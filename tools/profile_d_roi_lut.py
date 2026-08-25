#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cfg_file",
        default=DEFAULT_CFG
    )

    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT
    )

    p.add_argument(
        "--align",
        type=int,
        default=4
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=20
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=100
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "d_roi_latency_lut.json"
        )
    )

    return p.parse_args()


def find_backbone(model):
    for m in model.modules():
        if m.__class__.__name__ == "StreamDSGN2Backbone":
            return m

    raise RuntimeError(
        "StreamDSGN2Backbone not found"
    )


def run_d_stage(backbone, x):

    use_amp = bool(
        getattr(
            backbone,
            "use_amp",
            False
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=use_amp
    ):

        y = backbone.rpn3d_convs(x)

        if backbone.num_3dconvs_hg > 0:

            if backbone.num_3dconvs_hg == 1:

                pre, post = True, True

                for hg in backbone.rpn3d_hgs:
                    y, pre, post = hg(
                        y,
                        pre,
                        post
                    )

            else:

                for hg in backbone.rpn3d_hgs:
                    y = hg(
                        y,
                        None,
                        None
                    )

        y = backbone.rpn3d_pool(y)

    return y


def stats(values):

    x = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms": float(x.mean()),
        "p50_ms": float(
            np.percentile(x, 50)
        ),
        "p95_ms": float(
            np.percentile(x, 95)
        ),
        "p99_ms": float(
            np.percentile(x, 99)
        ),
    }


def measure(fn, warmup, repeat):

    with torch.no_grad():
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()

    values = []

    with torch.no_grad():

        for _ in range(repeat):

            st = torch.cuda.Event(
                enable_timing=True
            )

            ed = torch.cuda.Event(
                enable_timing=True
            )

            st.record()

            fn()

            ed.record()

            ed.synchronize()

            values.append(
                st.elapsed_time(ed)
            )

    return stats(values)


def aligned_sizes(full, align):

    # Dense sampling, especially in the practically useful
    # middle/high spatial-size range.
    ratios = [
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.50,
        0.5625,
        0.625,
        0.6875,
        0.75,
        0.8125,
        0.875,
        0.9375,
        1.0,
    ]

    values = set()

    for r in ratios:

        x = int(
            round(
                full * r
            )
        )

        x = max(
            align,
            (x // align) * align
        )

        x = min(
            full,
            x
        )

        values.add(x)

    values.add(full)

    return sorted(values)


def main():

    args = parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg
    )

    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    logger = common_utils.create_logger(
        rank=0
    )

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(
            cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False
    )

    model.cuda()
    model.eval()

    backbone = find_backbone(model)

    # --------------------------------------------------------
    # Capture real D-stage input
    # --------------------------------------------------------

    captured = {}

    def hook(module, inputs):
        captured["x"] = (
            inputs[0]
            .detach()
            .clone()
        )

    handle = (
        backbone.rpn3d_convs
        .register_forward_pre_hook(
            hook
        )
    )

    sample = dataset[0]

    batch = dataset.collate_batch(
        [sample]
    )

    load_data_to_gpu(batch)

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()

    handle.remove()

    x_full = (
        captured["x"]
        .contiguous()
    )

    _, C, Z, H, W = x_full.shape

    print()
    print(
        "D input:",
        tuple(x_full.shape)
    )

    print(
        "AMP:",
        bool(
            getattr(
                backbone,
                "use_amp",
                False
            )
        )
    )

    hs = aligned_sizes(
        H,
        args.align
    )

    ws = aligned_sizes(
        W,
        args.align
    )

    print(
        f"H candidates ({len(hs)}):",
        hs
    )

    print(
        f"W candidates ({len(ws)}):",
        ws
    )

    print()
    print(
        f"Total shapes: "
        f"{len(hs) * len(ws)}"
    )

    records = []

    print()
    print("=" * 88)

    print(
        f"{'H':>5s} "
        f"{'W':>5s} "
        f"{'Area%':>8s} "
        f"{'Aspect':>8s} "
        f"{'Mean':>9s} "
        f"{'P50':>9s} "
        f"{'P95':>9s} "
        f"{'P99':>9s}"
    )

    print("-" * 88)

    for h in hs:
        for w in ws:

            # Central crop.
            y0 = (
                H - h
            ) // 2

            x0 = (
                W - w
            ) // 2

            crop = (
                x_full[
                    ...,
                    y0:y0+h,
                    x0:x0+w
                ]
                .contiguous()
            )

            # Preallocate output destination so a realistic
            # output copy can also be measured.
            with torch.no_grad():
                y_ref = run_d_stage(
                    backbone,
                    crop
                )

            torch.cuda.synchronize()

            canvas = torch.empty_like(
                y_ref
            )

            def fn():

                # Keep crop materialization in timing.
                xx = (
                    x_full[
                        ...,
                        y0:y0+h,
                        x0:x0+w
                    ]
                    .contiguous()
                )

                yy = run_d_stage(
                    backbone,
                    xx
                )

                # Keep output copy/scatter-like cost.
                canvas.copy_(yy)

                return canvas

            s = measure(
                fn,
                args.warmup,
                args.repeat
            )

            area_ratio = (
                h * w
            ) / float(
                H * W
            )

            aspect = (
                w / float(h)
            )

            row = {
                "h": int(h),
                "w": int(w),
                "area_ratio": float(
                    area_ratio
                ),
                "aspect_ratio": float(
                    aspect
                ),
                **s,
            }

            records.append(row)

            print(
                f"{h:5d} "
                f"{w:5d} "
                f"{area_ratio*100:8.2f} "
                f"{aspect:8.3f} "
                f"{s['mean_ms']:9.3f} "
                f"{s['p50_ms']:9.3f} "
                f"{s['p95_ms']:9.3f} "
                f"{s['p99_ms']:9.3f}"
            )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            {
                "gpu":
                    torch.cuda.get_device_name(0),

                "input_shape":
                    list(x_full.shape),

                "align":
                    args.align,

                "warmup":
                    args.warmup,

                "repeat":
                    args.repeat,

                "records":
                    records,
            },
            indent=2
        )
    )

    print()
    print(
        "Saved:",
        out
    )


if __name__ == "__main__":
    main()
