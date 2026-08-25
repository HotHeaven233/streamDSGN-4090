#!/usr/bin/env python3

import argparse
import json
import math
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

    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)

    p.add_argument(
        "--lut",
        default="outputs/backbone_profile/d_roi_latency_lut.json"
    )

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.50]
    )

    p.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16]
    )

    p.add_argument("--halo", type=int, default=4)
    p.add_argument("--align", type=int, default=4)

    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeat", type=int, default=200)

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "latency_aware_packing_v2.json"
        )
    )

    return p.parse_args()


# ============================================================
# Model
# ============================================================

def find_backbone(model):
    for m in model.modules():
        if m.__class__.__name__ == "StreamDSGN2Backbone":
            return m

    raise RuntimeError("StreamDSGN2Backbone not found")


def run_d_stage(backbone, x):

    use_amp = bool(
        getattr(backbone, "use_amp", False)
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
                        y, pre, post
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


# ============================================================
# Timing
# ============================================================

def stats(values):
    x = np.asarray(values, dtype=np.float64)

    return {
        "mean_ms": float(x.mean()),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
    }


def measure(fn, warmup, repeat):

    with torch.no_grad():
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()

    values = []

    with torch.no_grad():

        for _ in range(repeat):

            st = torch.cuda.Event(enable_timing=True)
            ed = torch.cuda.Event(enable_timing=True)

            st.record()

            fn()

            ed.record()

            ed.synchronize()

            values.append(
                float(st.elapsed_time(ed))
            )

    return stats(values)


# ============================================================
# ROI utilities
# ============================================================

def roi_area(r):
    y0, y1, x0, x1 = r

    return (
        max(0, y1 - y0)
        * max(0, x1 - x0)
    )


def merge_bbox(a, b):
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def union_bbox(rois):
    out = rois[0]

    for r in rois[1:]:
        out = merge_bbox(out, r)

    return out


def expand_align(
    roi,
    halo,
    H,
    W,
    align
):
    y0, y1, x0, x1 = roi

    y0 = max(0, y0 - halo)
    x0 = max(0, x0 - halo)

    y1 = min(H, y1 + halo)
    x1 = min(W, x1 + halo)

    y0 = (y0 // align) * align
    x0 = (x0 // align) * align

    y1 = min(
        H,
        int(
            math.ceil(y1 / align)
            * align
        )
    )

    x1 = min(
        W,
        int(
            math.ceil(x1 / align)
            * align
        )
    )

    return y0, y1, x0, x1


def fragmentation_grid(n):

    known = {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
        16: (4, 4),
    }

    if n in known:
        return known[n]

    rows = max(
        1,
        int(math.sqrt(n))
    )

    while n % rows != 0:
        rows -= 1

    return rows, n // rows


def make_fragmented_rois(
    H,
    W,
    ratio,
    n
):
    gh, gw = fragmentation_grid(n)

    scale = math.sqrt(ratio)

    rois = []

    for iy in range(gh):

        sy0 = round(
            iy * H / gh
        )

        sy1 = round(
            (iy + 1) * H / gh
        )

        sh = sy1 - sy0

        for ix in range(gw):

            sx0 = round(
                ix * W / gw
            )

            sx1 = round(
                (ix + 1) * W / gw
            )

            sw = sx1 - sx0

            rh = max(
                1,
                round(sh * scale)
            )

            rw = max(
                1,
                round(sw * scale)
            )

            rh = min(rh, sh)
            rw = min(rw, sw)

            y0 = (
                sy0
                + (sh - rh) // 2
            )

            x0 = (
                sx0
                + (sw - rw) // 2
            )

            rois.append(
                (
                    y0,
                    y0 + rh,
                    x0,
                    x0 + rw,
                )
            )

    return rois


# ============================================================
# Shape-aware LUT
# ============================================================

class LatencyLUT:

    def __init__(self, path):

        data = json.load(
            open(path)
        )

        self.records = data["records"]

        self.hs = sorted({
            int(x["h"])
            for x in self.records
        })

        self.ws = sorted({
            int(x["w"])
            for x in self.records
        })

        self.table = {
            (
                int(x["h"]),
                int(x["w"])
            ):
            float(x["mean_ms"])

            for x in self.records
        }

    @staticmethod
    def nearest(values, x):
        return min(
            values,
            key=lambda v:
            abs(v - x)
        )

    def predict(self, h, w):

        # V1 deliberately uses nearest-neighbour LUT.
        # No fitted smooth function yet.
        hh = self.nearest(
            self.hs,
            h
        )

        ww = self.nearest(
            self.ws,
            w
        )

        return self.table[
            (hh, ww)
        ]

    def predict_exec_roi(
        self,
        roi,
        halo,
        H,
        W,
        align
    ):
        e = expand_align(
            roi,
            halo,
            H,
            W,
            align
        )

        h = e[1] - e[0]
        w = e[3] - e[2]

        return self.predict(
            h,
            w
        )


def predict_set_latency(
    lut,
    rois,
    halo,
    H,
    W,
    align
):
    return sum(
        lut.predict_exec_roi(
            r,
            halo,
            H,
            W,
            align
        )
        for r in rois
    )


# ============================================================
# Packing
# ============================================================

def area_greedy_to_cap(
    input_rois,
    cap,
    halo,
    H,
    W,
    align
):
    rois = list(input_rois)

    while len(rois) > cap:

        best = None

        for i in range(len(rois)):
            for j in range(i + 1, len(rois)):

                ri = rois[i]
                rj = rois[j]

                merged = merge_bbox(
                    ri,
                    rj
                )

                ei = expand_align(
                    ri,
                    halo,
                    H,
                    W,
                    align
                )

                ej = expand_align(
                    rj,
                    halo,
                    H,
                    W,
                    align
                )

                em = expand_align(
                    merged,
                    halo,
                    H,
                    W,
                    align
                )

                delta = (
                    roi_area(em)
                    - roi_area(ei)
                    - roi_area(ej)
                )

                if (
                    best is None
                    or delta < best[0]
                ):
                    best = (
                        delta,
                        i,
                        j,
                        merged
                    )

        _, i, j, merged = best

        rois = [
            r
            for k, r in enumerate(rois)
            if k not in (i, j)
        ] + [merged]

    return rois


def latency_aware_pack(
    input_rois,
    lut,
    halo,
    H,
    W,
    align,
    full_latency
):
    """
    Greedy hardware-aware packing.

    Start from all fine-grained ROIs.

    At every iteration:
      find the pair whose merge produces the largest
      predicted latency reduction.

    Stop when no merge improves predicted latency.

    Finally compare against Full D.
    """

    rois = list(input_rois)

    while len(rois) > 1:

        current_cost = predict_set_latency(
            lut,
            rois,
            halo,
            H,
            W,
            align
        )

        best_gain = 0.0
        best_new = None

        for i in range(len(rois)):
            for j in range(
                i + 1,
                len(rois)
            ):

                merged = merge_bbox(
                    rois[i],
                    rois[j]
                )

                candidate = [
                    r
                    for k, r in enumerate(rois)
                    if k not in (i, j)
                ]

                candidate.append(
                    merged
                )

                candidate_cost = (
                    predict_set_latency(
                        lut,
                        candidate,
                        halo,
                        H,
                        W,
                        align
                    )
                )

                gain = (
                    current_cost
                    - candidate_cost
                )

                if gain > best_gain:
                    best_gain = gain
                    best_new = candidate

        if best_new is None:
            break

        rois = best_new

    predicted = predict_set_latency(
        lut,
        rois,
        halo,
        H,
        W,
        align
    )

    if full_latency <= predicted:

        return {
            "mode": "full",
            "rois": [],
            "predicted_ms":
                float(full_latency),
        }

    return {
        "mode": "selective",
        "rois": rois,
        "predicted_ms":
            float(predicted),
    }


# ============================================================
# Actual execution
# ============================================================

def make_exec_fn(
    backbone,
    full_input,
    canvas,
    rois,
    halo,
    align
):
    H = full_input.shape[-2]
    W = full_input.shape[-1]

    pairs = []

    for core in rois:

        e = expand_align(
            core,
            halo,
            H,
            W,
            align
        )

        pairs.append(
            (core, e)
        )

    def run():

        for core, e in pairs:

            cy0, cy1, cx0, cx1 = core
            ey0, ey1, ex0, ex1 = e

            crop = (
                full_input[
                    ...,
                    ey0:ey1,
                    ex0:ex1
                ]
                .contiguous()
            )

            y = run_d_stage(
                backbone,
                crop
            )

            ry0 = cy0 - ey0
            rx0 = cx0 - ex0

            rh = cy1 - cy0
            rw = cx1 - cx0

            canvas[
                ...,
                cy0:cy1,
                cx0:cx1
            ].copy_(
                y[
                    ...,
                    ry0:ry0+rh,
                    rx0:rx0+rw
                ]
            )

        return canvas

    return run


# ============================================================
# Main
# ============================================================

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
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False
    )

    model.cuda().eval()

    backbone = find_backbone(model)

    captured = {}

    def hook(module, inputs):
        captured["x"] = (
            inputs[0]
            .detach()
            .clone()
        )

    h = (
        backbone.rpn3d_convs
        .register_forward_pre_hook(hook)
    )

    sample = dataset[0]

    batch = dataset.collate_batch(
        [sample]
    )

    load_data_to_gpu(batch)

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()
    h.remove()

    x = captured["x"].contiguous()

    _, _, _, H, W = x.shape

    with torch.no_grad():
        full_output = run_d_stage(
            backbone,
            x
        )

    torch.cuda.synchronize()

    canvas = torch.empty_like(
        full_output
    )

    def run_full():
        return run_d_stage(
            backbone,
            x
        )

    full_stats = measure(
        run_full,
        args.warmup,
        args.repeat
    )

    full_latency = full_stats["mean_ms"]

    lut = LatencyLUT(
        args.lut
    )

    print()
    print("D input:", tuple(x.shape))
    print(
        f"Full D = {full_latency:.3f} ms"
    )

    strategies = [
        "naive",
        "area_cap4",
        "area_cap2",
        "bbox1",
        "latency_v2",
    ]

    records = []

    print()
    print("=" * 122)

    print(
        f"{'Ratio':>7s} "
        f"{'Frag':>5s} "
        f"{'Strategy':>12s} "
        f"{'ROI':>4s} "
        f"{'Pred(ms)':>10s} "
        f"{'Real(ms)':>10s} "
        f"{'P95':>8s} "
        f"{'PredErr%':>9s} "
        f"{'Speedup':>8s} "
        f"{'Mode':>10s}"
    )

    print("-" * 122)

    for ratio in args.ratios:

        for frag in args.fragments:

            base = make_fragmented_rois(
                H,
                W,
                ratio,
                frag
            )

            for strategy in strategies:

                mode = "selective"

                if strategy == "naive":

                    rois = list(base)

                    pred = predict_set_latency(
                        lut,
                        rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                elif strategy == "area_cap4":

                    rois = area_greedy_to_cap(
                        base,
                        min(4, len(base)),
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                    pred = predict_set_latency(
                        lut,
                        rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                elif strategy == "area_cap2":

                    rois = area_greedy_to_cap(
                        base,
                        min(2, len(base)),
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                    pred = predict_set_latency(
                        lut,
                        rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                elif strategy == "bbox1":

                    rois = [
                        union_bbox(base)
                    ]

                    pred = predict_set_latency(
                        lut,
                        rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )

                elif strategy == "latency_v2":

                    result = latency_aware_pack(
                        base,
                        lut,
                        args.halo,
                        H,
                        W,
                        args.align,
                        full_latency
                    )

                    mode = result["mode"]
                    rois = result["rois"]
                    pred = result["predicted_ms"]

                else:
                    raise ValueError(strategy)

                if mode == "full":

                    real = full_stats

                else:

                    fn = make_exec_fn(
                        backbone,
                        x,
                        canvas,
                        rois,
                        args.halo,
                        args.align
                    )

                    real = measure(
                        fn,
                        args.warmup,
                        args.repeat
                    )

                err = (
                    (pred - real["mean_ms"])
                    / real["mean_ms"]
                    * 100.0
                )

                speedup = (
                    full_latency
                    / real["mean_ms"]
                )

                records.append({
                    "ratio": ratio,
                    "fragments": frag,
                    "strategy": strategy,
                    "mode": mode,
                    "num_rois": len(rois),
                    "predicted_ms": pred,
                    "real": real,
                    "prediction_error_percent": err,
                    "speedup_vs_full": speedup,
                    "rois": [
                        list(r)
                        for r in rois
                    ],
                })

                print(
                    f"{ratio*100:6.1f}% "
                    f"{frag:5d} "
                    f"{strategy:>12s} "
                    f"{len(rois):4d} "
                    f"{pred:10.3f} "
                    f"{real['mean_ms']:10.3f} "
                    f"{real['p95_ms']:8.3f} "
                    f"{err:9.2f} "
                    f"{speedup:8.3f} "
                    f"{mode:>10s}"
                )

    # ========================================================
    # Best strategy summary
    # ========================================================

    print()
    print("=" * 90)
    print("Best measured strategy per setting")
    print("=" * 90)

    for ratio in args.ratios:

        for frag in args.fragments:

            rows = [
                r
                for r in records
                if (
                    r["ratio"] == ratio
                    and
                    r["fragments"] == frag
                )
            ]

            best = min(
                rows,
                key=lambda r:
                r["real"]["mean_ms"]
            )

            lv2 = next(
                r
                for r in rows
                if r["strategy"] == "latency_v2"
            )

            gap = (
                lv2["real"]["mean_ms"]
                - best["real"]["mean_ms"]
            )

            print(
                f"ratio={ratio*100:5.1f}% "
                f"frag={frag:2d}  "
                f"oracle={best['strategy']:>12s} "
                f"{best['real']['mean_ms']:6.3f} ms  "
                f"latency_v2="
                f"{lv2['real']['mean_ms']:6.3f} ms  "
                f"gap={gap:+6.3f} ms"
            )

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.write_text(
        json.dumps(
            {
                "full_d": full_stats,
                "lut": args.lut,
                "halo": args.halo,
                "align": args.align,
                "records": records,
            },
            indent=2
        )
    )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
