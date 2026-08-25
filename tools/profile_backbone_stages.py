#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
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
    parser = argparse.ArgumentParser(
        "Profile StreamDSGN2Backbone stages"
    )

    parser.add_argument(
        "-n", "--num_frames",
        type=int,
        required=True
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50
    )

    parser.add_argument(
        "--cfg_file",
        default=DEFAULT_CFG
    )

    parser.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT
    )

    parser.add_argument(
        "--output",
        default="outputs/backbone_profile/backbone_stages.json"
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=20
    )

    return parser.parse_args()


class EventProfiler:
    def __init__(self):
        self.frame_events = defaultdict(list)
        self.handles = []

    def reset(self):
        self.frame_events.clear()

    def add_module(self, name, module):
        if module is None:
            return

        def pre_hook(mod, inputs):
            start = torch.cuda.Event(
                enable_timing=True
            )
            start.record()

            stack = getattr(
                mod,
                "_latency_event_stack",
                None
            )

            if stack is None:
                stack = []
                setattr(
                    mod,
                    "_latency_event_stack",
                    stack
                )

            stack.append(start)

        def post_hook(mod, inputs, output):
            stack = getattr(
                mod,
                "_latency_event_stack"
            )

            start = stack.pop()

            end = torch.cuda.Event(
                enable_timing=True
            )

            end.record()

            self.frame_events[name].append(
                (start, end)
            )

        self.handles.append(
            module.register_forward_pre_hook(
                pre_hook
            )
        )

        self.handles.append(
            module.register_forward_hook(
                post_hook
            )
        )

    def elapsed_calls(self, name):
        return [
            float(start.elapsed_time(end))
            for start, end
            in self.frame_events.get(name, [])
        ]

    def elapsed_sum(self, name):
        return float(
            sum(
                self.elapsed_calls(name)
            )
        )

    def close(self):
        for h in self.handles:
            h.remove()

        self.handles.clear()


def find_backbone(model):
    for module in model.modules():
        if (
            module.__class__.__name__
            == "StreamDSGN2Backbone"
        ):
            return module

    raise RuntimeError(
        "StreamDSGN2Backbone not found"
    )


def clear_stream_history(model):
    candidates = [
        "history_feature_queue",
        "history_queue",
    ]

    for name in candidates:
        obj = getattr(
            model,
            name,
            None
        )

        if obj is not None:
            try:
                obj.clear()
            except Exception:
                pass

    for name in [
        "prev_scene_token",
        "last_scene_token",
        "scene_token",
    ]:
        if hasattr(model, name):
            try:
                setattr(
                    model,
                    name,
                    None
                )
            except Exception:
                pass


def stats(values):
    x = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms": float(
            np.mean(x)
        ),
        "p50_ms": float(
            np.percentile(x, 50)
        ),
        "p95_ms": float(
            np.percentile(x, 95)
        ),
        "p99_ms": float(
            np.percentile(x, 99)
        ),
        "min_ms": float(
            np.min(x)
        ),
        "max_ms": float(
            np.max(x)
        ),
    }


def print_table(title, names, records, backbone_mean):
    print()
    print("=" * 86)
    print(title)
    print("=" * 86)

    print(
        f"{'Stage':32s}"
        f"{'Mean(ms)':>11s}"
        f"{'P50(ms)':>11s}"
        f"{'P95(ms)':>11s}"
        f"{'P99(ms)':>11s}"
        f"{'% Backbone':>12s}"
    )

    print("-" * 86)

    for name in names:
        s = stats(
            records[name]
        )

        ratio = (
            100.0
            * s["mean_ms"]
            / backbone_mean
            if backbone_mean > 0
            else 0.0
        )

        print(
            f"{name:32s}"
            f"{s['mean_ms']:11.3f}"
            f"{s['p50_ms']:11.3f}"
            f"{s['p95_ms']:11.3f}"
            f"{s['p99_ms']:11.3f}"
            f"{ratio:12.2f}"
        )


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

    backbone = find_backbone(
        model
    )

    print()
    print("Found backbone:")
    print(backbone.__class__.__name__)

    profiler = EventProfiler()

    # --------------------------------------------------------
    # Full backbone
    # --------------------------------------------------------
    profiler.add_module(
        "backbone_total",
        backbone
    )

    # --------------------------------------------------------
    # Stage A:
    # shared image frontend
    #
    # Same modules run once for left image and once for right.
    # We aggregate both calls.
    # --------------------------------------------------------
    profiler.add_module(
        "feature_backbone",
        backbone.feature_backbone
    )

    profiler.add_module(
        "feature_neck",
        backbone.feature_neck
    )

    # --------------------------------------------------------
    # Stage B:
    # stereo cost-volume / matching
    # --------------------------------------------------------
    if hasattr(
        backbone,
        "build_cost"
    ):
        profiler.add_module(
            "build_cost",
            backbone.build_cost
        )

    if hasattr(
        backbone,
        "dres0"
    ):
        profiler.add_module(
            "dres0",
            backbone.dres0
        )

    if hasattr(
        backbone,
        "dres1"
    ):
        profiler.add_module(
            "dres1",
            backbone.dres1
        )

    if hasattr(
        backbone,
        "hg_stereo"
    ):
        for i, module in enumerate(
            backbone.hg_stereo
        ):
            profiler.add_module(
                f"hg_stereo_{i}",
                module
            )

    # pred_stereo is currently normally skipped in this eval cfg
    # when drop_psv_loss=True, but hook it anyway in case config changes.
    if hasattr(
        backbone,
        "pred_stereo"
    ):
        for i, module in enumerate(
            backbone.pred_stereo
        ):
            profiler.add_module(
                f"pred_stereo_{i}",
                module
            )

    # --------------------------------------------------------
    # Stage D:
    # voxel / 3D processing
    # --------------------------------------------------------
    if hasattr(
        backbone,
        "squeeze_geo_conv"
    ):
        profiler.add_module(
            "squeeze_geo_conv",
            backbone.squeeze_geo_conv
        )

    profiler.add_module(
        "rpn3d_convs",
        backbone.rpn3d_convs
    )

    if hasattr(
        backbone,
        "rpn3d_hgs"
    ):
        for i, module in enumerate(
            backbone.rpn3d_hgs
        ):
            profiler.add_module(
                f"rpn3d_hg_{i}",
                module
            )

    profiler.add_module(
        "rpn3d_pool",
        backbone.rpn3d_pool
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    warmup_n = min(
        args.warmup,
        len(dataset)
    )

    print()
    print(
        f"Warmup: {warmup_n} frames"
    )

    with torch.no_grad():
        for i in range(warmup_n):
            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            load_data_to_gpu(
                batch
            )

            model(
                batch
            )

    torch.cuda.synchronize()

    clear_stream_history(
        model
    )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------
    N = min(
        args.num_frames,
        len(dataset)
    )

    records = defaultdict(list)

    with torch.no_grad():
        for i in range(N):
            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            # H2D is outside the measured model scope.
            load_data_to_gpu(
                batch
            )

            torch.cuda.synchronize()

            profiler.reset()

            model(
                batch
            )

            torch.cuda.synchronize()

            backbone_total = (
                profiler.elapsed_sum(
                    "backbone_total"
                )
            )

            # =================================================
            # Stage A
            # =================================================
            feature_backbone = (
                profiler.elapsed_sum(
                    "feature_backbone"
                )
            )

            feature_neck = (
                profiler.elapsed_sum(
                    "feature_neck"
                )
            )

            stage_A = (
                feature_backbone
                + feature_neck
            )

            # =================================================
            # Stage B
            # =================================================
            build_cost = (
                profiler.elapsed_sum(
                    "build_cost"
                )
            )

            dres0 = (
                profiler.elapsed_sum(
                    "dres0"
                )
            )

            dres1 = (
                profiler.elapsed_sum(
                    "dres1"
                )
            )

            hg_stereo = sum(
                profiler.elapsed_sum(
                    f"hg_stereo_{j}"
                )
                for j in range(
                    len(
                        getattr(
                            backbone,
                            "hg_stereo",
                            []
                        )
                    )
                )
            )

            pred_stereo = sum(
                profiler.elapsed_sum(
                    f"pred_stereo_{j}"
                )
                for j in range(
                    len(
                        getattr(
                            backbone,
                            "pred_stereo",
                            []
                        )
                    )
                )
            )

            stage_B = (
                build_cost
                + dres0
                + dres1
                + hg_stereo
                + pred_stereo
            )

            # =================================================
            # Stage D
            # =================================================
            squeeze_geo = (
                profiler.elapsed_sum(
                    "squeeze_geo_conv"
                )
            )

            rpn3d_convs = (
                profiler.elapsed_sum(
                    "rpn3d_convs"
                )
            )

            rpn3d_hgs = sum(
                profiler.elapsed_sum(
                    f"rpn3d_hg_{j}"
                )
                for j in range(
                    len(
                        getattr(
                            backbone,
                            "rpn3d_hgs",
                            []
                        )
                    )
                )
            )

            rpn3d_pool = (
                profiler.elapsed_sum(
                    "rpn3d_pool"
                )
            )

            stage_D = (
                squeeze_geo
                + rpn3d_convs
                + rpn3d_hgs
                + rpn3d_pool
            )

            # =================================================
            # Stage C
            #
            # Everything between stereo output and rpn3d that is
            # not represented by a standalone nn.Module:
            #
            # calibration / coordinate mapping
            # valid mask
            # cost-volume -> voxel grid_sample
            # tensor transforms / concatenation
            # etc.
            # =================================================
            stage_C = max(
                0.0,
                backbone_total
                - stage_A
                - stage_B
                - stage_D
            )

            records[
                "Backbone total"
            ].append(
                backbone_total
            )

            records[
                "A shared image frontend"
            ].append(
                stage_A
            )

            records[
                "B stereo core"
            ].append(
                stage_B
            )

            records[
                "C voxel mapping/projection"
            ].append(
                stage_C
            )

            records[
                "D voxel 3D processing"
            ].append(
                stage_D
            )

            # Sub-components
            sub_values = {
                "feature_backbone":
                    feature_backbone,

                "feature_neck":
                    feature_neck,

                "build_cost":
                    build_cost,

                "dres0":
                    dres0,

                "dres1":
                    dres1,

                "hg_stereo":
                    hg_stereo,

                "pred_stereo":
                    pred_stereo,

                "squeeze_geo_conv":
                    squeeze_geo,

                "rpn3d_convs":
                    rpn3d_convs,

                "rpn3d_hgs":
                    rpn3d_hgs,

                "rpn3d_pool":
                    rpn3d_pool,
            }

            for name, value in (
                sub_values.items()
            ):
                records[name].append(
                    value
                )

            # Left/right call-level information for shared frontend.
            fb_calls = (
                profiler.elapsed_calls(
                    "feature_backbone"
                )
            )

            fn_calls = (
                profiler.elapsed_calls(
                    "feature_neck"
                )
            )

            if len(fb_calls) >= 1:
                records[
                    "left feature_backbone"
                ].append(
                    fb_calls[0]
                )

            if len(fb_calls) >= 2:
                records[
                    "right feature_backbone"
                ].append(
                    fb_calls[1]
                )

            if len(fn_calls) >= 1:
                records[
                    "left feature_neck"
                ].append(
                    fn_calls[0]
                )

            if len(fn_calls) >= 2:
                records[
                    "right feature_neck"
                ].append(
                    fn_calls[1]
                )

            if (
                args.log_every > 0
                and (
                    (i + 1)
                    % args.log_every
                    == 0
                    or i + 1 == N
                )
            ):
                print(
                    f"[{i+1:4d}/{N}] "
                    f"total={backbone_total:7.3f} "
                    f"A={stage_A:6.3f} "
                    f"B={stage_B:6.3f} "
                    f"C={stage_C:6.3f} "
                    f"D={stage_D:6.3f}"
                )

    profiler.close()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    backbone_mean = stats(
        records[
            "Backbone total"
        ]
    )["mean_ms"]

    major_names = [
        "Backbone total",
        "A shared image frontend",
        "B stereo core",
        "C voxel mapping/projection",
        "D voxel 3D processing",
    ]

    print_table(
        "StreamDSGN2Backbone major stages",
        major_names,
        records,
        backbone_mean
    )

    sub_names = [
        "feature_backbone",
        "feature_neck",
        "left feature_backbone",
        "right feature_backbone",
        "left feature_neck",
        "right feature_neck",
        "build_cost",
        "dres0",
        "dres1",
        "hg_stereo",
        "pred_stereo",
        "squeeze_geo_conv",
        "rpn3d_convs",
        "rpn3d_hgs",
        "rpn3d_pool",
    ]

    sub_names = [
        n
        for n in sub_names
        if (
            n in records
            and len(records[n]) > 0
        )
    ]

    print_table(
        "Backbone sub-components",
        sub_names,
        records,
        backbone_mean
    )

    # --------------------------------------------------------
    # Suggested cut analysis
    # --------------------------------------------------------
    A_mean = stats(
        records[
            "A shared image frontend"
        ]
    )["mean_ms"]

    B_mean = stats(
        records[
            "B stereo core"
        ]
    )["mean_ms"]

    C_mean = stats(
        records[
            "C voxel mapping/projection"
        ]
    )["mean_ms"]

    D_mean = stats(
        records[
            "D voxel 3D processing"
        ]
    )["mean_ms"]

    adaptive_expensive = (
        B_mean
        + C_mean
        + D_mean
    )

    print()
    print("=" * 86)
    print("Candidate shared/adaptive split")
    print("=" * 86)

    print(
        f"Shared frontend candidate A : "
        f"{A_mean:.3f} ms"
    )

    print(
        f"Adaptive expensive B+C+D   : "
        f"{adaptive_expensive:.3f} ms"
    )

    print(
        f"Adaptive fraction          : "
        f"{100.0 * adaptive_expensive / backbone_mean:.2f}% "
        f"of backbone"
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "frames": N,
        "warmup": warmup_n,
        "gpu": torch.cuda.get_device_name(0),
        "stages": {
            name: stats(values)
            for name, values
            in records.items()
        },
        "candidate_split": {
            "shared_A_mean_ms":
                A_mean,

            "adaptive_BCD_mean_ms":
                adaptive_expensive,

            "adaptive_fraction_percent":
                float(
                    100.0
                    * adaptive_expensive
                    / backbone_mean
                ),
        },
    }

    output.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    print()
    print(
        "Saved JSON:",
        output
    )


if __name__ == "__main__":
    main()
