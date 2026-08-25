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


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        "D-stage ROI fragmentation / packing profiler"
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
        "--ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.50],
        help="Useful recompute area ratios."
    )

    parser.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="Number of fine-grained recompute fragments."
    )

    parser.add_argument(
        "--halo",
        type=int,
        default=4,
        help="Halo size in D-stage H/W cells."
    )

    parser.add_argument(
        "--align",
        type=int,
        default=4,
        help="Execution ROI spatial alignment."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=200
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "roi_fragmentation.json"
        )
    )

    return parser.parse_args()


# ============================================================
# Model helpers
# ============================================================

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


def run_d_stage(backbone, x):
    """
    Exactly follow the original D-stage path under the same
    AMP/autocast condition used by StreamDSGN inference.

        rpn3d_convs
        -> rpn3d_hgs
        -> rpn3d_pool
    """

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

                # Preserve original implementation semantics.
                pre, post = True, True

                for hg in backbone.rpn3d_hgs:
                    y, pre, post = hg(
                        y,
                        pre,
                        post
                    )

            else:

                pre, post = None, None

                for hg in backbone.rpn3d_hgs:
                    y = hg(
                        y,
                        pre,
                        post
                    )

        y = backbone.rpn3d_pool(y)

    return y


# ============================================================
# ROI utilities
#
# ROI format:
#     (y0, y1, x0, x1)
#
# Half-open ranges:
#     y0 <= y < y1
#     x0 <= x < x1
# ============================================================

def roi_area(roi):
    y0, y1, x0, x1 = roi

    return max(
        0,
        y1 - y0
    ) * max(
        0,
        x1 - x0
    )


def merge_bbox(a, b):
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def union_bbox(rois):
    if len(rois) == 0:
        raise ValueError("empty ROI list")

    out = rois[0]

    for roi in rois[1:]:
        out = merge_bbox(
            out,
            roi
        )

    return out


def expand_and_align(
    roi,
    halo,
    H,
    W,
    align
):
    """
    Expand ROI by halo and then outward-align boundaries.

    Full H/W are expected to be divisible by align.
    """

    y0, y1, x0, x1 = roi

    y0 = max(
        0,
        y0 - halo
    )

    y1 = min(
        H,
        y1 + halo
    )

    x0 = max(
        0,
        x0 - halo
    )

    x1 = min(
        W,
        x1 + halo
    )

    # Outward alignment
    y0 = (
        y0 // align
    ) * align

    x0 = (
        x0 // align
    ) * align

    y1 = min(
        H,
        int(
            math.ceil(
                y1 / align
            )
            * align
        )
    )

    x1 = min(
        W,
        int(
            math.ceil(
                x1 / align
            )
            * align
        )
    )

    if y1 <= y0 or x1 <= x0:
        raise RuntimeError(
            f"Invalid aligned ROI: "
            f"{(y0, y1, x0, x1)}"
        )

    return (
        y0,
        y1,
        x0,
        x1,
    )


def mask_union_area(
    rois,
    H,
    W
):
    mask = np.zeros(
        (H, W),
        dtype=np.bool_
    )

    for y0, y1, x0, x1 in rois:
        mask[
            y0:y1,
            x0:x1
        ] = True

    return int(
        mask.sum()
    )


def summed_execution_area(
    rois,
    halo,
    H,
    W,
    align
):
    total = 0

    for roi in rois:
        exec_roi = expand_and_align(
            roi,
            halo,
            H,
            W,
            align
        )

        total += roi_area(
            exec_roi
        )

    return total


def unique_execution_area(
    rois,
    halo,
    H,
    W,
    align
):
    exec_rois = [
        expand_and_align(
            roi,
            halo,
            H,
            W,
            align
        )
        for roi in rois
    ]

    return mask_union_area(
        exec_rois,
        H,
        W
    )


# ============================================================
# Generate controlled fragmented layouts
# ============================================================

def fragmentation_grid(n):
    """
    Exact regular layouts for our default fragmentation levels.

        1  -> 1 x 1
        2  -> 1 x 2
        4  -> 2 x 2
        8  -> 2 x 4
        16 -> 4 x 4
    """

    known = {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
        16: (4, 4),
    }

    if n in known:
        return known[n]

    # Generic fallback
    rows = int(
        math.floor(
            math.sqrt(n)
        )
    )

    rows = max(
        1,
        rows
    )

    while n % rows != 0:
        rows -= 1

    cols = n // rows

    return rows, cols


def make_fragmented_rois(
    H,
    W,
    useful_ratio,
    n_fragments
):
    """
    Divide full feature map into n placement cells.

    Inside every cell place one centered ROI.

    Each ROI occupies approximately useful_ratio of that
    placement cell, therefore total useful area remains
    approximately constant while number of fragments changes.

    This gives:
        same useful compute ratio
        + progressively more fragmented execution.
    """

    grid_h, grid_w = fragmentation_grid(
        n_fragments
    )

    if grid_h * grid_w != n_fragments:
        raise RuntimeError(
            "fragmentation grid mismatch"
        )

    scale = math.sqrt(
        useful_ratio
    )

    rois = []

    for gy in range(grid_h):
        slot_y0 = int(
            round(
                gy
                * H
                / grid_h
            )
        )

        slot_y1 = int(
            round(
                (gy + 1)
                * H
                / grid_h
            )
        )

        slot_h = (
            slot_y1
            - slot_y0
        )

        for gx in range(grid_w):

            slot_x0 = int(
                round(
                    gx
                    * W
                    / grid_w
                )
            )

            slot_x1 = int(
                round(
                    (gx + 1)
                    * W
                    / grid_w
                )
            )

            slot_w = (
                slot_x1
                - slot_x0
            )

            roi_h = max(
                1,
                int(
                    round(
                        slot_h
                        * scale
                    )
                )
            )

            roi_w = max(
                1,
                int(
                    round(
                        slot_w
                        * scale
                    )
                )
            )

            roi_h = min(
                roi_h,
                slot_h
            )

            roi_w = min(
                roi_w,
                slot_w
            )

            y0 = (
                slot_y0
                + (
                    slot_h
                    - roi_h
                )
                // 2
            )

            x0 = (
                slot_x0
                + (
                    slot_w
                    - roi_w
                )
                // 2
            )

            y1 = (
                y0
                + roi_h
            )

            x1 = (
                x0
                + roi_w
            )

            rois.append(
                (
                    y0,
                    y1,
                    x0,
                    x1,
                )
            )

    assert (
        len(rois)
        == n_fragments
    )

    return rois


# ============================================================
# Packing
# ============================================================

def greedy_merge_to_cap(
    input_rois,
    cap,
    halo,
    H,
    W,
    align
):
    """
    Greedily merge the pair with the smallest increase in
    halo-expanded execution area until <= cap ROIs remain.

    Important:
    this is NOT yet the final learned latency-aware packing.
    It is a deterministic first prototype.
    """

    rois = list(
        input_rois
    )

    while len(rois) > cap:

        best_pair = None
        best_delta = None
        best_merged = None

        for i in range(len(rois)):
            for j in range(
                i + 1,
                len(rois)
            ):

                ri = rois[i]
                rj = rois[j]

                merged = merge_bbox(
                    ri,
                    rj
                )

                separate_cost = (
                    summed_execution_area(
                        [ri, rj],
                        halo,
                        H,
                        W,
                        align
                    )
                )

                merged_cost = (
                    summed_execution_area(
                        [merged],
                        halo,
                        H,
                        W,
                        align
                    )
                )

                delta = (
                    merged_cost
                    - separate_cost
                )

                if (
                    best_delta is None
                    or delta < best_delta
                ):
                    best_delta = delta
                    best_pair = (
                        i,
                        j
                    )
                    best_merged = merged

        i, j = best_pair

        new_rois = []

        for k, roi in enumerate(rois):
            if (
                k != i
                and k != j
            ):
                new_rois.append(
                    roi
                )

        new_rois.append(
            best_merged
        )

        rois = new_rois

    return rois


def get_packed_rois(
    base_rois,
    strategy,
    halo,
    H,
    W,
    align
):
    if strategy == "naive":
        return list(
            base_rois
        )

    if strategy == "cap4":
        return greedy_merge_to_cap(
            base_rois,
            min(
                4,
                len(base_rois)
            ),
            halo,
            H,
            W,
            align
        )

    if strategy == "cap2":
        return greedy_merge_to_cap(
            base_rois,
            min(
                2,
                len(base_rois)
            ),
            halo,
            H,
            W,
            align
        )

    if strategy == "bbox1":
        return [
            union_bbox(
                base_rois
            )
        ]

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# Timing
# ============================================================

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
            np.percentile(
                x,
                50
            )
        ),
        "p95_ms": float(
            np.percentile(
                x,
                95
            )
        ),
        "p99_ms": float(
            np.percentile(
                x,
                99
            )
        ),
        "min_ms": float(
            np.min(x)
        ),
        "max_ms": float(
            np.max(x)
        ),
    }


def cuda_measure(
    fn,
    warmup,
    repeat
):
    with torch.no_grad():

        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()

    values = []

    with torch.no_grad():

        for _ in range(repeat):

            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            fn()

            end.record()

            end.synchronize()

            values.append(
                float(
                    start.elapsed_time(
                        end
                    )
                )
            )

    return stats(
        values
    )


def build_roi_execution_fn(
    backbone,
    full_input,
    output_canvas,
    packed_rois,
    halo,
    align
):
    """
    Timing includes:

        slice
        -> contiguous crop
        -> D-stage recomputation
        -> scatter

    It intentionally excludes:
        Decision Network
        scheduler
        P/U feature generation

    because this experiment isolates fragmentation cost.
    """

    H = full_input.shape[-2]
    W = full_input.shape[-1]

    exec_pairs = []

    for core_roi in packed_rois:

        exec_roi = expand_and_align(
            core_roi,
            halo,
            H,
            W,
            align
        )

        exec_pairs.append(
            (
                core_roi,
                exec_roi
            )
        )

    def run():

        for core_roi, exec_roi in exec_pairs:

            cy0, cy1, cx0, cx1 = core_roi
            ey0, ey1, ex0, ex1 = exec_roi

            # ROI extraction cost is included.
            crop = full_input[
                ...,
                ey0:ey1,
                ex0:ex1
            ].contiguous()

            y = run_d_stage(
                backbone,
                crop
            )

            # D-stage preserves H/W.
            rel_y0 = (
                cy0 - ey0
            )

            rel_y1 = (
                rel_y0
                + (
                    cy1 - cy0
                )
            )

            rel_x0 = (
                cx0 - ex0
            )

            rel_x1 = (
                rel_x0
                + (
                    cx1 - cx0
                )
            )

            # Scatter cost is included.
            output_canvas[
                ...,
                cy0:cy1,
                cx0:cx1
            ].copy_(
                y[
                    ...,
                    rel_y0:rel_y1,
                    rel_x0:rel_x1
                ]
            )

        return output_canvas

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

    logger = (
        common_utils.create_logger(
            rank=0
        )
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

    # --------------------------------------------------------
    # Capture the real input tensor immediately before D stage.
    # --------------------------------------------------------

    captured = {}

    def capture_hook(
        module,
        inputs
    ):
        captured[
            "d_input"
        ] = (
            inputs[0]
            .detach()
            .clone()
        )

    handle = (
        backbone.rpn3d_convs
        .register_forward_pre_hook(
            capture_hook
        )
    )

    sample = dataset[0]

    batch = (
        dataset.collate_batch(
            [sample]
        )
    )

    load_data_to_gpu(
        batch
    )

    with torch.no_grad():
        model(
            batch
        )

    torch.cuda.synchronize()

    handle.remove()

    if "d_input" not in captured:
        raise RuntimeError(
            "Failed to capture D-stage input"
        )

    full_input = (
        captured[
            "d_input"
        ]
        .contiguous()
    )

    print()
    print(
        "Captured D input:",
        tuple(
            full_input.shape
        ),
        full_input.dtype
    )

    print(
        "Backbone inference AMP:",
        bool(getattr(backbone, "use_amp", False))
    )

    _, _, Z, H, W = (
        full_input.shape
    )

    if (
        H % args.align != 0
        or W % args.align != 0
    ):
        print(
            "WARNING: H/W not divisible by align:",
            H,
            W,
            args.align
        )

    # --------------------------------------------------------
    # Get true full D output shape.
    # --------------------------------------------------------

    with torch.no_grad():
        full_output = run_d_stage(
            backbone,
            full_input
        )

    torch.cuda.synchronize()

    output_canvas = (
        torch.empty_like(
            full_output
        )
    )

    print(
        "D output:",
        tuple(
            full_output.shape
        )
    )

    # --------------------------------------------------------
    # Full D baseline
    # --------------------------------------------------------

    def run_full():
        return run_d_stage(
            backbone,
            full_input
        )

    print()
    print(
        "Measuring full D-stage baseline..."
    )

    full_stats = cuda_measure(
        run_full,
        args.warmup,
        args.repeat
    )

    full_mean = (
        full_stats[
            "mean_ms"
        ]
    )

    print(
        f"Full D: "
        f"mean={full_stats['mean_ms']:.3f} ms  "
        f"p50={full_stats['p50_ms']:.3f}  "
        f"p95={full_stats['p95_ms']:.3f}  "
        f"p99={full_stats['p99_ms']:.3f}"
    )

    # --------------------------------------------------------
    # Fragmentation experiment
    # --------------------------------------------------------

    strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    records = []

    full_area = (
        H * W
    )

    print()
    print("=" * 132)

    print(
        f"{'Ratio':>7s} "
        f"{'Frag':>5s} "
        f"{'Strategy':>10s} "
        f"{'ROI':>4s} "
        f"{'Useful%':>8s} "
        f"{'Packed%':>8s} "
        f"{'ExecSum%':>9s} "
        f"{'ExecUniq%':>10s} "
        f"{'Mean(ms)':>10s} "
        f"{'P95':>8s} "
        f"{'P99':>8s} "
        f"{'Speedup':>8s}"
    )

    print("-" * 132)

    for ratio in args.ratios:

        if not (
            0.0
            < ratio
            <= 1.0
        ):
            raise ValueError(
                f"Invalid ratio: {ratio}"
            )

        for n_frag in args.fragments:

            base_rois = (
                make_fragmented_rois(
                    H,
                    W,
                    ratio,
                    n_frag
                )
            )

            useful_area = (
                mask_union_area(
                    base_rois,
                    H,
                    W
                )
            )

            useful_ratio = (
                useful_area
                / full_area
            )

            for strategy in strategies:

                packed_rois = (
                    get_packed_rois(
                        base_rois,
                        strategy,
                        args.halo,
                        H,
                        W,
                        args.align
                    )
                )

                packed_area = (
                    mask_union_area(
                        packed_rois,
                        H,
                        W
                    )
                )

                exec_sum_area = (
                    summed_execution_area(
                        packed_rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )
                )

                exec_unique_area = (
                    unique_execution_area(
                        packed_rois,
                        args.halo,
                        H,
                        W,
                        args.align
                    )
                )

                run_roi = (
                    build_roi_execution_fn(
                        backbone,
                        full_input,
                        output_canvas,
                        packed_rois,
                        args.halo,
                        args.align
                    )
                )

                timing = cuda_measure(
                    run_roi,
                    args.warmup,
                    args.repeat
                )

                speedup = (
                    full_mean
                    / timing[
                        "mean_ms"
                    ]
                )

                row = {
                    "target_ratio":
                        float(
                            ratio
                        ),

                    "fragments":
                        int(
                            n_frag
                        ),

                    "strategy":
                        strategy,

                    "num_execution_rois":
                        len(
                            packed_rois
                        ),

                    "base_rois":
                        [
                            list(x)
                            for x in base_rois
                        ],

                    "packed_rois":
                        [
                            list(x)
                            for x in packed_rois
                        ],

                    "useful_area_ratio":
                        float(
                            useful_ratio
                        ),

                    "packed_core_area_ratio":
                        float(
                            packed_area
                            / full_area
                        ),

                    "summed_execution_area_ratio":
                        float(
                            exec_sum_area
                            / full_area
                        ),

                    "unique_execution_area_ratio":
                        float(
                            exec_unique_area
                            / full_area
                        ),

                    **timing,

                    "speedup_vs_full":
                        float(
                            speedup
                        ),
                }

                records.append(
                    row
                )

                print(
                    f"{ratio*100:6.1f}% "
                    f"{n_frag:5d} "
                    f"{strategy:>10s} "
                    f"{len(packed_rois):4d} "
                    f"{useful_ratio*100:8.2f} "
                    f"{packed_area/full_area*100:8.2f} "
                    f"{exec_sum_area/full_area*100:9.2f} "
                    f"{exec_unique_area/full_area*100:10.2f} "
                    f"{timing['mean_ms']:10.3f} "
                    f"{timing['p95_ms']:8.3f} "
                    f"{timing['p99_ms']:8.3f} "
                    f"{speedup:8.3f}"
                )

    # --------------------------------------------------------
    # Summary: naive fragmentation overhead
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "Naive fragmentation summary"
    )
    print("=" * 90)

    for ratio in args.ratios:

        selected = [
            r
            for r in records
            if (
                abs(
                    r[
                        "target_ratio"
                    ]
                    - ratio
                )
                < 1e-9
                and r[
                    "strategy"
                ]
                == "naive"
            )
        ]

        selected.sort(
            key=lambda x:
            x["fragments"]
        )

        print()
        print(
            f"Target recompute ratio: "
            f"{ratio*100:.1f}%"
        )

        base_latency = (
            selected[0][
                "mean_ms"
            ]
        )

        for r in selected:

            frag_overhead = (
                r["mean_ms"]
                / base_latency
            )

            print(
                f"  fragments={r['fragments']:2d}  "
                f"ROI={r['num_execution_rois']:2d}  "
                f"mean={r['mean_ms']:7.3f} ms  "
                f"vs-1ROI={frag_overhead:6.3f}x  "
                f"exec-area="
                f"{r['summed_execution_area_ratio']*100:6.2f}%"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {
        "gpu":
            torch.cuda.get_device_name(
                0
            ),

        "d_input_shape":
            list(
                full_input.shape
            ),

        "d_output_shape":
            list(
                full_output.shape
            ),

        "halo":
            args.halo,

        "align":
            args.align,

        "warmup":
            args.warmup,

        "repeat":
            args.repeat,

        "full_d":
            full_stats,

        "records":
            records,
    }

    output_path.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()
