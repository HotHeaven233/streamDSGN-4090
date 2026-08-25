#!/usr/bin/env python3

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

import profile_variable_latency as base

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils import model_nms_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser(
        "Detailed variable-latency analysis for StreamDSGN"
    )

    p.add_argument(
        "-n", "--num_frames",
        type=int,
        required=True
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=50
    )

    p.add_argument(
        "--cfg_file",
        default=DEFAULT_CFG
    )

    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT
    )

    p.add_argument(
        "--output_dir",
        default="outputs/variable_latency_profile_v2"
    )

    p.add_argument(
        "--log_every",
        type=int,
        default=50
    )

    p.add_argument(
        "--repeat_fa_frame",
        type=int,
        default=100,
        help="frame used for same-input FeatureAlignment repeat test"
    )

    p.add_argument(
        "--repeat_fa_times",
        type=int,
        default=300
    )

    p.add_argument(
        "--repeat_fa_warmup",
        type=int,
        default=30
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1024
    )

    args = p.parse_args()

    if args.num_frames <= 0:
        p.error("-n must be > 0")

    return args


def new_event():
    return torch.cuda.Event(
        enable_timing=True
    )


def safe_name(x):
    return "".join(
        c if c.isalnum() else "_"
        for c in str(x)
    )


def add_cpu_time(profiler, name, value):
    profiler.current[name] = (
        profiler.current.get(name, 0.0)
        + float(value)
    )


# ============================================================
# Detailed warp profiler
# ============================================================

def install_detailed_warp(profiler):
    """
    Replace only FeatureAlignment.warp_feature during profiling.

    Semantics are kept equivalent to the original implementation,
    while the operation is split into fine-grained timing regions.
    """

    original_warp = profiler.orig_warp
    fa = profiler.fa

    def detailed_warp(x, flow):

        if profiler.mode == "features":
            profiler.current[
                "fa_flow"
            ] = flow.detach()

            return original_warp(
                x,
                flow
            )

        if profiler.mode != "timing":
            return original_warp(
                x,
                flow
            )

        total_start = new_event()
        total_end = new_event()

        total_start.record()

        # ----------------------------------------------------
        # FP16 -> FP32
        # ----------------------------------------------------
        use_amp = (
            x.dtype == torch.float16
        )

        if use_amp:
            x, flow = profiler.timed_call(
                "fa_warp_cast_fp32",
                lambda a, b: (
                    a.float(),
                    b.float()
                ),
                x,
                flow
            )

        B, C, H, W = x.size()

        profiler.current[
            "fa_warp_B"
        ] = int(B)

        profiler.current[
            "fa_warp_C"
        ] = int(C)

        profiler.current[
            "fa_warp_H"
        ] = int(H)

        profiler.current[
            "fa_warp_W"
        ] = int(W)

        # ----------------------------------------------------
        # CPU grid creation
        # ----------------------------------------------------
        cpu_t0 = time.perf_counter()

        xx = (
            torch.arange(0, W)
            .view(1, -1)
            .repeat(H, 1)
        )

        yy = (
            torch.arange(0, H)
            .view(-1, 1)
            .repeat(1, W)
        )

        xx = (
            xx.view(1, 1, H, W)
            .repeat(B, 1, 1, 1)
        )

        yy = (
            yy.view(1, 1, H, W)
            .repeat(B, 1, 1, 1)
        )

        grid = torch.cat(
            (xx, yy),
            1
        ).float()

        add_cpu_time(
            profiler,
            "fa_warp_grid_cpu_ms",
            (
                time.perf_counter()
                - cpu_t0
            ) * 1000.0
        )

        # ----------------------------------------------------
        # grid CPU -> GPU
        # This is inside model.forward(), therefore included
        # in the user's definition of model execution.
        # ----------------------------------------------------
        if x.is_cuda:
            host_t0 = time.perf_counter()

            grid = profiler.timed_call(
                "fa_warp_grid_h2d",
                lambda z: z.cuda(),
                grid
            )

            add_cpu_time(
                profiler,
                "fa_warp_grid_h2d_host_ms",
                (
                    time.perf_counter()
                    - host_t0
                ) * 1000.0
            )

        # ----------------------------------------------------
        # Build normalized sampling grid
        # ----------------------------------------------------
        def build_vgrid(grid, flow):
            vgrid = grid + flow

            vgrid[:, 0, :, :] = (
                2.0
                * vgrid[:, 0, :, :]
                / max(W - 1, 1)
                - 1.0
            )

            vgrid[:, 1, :, :] = (
                2.0
                * vgrid[:, 1, :, :]
                / max(H - 1, 1)
                - 1.0
            )

            vgrid = vgrid.permute(
                0, 2, 3, 1
            )

            return vgrid

        vgrid = profiler.timed_call(
            "fa_warp_build_vgrid",
            build_vgrid,
            grid,
            flow
        )

        # ----------------------------------------------------
        # Feature grid_sample
        # ----------------------------------------------------
        x_warp = profiler.timed_call(
            "fa_warp_grid_sample_feature",
            lambda a, g:
                F.grid_sample(
                    a,
                    g,
                    padding_mode="zeros"
                ),
            x,
            vgrid
        )

        # ----------------------------------------------------
        # Mask CPU allocation
        # Original implementation:
        #
        # torch.ones(x.size()).cuda()
        #
        # We separate allocation and H2D for diagnosis.
        # ----------------------------------------------------
        cpu_t0 = time.perf_counter()

        mask = torch.ones(
            x.size(),
            requires_grad=False
        )

        add_cpu_time(
            profiler,
            "fa_warp_mask_cpu_ms",
            (
                time.perf_counter()
                - cpu_t0
            ) * 1000.0
        )

        if x.is_cuda:
            host_t0 = time.perf_counter()

            mask = profiler.timed_call(
                "fa_warp_mask_h2d",
                lambda z: z.cuda(),
                mask
            )

            add_cpu_time(
                profiler,
                "fa_warp_mask_h2d_host_ms",
                (
                    time.perf_counter()
                    - host_t0
                ) * 1000.0
            )

        # ----------------------------------------------------
        # Mask grid_sample
        # ----------------------------------------------------
        mask = profiler.timed_call(
            "fa_warp_grid_sample_mask",
            lambda a, g:
                F.grid_sample(
                    a,
                    g
                ),
            mask,
            vgrid
        )

        # ----------------------------------------------------
        # Threshold mask
        # ----------------------------------------------------
        mask = profiler.timed_call(
            "fa_warp_mask_threshold",
            lambda a:
                (a >= 1.0).float(),
            mask
        )

        # ----------------------------------------------------
        # Cast back
        # ----------------------------------------------------
        if use_amp:
            x_warp, mask = profiler.timed_call(
                "fa_warp_cast_fp16",
                lambda a, b: (
                    a.half(),
                    b.half()
                ),
                x_warp,
                mask
            )

        # ----------------------------------------------------
        # final multiply
        # ----------------------------------------------------
        output = profiler.timed_call(
            "fa_warp_output_mul",
            lambda a, b: a * b,
            x_warp,
            mask
        )

        total_end.record()

        profiler.records[
            "fa_warp"
        ].append(
            (
                total_start,
                total_end
            )
        )

        return output

    fa.warp_feature = detailed_warp


# ============================================================
# Per-class post-processing profiler
# ============================================================

def install_per_class_nms_profiler(
    profiler,
    class_names
):
    """
    Replace multi_classes_nms with an equivalent implementation
    that separately records each class's candidate count and time.
    """

    original_multi = (
        model_nms_utils
        .multi_classes_nms
    )

    def profiled_multi_classes_nms(
        cls_scores,
        box_preds,
        nms_config,
        score_thresh=None,
        label_preds=None
    ):
        pred_scores = []
        pred_labels = []
        pred_boxes = []

        if label_preds is None:
            num_classes = (
                cls_scores.shape[1]
            )
        else:
            num_classes = (
                label_preds.max().item()
            )

        for k in range(
            num_classes
        ):
            stage_name = (
                f"post_class_{k}"
            )

            if profiler.mode == "timing":
                stage_start = new_event()
                stage_end = new_event()

                stage_start.record()

            if score_thresh is not None:

                if label_preds is None:
                    scores_mask = (
                        cls_scores[:, k]
                        >= score_thresh
                    )

                    box_scores = (
                        cls_scores[
                            scores_mask,
                            k
                        ]
                    )

                    cur_box_preds = (
                        box_preds[
                            scores_mask
                        ]
                    )

                else:
                    scores_mask = (
                        (
                            cls_scores[:, 0]
                            >= score_thresh
                        )
                        &
                        (
                            label_preds
                            == k + 1
                        )
                    )

                    box_scores = (
                        cls_scores[
                            scores_mask,
                            0
                        ]
                    )

                    cur_box_preds = (
                        box_preds[
                            scores_mask
                        ]
                    )

            else:
                raise NotImplementedError

            candidate_count = int(
                box_scores.shape[0]
            )

            selected = []

            if candidate_count > 0:
                (
                    box_scores_nms,
                    indices
                ) = torch.topk(
                    box_scores,
                    k=min(
                        nms_config.NMS_PRE_MAXSIZE,
                        candidate_count
                    )
                )

                boxes_for_nms = (
                    cur_box_preds[
                        indices
                    ]
                )

                (
                    keep_idx,
                    selected_scores
                ) = getattr(
                    iou3d_nms_utils,
                    nms_config.NMS_TYPE
                )(
                    boxes_for_nms[:, 0:7],
                    box_scores_nms,
                    nms_config.NMS_THRESH,
                    **nms_config
                )

                selected = indices[
                    keep_idx[
                        :nms_config
                        .NMS_POST_MAXSIZE
                    ]
                ]

            keep_count = len(
                selected
            )

            if profiler.mode == "timing":
                stage_end.record()

                profiler.records[
                    stage_name
                ].append(
                    (
                        stage_start,
                        stage_end
                    )
                )

                profiler.current.setdefault(
                    "class_candidates",
                    {}
                )[k] = candidate_count

                profiler.current.setdefault(
                    "class_keeps",
                    {}
                )[k] = keep_count

            pred_scores.append(
                box_scores[
                    selected
                ]
            )

            pred_labels.append(
                box_scores.new_ones(
                    len(selected)
                ).long() * k
            )

            pred_boxes.append(
                cur_box_preds[
                    selected
                ]
            )

        pred_scores = torch.cat(
            pred_scores,
            dim=0
        )

        pred_labels = torch.cat(
            pred_labels,
            dim=0
        )

        pred_boxes = torch.cat(
            pred_boxes,
            dim=0
        )

        return (
            pred_scores,
            pred_labels,
            pred_boxes
        )

    model_nms_utils.multi_classes_nms = (
        profiled_multi_classes_nms
    )

    return original_multi


# ============================================================
# Regression
# ============================================================

def regression_metrics(
    y,
    pred
):
    err = pred - y
    abs_err = np.abs(err)

    mae = float(
        abs_err.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                err ** 2
            )
        )
    )

    p95 = float(
        np.percentile(
            abs_err,
            95
        )
    )

    p99 = float(
        np.percentile(
            abs_err,
            99
        )
    )

    denom = np.sum(
        (
            y - y.mean()
        ) ** 2
    )

    r2 = (
        1.0
        - np.sum(err ** 2)
        / denom
        if denom > 0
        else float("nan")
    )

    return {
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "P95_AE_ms": p95,
        "P99_AE_ms": p99,
        "R2": float(r2),
    }


def cross_validate_linear(
    X,
    y,
    folds=5,
    seed=1024
):
    rng = np.random.default_rng(
        seed
    )

    n = len(y)

    order = np.arange(n)
    rng.shuffle(order)

    fold_ids = np.array_split(
        order,
        folds
    )

    all_y = []
    all_pred = []

    for test_idx in fold_ids:
        train_mask = np.ones(
            n,
            dtype=bool
        )

        train_mask[
            test_idx
        ] = False

        train_idx = np.where(
            train_mask
        )[0]

        Xtr = X[train_idx]
        Xte = X[test_idx]

        ytr = y[train_idx]
        yte = y[test_idx]

        mean = Xtr.mean(
            axis=0
        )

        std = Xtr.std(
            axis=0
        )

        std[
            std < 1e-9
        ] = 1.0

        Xtr_n = (
            Xtr - mean
        ) / std

        Xte_n = (
            Xte - mean
        ) / std

        Xtr_d = np.concatenate(
            [
                np.ones(
                    (
                        len(Xtr_n),
                        1
                    )
                ),
                Xtr_n
            ],
            axis=1
        )

        Xte_d = np.concatenate(
            [
                np.ones(
                    (
                        len(Xte_n),
                        1
                    )
                ),
                Xte_n
            ],
            axis=1
        )

        beta = np.linalg.lstsq(
            Xtr_d,
            ytr,
            rcond=None
        )[0]

        pred = (
            Xte_d @ beta
        )

        all_y.append(
            yte
        )

        all_pred.append(
            pred
        )

    all_y = np.concatenate(
        all_y
    )

    all_pred = np.concatenate(
        all_pred
    )

    return regression_metrics(
        all_y,
        all_pred
    )


def evaluate_post_models(
    rows,
    class_names,
    seed
):
    y = np.asarray(
        [
            float(
                r[
                    "post_latency_ms"
                ]
            )
            for r in rows
        ],
        dtype=np.float64
    )

    class_columns = [
        "post_candidate_"
        + safe_name(name)
        for name in class_names
    ]

    X_class = np.asarray(
        [
            [
                float(
                    r.get(
                        col,
                        0
                    )
                )
                for col in class_columns
            ]
            for r in rows
        ],
        dtype=np.float64
    )

    total = X_class.sum(
        axis=1,
        keepdims=True
    )

    X_quad = np.concatenate(
        [
            X_class,
            X_class ** 2
        ],
        axis=1
    )

    X_pair = [
        X_class
    ]

    if X_class.shape[1] >= 2:
        pair_terms = []

        for i in range(
            X_class.shape[1]
        ):
            for j in range(
                i + 1,
                X_class.shape[1]
            ):
                pair_terms.append(
                    (
                        X_class[:, i]
                        * X_class[:, j]
                    )[:, None]
                )

        X_pair.append(
            np.concatenate(
                pair_terms,
                axis=1
            )
        )

    X_pair = np.concatenate(
        X_pair,
        axis=1
    )

    models = {
        "total_linear":
            total,

        "per_class_linear":
            X_class,

        "per_class_quadratic":
            X_quad,

        "per_class_interactions":
            X_pair,
    }

    results = {}

    print()
    print("=" * 90)
    print("Post-processing execution-time prediction: 5-fold CV")
    print("=" * 90)

    print(
        f"{'Model':28s}"
        f"{'MAE(ms)':>11s}"
        f"{'RMSE(ms)':>12s}"
        f"{'P95AE(ms)':>12s}"
        f"{'P99AE(ms)':>12s}"
        f"{'R2':>10s}"
    )

    print("-" * 85)

    for name, X in models.items():
        m = cross_validate_linear(
            X,
            y,
            folds=5,
            seed=seed
        )

        results[name] = m

        print(
            f"{name:28s}"
            f"{m['MAE_ms']:11.4f}"
            f"{m['RMSE_ms']:12.4f}"
            f"{m['P95_AE_ms']:12.4f}"
            f"{m['P99_AE_ms']:12.4f}"
            f"{m['R2']:10.4f}"
        )

    return results


# ============================================================
# Same-input FeatureAlignment repeat
# ============================================================

def capture_fa_input(
    model,
    dataset,
    profiler,
    frame_idx
):
    base.clear_history(
        model
    )

    captured = None

    with torch.no_grad():
        for i in range(
            frame_idx + 1
        ):
            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            load_data_to_gpu(
                batch
            )

            profiler.reset_frame()
            profiler.mode = "features"

            model(
                batch
            )

            profiler.mode = "off"

            if i == frame_idx:
                history_len = int(
                    profiler.current.get(
                        "history_len",
                        0
                    )
                )

                if history_len == 0:
                    raise RuntimeError(
                        f"frame {frame_idx} has no history; "
                        "choose a non-sequence-start frame"
                    )

                cur = (
                    profiler.current[
                        "fa_cur_feature"
                    ]
                    .detach()
                    .clone()
                )

                last = (
                    profiler.current[
                        "fa_last_feature"
                    ]
                    .detach()
                    .clone()
                )

                captured = (
                    cur,
                    last
                )

    torch.cuda.synchronize()

    return captured


def same_input_fa_test(
    model,
    profiler,
    cur,
    last,
    warmup,
    repeats
):
    feature_name = (
        profiler.fa
        .fusion_features_name[0]
    )

    def make_input():
        return {
            feature_name: cur,
            "history_features": [
                (
                    "fixed",
                    {
                        feature_name:
                            last
                    }
                )
            ]
        }

    print()
    print(
        f"[same-input FA] "
        f"warmup={warmup}, repeats={repeats}"
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=model
            .use_amp_dict["TEST"]
        ):
            for _ in range(
                warmup
            ):
                profiler.mode = "off"

                profiler.fa(
                    make_input()
                )

    torch.cuda.synchronize()

    records = []

    with torch.no_grad():
        with torch.cuda.amp.autocast(
            enabled=model
            .use_amp_dict["TEST"]
        ):
            for _ in range(
                repeats
            ):
                profiler.reset_frame()

                profiler.mode = "timing"

                profiler.fa(
                    make_input()
                )

                profiler.mode = "off"

                torch.cuda.synchronize()

                times = (
                    profiler.elapsed()
                )

                row = {
                    "fa_latency_ms":
                        times.get(
                            "FeatureAlignment",
                            float("nan")
                        )
                }

                warp_names = [
                    "fa_warp",
                    "fa_warp_cast_fp32",
                    "fa_warp_grid_h2d",
                    "fa_warp_build_vgrid",
                    "fa_warp_grid_sample_feature",
                    "fa_warp_mask_h2d",
                    "fa_warp_grid_sample_mask",
                    "fa_warp_mask_threshold",
                    "fa_warp_cast_fp16",
                    "fa_warp_output_mul",
                ]

                for name in warp_names:
                    row[
                        name + "_ms"
                    ] = times.get(
                        name,
                        0.0
                    )

                row[
                    "fa_warp_grid_cpu_ms"
                ] = profiler.current.get(
                    "fa_warp_grid_cpu_ms",
                    0.0
                )

                row[
                    "fa_warp_mask_cpu_ms"
                ] = profiler.current.get(
                    "fa_warp_mask_cpu_ms",
                    0.0
                )

                records.append(
                    row
                )

    print()
    print("=" * 90)
    print("Same-input FeatureAlignment repeat")
    print("=" * 90)

    keys = [
        "fa_latency_ms",
        "fa_warp_ms",
        "fa_warp_cast_fp32_ms",
        "fa_warp_grid_cpu_ms",
        "fa_warp_grid_h2d_ms",
        "fa_warp_build_vgrid_ms",
        "fa_warp_grid_sample_feature_ms",
        "fa_warp_mask_cpu_ms",
        "fa_warp_mask_h2d_ms",
        "fa_warp_grid_sample_mask_ms",
        "fa_warp_mask_threshold_ms",
        "fa_warp_cast_fp16_ms",
        "fa_warp_output_mul_ms",
    ]

    print(
        f"{'Stage':38s}"
        f"{'Mean':>10s}"
        f"{'P50':>10s}"
        f"{'P95':>10s}"
        f"{'P99':>10s}"
        f"{'CV':>9s}"
    )

    print("-" * 87)

    summary = {}

    for key in keys:
        vals = np.asarray(
            [
                float(
                    r.get(
                        key,
                        0.0
                    )
                )
                for r in records
            ],
            dtype=np.float64
        )

        mean = float(
            vals.mean()
        )

        std = float(
            vals.std()
        )

        cv = (
            std / mean
            if mean > 1e-12
            else 0.0
        )

        s = {
            "mean_ms": mean,
            "p50_ms": float(
                np.percentile(
                    vals,
                    50
                )
            ),
            "p95_ms": float(
                np.percentile(
                    vals,
                    95
                )
            ),
            "p99_ms": float(
                np.percentile(
                    vals,
                    99
                )
            ),
            "std_ms": std,
            "cv": cv,
        }

        summary[key] = s

        print(
            f"{key:38s}"
            f"{s['mean_ms']:10.4f}"
            f"{s['p50_ms']:10.4f}"
            f"{s['p95_ms']:10.4f}"
            f"{s['p99_ms']:10.4f}"
            f"{s['cv']:9.4f}"
        )

    return records, summary


# ============================================================
# CSV
# ============================================================

def save_csv(
    rows,
    path
):
    keys = set()

    for r in rows:
        keys.update(
            r.keys()
        )

    priority = [
        "frame_idx",
        "scene",
        "frame_id",
        "model_latency_ms",
        "fa_latency_ms",
        "post_latency_ms",
    ]

    fields = (
        priority
        +
        sorted(
            k
            for k in keys
            if k not in priority
        )
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields
        )

        w.writeheader()
        w.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    torch.backends.cudnn.benchmark = True

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

    profiler = base.Profiler(
        model
    )

    install_detailed_warp(
        profiler
    )

    original_multi = (
        install_per_class_nms_profiler(
            profiler,
            cfg.CLASS_NAMES
        )
    )

    base.warmup(
        model,
        dataset,
        args.warmup
    )

    N = min(
        args.num_frames,
        len(dataset)
    )

    rows = [
        base.get_frame_meta(
            dataset,
            i
        )
        for i in range(N)
    ]

    # ========================================================
    # PASS 1
    # ========================================================

    print()
    print(
        "PASS 1/2: detailed model latency"
    )

    base.clear_history(
        model
    )

    with torch.no_grad():
        for i in range(N):

            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            # Explicitly outside model latency.
            load_data_to_gpu(
                batch
            )

            torch.cuda.synchronize()

            profiler.reset_frame()

            model_start = new_event()
            model_end = new_event()

            profiler.mode = "timing"

            model_start.record()

            model(
                batch
            )

            model_end.record()

            profiler.mode = "off"

            torch.cuda.synchronize()

            times = (
                profiler.elapsed()
            )

            row = rows[i]

            row[
                "model_latency_ms"
            ] = float(
                model_start.elapsed_time(
                    model_end
                )
            )

            row[
                "fa_latency_ms"
            ] = times.get(
                "FeatureAlignment",
                float("nan")
            )

            row[
                "post_latency_ms"
            ] = times.get(
                "post_processing",
                float("nan")
            )

            # -----------------------------------------------
            # FA top-level
            # -----------------------------------------------

            fa_names = [
                "fa_pool",
                "fa_fusion_layer",
                "fa_offset_matching",
                "fa_feature_sweeping",
                "fa_shift_coord",
                "fa_upsample",
                "fa_warp",
            ]

            for name in fa_names:
                row[
                    name + "_ms"
                ] = times.get(
                    name,
                    0.0
                )

            # -----------------------------------------------
            # Detailed warp
            # -----------------------------------------------

            warp_gpu_names = [
                "fa_warp_cast_fp32",
                "fa_warp_grid_h2d",
                "fa_warp_build_vgrid",
                "fa_warp_grid_sample_feature",
                "fa_warp_mask_h2d",
                "fa_warp_grid_sample_mask",
                "fa_warp_mask_threshold",
                "fa_warp_cast_fp16",
                "fa_warp_output_mul",
            ]

            for name in warp_gpu_names:
                row[
                    name + "_ms"
                ] = times.get(
                    name,
                    0.0
                )

            warp_cpu_names = [
                "fa_warp_grid_cpu_ms",
                "fa_warp_grid_h2d_host_ms",
                "fa_warp_mask_cpu_ms",
                "fa_warp_mask_h2d_host_ms",
            ]

            for name in warp_cpu_names:
                row[name] = (
                    profiler.current.get(
                        name,
                        0.0
                    )
                )

            row[
                "fa_warp_cpu_total_ms"
            ] = (
                row[
                    "fa_warp_grid_cpu_ms"
                ]
                +
                row[
                    "fa_warp_mask_cpu_ms"
                ]
            )

            # -----------------------------------------------
            # Post-processing
            # -----------------------------------------------

            row[
                "post_nms_ms"
            ] = times.get(
                "post_nms",
                0.0
            )

            class_candidates = (
                profiler.current.get(
                    "class_candidates",
                    {}
                )
            )

            class_keeps = (
                profiler.current.get(
                    "class_keeps",
                    {}
                )
            )

            total_candidates = 0

            for k, class_name in enumerate(
                cfg.CLASS_NAMES
            ):
                tag = safe_name(
                    class_name
                )

                candidate = int(
                    class_candidates.get(
                        k,
                        0
                    )
                )

                keep = int(
                    class_keeps.get(
                        k,
                        0
                    )
                )

                class_ms = times.get(
                    f"post_class_{k}",
                    0.0
                )

                row[
                    f"post_candidate_{tag}"
                ] = candidate

                row[
                    f"post_keep_{tag}"
                ] = keep

                row[
                    f"post_class_{tag}_ms"
                ] = class_ms

                total_candidates += (
                    candidate
                )

            row[
                "post_candidate_total"
            ] = total_candidates

            row[
                "history_len"
            ] = int(
                profiler.current.get(
                    "history_len",
                    0
                )
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
                    f"[timing {i+1:4d}/{N}] "
                    f"model={row['model_latency_ms']:7.3f} "
                    f"FA={row['fa_latency_ms']:7.3f} "
                    f"warp={row['fa_warp_ms']:7.3f} "
                    f"post={row['post_latency_ms']:6.3f} "
                    f"cand={total_candidates}"
                )

    # ========================================================
    # PASS 2
    # ========================================================

    print()
    print(
        "PASS 2/2: FA workload features"
    )

    base.clear_history(
        model
    )

    with torch.no_grad():
        for i in range(N):

            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            load_data_to_gpu(
                batch
            )

            torch.cuda.synchronize()

            profiler.reset_frame()

            profiler.mode = "features"

            model(
                batch
            )

            profiler.mode = "off"

            torch.cuda.synchronize()

            rows[i].update(
                base.extract_fa_features(
                    profiler
                )
            )

            # Score-distribution features are still useful
            # for post-processing correlation analysis.
            rows[i].update(
                base.extract_post_features(
                    profiler,
                    model
                )
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
                    f"[feature {i+1:4d}/{N}] "
                    f"flow="
                    f"{rows[i].get('fa_flow_l2_mean', float('nan')):.3f}"
                )

    # ========================================================
    # Correlation tables
    # ========================================================

    warp_features = [
        "fa_warp_cast_fp32_ms",
        "fa_warp_grid_cpu_ms",
        "fa_warp_grid_h2d_ms",
        "fa_warp_grid_h2d_host_ms",
        "fa_warp_build_vgrid_ms",
        "fa_warp_grid_sample_feature_ms",
        "fa_warp_mask_cpu_ms",
        "fa_warp_mask_h2d_ms",
        "fa_warp_mask_h2d_host_ms",
        "fa_warp_grid_sample_mask_ms",
        "fa_warp_mask_threshold_ms",
        "fa_warp_cast_fp16_ms",
        "fa_warp_output_mul_ms",
    ]

    base.print_corr_table(
        "FA latency vs detailed warp substages",
        rows,
        "fa_latency_ms",
        warp_features
    )

    base.print_corr_table(
        "Warp latency vs detailed warp substages",
        rows,
        "fa_warp_ms",
        warp_features
    )

    fa_content_features = [
        "history_len",
        "history_valid",
        "fa_flow_abs_mean",
        "fa_flow_l2_mean",
        "fa_flow_l2_p95",
        "fa_flow_max",
        "fa_flow_nonzero_ratio",
        "fa_flow_boundary_ratio",
        "fa_match_conf_mean",
        "fa_match_conf_p10",
        "fa_match_conf_p90",
        "fa_match_entropy_mean",
        "fa_feature_diff_l1",
        "fa_feature_cosine",
    ]

    base.print_corr_table(
        "FA latency vs input/state features",
        rows,
        "fa_latency_ms",
        fa_content_features
    )

    post_features = [
        "post_nms_ms",
        "post_candidate_total",
    ]

    for name in cfg.CLASS_NAMES:
        tag = safe_name(
            name
        )

        post_features.extend(
            [
                f"post_candidate_{tag}",
                f"post_keep_{tag}",
                f"post_class_{tag}_ms",
            ]
        )

    base.print_corr_table(
        "Post-processing latency correlations",
        rows,
        "post_latency_ms",
        post_features
    )

    # Per-class candidate -> class latency.
    for name in cfg.CLASS_NAMES:
        tag = safe_name(
            name
        )

        base.print_corr_table(
            f"{name}: candidate count vs class NMS latency",
            rows,
            f"post_class_{tag}_ms",
            [
                f"post_candidate_{tag}",
                f"post_keep_{tag}",
            ]
        )

    # ========================================================
    # Prediction models
    # ========================================================

    regression_results = (
        evaluate_post_models(
            rows,
            cfg.CLASS_NAMES,
            args.seed
        )
    )

    # ========================================================
    # Same-input FA experiment
    # ========================================================

    repeat_idx = min(
        args.repeat_fa_frame,
        len(dataset) - 1
    )

    print()
    print(
        f"Capturing FA input at frame "
        f"{repeat_idx}"
    )

    cur, last = capture_fa_input(
        model,
        dataset,
        profiler,
        repeat_idx
    )

    (
        repeat_records,
        repeat_summary
    ) = same_input_fa_test(
        model,
        profiler,
        cur,
        last,
        args.repeat_fa_warmup,
        args.repeat_fa_times
    )

    # ========================================================
    # Save
    # ========================================================

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        out_dir
        / f"profile_v2_{N}_frames.csv"
    )

    json_path = (
        out_dir
        / f"profile_v2_{N}_frames.json"
    )

    repeat_json = (
        out_dir
        / "same_input_fa_repeat.json"
    )

    regression_json = (
        out_dir
        / "post_regression.json"
    )

    save_csv(
        rows,
        csv_path
    )

    with open(
        json_path,
        "w"
    ) as f:
        json.dump(
            rows,
            f,
            indent=2
        )

    with open(
        repeat_json,
        "w"
    ) as f:
        json.dump(
            {
                "frame_idx":
                    repeat_idx,
                "repeats":
                    repeat_records,
                "summary":
                    repeat_summary,
            },
            f,
            indent=2
        )

    with open(
        regression_json,
        "w"
    ) as f:
        json.dump(
            regression_results,
            f,
            indent=2
        )

    model_nms_utils.multi_classes_nms = (
        original_multi
    )

    profiler.close()

    print()
    print("Saved:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", repeat_json)
    print(" ", regression_json)


if __name__ == "__main__":
    main()
