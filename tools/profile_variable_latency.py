#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr, spearmanr

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.ops.feature_flow import feature_flow_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        "Profile FeatureAlignment and post-processing latency variation"
    )

    parser.add_argument(
        "-n", "--num_frames",
        type=int,
        required=True
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30
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
        "--output_dir",
        default="outputs/variable_latency_profile"
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024
    )

    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("-n must be > 0")

    return args


def clear_history(model):
    q = getattr(model, "history_feature_queue", None)
    if q is not None:
        q.clear()


def new_event():
    return torch.cuda.Event(enable_timing=True)


class Profiler:
    def __init__(self, model):
        self.model = model

        self.mode = "off"
        self.records = defaultdict(list)
        self.current = {}

        self.handles = []

        self.orig_offset_matching = feature_flow_utils.offset_matching_gpu
        self.orig_get_shift_coord = feature_flow_utils.get_shift_coord

        self.nms_name = model.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_TYPE
        self.orig_nms = getattr(
            iou3d_nms_utils,
            self.nms_name
        )

        self.orig_post = model.post_processing

        self.fa = None

        for m in model.fusion_module:
            if type(m).__name__ == "FeatureAlignment":
                self.fa = m
                break

        if self.fa is None:
            raise RuntimeError("FeatureAlignment not found")

        self.orig_warp = self.fa.warp_feature

        self._install()

    def reset_frame(self):
        self.records = defaultdict(list)
        self.current = {
            "nms_inputs": [],
            "nms_outputs": [],
        }

    def event_start(self, name):
        s = new_event()
        s.record()
        self.current.setdefault("_event_stack", defaultdict(list))
        self.current["_event_stack"][name].append(s)

    def event_end(self, name):
        e = new_event()
        e.record()

        s = self.current["_event_stack"][name].pop()

        self.records[name].append((s, e))

    def timed_call(self, name, fn, *args, **kwargs):
        if self.mode != "timing":
            return fn(*args, **kwargs)

        s = new_event()
        e = new_event()

        s.record()
        out = fn(*args, **kwargs)
        e.record()

        self.records[name].append((s, e))

        return out

    def _install(self):
        #
        # FeatureAlignment total
        #
        def fa_pre_hook(module, inputs):
            if self.mode == "off":
                return

            batch_dict = inputs[0]

            history = batch_dict.get(
                "history_features", []
            )

            self.current["history_len"] = len(history)

            if self.mode == "timing":
                self.event_start("FeatureAlignment")

            if self.mode == "features":
                name = module.fusion_features_name[0]

                cur = batch_dict[name]

                self.current["fa_cur_feature"] = cur.detach()

                if len(history) > 0:
                    last = history[-1][1][name]
                    self.current["fa_last_feature"] = last.detach()

        def fa_post_hook(module, inputs, output):
            if self.mode == "timing":
                self.event_end("FeatureAlignment")

        self.handles.append(
            self.fa.register_forward_pre_hook(
                fa_pre_hook
            )
        )

        self.handles.append(
            self.fa.register_forward_hook(
                fa_post_hook
            )
        )

        #
        # FeatureAlignment nn.Module substages
        #
        def register_submodule(name, module):
            if module is None:
                return

            def pre(_m, _inputs):
                if self.mode == "timing":
                    self.event_start(name)

            def post(_m, _inputs, _output):
                if self.mode == "timing":
                    self.event_end(name)

            self.handles.append(
                module.register_forward_pre_hook(pre)
            )

            self.handles.append(
                module.register_forward_hook(post)
            )

        register_submodule(
            "fa_pool",
            self.fa.pool
        )

        register_submodule(
            "fa_fusion_layer",
            getattr(self.fa, "fusion_layer", None)
        )

        register_submodule(
            "fa_upsample",
            self.fa.interp_coord
        )

        #
        # feature_sweeping CUDA extension
        #
        self.orig_sweep = (
            feature_flow_utils
            .feature_flow_cuda
            .feature_sweeping_gpu
        )

        def sweep_wrapper(*args, **kwargs):
            return self.timed_call(
                "fa_feature_sweeping",
                self.orig_sweep,
                *args,
                **kwargs
            )

        feature_flow_utils.feature_flow_cuda.feature_sweeping_gpu = (
            sweep_wrapper
        )

        #
        # offset matching
        #
        def offset_wrapper(*args, **kwargs):
            if self.mode == "timing":
                return self.timed_call(
                    "fa_offset_matching",
                    self.orig_offset_matching,
                    *args,
                    **kwargs
                )

            out = self.orig_offset_matching(
                *args,
                **kwargs
            )

            if self.mode == "features":
                self.current[
                    "fa_matching_dist"
                ] = out.detach()

            return out

        feature_flow_utils.offset_matching_gpu = offset_wrapper

        #
        # shift coord
        #
        def shift_wrapper(*args, **kwargs):
            if self.mode == "timing":
                return self.timed_call(
                    "fa_shift_coord",
                    self.orig_get_shift_coord,
                    *args,
                    **kwargs
                )

            out = self.orig_get_shift_coord(
                *args,
                **kwargs
            )

            if self.mode == "features":
                self.current[
                    "fa_shift_coord_raw"
                ] = out.detach()

            return out

        feature_flow_utils.get_shift_coord = shift_wrapper

        #
        # final warp
        #
        def warp_wrapper(x, flow):
            if self.mode == "timing":
                return self.timed_call(
                    "fa_warp",
                    self.orig_warp,
                    x,
                    flow
                )

            if self.mode == "features":
                self.current[
                    "fa_flow"
                ] = flow.detach()

            return self.orig_warp(
                x,
                flow
            )

        self.fa.warp_feature = warp_wrapper

        #
        # NMS
        #
        def nms_wrapper(
            boxes,
            scores,
            thresh,
            *args,
            **kwargs
        ):
            input_count = int(boxes.shape[0])

            if self.mode == "timing":
                s = new_event()
                e = new_event()

                s.record()

                out = self.orig_nms(
                    boxes,
                    scores,
                    thresh,
                    *args,
                    **kwargs
                )

                e.record()

                self.records[
                    "post_nms"
                ].append((s, e))

                self.current[
                    "nms_inputs"
                ].append(input_count)

                try:
                    output_count = int(
                        out[0].shape[0]
                    )
                except Exception:
                    output_count = 0

                self.current[
                    "nms_outputs"
                ].append(output_count)

                return out

            return self.orig_nms(
                boxes,
                scores,
                thresh,
                *args,
                **kwargs
            )

        setattr(
            iou3d_nms_utils,
            self.nms_name,
            nms_wrapper
        )

        #
        # Full post-processing
        #
        def post_wrapper(batch_dict):
            if self.mode == "timing":
                s = new_event()
                e = new_event()

                s.record()

                out = self.orig_post(
                    batch_dict
                )

                e.record()

                self.records[
                    "post_processing"
                ].append((s, e))

                return out

            out = self.orig_post(
                batch_dict
            )

            if self.mode == "features":
                cls_preds = batch_dict[
                    "batch_cls_preds"
                ]

                self.current[
                    "post_cls_preds"
                ] = cls_preds

                self.current[
                    "post_cls_normalized"
                ] = bool(
                    batch_dict[
                        "cls_preds_normalized"
                    ]
                )

                self.current[
                    "post_box_preds"
                ] = batch_dict[
                    "batch_box_preds"
                ]

                self.current[
                    "post_result"
                ] = out[0]

            return out

        self.model.post_processing = post_wrapper

    def elapsed(self):
        out = {}

        for name, pairs in self.records.items():
            out[name] = float(
                sum(
                    s.elapsed_time(e)
                    for s, e in pairs
                )
            )

        return out

    def close(self):
        for h in self.handles:
            h.remove()

        feature_flow_utils.offset_matching_gpu = (
            self.orig_offset_matching
        )

        feature_flow_utils.get_shift_coord = (
            self.orig_get_shift_coord
        )

        feature_flow_utils.feature_flow_cuda.feature_sweeping_gpu = (
            self.orig_sweep
        )

        self.fa.warp_feature = self.orig_warp

        setattr(
            iou3d_nms_utils,
            self.nms_name,
            self.orig_nms
        )

        self.model.post_processing = self.orig_post


def warmup(model, dataset, n):
    if n <= 0:
        return

    print(f"[warmup] {n} frames")

    clear_history(model)

    with torch.no_grad():
        for i in range(n):
            sample = dataset[
                i % len(dataset)
            ]

            batch = dataset.collate_batch(
                [sample]
            )

            load_data_to_gpu(batch)

            model(batch)

    torch.cuda.synchronize()

    clear_history(model)


def get_frame_meta(dataset, idx):
    info = dataset.kitti_infos[idx]

    sample_idx = info["sample_idx"]

    return {
        "frame_idx": idx,
        "scene": str(
            sample_idx["scene"]
        ),
        "frame_id": str(
            sample_idx[
                "frame_tag"
            ]["token"]
        ),
    }


def safe_item(x):
    return float(x.item())


def extract_fa_features(profiler):
    c = profiler.current

    result = {}

    history_len = int(
        c.get("history_len", 0)
    )

    result["history_len"] = history_len
    result["history_valid"] = int(
        history_len > 0
    )

    if history_len == 0:
        return result

    flow = c.get(
        "fa_flow", None
    )

    matching = c.get(
        "fa_matching_dist", None
    )

    cur = c.get(
        "fa_cur_feature", None
    )

    last = c.get(
        "fa_last_feature", None
    )

    #
    # Flow statistics
    #
    if flow is not None:
        flow_f = flow.float()

        abs_flow = flow_f.abs()

        l2 = torch.sqrt(
            torch.sum(
                flow_f * flow_f,
                dim=1
            )
        )

        result[
            "fa_flow_abs_mean"
        ] = safe_item(
            abs_flow.mean()
        )

        result[
            "fa_flow_l2_mean"
        ] = safe_item(
            l2.mean()
        )

        result[
            "fa_flow_l2_p95"
        ] = safe_item(
            torch.quantile(
                l2.flatten(),
                0.95
            )
        )

        result[
            "fa_flow_max"
        ] = safe_item(
            abs_flow.max()
        )

        result[
            "fa_flow_nonzero_ratio"
        ] = safe_item(
            (
                l2 > 0.5
            ).float().mean()
        )

        max_shift = float(
            torch.abs(
                profiler.fa.shift_range_last
            ).max().item()
            * profiler.fa.factor
        )

        result[
            "fa_flow_boundary_ratio"
        ] = safe_item(
            (
                l2 >= max_shift * 0.9
            ).float().mean()
        )

    #
    # Matching confidence
    #
    if matching is not None:
        matching_f = matching.float()

        conf = torch.max(
            matching_f,
            dim=1
        ).values

        result[
            "fa_match_conf_mean"
        ] = safe_item(
            conf.mean()
        )

        result[
            "fa_match_conf_p10"
        ] = safe_item(
            torch.quantile(
                conf.flatten(),
                0.10
            )
        )

        result[
            "fa_match_conf_p90"
        ] = safe_item(
            torch.quantile(
                conf.flatten(),
                0.90
            )
        )

        entropy = -(
            matching_f
            * torch.log(
                matching_f + 1e-8
            )
        ).sum(dim=1)

        result[
            "fa_match_entropy_mean"
        ] = safe_item(
            entropy.mean()
        )

    #
    # Current/history feature difference
    #
    if (
        cur is not None
        and last is not None
    ):
        cur_f = cur.float()
        last_f = last.float()

        result[
            "fa_feature_diff_l1"
        ] = safe_item(
            (
                cur_f - last_f
            ).abs().mean()
        )

        cosine = F.cosine_similarity(
            cur_f,
            last_f,
            dim=1,
            eps=1e-6
        )

        result[
            "fa_feature_cosine"
        ] = safe_item(
            cosine.mean()
        )

    return result


def normalize_cls_preds(preds):
    if isinstance(preds, list):
        return preds
    return [preds]


def extract_post_features(
    profiler,
    model
):
    c = profiler.current

    result = {}

    if "post_cls_preds" not in c:
        return result

    preds_list = normalize_cls_preds(
        c["post_cls_preds"]
    )

    normalized = c[
        "post_cls_normalized"
    ]

    threshold = float(
        model.model_cfg
        .POST_PROCESSING
        .SCORE_THRESH
    )

    all_scores = []

    candidate_total = 0
    candidate_per_class = defaultdict(int)

    thresholds = [
        0.05,
        0.10,
        0.20,
        0.50
    ]

    threshold_counts = {
        t: 0
        for t in thresholds
    }

    class_offset = 0

    for preds in preds_list:
        x = preds

        if x.ndim == 3:
            x = x[0]

        if not normalized:
            x = torch.sigmoid(x)

        x = x.float()

        all_scores.append(
            x.flatten()
        )

        num_classes = x.shape[-1]

        for cls_id in range(
            num_classes
        ):
            scores = x[:, cls_id]

            count = int(
                (
                    scores >= threshold
                ).sum().item()
            )

            candidate_total += count

            candidate_per_class[
                class_offset + cls_id
            ] += count

        for t in thresholds:
            threshold_counts[t] += int(
                (
                    x >= t
                ).sum().item()
            )

        class_offset += num_classes

    scores = torch.cat(
        all_scores
    )

    result[
        "post_candidate_total"
    ] = candidate_total

    for cls_id, count in (
        candidate_per_class.items()
    ):
        result[
            f"post_candidate_class_{cls_id}"
        ] = count

    for t, count in threshold_counts.items():
        tag = str(t).replace(
            ".", "p"
        )

        result[
            f"post_count_ge_{tag}"
        ] = count

    result[
        "post_score_mean"
    ] = safe_item(
        scores.mean()
    )

    result[
        "post_score_max"
    ] = safe_item(
        scores.max()
    )

    result[
        "post_score_p99"
    ] = safe_item(
        torch.quantile(
            scores,
            0.99
        )
    )

    box_preds = c[
        "post_box_preds"
    ]

    if box_preds.ndim == 3:
        raw_box_count = int(
            box_preds.shape[1]
        )
    else:
        raw_box_count = int(
            box_preds.shape[0]
        )

    result[
        "post_raw_box_count"
    ] = raw_box_count

    final_count = 0

    for pred in c.get(
        "post_result", []
    ):
        if (
            isinstance(pred, dict)
            and "pred_boxes" in pred
        ):
            final_count += int(
                pred["pred_boxes"].shape[0]
            )

    result[
        "post_final_box_count"
    ] = final_count

    return result


def corr(
    rows,
    target,
    feature
):
    xs = []
    ys = []

    for r in rows:
        x = r.get(
            feature, np.nan
        )

        y = r.get(
            target, np.nan
        )

        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        if (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            xs.append(x)
            ys.append(y)

    if len(xs) < 4:
        return None

    x = np.asarray(xs)
    y = np.asarray(ys)

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return None

    p = pearsonr(x, y).statistic
    s = spearmanr(x, y).statistic

    return {
        "n": len(x),
        "pearson": float(p),
        "spearman": float(s),
    }


def print_corr_table(
    title,
    rows,
    target,
    features
):
    values = []

    for f in features:
        c = corr(
            rows,
            target,
            f
        )

        if c is not None:
            values.append(
                (
                    f,
                    c["pearson"],
                    c["spearman"],
                    c["n"]
                )
            )

    values.sort(
        key=lambda x:
        abs(x[2]),
        reverse=True
    )

    print()
    print("=" * 85)
    print(title)
    print("=" * 85)

    print(
        f"{'Feature':38s}"
        f"{'Pearson':>12s}"
        f"{'Spearman':>12s}"
        f"{'N':>8s}"
    )

    print("-" * 72)

    for name, p, s, n in values:
        print(
            f"{name:38s}"
            f"{p:12.4f}"
            f"{s:12.4f}"
            f"{n:8d}"
        )


def save_csv(rows, path):
    all_keys = set()

    for r in rows:
        all_keys.update(
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

    other = sorted(
        k
        for k in all_keys
        if k not in priority
    )

    keys = priority + other

    with open(
        path,
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        for r in rows:
            writer.writerow(r)


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

    #
    # Make sure original forward_test is used.
    #
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

    profiler = Profiler(
        model
    )

    warmup(
        model,
        dataset,
        args.warmup
    )

    N = min(
        args.num_frames,
        len(dataset)
    )

    rows = [
        get_frame_meta(
            dataset,
            i
        )
        for i in range(N)
    ]

    #
    # ==========================================================
    # PASS 1: timing only
    # ==========================================================
    #
    print()
    print(
        "PASS 1/2: latency measurement"
    )

    clear_history(model)

    with torch.no_grad():
        for i in range(N):
            sample = dataset[i]

            batch = dataset.collate_batch(
                [sample]
            )

            #
            # H2D is explicitly outside model timing.
            #
            load_data_to_gpu(
                batch
            )

            torch.cuda.synchronize()

            profiler.reset_frame()

            mem_before = (
                torch.cuda.memory_stats()
            )

            model_start = new_event()
            model_end = new_event()

            profiler.mode = "timing"

            model_start.record()

            model(batch)

            model_end.record()

            profiler.mode = "off"

            torch.cuda.synchronize()

            mem_after = (
                torch.cuda.memory_stats()
            )

            times = profiler.elapsed()

            model_ms = float(
                model_start.elapsed_time(
                    model_end
                )
            )

            fa_ms = times.get(
                "FeatureAlignment",
                float("nan")
            )

            post_ms = times.get(
                "post_processing",
                float("nan")
            )

            nms_ms = times.get(
                "post_nms",
                0.0
            )

            row = rows[i]

            row[
                "model_latency_ms"
            ] = model_ms

            row[
                "fa_latency_ms"
            ] = fa_ms

            row[
                "post_latency_ms"
            ] = post_ms

            row[
                "fa_pool_ms"
            ] = times.get(
                "fa_pool",
                0.0
            )

            row[
                "fa_fusion_layer_ms"
            ] = times.get(
                "fa_fusion_layer",
                0.0
            )

            row[
                "fa_offset_matching_ms"
            ] = times.get(
                "fa_offset_matching",
                0.0
            )

            row[
                "fa_feature_sweeping_ms"
            ] = times.get(
                "fa_feature_sweeping",
                0.0
            )

            row[
                "fa_shift_coord_ms"
            ] = times.get(
                "fa_shift_coord",
                0.0
            )

            row[
                "fa_upsample_ms"
            ] = times.get(
                "fa_upsample",
                0.0
            )

            row[
                "fa_warp_ms"
            ] = times.get(
                "fa_warp",
                0.0
            )

            row[
                "fa_matching_other_ms"
            ] = max(
                0.0,
                row[
                    "fa_offset_matching_ms"
                ]
                -
                row[
                    "fa_feature_sweeping_ms"
                ]
            )

            row[
                "post_nms_ms"
            ] = nms_ms

            row[
                "post_other_ms"
            ] = max(
                0.0,
                post_ms - nms_ms
            )

            nms_inputs = profiler.current.get(
                "nms_inputs",
                []
            )

            nms_outputs = profiler.current.get(
                "nms_outputs",
                []
            )

            row[
                "nms_input_total"
            ] = int(
                sum(nms_inputs)
            )

            row[
                "nms_input_max"
            ] = int(
                max(
                    nms_inputs,
                    default=0
                )
            )

            row[
                "nms_keep_total"
            ] = int(
                sum(nms_outputs)
            )

            row[
                "history_len"
            ] = int(
                profiler.current.get(
                    "history_len",
                    0
                )
            )

            row[
                "history_valid"
            ] = int(
                row["history_len"] > 0
            )

            row[
                "allocator_retry_delta"
            ] = int(
                mem_after.get(
                    "num_alloc_retries",
                    0
                )
                -
                mem_before.get(
                    "num_alloc_retries",
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
                    f"model={model_ms:7.3f} "
                    f"FA={fa_ms:7.3f} "
                    f"post={post_ms:6.3f} "
                    f"NMSin={row['nms_input_total']}"
                )

    #
    # ==========================================================
    # PASS 2: workload / content features only
    # ==========================================================
    #
    print()
    print(
        "PASS 2/2: workload feature collection"
    )

    clear_history(model)

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

            model(batch)

            profiler.mode = "off"

            torch.cuda.synchronize()

            fa_features = (
                extract_fa_features(
                    profiler
                )
            )

            post_features = (
                extract_post_features(
                    profiler,
                    model
                )
            )

            #
            # These reductions are intentionally outside
            # the measured pass.
            #
            torch.cuda.synchronize()

            rows[i].update(
                fa_features
            )

            rows[i].update(
                post_features
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
                    f"history={rows[i].get('history_len', 0)} "
                    f"flow={rows[i].get('fa_flow_l2_mean', float('nan')):.3f} "
                    f"cand={rows[i].get('post_candidate_total', 0)}"
                )

    profiler.close()

    #
    # Statistics
    #
    fa_substage_features = [
        "fa_pool_ms",
        "fa_fusion_layer_ms",
        "fa_offset_matching_ms",
        "fa_feature_sweeping_ms",
        "fa_matching_other_ms",
        "fa_shift_coord_ms",
        "fa_upsample_ms",
        "fa_warp_ms",
    ]

    fa_data_features = [
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
        "allocator_retry_delta",
    ]

    post_features = [
        "post_nms_ms",
        "post_other_ms",
        "nms_input_total",
        "nms_input_max",
        "nms_keep_total",
        "post_candidate_total",
        "post_candidate_class_0",
        "post_candidate_class_1",
        "post_candidate_class_2",
        "post_count_ge_0p05",
        "post_count_ge_0p1",
        "post_count_ge_0p2",
        "post_count_ge_0p5",
        "post_raw_box_count",
        "post_final_box_count",
        "post_score_mean",
        "post_score_max",
        "post_score_p99",
        "allocator_retry_delta",
    ]

    print_corr_table(
        "FeatureAlignment latency vs internal substage latency",
        rows,
        "fa_latency_ms",
        fa_substage_features
    )

    print_corr_table(
        "FeatureAlignment latency vs state / data features",
        rows,
        "fa_latency_ms",
        fa_data_features
    )

    print_corr_table(
        "Post-processing latency correlations",
        rows,
        "post_latency_ms",
        post_features
    )

    #
    # Basic percentile summary
    #
    def summary(name):
        vals = np.asarray([
            float(r[name])
            for r in rows
            if name in r
            and math.isfinite(
                float(r[name])
            )
        ])

        print(
            f"{name:24s}: "
            f"mean={vals.mean():7.3f} "
            f"p50={np.percentile(vals,50):7.3f} "
            f"p95={np.percentile(vals,95):7.3f} "
            f"p99={np.percentile(vals,99):7.3f}"
        )

    print()
    print("=" * 85)
    print("Latency summary")
    print("=" * 85)

    summary(
        "model_latency_ms"
    )

    summary(
        "fa_latency_ms"
    )

    summary(
        "post_latency_ms"
    )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        out_dir
        / f"profile_{N}_frames.csv"
    )

    json_path = (
        out_dir
        / f"profile_{N}_frames.json"
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

    print()
    print(
        "Saved CSV :",
        csv_path
    )

    print(
        "Saved JSON:",
        json_path
    )


if __name__ == "__main__":
    main()
