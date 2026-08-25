#!/usr/bin/env python3
import argparse
import copy
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    p = argparse.ArgumentParser("timestamp-driven real streaming evaluation")
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", default="outputs/realtime_stream")
    p.add_argument("--latency_scope",
                   choices=["model", "gpu_pipeline", "wall_e2e"],
                   default="gpu_pipeline",
                   help="model=forward only; gpu_pipeline=H2D+forward; wall_e2e=disk/preprocess+H2D+forward")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--timestamp_json", type=str, default=None,
                   help='optional mapping {"scene/frame_id": timestamp_ms}; otherwise use ANNOS_FREQUENCY')
    p.add_argument("--reset_history_on_gap", action="store_true",
                   help="ablation only; default keeps last processed feature across skipped frames")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--full_metrics", action="store_true",
                   help="print full KITTI metrics instead of paper-only summary")
    return p.parse_args()


def clear_history(model):
    if hasattr(model, "history_feature_queue") and model.history_feature_queue is not None:
        model.history_feature_queue.clear()


def empty_det():
    return {
        "name": np.array([], dtype=str),
        "truncated": np.array([], dtype=np.float32),
        "occluded": np.array([], dtype=np.float32),
        "alpha": np.array([], dtype=np.float32),
        "bbox": np.zeros((0, 4), dtype=np.float32),
        "dimensions": np.zeros((0, 3), dtype=np.float32),
        "location": np.zeros((0, 3), dtype=np.float32),
        "rotation_y": np.array([], dtype=np.float32),
        "score": np.array([], dtype=np.float32),
        "boxes_lidar": np.zeros((0, 7), dtype=np.float32),
    }


def format_paper_metrics(result_text):
    lines = result_text.splitlines()
    targets = {
        "Car": {
            "0.5": "Car AP_R40@0.70, 0.50, 0.50:",
            "0.7": "Car AP_R40@0.70, 0.70, 0.70:",
        },
        "Pedestrian": {
            "0.5": "Pedestrian AP_R40@0.50, 0.25, 0.25:",
            "0.7": "Pedestrian AP_R40@0.50, 0.50, 0.50:",
        },
        "Cyclist": {
            "0.5": "Cyclist AP_R40@0.50, 0.25, 0.25:",
            "0.7": "Cyclist AP_R40@0.50, 0.50, 0.50:",
        },
    }

    def extract(header):
        try:
            i = next(i for i, x in enumerate(lines) if x.strip() == header)
        except StopIteration:
            return None, None
        bev = d3 = None
        for x in lines[i + 1:]:
            y = x.strip()
            if y.endswith(":") and (" AP@" in y or " AP_R40@" in y):
                break
            if y.startswith("bev") and "AP:" in y:
                bev = y.split("AP:", 1)[1].strip()
            elif y.startswith("3d") and "AP:" in y:
                d3 = y.split("AP:", 1)[1].strip()
        return bev, d3

    out = ["Timestamp-stream metrics (AP_R40)",
           "Values: Easy / Moderate / Hard"]
    found = False
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        out += ["", cls]
        for iou in ["0.5", "0.7"]:
            bev, d3 = extract(targets[cls][iou])
            if bev is None and d3 is None:
                continue
            found = True
            out.append(f"  IoU={iou}")
            if bev is not None:
                out.append(f"    sAPBEV: {bev}")
            if d3 is not None:
                out.append(f"    sAP3D : {d3}")
    return "\n".join(out) if found else result_text


def load_timestamp_map(path):
    if path is None:
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def build_metadata(dataset, frame_period_ms, timestamp_map, max_samples=None):
    metas = []
    scene_ord = defaultdict(int)
    global_ord = 0

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(n):
        info = dataset.kitti_infos[idx]
        scene = str(info["sample_idx"]["scene"])
        frame_id = str(info["sample_idx"]["frame_tag"]["token"])
        key = f"{scene}/{frame_id}"

        ordinal = scene_ord[scene]
        scene_ord[scene] += 1

        if timestamp_map is not None:
            if key not in timestamp_map:
                raise KeyError(f"missing timestamp for {key}")
            scene_ts = float(timestamp_map[key])
        else:
            # KITTI Tracking is sampled at a fixed annotation frequency.
            # Prefer the numeric frame id so any frame-id gaps remain real time gaps.
            try:
                scene_ts = int(frame_id) * frame_period_ms
            except ValueError:
                scene_ts = ordinal * frame_period_ms

        metas.append({
            "dataset_index": idx,
            "scene": scene,
            "frame_id": frame_id,
            "scene_ordinal": ordinal,
            "scene_timestamp_ms": scene_ts,
            "global_timestamp_ms": global_ord * frame_period_ms,
        })
        global_ord += 1
    return metas


def warmup(model, dataset, n):
    if n <= 0:
        return
    print(f"[warmup] {n} iterations")
    for i in range(n):
        clear_history(model)
        sample = dataset[i % min(2, len(dataset))]
        batch = dataset.collate_batch([sample])
        load_data_to_gpu(batch)
        with torch.no_grad():
            model(batch)
    torch.cuda.synchronize()
    clear_history(model)


def percentile(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else float("nan")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.LOCAL_RANK = 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / "log_realtime_stream.txt", rank=0)

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
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    ann_freq = float(cfg.DATA_CONFIG.get("ANNOS_FREQUENCY", 10))
    frame_period_ms = 1000.0 / ann_freq
    ts_map = load_timestamp_map(args.timestamp_json)
    metas = build_metadata(dataset, frame_period_ms, ts_map, args.max_samples)
    meta_by_key = {(m["scene"], m["frame_id"]): m for m in metas}

    logger.info(f"frame frequency = {ann_freq:.3f} Hz")
    logger.info(f"frame period = {frame_period_ms:.3f} ms")
    logger.info(f"latency scope = {args.latency_scope}")
    logger.info("drop policy = drop frame when accelerator is busy")
    logger.info(f"reset history on gap = {args.reset_history_on_gap}")
    logger.info("timestamp source = " + ("external JSON" if ts_map is not None else "numeric frame_id / ANNOS_FREQUENCY"))

    warmup(model, dataset, args.warmup)

    output_events = defaultdict(list)
    trace = []
    latencies = []
    processed = 0
    skipped = 0
    gap_events = 0
    max_gap = 0

    current_scene = None
    busy_until_ms = -float("inf")
    last_processed_ordinal = None
    last_processed_frame_id = None

    for m in metas:
        scene = m["scene"]
        input_ts = m["scene_timestamp_ms"]

        if scene != current_scene:
            current_scene = scene
            busy_until_ms = -float("inf")
            last_processed_ordinal = None
            last_processed_frame_id = None
            clear_history(model)

        if input_ts + 1e-9 < busy_until_ms:
            skipped += 1
            trace.append({
                **m,
                "status": "skipped_busy",
                "busy_until_ms": float(busy_until_ms),
                "history_source_frame_id": last_processed_frame_id,
            })
            continue

        gap = 0 if last_processed_ordinal is None else max(
            0, m["scene_ordinal"] - last_processed_ordinal - 1
        )
        if gap > 0:
            gap_events += 1
            max_gap = max(max_gap, gap)
            if args.reset_history_on_gap:
                clear_history(model)

        if args.latency_scope == "wall_e2e":
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()

        sample = dataset[m["dataset_index"]]
        batch = dataset.collate_batch([sample])

        if args.latency_scope == "gpu_pipeline":
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()

        load_data_to_gpu(batch)

        if args.latency_scope == "model":
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()

        with torch.no_grad():
            pred_dicts, _ = model(batch)

        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1e6

        ann = dataset.generate_prediction_dicts(
            batch, pred_dicts, cfg.CLASS_NAMES
        )[0]

        output_ts = input_ts + latency_ms
        busy_until_ms = output_ts

        next_frame_id = ann.get("next_frame_id", None)
        if next_frame_id is not None:
            next_frame_id = str(next_frame_id)
        target_meta = meta_by_key.get((scene, next_frame_id))
        target_ts = None if target_meta is None else target_meta["scene_timestamp_ms"]

        event = {
            "scene": scene,
            "input_frame_id": m["frame_id"],
            "target_frame_id": next_frame_id,
            "input_timestamp_ms": float(input_ts),
            "output_timestamp_ms": float(output_ts),
            "target_timestamp_ms": None if target_ts is None else float(target_ts),
            "latency_ms": float(latency_ms),
            "history_source_frame_id": last_processed_frame_id,
            "history_gap_frames": int(gap),
            "annotation": ann,
        }
        output_events[scene].append(event)

        trace.append({
            **m,
            "status": "processed",
            "input_timestamp_ms": float(input_ts),
            "output_timestamp_ms": float(output_ts),
            "latency_ms": float(latency_ms),
            "target_frame_id": next_frame_id,
            "target_timestamp_ms": None if target_ts is None else float(target_ts),
            "history_source_frame_id": last_processed_frame_id,
            "history_gap_frames": int(gap),
        })

        latencies.append(latency_ms)
        processed += 1
        last_processed_ordinal = m["scene_ordinal"]
        last_processed_frame_id = m["frame_id"]

        if processed % args.log_every == 0:
            logger.info(
                f"processed={processed}, skipped={skipped}, "
                f"latest={scene}/{m['frame_id']}, latency={latency_ms:.2f} ms, gap={gap}"
            )

    gt_annos = []
    aligned_det_annos = []
    alignment_trace = []

    scene_outputs = {s: sorted(v, key=lambda x: x["output_timestamp_ms"])
                     for s, v in output_events.items()}
    ptr = defaultdict(int)
    latest = {}

    for m in metas:
        scene = m["scene"]
        q = m["scene_timestamp_ms"]
        events = scene_outputs.get(scene, [])
        p = ptr[scene]

        while p < len(events) and events[p]["output_timestamp_ms"] <= q + 1e-9:
            latest[scene] = events[p]
            p += 1
        ptr[scene] = p

        gt = copy.deepcopy(dataset.kitti_infos[m["dataset_index"]]["infos"]["token"]["annos"])
        gt_annos.append(gt)

        ev = latest.get(scene)
        if ev is None:
            aligned_det_annos.append(empty_det())
            alignment_trace.append({
                "scene": scene,
                "gt_frame_id": m["frame_id"],
                "gt_timestamp_ms": float(q),
                "selected_input_frame_id": None,
                "selected_target_frame_id": None,
                "selected_output_timestamp_ms": None,
                "target_staleness_ms": None,
            })
        else:
            aligned_det_annos.append(copy.deepcopy(ev["annotation"]))
            target_ts = ev["target_timestamp_ms"]
            alignment_trace.append({
                "scene": scene,
                "gt_frame_id": m["frame_id"],
                "gt_timestamp_ms": float(q),
                "selected_input_frame_id": ev["input_frame_id"],
                "selected_target_frame_id": ev["target_frame_id"],
                "selected_output_timestamp_ms": ev["output_timestamp_ms"],
                "target_staleness_ms": None if target_ts is None else float(q - target_ts),
            })

    result_str, result_dict = dataset.evaluation_offline(
        gt_annos, aligned_det_annos, cfg.CLASS_NAMES, "3d"
    )

    paper_str = format_paper_metrics(result_str)

    all_events = [e for scene in output_events.values() for e in scene]
    with open(out_dir / "stream_outputs.pkl", "wb") as f:
        pickle.dump(all_events, f)
    with open(out_dir / "timestamp_aligned_det_annos.pkl", "wb") as f:
        pickle.dump(aligned_det_annos, f)
    with open(out_dir / "metric_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    with open(out_dir / "trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    with open(out_dir / "alignment_trace.json", "w") as f:
        json.dump(alignment_trace, f, indent=2)
    (out_dir / "metrics_full.txt").write_text(result_str)
    (out_dir / "metrics_paper.txt").write_text(paper_str)

    total = processed + skipped
    miss = sum(x > frame_period_ms for x in latencies)
    summary = {
        "frames_total": total,
        "frames_processed": processed,
        "frames_skipped": skipped,
        "skip_rate": skipped / max(total, 1),
        "history_gap_events": gap_events,
        "max_history_gap_frames": max_gap,
        "frame_period_ms": frame_period_ms,
        "latency_scope": args.latency_scope,
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "latency_max_ms": float(np.max(latencies)) if latencies else None,
        "deadline_miss_rate": miss / max(processed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("REAL TIMESTAMP STREAM SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("TIMESTAMP-ALIGNED KITTI RESULT")
    print("=" * 80)
    print(result_str if args.full_metrics else paper_str)
    print("=" * 80)
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()