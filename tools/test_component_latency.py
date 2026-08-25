#!/usr/bin/env python3

import argparse
import json
import time
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
        description="Profile StreamDSGN component latency"
    )

    parser.add_argument(
        "-n",
        "--num_frames",
        type=int,
        required=True,
        help="number of measured inference frames"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="number of warmup frames, not counted in N"
    )

    parser.add_argument(
        "--cfg_file",
        type=str,
        default=DEFAULT_CFG
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_CKPT
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024
    )

    args = parser.parse_args()

    if args.num_frames <= 0:
        parser.error("--num_frames must be > 0")

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    return args


def clear_history(model):
    if (
        hasattr(model, "history_feature_queue")
        and model.history_feature_queue is not None
    ):
        model.history_feature_queue.clear()


class ComponentProfiler:
    """
    CUDA-event profiler for StreamDSGN top-level modules.

    No torch.cuda.synchronize() is inserted between individual modules,
    so the model execution is not artificially serialized.
    """

    def __init__(self):
        self.enabled = False
        self.handles = []
        self.reset_frame()

    def reset_frame(self):
        self.records = defaultdict(list)
        self.active = defaultdict(list)

    @staticmethod
    def new_event():
        return torch.cuda.Event(enable_timing=True)

    def register_module(self, name, module):

        def pre_hook(_module, _inputs):
            if not self.enabled:
                return

            start = self.new_event()
            start.record()
            self.active[name].append(start)

        def post_hook(_module, _inputs, _output):
            if not self.enabled:
                return

            start = self.active[name].pop()

            end = self.new_event()
            end.record()

            self.records[name].append((start, end))

        self.handles.append(
            module.register_forward_pre_hook(pre_hook)
        )

        self.handles.append(
            module.register_forward_hook(post_hook)
        )

    def measure_function(self, name, func, *args, **kwargs):
        if not self.enabled:
            return func(*args, **kwargs)

        start = self.new_event()
        end = self.new_event()

        start.record()
        output = func(*args, **kwargs)
        end.record()

        self.records[name].append((start, end))

        return output

    def elapsed_ms(self):
        result = {}

        for name, pairs in self.records.items():
            total = 0.0

            for start, end in pairs:
                total += start.elapsed_time(end)

            result[name] = float(total)

        return result

    def close(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def attach_profiler(model, profiler):
    """
    Register top-level components in the same order as STREAM.forward_test().
    """

    stage_names = []
    counter = defaultdict(int)

    module_groups = [
        model.feature_extractor,
        model.fusion_module,
        model.after_fusion_blocks,
    ]

    for group in module_groups:
        for module in group:

            base_name = type(module).__name__

            counter[base_name] += 1

            if counter[base_name] == 1:
                name = base_name
            else:
                name = f"{base_name}#{counter[base_name]}"

            profiler.register_module(name, module)
            stage_names.append(name)

    #
    # post_processing is a function, not nn.Module.
    #
    original_post_processing = model.post_processing

    def timed_post_processing(batch_dict):
        return profiler.measure_function(
            "post_processing",
            original_post_processing,
            batch_dict
        )

    object.__setattr__(
        model,
        "post_processing",
        timed_post_processing
    )

    stage_names.append("post_processing")

    return stage_names


def warmup(model, dataset, num_warmup):
    if num_warmup <= 0:
        return

    print(f"[warmup] {num_warmup} frames")

    clear_history(model)

    with torch.no_grad():
        for i in range(num_warmup):

            idx = i % len(dataset)

            sample = dataset[idx]
            batch = dataset.collate_batch([sample])

            load_data_to_gpu(batch)
            model(batch)

    torch.cuda.synchronize()

    #
    # Measured sequence starts from a clean streaming history.
    #
    clear_history(model)


def stats(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "count": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def main():

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cfg_from_yaml_file(args.cfg_file, cfg)

    cfg.LOCAL_RANK = 0

    #
    # Make sure normal forward_test() is used.
    #
    cfg.MODEL.SAVE_TIME = False

    logger = common_utils.create_logger(rank=0)

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

    model.cuda()
    model.eval()

    profiler = ComponentProfiler()

    stage_names = attach_profiler(
        model,
        profiler
    )

    warmup(
        model,
        dataset,
        args.warmup
    )

    num_frames = min(
        args.num_frames,
        len(dataset)
    )

    if num_frames != args.num_frames:
        print(
            f"[warning] requested {args.num_frames} frames, "
            f"dataset only has {len(dataset)} frames"
        )

    component_times = defaultdict(list)
    aggregate_times = defaultdict(list)

    clear_history(model)

    print()
    print("========================================")
    print("StreamDSGN component latency profiling")
    print("========================================")
    print("GPU          :", torch.cuda.get_device_name(0))
    print("Torch        :", torch.__version__)
    print("Torch CUDA   :", torch.version.cuda)
    print("Frames       :", num_frames)
    print("Warmup       :", args.warmup)
    print("========================================")
    print()

    with torch.no_grad():

        for frame_idx in range(num_frames):

            #
            # Full wall-E2E starts before dataset loading.
            #
            wall_start = time.perf_counter()

            #
            # Disk + image decoding + preprocessing + collate
            #
            cpu_start = time.perf_counter()

            sample = dataset[frame_idx]
            batch = dataset.collate_batch([sample])

            cpu_prepare_ms = (
                time.perf_counter() - cpu_start
            ) * 1000.0

            profiler.reset_frame()

            #
            # GPU timing events
            #
            h2d_start = torch.cuda.Event(
                enable_timing=True
            )

            h2d_end = torch.cuda.Event(
                enable_timing=True
            )

            model_start = torch.cuda.Event(
                enable_timing=True
            )

            model_end = torch.cuda.Event(
                enable_timing=True
            )

            #
            # H2D
            #
            h2d_start.record()

            load_data_to_gpu(batch)

            h2d_end.record()

            #
            # Model forward
            #
            profiler.enabled = True

            model_start.record()

            model(batch)

            model_end.record()

            profiler.enabled = False

            #
            # Single synchronization per measured frame.
            #
            torch.cuda.synchronize()

            wall_e2e_ms = (
                time.perf_counter() - wall_start
            ) * 1000.0

            h2d_ms = float(
                h2d_start.elapsed_time(h2d_end)
            )

            model_ms = float(
                model_start.elapsed_time(model_end)
            )

            gpu_pipeline_ms = float(
                h2d_start.elapsed_time(model_end)
            )

            frame_components = profiler.elapsed_ms()

            component_sum_ms = sum(
                frame_components.values()
            )

            #
            # history clone / queue management / Python logic etc.
            #
            other_model_overhead = max(
                0.0,
                model_ms - component_sum_ms
            )

            for name, value in frame_components.items():
                component_times[name].append(
                    float(value)
                )

            component_times[
                "other_model_overhead"
            ].append(
                float(other_model_overhead)
            )

            aggregate_times[
                "cpu_data_prepare"
            ].append(cpu_prepare_ms)

            aggregate_times[
                "h2d"
            ].append(h2d_ms)

            aggregate_times[
                "model_total"
            ].append(model_ms)

            aggregate_times[
                "gpu_pipeline"
            ].append(gpu_pipeline_ms)

            aggregate_times[
                "wall_e2e"
            ].append(wall_e2e_ms)

            if (
                args.log_every > 0
                and (
                    (frame_idx + 1) % args.log_every == 0
                    or frame_idx + 1 == num_frames
                )
            ):
                print(
                    f"[{frame_idx + 1:4d}/{num_frames}] "
                    f"model={model_ms:7.3f} ms  "
                    f"pipeline={gpu_pipeline_ms:7.3f} ms"
                )

    profiler.close()

    component_stats = {}

    for name in stage_names:
        if name in component_times:
            component_stats[name] = stats(
                component_times[name]
            )

    component_stats[
        "other_model_overhead"
    ] = stats(
        component_times["other_model_overhead"]
    )

    aggregate_stats = {
        name: stats(values)
        for name, values in aggregate_times.items()
    }

    model_mean = aggregate_stats[
        "model_total"
    ]["mean_ms"]

    print()
    print("============================================================")
    print("Component latency")
    print("============================================================")

    print(
        f"{'Component':32s}"
        f"{'Mean(ms)':>12s}"
        f"{'P50(ms)':>12s}"
        f"{'P95(ms)':>12s}"
        f"{'% Model':>10s}"
    )

    print("-" * 78)

    output_order = (
        stage_names
        + ["other_model_overhead"]
    )

    for name in output_order:

        if name not in component_stats:
            continue

        s = component_stats[name]

        ratio = (
            100.0 * s["mean_ms"] / model_mean
            if model_mean > 0
            else float("nan")
        )

        print(
            f"{name:32s}"
            f"{s['mean_ms']:12.3f}"
            f"{s['p50_ms']:12.3f}"
            f"{s['p95_ms']:12.3f}"
            f"{ratio:9.2f}%"
        )

    print()
    print("============================================================")
    print("Aggregate latency")
    print("============================================================")

    print(
        f"{'Scope':32s}"
        f"{'Mean(ms)':>12s}"
        f"{'P50(ms)':>12s}"
        f"{'P95(ms)':>12s}"
    )

    print("-" * 68)

    aggregate_order = [
        "cpu_data_prepare",
        "h2d",
        "model_total",
        "gpu_pipeline",
        "wall_e2e",
    ]

    for name in aggregate_order:

        s = aggregate_stats[name]

        print(
            f"{name:32s}"
            f"{s['mean_ms']:12.3f}"
            f"{s['p50_ms']:12.3f}"
            f"{s['p95_ms']:12.3f}"
        )

    #
    # Save results
    #
    if args.output_json is None:

        output_path = Path(
            "outputs/component_latency/"
            f"component_latency_{num_frames}.json"
        )

    else:
        output_path = Path(
            args.output_json
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {
        "num_frames_requested": args.num_frames,
        "num_frames_measured": num_frames,
        "warmup_frames": args.warmup,
        "cfg_file": args.cfg_file,
        "ckpt": args.ckpt,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "components": component_stats,
        "aggregate": aggregate_stats,
    }

    with open(output_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print("Saved JSON:", output_path)


if __name__ == "__main__":
    main()
