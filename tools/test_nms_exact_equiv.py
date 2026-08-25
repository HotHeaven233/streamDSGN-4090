#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils


def make_case(n, thresh, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    boxes = torch.empty((n, 7), device="cuda", dtype=torch.float32)

    boxes[:, 0] = torch.rand(n, generator=g, device="cuda") * 50.0
    boxes[:, 1] = torch.rand(n, generator=g, device="cuda") * 50.0
    boxes[:, 2] = torch.rand(n, generator=g, device="cuda") * 4.0 - 2.0

    boxes[:, 3] = torch.rand(n, generator=g, device="cuda") * 4.0 + 0.2
    boxes[:, 4] = torch.rand(n, generator=g, device="cuda") * 2.0 + 0.2
    boxes[:, 5] = torch.rand(n, generator=g, device="cuda") * 2.0 + 0.2

    boxes[:, 6] = (
        torch.rand(n, generator=g, device="cuda") * 6.283185307179586
        - 3.141592653589793
    )

    scores = torch.rand(n, generator=g, device="cuda")

    selected, _ = iou3d_nms_utils.nms_gpu(
        boxes,
        scores,
        thresh,
        pre_maxsize=4096
    )

    torch.cuda.synchronize()

    return {
        "n": n,
        "thresh": thresh,
        "boxes": boxes.cpu(),
        "scores": scores.cpu(),
        "selected": selected.cpu(),
    }


def run_from_saved(case):
    boxes = case["boxes"].cuda()
    scores = case["scores"].cuda()

    selected, _ = iou3d_nms_utils.nms_gpu(
        boxes,
        scores,
        float(case["thresh"]),
        pre_maxsize=4096
    )

    torch.cuda.synchronize()

    return selected.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "compare"])
    parser.add_argument(
        "--file",
        default="outputs/nms_exact_baseline.pt"
    )
    args = parser.parse_args()

    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "save":
        specs = [
            (1,    0.25),
            (17,   0.25),
            (64,   0.25),
            (257,  0.25),
            (1024, 0.25),
            (4096, 0.25),
            (1024, 0.10),
            (1024, 0.50),
        ]

        cases = []

        for i, (n, thresh) in enumerate(specs):
            print(f"[save] n={n:4d}, thresh={thresh}")
            cases.append(
                make_case(
                    n=n,
                    thresh=thresh,
                    seed=1024 + i
                )
            )

        torch.save(cases, path)
        print("Saved:", path)
        return

    cases = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    all_ok = True

    for i, case in enumerate(cases):
        actual = run_from_saved(case)
        expected = case["selected"]

        equal = torch.equal(actual, expected)

        print(
            f"[{i}] n={case['n']:4d} "
            f"thresh={case['thresh']:.2f} "
            f"keep={len(expected):4d} "
            f"exact={equal}"
        )

        if not equal:
            all_ok = False

            print("expected:", expected[:30])
            print("actual  :", actual[:30])

    if not all_ok:
        raise RuntimeError("NMS output changed!")

    print()
    print("PASS: all optimized NMS outputs are exactly identical.")


if __name__ == "__main__":
    main()
