#!/usr/bin/env python3

import torch

from pcdet.models.fusion_module.feature_alignment import (
    FeatureAlignment,
    warp_feature,
)

from easydict import EasyDict as edict


def main():
    cfg = edict({
        'FUSION_TYPE': 'avg',
        'GN': False,
        'HISTORY_TAG': ['prev'],
        'HISTORY_FEATURES_NAME': ['spatial_features'],
        'MATCHING_RANGE': [-3, 3, -3, 3],
        'POOL_CFG': {
            'type': 'max',
            'factor': 2,
        },
        'INTERP_TYPE': 'bilinear',
        'DO_WARPING': True,
        'PIXEL_LEVEL_WARPING': False,
    })

    model = FeatureAlignment(
        model_cfg=cfg,
        input_channels=96
    ).cuda().eval()

    torch.manual_seed(1024)

    # Use the actual BEV feature resolution appearing in this project.
    x = torch.randn(
        1, 96, 288, 256,
        device='cuda',
        dtype=torch.float16
    )

    flow = torch.randn(
        1, 2, 288, 256,
        device='cuda',
        dtype=torch.float16
    )

    with torch.no_grad():
        y_old = warp_feature(
            x.clone(),
            flow.clone()
        )

        y_new = model.warp_feature_cached(
            x.clone(),
            flow.clone()
        )

    torch.cuda.synchronize()

    diff = (
        y_old.float()
        - y_new.float()
    ).abs()

    print("shape          :", tuple(y_old.shape))
    print("dtype old      :", y_old.dtype)
    print("dtype new      :", y_new.dtype)
    print("torch.equal    :", torch.equal(y_old, y_new))
    print("max_abs_diff   :", diff.max().item())
    print("mean_abs_diff  :", diff.mean().item())

    assert torch.equal(
        y_old,
        y_new
    ), "ERROR: cached warp is not bitwise identical"

    print()
    print("PASS: cached warp is bitwise identical to original warp.")


if __name__ == "__main__":
    main()
