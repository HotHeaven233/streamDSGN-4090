import numpy as np

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """
    CPU-only compatibility implementation for KITTI evaluation.

    Parameters
    ----------
    boxes : ndarray, shape [N, 5]
        [cx, cy, dx, dy, angle]
    query_boxes : ndarray, shape [K, 5]
        [cx, cy, dx, dy, angle]
    criterion :
        -1 -> IoU
         0 -> intersection / area(boxes)
         1 -> intersection / area(query_boxes)
        else -> intersection area

    Notes
    -----
    The original KITTI evaluator's Numba implementation uses the opposite
    angle sign convention from OpenPCDet's C++ rotated-box implementation,
    so heading is negated during conversion.
    """

    box_dtype = boxes.dtype

    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)

    N = boxes.shape[0]
    K = query_boxes.shape[0]

    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=box_dtype)

    # OpenPCDet CPU rotated IoU expects:
    # [x, y, z, dx, dy, dz, heading]
    boxes7 = np.zeros((N, 7), dtype=np.float32)
    query7 = np.zeros((K, 7), dtype=np.float32)

    boxes7[:, 0] = boxes[:, 0]
    boxes7[:, 1] = boxes[:, 1]
    boxes7[:, 3] = boxes[:, 2]
    boxes7[:, 4] = boxes[:, 3]
    boxes7[:, 5] = 1.0
    boxes7[:, 6] = -boxes[:, 4]

    query7[:, 0] = query_boxes[:, 0]
    query7[:, 1] = query_boxes[:, 1]
    query7[:, 3] = query_boxes[:, 2]
    query7[:, 4] = query_boxes[:, 3]
    query7[:, 5] = 1.0
    query7[:, 6] = -query_boxes[:, 4]

    iou = boxes_bev_iou_cpu(
        boxes7,
        query7
    ).astype(np.float32, copy=False)

    if criterion == -1:
        result = iou

    else:
        area_a = (
            boxes[:, 2] * boxes[:, 3]
        ).reshape(-1, 1)

        area_b = (
            query_boxes[:, 2] * query_boxes[:, 3]
        ).reshape(1, -1)

        # IoU = I / (A + B - I)
        # => I = IoU * (A + B) / (1 + IoU)
        inter = (
            iou * (area_a + area_b)
            / np.maximum(1.0 + iou, 1e-8)
        )

        if criterion == 0:
            result = inter / np.maximum(area_a, 1e-8)

        elif criterion == 1:
            result = inter / np.maximum(area_b, 1e-8)

        else:
            # KITTI d3_box_overlap uses criterion=2 for raw intersection.
            result = inter

    return result.astype(box_dtype, copy=False)
