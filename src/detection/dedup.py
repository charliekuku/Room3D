"""Lightweight overlap helpers shared by 2-D and 3-D detection stages."""
from __future__ import annotations

import numpy as np


def box_iou(a, b) -> float:
    """Intersection-over-union for two XYXY boxes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    wh = np.maximum(hi - lo, 0.0)
    intersection = float(wh[0] * wh[1])
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def suppress_overlapping_boxes(boxes, labels, scores,
                               iou_threshold: float = 0.70) -> list[int]:
    """Score-ordered NMS within each snapped label.

    Grounding DINO performs phrase grounding, so one physical region can be
    returned repeatedly. Cross-label candidates are intentionally retained here
    so multi-frame 3-D evidence can decide whether ``desk`` or ``table`` is the
    better label; this pass removes only redundant copies of the same phrase.
    """
    boxes = np.asarray(boxes, float)
    scores = np.asarray(scores, float)
    labels = list(labels)
    if len(boxes) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    for idx in order:
        if all(
            labels[idx] != labels[other] or
            box_iou(boxes[idx], boxes[other]) < iou_threshold
            for other in kept
        ):
            kept.append(int(idx))
    return sorted(kept)
