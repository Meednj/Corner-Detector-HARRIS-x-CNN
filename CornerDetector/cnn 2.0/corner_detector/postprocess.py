from __future__ import annotations

import cv2
import numpy as np


def find_corners(
    heatmap: np.ndarray,
    threshold: float = 0.35,
    nms_kernel: int = 7,
    max_corners: int = 200,
    border_margin: int = 0,
) -> list[tuple[int, int, float]]:
    """Return corner detections as (x, y, score)."""
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    if nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be odd")
    if border_margin < 0:
        raise ValueError("border_margin must be non-negative")

    heatmap = heatmap.astype(np.float32).copy()
    if border_margin:
        heatmap[:border_margin, :] = 0
        heatmap[-border_margin:, :] = 0
        heatmap[:, :border_margin] = 0
        heatmap[:, -border_margin:] = 0

    pooled = cv2.dilate(heatmap, np.ones((nms_kernel, nms_kernel), dtype=np.uint8))
    peaks = (heatmap == pooled) & (heatmap >= threshold)
    ys, xs = np.where(peaks)
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)[:max_corners]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]
