from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=False)
class SyntheticConfig:
    image_size: int = 128
    min_shapes: int = 1
    max_shapes: int = 3
    heatmap_sigma: float = 2.0
    noise_std: float = 0.05
    blur_probability: float = 0.4


def draw_gaussian(heatmap: np.ndarray, x: int, y: int, sigma: float) -> None:
    radius = max(1, int(3 * sigma))
    h, w = heatmap.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    patch = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)


def polygon_points(cx: float, cy: float, radius: float, sides: int, angle: float) -> np.ndarray:
    points = []
    for i in range(sides):
        theta = angle + 2 * math.pi * i / sides
        points.append([cx + radius * math.cos(theta), cy + radius * math.sin(theta)])
    return np.array(points, dtype=np.int32)


class SyntheticCornerDataset(Dataset):
    """Generates simple images with known geometric corners."""

    def __init__(self, length: int, config: SyntheticConfig | None = None, seed: int = 13) -> None:
        self.length = length
        self.config = config or SyntheticConfig()
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + index)
        image, heatmap = make_synthetic_sample(self.config, rng)
        image_t = torch.from_numpy(image[None, ...]).float()
        heatmap_t = torch.from_numpy(heatmap[None, ...]).float()
        return image_t, heatmap_t


def make_synthetic_sample(config: SyntheticConfig, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    size = config.image_size
    image = np.zeros((size, size), dtype=np.float32)
    heatmap = np.zeros((size, size), dtype=np.float32)
    shape_count = rng.randint(config.min_shapes, config.max_shapes)

    for _ in range(shape_count):
        shape_type = rng.choice(["rectangle", "polygon", "polyline"])
        intensity = rng.uniform(0.55, 1.0)
        thickness = rng.randint(1, 3)

        if shape_type == "rectangle":
            w = rng.randint(size // 8, size // 3)
            h = rng.randint(size // 8, size // 3)
            x = rng.randint(5, size - w - 6)
            y = rng.randint(5, size - h - 6)
            points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=float(intensity), thickness=thickness)

        elif shape_type == "polygon":
            sides = rng.randint(3, 7)
            radius = rng.uniform(size * 0.08, size * 0.2)
            cx = rng.uniform(radius + 5, size - radius - 5)
            cy = rng.uniform(radius + 5, size - radius - 5)
            angle = rng.uniform(0, 2 * math.pi)
            points = polygon_points(cx, cy, radius, sides, angle)
            cv2.polylines(image, [points], isClosed=True, color=float(intensity), thickness=thickness)

        else:
            count = rng.randint(3, 6)
            points = []
            margin = 8
            for _point in range(count):
                points.append([rng.randint(margin, size - margin), rng.randint(margin, size - margin)])
            points = np.array(points, dtype=np.int32)
            cv2.polylines(image, [points], isClosed=False, color=float(intensity), thickness=thickness)

        for px, py in points:
            draw_gaussian(heatmap, int(px), int(py), config.heatmap_sigma)

    if rng.random() < config.blur_probability:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    noise = np.random.default_rng(self_seed(config, rng)).normal(0, config.noise_std, image.shape)
    image = np.clip(image + noise.astype(np.float32), 0.0, 1.0)
    # Random contrast/brightness shift
    alpha = rng.uniform(0.7, 1.3)   # contrast
    beta = rng.uniform(-0.1, 0.1)   # brightness
    image = np.clip(alpha * image + beta, 0.0, 1.0)
    return image.astype(np.float32), heatmap.astype(np.float32)


def self_seed(config: SyntheticConfig, rng: random.Random) -> int:
    return rng.randint(0, 2**31 - 1) + config.image_size

