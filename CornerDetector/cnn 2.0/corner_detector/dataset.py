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
    max_shapes: int = 4
    heatmap_sigma: float = 2.0
    noise_std: float = 0.04
    blur_probability: float = 0.35
    filled_probability: float = 0.55
    cube_probability: float = 0.25
    checkerboard_probability: float = 0.15
    negative_probability: float = 0.18


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


def add_points_to_heatmap(heatmap: np.ndarray, points: np.ndarray, sigma: float) -> None:
    h, w = heatmap.shape
    for px, py in points:
        x = int(np.clip(round(float(px)), 0, w - 1))
        y = int(np.clip(round(float(py)), 0, h - 1))
        draw_gaussian(heatmap, x, y, sigma)


def draw_antialiased_polyline(
    image: np.ndarray,
    points: np.ndarray,
    intensity: float,
    thickness: int,
    closed: bool = True,
) -> None:
    cv2.polylines(
        image,
        [points.astype(np.int32)],
        isClosed=closed,
        color=float(intensity),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def make_background(size: int, rng: random.Random) -> np.ndarray:
    base = np.full((size, size), rng.uniform(0.0, 0.18), dtype=np.float32)
    if rng.random() < 0.55:
        x_grad = np.linspace(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), size, dtype=np.float32)
        y_grad = np.linspace(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), size, dtype=np.float32)[:, None]
        base = base + x_grad + y_grad
    if rng.random() < 0.35:
        stripe_period = rng.randint(6, 18)
        stripes = (np.arange(size, dtype=np.float32) % stripe_period) / stripe_period
        base = base + 0.04 * stripes[None, :]
    return np.clip(base, 0.0, 1.0)


def draw_filled_polygon(image: np.ndarray, points: np.ndarray, rng: random.Random) -> None:
    fill = rng.uniform(0.45, 0.95)
    cv2.fillPoly(image, [points.astype(np.int32)], color=float(fill), lineType=cv2.LINE_AA)
    draw_antialiased_polyline(
        image,
        points,
        intensity=float(np.clip(fill + rng.uniform(-0.25, 0.2), 0.15, 1.0)),
        thickness=rng.randint(1, 3),
        closed=True,
    )


def draw_checkerboard(image: np.ndarray, heatmap: np.ndarray, config: SyntheticConfig, rng: random.Random) -> None:
    size = config.image_size
    cells = rng.randint(6, 10)
    board_size = rng.randint(int(size * 0.58), int(size * 0.9))
    x0 = rng.randint(2, size - board_size - 2)
    y0 = rng.randint(2, size - board_size - 2)
    cell = board_size / cells
    light = rng.uniform(0.72, 1.0)
    dark = rng.uniform(0.0, 0.18)

    for row in range(cells):
        for col in range(cells):
            x1 = int(round(x0 + col * cell))
            y1 = int(round(y0 + row * cell))
            x2 = int(round(x0 + (col + 1) * cell))
            y2 = int(round(y0 + (row + 1) * cell))
            color = light if (row + col) % 2 == 0 else dark
            cv2.rectangle(image, (x1, y1), (x2, y2), color=float(color), thickness=-1)

    points = []
    for row in range(cells + 1):
        for col in range(cells + 1):
            points.append([x0 + col * cell, y0 + row * cell])
    add_points_to_heatmap(heatmap, np.array(points, dtype=np.float32), config.heatmap_sigma)


def draw_cube(image: np.ndarray, heatmap: np.ndarray, config: SyntheticConfig, rng: random.Random) -> None:
    size = config.image_size
    cx = rng.uniform(size * 0.38, size * 0.62)
    cy = rng.uniform(size * 0.3, size * 0.48)
    half_w = rng.uniform(size * 0.16, size * 0.28)
    half_d = rng.uniform(size * 0.1, size * 0.18)
    height = rng.uniform(size * 0.22, size * 0.38)

    top = np.array([cx, cy - half_d], dtype=np.float32)
    left = np.array([cx - half_w, cy], dtype=np.float32)
    right = np.array([cx + half_w, cy], dtype=np.float32)
    center = np.array([cx, cy + half_d], dtype=np.float32)
    left_bottom = left + np.array([0, height], dtype=np.float32)
    right_bottom = right + np.array([0, height], dtype=np.float32)
    bottom = center + np.array([0, height], dtype=np.float32)

    shade = rng.uniform(0.45, 0.9)
    faces = [
        (np.array([top, right, center, left], dtype=np.int32), shade + 0.12),
        (np.array([left, center, bottom, left_bottom], dtype=np.int32), shade - 0.08),
        (np.array([center, right, right_bottom, bottom], dtype=np.int32), shade),
    ]
    for face, color in faces:
        cv2.fillPoly(image, [face], color=float(np.clip(color, 0.1, 1.0)), lineType=cv2.LINE_AA)
    for edge in [
        np.array([top, left, left_bottom, bottom, right_bottom, right, top], dtype=np.int32),
        np.array([left, center, right], dtype=np.int32),
        np.array([center, bottom], dtype=np.int32),
    ]:
        draw_antialiased_polyline(image, edge, intensity=float(np.clip(shade - 0.2, 0.05, 0.85)), thickness=1, closed=False)

    points = np.array([top, left, right, center, left_bottom, right_bottom, bottom], dtype=np.float32)
    add_points_to_heatmap(heatmap, points, config.heatmap_sigma)


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
    image = make_background(size, rng)
    heatmap = np.zeros((size, size), dtype=np.float32)
    if rng.random() < config.negative_probability:
        noise = np.random.default_rng(self_seed(config, rng)).normal(0, config.noise_std, image.shape)
        image = np.clip(image + noise.astype(np.float32), 0.0, 1.0)
        return image.astype(np.float32), heatmap.astype(np.float32)

    if rng.random() < config.checkerboard_probability:
        draw_checkerboard(image, heatmap, config, rng)
    if rng.random() < config.cube_probability:
        draw_cube(image, heatmap, config, rng)

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
            if rng.random() < config.filled_probability:
                draw_filled_polygon(image, points, rng)
            else:
                draw_antialiased_polyline(image, points, intensity, thickness)

        elif shape_type == "polygon":
            sides = rng.randint(3, 7)
            radius = rng.uniform(size * 0.08, size * 0.2)
            cx = rng.uniform(radius + 5, size - radius - 5)
            cy = rng.uniform(radius + 5, size - radius - 5)
            angle = rng.uniform(0, 2 * math.pi)
            points = polygon_points(cx, cy, radius, sides, angle)
            if rng.random() < config.filled_probability:
                draw_filled_polygon(image, points, rng)
            else:
                draw_antialiased_polyline(image, points, intensity, thickness)

        else:
            count = rng.randint(3, 6)
            points = []
            margin = 8
            for _point in range(count):
                points.append([rng.randint(margin, size - margin), rng.randint(margin, size - margin)])
            points = np.array(points, dtype=np.int32)
            draw_antialiased_polyline(image, points, intensity, thickness, closed=False)

        add_points_to_heatmap(heatmap, points, config.heatmap_sigma)

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
