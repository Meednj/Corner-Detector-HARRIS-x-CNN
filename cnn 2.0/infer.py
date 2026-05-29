from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from corner_detector.model import load_model
from corner_detector.postprocess import find_corners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN corner detection on an image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="prediction.png")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--nms-kernel", type=int, default=15)
    parser.add_argument("--max-corners", type=int, default=200)
    parser.add_argument("--border-margin", type=int, default=10)
    parser.add_argument(
        "--preprocess",
        choices=("gray", "canny", "invert", "sobel"),
        default="canny",
        help="Input transform before the CNN. Canny is closer to the synthetic line training data.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_grayscale(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return rgb, gray


def preprocess_image(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return gray
    if mode == "invert":
        return 1.0 - gray
    if mode == "canny":
        gray_u8 = (gray * 255).astype(np.uint8)
        return cv2.Canny(gray_u8, 50, 150).astype(np.float32) / 255.0
    if mode == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        return np.clip(cv2.magnitude(grad_x, grad_y) * 4.0, 0.0, 1.0)
    raise ValueError(f"unknown preprocess mode: {mode}")


@torch.no_grad()
def predict_heatmap(model: torch.nn.Module, gray: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(gray[None, None, ...]).float().to(device)
    logits = model(tensor)
    heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return heatmap


def overlay_corners(rgb: np.ndarray, corners: list[tuple[int, int, float]]) -> np.ndarray:
    output = rgb.copy()
    for x, y, score in corners:
        radius = 3 if score < 0.7 else 4
        cv2.circle(output, (x, y), radius, (255, 40, 40), thickness=2)
        cv2.circle(output, (x, y), 1, (255, 255, 255), thickness=-1)
    return output


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    rgb, gray = load_grayscale(args.image)
    model_input = preprocess_image(gray, args.preprocess)
    heatmap = predict_heatmap(model, model_input, device)
    corners = find_corners(
        heatmap,
        threshold=args.threshold,
        nms_kernel=args.nms_kernel,
        max_corners=args.max_corners,
        border_margin=args.border_margin,
    )
    output = overlay_corners(rgb, corners)
    Image.fromarray(output).save(args.out)
    print(f"detected_corners={len(corners)} saved={args.out}")


if __name__ == "__main__":
    main()
