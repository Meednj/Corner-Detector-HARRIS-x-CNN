from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from corner_detector.model import load_model
from corner_detector.postprocess import find_corners

#parametre que vous m'avez demander de faire dans la derniere reunion pfe
IMAGE_PATH = Path("path/to/godmode.png")
THRESHOLD = -1.0 #-1 est la valeur automatique
NMS_KERNEL = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN corner detection on an image.")
    parser.add_argument("--checkpoint", default="runs/corner_detector_best.pt")
    parser.add_argument("--image", default=str(IMAGE_PATH), help="Image to process. Defaults to IMAGE_PATH.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Detection threshold (0-1). Use -1 for auto (recommended).")
    parser.add_argument("--nms-kernel", type=int, default=NMS_KERNEL)
    parser.add_argument("--max-corners", type=int, default=200)
    parser.add_argument("--border-margin", type=int, default=10)
    parser.add_argument(
        "--detector",
        choices=("harris", "cnn"),
        default="cnn",
    )
    parser.add_argument(
        "--preprocess",
        choices=("gray", "canny", "invert", "sobel"),
        default="sobel",
    )
    parser.add_argument("--quality-level", type=float, default=0.04)
    parser.add_argument("--min-distance", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--harris-k", type=float, default=0.04)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-heatmap", action="store_true",
                        help="Also save a coloured heatmap for debugging.")
    return parser.parse_args()


def load_grayscale(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return rgb, gray


def preprocess_image(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        result = gray.copy()
    elif mode == "invert":
        result = 1.0 - gray
    elif mode == "canny":
        gray_u8 = (gray * 255).astype(np.uint8)
        result = cv2.Canny(gray_u8, 50, 150).astype(np.float32) / 255.0
    elif mode == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        result = np.clip(cv2.magnitude(grad_x, grad_y) * 4.0, 0.0, 1.0)
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")

    # Normalize to [0, 1] per-image so real photos match synthetic training distribution
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-6:
        result = (result - lo) / (hi - lo)
    return result


def auto_threshold(heatmap: np.ndarray) -> float:
    """Derive threshold from the heatmap itself: mean of top-1% minus one std."""
    flat = heatmap.flatten()
    top = flat[flat >= np.percentile(flat, 99.0)]
    if len(top) == 0:
        return 0.6
    return float(np.clip(top.mean() - top.std(), 0.6, 0.95))


@torch.no_grad()
def predict_heatmap(model: torch.nn.Module, gray: np.ndarray, device: torch.device) -> np.ndarray:
    H, W = gray.shape

    model_size = getattr(model, "image_size", 128)
    resized = cv2.resize(gray, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized[None, None, ...]).float().to(device)
    logits = model(tensor)
    heatmap_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_LINEAR)


def overlay_corners(rgb: np.ndarray, corners: list[tuple[int, int, float]]) -> np.ndarray:
    output = rgb.copy()
    for x, y, score in corners:
        radius = 3 if score < 0.7 else 4
        cv2.circle(output, (x, y), radius, (255, 40, 40), thickness=2)
        cv2.circle(output, (x, y), 1, (255, 255, 255), thickness=-1)
    return output


def detect_harris_corners(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: int,
    block_size: int,
    harris_k: float,
    border_margin: int,
) -> list[tuple[int, int, float]]:
    gray_u8 = (gray * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        gray_u8,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=True,
        k=harris_k,
    )
    if corners is None:
        return []

    h, w = gray.shape
    detections = []
    for index, [[x, y]] in enumerate(corners):
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        if (
            x_i < border_margin
            or y_i < border_margin
            or x_i >= w - border_margin
            or y_i >= h - border_margin
        ):
            continue
        score = 1.0 - index / max(1, len(corners))
        detections.append((x_i, y_i, score))
    return detections


def build_output_path(image_path: str | Path) -> Path:
    image = Path(image_path)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{image.stem}_pred.png"


def main() -> None:
    args = parse_args()
    rgb, gray = load_grayscale(args.image)
    output_path = build_output_path(args.image)

    if args.detector == "harris":
        corners = detect_harris_corners(
            gray,
            max_corners=args.max_corners,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
            block_size=args.block_size,
            harris_k=args.harris_k,
            border_margin=args.border_margin,
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --detector cnn")

        device = torch.device(args.device)
        model = load_model(args.checkpoint, device=device)
        model_input = preprocess_image(gray, args.preprocess)
        heatmap = predict_heatmap(model, model_input, device)

        # Use auto-threshold when the user hasn't set one explicitly
        threshold = auto_threshold(heatmap) if args.threshold < 0 else args.threshold
        print(f"using threshold={threshold:.4f}")

        if args.save_heatmap:
            heatmap_vis = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)
            heatmap_path = output_path.with_name(f"{output_path.stem}_heatmap.png")
            cv2.imwrite(str(heatmap_path), heatmap_color)
            print(f"heatmap saved to {heatmap_path}")

        corners = find_corners(
            heatmap,
            threshold=threshold,
            nms_kernel=args.nms_kernel,
            max_corners=args.max_corners,
            border_margin=args.border_margin,
        )

    output = overlay_corners(rgb, corners)
    Image.fromarray(output).save(output_path)
    print(f"detected_corners={len(corners)} saved={output_path}")


if __name__ == "__main__":
    main()
