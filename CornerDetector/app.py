from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
CNN2_DIR = ROOT_DIR / "cnn 2.0"
HARRIS_DIR = ROOT_DIR / "HarrisDetector"
RESULTS_DIR = ROOT_DIR / "combined_results"
CHECKPOINT_PATH = CNN2_DIR / "runs" / "corner_detector_best.pt"

if str(CNN2_DIR) not in sys.path:
    sys.path.insert(0, str(CNN2_DIR))

from corner_detector.model import load_model
from corner_detector.postprocess import find_corners


app = Flask(__name__)
RESULTS_DIR.mkdir(exist_ok=True)

device: torch.device | None = None
model: torch.nn.Module | None = None


def initialize() -> None:
    global device, model
    if model is not None:
        return
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device=device)


def encode_png(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image as PNG")
    return base64.b64encode(buffer).decode("ascii")


def colorize_response(response: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    response = response.astype(np.float32)
    finite = np.isfinite(response)
    if not finite.any():
        normalized = np.zeros(response.shape, dtype=np.uint8)
    else:
        clean = np.where(finite, response, 0.0)
        lo = float(clean.min())
        hi = float(clean.max())
        if hi - lo < 1e-8:
            normalized = np.zeros(clean.shape, dtype=np.uint8)
        else:
            normalized = ((clean - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, colormap)


def load_upload_image(file_storage) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(file_storage.stream).convert("RGB"))
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

    lo = float(result.min())
    hi = float(result.max())
    if hi - lo > 1e-6:
        result = (result - lo) / (hi - lo)
    return result.astype(np.float32)


def auto_threshold(heatmap: np.ndarray) -> float:
    top = heatmap[heatmap >= np.percentile(heatmap, 99.0)]
    if top.size == 0:
        return 0.6
    return float(np.clip(top.mean() - top.std(), 0.6, 0.95))


def find_matlab_executable() -> str:
    candidates = [
        os.environ.get("MATLAB_EXE"),
        shutil.which("matlab"),
        shutil.which("matlab.exe"),
        r"C:\Program Files\MATLAB\MATLAB Production Server\R2015a\bin\matlab.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "MATLAB executable not found. Set MATLAB_EXE or install MATLAB/Matlab Production Server."
    )


def matlab_quote(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def run_harris_detector_with_matlab(
    gray: np.ndarray,
    k: float,
    sigma: float,
    gaussian_size: int,
    threshold_ratio: float,
    nms_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    matlab_exe = find_matlab_executable()

    with tempfile.TemporaryDirectory(prefix="harris_matlab_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.png"
        response_path = temp_dir / "response.txt"
        mask_path = temp_dir / "mask.txt"
        script_path = temp_dir / "run_harris.m"

        cv2.imwrite(str(input_path), np.clip(gray * 255.0, 0, 255).astype(np.uint8))

        harris_dir = matlab_quote(HARRIS_DIR)
        input_file = matlab_quote(input_path)
        response_file = matlab_quote(response_path)
        mask_file = matlab_quote(mask_path)

        script_path.write_text(
            textwrap.dedent(
                f"""
                addpath(genpath('{harris_dir}'));
                I = imread('{input_file}');
                if size(I, 3) == 3
                    I = rgb2gray(I);
                end
                I = double(I);
                [R, coins] = harris_detector(I, {k}, {sigma}, {gaussian_size}, {threshold_ratio}, {nms_kernel});
                save('{response_file}', 'R', '-ascii');
                save('{mask_file}', 'coins', '-ascii');
                exit;
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                matlab_exe,
                "-nosplash",
                "-nodesktop",
                "-noFigureWindows",
                "-wait",
                "-r",
                f"run('{matlab_quote(script_path)}');",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "MATLAB Harris detector failed: "
                + (result.stderr.strip() or result.stdout.strip() or "unknown error")
            )

        response = np.loadtxt(response_path, dtype=np.float32)
        mask = np.loadtxt(mask_path, dtype=np.float32)

        response = np.atleast_2d(response)
        mask = np.atleast_2d(mask)
        if response.shape != gray.shape and response.size == gray.size:
            response = response.reshape(gray.shape)
        if mask.shape != gray.shape and mask.size == gray.size:
            mask = mask.reshape(gray.shape)

        return response.astype(np.float32), mask.astype(np.float32)


def extract_harris_corners(
    response: np.ndarray,
    mask: np.ndarray,
    max_corners: int,
    border_margin: int,
) -> list[tuple[int, int, float]]:
    work_mask = mask.astype(bool).copy()
    if border_margin:
        work_mask[:border_margin, :] = False
        work_mask[-border_margin:, :] = False
        work_mask[:, :border_margin] = False
        work_mask[:, -border_margin:] = False

    ys, xs = np.where(work_mask)
    if xs.size == 0:
        return []

    scores = response[ys, xs].astype(np.float32)
    max_score = float(np.max(response)) if response.size else 0.0
    if max_score > 0:
        scores = scores / max_score

    order = np.argsort(-scores)[:max_corners]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


@torch.no_grad()
def predict_heatmap(model: torch.nn.Module, gray: np.ndarray, device: torch.device) -> np.ndarray:
    height, width = gray.shape
    model_size = getattr(model, "image_size", 128)
    resized = cv2.resize(gray, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized[None, None, ...]).float().to(device)
    logits = model(tensor)
    heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_corners(
    rgb: np.ndarray,
    corners: list[tuple[int, int, float]],
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = rgb.copy()
    for x, y, score in corners:
        radius = 3 if score < 0.7 else 4
        cv2.circle(overlay, (x, y), radius, color, thickness=2)
        cv2.circle(overlay, (x, y), 1, (255, 255, 255), thickness=-1)
    return overlay


def harris_response(
    gray: np.ndarray,
    k: float,
    sigma: float,
    gaussian_size: int,
) -> np.ndarray:
    gray32 = gray.astype(np.float32)
    ix = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)

    ix2 = cv2.GaussianBlur(ix * ix, (gaussian_size, gaussian_size), sigmaX=sigma)
    iy2 = cv2.GaussianBlur(iy * iy, (gaussian_size, gaussian_size), sigmaX=sigma)
    ixy = cv2.GaussianBlur(ix * iy, (gaussian_size, gaussian_size), sigmaX=sigma)

    return (ix2 * iy2 - ixy * ixy) - k * np.square(ix2 + iy2)


def detect_harris_corners(
    response: np.ndarray,
    threshold_ratio: float,
    nms_kernel: int,
    max_corners: int,
    border_margin: int,
) -> list[tuple[int, int, float]]:
    if nms_kernel % 2 == 0:
        nms_kernel += 1

    work = response.astype(np.float32).copy()
    work[work < 0] = 0
    if border_margin:
        work[:border_margin, :] = 0
        work[-border_margin:, :] = 0
        work[:, :border_margin] = 0
        work[:, -border_margin:] = 0

    max_response = float(work.max()) if work.size else 0.0
    if max_response <= 0:
        return []

    threshold = threshold_ratio * max_response
    pooled = cv2.dilate(work, np.ones((nms_kernel, nms_kernel), dtype=np.uint8))
    peaks = (work == pooled) & (work >= threshold)
    ys, xs = np.where(peaks)
    scores = work[ys, xs] / max_response
    order = np.argsort(-scores)[:max_corners]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


def save_session(payload: dict[str, object], images: dict[str, np.ndarray]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, image in images.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)

    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return str(out_dir.relative_to(ROOT_DIR))


@app.route("/")
def index():
    return send_from_directory(ROOT_DIR, "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        initialize()
        assert model is not None
        assert device is not None

        cnn_threshold = float(request.form.get("cnn_threshold", -1.0))
        cnn_nms_kernel = int(request.form.get("cnn_nms_kernel", 21))
        max_corners = int(request.form.get("max_corners", 200))
        border_margin = int(request.form.get("border_margin", 10))
        preprocess_mode = request.form.get("preprocess", "gray")

        harris_threshold = float(request.form.get("harris_threshold", 0.01))
        harris_nms_kernel = int(request.form.get("harris_nms_kernel", 7))
        harris_k = float(request.form.get("harris_k", 0.04))
        harris_sigma = float(request.form.get("harris_sigma", 1.5))
        harris_gaussian_size = int(request.form.get("harris_gaussian_size", 7))
        if harris_gaussian_size % 2 == 0:
            harris_gaussian_size += 1

        save_results = request.form.get("save_results", "true").lower() == "true"

        rgb, gray = load_upload_image(request.files["image"])
        cnn_input = preprocess_image(gray, preprocess_mode)
        cnn_heatmap = predict_heatmap(model, cnn_input, device)
        resolved_threshold = auto_threshold(cnn_heatmap) if cnn_threshold < 0 else cnn_threshold
        cnn_corners = find_corners(
            cnn_heatmap,
            threshold=resolved_threshold,
            nms_kernel=cnn_nms_kernel,
            max_corners=max_corners,
            border_margin=border_margin,
        )

        harris_map, harris_mask = run_harris_detector_with_matlab(
            gray,
            k=harris_k,
            sigma=harris_sigma,
            gaussian_size=harris_gaussian_size,
            threshold_ratio=harris_threshold,
            nms_kernel=harris_nms_kernel,
        )
        harris_corners = extract_harris_corners(
            harris_map,
            harris_mask,
            max_corners=max_corners,
            border_margin=border_margin,
        )

        cnn_overlay = draw_corners(rgb, cnn_corners, (255, 45, 45))
        harris_overlay = draw_corners(rgb, harris_corners, (35, 195, 255))
        cnn_heatmap_vis = cv2.cvtColor(colorize_response(cnn_heatmap), cv2.COLOR_BGR2RGB)
        harris_heatmap_vis = cv2.cvtColor(colorize_response(np.maximum(harris_map, 0), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "cnn": {
                "corner_count": len(cnn_corners),
                "threshold": resolved_threshold,
                "corners": [{"x": x, "y": y, "score": score} for x, y, score in cnn_corners],
            },
            "harris": {
                "corner_count": len(harris_corners),
                "threshold_ratio": harris_threshold,
                "corners": [{"x": x, "y": y, "score": score} for x, y, score in harris_corners],
            },
        }

        result_path = None
        if save_results:
            result_path = save_session(
                metadata,
                {
                    "original": rgb,
                    "cnn_overlay": cnn_overlay,
                    "cnn_heatmap": cnn_heatmap_vis,
                    "harris_overlay": harris_overlay,
                    "harris_heatmap": harris_heatmap_vis,
                },
            )

        return jsonify(
            {
                "original": encode_png(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                "cnn_overlay": encode_png(cv2.cvtColor(cnn_overlay, cv2.COLOR_RGB2BGR)),
                "cnn_heatmap": encode_png(cv2.cvtColor(cnn_heatmap_vis, cv2.COLOR_RGB2BGR)),
                "harris_overlay": encode_png(cv2.cvtColor(harris_overlay, cv2.COLOR_RGB2BGR)),
                "harris_heatmap": encode_png(cv2.cvtColor(harris_heatmap_vis, cv2.COLOR_RGB2BGR)),
                "cnn_corner_count": len(cnn_corners),
                "harris_corner_count": len(harris_corners),
                "cnn_threshold": resolved_threshold,
                "saved": save_results,
                "result_path": result_path,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, port=port, use_reloader=False)
