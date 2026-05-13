"""
app.py — Flask server for the CNN Corner Detector UI
Run:  python app.py
Then open:  http://localhost:5000
"""

import io
import base64
import json
import numpy as np
import cv2
import torch
import os
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

# ── Import your model ─────────────────────────────────────────────────────────
from corner_cnn import PureCornerCNN, train_corner_cnn, detect_corners_pure_cnn

app = Flask(__name__, static_folder="static")

# ── Create results folder ───────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load / train model once at startup ───────────────────────────────────────
MODEL_PATH = "corner_cnn.pth"
model = PureCornerCNN()

# Try to load existing weights, but handle architecture changes gracefully
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Loaded saved model weights.")
        model_loaded = True
    except RuntimeError as e:
        print(f"⚠ Could not load existing weights (architecture changed): {str(e)[:100]}...")
        print("Training new model with improved architecture...")
        os.remove(MODEL_PATH)

if not model_loaded:
    print("Training improved CNN corner detector (this takes ~3-5 minutes)…")
    model = train_corner_cnn(model, epochs=50, batch_size=32)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Weights saved to {MODEL_PATH}")

model.eval()


def pil_to_gray_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("L"))


def np_to_b64_png(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode("utf-8")


def save_results(corners, original, cnn_response, overlay, heatmap_color, harris_color, filename_prefix=None):
    """
    Save detection results to files in the results folder.
    Returns path to saved results.
    """
    if filename_prefix is None:
        filename_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_subdir = os.path.join(RESULTS_DIR, filename_prefix)
    os.makedirs(result_subdir, exist_ok=True)
    
    # Save images
    cv2.imwrite(os.path.join(result_subdir, "original.png"), original)
    cv2.imwrite(os.path.join(result_subdir, "heatmap.png"), heatmap_color)
    cv2.imwrite(os.path.join(result_subdir, "overlay.png"), overlay)
    cv2.imwrite(os.path.join(result_subdir, "harris_comparison.png"), harris_color)
    
    # Save corner response map
    response_normalized = cv2.normalize(cnn_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(result_subdir, "corner_response.png"), response_normalized)
    
    # Save results as JSON
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "corner_count": len(corners),
        "corners": [{"x": int(x), "y": int(y), "score": float(score)} for x, y, score in corners],
        "stats": {
            "min_score": float(np.min(cnn_response)) if len(corners) > 0 else 0,
            "max_score": float(np.max(cnn_response)) if len(corners) > 0 else 0,
            "mean_score": float(np.mean([s for _, _, s in corners])) if len(corners) > 0 else 0,
            "image_size": list(original.shape)
        }
    }
    
    json_path = os.path.join(result_subdir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {result_subdir}")
    return result_subdir, results_json


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    pil_img = Image.open(file.stream)
    gray = pil_to_gray_np(pil_img)

    # Save the upload to a real OS temp file so OpenCV can read it on Windows too.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    if not cv2.imwrite(tmp_path, gray):
        os.remove(tmp_path)
        return jsonify({"error": "Failed to save uploaded image"}), 500

    threshold = float(request.form.get("threshold", 0.35))
    nms_radius = int(request.form.get("nms_radius", 5))
    save_results_flag = request.form.get("save_results", "true").lower() == "true"

    try:
        original, cnn_response, corners = detect_corners_pure_cnn(
            model, tmp_path, threshold=threshold, nms_radius=nms_radius
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Build corner overlay
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for (x, y, score) in corners:
        cv2.circle(overlay, (x, y), 5, (0, 255, 120), -1)
        cv2.circle(overlay, (x, y), 6, (255, 255, 255), 1)

    # Heatmap of response
    heatmap_norm = cv2.normalize(cnn_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)

    # Classical Harris for comparison panel
    harris = cv2.cornerHarris(original, blockSize=2, ksize=3, k=0.04)
    harris = np.clip(harris, 0, None)
    harris = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    harris_color = cv2.applyColorMap(harris, cv2.COLORMAP_INFERNO)

    # Save results if requested
    result_dir = None
    results_json = None
    if save_results_flag:
        result_dir, results_json = save_results(
            corners, original, cnn_response, overlay, heatmap_color, harris_color
        )

    return jsonify({
        "original":   np_to_b64_png(original),
        "heatmap":    np_to_b64_png(heatmap_color),
        "overlay":    np_to_b64_png(overlay),
        "harris":     np_to_b64_png(harris_color),
        "corner_count": len(corners),
        "saved": save_results_flag,
        "result_path": result_dir
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
