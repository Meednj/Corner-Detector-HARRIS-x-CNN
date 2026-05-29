"""
app.py — Flask server for CNN Corner Detector (Homographic Adaptation)
Run:  python app.py
Open: http://localhost:5000
"""

import base64, json, math, os, tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from corner_cnn import (CornerNet, train_model, detect_corners,
                         download_coco, IMG_SIZE, N_HOMO)

app = Flask(__name__, static_folder="static")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH  = "corner_cnn.pth"

device = None
model  = None


def initialize():
    global model, device
    if model is not None: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CornerNet(base=24)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded ← {MODEL_PATH}")
        except RuntimeError:
            print("Architecture changed — retraining…")
            os.remove(MODEL_PATH)
            _train()
    else:
        print("No weights found — training…")
        _train()

    model.to(device).eval()


def _train():
    global model
    paths = download_coco(600)
    model = train_model(model, paths,
                        epochs=25, batch_size=16,
                        size=IMG_SIZE, n_homo=N_HOMO)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved → {MODEL_PATH}")


def encode_png(arr):
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode()

def save_session(corners, gray, overlay):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / ts; out.mkdir(exist_ok=True)
    cv2.imwrite(str(out/"original.png"),        gray)
    cv2.imwrite(str(out/"corners_overlay.png"), overlay)
    with open(out/"results.json","w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "corner_count": len(corners),
                   "corners": [{"x":int(x),"y":int(y),"score":float(s)}
                                for x,y,s in corners]}, f, indent=2)
    return str(out)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    initialize()

    threshold  = float(request.form.get("threshold",  0.55))
    nms_radius = int(request.form.get("nms_radius",   18))
    max_pts    = int(request.form.get("max_corners",  50))
    do_save    = request.form.get("save_results","true").lower()=="true"

    gray = np.array(Image.open(request.files["image"].stream).convert("L"))
    fd,tmp = tempfile.mkstemp(suffix=".png"); os.close(fd)
    try:
        cv2.imwrite(tmp, gray)
        heatmap, response, corners = detect_corners(
                                    model, tmp,
                                    threshold=threshold,
                                    nms_radius=nms_radius,
                                    max_corners=max_pts,
                                    device=device
                                )
    finally:
        if os.path.exists(tmp): os.remove(tmp)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x,y,score in corners:
        norm  = min(score,1.0)
        color = (0,int(255*(1-norm*0.7)),int(255*norm))
        cv2.circle(overlay,(x,y),5,color,-1)
        cv2.circle(overlay,(x,y),6,(255,255,255),1)

    heatmap_vis = (heatmap * 255).astype(np.uint8)

    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_INFERNO)

    # Harris reference
    harris = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    harris_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    harris_vis[harris > 0.01 * harris.max()] = [0, 140, 255]
    result_path = save_session(corners,gray,overlay) if do_save else None
    return jsonify({
                    "original": encode_png(gray),
                    "heatmap": encode_png(heatmap_vis),
                    "overlay": encode_png(overlay),
                    "harris": encode_png(harris_vis),
                    "corner_count": len(corners),
                    "corners": [{"x":int(x),"y":int(y),"score":float(s)}
                                 for x,y,s in corners],
                    "saved": do_save, "result_path": result_path})


if __name__ == "__main__":
    initialize()
    app.run(debug=True, port=5000, use_reloader=False)