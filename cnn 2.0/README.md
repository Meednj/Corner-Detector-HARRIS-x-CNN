# CNN Corner Detector

A complete PyTorch corner detector that trains on synthetic geometric images and predicts corner heatmaps.

The project includes:

- A small U-Net style CNN for dense corner heatmap prediction
- Synthetic data generation for rectangles, polygons, and line structures
- Training script with validation
- Inference script with non-maximum suppression
- Checkpoint save/load helpers

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 20 --batch-size 16 --device cpu
```

If you have CUDA:

```bash
python train.py --epochs 20 --batch-size 32 --device cuda
```

The best model is saved to `runs/corner_detector_best.pt`.

## Run Inference

```bash
python infer.py --checkpoint runs/corner_detector_best.pt --image path/to/image.png --out prediction.png
```

The output image overlays detected corners on the input image.
For real images, the default inference path uses Canny preprocessing because the model was trained on synthetic line drawings. To run on the raw grayscale image instead:

```bash
python infer.py --checkpoint runs/corner_detector_best.pt --image path/to/image.png --out prediction.png --preprocess gray --threshold 0.35 --nms-kernel 7 --border-margin 0
```

## Model Output

The CNN predicts a single-channel heatmap with values in `[0, 1]`. Bright peaks indicate likely corners. During inference, local maxima above a threshold are returned as `(x, y, score)` detections.
