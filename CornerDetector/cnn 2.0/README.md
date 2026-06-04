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
The default detector is the trained CNN. The script uses the saved training
resolution from the checkpoint and chooses a per-image threshold unless you pass
`--threshold`.

```bash
python infer.py --image path/to/test.png --out prediction.png
```

To compare against a classical Harris baseline explicitly:

```bash
python infer.py --detector harris --image path/to/test.png --out harris_prediction.png
```

For clean synthetic-looking images such as the cube, CNN+Canny can find more edge corners:

```bash
python infer.py --checkpoint runs/corner_detector_best.pt --image path/to/image.png --out prediction.png --preprocess canny --threshold 0.95 --nms-kernel 15
```

## Model Output

The CNN predicts a single-channel heatmap with values in `[0, 1]`. Bright peaks indicate likely corners. During inference, local maxima above a threshold are returned as `(x, y, score)` detections.
