from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from corner_detector.dataset import SyntheticConfig, make_synthetic_sample


def main() -> None:
    out_dir = Path("demo")
    out_dir.mkdir(exist_ok=True)
    config = SyntheticConfig(image_size=128)

    import random

    image, heatmap = make_synthetic_sample(config, random.Random(42))
    image_u8 = (image * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.65 * np.stack([image_u8] * 3, axis=-1) + 0.35 * heatmap_color, 0, 255).astype(np.uint8)

    Image.fromarray(image_u8).save(out_dir / "synthetic_image.png")
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(out_dir / "corner_heatmap.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    print(f"saved demo images to {out_dir}")


if __name__ == "__main__":
    main()

