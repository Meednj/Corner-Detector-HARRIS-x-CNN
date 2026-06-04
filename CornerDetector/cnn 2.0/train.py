from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from corner_detector.dataset import SyntheticConfig, SyntheticCornerDataset
from corner_detector.model import CornerDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN corner detector on synthetic geometry.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="runs")
    return parser.parse_args()


def weighted_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Corners occupy very few pixels. Focal weighting keeps the model from
    # learning the easy "mostly background" solution.
    probs = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = (1.0 - pt).pow(2.0)
    positive_weight = 1.0 + 35.0 * targets
    return (bce * focal * positive_weight).mean()


@torch.no_grad()
def evaluate(model: CornerDetector, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = weighted_heatmap_loss(logits, targets)
        total += float(loss.item()) * images.size(0)
        count += images.size(0)
    return total / max(1, count)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SyntheticConfig(image_size=args.image_size)
    train_ds = SyntheticCornerDataset(args.train_samples, config=config, seed=100)
    val_ds = SyntheticCornerDataset(args.val_samples, config=config, seed=100_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_args = {"in_channels": 1, "base_channels": args.base_channels}
    model = CornerDetector(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, disable=True)
        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = weighted_heatmap_loss(logits, targets)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * images.size(0)
            seen += images.size(0)
            progress.set_postfix(loss=running / max(1, seen))

        train_loss = running / max(1, seen)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "model_args": model_args,
                "image_size": args.image_size,
                "val_loss": best_val,
            }
            torch.save(checkpoint, out_dir / "corner_detector_best.pt")

    print(f"best_val_loss={best_val:.5f}")


if __name__ == "__main__":
    main()
