import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import Dataset, DataLoader


class PureCornerCNN(nn.Module):
    """
    Improved CNN for corner detection with better architecture.
    - Deeper encoder-decoder with batch normalization
    - Dropout for regularization
    - Dilated convolutions for larger receptive field
    - Multi-scale feature fusion
    """
    def __init__(self):
        super(PureCornerCNN, self).__init__()

        # Encoder with batch norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Note: dilation=2 with kernel=3 needs padding=2 to preserve spatial dimensions
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder with skip connections and batch norm
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        # Corner-specific head: multi-scale branches
        # Use padding=1 for all branches to maintain spatial dimensions
        self.corner_branch_a = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.corner_branch_b = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.corner_branch_c = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        # Merge and output
        self.merge = nn.Conv2d(48, 24, kernel_size=1)
        self.merge_bn = nn.BatchNorm2d(24)
        self.output = nn.Conv2d(24, 1, kernel_size=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        # Encoder with batch norm and dropout
        x1 = self.relu(self.bn1(self.conv1(x)))    # (B, 32, H, W)
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.bn2(self.conv2(x1)))   # (B, 64, H, W)
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.bn3(self.conv3(x2)))   # (B, 128, H, W)
        x3 = self.dropout(x3)
        
        x4 = self.relu(self.bn4(self.conv4(x3)))   # (B, 256, H, W)
        x4 = self.dropout(x4)

        # Decoder with skip connections
        x5 = self.relu(self.bn5(self.conv5(x4)))
        x5 = x5 + x3                                # skip from encoder level 3

        x6 = self.relu(self.bn6(self.conv6(x5)))
        x6 = x6 + x2                                # skip from encoder level 2

        x7 = self.relu(self.bn7(self.conv7(x6)))
        x7 = x7 + x1                                # skip from encoder level 1

        # Multi-scale branches for better feature extraction
        ba = self.relu(self.corner_branch_a(x7))
        bb = self.relu(self.corner_branch_b(x7))
        bc = self.relu(self.corner_branch_c(x7))

        # Concatenate and merge
        merged = torch.cat([ba, bb, bc], dim=1)    # (B, 48, H, W)
        merged = self.relu(self.merge_bn(self.merge(merged)))

        corners = self.sigmoid(self.output(merged))
        return corners


def harris_corners_numpy(img: np.ndarray, k: float = 0.04, threshold_percentile: int = 95) -> np.ndarray:
    """
    Compute Harris corner map as ground truth for training.
    Returns binary mask of corner locations.
    """
    img_f = img.astype(np.float32)

    # Gradients
    Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor elements, smoothed with Gaussian
    Ixx = cv2.GaussianBlur(Ix * Ix, (5, 5), sigmaX=1.0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (5, 5), sigmaX=1.0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (5, 5), sigmaX=1.0)

    # Harris response R = det(M) - k * trace(M)^2
    det_M   = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)

    # Keep only positive (corner) responses above threshold
    R = np.clip(R, 0, None)
    threshold = np.percentile(R, threshold_percentile)
    corner_map = (R > threshold).astype(np.float32)

    # Dilate slightly so the network has a visible target to learn
    corner_map = cv2.dilate(corner_map, np.ones((3, 3), np.uint8), iterations=1)

    return corner_map


class SyntheticCornerDataset(Dataset):
    """
    Generates synthetic images rich in corners with enhanced data augmentation:
    - Checkerboard patches
    - Random polygons / rectangles
    - L-shapes and T-intersections
    - Grid patterns
    - Enhanced noise and texture variety
    Ground truth is the Harris corner map of each image.
    """
    def __init__(self, num_samples: int = 5000, size: int = 64):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.zeros((self.size, self.size), dtype=np.float32)

        shape_type = np.random.choice(
            ['rectangle', 'polygon', 'checkerboard', 'l_shape', 't_shape', 'cross', 'grid', 'star'],
            p=[0.15, 0.12, 0.18, 0.12, 0.12, 0.12, 0.10, 0.09]
        )

        if shape_type == 'rectangle':
            x1, y1 = np.random.randint(5, self.size // 2, 2)
            x2, y2 = np.random.randint(self.size // 2, self.size - 5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), 1.0, thickness=-1)

        elif shape_type == 'polygon':
            n_pts = np.random.randint(4, 8)
            pts = np.random.randint(10, self.size - 10, (n_pts, 2))
            cv2.fillPoly(img, [pts], color=1.0)

        elif shape_type == 'checkerboard':
            cell = np.random.randint(6, 14)
            for r in range(0, self.size, cell):
                for c in range(0, self.size, cell):
                    if (r // cell + c // cell) % 2 == 0:
                        img[r:r+cell, c:c+cell] = 1.0

        elif shape_type == 'l_shape':
            cx, cy = np.random.randint(10, self.size - 20, 2)
            w, h = np.random.randint(10, 25, 2)
            t = np.random.randint(4, 8)
            cv2.rectangle(img, (cx, cy), (cx + w, cy + t), 1.0, -1)
            cv2.rectangle(img, (cx, cy), (cx + t, cy + h), 1.0, -1)

        elif shape_type == 't_shape':
            cx, cy = np.random.randint(10, self.size - 20, 2)
            w = np.random.randint(12, 24)
            h = np.random.randint(10, 20)
            t = np.random.randint(4, 8)
            cv2.rectangle(img, (cx, cy), (cx + w, cy + t), 1.0, -1)
            cv2.rectangle(img, (cx + w // 2 - t // 2, cy), (cx + w // 2 + t // 2, cy + h), 1.0, -1)

        elif shape_type == 'cross':
            cx, cy = np.random.randint(15, self.size - 15, 2)
            arm = np.random.randint(8, 18)
            t = np.random.randint(4, 8)
            cv2.rectangle(img, (cx - arm, cy - t // 2), (cx + arm, cy + t // 2), 1.0, -1)
            cv2.rectangle(img, (cx - t // 2, cy - arm), (cx + t // 2, cy + arm), 1.0, -1)

        elif shape_type == 'grid':
            spacing = np.random.randint(8, 16)
            thickness = np.random.randint(1, 3)
            for i in range(0, self.size, spacing):
                cv2.line(img, (0, i), (self.size, i), 1.0, thickness)
                cv2.line(img, (i, 0), (i, self.size), 1.0, thickness)

        elif shape_type == 'star':
            cx, cy = self.size // 2, self.size // 2
            n_arms = np.random.randint(5, 10)
            radius = np.random.randint(10, 20)
            t = np.random.randint(2, 5)
            for i in range(n_arms):
                angle = 2 * np.pi * i / n_arms
                x_end = cx + int(radius * np.cos(angle))
                y_end = cy + int(radius * np.sin(angle))
                cv2.line(img, (cx, cy), (x_end, y_end), 1.0, t)

        # Enhanced texture and noise with variety
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'texture'], p=[0.5, 0.2, 0.3])
        
        if noise_type == 'gaussian':
            img += np.random.randn(self.size, self.size).astype(np.float32) * np.random.uniform(0.02, 0.08)
        elif noise_type == 'salt_pepper':
            sp_noise = np.random.choice([0, 1, -1], (self.size, self.size), p=[0.95, 0.025, 0.025]).astype(np.float32)
            img = img + sp_noise * 0.1
        else:  # texture
            texture = np.random.rand(self.size, self.size).astype(np.float32) * np.random.uniform(0.05, 0.15)
            img = img + texture
        
        img = np.clip(img, 0, 1)

        # Ground truth: Harris corner map with variable threshold
        img_uint8 = (img * 255).astype(np.uint8)
        threshold_percentile = np.random.randint(88, 96)
        corner_gt = harris_corners_numpy(img_uint8, k=0.04, threshold_percentile=threshold_percentile)

        img_tensor    = torch.FloatTensor(img).unsqueeze(0)
        corner_tensor = torch.FloatTensor(corner_gt).unsqueeze(0)

        return img_tensor, corner_tensor


def corner_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted BCE: corners are rare, so we upweight positive pixels.
    Also adds a spatial smoothness penalty to avoid scattered noise predictions.
    """
    pos_weight = torch.tensor([20.0])
    bce = nn.functional.binary_cross_entropy(pred, target,
                                              weight=pos_weight * target + (1 - target))

    # Encourage spatial smoothness in non-corner regions
    diff_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    diff_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    smoothness = (diff_x.mean() + diff_y.mean()) * 0.01

    return bce + smoothness


def train_corner_cnn(model: nn.Module, epochs: int = 15, batch_size: int = 64) -> nn.Module:
    # ── Optimize for speed ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset   = SyntheticCornerDataset(num_samples=2000, size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training on {device}...", flush=True)
    print("Training Improved CNN Corner Detector (Harris-style GT)...", flush=True)
    print("=" * 55, flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = corner_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:>3}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.5f}", flush=True)

    print("=" * 55, flush=True)
    print("Training complete! CNN has learned Harris-style corner detection.", flush=True)
    return model


def detect_corners_pure_cnn(model: nn.Module, image_path: str,
                             threshold: float = 0.4, nms_radius: int = 5):
    """
    Run the trained CNN on a real image with multi-scale detection and improved NMS.
    Returns: (original_gray, corner_response_map, corner_points_list)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    original_size = img.shape
    
    # Multi-scale detection
    scales = [0.8, 1.0, 1.2]
    response_multi = None
    
    model.eval()
    with torch.no_grad():
        for scale in scales:
            if scale != 1.0:
                h_scaled = int(img.shape[0] * scale)
                w_scaled = int(img.shape[1] * scale)
                img_scaled = cv2.resize(img, (w_scaled, h_scaled))
            else:
                img_scaled = img
            
            img_resized = cv2.resize(img_scaled, (256, 256))
            img_tensor = torch.FloatTensor(img_resized / 255.0).unsqueeze(0).unsqueeze(0)
            
            response = model(img_tensor)
            response = response.squeeze().numpy()
            response = cv2.resize(response, (original_size[1], original_size[0]))
            
            if response_multi is None:
                response_multi = response / len(scales)
            else:
                response_multi += response / len(scales)

    response = response_multi

    # Improved NMS with better corner selection
    corners = []
    binary = (response > threshold).astype(np.uint8)
    
    # Use morphological operations for better NMS
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_radius * 2 + 1, nms_radius * 2 + 1))
    dilated = cv2.dilate(response, kernel)
    
    # Apply NMS: keep only local maxima
    nms_mask = (response == dilated) & (binary == 1)
    ys, xs = np.where(nms_mask)
    
    # Sort by response strength
    corner_list = [(int(x), int(y), float(response[y, x])) for y, x in zip(ys, xs)]
    corner_list.sort(key=lambda c: c[2], reverse=True)
    
    # Apply additional distance-based NMS
    corners = []
    for x, y, score in corner_list:
        # Check if this corner is far enough from already selected corners
        is_valid = True
        for cx, cy, _ in corners:
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < nms_radius:
                is_valid = False
                break
        if is_valid:
            corners.append((x, y, score))

    return img, response, corners


def visualize_results(original, response_map, corners, cnn_corners_img,
                       harris_ref, save_path="cnn_corners_result.jpg"):
    # Build overlay: draw detected corners on original
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for (x, y, score) in corners:
        color = (0, int(255 * score), 255 - int(200 * score))
        cv2.circle(overlay, (x, y), 4, color, -1)
        cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')

    titles = ['Original Image', 'CNN Corner Response',
              f'CNN Corners ({len(corners)} pts)', 'Harris Reference']
    imgs   = [original, response_map, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), harris_ref]
    cmaps  = ['gray', 'hot', None, 'hot']

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    plt.suptitle('Pure CNN Corner Detector  ·  Harris-style learned response',
                 color='#e0d7ff', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.show()
    print(f"Result saved as '{save_path}'")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Creating CNN with randomly initialized filters...")
    cnn_corner_detector = PureCornerCNN()

    total_params = sum(p.numel() for p in cnn_corner_detector.parameters())
    print(f"Model has {total_params:,} trainable parameters\n")

    trained_model = train_corner_cnn(cnn_corner_detector, epochs=20, batch_size=32)

    # ── Optionally save / load weights ────────────────────────────────────────
    # torch.save(trained_model.state_dict(), "corner_cnn.pth")
    # trained_model.load_state_dict(torch.load("corner_cnn.pth"))

    # ── Download test image if needed ─────────────────────────────────────────
    test_image = "test_image.jpg"
    if not os.path.exists(test_image):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            test_image,
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    original, cnn_response, corners = detect_corners_pure_cnn(
        trained_model, test_image, threshold=0.35, nms_radius=5
    )

    # Classical Harris for comparison
    harris_ref = cv2.cornerHarris(original, blockSize=2, ksize=3, k=0.04)
    harris_ref = np.clip(harris_ref, 0, None)
    harris_ref = cv2.normalize(harris_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── Build corner overlay image ─────────────────────────────────────────────
    cnn_corners_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for (x, y, score) in corners:
        cv2.circle(cnn_corners_img, (x, y), 4, (0, 255, 100), -1)

    visualize_results(original, cnn_response, corners,
                      cnn_corners_img, harris_ref,
                      save_path="cnn_corners_result.jpg")

    print(f"\nDetected {len(corners)} corners.")
    print("Done!")