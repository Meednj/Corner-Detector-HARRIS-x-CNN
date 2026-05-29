"""
corner_cnn.py — CNN Corner Detector with GPU Homographic Adaptation
====================================================================

WHY PREVIOUS ATTEMPTS FAILED
------------------------------
Synthetic shapes (rectangles, L-shapes) on white backgrounds look NOTHING
like a real cube with lighting/shadows/gradients. The CNN learned to detect
"white pixel → black pixel edges on a clean background", which is useless
on real photos. That's why it detected border noise instead of cube corners.

THE CORRECT APPROACH
---------------------
1. Download ~500 real images (COCO val, no labels needed).
2. For each image, apply N random homographic warps ON THE GPU.
3. Run a GPU-accelerated Harris detector on every warped image.
4. Warp all Harris responses BACK and average them.
   → True corners are stable across viewpoints; edges cancel out.
5. Train the CNN on (real image → consensus corner map).

The CNN then learns what corners LOOK LIKE in real photos.

SPEED
------
- Harris is computed in parallel on GPU using F.conv2d (no OpenCV in loop)
- All warping via F.grid_sample (GPU, sub-pixel accurate)
- Dataset cached as .npy to avoid re-computing on restart
- ~10-15 min total on RTX 3050
"""

import os, math, argparse, urllib.request, zipfile, random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════

def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark  = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {torch.cuda.get_device_name(0)}  ({total:.1f} GB)", flush=True)
    else:
        torch.set_num_threads(os.cpu_count() or 1)
        print("[CPU]", flush=True)
    return dev


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class CBR(nn.Sequential):
    def __init__(self, ci, co, k=3, d=1):
        super().__init__(
            nn.Conv2d(ci, co, k, padding=d*(k//2), dilation=d, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class CornerNet(nn.Module):
    """Fully-dilated U-Net, no pooling → pixel-accurate. Fits in 4 GB."""
    def __init__(self, base=24):
        super().__init__()
        b = base
        self.e1  = nn.Sequential(CBR(1,b),    CBR(b,b))
        self.e2  = nn.Sequential(CBR(b,b*2),  CBR(b*2,b*2))
        self.e3  = nn.Sequential(CBR(b*2,b*4),CBR(b*4,b*4))
        self.bot = nn.Sequential(CBR(b*4,b*4,d=2),CBR(b*4,b*4,d=4),CBR(b*4,b*4,d=2))
        self.p3  = nn.Conv2d(b*4,b*4,1,bias=False)
        self.p2  = nn.Conv2d(b*2,b*4,1,bias=False)
        self.p1  = nn.Conv2d(b,  b*2,1,bias=False)
        self.d3  = nn.Sequential(CBR(b*4,b*4),CBR(b*4,b*4))
        self.d2  = nn.Sequential(CBR(b*4,b*2),CBR(b*2,b*2))
        self.d1  = nn.Sequential(CBR(b*2,b),  CBR(b,b))
        self.head= nn.Sequential(CBR(b,b//2), nn.Conv2d(b//2,1,1))
        self.drop= nn.Dropout2d(0.1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.drop(e1)); e3=self.e3(self.drop(e2))
        b=self.bot(e3)
        d=self.d3(b+self.p3(e3)); d=self.d2(d+self.p2(e2)); d=self.d1(d+self.p1(e1))
        return self.head(d)


# ══════════════════════════════════════════════════════════════
#  GPU HARRIS DETECTOR
# ══════════════════════════════════════════════════════════════

def _make_sobel(dev):
    """Returns Sobel X and Y kernels as (1,1,3,3) tensors."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                      dtype=torch.float32, device=dev).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                      dtype=torch.float32, device=dev).view(1,1,3,3)
    return kx, ky

def _make_gauss(size=5, sigma=1.0, dev='cpu'):
    """Returns a (1,1,size,size) Gaussian blur kernel."""
    ax  = torch.arange(size, dtype=torch.float32, device=dev) - size//2
    g   = torch.exp(-ax**2/(2*sigma**2))
    k   = torch.outer(g, g); k /= k.sum()
    return k.view(1,1,size,size)

@torch.no_grad()
def harris_gpu(imgs: torch.Tensor, k: float = 0.04) -> torch.Tensor:
    """
    Vectorised Harris corner response on GPU.
    imgs : (B, 1, H, W)  float32 in [0,1]
    returns: (B, 1, H, W)  Harris R score (clipped ≥0, normalised per image)
    """
    dev  = imgs.device
    B, _, H, W = imgs.shape
    kx, ky = _make_sobel(dev)
    kg     = _make_gauss(5, 1.0, dev).expand(B, -1, -1, -1)  # wrong — fix below

    # Sobel gradients
    Ix = F.conv2d(imgs, kx, padding=1)   # (B,1,H,W)
    Iy = F.conv2d(imgs, ky, padding=1)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Gaussian smoothing of structure tensor  (per-channel, groups=B)
    gauss = _make_gauss(5, 1.0, dev)   # (1,1,5,5)

    def blur(t):
        # t: (B,1,H,W) → treat as (1, B, H, W) to share the single kernel
        return F.conv2d(t.view(1,B,H,W), gauss.expand(B,1,5,5),
                        padding=2, groups=B).view(B,1,H,W)

    Ixx = blur(Ixx); Iyy = blur(Iyy); Ixy = blur(Ixy)

    # Harris response R = det(M) - k * trace(M)^2
    det   = Ixx*Iyy - Ixy*Ixy
    trace = Ixx + Iyy
    R     = det - k * trace**2

    # Clip negatives, normalise per image to [0,1]
    R = R.clamp(min=0)
    mx = R.view(B,-1).max(1).values.view(B,1,1,1).clamp(min=1e-8)
    return R / mx   # (B,1,H,W)


# ══════════════════════════════════════════════════════════════
#  GPU HOMOGRAPHIC WARP
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def random_homography_gpu(B: int, H: int, W: int,
                           scale: float, dev) -> torch.Tensor:
    """
    Returns (B, 3, 3) random homography matrices on GPU.
    scale controls max perspective distortion (0.12 is mild).
    """
    # Start from identity, add random noise to corners
    pts_src = torch.tensor([[0,0],[W,0],[W,H],[0,H]],
                            dtype=torch.float32, device=dev)           # (4,2)
    pts_src = pts_src.unsqueeze(0).expand(B,-1,-1)                    # (B,4,2)
    noise   = (torch.rand(B,4,2,device=dev)*2-1) * scale \
              * torch.tensor([W,H],dtype=torch.float32,device=dev)
    pts_dst = pts_src + noise                                          # (B,4,2)

    # Compute H matrix via DLT (vectorised)
    Ms = []
    for b in range(B):
        src = pts_src[b].cpu().numpy()
        dst = pts_dst[b].cpu().numpy()
        M, _ = cv2.findHomography(src, dst)
        if M is None: M = np.eye(3, dtype=np.float32)
        Ms.append(torch.from_numpy(M.astype(np.float32)))
    return torch.stack(Ms, 0).to(dev)   # (B,3,3)


@torch.no_grad()
def warp_images(imgs: torch.Tensor, Ms: torch.Tensor) -> torch.Tensor:
    """
    Warp (B,1,H,W) images by (B,3,3) homographies using grid_sample.
    """
    B, _, H, W = imgs.shape
    dev = imgs.device

    # Build normalised grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=dev),
        torch.linspace(-1,1,W,device=dev), indexing='ij')
    ones = torch.ones_like(xx)
    grid = torch.stack([xx,yy,ones], -1).view(1,H*W,3) \
               .expand(B,-1,-1)   # (B, H*W, 3)

    # Convert normalised → pixel coords, apply H, convert back
    Hs = Ms.clone()
    Hs[:,0,2] /= (W/2); Hs[:,1,2] /= (H/2)  # rough norm (good enough)

    # Use affine_grid approach: convert H to sampling coords
    # Full perspective warp via manual grid computation
    pixel_grid = torch.stack([
        (xx+1)/2*(W-1), (yy+1)/2*(H-1), ones], -1).view(1,H*W,3).expand(B,-1,-1)

    warped_pts = torch.bmm(pixel_grid, Ms.transpose(1,2))   # (B,H*W,3)
    wx = warped_pts[:,:,0] / warped_pts[:,:,2].clamp(min=1e-6)
    wy = warped_pts[:,:,1] / warped_pts[:,:,2].clamp(min=1e-6)

    # Normalise back to [-1,1]
    wx = (wx / (W-1)) * 2 - 1
    wy = (wy / (H-1)) * 2 - 1
    sample_grid = torch.stack([wx, wy], -1).view(B, H, W, 2)

    return F.grid_sample(imgs, sample_grid,
                         mode='bilinear', padding_mode='zeros',
                         align_corners=True)


# ══════════════════════════════════════════════════════════════
#  DATASET — real images + GPU homographic adaptation GT
# ══════════════════════════════════════════════════════════════

DATA_DIR   = Path("ha_data")
IMAGES_DIR = DATA_DIR / "images"
GT_DIR     = DATA_DIR / "gt"
COCO_URL   = "http://images.cocodataset.org/zips/val2017.zip"
IMG_SIZE   = 240   # train resolution — fits in 4 GB
N_HOMO     = 15    # homographies per image for GT generation


def download_coco(max_images=600):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(IMAGES_DIR.glob("*.jpg"))
    if len(existing) >= max_images:
        print(f"Found {len(existing)} images.", flush=True)
        return [str(p) for p in existing[:max_images]]

    zip_path = DATA_DIR / "val2017.zip"
    if not zip_path.exists():
        print("Downloading COCO val2017 (~1 GB) …", flush=True)
        def _prog(b, bs, t):
            pct = min(b*bs/t*100, 100)
            print(f"\r  {pct:.0f}%", end="", flush=True)
        urllib.request.urlretrieve(COCO_URL, zip_path, reporthook=_prog)
        print()
    print("Extracting …", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith('.jpg')][:max_images]
        for m in members:
            dst = IMAGES_DIR / Path(m).name
            if not dst.exists():
                with z.open(m) as src, open(dst,'wb') as f:
                    f.write(src.read())
    paths = sorted(IMAGES_DIR.glob("*.jpg"))[:max_images]
    print(f"Ready: {len(paths)} images.", flush=True)
    return [str(p) for p in paths]

def gt_nms(gt, radius=10):
    """
    Non-maximum suppression for GT maps.
    Keeps only local maxima.
    """
    if gt.max() <= 0:
        return gt

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1)
    )

    dilated = cv2.dilate(gt, kernel)

    peaks = (gt == dilated) & (gt > 0)

    out = np.zeros_like(gt, dtype=np.float32)
    out[peaks] = gt[peaks]

    return out

@torch.no_grad()
def compute_gt_gpu(gray_np: np.ndarray, dev,
                   n_homo: int = N_HOMO,
                   size: int = IMG_SIZE) -> np.ndarray:
    """
    GPU homographic adaptation:
    Apply n_homo random warps, run GPU Harris on each,
    warp responses back, average → robust corner map.
    """
    gray_r = cv2.resize(gray_np, (size, size))
    img_t  = torch.from_numpy(gray_r.astype(np.float32)/255.0) \
                  .unsqueeze(0).unsqueeze(0).to(dev)   # (1,1,H,W)
    H, W   = size, size

    accum  = torch.zeros(1,1,H,W,device=dev)

    for _ in range(n_homo):
        Ms    = random_homography_gpu(1, H, W, scale=0.12, dev=dev)
        Ms_inv= torch.linalg.inv(Ms)

        warped = warp_images(img_t, Ms)              # (1,1,H,W)
        resp   = harris_gpu(warped)                  # (1,1,H,W) in [0,1]
        resp_bk= warp_images(resp, Ms_inv)           # warp back
        accum += resp_bk

    # Also add the straight Harris
    accum += harris_gpu(img_t) * (n_homo * 0.3)

    accum = accum.squeeze().cpu().numpy()
    accum /= accum.max() + 1e-8

    # Normalize
    accum /= accum.max() + 1e-8

    # VERY strict threshold (top 2-5%)
    valid = accum[accum > 0.02]

    if len(valid) > 0:
        thr = np.percentile(valid, 97)
    else:
        thr = 0.0

    # Hard threshold
    gt = np.zeros_like(accum, dtype=np.float32)
    gt[accum >= thr] = accum[accum >= thr]

    # GT non-maximum suppression
    gt = gt_nms(gt, radius=10)

    # Re-normalize
    if gt.max() > 0:
        gt /= gt.max()

    # Optional sharpening
    gt = np.power(gt, 2.5)

    return gt.astype(np.float32)


def build_dataset(image_paths, dev, size=IMG_SIZE, n_homo=N_HOMO):
    """Pre-compute and cache GT maps, then load everything into GPU RAM."""
    GT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-compute missing GT maps
    missing = [p for p in image_paths
               if not (GT_DIR / (Path(p).stem+".npy")).exists()]
    if missing:
        print(f"Computing GT for {len(missing)} images on GPU …", flush=True)
        for i, path in enumerate(missing):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            gt = compute_gt_gpu(gray, dev, n_homo, size)
            np.save(GT_DIR / (Path(path).stem+".npy"), gt)
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(missing)}", flush=True)
        print("GT done.", flush=True)
    else:
        print("GT cache found.", flush=True)

    # Load into GPU RAM
    print("Loading dataset into GPU …", flush=True)
    imgs_l, gts_l = [], []
    for path in image_paths:
        gt_path = GT_DIR / (Path(path).stem+".npy")
        if not gt_path.exists(): continue
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None: continue
        gray_r = cv2.resize(gray, (size,size)).astype(np.float32)/255.0
        gt     = np.load(gt_path)
        imgs_l.append(torch.from_numpy(gray_r).unsqueeze(0))
        gts_l.append(torch.from_numpy(gt).unsqueeze(0))

    imgs = torch.stack(imgs_l).to(dev)   # (N,1,H,W) on GPU
    gts  = torch.stack(gts_l).to(dev)
    print(f"Dataset: {imgs.shape[0]} images on {dev}.", flush=True)
    return imgs, gts


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION (GPU, in-place)
# ══════════════════════════════════════════════════════════════

def augment(x, y):
    if random.random() > 0.5: x=x.flip(-1); y=y.flip(-1)
    if random.random() > 0.5: x=x.flip(-2); y=y.flip(-2)
    # brightness jitter
    x = (x * (0.6 + random.random()*0.8)).clamp(0,1)
    # gaussian noise
    x = (x + torch.randn_like(x)*0.02).clamp(0,1)
    return x, y


# ══════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════

def focal_loss(pred, gt, pos_w=30.0, gamma=2.0):
    prob   = torch.sigmoid(pred)
    ce     = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    pt     = gt*prob + (1-gt)*(1-prob)
    return ((1-pt)**gamma * ce * (pos_w*gt + (1-gt))).mean()


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def train_model(model, image_paths,
                epochs=25, batch_size=16,
                size=IMG_SIZE, n_homo=N_HOMO, lr=3e-4):

    dev   = get_device()
    model = model.to(dev)

    imgs, gts = build_dataset(image_paths, dev, size, n_homo)
    N = imgs.shape[0]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type=="cuda")

    print(f"\nTraining  epochs={epochs}  batch={batch_size}  N={N}", flush=True)
    print("="*55, flush=True)

    best_loss=float('inf'); best_sd=None

    for epoch in range(epochs):
        model.train()
        perm=torch.randperm(N,device=dev); total=0.0; steps=0

        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            x, y = augment(imgs[idx], gts[idx])

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16,
                                enabled=dev.type=="cuda"):
                loss = focal_loss(model(x), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += loss.item(); steps += 1

        scheduler.step()
        avg = total/steps
        if avg < best_loss:
            best_loss=avg
            best_sd={k:v.clone() for k,v in model.state_dict().items()}

        print(f"  Epoch {epoch+1:>3}/{epochs}  "
              f"loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    model.load_state_dict(best_sd)
    print("="*55)
    print(f"Done. Best loss={best_loss:.5f}", flush=True)
    return model


# ══════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════

def detect_corners(model, image_path, threshold=0.55,
                   nms_radius=18, max_corners=50, device=None):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError(f"Cannot load: {image_path}")
    if device is None: device = get_device()
    H, W = gray.shape
    model.to(device).eval()
    response = np.zeros((H,W), dtype=np.float32)

    with torch.no_grad():
        for s in [0.75, 1.0, 1.25]:
            rs  = cv2.resize(gray,(int(W*s),int(H*s)))
            inp = torch.from_numpy(rs.astype(np.float32)/255.0) \
                      .unsqueeze(0).unsqueeze(0).to(device)
            out = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            response += cv2.resize(out,(W,H)) / 3.0

    response[response < 0.05] = 0.0
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(nms_radius*2+1,nms_radius*2+1))
    nms_map = (response==cv2.dilate(response,kernel)) & (response>threshold)

    ys,xs = np.where(nms_map)
    cands = sorted([(int(x),int(y),float(response[y,x]))
                    for y,x in zip(ys,xs)], key=lambda c:c[2], reverse=True)
    corners=[]
    for x,y,sc in cands:
        if all(math.hypot(x-cx,y-cy)>=nms_radius for cx,cy,_ in corners):
            corners.append((x,y,sc))
        if len(corners)>=max_corners: break
    return gray, response, corners


# ══════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════

def visualize(gray, corners, save_path="corners_result.jpg"):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x,y,score in corners:
        norm  = min(score,1.0)
        color = (0, int(255*(1-norm*0.7)), int(255*norm))
        cv2.circle(overlay,(x,y),5,color,-1)
        cv2.circle(overlay,(x,y),6,(255,255,255),1)
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    fig.patch.set_facecolor('#0f172a')
    axes[0].imshow(gray,cmap='gray')
    axes[0].set_title('Original',color='white',fontsize=13,fontweight='bold',pad=10)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected Corners ({len(corners)} pts)',
                      color='white',fontsize=13,fontweight='bold',pad=10)
    axes[1].axis('off')
    plt.suptitle('CNN Corner Detector — Homographic Adaptation',
                 color='#818cf8',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout()
    plt.savefig(save_path,dpi=150,bbox_inches='tight',
                facecolor='#0f172a',edgecolor='none')
    plt.show()
    print(f"Saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",         default="test_image.jpg")
    ap.add_argument("--images_dir",    default=None)
    ap.add_argument("--max_images",    type=int,   default=600)
    ap.add_argument("--epochs",        type=int,   default=25)
    ap.add_argument("--batch",         type=int,   default=16)
    ap.add_argument("--size",          type=int,   default=240)
    ap.add_argument("--n_homo",        type=int,   default=15)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--nms_radius", type=int, default=18)
    ap.add_argument("--max_corners",   type=int,   default=50)
    ap.add_argument("--weights",       default="corner_cnn.pth")
    ap.add_argument("--retrain",       action="store_true")
    args = ap.parse_args()

    net = CornerNet(base=24)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    if os.path.exists(args.weights) and not args.retrain:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))
        print(f"Loaded ← {args.weights}")
    else:
        if args.images_dir:
            paths = [str(p) for p in Path(args.images_dir).glob("*.jpg")]
            paths+= [str(p) for p in Path(args.images_dir).glob("*.png")]
            paths = paths[:args.max_images]
        else:
            paths = download_coco(args.max_images)
        net = train_model(net, paths,
                          epochs=args.epochs, batch_size=args.batch,
                          size=args.size, n_homo=args.n_homo)
        torch.save(net.state_dict(), args.weights)

    if not os.path.exists(args.image):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            args.image)

    gray, response, corners = detect_corners(
        net, args.image,
        threshold=args.threshold, nms_radius=args.nms_radius,
        max_corners=args.max_corners)
    visualize(gray, corners)
    print(f"Detected {len(corners)} corners.")