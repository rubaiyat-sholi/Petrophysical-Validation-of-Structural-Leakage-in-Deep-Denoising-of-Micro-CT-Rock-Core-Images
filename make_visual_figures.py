"""
Generate two publication figures:
  figures/fig_visual_comparison.pdf/png  -- LQ|Gaussian|NLM|DnCNN|UNet|UNet+Leakage|HQ
                                            with zoomed pore-detail inset
  figures/fig_3d_ortho.pdf/png           -- 3-D orthogonal view from stacked slices 37-99
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models import DnCNN, UNet, UNetLeakage

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
HQ_DIR = os.path.join(BASE, "shard_data", "initialOilHigh")
LQ_DIR = os.path.join(BASE, "shard_data", "initialQilQuick")
CKPTS  = {
    "DnCNN":        os.path.join(BASE, "runs", "dncnn_20260629_151815",      "best.pt"),
    "UNet":         os.path.join(BASE, "runs", "unet_20260629_163658",       "best.pt"),
    "UNet+Leakage": os.path.join(BASE, "runs", "unet_leakage_20260629_182602", "best.pt"),
}
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "figure.dpi":       150,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

NORM  = 65535.0
SLICE = 92          # representative test slice used throughout paper
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Okabe-Ito palette for later use
OKI = ["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]

# ── helpers ────────────────────────────────────────────────────────────────────
def sorted_tiffs(d):
    import glob
    return sorted(glob.glob(os.path.join(d, "*.tif")))

def load_slice_f32(path):
    return tifffile.imread(path).astype(np.float32) / NORM

def load_model(cls, path):
    m = cls()
    ckpt = torch.load(path, map_location=DEVICE)
    # checkpoints may be wrapped: {"model": state_dict, "epoch": ..., ...}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    m.load_state_dict(state)
    m.to(DEVICE).eval()
    return m

def infer(model, arr_f32):
    """Run model on a full 1024x1024 float32 array, return float32 array."""
    with torch.no_grad():
        t = torch.from_numpy(arr_f32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = model(t)
    return out.squeeze().cpu().numpy().astype(np.float32)

def apply_gaussian(arr):
    return gaussian_filter(arr, sigma=1.0).astype(np.float32)

def apply_nlm(arr):
    sigma = float(np.mean(estimate_sigma(arr)))
    return denoise_nl_means(arr, h=1.15*sigma,
                            patch_size=5, patch_distance=6,
                            fast_mode=True).astype(np.float32)

# ── load all denoisers ─────────────────────────────────────────────────────────
print("Loading models …")
dncnn  = load_model(DnCNN,        CKPTS["DnCNN"])
unet   = load_model(UNet,         CKPTS["UNet"])
unetlk = load_model(UNetLeakage,  CKPTS["UNet+Leakage"])

# ── load slice 92 ─────────────────────────────────────────────────────────────
hq_paths = sorted_tiffs(HQ_DIR)
lq_paths = sorted_tiffs(LQ_DIR)

lq_arr = load_slice_f32(lq_paths[SLICE])
hq_arr = load_slice_f32(hq_paths[SLICE])

print(f"Slice {SLICE}: LQ range [{lq_arr.min():.3f}, {lq_arr.max():.3f}]  "
      f"HQ range [{hq_arr.min():.3f}, {hq_arr.max():.3f}]")

print("Applying Gaussian …")
gauss_arr = apply_gaussian(lq_arr)

print("Applying NLM (may take ~60 s) …")
nlm_arr = apply_nlm(lq_arr)

print("Running DnCNN …")
dncnn_arr = infer(dncnn, lq_arr)

print("Running UNet …")
unet_arr = infer(unet, lq_arr)

print("Running UNet+Leakage …")
unetlk_arr = infer(unetlk, lq_arr)

METHODS = {
    "LQ (noisy)":     lq_arr,
    "Gaussian":       gauss_arr,
    "NLM":            nlm_arr,
    "DnCNN":          dncnn_arr,
    "U-Net":          unet_arr,
    "U-Net+Leakage":  unetlk_arr,
    "HQ (reference)": hq_arr,
}
LABELS = list(METHODS.keys())
IMAGES = list(METHODS.values())
N = len(LABELS)

# consistent display range: 1st – 99th percentile of HQ
vlo, vhi = np.percentile(hq_arr, 1), np.percentile(hq_arr, 99)

# ── zoom crop — pick a pore-rich region ───────────────────────────────────────
# use Otsu on HQ to find oil-rich region
thr = threshold_otsu(hq_arr)
oil_mask = hq_arr > thr
# find centre of mass of oil region
ys, xs = np.where(oil_mask)
cy, cx = int(np.median(ys)), int(np.median(xs))
cy = np.clip(cy, 128, hq_arr.shape[0]-128)
cx = np.clip(cx, 128, hq_arr.shape[1]-128)
CROP = 200   # half-size of zoom box
r0, r1 = cy-CROP, cy+CROP
c0, c1 = cx-CROP, cx+CROP
print(f"Zoom crop: rows {r0}:{r1}, cols {c0}:{c1}")

# ── FIG 1: VISUAL COMPARISON PANEL ────────────────────────────────────────────
print("\nGenerating fig_visual_comparison …")

fig = plt.figure(figsize=(18, 6.5))
gs  = gridspec.GridSpec(2, N, figure=fig,
                        hspace=0.04, wspace=0.03,
                        left=0.01, right=0.99, top=0.92, bottom=0.02)

for col, (lbl, img) in enumerate(zip(LABELS, IMAGES)):
    # ── full slice (top row) ──
    ax_top = fig.add_subplot(gs[0, col])
    ax_top.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
    ax_top.set_title(lbl, fontsize=8, pad=3)
    ax_top.axis("off")

    # draw zoom rectangle on top row
    rect = patches.Rectangle((c0, r0), c1-c0, r1-r0,
                              linewidth=1.2, edgecolor="#E69F00",
                              facecolor="none")
    ax_top.add_patch(rect)

    # ── zoomed inset (bottom row) ──
    ax_bot = fig.add_subplot(gs[1, col])
    crop = img[r0:r1, c0:c1]
    ax_bot.imshow(crop, cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
    ax_bot.axis("off")

    # highlight border on HQ
    if col == N-1:
        for spine in ax_bot.spines.values():
            spine.set_edgecolor("#009E73")
            spine.set_linewidth(2)
            spine.set_visible(True)

# row labels
fig.text(0.003, 0.74, "Full slice", va="center", rotation=90, fontsize=8)
fig.text(0.003, 0.25, "Zoomed region", va="center", rotation=90, fontsize=8)

fig.suptitle(
    f"Qualitative denoising comparison — Ketton carbonate slice {SLICE}",
    fontsize=9, y=0.97
)

out_stem = os.path.join(FIG_DIR, "fig_visual_comparison")
fig.savefig(out_stem + ".pdf")
fig.savefig(out_stem + ".png", dpi=300)
plt.close(fig)
print(f"  Saved {out_stem}.pdf / .png")

# ── FIG 2: 3-D ORTHOGONAL VIEW ────────────────────────────────────────────────
print("\nBuilding 3-D HQ volume from slices 37-99 …")

hq_vol = []
for idx in range(37, 100):
    if idx < len(hq_paths):
        sl = tifffile.imread(hq_paths[idx]).astype(np.float32) / NORM
        if sl.shape == (1024, 1024):
            hq_vol.append(sl)

vol = np.stack(hq_vol, axis=0)   # (D, H, W)
print(f"  Volume shape: {vol.shape}  "
      f"range [{vol.min():.3f}, {vol.max():.3f}]")

# also build LQ volume for side-by-side
lq_vol = []
for idx in range(37, 100):
    if idx < len(lq_paths):
        sl = tifffile.imread(lq_paths[idx]).astype(np.float32) / NORM
        if sl.shape == (1024, 1024):
            lq_vol.append(sl)
lq_vol = np.stack(lq_vol, axis=0)

D, H, W = vol.shape
mid_z = D // 2
mid_y = H // 2
mid_x = W // 2

vlo3, vhi3 = np.percentile(vol, 1), np.percentile(vol, 99)

# display range: subsample axial views to avoid huge display
subsample = 2   # show every 2nd pixel in planar views

fig3, axes = plt.subplots(2, 3, figsize=(13, 9))

pairs = [
    ("HQ reference", vol),
    ("LQ (noisy)",   lq_vol),
]

planes = [
    ("XY  (axial)",       lambda v, mz, my, mx: v[mz, ::subsample, ::subsample],    "Slice axis (Z)", "Col (X)"),
    ("XZ  (coronal)",     lambda v, mz, my, mx: v[::1, my, ::subsample],             "Slice (Z)",      "Col (X)"),
    ("YZ  (sagittal)",    lambda v, mz, my, mx: v[::1, ::subsample, mx],             "Slice (Z)",      "Row (Y)"),
]

for row, (vol_lbl, v) in enumerate(pairs):
    for col, (plane_lbl, extractor, ylabel, xlabel) in enumerate(planes):
        ax = axes[row, col]
        plane_img = extractor(v, mid_z, mid_y, mid_x)
        ax.imshow(plane_img, cmap="gray", vmin=vlo3, vmax=vhi3,
                  aspect="auto", interpolation="nearest")
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel if col == 0 else "", fontsize=7)
        ax.tick_params(labelsize=6)
        if row == 0:
            ax.set_title(plane_lbl, fontsize=8, pad=3)
        if col == 0:
            ax.set_ylabel(f"{vol_lbl}\n{ylabel}", fontsize=7)

# add cross-hair lines to all panels
for row in range(2):
    v = pairs[row][1]
    for col in range(3):
        ax = axes[row, col]
        yl, xl = ax.get_xlim(), ax.get_ylim()
        img_h, img_w = [
            (v[mid_z, ::subsample, ::subsample].shape,),
            (v[::1, mid_y, ::subsample].shape,),
            (v[::1, ::subsample, mid_x].shape,),
        ][col][0]
        ax.axhline(img_h // 2, color="#E69F00", linewidth=0.8, alpha=0.7)
        ax.axvline(img_w // 2, color="#E69F00", linewidth=0.8, alpha=0.7)

fig3.suptitle(
    "3-D orthogonal cross-sections — Ketton carbonate (HQ vs LQ)\n"
    f"Volume: {D}×{H}×{W} voxels at 6 µm · slices 37–99",
    fontsize=9
)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out3 = os.path.join(FIG_DIR, "fig_3d_ortho")
fig3.savefig(out3 + ".pdf")
fig3.savefig(out3 + ".png", dpi=300)
plt.close(fig3)
print(f"  Saved {out3}.pdf / .png")

print("\nDone.")
