"""
Evaluation script — compare all denoisers on image quality + physics.

Usage
-----
py -3 evaluate.py --dncnn runs/dncnn_*/best.pt --unet runs/unet_*/best.pt \
                  --unet_leakage runs/unet_leakage_*/best.pt

What it does
------------
1. Loads the test split (15 slices, full 1024×1024).
2. Runs each denoiser (classical + neural).
3. Computes image metrics (PSNR, SSIM) and faithfulness metrics (leakage_corr, grad_ratio).
4. Derives two global Otsu thresholds (grain / oil) from the clean HQ test slices.
5. Computes per-slice physics (oil_saturation, euler_number, n_blobs) for:
   - HQ reference
   - LQ (noisy baseline, no denoising)
   - each denoiser output
6. Saves results/metrics.csv and results/physics.csv.
7. Generates summary plots in results/.
"""

import argparse, os, glob, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data        import CTDenoisePairDataset, load_slice, HQ_DIR, LQ_DIR
from src.models      import DnCNN, UNet, UNetLeakage, GaussDenoiser, NLMDenoiser
from src.metrics     import image_metrics, leakage_metric
from src.petrophysics import (threshold_stack, two_phase_labels, oil_saturation,
                               euler_connectivity, blob_size_distribution,
                               slice_physics)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────
def load_neural(model_cls, ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = model_cls().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_neural(model, lq_np: np.ndarray) -> np.ndarray:
    """Run a GPU model on a full (H,W) slice. Returns (H,W) float32."""
    t = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    return out.squeeze().cpu().numpy()


def load_test_slices():
    """Returns lists of (hq, lq) numpy arrays for test split (indices 85-99)."""
    import glob as _glob
    hq_paths = sorted(_glob.glob(os.path.join(HQ_DIR, "*.tif")))[85:100]
    lq_paths = sorted(_glob.glob(os.path.join(LQ_DIR, "*.tif")))[85:100]
    hqs = [load_slice(p) for p in hq_paths]
    lqs = [load_slice(p) for p in lq_paths]
    return hqs, lqs


def compute_thresholds(hqs: list) -> tuple[float, float]:
    """
    Compute grain and oil Otsu thresholds from the HQ stack.
    grain_thr > oil_thr: stack → Otsu on full stack for grain,
    then Otsu on pore-only pixels for oil.
    """
    stack = np.stack(hqs, axis=0)
    flat  = stack.ravel()

    # grain threshold: Otsu on all pixels
    from skimage.filters import threshold_otsu
    grain_thr = float(threshold_otsu(flat))

    # oil threshold: Otsu only on sub-grain pixels
    pore_pixels = flat[flat < grain_thr]
    oil_thr     = float(threshold_otsu(pore_pixels)) if len(pore_pixels) > 0 else grain_thr * 0.5

    print(f"Thresholds — grain: {grain_thr:.4f}  oil: {oil_thr:.4f}")
    return grain_thr, oil_thr


# ── evaluation loop ───────────────────────────────────────────────────────────
def evaluate_all(denoisers: dict, hqs, lqs, grain_thr, oil_thr):
    """
    denoisers: {name: callable(lq_np) -> pred_np}
    Returns (img_records, phys_records).
    """
    img_rows, phys_rows = [], []

    for i, (hq, lq) in enumerate(zip(hqs, lqs)):
        slice_id = 85 + i

        # LQ baseline physics (no denoising)
        lq_phys = slice_physics(lq, grain_thr=grain_thr, oil_thr=oil_thr)
        hq_phys = slice_physics(hq, grain_thr=grain_thr, oil_thr=oil_thr)

        for name, fn in denoisers.items():
            pred = fn(lq)

            # image metrics (needs reference)
            im = image_metrics(pred, hq)
            lk = leakage_metric(lq, pred)
            img_rows.append({
                "slice": slice_id, "denoiser": name,
                "psnr": im["psnr"], "ssim": im["ssim"],
                "leakage_corr": lk["leakage_corr"],
                "grad_ratio":   lk["grad_ratio"],
                "residual_std": lk["residual_std"],
            })

            # physics
            ph = slice_physics(pred, grain_thr=grain_thr, oil_thr=oil_thr)
            phys_rows.append({
                "slice": slice_id, "denoiser": name,
                "oil_sat":   ph["oil_saturation"],
                "euler":     ph["euler_number"],
                "n_blobs":   ph["n_blobs"],
            })

        # HQ reference and LQ baseline
        for tag, phys in [("HQ_reference", hq_phys), ("LQ_noisy", lq_phys)]:
            phys_rows.append({
                "slice": slice_id, "denoiser": tag,
                "oil_sat":   phys["oil_saturation"],
                "euler":     phys["euler_number"],
                "n_blobs":   phys["n_blobs"],
            })

        print(f"  Slice {slice_id} done")

    return img_rows, phys_rows


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_results(img_df, phys_df):
    denoiser_order = [d for d in img_df["denoiser"].unique()]
    colors = plt.cm.tab10(np.linspace(0, 1, len(denoiser_order)))

    # PSNR / SSIM bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(axes, ["psnr", "ssim"], ["PSNR (dB)", "SSIM"]):
        means = img_df.groupby("denoiser")[metric].mean().reindex(denoiser_order)
        stds  = img_df.groupby("denoiser")[metric].std().reindex(denoiser_order)
        ax.bar(denoiser_order, means, yerr=stds, capsize=4, color=colors)
        ax.set_title(label); ax.set_ylabel(label)
        ax.set_xticklabels(denoiser_order, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "psnr_ssim.png"), dpi=150)
    plt.close()

    # Leakage
    fig, ax = plt.subplots(figsize=(8, 4))
    means = img_df.groupby("denoiser")["leakage_corr"].mean().reindex(denoiser_order)
    ax.bar(denoiser_order, means, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Structural-Leakage Correlation (lower = more faithful)")
    ax.set_ylabel("|corr(residual, output)|")
    ax.set_xticklabels(denoiser_order, rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "leakage.png"), dpi=150)
    plt.close()

    # Oil saturation deviation from HQ
    hq_oil  = phys_df[phys_df["denoiser"] == "HQ_reference"].set_index("slice")["oil_sat"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in denoiser_order:
        sub = phys_df[phys_df["denoiser"] == name].set_index("slice")
        dev = sub["oil_sat"] - hq_oil
        ax.plot(dev.index, dev.values, label=name, marker="o", ms=4)
    ax.axhline(0, color="k", lw=0.8, ls="--", label="HQ reference")
    ax.set_title("Oil Saturation Deviation from HQ Reference")
    ax.set_xlabel("Slice index"); ax.set_ylabel("ΔS_o")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "oil_saturation_dev.png"), dpi=150)
    plt.close()

    print("Plots saved to results/")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dncnn",         default=None)
    p.add_argument("--unet",          default=None)
    p.add_argument("--unet_leakage",  default=None)
    p.add_argument("--gauss_sigma",   type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading test slices...")
    hqs, lqs = load_test_slices()
    print(f"  {len(hqs)} slices loaded")

    print("Computing segmentation thresholds from HQ...")
    grain_thr, oil_thr = compute_thresholds(hqs)

    # ── build denoiser dict ───────────────────────────────────────────────────
    denoisers = {}

    # Classical
    gauss = GaussDenoiser(sigma=args.gauss_sigma)
    denoisers["Gaussian"] = gauss

    nlm = NLMDenoiser()
    denoisers["NLM"] = nlm

    # Neural
    if args.dncnn:
        ck = sorted(glob.glob(args.dncnn))[-1] if "*" in args.dncnn else args.dncnn
        m  = load_neural(DnCNN, ck)
        denoisers["DnCNN"] = lambda lq, _m=m: run_neural(_m, lq)
        print(f"  DnCNN  loaded from {ck}")

    if args.unet:
        ck = sorted(glob.glob(args.unet))[-1] if "*" in args.unet else args.unet
        m  = load_neural(UNet, ck)
        denoisers["UNet"] = lambda lq, _m=m: run_neural(_m, lq)
        print(f"  UNet   loaded from {ck}")

    if args.unet_leakage:
        ck = sorted(glob.glob(args.unet_leakage))[-1] if "*" in args.unet_leakage else args.unet_leakage
        m  = load_neural(UNetLeakage, ck)
        denoisers["UNet+Leakage"] = lambda lq, _m=m: run_neural(_m, lq)
        print(f"  UNet+Leakage loaded from {ck}")

    print(f"\nRunning evaluation over {len(hqs)} test slices × {len(denoisers)} denoisers...")
    img_rows, phys_rows = evaluate_all(denoisers, hqs, lqs, grain_thr, oil_thr)

    img_df  = pd.DataFrame(img_rows)
    phys_df = pd.DataFrame(phys_rows)

    img_df.to_csv( os.path.join(RESULTS_DIR, "metrics.csv"),  index=False)
    phys_df.to_csv(os.path.join(RESULTS_DIR, "physics.csv"),  index=False)
    print("\nSaved results/metrics.csv and results/physics.csv")

    print("\n=== Image Quality Summary (mean ± std) ===")
    summary = img_df.groupby("denoiser")[["psnr","ssim","leakage_corr","grad_ratio"]].agg(["mean","std"])
    print(summary.to_string())

    print("\n=== Physics Summary (mean ± std) ===")
    psummary = phys_df.groupby("denoiser")[["oil_sat","euler","n_blobs"]].agg(["mean","std"])
    print(psummary.to_string())

    plot_results(img_df, phys_df)


if __name__ == "__main__":
    main()
