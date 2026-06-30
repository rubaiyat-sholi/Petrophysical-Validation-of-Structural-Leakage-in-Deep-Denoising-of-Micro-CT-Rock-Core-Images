"""
Publication-quality figures for:
  "Structural Leakage Regularization for Petrophysics-Faithful Deep Denoising
   of Micro-CT Rock-Core Images"
Target: Computers & Geosciences (Elsevier)

Figures produced
----------------
fig1_training_curves.pdf   -- validation-loss curves (all 3 DL models)
fig2_image_quality.pdf     -- PSNR, SSIM, leakage, gradient-ratio panel
fig3_petrophysics.pdf      -- oil saturation, Euler number, ganglion count panel
fig4_leakage_vs_euler.pdf  -- scatter: structural leakage vs Euler number
fig5_qualitative_zoom.pdf  -- HQ/LQ/denoised slice comparison with zooms
fig6_gray_histograms.pdf   -- gray-value distributions for representative slice
fig7_slice_trends.pdf      -- per-slice metric and petrophysical trends
fig8_full_workflow.pdf     -- end-to-end experimental workflow
fig9_architecture_workflow.pdf -- model and loss architecture workflow
fig10_segmentation_comparison.pdf -- CT-DiffNet-style segmentation comparison
fig11_ganglion_distribution.pdf -- oil-ganglion size distributions
fig12_metric_distributions.pdf -- PSNR/SSIM distributions across test slices
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from src.data import HQ_DIR, LQ_DIR, load_slice
from src.models import DnCNN, GaussDenoiser, NLMDenoiser, UNet, UNetLeakage
from src.petrophysics import two_phase_labels

# ── Output directory ─────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global aesthetics ─────────────────────────────────────────────────────────
# Use Okabe-Ito colorblind-safe palette throughout
OI = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "gray":    "#888888",
}

METHOD_COLORS = {
    "Gaussian":      OI["blue"],
    "NLM":           OI["skyblue"],
    "DnCNN":         OI["orange"],
    "U-Net":         OI["green"],
    "U-Net+Leakage": OI["vermil"],
    "HQ reference":  OI["black"],
    "LQ noisy":      OI["gray"],
}

METHOD_MARKERS = {
    "Gaussian":      "s",
    "NLM":           "^",
    "DnCNN":         "o",
    "U-Net":         "D",
    "U-Net+Leakage": "P",
    "HQ reference":  "*",
    "LQ noisy":      "X",
}

# Journal-ready rcParams (two-column Elsevier ~ 190 mm full width)
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "grid.color":         "0.88",
    "grid.linestyle":     "-",
    "axes.grid":          True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

# ── Data paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
RUNS = os.path.join(ROOT, "runs")

DNCNN_LOG    = os.path.join(RUNS, "dncnn_20260629_151815",      "train_log.csv")
UNET_LOG     = os.path.join(RUNS, "unet_20260629_163658",       "train_log.csv")
LEAKAGE_LOG  = os.path.join(RUNS, "unet_leakage_20260629_182602", "train_log.csv")
METRICS_CSV  = os.path.join(ROOT, "results", "metrics.csv")
PHYSICS_CSV  = os.path.join(ROOT, "results", "physics.csv")
DNCNN_CKPT   = os.path.join(RUNS, "dncnn_20260629_151815", "best.pt")
UNET_CKPT    = os.path.join(RUNS, "unet_20260629_163658", "best.pt")
LEAK_CKPT    = os.path.join(RUNS, "unet_leakage_20260629_182602", "best.pt")

# ── Load data ─────────────────────────────────────────────────────────────────
dncnn_log   = pd.read_csv(DNCNN_LOG)
unet_log    = pd.read_csv(UNET_LOG)
leakage_log = pd.read_csv(LEAKAGE_LOG)

metrics_raw = pd.read_csv(METRICS_CSV)
physics_raw = pd.read_csv(PHYSICS_CSV)

# ── Compute per-method summary stats from test slices ─────────────────────────
DENOISER_ORDER = ["Gaussian", "NLM", "DnCNN", "UNet", "UNet+Leakage"]
DISPLAY_NAMES  = {
    "Gaussian":    "Gaussian",
    "NLM":         "NLM",
    "DnCNN":       "DnCNN",
    "UNet":        "U-Net",
    "UNet+Leakage":"U-Net+Leakage",
}

def summarise(df, col):
    g = df.groupby("denoiser")[col]
    return g.mean(), g.std()


def _diagram_box(ax, xy, width, height, text, fc="white", ec="0.25",
                 fontsize=8, lw=1.1):
    box = mpatches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=fontsize, linespacing=1.2
    )
    return box


def _diagram_arrow(ax, start, end, color="0.25", lw=1.2, rad=0.0):
    arrow = mpatches.FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arrow)
    return arrow

# Image quality
psnr_mean, psnr_std     = summarise(metrics_raw, "psnr")
ssim_mean, ssim_std     = summarise(metrics_raw, "ssim")
leak_mean, leak_std     = summarise(metrics_raw, "leakage_corr")
grad_mean, grad_std     = summarise(metrics_raw, "grad_ratio")

# Physics – mean across 15 test slices
phys_mean = physics_raw.groupby("denoiser").mean()

# HQ reference and LQ noisy (already in physics_raw)
HQ_EULER    = phys_mean.loc["HQ_reference", "euler"]
HQ_OILSAT   = phys_mean.loc["HQ_reference", "oil_sat"]
HQ_BLOBS    = phys_mean.loc["HQ_reference", "n_blobs"]
LQ_EULER    = phys_mean.loc["LQ_noisy",     "euler"]
LQ_OILSAT   = phys_mean.loc["LQ_noisy",     "oil_sat"]
LQ_BLOBS    = phys_mean.loc["LQ_noisy",     "n_blobs"]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig1_training_curves():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    epochs = np.arange(1, 51)

    # --- DnCNN (smooth)
    c_dncnn = OI["orange"]
    ax.plot(epochs, dncnn_log["train_loss"], color=c_dncnn,
            lw=1.2, ls="--", alpha=0.55)
    ax.plot(epochs, dncnn_log["val_loss"], color=c_dncnn,
            lw=1.8, label="DnCNN (val)")

    # --- U-Net
    c_unet = OI["green"]
    ax.plot(epochs, unet_log["train_loss"], color=c_unet,
            lw=1.2, ls="--", alpha=0.55)
    ax.plot(epochs, unet_log["val_loss"], color=c_unet,
            lw=1.8, label="U-Net (val)")

    # --- U-Net+Leakage (total loss)
    c_leak = OI["vermil"]
    ax.plot(epochs, leakage_log["train_loss"], color=c_leak,
            lw=1.2, ls="--", alpha=0.55)
    ax.plot(epochs, leakage_log["val_loss"], color=c_leak,
            lw=1.8, label="U-Net+Leakage (val)")

    # --- Annotate collapse spikes on U-Net+Leakage
    collapse_mask = leakage_log["val_loss"] > 0.4
    collapse_epochs = epochs[collapse_mask.values]
    for ep in collapse_epochs:
        ax.axvline(ep, color=c_leak, lw=0.6, ls=":", alpha=0.5)

    # --- Best checkpoint markers
    best_dncnn = dncnn_log["val_loss"].idxmin()
    best_unet  = unet_log["val_loss"].idxmin()
    best_leak  = leakage_log["val_loss"].idxmin()
    ax.scatter(epochs[best_dncnn], dncnn_log["val_loss"].iloc[best_dncnn],
               s=55, color=c_dncnn, zorder=5, marker="*",
               edgecolors="k", linewidths=0.4)
    ax.scatter(epochs[best_unet], unet_log["val_loss"].iloc[best_unet],
               s=55, color=c_unet, zorder=5, marker="*",
               edgecolors="k", linewidths=0.4)
    ax.scatter(epochs[best_leak], leakage_log["val_loss"].iloc[best_leak],
               s=55, color=c_leak, zorder=5, marker="*",
               edgecolors="k", linewidths=0.4)

    # Clip y to show convergence clearly; note spikes in caption
    ax.set_ylim(0.04, 0.32)
    ax.set_xlim(1, 50)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Custom legend including train style
    leg_handles = [
        Line2D([0], [0], color=c_dncnn,  lw=1.8, label="DnCNN"),
        Line2D([0], [0], color=c_unet,   lw=1.8, label="U-Net"),
        Line2D([0], [0], color=c_leak,   lw=1.8, label="U-Net+Leakage"),
        Line2D([0], [0], color="k",      lw=1.2, ls="--", alpha=0.6, label="Train loss"),
        Line2D([0], [0], color="k",      lw=1.8,                    label="Val loss"),
        Line2D([0], [0], color="k",      lw=0,   marker="*", ms=6,
               markeredgecolor="k", markeredgewidth=0.4,           label="Best checkpoint"),
    ]
    ax.legend(handles=leg_handles, loc="upper right", ncol=2,
              handlelength=1.6, columnspacing=1.0)

    # Annotate collapse region
    ax.annotate("BN collapse\nevents",
                xy=(17, 0.285), xytext=(22, 0.30),
                fontsize=7, color=c_leak,
                arrowprops=dict(arrowstyle="-|>", color=c_leak, lw=0.8))

    ax.set_title("(a) Training and validation loss curves", loc="left", pad=4)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig1_training_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    out_png = os.path.join(FIGURES_DIR, "fig1_training_curves.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Image quality metrics (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_image_quality():
    methods_plot  = ["Gaussian", "NLM", "DnCNN", "UNet", "UNet+Leakage"]
    labels_plot   = [DISPLAY_NAMES[m] for m in methods_plot]
    colors_plot   = [METHOD_COLORS[DISPLAY_NAMES[m]] for m in methods_plot]

    psnr_m  = [psnr_mean[m] for m in methods_plot]
    psnr_e  = [psnr_std[m]  for m in methods_plot]
    ssim_m  = [ssim_mean[m] for m in methods_plot]
    ssim_e  = [ssim_std[m]  for m in methods_plot]
    leak_m  = [leak_mean[m] for m in methods_plot]
    leak_e  = [leak_std[m]  for m in methods_plot]
    grad_m  = [grad_mean[m] for m in methods_plot]
    grad_e  = [grad_std[m]  for m in methods_plot]

    x = np.arange(len(methods_plot))
    bar_w = 0.62

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax_psnr, ax_ssim, ax_leak, ax_grad = axes.flat

    kw_err = dict(ecolor="0.35", elinewidth=0.8, capsize=2.5, capthick=0.8)

    def _bar(ax, vals, errs, ylabel, title, ylim=None, hline=None, hline_lbl=None):
        bars = ax.bar(x, vals, bar_w, color=colors_plot,
                      yerr=errs, error_kw=kw_err,
                      edgecolor="white", linewidth=0.4, zorder=3)
        if hline is not None:
            ax.axhline(hline, color="0.3", lw=1.0, ls="--", zorder=2)
            if hline_lbl:
                ax.text(len(methods_plot) - 0.4, hline * 1.01, hline_lbl,
                        ha="right", va="bottom", fontsize=7, color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)
        if ylim:
            ax.set_ylim(ylim)
        # Value labels on bars
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + e + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    _bar(ax_psnr, psnr_m, psnr_e,
         "PSNR (dB)", "(a) Peak signal-to-noise ratio",
         ylim=(0, 38))

    _bar(ax_ssim, ssim_m, ssim_e,
         "SSIM", "(b) Structural similarity index",
         ylim=(0.40, 0.94))

    _bar(ax_leak, leak_m, leak_e,
         "Leakage correlation $|\\rho_{r,\\hat{x}}|$",
         "(c) Structural leakage",
         ylim=(0, 0.82))

    _bar(ax_grad, grad_m, grad_e,
         "Gradient ratio", "(d) Spatial gradient ratio",
         ylim=(0, 8.5),
         hline=1.0, hline_lbl="Input level")

    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    out = os.path.join(FIGURES_DIR, "fig2_image_quality.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Petrophysical metrics (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_petrophysics():
    """
    Horizontal bar chart for each of the three petrophysical scalars.
    Methods ordered from top to bottom: HQ ref, Gaussian, NLM, DnCNN,
    U-Net, U-Net+Leakage, LQ noisy.
    A vertical dashed line marks the HQ reference value.
    """
    methods_all = ["HQ_reference", "Gaussian", "NLM", "DnCNN",
                   "UNet", "UNet+Leakage", "LQ_noisy"]
    labels_all  = ["HQ reference", "Gaussian", "NLM", "DnCNN",
                   "U-Net", "U-Net+Leakage", "LQ noisy"]
    colors_all  = [METHOD_COLORS[l] for l in labels_all]

    oil_sat = [phys_mean.loc[m, "oil_sat"] for m in methods_all]
    euler   = [phys_mean.loc[m, "euler"]   for m in methods_all]
    n_blobs = [phys_mean.loc[m, "n_blobs"] for m in methods_all]

    y = np.arange(len(methods_all))
    bar_h = 0.62

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.6))
    ax_oil, ax_eul, ax_blb = axes

    def _hbar(ax, vals, ref_val, xlabel, title, xfmt="{:.3f}",
              log=False, xlim=None):
        ax.barh(y, vals, bar_h, color=colors_all,
                edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(ref_val, color=colors_all[0], lw=1.2, ls="--", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_all)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=3)
        if log:
            ax.set_xscale("log")
        if xlim:
            ax.set_xlim(xlim)
        # Value annotations inside/outside bars
        for i, v in enumerate(vals):
            offset = max(vals) * 0.02
            ax.text(v + offset, i, xfmt.format(v),
                    ha="left", va="center", fontsize=6.5)
        ax.invert_yaxis()

    _hbar(ax_oil, oil_sat, HQ_OILSAT,
          "Oil saturation $S_o$", "(a) Oil saturation",
          xfmt="{:.4f}", xlim=(0, 0.085))

    _hbar(ax_eul, euler, HQ_EULER,
          "Euler number $E$", "(b) Euler connectivity",
          xfmt="{:.0f}", xlim=(0, 4000))

    _hbar(ax_blb, n_blobs, HQ_BLOBS,
          "Ganglion count $N_{\\mathrm{blob}}$", "(c) Ganglion count",
          xfmt="{:.0f}", xlim=(0, 4200))

    # Shared legend for the reference line
    ref_patch = Line2D([0], [0], color=OI["black"], lw=1.2, ls="--",
                       label="HQ reference value")
    fig.legend(handles=[ref_patch], loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=1, fontsize=8,
               framealpha=0.9)

    fig.tight_layout(w_pad=2.5)
    out = os.path.join(FIGURES_DIR, "fig3_petrophysics.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Structural leakage vs Euler number
# ─────────────────────────────────────────────────────────────────────────────
def fig4_leakage_vs_euler():
    """
    Scatter plot: x = structural leakage correlation,
                  y = Euler connectivity number.
    Classical filters: square markers.
    Deep learning:     round markers.
    HQ reference:      star.
    LQ noisy:          X.
    """
    # Gather data: (name, leakage, euler, color, marker, ms, category)
    dl_methods = [
        ("DnCNN",          leak_mean["DnCNN"],         phys_mean.loc["DnCNN",     "euler"],
         OI["orange"], "o", 70),
        ("U-Net",          leak_mean["UNet"],           phys_mean.loc["UNet",      "euler"],
         OI["green"],  "D", 70),
        ("U-Net+Leakage",  leak_mean["UNet+Leakage"],   phys_mean.loc["UNet+Leakage","euler"],
         OI["vermil"], "P", 80),
    ]
    cl_methods = [
        ("Gaussian",  leak_mean["Gaussian"], phys_mean.loc["Gaussian","euler"],
         OI["blue"],    "s", 70),
        ("NLM",       leak_mean["NLM"],      phys_mean.loc["NLM",     "euler"],
         OI["skyblue"],"^", 70),
    ]
    specials = [
        ("HQ reference", None,               HQ_EULER,  OI["black"], "*", 120),
        ("LQ noisy",     None,               LQ_EULER,  OI["gray"],  "X", 70),
    ]

    # Leakage error bars for DL and classical
    dl_leak_err = [
        leak_std["DnCNN"],
        leak_std["UNet"],
        leak_std["UNet+Leakage"],
    ]
    cl_leak_err = [
        leak_std["Gaussian"],
        leak_std["NLM"],
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    # DL methods
    for (name, lk, eu, col, mk, ms), err in zip(dl_methods, dl_leak_err):
        ax.errorbar(lk, eu, xerr=err, fmt=mk, color=col,
                    ms=np.sqrt(ms), mew=0.8, mec="k",
                    elinewidth=0.9, capsize=2.5, capthick=0.8,
                    label=name, zorder=5)

    # Classical methods
    for (name, lk, eu, col, mk, ms), err in zip(cl_methods, cl_leak_err):
        ax.errorbar(lk, eu, xerr=err, fmt=mk, color=col,
                    ms=np.sqrt(ms), mew=0.8, mec="k",
                    elinewidth=0.9, capsize=2.5, capthick=0.8,
                    label=name, zorder=5)

    # Special markers (HQ ref has no leakage)
    ax.scatter(0, HQ_EULER,  marker="*", s=180, color=OI["black"],
               edgecolors="k", linewidths=0.5, zorder=6, label="HQ reference")
    ax.scatter(0, LQ_EULER,  marker="X", s=80,  color=OI["gray"],
               edgecolors="k", linewidths=0.5, zorder=6, label="LQ noisy")

    # Annotations
    offset_x = 0.006
    offset_y = 40
    annots = [
        ("Gaussian",       leak_mean["Gaussian"],        phys_mean.loc["Gaussian","euler"]),
        ("NLM",            leak_mean["NLM"],             phys_mean.loc["NLM","euler"]),
        ("DnCNN",          leak_mean["DnCNN"],           phys_mean.loc["DnCNN","euler"]),
        ("U-Net",          leak_mean["UNet"],            phys_mean.loc["UNet","euler"]),
        ("U-Net+Leakage",  leak_mean["UNet+Leakage"],    phys_mean.loc["UNet+Leakage","euler"]),
    ]
    nudge = {
        "Gaussian":      (offset_x,  offset_y),
        "NLM":           (offset_x,  offset_y),
        "DnCNN":         (offset_x, -80),
        "U-Net":         (offset_x,  offset_y),
        "U-Net+Leakage": (offset_x, -80),
    }
    for name, lk, eu in annots:
        dx, dy = nudge[name]
        ax.annotate(name, (lk, eu), xytext=(lk + dx, eu + dy),
                    fontsize=7, ha="left",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))

    ax.annotate("HQ ref.", (0, HQ_EULER),
                xytext=(0.025, HQ_EULER + 60), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    ax.annotate("LQ noisy", (0, LQ_EULER),
                xytext=(0.025, LQ_EULER - 200), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))

    # Shaded regions for family separation
    ax.axvspan(0, 0.20, alpha=0.06, color=OI["blue"],    label="__nolegend__")
    ax.axvspan(0.55, 0.80, alpha=0.06, color=OI["vermil"], label="__nolegend__")
    ax.text(0.09, 200, "Classical\nfilters", ha="center", fontsize=7,
            color=OI["blue"], style="italic")
    ax.text(0.69, 200, "Deep\nlearning", ha="center", fontsize=7,
            color=OI["vermil"], style="italic")

    ax.set_xlabel("Structural leakage $|\\rho_{r,\\hat{x}}|$")
    ax.set_ylabel("Euler connectivity number $E$")
    ax.set_xlim(-0.04, 0.82)
    ax.set_ylim(-100, 4000)
    ax.set_title("(a) Structural leakage vs Euler connectivity", loc="left", pad=3)

    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter __nolegend__ entries
    hl = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    ax.legend([h for h, l in hl], [l for h, l in hl],
              loc="upper left", fontsize=7, handlelength=1.2,
              borderpad=0.6, labelspacing=0.4)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig4_leakage_vs_euler.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
def _load_checkpoint_model(model_cls, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = model_cls().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _run_model(model, device, img):
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return pred.squeeze().cpu().numpy()


def _representative_predictions(slice_id=92):
    hq_paths = sorted(
        os.path.join(HQ_DIR, name) for name in os.listdir(HQ_DIR)
        if name.lower().endswith(".tif")
    )
    lq_paths = sorted(
        os.path.join(LQ_DIR, name) for name in os.listdir(LQ_DIR)
        if name.lower().endswith(".tif")
    )

    hq = load_slice(hq_paths[slice_id])
    lq = load_slice(lq_paths[slice_id])

    dncnn, device = _load_checkpoint_model(DnCNN, DNCNN_CKPT)
    unet, _ = _load_checkpoint_model(UNet, UNET_CKPT)
    leak, _ = _load_checkpoint_model(UNetLeakage, LEAK_CKPT)

    return {
        "LQ input": lq,
        "HQ reference": hq,
        "Gaussian": GaussDenoiser(sigma=1.0)(lq),
        "NLM": NLMDenoiser()(lq),
        "DnCNN": _run_model(dncnn, device, lq),
        "U-Net": _run_model(unet, device, lq),
        "U-Net+Leakage": _run_model(leak, device, lq),
    }


def fig5_qualitative_zoom():
    predictions = _representative_predictions(slice_id=92)
    names = ["LQ input", "HQ reference", "NLM", "DnCNN", "U-Net", "U-Net+Leakage"]
    crop_y, crop_x, crop_size = 360, 420, 220

    all_pixels = np.concatenate([predictions[n].ravel() for n in names])
    vmin, vmax = np.percentile(all_pixels, [0.5, 99.5])

    fig, axes = plt.subplots(2, len(names), figsize=(7.5, 3.2))
    for col, name in enumerate(names):
        img = predictions[name]
        axes[0, col].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].add_patch(
            mpatches.Rectangle(
                (crop_x, crop_y), crop_size, crop_size,
                fill=False, edgecolor=OI["yellow"], linewidth=1.1
            )
        )
        axes[0, col].set_title(name, fontsize=7, pad=2)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        crop = img[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        axes[1, col].imshow(crop, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Full slice", fontsize=8)
    axes[1, 0].set_ylabel("Zoom", fontsize=8)
    fig.suptitle("Representative test slice and local pore-scale zoom", y=1.02, fontsize=9)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)

    out = os.path.join(FIGURES_DIR, "fig5_qualitative_zoom.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig6_gray_histograms():
    predictions = _representative_predictions(slice_id=92)
    names = ["LQ input", "HQ reference", "NLM", "DnCNN", "U-Net", "U-Net+Leakage"]
    colors = {
        "LQ input": OI["gray"],
        "HQ reference": OI["black"],
        "NLM": OI["skyblue"],
        "DnCNN": OI["orange"],
        "U-Net": OI["green"],
        "U-Net+Leakage": OI["vermil"],
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    sample_mask = (predictions["HQ reference"] > 0.02) & (predictions["LQ input"] > 0.02)
    for name in names:
        vals = predictions[name][sample_mask].ravel()
        vals = vals[(vals > 0.02) & (vals < 0.98)]
        ax.hist(vals, bins=180, range=(0, 1), density=True,
                histtype="step", linewidth=1.3, color=colors[name],
                label=name)

    ax.set_xlabel("Normalized gray value")
    ax.set_ylabel("Density")
    ax.set_title("(a) Gray-value distributions inside the rock core", loc="left", pad=3)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xlim(0, 1)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "fig6_gray_histograms.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig7_slice_trends():
    method_order = ["Gaussian", "NLM", "DnCNN", "UNet", "UNet+Leakage"]
    display = [DISPLAY_NAMES[m] for m in method_order]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharex=True)
    ax_psnr, ax_ssim, ax_leak, ax_euler = axes.flat

    for method, label in zip(method_order, display):
        img = metrics_raw[metrics_raw["denoiser"] == method].sort_values("slice")
        phys = physics_raw[physics_raw["denoiser"] == method].sort_values("slice")
        color = METHOD_COLORS[label]
        ax_psnr.plot(img["slice"], img["psnr"], marker="o", ms=3, color=color, label=label)
        ax_ssim.plot(img["slice"], img["ssim"], marker="o", ms=3, color=color)
        ax_leak.plot(img["slice"], img["leakage_corr"], marker="o", ms=3, color=color)
        ax_euler.plot(phys["slice"], phys["euler"], marker="o", ms=3, color=color)

    hq = physics_raw[physics_raw["denoiser"] == "HQ_reference"].sort_values("slice")
    lq = physics_raw[physics_raw["denoiser"] == "LQ_noisy"].sort_values("slice")
    ax_euler.plot(hq["slice"], hq["euler"], color=OI["black"], lw=1.2, ls="--",
                  label="HQ reference")
    ax_euler.plot(lq["slice"], lq["euler"], color=OI["gray"], lw=1.2, ls=":",
                  label="LQ noisy")

    ax_psnr.set_ylabel("PSNR (dB)")
    ax_ssim.set_ylabel("SSIM")
    ax_leak.set_ylabel("Leakage $|\\rho|$")
    ax_euler.set_ylabel("Euler number $E$")
    ax_leak.set_xlabel("Test slice index")
    ax_euler.set_xlabel("Test slice index")
    ax_psnr.set_title("(a) PSNR", loc="left", pad=3)
    ax_ssim.set_title("(b) SSIM", loc="left", pad=3)
    ax_leak.set_title("(c) Structural leakage", loc="left", pad=3)
    ax_euler.set_title("(d) Euler connectivity", loc="left", pad=3)
    ax_psnr.legend(loc="lower left", fontsize=6.5, ncol=2, framealpha=0.9)
    ax_euler.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlim(85, 99)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    out = os.path.join(FIGURES_DIR, "fig7_slice_trends.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def _segmentation_thresholds():
    hq_paths = sorted(
        os.path.join(HQ_DIR, name) for name in os.listdir(HQ_DIR)
        if name.lower().endswith(".tif")
    )[85:100]
    hqs = [load_slice(path) for path in hq_paths]
    flat = np.stack(hqs, axis=0).ravel()
    grain_thr = float(threshold_otsu(flat))
    pore_pixels = flat[flat < grain_thr]
    oil_thr = float(threshold_otsu(pore_pixels))
    return grain_thr, oil_thr


def _oil_radii(label_img):
    oil = label_img == 1
    labeled, _ = ndi_label(oil)
    radii = []
    for prop in regionprops(labeled):
        if prop.area >= 3:
            radii.append(np.sqrt(prop.area / np.pi))
    return np.asarray(radii, dtype=float)


def fig10_segmentation_comparison():
    predictions = _representative_predictions(slice_id=92)
    names = ["LQ input", "HQ reference", "NLM", "DnCNN", "U-Net", "U-Net+Leakage"]
    grain_thr, oil_thr = _segmentation_thresholds()
    crop_y, crop_x, crop_size = 330, 360, 300
    phase_cmap = ListedColormap(["#1F4E79", "#D95F02", "#D8D8D8"])

    fig, axes = plt.subplots(2, len(names), figsize=(7.5, 2.9))
    for col, name in enumerate(names):
        img = predictions[name]
        crop = img[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        labels = two_phase_labels(crop, grain_thr=grain_thr, oil_thr=oil_thr)

        axes[0, col].imshow(crop, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(name, fontsize=7, pad=2)
        axes[1, col].imshow(labels, cmap=phase_cmap, vmin=0, vmax=2,
                            interpolation="nearest")
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("Intensity", fontsize=8)
    axes[1, 0].set_ylabel("Segmentation", fontsize=8)
    legend_items = [
        mpatches.Patch(color="#1F4E79", label="brine/pore"),
        mpatches.Patch(color="#D95F02", label="oil"),
        mpatches.Patch(color="#D8D8D8", label="grain"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Representative slice segmentation from denoised outputs",
                 y=1.03, fontsize=9)
    fig.tight_layout(w_pad=0.45, h_pad=0.3)

    out = os.path.join(FIGURES_DIR, "fig10_segmentation_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig11_ganglion_distribution():
    predictions = _representative_predictions(slice_id=92)
    names = ["HQ reference", "LQ input", "NLM", "DnCNN", "U-Net", "U-Net+Leakage"]
    grain_thr, oil_thr = _segmentation_thresholds()

    radii = []
    labels = []
    colors = []
    for name in names:
        lbl = two_phase_labels(predictions[name], grain_thr=grain_thr, oil_thr=oil_thr)
        vals = _oil_radii(lbl)
        vals = vals[(vals >= 1.0) & (vals <= 30.0)]
        radii.append(vals)
        labels.append(name)
        colors.append(METHOD_COLORS.get(name, OI["gray"]))

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    parts = ax.violinplot(radii, showmeans=False, showmedians=True,
                          showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("0.25")
        body.set_alpha(0.45)
    parts["cmedians"].set_color("0.15")
    parts["cmedians"].set_linewidth(1.2)

    for idx, vals in enumerate(radii, start=1):
        if len(vals) == 0:
            continue
        rng = np.random.default_rng(1200 + idx)
        sample = rng.choice(vals, size=min(250, len(vals)), replace=False)
        jitter = rng.normal(0, 0.035, size=len(sample))
        ax.scatter(np.full(len(sample), idx) + jitter, sample,
                   s=4, alpha=0.25, color=colors[idx - 1], linewidths=0)
        ax.text(idx, 31.5, f"n={len(vals)}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Oil-ganglion equivalent radius (pixels)")
    ax.set_ylim(0, 34)
    ax.set_title("(a) Oil-ganglion size distributions for representative slice 92",
                 loc="left", pad=3)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "fig11_ganglion_distribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig12_metric_distributions():
    method_order = ["Gaussian", "NLM", "DnCNN", "UNet", "UNet+Leakage"]
    labels = [DISPLAY_NAMES[m] for m in method_order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, metric, ylabel, title in [
        (axes[0], "psnr", "PSNR (dB)", "(a) PSNR across held-out slices"),
        (axes[1], "ssim", "SSIM", "(b) SSIM across held-out slices"),
    ]:
        data = [
            metrics_raw[metrics_raw["denoiser"] == method][metric].to_numpy()
            for method in method_order
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        showfliers=False, medianprops={"color": "0.1", "lw": 1.1})
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(METHOD_COLORS[label])
            patch.set_alpha(0.45)
            patch.set_edgecolor("0.25")
        for item in bp["whiskers"] + bp["caps"]:
            item.set_color("0.25")
        for i, vals in enumerate(data, start=1):
            ax.scatter(np.full(len(vals), i), vals, s=12, color="0.15",
                       alpha=0.55, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=3)

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(FIGURES_DIR, "fig12_metric_distributions.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig8_full_workflow():
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    lq = load_slice(os.path.join(LQ_DIR, "InitialOilQuick0092.tif"))
    hq = load_slice(os.path.join(HQ_DIR, "InitialOilHigh0092.tif"))
    crop = np.s_[280:620, 330:670]

    def inset_image(bounds, img, title):
        iax = ax.inset_axes(bounds)
        iax.imshow(img[crop], cmap="gray", vmin=0, vmax=1)
        iax.set_xticks([])
        iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.25")
        ax.text(bounds[0] + bounds[2] / 2, bounds[1] - 0.025, title,
                ha="center", va="top", fontsize=7.5)

    ax.text(0.03, 0.96, "(a) Data preparation and training",
            fontsize=10, fontweight="bold", ha="left", va="center")
    ax.text(0.03, 0.48, "(b) Held-out testing and petrophysical validation",
            fontsize=10, fontweight="bold", ha="left", va="center")

    inset_image([0.04, 0.69, 0.095, 0.18], lq, "LQ slice")
    inset_image([0.105, 0.72, 0.095, 0.18], hq, "HQ slice")
    ax.text(0.12, 0.63, "paired 16-bit\nmicro-CT images", ha="center",
            va="top", fontsize=7.5)

    top_boxes = [
        (0.25, 0.70, 0.14, 0.14, "Normalize\n[0, 1]"),
        (0.45, 0.70, 0.14, 0.14, "Extract\n128 x 128\npatches"),
        (0.65, 0.70, 0.14, 0.14, "Train/val split\nslices 1-85"),
        (0.83, 0.70, 0.13, 0.14, "Best model\ncheckpoint"),
    ]
    for box in top_boxes:
        _diagram_box(ax, box[:2], box[2], box[3], box[4], fc="#F7F7F7",
                     ec="0.25", fontsize=7.8, lw=0.9)
    for start, end in [((0.20, 0.77), (0.25, 0.77)),
                       ((0.39, 0.77), (0.45, 0.77)),
                       ((0.59, 0.77), (0.65, 0.77)),
                       ((0.79, 0.77), (0.83, 0.77))]:
        _diagram_arrow(ax, start, end, lw=1.0)

    _diagram_box(ax, (0.04, 0.23), 0.16, 0.13, "Held-out test\nslices 86-100",
                 fc="#EEF5FB", ec="0.25", fontsize=8, lw=0.9)
    _diagram_box(ax, (0.27, 0.23), 0.18, 0.13,
                 "Denoisers\nGaussian, NLM\nDnCNN, U-Net\nU-Net+Leakage",
                 fc="#F7F7F7", ec="0.25", fontsize=7.3, lw=0.9)
    _diagram_box(ax, (0.52, 0.29), 0.18, 0.12,
                 "Image fidelity\nPSNR, SSIM\nleakage, gradients",
                 fc="#F8F0E5", ec="0.25", fontsize=7.5, lw=0.9)
    _diagram_box(ax, (0.52, 0.12), 0.18, 0.12,
                 "Rock physics\nsaturation, Euler\nconnected ganglia",
                 fc="#F8F0E5", ec="0.25", fontsize=7.5, lw=0.9)
    _diagram_box(ax, (0.78, 0.20), 0.18, 0.16,
                 "Rank denoisers by\npixel accuracy and\nstructural fidelity",
                 fc="#EEF7EC", ec="0.25", fontsize=7.8, lw=0.9)

    for start, end in [((0.20, 0.295), (0.27, 0.295)),
                       ((0.45, 0.295), (0.52, 0.35)),
                       ((0.45, 0.295), (0.52, 0.18)),
                       ((0.70, 0.35), (0.78, 0.30)),
                       ((0.70, 0.18), (0.78, 0.26))]:
        _diagram_arrow(ax, start, end, lw=1.0)

    ax.plot([0.03, 0.97], [0.55, 0.55], color="0.82", lw=0.8)

    out = os.path.join(FIGURES_DIR, "fig8_full_workflow.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def fig9_architecture_workflow():
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def fmap(x, y, w, h, color, label, channels=None):
        for off in [0.012, 0.006, 0]:
            rect = mpatches.Rectangle((x + off, y + off), w, h, linewidth=0.8,
                                      edgecolor="0.25", facecolor=color)
            ax.add_patch(rect)
        ax.text(x + w / 2 + 0.006, y - 0.035, label, ha="center",
                va="top", fontsize=7)
        if channels:
            ax.text(x + w / 2 + 0.006, y + h + 0.025, channels, ha="center",
                    va="bottom", fontsize=7)

    ax.text(0.03, 0.95, "(a) Residual CNN denoiser",
            fontsize=10, fontweight="bold", ha="left", va="center")
    ax.text(0.03, 0.54, "(b) U-Net and leakage-regularized objective",
            fontsize=10, fontweight="bold", ha="left", va="center")

    fmap(0.05, 0.74, 0.055, 0.10, "#DCECF5", "LQ", "1 ch")
    _diagram_box(ax, (0.18, 0.72), 0.22, 0.13,
                 "17-layer DnCNN\nConv + BN + ReLU\nresidual noise estimate",
                 fc="#F7F7F7", ec="0.25", fontsize=7.4, lw=0.9)
    fmap(0.47, 0.74, 0.055, 0.10, "#F8E4D8", "r", "1 ch")
    ax.text(0.62, 0.79, r"$\hat{x}=\mathrm{clip}(\mathrm{LQ}-r)$",
            ha="center", va="center", fontsize=9)
    fmap(0.78, 0.74, 0.055, 0.10, "#E8F2E5", "output", "1 ch")
    for start, end in [((0.12, 0.79), (0.18, 0.79)),
                       ((0.40, 0.79), (0.47, 0.79)),
                       ((0.53, 0.79), (0.55, 0.79)),
                       ((0.69, 0.79), (0.72, 0.79)),
                       ((0.72, 0.79), (0.78, 0.79))]:
        _diagram_arrow(ax, start, end, lw=1.0)

    x_positions = [0.06, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.88]
    heights = [0.16, 0.13, 0.105, 0.085, 0.105, 0.13, 0.16, 0.16]
    widths = [0.045, 0.05, 0.055, 0.06, 0.055, 0.05, 0.045, 0.04]
    labels = ["input", "enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2/1", "output"]
    channels = ["1", "32", "64", "128/256", "512", "256/128", "64/32", "1"]
    y_base = 0.30
    for x, h, w, lab, ch in zip(x_positions, heights, widths, labels, channels):
        fmap(x, y_base + (0.16 - h) / 2, w, h, "#E8F2E5", lab, ch)

    for i in range(len(x_positions) - 1):
        _diagram_arrow(
            ax,
            (x_positions[i] + widths[i] + 0.02, y_base + 0.08),
            (x_positions[i + 1] - 0.01, y_base + 0.08),
            lw=1.0
        )

    for left, right, yarc in [(0.20, 0.77, 0.455), (0.32, 0.66, 0.435), (0.44, 0.55, 0.415)]:
        _diagram_arrow(ax, (left, yarc), (right, yarc), color=OI["green"],
                       lw=1.0, rad=-0.08)
    ax.text(0.50, 0.462, "skip connections", ha="center", va="bottom",
            fontsize=7.3, color=OI["green"])

    _diagram_box(ax, (0.12, 0.07), 0.21, 0.11,
                 r"standard loss" "\n" r"$L_1 + 0.5(1-\mathrm{SSIM})$",
                 fc="#F8F0E5", ec="0.25", fontsize=7.6, lw=0.9)
    _diagram_box(ax, (0.40, 0.07), 0.25, 0.11,
                 r"leakage penalty" "\n"
                 r"$0.1|\rho(\mathrm{LQ}-\hat{x},\hat{x})|$",
                 fc="#F8F0E5", ec="0.25", fontsize=7.6, lw=0.9)
    _diagram_box(ax, (0.73, 0.07), 0.18, 0.11,
                 "U-Net+Leakage\nsame architecture\nmodified objective",
                 fc="#EFE7F1", ec="0.25", fontsize=7.2, lw=0.9)
    _diagram_arrow(ax, (0.33, 0.125), (0.40, 0.125), lw=1.0)
    _diagram_arrow(ax, (0.65, 0.125), (0.73, 0.125), lw=1.0)

    ax.plot([0.03, 0.97], [0.61, 0.61], color="0.82", lw=0.8)

    out = os.path.join(FIGURES_DIR, "fig9_architecture_workflow.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig8_full_workflow()
    fig9_architecture_workflow()
    fig1_training_curves()
    fig2_image_quality()
    fig3_petrophysics()
    fig4_leakage_vs_euler()
    fig5_qualitative_zoom()
    fig6_gray_histograms()
    fig7_slice_trends()
    fig10_segmentation_comparison()
    fig11_ganglion_distribution()
    fig12_metric_distributions()
    print("Done. All figures saved to ./figures/")
