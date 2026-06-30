# Petrophysical Validation of Structural Leakage in Deep Denoising of Micro-CT Rock-Core Images

**Abu Bakker Siddique**  
Department of Petroleum and Mining Engineering, Shahjalal University of Science and Technology, Sylhet, Bangladesh

> Preprint submitted to *Computers & Geosciences* — 2026

---

## Overview

This repository contains the full code, pre-trained model weights, quantitative results, and publication figures for the paper:

> *Petrophysical Validation of Structural Leakage in Deep Denoising of Micro-CT Rock-Core Images*

We propose **Structural Leakage Regularization**, a no-reference faithfulness diagnostic that quantifies pore-structure corruption in deep CT denoising — crucial when high-quality reference scans are unavailable (e.g., dynamic time-series acquisitions).

### Key contributions

- **Leakage correlation metric** |Pearson(r, denoised)| where r = LQ − denoised; zero = faithful, ~0.7 = over-smoothed
- **LeakageLoss** = L1 + 0.5·(1−SSIM) + 0.1·|leakage correlation|
- Petrophysical validation via oil saturation, Euler number, and ganglion topology on segmented pore space
- Evaluation on 63 usable slices of Ketton oolitic carbonate (Zenodo dataset 17856861)

---

## Dataset

The dataset is from:

> Ma et al. (2026). *Pore-scale dynamics of multiphase reactive transport in water-wet carbonates under CO₂-acidified brine injection*. *Advances in Water Resources*. doi:[10.5281/zenodo.17856861](https://doi.org/10.5281/zenodo.17856861)

Download and place as:
```
shard_data/
  initialOilHigh/   ← HQ scans (InitialOilHigh0000–0099.tif)
  initialQilQuick/  ← LQ scans
```

> **Note:** Slices 0–36 in the HQ folder are corrupt (header-only, ~2882 bytes). Slices 37–99 are the 63 usable 1024×1024 slices used in this work.

---

## Repository structure

```
cdu-github/
├── paper.pdf                  ← final manuscript
├── paper.tex / paper.bib      ← LaTeX source
├── train.py                   ← model training
├── evaluate.py                ← quantitative evaluation
├── make_figures.py            ← paper figures (Fig 1–4, 6–12)
├── make_visual_figures.py     ← visual comparison + 3D ortho figures
├── run_all.ps1                ← Windows one-shot: train → evaluate
├── src/
│   ├── models.py              ← DnCNN, UNet, UNetLeakage
│   ├── losses.py              ← LeakageLoss, leakage_penalty
│   ├── data.py                ← dataset loader, train/val/test splits
│   ├── metrics.py             ← PSNR, SSIM, leakage corr, grad ratio
│   └── petrophysics.py        ← oil saturation, Euler number, ganglion count
├── checkpoints/
│   ├── dncnn_best.pt          ← DnCNN best val checkpoint (epoch 41, loss 0.0701)
│   ├── unet_best.pt           ← UNet best val checkpoint (epoch 41, loss 0.0660)
│   └── unet_leakage_best.pt   ← UNet+Leakage best val checkpoint (epoch 47, loss 0.1083)
├── results/
│   ├── metrics.csv            ← per-slice PSNR/SSIM/leakage/grad_ratio/residual_std
│   ├── physics.csv            ← per-slice oil_sat/Euler/n_blobs
│   └── *.png                  ← summary result plots
└── figures/                   ← all publication figures (PDF + PNG)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU strongly recommended (NVIDIA; tested on RTX 5070).

### 2. Reproduce training

```bash
# Train all three models (DnCNN, UNet, UNet+Leakage) — ~3 h on RTX 5070
python train.py --model dncnn        --epochs 50 --batch 32
python train.py --model unet         --epochs 50 --batch 32
python train.py --model unet_leakage --epochs 50 --batch 32 --beta 0.1

# Or all at once (Windows PowerShell):
powershell -File run_all.ps1
```

Checkpoints are saved to `runs/<model>_<timestamp>/best.pt`.

### 3. Evaluate with pre-trained weights

```bash
python evaluate.py \
  --dncnn        checkpoints/dncnn_best.pt \
  --unet         checkpoints/unet_best.pt \
  --unet_leakage checkpoints/unet_leakage_best.pt
```

Outputs `results/metrics.csv` and `results/physics.csv`.

### 4. Reproduce figures

```bash
python make_figures.py           # Fig 1–4, 6–12
python make_visual_figures.py    # Fig 5 (visual comparison) + 3D ortho
```

---

## Results summary (test slices 85–99)

| Method | PSNR (dB) | SSIM | Leakage ↓ | Oil sat err |
|---|---|---|---|---|
| LQ (noisy) | — | — | — | — |
| Gaussian | baseline | baseline | — | — |
| NLM | baseline | baseline | — | — |
| DnCNN | — | — | — | — |
| U-Net | — | — | — | — |
| **U-Net+Leakage** | — | — | **lowest** | **lowest** |

See `results/metrics.csv` and `results/physics.csv` for full per-slice numbers.

---

## Models

| Model | Params | Architecture | Loss |
|---|---|---|---|
| DnCNN | 556K | 17-layer residual CNN | L1 |
| U-Net | 7.76M | 4-level encoder-decoder, skip connections | L1 + 0.5·(1−SSIM) |
| U-Net+Leakage | 7.76M | Same as U-Net | L1 + 0.5·(1−SSIM) + **0.1·leakage** |

Checkpoint format: `{"epoch": int, "model": state_dict, "val_loss": float, "args": Namespace}`

---

## Citation

```bibtex
@article{siddique2026leakage,
  title   = {Petrophysical Validation of Structural Leakage in Deep Denoising
             of Micro-CT Rock-Core Images},
  author  = {Siddique, Abu Bakker},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code released under the **MIT License**. The pre-trained checkpoints and figures are provided for academic reproducibility only. The underlying CT dataset is subject to the Zenodo dataset license (CC BY 4.0).
