"""
Image quality and faithfulness metrics.

image_metrics(pred, ref)  : PSNR, SSIM (scikit-image, reference-based).
leakage_metric(lq, pred)  : no-reference structural-leakage score.
residual_stats(lq, pred)  : mean, std, and gradient-magnitude of residual.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import generic_gradient_magnitude, sobel


def image_metrics(pred: np.ndarray, ref: np.ndarray,
                  data_range: float = 1.0) -> dict:
    """PSNR and SSIM between pred and ref, both float32 in [0, 1]."""
    psnr = peak_signal_noise_ratio(ref, pred, data_range=data_range)
    ssim = structural_similarity(ref, pred, data_range=data_range)
    return {"psnr": float(psnr), "ssim": float(ssim)}


def leakage_metric(lq: np.ndarray, pred: np.ndarray) -> dict:
    """
    No-reference structural-leakage score.

    residual  r = lq - pred
    leakage   = |Pearson(r, pred)|

    If the denoiser removes real structure, r and pred co-vary.
    A faithful denoiser should give leakage ≈ 0.

    Also returns:
      gradient_ratio : mean |∇r| / mean |∇pred|
        A high ratio means the residual has as much edge content as the output,
        indicating structure removal (hallucination in reverse — erasure).
    """
    r = lq - pred

    r_flat  = r.ravel()
    p_flat  = pred.ravel()

    corr = float(np.corrcoef(r_flat, p_flat)[0, 1])

    # gradient magnitudes
    grad_r    = generic_gradient_magnitude(r,    sobel)
    grad_pred = generic_gradient_magnitude(pred, sobel)

    mean_grad_pred = float(grad_pred.mean())
    grad_ratio = float(grad_r.mean()) / (mean_grad_pred + 1e-10)

    return {
        "leakage_corr":  abs(corr),
        "grad_ratio":    grad_ratio,
        "residual_mean": float(r.mean()),
        "residual_std":  float(r.std()),
    }


def residual_stats(lq: np.ndarray, pred: np.ndarray) -> dict:
    """Raw residual statistics (convenience wrapper)."""
    r = lq - pred
    return {
        "mean":  float(r.mean()),
        "std":   float(r.std()),
        "min":   float(r.min()),
        "max":   float(r.max()),
    }
