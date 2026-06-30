"""
Loss functions for the denoising models.

Standard   : L1 + SSIM loss (pixel-fidelity).
Leakage    : Standard + structural-leakage penalty (faithfulness).

Structural-leakage penalty (Buades et al. 2005 "method noise" framing):
  residual   r = lq - denoised
  The residual should be structureless (zero correlation with the output).
  We penalise the Pearson correlation between r and denoised.
  This forces the model NOT to erase real structure from the output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SSIM (differentiable) ────────────────────────────────────────────────────
def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)


_SSIM_KERNEL = None


def _get_ssim_kernel(device):
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != device:
        _SSIM_KERNEL = _gaussian_kernel().to(device)
    return _SSIM_KERNEL


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """1 - mean SSIM (differentiable, single-channel)."""
    k = _get_ssim_kernel(pred.device)
    mu_x  = F.conv2d(pred,   k, padding=5, groups=1)
    mu_y  = F.conv2d(target, k, padding=5, groups=1)
    mu_xx = F.conv2d(pred**2,        k, padding=5)
    mu_yy = F.conv2d(target**2,      k, padding=5)
    mu_xy = F.conv2d(pred * target,  k, padding=5)

    sigma_x  = mu_xx - mu_x**2
    sigma_y  = mu_yy - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y

    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return 1.0 - (num / den).mean()


# ── Standard loss ────────────────────────────────────────────────────────────
class StandardLoss(nn.Module):
    """L1 + α·SSIM."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1   = F.l1_loss(pred, target)
        ssim = ssim_loss(pred, target)
        return l1 + self.alpha * ssim


# ── Structural-leakage penalty ────────────────────────────────────────────────
def leakage_penalty(lq: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation between residual r=(lq-denoised) and denoised,
    computed per image in the batch and averaged.

    A positive leakage means residual and output co-vary — i.e., the model
    is suppressing signal (not just noise). We penalise |corr|.

    Stability note: when the model output is near-constant (std_d ~ 0),
    the Pearson gradient blows up. We mask out those samples so the
    penalty only fires once the output has meaningful spatial variation.
    """
    r = lq - denoised
    # flatten spatial dims: (B, H*W)
    r_flat = r.view(r.size(0), -1)
    d_flat = denoised.view(denoised.size(0), -1)

    r_mean = r_flat.mean(dim=1, keepdim=True)
    d_mean = d_flat.mean(dim=1, keepdim=True)

    r_c = r_flat - r_mean
    d_c = d_flat - d_mean

    std_r  = r_c.std(dim=1)
    std_d  = d_c.std(dim=1)

    # Only compute penalty for samples where both signals have variance.
    # Detached mask: no gradient flows through the gating decision itself.
    valid = ((std_r > 1e-4) & (std_d > 1e-4)).detach().float()

    cov    = (r_c * d_c).mean(dim=1)
    corr   = cov / (std_r.clamp(min=1e-8) * std_d.clamp(min=1e-8))  # (B,)
    return (corr.abs() * valid).mean()


class LeakageLoss(nn.Module):
    """StandardLoss + β·leakage_penalty."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self._std  = StandardLoss(alpha=alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                lq: torch.Tensor) -> dict:
        std  = self._std(pred, target)
        leak = leakage_penalty(lq, pred)
        total = std + self.beta * leak
        return {"total": total, "std": std, "leakage": leak}
