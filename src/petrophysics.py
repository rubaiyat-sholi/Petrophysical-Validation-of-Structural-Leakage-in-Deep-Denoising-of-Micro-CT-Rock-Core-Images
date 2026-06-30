"""
Petrophysics pipeline: segmentation → physics metrics.

Metrics
-------
oil_saturation(seg)        : fraction of pore voxels occupied by oil.
euler_characteristic(seg)  : Euler number (connectivity proxy, Vogel 2002).
specific_interfacial_area  : oil-brine interface area per unit volume
                             (marching-cubes surface area / image volume).
blob_size_distribution     : oil-blob (ganglion) equivalent radii histogram.

Segmentation
------------
threshold_otsu_stack(imgs) : Otsu threshold on the stack median.
segment_slice(img, thr)    : hard threshold → binary (True = solid/oil).

Notes
-----
In the Initial-Oil state, the two-phase system is:
  brine (darker)  ↔  oil (brighter)
  solid grains   ↔  pore space (brine + oil)
We label: void/pore (low intensity) vs solid+oil (high intensity).
A second threshold separates oil from brine within the pore space.
"""

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import marching_cubes, mesh_surface_area, regionprops, euler_number


# ── segmentation ──────────────────────────────────────────────────────────────
def threshold_stack(slices: list[np.ndarray]) -> float:
    """Global Otsu threshold from the stack median image."""
    stack  = np.stack(slices, axis=0)
    median = np.median(stack, axis=0)
    return float(threshold_otsu(median))


def segment_slice(img: np.ndarray, thr: float,
                  smooth_sigma: float = 1.0) -> np.ndarray:
    """
    Return a binary mask: True = oil/solid, False = brine/pore.
    Applies light Gaussian smoothing before threshold.
    """
    smoothed = gaussian(img, sigma=smooth_sigma)
    return smoothed > thr


def two_phase_labels(img: np.ndarray,
                     grain_thr: float,
                     oil_thr: float) -> np.ndarray:
    """
    Assign pixel labels:
      0 = brine (pore, dark)
      1 = oil   (pore, intermediate)
      2 = grain  (solid, bright)
    grain_thr > oil_thr.
    """
    labels = np.zeros(img.shape, dtype=np.uint8)
    labels[img > oil_thr]   = 1   # oil
    labels[img > grain_thr] = 2   # grain (overwrite oil)
    return labels


# ── physics metrics ───────────────────────────────────────────────────────────
def oil_saturation(label_img: np.ndarray) -> float:
    """
    S_o = n_oil / (n_oil + n_brine)
    label_img: 0=brine, 1=oil, 2=grain
    """
    n_oil   = int((label_img == 1).sum())
    n_brine = int((label_img == 0).sum())
    pore    = n_oil + n_brine
    return n_oil / pore if pore > 0 else 0.0


def euler_connectivity(binary_oil: np.ndarray) -> float:
    """
    Euler characteristic of the oil phase (2D).
    Lower (more negative) → more connected / more loops.
    """
    return float(euler_number(binary_oil))


def specific_interfacial_area(label_3d: np.ndarray,
                               voxel_size: float = 1.0) -> float:
    """
    Oil–brine specific interfacial area (SIA) via marching cubes on the
    oil-phase binary volume.

    label_3d  : 3-D label array (0=brine, 1=oil, 2=grain), shape (Z, H, W).
    voxel_size: physical size of one voxel side (arbitrary units).
    Returns   : SIA = surface_area / total_volume  (units: 1/length).
    """
    oil_binary = (label_3d == 1).astype(np.float32)
    verts, faces, _, _ = marching_cubes(oil_binary, level=0.5,
                                         spacing=(voxel_size,)*3)
    area   = float(mesh_surface_area(verts, faces))
    volume = float(np.prod(np.array(label_3d.shape)) * voxel_size**3)
    return area / volume if volume > 0 else 0.0


def blob_size_distribution(label_2d: np.ndarray) -> dict:
    """
    Connected-component analysis of the oil phase in 2D.
    Returns dict with equivalent_radii list and summary stats.
    """
    oil_binary = (label_2d == 1)
    labeled, _  = label(oil_binary)
    props       = regionprops(labeled)

    radii = [np.sqrt(p.area / np.pi) for p in props if p.area > 0]
    radii = np.array(radii, dtype=np.float32)

    if len(radii) == 0:
        return {"mean_r": 0.0, "median_r": 0.0, "n_blobs": 0, "radii": []}

    return {
        "mean_r":   float(radii.mean()),
        "median_r": float(np.median(radii)),
        "std_r":    float(radii.std()),
        "n_blobs":  int(len(radii)),
        "radii":    radii.tolist(),
    }


# ── slice-level convenience ───────────────────────────────────────────────────
def slice_physics(img: np.ndarray, grain_thr: float, oil_thr: float) -> dict:
    """All 2-D physics metrics for a single slice."""
    lbl = two_phase_labels(img, grain_thr=grain_thr, oil_thr=oil_thr)
    so  = oil_saturation(lbl)
    eu  = euler_connectivity(lbl == 1)
    bd  = blob_size_distribution(lbl)
    return {
        "oil_saturation":    so,
        "euler_number":      eu,
        **{f"blob_{k}": v for k, v in bd.items() if k != "radii"},
        "n_blobs":           bd["n_blobs"],
    }
