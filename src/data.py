"""
TIFF dataset loader for paired clean/noisy CT slices.

Split strategy: slice-disjoint (train 0-69, val 70-84, test 85-99).
Patch extraction: 128×128 non-overlapping patches per slice (64 patches per 1024² slice).
Normalisation: map [0, 65535] to [0, 1] float32.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile

# ── constants ────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "shard_data")
HQ_DIR    = os.path.join(DATA_ROOT, "initialOilHigh")
LQ_DIR    = os.path.join(DATA_ROOT, "initialQilQuick")

PATCH_SIZE  = 128
NORM_SCALE  = 65535.0

SPLIT_TRAIN = (0,  70)   # slices [0, 70)
SPLIT_VAL   = (70, 85)   # slices [70, 85)
SPLIT_TEST  = (85, 100)  # slices [85, 100)

# ── helpers ───────────────────────────────────────────────────────────────────
def _sorted_tiffs(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"No TIFFs found in {directory}")
    return paths


def load_slice(path: str) -> np.ndarray:
    """Return a (H, W) float32 array normalised to [0, 1]."""
    img = tifffile.imread(path).astype(np.float32)
    return img / NORM_SCALE


def extract_patches(img: np.ndarray, patch_size: int = PATCH_SIZE):
    """Extract non-overlapping patches from a 2-D image. Returns (N, P, P)."""
    H, W = img.shape
    rows = H // patch_size
    cols = W // patch_size
    patches = []
    for r in range(rows):
        for c in range(cols):
            p = img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches.append(p)
    return np.stack(patches, axis=0)  # (N, P, P)


# ── dataset ───────────────────────────────────────────────────────────────────
class CTDenoisePairDataset(Dataset):
    """
    Yields (lq_patch, hq_patch) tensors of shape (1, P, P) in [0, 1].

    split : 'train' | 'val' | 'test'
    patch : if False, returns full slices instead of patches (for evaluation).
    """

    def __init__(self, split: str = "train", patch: bool = True,
                 patch_size: int = PATCH_SIZE):
        super().__init__()
        self.patch      = patch
        self.patch_size = patch_size

        hq_paths = _sorted_tiffs(HQ_DIR)
        lq_paths = _sorted_tiffs(LQ_DIR)

        assert len(hq_paths) == len(lq_paths), "HQ / LQ count mismatch"

        bounds = dict(train=SPLIT_TRAIN, val=SPLIT_VAL, test=SPLIT_TEST)[split]
        self.hq_paths = hq_paths[bounds[0]:bounds[1]]
        self.lq_paths = lq_paths[bounds[0]:bounds[1]]

        self._build_index()

    def _build_index(self):
        """Pre-compute slice→patch mapping so __len__ is O(1)."""
        if not self.patch:
            self._n = len(self.hq_paths)
            return

        # number of patches per slice (same for all 1024² images)
        tmp = load_slice(self.hq_paths[0])
        n_per_slice = (tmp.shape[0] // self.patch_size) * (tmp.shape[1] // self.patch_size)
        self._n_per_slice = n_per_slice
        self._n = len(self.hq_paths) * n_per_slice

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if not self.patch:
            hq = load_slice(self.hq_paths[idx])
            lq = load_slice(self.lq_paths[idx])
            return (
                torch.from_numpy(lq).unsqueeze(0),
                torch.from_numpy(hq).unsqueeze(0),
            )

        slice_idx  = idx // self._n_per_slice
        patch_idx  = idx %  self._n_per_slice

        hq = load_slice(self.hq_paths[slice_idx])
        lq = load_slice(self.lq_paths[slice_idx])

        hq_p = extract_patches(hq, self.patch_size)[patch_idx]
        lq_p = extract_patches(lq, self.patch_size)[patch_idx]

        return (
            torch.from_numpy(lq_p).unsqueeze(0),  # (1, P, P)
            torch.from_numpy(hq_p).unsqueeze(0),
        )


def make_loader(split: str, batch_size: int = 32, patch: bool = True,
                num_workers: int = 4, **kwargs) -> DataLoader:
    ds = CTDenoisePairDataset(split=split, patch=patch)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, **kwargs)
