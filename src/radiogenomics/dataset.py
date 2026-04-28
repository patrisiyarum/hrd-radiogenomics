"""VolumeDataset: serves 96^3 preprocessed CT volumes keyed by patient.

Expects the preprocess_all Snakemake rule to have run already, writing a
single .npy per patient under `data/preprocessed/<bcr_patient_barcode>.npy`.
Loading from disk avoids re-running the (slow) DICOM series read on every
epoch.

Training loaders pass `augment=True` to get random flips, rotations, and
intensity jitter — cheap regularisation that typically adds 0.03-0.05
AUROC on small radiomics cohorts. Validation loaders pass `augment=False`
so the val metric is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PREPROCESSED_DIR = Path("data/preprocessed")


def _build_augment_pipeline(strong: bool = False):
    """MONAI augmentation stack.

    Two presets:
      - default (`strong=False`, v1): light flip + small rotation + intensity
        jitter. Kept conservative for the 135-patient cohort.
      - strong (`strong=True`, v2): adds scanner-style perturbations
        (Gaussian noise, blur, gamma) and a wider rotation range. Targets
        the cross-fold variance issue v1 showed (folds 1-4 disagreed
        substantially with fold 0 on the test set). Scanner aug specifically
        is the standard fix for "model overfit to one scanner brand".
    """
    import monai.transforms as T

    if not strong:
        return T.Compose(
            [
                T.RandFlip(prob=0.5, spatial_axis=0),
                T.RandFlip(prob=0.5, spatial_axis=1),
                T.RandFlip(prob=0.5, spatial_axis=2),
                T.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, mode="bilinear"),
                T.RandScaleIntensity(factors=0.1, prob=0.3),
                T.RandShiftIntensity(offsets=0.05, prob=0.3),
            ]
        )

    return T.Compose(
        [
            # Geometric — wider rotation than v1, more aggressive flip.
            T.RandFlip(prob=0.5, spatial_axis=0),
            T.RandFlip(prob=0.5, spatial_axis=1),
            T.RandFlip(prob=0.5, spatial_axis=2),
            T.RandRotate(
                range_x=0.2, range_y=0.2, range_z=0.2,
                prob=0.7, mode="bilinear",
            ),
            T.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3, mode="trilinear"),
            # Intensity — same as v1.
            T.RandScaleIntensity(factors=0.15, prob=0.5),
            T.RandShiftIntensity(offsets=0.08, prob=0.5),
            # Scanner-style perturbations — the core of the v2 changes.
            # Different scanners produce different noise profiles, slightly
            # different reconstruction kernels (≈ blur), and different
            # contrast curves (≈ gamma). Training the model to handle
            # those should improve generalization to unseen scanners.
            T.RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
            T.RandGaussianSmooth(
                sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0),
                prob=0.2,
            ),
            T.RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
        ]
    )


class VolumeDataset(Dataset):
    """One patient per row. Returns (volume, label) where:
        volume: torch.float32 of shape (1, 96, 96, 96)   (channel dim added)
        label:  torch.long   — 1 for HRD, 0 for non-HRD
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        preprocessed_dir: Path = PREPROCESSED_DIR,
        augment: bool = False,
        strong_augment: bool = False,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.preprocessed_dir = preprocessed_dir
        self.labels = self.manifest["hrd_class"].map({"HRD": 1, "non-HRD": 0}).values
        self.augment = augment
        self._aug = (
            _build_augment_pipeline(strong=strong_augment) if augment else None
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        barcode = self.manifest.iloc[i]["bcr_patient_barcode"]
        path = self.preprocessed_dir / f"{barcode}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"no preprocessed volume for {barcode} at {path} — did you "
                "run `snakemake preprocess_all`?"
            )
        vol = np.load(path).astype(np.float32)
        # (Z, Y, X) → (1, Z, Y, X) so the CNN sees channel=1.
        vol = vol[np.newaxis, ...]
        if self._aug is not None:
            vol = np.asarray(self._aug(vol))
        label = int(self.labels[i])
        return torch.from_numpy(vol).float(), torch.tensor(label, dtype=torch.long)
