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


def _build_augment_pipeline():
    """Light MONAI augmentation stack.

    Kept intentionally conservative for a 135-patient cohort — aggressive
    aug on tiny datasets destroys the signal we're trying to learn.
    """
    import monai.transforms as T

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
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.preprocessed_dir = preprocessed_dir
        self.labels = self.manifest["hrd_class"].map({"HRD": 1, "non-HRD": 0}).values
        self.augment = augment
        self._aug = _build_augment_pipeline() if augment else None

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
