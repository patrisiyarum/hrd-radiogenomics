"""VolumeDataset: serves 96^3 preprocessed CT volumes keyed by patient.

Expects the preprocess_all Snakemake rule to have run already, writing a
single .npy per patient under `data/preprocessed/<bcr_patient_barcode>.npy`.
Loading from disk avoids re-running the (slow) DICOM series read on every
epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PREPROCESSED_DIR = Path("data/preprocessed")


class VolumeDataset(Dataset):
    """One patient per row. Returns (volume, label) where:
        volume: torch.float32 of shape (1, 96, 96, 96)   (channel dim added)
        label:  torch.long   — 1 for HRD, 0 for non-HRD
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        preprocessed_dir: Path = PREPROCESSED_DIR,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.preprocessed_dir = preprocessed_dir
        self.labels = self.manifest["hrd_class"].map({"HRD": 1, "non-HRD": 0}).values

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
        label = int(self.labels[i])
        return torch.from_numpy(vol), torch.tensor(label, dtype=torch.long)
