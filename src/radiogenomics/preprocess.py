"""Volume preprocessing shared with drug-cell-viz.

Contract (identical to drug-cell-viz's services/radiogenomics.py):

    load_volume(raw_bytes, filename) -> (numpy.ndarray[Z,Y,X], VolumeMetadata)
    preprocess(volume, metadata)     -> numpy.ndarray[96, 96, 96] float32 in [0, 1]

Why duplicate? The research repo owns training, which happens offline on
lots of pre-downloaded DICOM files in bulk. The web app's version lives
in the backend module with no external disk access. Both must produce
the same tensors so a checkpoint trained here deploys there without
re-preprocessing the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from radiogenomics import HU_WINDOW, TARGET_SHAPE


@dataclass(frozen=True)
class VolumeMetadata:
    modality: Literal["CT", "MR", "PT", "UNKNOWN"]
    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    target_shape: tuple[int, int, int]
    hu_window: tuple[float, float]


def load_dicom_dir(directory: Path) -> tuple[np.ndarray, VolumeMetadata]:
    """Read a directory of DICOM files into a single 3D volume via SimpleITK.

    Training reads from pre-downloaded DICOM directories (TCIA dumps one
    series per folder); this is the fast in-bulk loader. The web app's
    variant reads a DICOM zip from an HTTP upload instead.
    """
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise FileNotFoundError(f"no DICOM series in {directory}")
    files = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    vol = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
    spacing = tuple(float(s) for s in reversed(image.GetSpacing()))  # (Z, Y, X)
    return vol, VolumeMetadata(
        modality="CT",
        original_shape=tuple(int(d) for d in vol.shape),  # type: ignore[arg-type]
        original_spacing_mm=spacing,
        target_shape=TARGET_SHAPE,
        hu_window=HU_WINDOW,
    )


def preprocess(volume: np.ndarray) -> np.ndarray:
    """Apply the same crop-resample-normalise the web app runs.

    Uses MONAI transforms for consistency with the training pipeline.
    """
    import monai.transforms as T

    tfm = T.Compose(
        [
            T.EnsureChannelFirst(channel_dim="no_channel"),
            T.CropForeground(
                source_key="image",
                select_fn=lambda x: (x >= HU_WINDOW[0]) & (x <= HU_WINDOW[1]),
                margin=4,
            ),
            T.Resize(spatial_size=TARGET_SHAPE, mode="trilinear"),
            T.ScaleIntensityRange(
                a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )
    return np.asarray(tfm(volume))[0]  # drop the channel dim
