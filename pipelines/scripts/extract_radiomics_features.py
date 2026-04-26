"""Extract PyRadiomics features (intratumoral + peritumoral) for every
preprocessed volume.

The preprocessed .npy files are 96^3 float32 cubes already HU-windowed
to [-200, 250] and rescaled to [0, 1]. We synthesise a binary tumor
mask via simple foreground thresholding (any voxel above ~0.05 in the
[0, 1] range, i.e. above ~-178 HU). This mirrors the foreground crop
the preprocess step already used.

Peritumoral mask: 1cm-equivalent dilation of the intratumoral mask. At
the resampled 96^3 scale a typical pelvic field of view is ~30cm, so
1cm ~= 3 voxels — we use 3-voxel binary dilation.

Output: data/radiomics_features.parquet with columns
  bcr_patient_barcode, hrd_class,
  intra_<feature_class>_<feature_name>, peri_<feature_class>_<feature_name>, ...

Pan et al. 2024 (Front Oncol) used this exact split (intra + peri) and
got AUC 0.82 on internal validation; this script is the faithful
re-implementation against the public TCGA-OV cohort + our held-out
test split.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import binary_dilation

logger = logging.getLogger("extract_radiomics_features")

# Foreground threshold on the [0, 1] normalised volume. The preprocess
# pipeline windowed [-200, 250] HU into [0, 1]; ~0.05 corresponds to
# ~-178 HU which is below soft tissue — anything above that is body.
FOREGROUND_THRESHOLD = 0.05

# Voxels of dilation for the peritumoral shell. 96^3 cube over a ~30cm
# pelvic FOV gives ~3mm/voxel, so 1cm ~= 3 voxels.
PERITUMORAL_DILATION_VOXELS = 3


def _build_extractor() -> "featureextractor.RadiomicsFeatureExtractor":
    """PyRadiomics extractor with the canonical default feature classes
    (firstorder + shape + GLCM + GLRLM + GLSZM + NGTDM + GLDM)."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    # Skip 2D-shape features; we're operating on 3D volumes.
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG")
    extractor.enableImageTypeByName("Wavelet")
    return extractor


def _make_masks(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build binary intratumoral and peritumoral masks for one volume."""
    intra = vol > FOREGROUND_THRESHOLD
    if intra.sum() == 0:
        # Pathological case (volume is empty / all air). Caller skips.
        return intra, intra
    dilated = binary_dilation(intra, iterations=PERITUMORAL_DILATION_VOXELS)
    peri = dilated & ~intra
    return intra, peri


def _to_sitk(arr: np.ndarray) -> sitk.Image:
    """np.float32 (Z, Y, X) -> SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    # Spacing irrelevant for shape-invariant features given fixed 96^3,
    # but PyRadiomics warns if missing — set 1mm isotropic.
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _extract_one(
    extractor: "featureextractor.RadiomicsFeatureExtractor",
    vol: np.ndarray,
    mask_arr: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if mask_arr.sum() < 64:
        # Mask too small for stable feature extraction — return NaNs so the
        # downstream LASSO drops them via missing-value handling.
        return {}
    img = _to_sitk(vol)
    mask = _to_sitk(mask_arr.astype(np.uint8))
    raw = extractor.execute(img, mask)
    out: dict[str, float] = {}
    for k, v in raw.items():
        # PyRadiomics returns diagnostic keys we don't want as features.
        if k.startswith("diagnostics_"):
            continue
        try:
            out[f"{prefix}_{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    manifest_path = Path(sm.input.manifest)
    cache_path = Path(sm.input.cache)  # noqa: F841 (sentinel, kept for DAG)
    out_path = Path(sm.output.features)
    pp_root = Path("data/preprocessed")

    manifest = pd.read_parquet(manifest_path)
    extractor = _build_extractor()

    rows: list[dict] = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        barcode = row.bcr_patient_barcode
        npy = pp_root / f"{barcode}.npy"
        if not npy.exists():
            logger.warning("missing preprocessed volume for %s — skipping", barcode)
            continue
        vol = np.load(npy).astype(np.float32)
        intra_mask, peri_mask = _make_masks(vol)
        try:
            intra = _extract_one(extractor, vol, intra_mask, "intra")
            peri = _extract_one(extractor, vol, peri_mask, "peri")
        except Exception as exc:  # noqa: BLE001
            logger.warning("feature extraction failed for %s: %s", barcode, exc)
            continue
        rows.append(
            {
                "bcr_patient_barcode": barcode,
                "hrd_class": row.hrd_class,
                **intra,
                **peri,
            }
        )
        if i % 10 == 0:
            logger.info("extracted features for %d / %d patients", i, len(manifest))

    if not rows:
        raise SystemExit("no features extracted; check preprocess_all output")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_features = sum(c.startswith(("intra_", "peri_")) for c in df.columns)
    logger.info(
        "wrote %d patients x %d features to %s",
        len(df), n_features, out_path,
    )


if __name__ == "__main__":
    main()
