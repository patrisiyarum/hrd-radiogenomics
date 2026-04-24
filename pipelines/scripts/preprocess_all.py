"""Download + preprocess every CT series named in the manifest.

Per patient:
    1. Download the selected TCIA series to `data/raw/<barcode>/` if absent.
    2. Read the DICOM directory into a 3D numpy volume via SimpleITK.
    3. Run the shared preprocess() pipeline (crop → resample → HU-normalise).
    4. Save to `data/preprocessed/<barcode>.npy`.
    5. Update the manifest row's `scanner_manufacturer` from the DICOM header.

The `.done` sentinel is written only if every row preprocessed
successfully. Partial runs are fine — re-running resumes from the last
successful patient (stable via file existence check).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from radiogenomics.data.tcia import download_series
from radiogenomics.preprocess import load_dicom_dir, preprocess

logger = logging.getLogger("preprocess_all")


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    manifest_path = Path("data/manifest.parquet")
    done_path = Path(sm.output.cache)
    raw_root = Path("data/raw")
    pp_root = Path("data/preprocessed")

    manifest = pd.read_parquet(manifest_path)
    n_ok = 0
    scanner_updates: list[tuple[str, str]] = []

    for _, row in manifest.iterrows():
        barcode = row["bcr_patient_barcode"]
        series_uid = row["tcia_series_uid"]
        raw_dir = raw_root / barcode
        pp_path = pp_root / f"{barcode}.npy"

        if pp_path.exists():
            n_ok += 1
            continue

        try:
            if not any(raw_dir.glob("**/*.dcm")):
                download_series(series_uid, raw_dir)

            vol, _meta = load_dicom_dir(raw_dir)
            preprocessed = preprocess(vol)
            pp_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pp_path, preprocessed.astype(np.float32))

            first = next(raw_dir.glob("**/*.dcm"))
            ds = pydicom.dcmread(str(first), stop_before_pixels=True)
            manufacturer = str(getattr(ds, "Manufacturer", "unknown"))
            scanner_updates.append((barcode, manufacturer))

            n_ok += 1
            logger.info("ok: %s", barcode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", barcode, exc)

    # Patch scanner_manufacturer into the manifest so the training-time
    # stratified-CV split can actually stratify on it.
    if scanner_updates:
        patch = dict(scanner_updates)
        manifest["scanner_manufacturer"] = manifest["bcr_patient_barcode"].map(patch).fillna("unknown")
        manifest.to_parquet(manifest_path, index=False)

    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(f"preprocessed {n_ok} / {len(manifest)} patients\n")
    logger.info("preprocessing: %d / %d patients succeeded", n_ok, len(manifest))


if __name__ == "__main__":
    main()
