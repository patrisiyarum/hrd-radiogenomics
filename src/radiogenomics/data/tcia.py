"""TCIA (The Cancer Imaging Archive) search + download.

TCIA exposes a public REST API at https://services.cancerimagingarchive.net/
with no authentication for most collections. We use it to:
  1. Look up CT series for a given TCGA patient barcode.
  2. Download a specific series into a local directory.

For the TCGA-OV paired imaging analysis, the relevant TCIA collection is
"TCGA-OV" which contains ~143 patients with preoperative imaging of the
primary pelvic tumor plus follow-up scans.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TCIA_API = "https://services.cancerimagingarchive.net/services/v4/TCIA"


def search_tcia_ct_series(
    patient_barcodes: list[str],
    collection: str = "TCGA-OV",
    modality: str = "CT",
) -> pd.DataFrame:
    """Return a dataframe of (patient, study, series) tuples for which TCIA has
    CT data.

    Each row: bcr_patient_barcode, study_uid, series_uid, num_images,
    body_part_examined. One patient can have multiple series — later stages
    of the pipeline pick one representative series per patient.
    """
    rows: list[dict] = []
    for barcode in patient_barcodes:
        resp = requests.get(
            f"{TCIA_API}/query/getSeries",
            params={
                "Collection": collection,
                "PatientID": barcode,
                "Modality": modality,
            },
            timeout=60,
        )
        if not resp.ok:
            logger.warning("TCIA query failed for %s: %s", barcode, resp.status_code)
            continue
        for entry in resp.json():
            rows.append(
                {
                    "bcr_patient_barcode": barcode,
                    "study_uid": entry.get("StudyInstanceUID"),
                    "series_uid": entry.get("SeriesInstanceUID"),
                    "num_images": int(entry.get("ImageCount", 0)),
                    "body_part_examined": entry.get("BodyPartExamined"),
                }
            )
    df = pd.DataFrame(rows)
    logger.info(
        "TCIA: %d series found across %d patients in %s",
        len(df),
        df["bcr_patient_barcode"].nunique() if len(df) else 0,
        collection,
    )
    return df


def download_series(series_uid: str, out_dir: Path) -> Path:
    """Fetch every DICOM in a series into `out_dir`. Returns the directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{TCIA_API}/query/getImage",
        params={"SeriesInstanceUID": series_uid},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    zip_path = out_dir / f"{series_uid}.zip"
    with zip_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    # TCIA returns a zip per series; unpack then drop the archive.
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    return out_dir
