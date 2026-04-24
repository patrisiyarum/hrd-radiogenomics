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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

TCIA_API = "https://services.cancerimagingarchive.net/services/v4/TCIA"


def _make_session() -> requests.Session:
    """Session with retries so transient TCIA timeouts don't kill the whole run."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,  # 1.5s, 3s, 6s, 12s, 24s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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
    session = _make_session()
    rows: list[dict] = []
    for i, barcode in enumerate(patient_barcodes, 1):
        try:
            resp = session.get(
                f"{TCIA_API}/query/getSeries",
                params={
                    "Collection": collection,
                    "PatientID": barcode,
                    "Modality": modality,
                },
                timeout=(15, 120),  # (connect, read)
            )
        except requests.RequestException as e:
            logger.warning("TCIA request error for %s: %s", barcode, e)
            continue
        if not resp.ok:
            logger.warning("TCIA query failed for %s: %s", barcode, resp.status_code)
            continue
        if i % 50 == 0:
            logger.info("TCIA: queried %d / %d patients (%d series so far)",
                        i, len(patient_barcodes), len(rows))
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
