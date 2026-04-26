"""Download TCGA-OV clinical metadata from GDC for every patient in the
manifest, parse the prognostic fields into numeric columns, write a
parquet that train_radiomics.py / train.py can join in as extra
features alongside the imaging ones.

Pan et al. 2024 used Ki-67 + family history. TCGA-OV reliably exposes
age, FIGO stage, and tumor grade for ~all patients (Ki-67 is not in the
GDC clinical schema for OV; it would need a separate IHC reannotation).
This script grabs the three reliable fields and parses them to numeric.

Output: data/clinical_features.parquet with columns:
    bcr_patient_barcode
    clin_age_years          float  -- age at diagnosis
    clin_figo_stage_num     float  -- 1..4 (NaN if unknown)
    clin_grade_num          float  -- 1..4 (NaN if unknown / GX)
"""

import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("download_clinical_features")

GDC_CASES = "https://api.gdc.cancer.gov/cases"
TCIA_FIELDS = ",".join([
    "submitter_id",
    "diagnoses.age_at_diagnosis",
    "diagnoses.figo_stage",
    "diagnoses.tumor_grade",
])

_FIGO_RE = re.compile(r"stage\s*(IV|III|II|I)", re.IGNORECASE)
_FIGO_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}
_GRADE_RE = re.compile(r"G\s*([1-4])")


def _parse_figo(s: str | None) -> float:
    if not s:
        return float("nan")
    m = _FIGO_RE.search(str(s))
    if not m:
        return float("nan")
    return float(_FIGO_MAP[m.group(1).upper()])


def _parse_grade(s: str | None) -> float:
    if not s:
        return float("nan")
    m = _GRADE_RE.search(str(s))
    if not m:
        return float("nan")
    return float(m.group(1))


def _fetch_clinical(barcodes: list[str]) -> list[dict]:
    """Hit GDC's /cases endpoint with the manifest barcodes. The API limits
    the filter list size; we batch in chunks of 100 to be safe."""
    out: list[dict] = []
    for start in range(0, len(barcodes), 100):
        batch = barcodes[start : start + 100]
        body = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "=",
                        "content": {
                            "field": "project.project_id",
                            "value": "TCGA-OV",
                        },
                    },
                    {
                        "op": "in",
                        "content": {
                            "field": "submitter_id",
                            "value": batch,
                        },
                    },
                ],
            },
            "fields": TCIA_FIELDS,
            "size": str(len(batch) + 10),
            "format": "JSON",
        }
        resp = requests.post(GDC_CASES, json=body, timeout=120)
        resp.raise_for_status()
        out.extend(resp.json().get("data", {}).get("hits", []))
    return out


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    manifest_path = Path(sm.input.manifest)
    out_path = Path(sm.output.clinical)

    manifest = pd.read_parquet(manifest_path)
    barcodes = manifest["bcr_patient_barcode"].tolist()
    logger.info("requesting clinical metadata for %d patients", len(barcodes))

    hits = _fetch_clinical(barcodes)
    logger.info("GDC returned %d cases", len(hits))

    rows: list[dict] = []
    for h in hits:
        diagnoses = h.get("diagnoses") or [{}]
        diag = diagnoses[0] if diagnoses else {}
        age_days = diag.get("age_at_diagnosis")
        rows.append(
            {
                "bcr_patient_barcode": h["submitter_id"],
                "clin_age_years": (
                    float(age_days) / 365.0 if age_days is not None else float("nan")
                ),
                "clin_figo_stage_num": _parse_figo(diag.get("figo_stage")),
                "clin_grade_num": _parse_grade(diag.get("tumor_grade")),
            },
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d clinical rows to %s", len(df), out_path)
    logger.info(
        "missingness: age=%.1f%%, stage=%.1f%%, grade=%.1f%%",
        100 * df["clin_age_years"].isna().mean(),
        100 * df["clin_figo_stage_num"].isna().mean(),
        100 * df["clin_grade_num"].isna().mean(),
    )


if __name__ == "__main__":
    main()
