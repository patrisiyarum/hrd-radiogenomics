"""TCGA HRD label loader — GDC PanCanAtlas publication page.

HRD scores for TCGA come from Knijnenburg et al. (Cell Rep 2018,
"Genomic and molecular landscape of DNA damage repair deficiency across
The Cancer Genome Atlas"). The GDC publication page hosts them as
`TCGA.HRD_withSampleID.txt`, a tab-separated file with columns:

    sampleID   ai1   lst1   hrd-loh   HRD

Where:
  - `sampleID` looks like `TCGA-02-0001-01` — we derive `bcr_patient_barcode`
    by dropping the trailing `-01` (TCGA's per-sample suffix).
  - `ai1` = NtAI count (telomeric allelic imbalance).
  - `lst1` = LST count (large-scale state transitions).
  - `hrd-loh` = HRD-LOH count.
  - `HRD` = the summed score = ai1 + lst1 + hrd-loh.

The file covers every TCGA project (10,648 samples total); we don't filter
to a specific cancer type here because TCIA's per-collection query in
`manifest` does that filtering naturally — we only care about patients
who have both an HRD score AND a matched CT series, and the TCIA query
only returns TCGA-OV patients to begin with.

`hrd_class` is derived from `HRD >= 42` (Myriad myChoice clinical cutoff).
BRCA1/2 mutation status isn't in this file; the research-repo follow-up
step joins it from a separate PanCanAtlas mutation table when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SUPPLEMENTARY = Path("data/raw/knijnenburg_2018_tableS2.tsv")

# Myriad myChoice's clinical cutoff for "HRD-positive" tumors.
HRD_POSITIVE_CUTOFF = 42


def load_hrd_labels(
    path: Path = DEFAULT_SUPPLEMENTARY,
    # Kept for backwards compatibility — older callers passed cohort="OV".
    # The file doesn't carry cancer-type info, so this arg is a no-op.
    cohort: str | None = None,  # noqa: ARG001
) -> pd.DataFrame:
    """Return a dataframe with columns:
        bcr_patient_barcode, hrd_sum, loh, lst, ntai, hrd_class, brca_status.
    `brca_status` is "unknown" for every row — this file doesn't carry it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `snakemake download_knijnenburg` first"
        )

    df = pd.read_csv(path, sep="\t")

    # Derive bcr_patient_barcode from sampleID: drop the -XX suffix that
    # labels the per-sample vial/replicate.
    df["bcr_patient_barcode"] = df["sampleID"].str.rsplit("-", n=1).str[0]

    df = df.rename(
        columns={
            "ai1": "ntai",
            "lst1": "lst",
            "hrd-loh": "loh",
            "HRD": "hrd_sum",
        }
    )
    df["hrd_class"] = df["hrd_sum"].apply(
        lambda s: "HRD" if s >= HRD_POSITIVE_CUTOFF else "non-HRD"
    )
    df["brca_status"] = "unknown"

    # Collapse multiple samples per patient down to the primary tumor
    # (sample-type suffix "-01"). Anything else is a recurrence / normal /
    # replicate that shouldn't drive the imaging match.
    df = df[df["sampleID"].str.endswith("-01")]
    df = df.drop_duplicates(subset="bcr_patient_barcode", keep="first")

    keep = [
        "bcr_patient_barcode", "hrd_sum", "loh", "lst", "ntai",
        "hrd_class", "brca_status",
    ]
    df = df[keep].reset_index(drop=True)
    logger.info(
        "loaded %d TCGA primary-tumor samples with HRD labels "
        "(%d HRD-positive at cutoff %d, %d non-HRD)",
        len(df),
        (df["hrd_class"] == "HRD").sum(),
        HRD_POSITIVE_CUTOFF,
        (df["hrd_class"] == "non-HRD").sum(),
    )
    return df
