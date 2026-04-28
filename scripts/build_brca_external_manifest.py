"""Build a TCGA-BRCA external-validation manifest.

Mirrors `build_manifest.py` but for the breast-cancer cohort instead of
ovarian. Produces `data/manifest_external_brca.parquet` — same schema as
the OV manifest, used as a held-out external-validation cohort.

Why this exists:
    The v1 / v2 model is trained on TCGA-OV. Reporting test AUROC on a
    held-out 27-patient OV split tells us in-distribution performance.
    Reporting AUROC on TCGA-BRCA tells us whether the model is learning
    *cancer-imaging signatures of HRD* (which transfer) or *ovarian-pelvis
    artifacts* (which don't). The latter would be an over-fit story.

    BRCA is the right external cohort: it has Knijnenburg HRD labels,
    a TCIA imaging collection, and HRD-driven biology shares with OV
    (BRCA1/2 mutations cause both cancers via the same DNA-repair
    pathway). If a CT-trained HRD predictor generalizes anywhere, it
    should generalize here.

Usage (on Lambda or any machine with network access):
    uv run python scripts/build_brca_external_manifest.py --max-patients 50

The --max-patients cap exists because BRCA has ~300 patients with both
labels and imaging; downloading + preprocessing all of them eats hours
of Lambda time. 50 is enough for a first-pass external-validation AUROC
with reasonable confidence intervals. Pass --max-patients 0 for "all".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from radiogenomics.data.tcga_ov import load_hrd_labels
from radiogenomics.data.tcia import search_tcia_ct_series

logger = logging.getLogger("build_brca_external_manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hrd-table",
        type=Path,
        default=Path("data/raw/knijnenburg_2018_tableS2.tsv"),
        help="Knijnenburg supplementary table (already downloaded for OV)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/manifest_external_brca.parquet"),
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=50,
        help="Cap patient count to keep download + preprocess tractable. "
             "0 = no cap (~300 patients).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Sampling seed when --max-patients caps the cohort. Deterministic.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    if not args.hrd_table.exists():
        sys.exit(
            f"missing {args.hrd_table}. Run `snakemake "
            "data/raw/knijnenburg_2018_tableS2.tsv` first to fetch the "
            "Knijnenburg table — same file the OV manifest uses."
        )

    # Same loader, different cohort filter.
    hrd = load_hrd_labels(args.hrd_table, cohort="BRCA")
    logger.info(
        "Knijnenburg BRCA: %d patients (%d HRD / %d non-HRD)",
        len(hrd),
        (hrd["hrd_class"] == "HRD").sum(),
        (hrd["hrd_class"] == "non-HRD").sum(),
    )

    # Stratified subsample if requested. Keep the HRD/non-HRD ratio the same
    # as the full cohort so the external-validation AUROC isn't biased by a
    # weird class balance. Done as two separate samples + concat instead of
    # groupby+apply because pandas 2.x groupby+apply has surprising
    # interactions with index-vs-column membership of the grouping key.
    if args.max_patients > 0 and len(hrd) > args.max_patients:
        total = len(hrd)
        cap = args.max_patients
        parts = []
        for cls in ("HRD", "non-HRD"):
            sub = hrd[hrd["hrd_class"] == cls]
            n = max(1, int(round(cap * len(sub) / total)))
            n = min(n, len(sub))
            parts.append(sub.sample(n=n, random_state=args.seed))
        hrd = pd.concat(parts, ignore_index=True)
        logger.info(
            "subsampled to %d patients (%d HRD / %d non-HRD)",
            len(hrd),
            (hrd["hrd_class"] == "HRD").sum(),
            (hrd["hrd_class"] == "non-HRD").sum(),
        )

    # TCIA: pull CT series for the BRCA cohort. Note: TCGA-BRCA on TCIA has
    # MULTIPLE collection identifiers historically (TCGA-BRCA is the canonical
    # one). Most BRCA imaging is mammograms, but a subset has chest CTs which
    # is what the model was trained on (HU values + soft-tissue window).
    tcia = search_tcia_ct_series(
        patient_barcodes=hrd["bcr_patient_barcode"].tolist(),
        collection="TCGA-BRCA",
        modality="CT",
    )
    if tcia.empty:
        sys.exit(
            "no TCIA CT series found for TCGA-BRCA. The BRCA collection on "
            "TCIA is mostly MR + mammography; CT is sparse. Try the wider "
            "BRCA-Diagnosis or BREAST-DIAGNOSIS collections, or fall back to "
            "MR if you adapt the preprocessing pipeline."
        )
    logger.info("TCIA: %d series found across BRCA cohort", len(tcia))

    tcia_top = (
        tcia.sort_values("num_images", ascending=False)
            .drop_duplicates("bcr_patient_barcode")
            .reset_index(drop=True)
    )

    manifest = hrd.merge(tcia_top, on="bcr_patient_barcode", how="inner")
    manifest = manifest.rename(
        columns={
            "series_uid": "tcia_series_uid",
            "study_uid": "tcia_study_uid",
            "manufacturer": "tcia_manufacturer",
        }
    )
    manifest["scanner_manufacturer"] = "unknown"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)
    logger.info(
        "wrote %s: %d patients (%d HRD / %d non-HRD)",
        args.out, len(manifest),
        (manifest["hrd_class"] == "HRD").sum(),
        (manifest["hrd_class"] == "non-HRD").sum(),
    )

    print()
    print("=" * 80)
    print(f"BRCA external-validation manifest: {args.out}")
    print(f"  {len(manifest)} patients with both HRD labels + CT imaging")
    print(f"  HRD-positive: {(manifest['hrd_class'] == 'HRD').sum()}")
    print(f"  non-HRD:      {(manifest['hrd_class'] == 'non-HRD').sum()}")
    print()
    print("Next: download + preprocess these CTs, then score with v1 model.")
    print("    uv run python scripts/preprocess_external_cohort.py \\")
    print(f"        --manifest {args.out}")


if __name__ == "__main__":
    main()
