"""Build an external-validation manifest from a different TCGA cohort.

Mirrors `build_manifest.py` but pulls a different cancer cohort instead
of ovarian. Default cohort is **TCGA-UCEC (uterine)** because:

  - It has CT imaging on TCIA (TCGA-BRCA does NOT — only MG + MRI, so
    CT-trained models can't be validated on it).
  - Same body region (pelvis) as OV → preprocessing pipeline works
    without modification.
  - Gynecologic cancer with shared HRD biology (BRCA1/2 mutations
    drive a subset of UCEC via the same DNA-repair pathway).
  - Has Knijnenburg HRD labels.

Other CT-bearing TCGA cohorts you can pass via --collection:
  - TCGA-OV   (training cohort — use for sanity check, not external eval)
  - TCGA-LUAD (lung adeno — different organ, different anatomy)
  - TCGA-COAD (colon)
  - TCGA-LIHC (liver)

Why not TCGA-BRCA: TCIA's TCGA-BRCA only has mammography (MG) and MRI,
no CT modality. A CT-trained model has nothing to score on it.

Usage (on Lambda or any machine with network access):
    uv run python scripts/build_brca_external_manifest.py \\
        --collection TCGA-UCEC --max-patients 50

The --max-patients cap exists because cohorts are ~200-500 patients;
downloading + preprocessing all of them eats hours of Lambda time. 50 is
enough for a first-pass external-validation AUROC with reasonable
confidence intervals. Pass --max-patients 0 for "all".

Output filename includes the cohort: `data/manifest_external_<cohort>.parquet`.
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
        "--collection",
        type=str,
        default="TCGA-UCEC",
        help="TCIA collection name (default TCGA-UCEC — uterine cancer; "
             "shares pelvis anatomy + HRD biology with OV, has CT modality)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default data/manifest_external_<cohort>.parquet)",
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

    if args.out is None:
        suffix = args.collection.lower().replace("-", "_")
        args.out = Path(f"data/manifest_external_{suffix}.parquet")

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

    # Knijnenburg covers all 33 TCGA cohorts; the inner-join with TCIA's
    # per-collection query does the cancer-type filtering.
    hrd = load_hrd_labels(args.hrd_table)
    logger.info(
        "Knijnenburg full table: %d patients (%d HRD / %d non-HRD)",
        len(hrd),
        (hrd["hrd_class"] == "HRD").sum(),
        (hrd["hrd_class"] == "non-HRD").sum(),
    )

    # TCIA: pull CT series for the chosen cohort. The collection query
    # filters Knijnenburg's full TCGA table down to just the cancer type
    # we want. (Important: we have to query TCIA *before* subsampling
    # Knijnenburg — otherwise a random 50-patient sample of all of TCGA
    # is unlikely to overlap with the ~65 UCEC patients on TCIA.)
    tcia = search_tcia_ct_series(
        patient_barcodes=hrd["bcr_patient_barcode"].tolist(),
        collection=args.collection,
        modality="CT",
    )
    if tcia.empty:
        sys.exit(
            f"no TCIA CT series found for {args.collection}. Check that "
            "this collection has CT modality (some TCGA cohorts only have "
            "MR / MG / SR)."
        )
    logger.info("TCIA: %d series found across %s cohort", len(tcia), args.collection)

    # Intersect HRD labels with imaging-available patients now, BEFORE
    # subsampling. Otherwise a random sample of the full TCGA table almost
    # never overlaps with the cohort's smaller imaging set.
    hrd_with_imaging = hrd[hrd["bcr_patient_barcode"].isin(tcia["bcr_patient_barcode"])]
    logger.info(
        "intersection (HRD label + imaging in %s): %d patients (%d HRD / %d non-HRD)",
        args.collection,
        len(hrd_with_imaging),
        (hrd_with_imaging["hrd_class"] == "HRD").sum(),
        (hrd_with_imaging["hrd_class"] == "non-HRD").sum(),
    )
    if len(hrd_with_imaging) == 0:
        sys.exit(
            f"intersection of Knijnenburg and {args.collection} imaging is "
            "empty. Either the collection has no patients in Knijnenburg's "
            "table, or barcodes don't match."
        )
    hrd = hrd_with_imaging

    # Stratified subsample if requested. Keep the HRD/non-HRD ratio the same
    # as the cohort so the external-validation AUROC isn't biased by a
    # weird class balance.
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
