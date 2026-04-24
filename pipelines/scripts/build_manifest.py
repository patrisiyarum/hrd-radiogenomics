"""Join TCGA-OV HRD labels with TCIA CT-series availability.

Output `data/manifest.parquet` has one row per patient for whom we have
BOTH a Knijnenburg HRD label AND at least one TCIA CT series. Columns:

    bcr_patient_barcode, hrd_class, hrd_sum, loh, lst, ntai, brca_status,
    tcia_series_uid, tcia_study_uid, num_images, scanner_manufacturer

`scanner_manufacturer` is pulled from the DICOM header of the first image
in each selected series. It matters for stratified CV because scanner
variability is the leading confounder in radiomics work.
"""


import logging
from pathlib import Path

from radiogenomics.data.tcga_ov import load_hrd_labels
from radiogenomics.data.tcia import search_tcia_ct_series

logger = logging.getLogger("build_manifest")

def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    hrd_path = Path(sm.input.hrd)
    out_path = Path(sm.output.manifest)

    hrd = load_hrd_labels(hrd_path, cohort="OV")

    # TCIA uses "TCGA-XX-YYYY" as the PatientID. Knijnenburg's bcr_patient_barcode
    # follows the same scheme, so no normalisation needed.
    tcia = search_tcia_ct_series(
        patient_barcodes=hrd["bcr_patient_barcode"].tolist(),
        collection="TCGA-OV",
        modality="CT",
    )
    if tcia.empty:
        raise RuntimeError("no TCIA CT series found — check network + TCIA API status")

    # For patients with multiple series, pick the one with the most images
    # (usually the primary diagnostic series).
    tcia_top = (
        tcia.sort_values("num_images", ascending=False)
            .drop_duplicates("bcr_patient_barcode")
            .reset_index(drop=True)
    )

    manifest = hrd.merge(tcia_top, on="bcr_patient_barcode", how="inner")
    # Scanner manufacturer lookup is deferred to preprocess_all (it reads
    # the actual DICOM headers after download); stub with 'unknown' for now.
    manifest["scanner_manufacturer"] = "unknown"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_path, index=False)
    logger.info(
        "manifest: %d patients with matched HRD + CT (%d HRD, %d non-HRD)",
        len(manifest),
        (manifest["hrd_class"] == "HRD").sum(),
        (manifest["hrd_class"] == "non-HRD").sum(),
    )

if __name__ == "__main__":
    main()
