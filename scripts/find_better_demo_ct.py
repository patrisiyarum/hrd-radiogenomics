"""Find a better Maya CT fixture for the Kintsugi demo.

The current Maya fixture (TCGA-09-1659) sits below the v1 CNN's decision
boundary, so the imaging tile predicts HR-proficient. We want a CT where
the patient is genuinely HRD AND both the CNN and the LASSO classifier
agree on HR-deficient — that's the cleanest demo story.

What this script does:
    1. Loads the v1 CNN ensemble (5 fold checkpoints) from `models/radiogen_v1`.
    2. Loads each test-set patient's preprocessed 96^3 volume from
       `data/preprocessed/<barcode>.npy`.
    3. Runs the ensemble (mean of 5 sigmoid outputs) on every test patient.
    4. (Optional) Loads the LASSO test predictions from
       `data/radiomics_test_eval.json` if it exists, so we can find a
       candidate both classifiers agree on.
    5. Prints a ranked table:
           barcode       hrd_class   ensemble_p   lasso_p   both_agree
    6. Re-downloads the chosen patient's DICOM from TCIA, builds a NIfTI,
       and writes it to `data/maya_ct_scan_v2.nii.gz` ready to drop into
       drug-cell-viz/apps/web/public/fixtures/.

Usage (on Lambda):
    cd hrd-radiogenomics
    uv run python scripts/find_better_demo_ct.py --top 5
        # ranks the test set, picks the highest-scoring HRD patient as
        # the new Maya fixture, downloads + writes the NIfTI

    uv run python scripts/find_better_demo_ct.py --barcode TCGA-XX-XXXX
        # uses an explicit barcode you picked from the ranking

Outputs:
    data/demo_ct_ranking.json          — full per-patient table (audit trail)
    data/maya_ct_scan_v2.nii.gz        — the new Maya fixture

Then SCP `maya_ct_scan_v2.nii.gz` back to:
    drug-cell-viz/apps/web/public/fixtures/maya_ct_scan.nii.gz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radiogenomics.data.tcia import download_series
from radiogenomics.preprocess import load_dicom_dir
from radiogenomics.train import _build_model

logger = logging.getLogger("find_better_demo_ct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/radiogen_v1"),
        help="Directory containing fold*.pt checkpoints (default: v1)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Pipeline data dir (default: data)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top-ranked candidates to print (default 5)",
    )
    parser.add_argument(
        "--barcode",
        type=str,
        default=None,
        help="Skip ranking and use this specific patient's CT as the new fixture",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Just print the ranking table; don't fetch the chosen CT",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    test_manifest_path = args.data_dir / "manifest_test.parquet"
    if not test_manifest_path.exists():
        sys.exit(
            f"missing {test_manifest_path}. Run `snakemake split_holdout_test "
            "--cores 4` first."
        )

    test_df = pd.read_parquet(test_manifest_path)
    logger.info("test cohort: %d patients", len(test_df))

    # --- Score every test patient with the v1 CNN ensemble ----------------
    fold_paths = sorted(args.model_dir.glob("fold*.pt"))
    if not fold_paths:
        sys.exit(f"no fold*.pt files in {args.model_dir}")
    logger.info("found %d fold checkpoints", len(fold_paths))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    pp_dir = args.data_dir / "preprocessed"

    # Load all volumes once into memory (108 dev + 27 test = ~10 MB total
    # at float32 96^3, trivial).
    volumes: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for barcode in test_df["bcr_patient_barcode"]:
        path = pp_dir / f"{barcode}.npy"
        if not path.exists():
            missing.append(barcode)
            continue
        volumes[barcode] = np.load(path).astype(np.float32)
    if missing:
        logger.warning(
            "skipping %d patients with missing preprocessed volumes: %s",
            len(missing), missing[:5],
        )

    barcodes = list(volumes.keys())
    n = len(barcodes)
    logger.info("scoring %d patients across %d folds", n, len(fold_paths))

    fold_probs = np.zeros((len(fold_paths), n), dtype=np.float32)

    for fold_idx, ckpt_path in enumerate(fold_paths):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        backbone = ckpt.get("model_card", {}).get("backbone", "med3d")
        model = _build_model(backbone)
        # _build_model returns a wrapped module whose state_dict already has
        # the "backbone." prefix. The saved checkpoint matches that exact
        # layout, so we load it directly — DO NOT strip the prefix here.
        # (The API service builds a bare backbone and so does need to strip;
        # this script uses the training-repo wrapper and doesn't.)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state"], strict=False,
        )
        if missing or unexpected:
            logger.warning(
                "fold %d: %d missing / %d unexpected keys — load mismatch",
                fold_idx, len(missing), len(unexpected),
            )
        model.to(device).eval()

        with torch.no_grad():
            for i, barcode in enumerate(barcodes):
                vol = volumes[barcode]
                tensor = (
                    torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
                )
                logit = model(tensor).detach().cpu().view(-1)[0].item()
                fold_probs[fold_idx, i] = float(torch.sigmoid(torch.tensor(logit)).item())

        logger.info("  fold %d done (file: %s)", fold_idx + 1, ckpt_path.name)

    ensemble_p = fold_probs.mean(axis=0)

    # --- Pull LASSO predictions if available ------------------------------
    lasso_p_by_barcode: dict[str, float] = {}
    lasso_path = args.data_dir / "radiomics_test_eval.json"
    if lasso_path.exists():
        lasso_data = json.loads(lasso_path.read_text())
        for row in lasso_data.get("per_patient", []):
            lasso_p_by_barcode[row["barcode"]] = float(row["p_hrd"])
        logger.info("loaded LASSO predictions for %d patients", len(lasso_p_by_barcode))
    else:
        logger.info(
            "no LASSO predictions at %s — skipping (rank by CNN only). "
            "Run `snakemake data/radiomics_test_eval.json --cores 8` to "
            "include LASSO in the ranking.", lasso_path,
        )

    # --- Build the ranking table ------------------------------------------
    label_by_barcode = dict(
        zip(test_df["bcr_patient_barcode"], test_df["hrd_class"], strict=False)
    )

    rows = []
    for i, barcode in enumerate(barcodes):
        rows.append(
            {
                "barcode": barcode,
                "hrd_class": label_by_barcode.get(barcode, "?"),
                "ensemble_p": float(ensemble_p[i]),
                "lasso_p": lasso_p_by_barcode.get(barcode),
            },
        )

    # The best Maya fixture is a patient who:
    #   1. Has ground-truth HRD class = "HRD" (truthful demo)
    #   2. CNN ensemble predicts > 0.5 (would label HR-deficient)
    #   3. LASSO also predicts > 0.5 if we have it (both models agree)
    # Sort by min(CNN, LASSO) when both available, else CNN — picks the
    # patient where the WEAKER model still confidently calls HRD.
    def score(r: dict) -> float:
        c = r["ensemble_p"]
        l = r["lasso_p"]
        return min(c, l) if l is not None else c

    hrd_rows = [r for r in rows if r["hrd_class"] == "HRD"]
    hrd_rows.sort(key=score, reverse=True)

    print()
    print("=" * 86)
    print("Top candidates (ground-truth HRD, ranked by min(CNN, LASSO)):")
    print("=" * 86)
    print(f"{'barcode':<14} {'hrd_class':<10} {'cnn_p':>8} {'lasso_p':>9} {'agree':>6}")
    print("-" * 86)
    for r in hrd_rows[: args.top]:
        cnn = f"{r['ensemble_p']:.3f}"
        lasso = f"{r['lasso_p']:.3f}" if r["lasso_p"] is not None else "    -"
        agree_flag = ""
        if r["lasso_p"] is not None:
            agree_flag = "✓" if (r["ensemble_p"] > 0.5 and r["lasso_p"] > 0.5) else " "
        print(f"{r['barcode']:<14} {r['hrd_class']:<10} {cnn:>8} {lasso:>9} {agree_flag:>6}")
    print()

    out_ranking = args.data_dir / "demo_ct_ranking.json"
    out_ranking.write_text(json.dumps(rows, indent=2))
    logger.info("wrote full ranking to %s", out_ranking)

    if args.no_download:
        return

    # --- Pick the chosen barcode and rebuild the NIfTI fixture ------------
    if args.barcode:
        chosen = args.barcode
        if chosen not in label_by_barcode:
            sys.exit(f"barcode {chosen} not in test manifest")
    else:
        if not hrd_rows:
            sys.exit("no HRD-positive candidates in the test set?")
        chosen = hrd_rows[0]["barcode"]

    chosen_row = test_df[test_df["bcr_patient_barcode"] == chosen].iloc[0]
    series_uid = chosen_row["tcia_series_uid"]
    logger.info("chosen Maya fixture: %s (series %s)", chosen, series_uid)

    raw_dir = args.data_dir / "raw" / chosen
    if not any(raw_dir.glob("**/*.dcm")):
        logger.info("downloading DICOMs for %s from TCIA...", chosen)
        download_series(series_uid, raw_dir)

    # Re-load the original DICOM volume so we can write a NIfTI at the
    # patient's native (non-cropped, non-resampled) resolution. The web
    # app's slideshow viewer renders the NIfTI directly; the radiogenomics
    # endpoint will crop + resample on its own when the upload is scored.
    vol, meta = load_dicom_dir(raw_dir)

    # Re-export as NIfTI with the same orientation conventions the web app
    # expects (axial, RAS+). load_dicom_dir returns (Z, Y, X) HU values;
    # nibabel canonical is (X, Y, Z).
    import nibabel as nib

    out_nifti = args.data_dir / "maya_ct_scan_v2.nii.gz"
    vol_xyz = np.transpose(vol.astype(np.int16), (2, 1, 0))
    sz, sy, sx = (
        meta.spacing if hasattr(meta, "spacing") else (1.0, 1.0, 1.0)
    )
    affine = np.diag([float(sx), float(sy), float(sz), 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(vol_xyz, affine), str(out_nifti))
    logger.info(
        "wrote %s (shape %s, %.1f MB)",
        out_nifti, vol.shape, out_nifti.stat().st_size / 1_000_000,
    )

    print()
    print("=" * 86)
    print("New Maya fixture written.")
    print("=" * 86)
    print(f"  {out_nifti}  ({vol.shape}, {out_nifti.stat().st_size / 1_000_000:.1f} MB)")
    print()
    print("Next steps:")
    print(f"  1. SCP back: scp lambda:{out_nifti} drug-cell-viz/apps/web/public/fixtures/maya_ct_scan.nii.gz")
    print("  2. (optional) Crop in Python with the body-bounding-box trick if it has wide field-of-view.")
    print("  3. Restart the dev server, reload Maya's clinical analysis page.")


if __name__ == "__main__":
    main()
