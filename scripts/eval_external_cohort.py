"""End-to-end external-validation scorer.

Takes a manifest produced by build_brca_external_manifest.py (or any
compatible manifest with bcr_patient_barcode + hrd_class + tcia_series_uid
columns), downloads the DICOMs from TCIA, preprocesses them through the
same crop → resample → HU-normalize pipeline as training, scores each
patient with the loaded v1 model, and reports:

    - point AUROC + 95% bootstrap CI
    - per-fold AUROCs (so cross-fold variance is visible)
    - confusion matrix at thresholds 0.50 and 0.85
    - per-patient predictions for audit

Usage (on Lambda):
    uv run python scripts/eval_external_cohort.py \
        --manifest data/manifest_external_brca.parquet \
        --model-dir models/radiogen_v1 \
        --n-boot 1000

Outputs `data/external_brca_eval.json` with the full report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from radiogenomics.data.tcia import download_series
from radiogenomics.preprocess import load_dicom_dir, preprocess
from radiogenomics.train import _build_model

logger = logging.getLogger("eval_external_cohort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/radiogen_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threshold-default", type=float, default=0.50,
        help="Naive threshold (matches training-time default)",
    )
    parser.add_argument(
        "--threshold-calibrated", type=float, default=0.85,
        help="Calibrated threshold the production API uses",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    if not args.manifest.exists():
        sys.exit(f"missing {args.manifest}")
    df = pd.read_parquet(args.manifest)
    logger.info(
        "external cohort: %d patients (%d HRD / %d non-HRD)",
        len(df),
        (df["hrd_class"] == "HRD").sum(),
        (df["hrd_class"] == "non-HRD").sum(),
    )

    raw_dir = args.cache_dir / "raw"
    pp_dir = args.cache_dir / "preprocessed"
    pp_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleanup = os.environ.get("CLEANUP_RAW", "").lower() in ("1", "true", "yes")

    # Download + preprocess each patient. Resumable: skips any patient
    # whose preprocessed .npy already exists.
    volumes: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        bc = row["bcr_patient_barcode"]
        pp_path = pp_dir / f"{bc}.npy"
        if pp_path.exists():
            volumes[bc] = np.load(pp_path).astype(np.float32)
            continue
        rd = raw_dir / bc
        if not any(rd.glob("**/*.dcm")):
            try:
                download_series(row["tcia_series_uid"], rd)
            except Exception as exc:  # noqa: BLE001
                logger.warning("download failed for %s: %s", bc, exc)
                continue
        try:
            vol, _meta = load_dicom_dir(rd)
            arr = preprocess(vol).astype(np.float32)
            np.save(pp_path, arr)
            volumes[bc] = arr
            logger.info("preprocessed %s", bc)
            if cleanup:
                import shutil
                shutil.rmtree(rd, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preprocess failed for %s: %s", bc, exc)

    # Filter manifest to patients we successfully preprocessed.
    df = df[df["bcr_patient_barcode"].isin(volumes.keys())].reset_index(drop=True)
    barcodes = df["bcr_patient_barcode"].tolist()
    y = (df["hrd_class"] == "HRD").astype(int).to_numpy()
    logger.info(
        "scoring %d preprocessed patients (%d HRD / %d non-HRD)",
        len(df), int(y.sum()), int((1 - y).sum()),
    )

    # Score each fold + ensemble.
    fold_paths = sorted(args.model_dir.glob("fold*.pt"))
    if not fold_paths:
        sys.exit(f"no fold*.pt in {args.model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s, %d folds", device, len(fold_paths))

    fold_probs = np.zeros((len(fold_paths), len(barcodes)), dtype=np.float32)
    for fi, p in enumerate(fold_paths):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        backbone = ckpt.get("model_card", {}).get("backbone", "med3d")
        model = _build_model(backbone)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.to(device).eval()
        with torch.no_grad():
            for i, bc in enumerate(barcodes):
                v = volumes[bc]
                t = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device)
                logit = model(t).item()
                fold_probs[fi, i] = float(torch.sigmoid(torch.tensor(logit)).item())
        logger.info("  fold %d done", fi)

    ensemble = fold_probs.mean(axis=0)
    rng = np.random.default_rng(args.seed)

    def bootstrap(probs: np.ndarray) -> dict:
        point = float(roc_auc_score(y, probs))
        ap = float(average_precision_score(y, probs))
        boot = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, len(y), len(y))
            ys, ps = y[idx], probs[idx]
            if ys.sum() == 0 or ys.sum() == len(ys):
                continue
            boot.append(roc_auc_score(ys, ps))
        return {
            "auroc": point,
            "auprc": ap,
            "auroc_ci_low": float(np.percentile(boot, 2.5)),
            "auroc_ci_high": float(np.percentile(boot, 97.5)),
        }

    def confusion(probs: np.ndarray, thr: float) -> dict:
        pred = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "threshold": thr,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
        }

    out = {
        "manifest": str(args.manifest),
        "n_total": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_boot": args.n_boot,
        "ensemble": bootstrap(ensemble),
        "ensemble_confusion": [
            confusion(ensemble, args.threshold_default),
            confusion(ensemble, args.threshold_calibrated),
        ],
        "per_fold": [
            {"fold": fi, **bootstrap(fold_probs[fi])} for fi in range(len(fold_paths))
        ],
        "per_patient": [
            {
                "barcode": str(bc),
                "label": int(yy),
                "ensemble_p": float(ensemble[i]),
                "fold_p": [float(fold_probs[fi, i]) for fi in range(len(fold_paths))],
            }
            for i, (bc, yy) in enumerate(zip(barcodes, y, strict=False))
        ],
    }

    out_path = args.data_dir / "external_brca_eval.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    print()
    print("=" * 80)
    print(f"External validation on TCGA-BRCA: {len(df)} patients")
    print("=" * 80)
    e = out["ensemble"]
    print(f"  Ensemble AUROC:  {e['auroc']:.3f}  (95% CI: {e['auroc_ci_low']:.3f} – {e['auroc_ci_high']:.3f})")
    print(f"  Ensemble AUPRC:  {e['auprc']:.3f}")
    print()
    for f in out["per_fold"]:
        print(f"  Fold {f['fold']} AUROC:    {f['auroc']:.3f}  ({f['auroc_ci_low']:.3f} – {f['auroc_ci_high']:.3f})")
    print()
    print("Confusion at thresholds:")
    for c in out["ensemble_confusion"]:
        print(f"  τ={c['threshold']:.2f}: "
              f"sens={c['sensitivity']:.2f}  spec={c['specificity']:.2f}  "
              f"(TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']})")
    print()


if __name__ == "__main__":
    main()
