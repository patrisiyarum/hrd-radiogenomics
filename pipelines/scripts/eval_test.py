"""Ensemble evaluation on the held-out test split.

Loads all 5 fold checkpoints, predicts each test patient with every
fold model, averages the sigmoid probabilities to produce one ensemble
score per patient, then computes test AUROC + AUPRC + per-patient rows.

This is the number that turns "validation AUROC" into "held-out test
AUROC" — the true never-seen-during-training generalisation metric.
The 27 test patients in manifest_test.parquet were split off BEFORE
train_cv ran, so no fold model has ever seen them.

Output: data/test_eval.json with
    {
      "n": 27,
      "backbone": "med3d",
      "test_auroc": 0.78,
      "test_auprc": 0.71,
      "cv_mean_auroc": 0.63,
      "auroc_delta_test_minus_cv": 0.15,
      "per_patient": [
        {"barcode": "TCGA-...", "label": 1, "ensemble_p": 0.84,
         "fold_p": [0.82, 0.86, 0.79, 0.88, 0.85]},
        ...
      ]
    }
"""


import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from radiogenomics.dataset import VolumeDataset
from radiogenomics.train import _build_model

logger = logging.getLogger("eval_test")


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    summary_path = Path(sm.input.summary)
    test_manifest_path = Path(sm.input.test_manifest)
    out_path = Path(sm.output.report)
    model_dir = summary_path.parent

    summary = json.loads(summary_path.read_text())
    test_df = pd.read_parquet(test_manifest_path)
    n = len(test_df)
    logger.info("test set: %d patients", n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    # Load each fold checkpoint, run inference once per fold, accumulate
    # sigmoid outputs into a (n_folds, n_test) matrix. Keep batch_size=1 so
    # the order is rock-solid stable across folds (no DataLoader shuffle).
    fold_paths = sorted(model_dir.glob("fold*.pt"))
    logger.info("found %d fold checkpoints: %s", len(fold_paths),
                [p.name for p in fold_paths])

    all_fold_probs: list[list[float]] = []
    backbone = None
    labels: list[int] | None = None
    barcodes: list[str] | None = None

    for ckpt_path in fold_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        card = ckpt.get("model_card", {})
        backbone = card.get("backbone", backbone)
        model = _build_model(card.get("backbone", "monai_densenet"))
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device).eval()

        loader = DataLoader(VolumeDataset(test_df), batch_size=1, shuffle=False)
        fold_probs: list[float] = []
        fold_labels: list[int] = []
        with torch.no_grad():
            for x, y in loader:
                p = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel().tolist()
                fold_probs.extend(p)
                fold_labels.extend(y.tolist())
        all_fold_probs.append(fold_probs)
        if labels is None:
            labels = fold_labels
            barcodes = test_df["bcr_patient_barcode"].tolist()
        logger.info("    %s done — %d predictions", ckpt_path.name, len(fold_probs))
        # Free GPU before loading next fold.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    fold_arr = np.asarray(all_fold_probs)         # (n_folds, n_test)
    ensemble_p = fold_arr.mean(axis=0)            # (n_test,)
    y = np.asarray(labels, dtype=int)

    # The whole point of this script:
    test_auroc = float(roc_auc_score(y, ensemble_p))
    test_auprc = float(average_precision_score(y, ensemble_p))
    cv_mean = float(summary.get("mean_auroc", float("nan")))

    per_patient = [
        {
            "barcode": barcodes[i],            # type: ignore[index]
            "label": int(y[i]),
            "ensemble_p": float(ensemble_p[i]),
            "fold_p": [float(fold_arr[k, i]) for k in range(len(fold_paths))],
        }
        for i in range(n)
    ]

    metrics = {
        "n": n,
        "backbone": backbone,
        "n_folds_in_ensemble": len(fold_paths),
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "cv_mean_auroc": cv_mean,
        "auroc_delta_test_minus_cv": test_auroc - cv_mean,
        "per_patient": per_patient,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("=" * 60)
    logger.info("HELD-OUT TEST AUROC: %.3f  (CV mean was %.3f)", test_auroc, cv_mean)
    logger.info("HELD-OUT TEST AUPRC: %.3f", test_auprc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
