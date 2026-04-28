"""Bootstrap confidence intervals for the held-out test AUROC.

The 27-patient test set is small enough that a single AUROC point estimate
hides a lot of uncertainty. Bootstrap resamples (with replacement) the test
set N times, computes AUROC each time, and reports the 2.5th / 97.5th
percentiles → a 95% confidence interval for the AUROC.

Usage (on Lambda or any box with the preprocessed cubes + checkpoints):
    uv run python scripts/bootstrap_test_auroc.py \
        --model-dir models/radiogen_v1 --n-boot 1000

Outputs `data/bootstrap_test_auroc.json`:
    {
      "n_test": 27,
      "n_boot": 1000,
      "auroc_point": 0.91,
      "auroc_ci_low": 0.79,
      "auroc_ci_high": 0.99,
      "fold0_auroc_point": 0.91,
      ...
    }

Reuses the same fold-loading logic as find_better_demo_ct.py (no prefix
stripping when going via the training-repo wrapper).
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
from sklearn.metrics import roc_auc_score

from radiogenomics.train import _build_model

logger = logging.getLogger("bootstrap_test_auroc")


def _score_fold(model, volumes: dict, barcodes, device: str) -> np.ndarray:
    probs = np.zeros(len(barcodes), dtype=np.float32)
    with torch.no_grad():
        for i, bc in enumerate(barcodes):
            v = volumes[bc]
            t = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device)
            logit = model(t).item()
            probs[i] = float(torch.sigmoid(torch.tensor(logit)).item())
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/radiogen_v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--n-boot", type=int, default=1000,
        help="Number of bootstrap resamples (default 1000).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    test_path = args.data_dir / "manifest_test.parquet"
    if not test_path.exists():
        sys.exit(f"missing {test_path}; run snakemake split_holdout_test first.")
    test_df = pd.read_parquet(test_path)
    barcodes = list(test_df["bcr_patient_barcode"])
    y = (test_df["hrd_class"] == "HRD").astype(int).to_numpy()
    logger.info("test cohort: %d patients (%d HRD / %d non-HRD)",
                len(test_df), int(y.sum()), int((1 - y).sum()))

    pp_dir = args.data_dir / "preprocessed"
    volumes: dict = {}
    for bc in barcodes:
        path = pp_dir / f"{bc}.npy"
        if not path.exists():
            sys.exit(f"missing {path}; run snakemake preprocess_all first.")
        volumes[bc] = np.load(path).astype(np.float32)

    fold_paths = sorted(args.model_dir.glob("fold*.pt"))
    if not fold_paths:
        sys.exit(f"no fold*.pt in {args.model_dir}")
    logger.info("found %d fold checkpoints", len(fold_paths))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    # Score each fold separately, also the ensemble. We bootstrap on each
    # so the report shows whether ensembling actually helps.
    all_fold_probs = np.zeros((len(fold_paths), len(barcodes)), dtype=np.float32)
    for fi, p in enumerate(fold_paths):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        backbone = ckpt.get("model_card", {}).get("backbone", "med3d")
        model = _build_model(backbone)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.to(device).eval()
        all_fold_probs[fi] = _score_fold(model, volumes, barcodes, device)
        logger.info("  fold %d done", fi)

    ensemble = all_fold_probs.mean(axis=0)

    rng = np.random.default_rng(args.seed)
    n = len(y)

    def bootstrap(probs: np.ndarray) -> tuple[float, float, float]:
        point = float(roc_auc_score(y, probs))
        boot_aurocs = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            ys = y[idx]
            ps = probs[idx]
            # Skip resamples where one class is missing (AUROC undefined).
            if ys.sum() == 0 or ys.sum() == n:
                continue
            boot_aurocs.append(roc_auc_score(ys, ps))
        lo = float(np.percentile(boot_aurocs, 2.5))
        hi = float(np.percentile(boot_aurocs, 97.5))
        return point, lo, hi

    out = {
        "n_test": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_boot": args.n_boot,
        "ensemble": dict(zip(("auroc", "ci_low", "ci_high"), bootstrap(ensemble))),
    }
    for fi in range(len(fold_paths)):
        out[f"fold{fi}"] = dict(zip(
            ("auroc", "ci_low", "ci_high"), bootstrap(all_fold_probs[fi]),
        ))

    out_path = args.data_dir / "bootstrap_test_auroc.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    print()
    print("=" * 80)
    print(f"Held-out test AUROC, 95% bootstrap CI (n={n}, n_boot={args.n_boot})")
    print("=" * 80)
    print(f"{'series':<12} {'AUROC':>7}   95% CI")
    print("-" * 80)
    for key in ["ensemble"] + [f"fold{i}" for i in range(len(fold_paths))]:
        row = out[key]
        print(f"{key:<12} {row['auroc']:.3f}   [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
    print()


if __name__ == "__main__":
    main()
