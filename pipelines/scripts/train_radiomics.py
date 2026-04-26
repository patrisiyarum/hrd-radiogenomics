"""Train a classical-radiomics LASSO + logistic-regression classifier.

Reads radiomics_features.parquet and manifest_dev.parquet (the 108-patient
development split — same one the deep model uses, so the apples-to-apples
comparison against eval_test is honest), runs the classical radiomics
recipe from Pan et al. 2024:

    1. Standardise features (zero mean, unit variance) on the dev set.
    2. Univariate filter with Mann-Whitney U: drop features whose
       distributions don't differ between HRD+ / non-HRD at p < 0.05.
    3. LASSO logistic regression with cross-validated alpha (LogisticRegressionCV
       with l1 penalty + saga solver). 5-fold internal CV picks the
       sparsity level.
    4. Refit on the full dev set with the LASSO-selected features.

Persists the fitted scaler + selected-feature names + final logistic
model so eval_radiomics_test can reload it without re-running training.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("train_radiomics")

UNIVARIATE_P_THRESHOLD = 0.05


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    features_path = Path(sm.input.features)
    dev_manifest_path = Path(sm.input.dev_manifest)
    out_model_path = Path(sm.output.model)
    out_summary_path = Path(sm.output.summary)

    features = pd.read_parquet(features_path)
    dev = pd.read_parquet(dev_manifest_path)
    df = features.merge(
        dev[["bcr_patient_barcode"]], on="bcr_patient_barcode", how="inner",
    )
    logger.info("dev cohort: %d patients with extracted features", len(df))

    feat_cols = [c for c in df.columns if c.startswith(("intra_", "peri_"))]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    # Replace any inf/NaN that PyRadiomics occasionally emits so sklearn doesn't choke.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (df["hrd_class"] == "HRD").astype(int).to_numpy()

    # Standardise. RobustScaler would be safer for radiomics tails, but
    # with n=108 we're fine on StandardScaler + the LASSO regularisation
    # absorbs the scale anyway.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Univariate filter: drop features whose Mann-Whitney p > 0.05.
    keep_mask = np.ones(len(feat_cols), dtype=bool)
    pos = X_std[y == 1]
    neg = X_std[y == 0]
    for j in range(len(feat_cols)):
        a = pos[:, j]
        b = neg[:, j]
        if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
            keep_mask[j] = False
            continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            keep_mask[j] = False
            continue
        if p > UNIVARIATE_P_THRESHOLD:
            keep_mask[j] = False
    kept_features = [f for f, k in zip(feat_cols, keep_mask, strict=False) if k]
    logger.info(
        "univariate filter: %d / %d features survive p < %.2f",
        len(kept_features), len(feat_cols), UNIVARIATE_P_THRESHOLD,
    )
    if len(kept_features) == 0:
        raise SystemExit(
            "no features survived the Mann-Whitney filter; classes may be too "
            "small or features are degenerate. Check the input parquet.",
        )

    X_filtered = X_std[:, keep_mask]

    # LASSO logistic with internal 5-fold CV to pick alpha.
    model = LogisticRegressionCV(
        Cs=20,
        cv=5,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        max_iter=5000,
        class_weight="balanced",
        random_state=0,
    )
    model.fit(X_filtered, y)
    coef = model.coef_.ravel()
    nonzero_idx = np.where(np.abs(coef) > 1e-9)[0]
    selected_features = [kept_features[i] for i in nonzero_idx]
    logger.info("LASSO retained %d features", len(selected_features))
    for f, c in sorted(
        zip(selected_features, coef[nonzero_idx], strict=False),
        key=lambda t: -abs(t[1]),
    )[:15]:
        logger.info("    coef %+.3f  %s", c, f)

    # Persist scaler + LASSO mask + selected feature names + model.
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    with out_model_path.open("wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "all_features": feat_cols,
                "kept_features": kept_features,
                "selected_features": selected_features,
                "model": model,
            },
            f,
        )

    summary = {
        "n_train": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_features_total": len(feat_cols),
        "n_features_after_univariate": len(kept_features),
        "n_features_after_lasso": len(selected_features),
        "lasso_C_chosen": float(model.C_[0]),
        "selected_features": selected_features,
    }
    out_summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote model to %s and summary to %s", out_model_path, out_summary_path)


if __name__ == "__main__":
    main()
