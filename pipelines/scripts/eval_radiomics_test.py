"""Score the held-out 27-patient test split with the trained classical
radiomics LASSO model.

This is the apples-to-apples comparison number to the deep model's
test_eval.json. Same train/test split, same patients, same evaluation
discipline — different feature representation and classifier."""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("eval_radiomics_test")


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    features_path = Path(sm.input.features)
    test_manifest_path = Path(sm.input.test_manifest)
    model_path = Path(sm.input.model)
    clinical_path = Path(getattr(sm.input, "clinical", "")) if hasattr(sm.input, "clinical") else None
    out_path = Path(sm.output.report)

    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    scaler = bundle["scaler"]
    all_features = bundle["all_features"]
    kept_features = bundle["kept_features"]
    selected_features = bundle["selected_features"]
    model = bundle["model"]

    features = pd.read_parquet(features_path)
    test = pd.read_parquet(test_manifest_path)
    df = features.merge(
        test[["bcr_patient_barcode"]], on="bcr_patient_barcode", how="inner",
    )

    # Same clinical-merge step as training. Missing values get median-imputed
    # using the test set's medians (the trainer already did its own imputation
    # and the LASSO baked the relationship in; tiny test-set bias is fine
    # given the alternative is dropping rows for one missing FIGO stage).
    if clinical_path and clinical_path.exists():
        clin = pd.read_parquet(clinical_path)
        df = df.merge(clin, on="bcr_patient_barcode", how="left")
        clin_cols = [c for c in df.columns if c.startswith("clin_")]
        for c in clin_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

    logger.info("test cohort: %d patients", len(df))

    X = df[all_features].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_std = scaler.transform(X)
    # Restrict to kept_features (post-Mann-Whitney) — the model was fit
    # on these columns only.
    keep_idx = [all_features.index(f) for f in kept_features]
    X_filtered = X_std[:, keep_idx]

    proba = model.predict_proba(X_filtered)[:, 1]
    y = (df["hrd_class"] == "HRD").astype(int).to_numpy()

    test_auroc = float(roc_auc_score(y, proba))
    test_auprc = float(average_precision_score(y, proba))

    per_patient = [
        {
            "barcode": str(b),
            "label": int(yy),
            "p_hrd": float(p),
        }
        for b, yy, p in zip(df["bcr_patient_barcode"], y, proba, strict=False)
    ]

    metrics = {
        "approach": "classical_radiomics_lasso_logistic",
        "n": int(len(df)),
        "n_features_in_model": len(selected_features),
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "per_patient": per_patient,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))

    logger.info("=" * 60)
    logger.info("RADIOMICS HELD-OUT TEST AUROC: %.3f", test_auroc)
    logger.info("RADIOMICS HELD-OUT TEST AUPRC: %.3f", test_auprc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
