"""Carve a true held-out test set out of the full manifest.

Splits 135 patients into 108 dev / 27 test (80/20), stratified on
(HRD class x scanner manufacturer) so both buckets preserve the class
balance and the scanner mix that the leading radiomics confounder.

The test bucket is touched by NO downstream training step. train_cv reads
manifest_dev.parquet only; the eval_test rule scores the trained ensemble
on manifest_test.parquet exactly once at the end.

This is what turns "validation AUROC" into "held-out test AUROC".

Random seed is locked at 0 (matches train.py default) so the split is
reproducible across runs of the same code.
"""


import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("split_holdout_test")

TEST_FRACTION = 0.20
RANDOM_STATE = 0


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    manifest_path = Path(sm.input.manifest)
    dev_out = Path(sm.output.dev)
    test_out = Path(sm.output.test)

    manifest = pd.read_parquet(manifest_path)
    n = len(manifest)
    logger.info("loaded %d patients from %s", n, manifest_path)

    # Stratify on (scanner manufacturer x HRD class). If a stratum has only
    # one member, sklearn rejects the split — fall back to HRD-class-only
    # stratification with a warning. With 135 patients this fallback is rare
    # but safe (manufacturer groups can be tiny).
    scanner = manifest["scanner_manufacturer"].fillna("unknown")
    strata = scanner + "_" + manifest["hrd_class"]
    if strata.value_counts().min() < 2:
        logger.warning(
            "scanner x HRD strata too small for stratification; falling back "
            "to HRD-class-only stratification (%d strata, min count %d)",
            strata.nunique(), strata.value_counts().min(),
        )
        strata = manifest["hrd_class"]

    dev_df, test_df = train_test_split(
        manifest,
        test_size=TEST_FRACTION,
        stratify=strata,
        random_state=RANDOM_STATE,
    )

    dev_out.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_parquet(dev_out, index=False)
    test_df.to_parquet(test_out, index=False)

    def _balance(df: pd.DataFrame) -> str:
        pos = int((df["hrd_class"] == "HRD").sum())
        neg = int((df["hrd_class"] == "non-HRD").sum())
        return f"{len(df)} ({pos} HRD / {neg} non-HRD)"

    logger.info("split: %d total -> dev %s + test %s",
                n, _balance(dev_df), _balance(test_df))
    logger.info(
        "    dev   = %s  (5-fold CV will run on this)\n"
        "    test  = %s  (touched only by eval_test rule)",
        dev_out, test_out,
    )


if __name__ == "__main__":
    main()
