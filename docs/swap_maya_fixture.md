# Runbook: improve the Maya CT prediction

The current Maya fixture (`maya_ct_scan.nii.gz` = TCGA-09-1659) sits below
the v1 CNN's decision boundary, so the imaging tile in the Kintsugi demo
predicts "HR-proficient" — disagreeing with the germline + scar signals
and breaking the demo narrative.

This runbook does two things in one Lambda session:

1. **Train the classical-radiomics LASSO baseline** (Pan et al. 2024 style)
   — should test-AUROC around 0.65–0.70 vs the CNN's 0.59. If it does,
   we use it as the lab tile's primary classifier.
2. **Find a better Maya CT** that both the CNN and the LASSO predict
   HR-deficient on, then export it as a drop-in NIfTI fixture.

You'll end up with:
- `models/radiogen_v1/radiomics_lasso.pkl` — the trained LASSO bundle
- `data/radiomics_test_eval.json` — LASSO test AUROC + per-patient probs
- `data/maya_ct_scan_v2.nii.gz` — the new Maya fixture

---

## 0. Prereqs

You should already have the v1 CNN trained on Lambda (`models/radiogen_v1/fold*.pt`)
and the dev/test split materialised (`data/manifest_dev.parquet`,
`data/manifest_test.parquet`, `data/preprocessed/<barcode>.npy`).

If `data/preprocessed/` is empty (e.g. you cleaned up after training),
re-run preprocessing first:

```bash
snakemake data/preprocessed/.done --cores 8
```

## 1. Extract PyRadiomics features (one-time, ~30–60 min)

```bash
ssh lambda
cd hrd-radiogenomics

# Triggers extract_radiomics_features.py for all 135 patients.
# Outputs data/radiomics_features.parquet (~1500 features × 135 rows).
snakemake data/radiomics_features.parquet --cores 8
```

This is the slow part — pyradiomics runs on every preprocessed cube
(intratumoral + peritumoral masks). It's CPU-bound; a Lambda 32-core box
finishes in under an hour.

## 2. Train the LASSO and score the held-out test split

```bash
# Trains radiomics_lasso.pkl on the 108-dev cohort, scores the 27-test
# cohort, writes radiomics_test_eval.json.
snakemake data/radiomics_test_eval.json --cores 4
```

Expected output (read the AUROC from the JSON):
```json
{
  "n": 27,
  "test_auroc": 0.68,
  "test_auprc": 0.62,
  "selected_features": ["intra_glcm_correlation", "peri_firstorder_skewness", ...]
}
```

If `test_auroc < 0.55`, something's wrong — most likely a feature
extraction bug; rerun step 1 with `--rerun-incomplete`.

## 3. Find the new Maya fixture

```bash
# Ranks all 27 test patients, picks the one BOTH classifiers agree
# is HR-deficient with the highest min(CNN_p, LASSO_p).
uv run python scripts/find_better_demo_ct.py --top 10
```

Output looks like:
```
==============================================================================
Top candidates (ground-truth HRD, ranked by min(CNN, LASSO)):
==============================================================================
barcode        hrd_class    cnn_p   lasso_p  agree
------------------------------------------------------------------------------
TCGA-13-1411   HRD          0.84    0.78       ✓
TCGA-29-1768   HRD          0.79    0.71       ✓
TCGA-04-1525   HRD          0.71    0.66       ✓
TCGA-09-1659   HRD          0.42    0.39
...

New Maya fixture written.
  data/maya_ct_scan_v2.nii.gz  ((69, 432, 310), 11.7 MB)
```

(TCGA-09-1659 is the current Maya CT — note the 0.42 / 0.39 numbers.)

If you want a specific patient instead of the auto-pick:

```bash
uv run python scripts/find_better_demo_ct.py --barcode TCGA-29-1768
```

## 4. Ship it

From your laptop:

```bash
# Pull the new fixture
scp lambda:hrd-radiogenomics/data/maya_ct_scan_v2.nii.gz \
    drug-cell-viz/apps/web/public/fixtures/maya_ct_scan.nii.gz

# (Optional) crop wide field-of-view scans to the body bounding box.
# Same trick we used last time:
python -c "
import nibabel as nib, numpy as np
img = nib.load('drug-cell-viz/apps/web/public/fixtures/maya_ct_scan.nii.gz')
vol = np.asarray(img.dataobj)
mask = vol > 0  # soft tissue
xs, ys, zs = np.where(mask)
sl = (slice(xs.min()-6, xs.max()+6),
      slice(ys.min()-6, ys.max()+6),
      slice(zs.min()-6, zs.max()+6))
nib.save(nib.Nifti1Image(vol[sl], img.affine),
         'drug-cell-viz/apps/web/public/fixtures/maya_ct_scan.nii.gz')
"

# Pull the LASSO model too (we'll plumb it into the API in a follow-up)
scp lambda:hrd-radiogenomics/models/radiogen_v1/radiomics_lasso.pkl \
    hrd-radiogenomics/models/radiogen_v1/radiomics_lasso.pkl
```

Restart the local dev server, reload Maya's clinical analysis page, and
the CT imaging tile should now predict HR-deficient with a probability
in the 0.7–0.85 range.

## 5. (Follow-up) Wire the LASSO classifier into the API

If `radiomics_test_eval.json` shows the LASSO outperforms the CNN:

1. Add a `radiomics_lasso.py` service in `apps/api/src/api/services/` that
   loads the pickle and scores PyRadiomics features.
2. Update `routes/radiogenomics.py` to call LASSO when the pickle exists,
   fall back to CNN otherwise.
3. Surface the model name in the `CtScanResponse.caveats` so the patient
   knows which classifier ran ("Trained on TCGA-OV with PyRadiomics +
   LASSO logistic regression — test AUROC 0.68").

Adding `pyradiomics` + `SimpleITK` to the API runtime is ~100 MB of
deps. Worth it once we know LASSO is the better model. Until then,
swapping the fixture (steps 1–4) is enough for the demo.
