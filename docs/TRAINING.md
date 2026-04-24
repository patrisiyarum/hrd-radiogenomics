# Training hrd-radiogenomics

Concrete end-to-end guide for going from an empty clone to a trained
CT → HRD model. All paths are relative to the repo root unless noted.

## What you need

### Data (fully public path — ~150 GB)

| What | Source | Access | Size |
|---|---|---|---|
| Knijnenburg 2018 HRD labels | Cell Rep 2018, Table S2 | Public | ~100 KB |
| TCGA-OV clinical barcodes | Included in Knijnenburg table | Public | — |
| TCIA CT series (TCGA-OV collection) | `services.cancerimagingarchive.net` REST | Public, no auth | ~150 GB across ~140 patients |

### Data (access-restricted path for external validation)

| What | Source | Access | Timeline |
|---|---|---|---|
| Hartwig imaging + HRD | Hartwig Medical Foundation | Institutional DAR | 3–6 months |
| PAOLA-1 | Bergonié / AGO-OVAR | Collaboration request | varies |
| Held-out TCGA-OV split | Already in TCIA | — | — |

Start with the public path. Do the held-out TCIA split as your external
validation and publish with that. Hartwig unlocks stronger generalisation
claims but isn't strictly required for a first release.

### Compute

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 16 GB (RTX 3090, V100, T4 w/ small batch) | A100 40 GB |
| RAM | 16 GB | 32 GB |
| Storage | 200 GB free | 500 GB (cache + checkpoints) |
| Python | 3.11+ | 3.11 |

If you don't have a GPU locally, any of these work fine:

- **Google Colab Pro+** — A100 access, ~$50/mo
- **Lambda Labs** — on-demand A100s, ~$1.29/hr
- **Paperspace Gradient** — free T4 tier or paid A100s
- **Modal / Replicate** — pay-per-call GPU
- **University HPC** — usually free if you have an affiliation

## Exact steps

### 1. Clone + install

```bash
git clone https://github.com/patrisiyarum/hrd-radiogenomics
cd hrd-radiogenomics
uv sync --extra dev
```

`uv sync` pulls torch, MONAI, pydicom, nibabel, SimpleITK, scikit-learn,
and Snakemake. Takes ~2 minutes on a fresh cache.

### 2. Download the HRD labels (~30 seconds)

```bash
uv run snakemake --snakefile pipelines/Snakefile --cores 1 download_knijnenburg
```

Downloads to `data/raw/knijnenburg_2018_tableS2.tsv`. If both mirror URLs
are down, grab the supplementary manually from the Cell Rep paper
(Knijnenburg et al., Cell Rep 2018) and drop it at that path yourself.

### 3. Build the patient manifest (~2–5 minutes)

```bash
uv run snakemake --snakefile pipelines/Snakefile --cores 1 manifest
```

Queries TCIA for every TCGA-OV patient in the HRD label table, keeps
those with at least one CT series, and writes `data/manifest.parquet`.
Expect **~140 rows** — Knijnenburg has HRD labels for ~580 TCGA-OV
patients but TCIA only holds imaging for a subset.

### 4. Download + preprocess every CT (~3–6 hours, ~150 GB disk)

```bash
uv run snakemake --snakefile pipelines/Snakefile --cores 8 preprocess_all
```

Per patient: downloads their DICOM series (~1 GB), reads with
SimpleITK, applies the MONAI preprocess pipeline (crop → resample to
96³ → HU window to [-200, 250] → clamp to [0, 1]), saves a single
`.npy` file per patient. Total ~150 GB of DICOM input, ~100 MB of
preprocessed `.npy` output (a 96³ float32 is ~3.5 MB). Scanner
manufacturer is patched into the manifest at the same time.

Safe to interrupt and resume — the rule skips patients whose `.npy`
already exists.

### 5. Train with 5-fold stratified CV (~4–8 hours on A100)

```bash
uv run snakemake --snakefile pipelines/Snakefile --cores 1 train_cv
```

Or invoke the training loop directly if you want to tune flags:

```bash
uv run python -m radiogenomics.train \
    --manifest data/manifest.parquet \
    --backbone monai_densenet \
    --output models/radiogen_v0/ \
    --epochs 50 \
    --batch-size 4
```

Produces 5 checkpoints (`fold0.pt` through `fold4.pt`) and a
`cv_summary.json` with the per-fold val AUROC and the mean.

**Expected numbers (v0):**

- CV mean AUROC: **0.65–0.80** depending on preprocessing quality
- Per-scanner AUROC spread: often ± 0.10 between manufacturers
- Per-fold val AUROC: usually 0.6–0.85, with one weak fold pulling the
  mean down when a manufacturer is over-represented in one split

If you get > 0.85 on internal CV, be suspicious — usually means a data
leak (patient appearing in both train and val, or a preprocessing shortcut
the model found).

### 6. External validation (~30 minutes)

Drop an external manifest at `data/external_manifest.parquet` with the
same schema as the internal one, then:

```bash
uv run snakemake --snakefile pipelines/Snakefile --cores 1 eval_external
```

Writes `data/external_eval.json` with:

```json
{
  "backbone": "monai_densenet",
  "best_internal_auroc": 0.78,
  "external_auroc": 0.71,
  "auroc_delta": -0.07,
  "external_auprc": 0.66,
  "n": 45
}
```

**Green-lit metric:** external AUROC ≥ 0.70. If you land there, you
have a publishable result. If the delta is < -0.15, the model likely
overfits to scanner-specific artifacts — iterate on:
- stronger spacing + intensity augmentation (MONAI has these built in)
- cross-scanner domain adaptation
- an ensemble across the 5 CV folds

### 7. Saliency analysis

Once you've got a working model, generate Grad-CAM 3D overlays to see
*what* the model is looking at:

```python
from radiogenomics.backbones import load_monai_densenet
from radiogenomics.interpret import grad_cam_3d, save_overlay
import numpy as np, torch

model = load_monai_densenet()
ckpt = torch.load("models/radiogen_v0/fold0.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])

vol = np.load("data/preprocessed/TCGA-04-1332.npy")
heat = grad_cam_3d(model, vol, target_class=1)
save_overlay(vol, heat, out_path="reports/saliency/TCGA-04-1332.png")
```

This is what makes the project publishable — not just "we predict HRD"
but "we predict HRD and the model is looking at the following imaging
features: necrosis patterns, enhancement heterogeneity, tumor-border
irregularity, whatever else shows up."

## Deploy back into drug-cell-viz

Once you have a checkpoint you trust:

```python
# apps/api/src/api/main.py — inside the lifespan handler
from api.services.radiogenomics import set_model_weights
set_model_weights(Path("/data/models/radiogen_v0/fold0.pt"))
```

Add the same code path + point the backend's `MODEL_WEIGHTS_PATH` at
the checkpoint, restart the API, and `/api/radiogenomics/upload` starts
serving real predictions instead of the stub.

## If you don't have the data or compute

A lean first pass that still teaches you the mechanics:

1. Skip TCIA. Use publicly-released 3D CT fixture volumes (e.g. the
   MSD ovarian cancer subset) — ~4 GB of permissively-licensed data.
2. Train on a subset (30–50 patients) just to get the full pipeline
   running end-to-end. Expect AUROC to be noisy but the training loop
   will run.
3. Once you're confident the code is correct, scale up on a rented GPU.

## Time budget

| Step | Wall time | Human effort |
|---|---|---|
| Clone + install | 5 min | 5 min |
| Download labels + manifest | 10 min | 2 min |
| Download + preprocess CTs | 3–6 hours | passive |
| Train 5-fold CV (A100) | 4–8 hours | passive |
| External evaluation | 30 min | passive |
| Interpret results + iterate | varies | 1–2 weeks |

**Total:** one realistic training run from zero takes ~1 working week
if the data access path is public. Count multiples if you're iterating
on preprocessing or backbone choices.
