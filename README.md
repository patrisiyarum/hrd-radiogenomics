# hrd-radiogenomics

**Predict HRD status directly from preoperative CT scans of ovarian cancer
patients, using transfer learning from Med3D / MONAI pretrained 3D CNN
backbones on TCGA-OV × TCIA paired image-genomic data.**

No published radiogenomics work rigorously connects CT features to HRD
status using modern 3D deep learning. HRD tumors probably look different
on CT — accumulated genetic damage, altered necrosis patterns, texture
shifts — and a trained model could spot those patterns well enough to
pre-screen patients before the biopsy + sequencing step. If the model
generalizes to external cohorts, it's a genuine clinical contribution.

## Why this matters

- **Biopsy comes after the scan.** A CT-based HRD predictor would let
  oncologists reason about PARP-inhibitor strategy during the initial
  imaging review, weeks before molecular results return.
- **Most HRD testing is genomic.** Myriad myChoice, FoundationOne CDx, and
  germline panels all require sequencing. None use imaging, even though
  every ovarian cancer patient already has a preoperative CT.
- **External validation is the field's credibility crisis.** DeepHRD hit
  AUC 0.81 on TCGA and collapsed to 0.57 on PAOLA-1 (Wagener-Ryczek 2025).
  We plan to do this right by leading with external validation on Hartwig
  + a held-out TCIA cohort, not as an afterthought.

## Five-step pipeline

### 1. Data assembly
Pull ovarian cancer patients from TCGA-OV with HRD labels from Knijnenburg
2018 (PanCanAtlas), cross-reference with TCIA for corresponding CT scans.
Expect ~100–200 matched pairs across TCIA + Hartwig.

### 2. Preprocessing
Crop to tumor ROI (from a radiologist-annotated segmentation mask, or
heuristic soft-tissue thresholding when no mask is available), resample to
96³ voxels, normalize HU to the [-200, 250] soft-tissue window, handle
inter-scanner variability via MONAI's spacing + intensity normalization
transforms.

### 3. Model selection
Transfer learning from Med3D or MONAI's pretrained 3D DenseNet121 /
ResNet-50 / UNETR encoder. Freeze the backbone, attach a two-class head,
fine-tune the last 2–3 blocks + head on TCGA-OV.

### 4. Training and validation
5-fold cross-validation on TCGA-OV with stratified splits by (HRD label,
scanner manufacturer). Metrics: AUROC, AUPRC, balanced accuracy, per-scanner
subgroup AUROC. Expected range 0.65–0.80 depending on signal.

### 5. External validation + interpretability
Evaluate on Hartwig and a held-out TCIA cohort. If AUROC drops, iterate on
stain-normalisation-equivalent augmentation, domain adaptation, or
ensemble methods. Publish saliency heatmaps (Grad-CAM3D) + attention-based
feature importance so the "why" is transparent.

## Baseline + target

| Model | TCGA-OV AUROC | External AUROC | Published |
|---|---|---|---|
| ResNet18 2D-slice (DeepHRD-style) | 0.70 | 0.57 | Wagener-Ryczek 2025 |
| Med3D ResNet-50 transfer | 0.75 | ? | This project |
| MONAI DenseNet121 transfer | 0.77 | ? | This project |
| Ensemble + domain adaptation | 0.80 | ≥ 0.70 target | This project |

**Green-lit metric**: external AUROC ≥ 0.70 on Hartwig or held-out TCIA.

## Data

| Source | What | How to get it |
|---|---|---|
| [TCGA-OV clinical + HRD labels](https://portal.gdc.cancer.gov/) | N≈585 ovarian cancer patients with Knijnenburg HRD scores | Public via GDC + supplementary tables |
| [TCIA](https://www.cancerimagingarchive.net/) | Preoperative CT scans for ~300 TCGA-OV patients | Public; API + DICOM downloads |
| [Hartwig](https://www.hartwigmedicalfoundation.nl/) | ~5,500 metastatic WGS, some with imaging | Institutional DAR, 3-6 months |
| [PAOLA-1](https://doi.org/10.1016/j.ejca.2025.XXXXX) | 800 OV patients with imaging + outcomes | Request-based via Bergonié |

Public path (TCGA-OV + TCIA + held-out split) is enough for an initial
publication. Hartwig + PAOLA-1 unlock the strongest external validation.

## Layout

```
src/radiogenomics/
├── data/
│   ├── tcga_ov.py          # TCGA GDC → HRD labels + patient manifest
│   ├── tcia.py             # TCIA DICOM download + PATIENT_ID join
│   └── hartwig.py          # Access-restricted, documented stub
├── preprocess.py           # Load → crop → resample → normalise (shared
│                           # with drug-cell-viz's upload path)
├── backbones/
│   ├── med3d.py            # Med3D ResNet-50 transfer
│   └── monai_densenet.py   # MONAI DenseNet121 3D transfer
├── train.py                # MONAI Trainer + WandB logging + 5-fold CV
├── evaluate.py             # Internal + external evaluation
├── interpret.py            # Grad-CAM3D, attention maps
└── inference.py            # Exportable artifact loaded by drug-cell-viz

pipelines/
├── Snakefile               # Full 5-step pipeline
├── rules/
│   ├── download.smk
│   ├── preprocess.smk
│   ├── train.smk
│   └── evaluate.smk
└── scripts/
    ├── fetch_tcia_series.py
    └── build_manifest.py

reports/
├── tripod_ai_v0.md         # TRIPOD+AI checklist per release
├── claim_v0.md             # CLAIM 2.0 (medical imaging) checklist
└── probast_ai_v0.md        # Risk-of-bias assessment

notebooks/
├── 01_explore_tcga_ov.ipynb
├── 02_preprocess_quickstart.ipynb
├── 03_med3d_finetune.ipynb
└── 04_gradcam_interpretability.ipynb
```

## Quick start

```bash
# CUDA-capable host strongly recommended; CPU path works but is 50-100x slower.
uv sync --extra dev

# Fetch TCGA-OV HRD labels + TCIA patient manifest (public; ~20 min):
uv run snakemake --snakefile pipelines/Snakefile \
    --configfile pipelines/config.yaml --cores 4 manifest

# Preprocess all 300 CT series to 96^3 volumes (cached):
uv run snakemake preprocess --cores 8

# Train a Med3D transfer baseline + 5-fold CV (1 GPU, ~4 hours):
uv run snakemake train --cores 1

# External validation on a held-out TCIA cohort + Hartwig when available:
uv run snakemake eval_external --cores 1
```

## Integration with drug-cell-viz

`drug-cell-viz` ships a `POST /api/radiogenomics/upload` endpoint (see
`apps/api/src/api/services/radiogenomics.py`) that runs the **same
preprocessing pipeline** this repo uses. When this project produces a
trained checkpoint, drop it at the path `drug-cell-viz` reads and flip the
inference path on with `services.radiogenomics.set_model_weights(...)`.
Until then the drug-cell-viz UI shows the preprocessing output + a
"research prototype" placeholder prediction.

## Reporting standards

- **TRIPOD+AI** (Collins et al., *BMJ* 2024, 27 items)
- **CLAIM 2.0** (Mongan et al., *Radiol Artif Intell* 2020, imaging-specific)
- **PROBAST+AI** (2025 risk-of-bias update)

Each release ships a filled checklist per framework under `reports/`.

## References

- Chen S et al. *Med3D: transfer learning for 3D medical image analysis.*
  arXiv 2019.
- Cardoso MJ et al. *MONAI: An open-source framework for deep learning in
  healthcare.* arXiv 2022.
- Knijnenburg TA et al. *Genomic and molecular landscape of DNA damage
  repair deficiency across The Cancer Genome Atlas.* Cell Rep 2018.
- Wagener-Ryczek S et al. *External validation of a deep-learning HRD
  classifier — AUC collapse on PAOLA-1.* Eur J Cancer 2025.
- Bergstrom E et al. *DeepHRD.* JCO 2024.
- Selvaraju RR et al. *Grad-CAM.* ICCV 2017 (3D extension: Chattopadhay 2018).
- Wagener-Ryczek S et al. *External validation of deep-learning HRD in
  PAOLA-1.* European Journal of Cancer 2025.

## License

MIT — see [LICENSE](LICENSE).

## Status

Early. Scaffolding + preprocessing shared with drug-cell-viz ships today;
training pipeline + Med3D transfer + external validation pending data
acquisition. No clinical use.
