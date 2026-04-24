# TRIPOD+AI checklist — radiogenomics v0 (pre-release)

**Status:** pre-release. Scaffolding shipped; training pipeline, external
validation, and Grad-CAM saliency analysis pending. Do not use for any
clinical decision.

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Title + abstract | ✅ | See README.md |
| 2 | Introduction / clinical problem | ✅ | CT-based HRD pre-screen before biopsy + sequencing |
| 3 | Source of data | ✅ | TCGA-OV (GDC) + TCIA (public API) + Hartwig (DAR) |
| 4 | Eligibility criteria | ✅ | Ovarian serous carcinoma with preoperative CT + genomic HRD label |
| 5 | Outcome definition | ✅ | Binary HRD vs non-HRD per Knijnenburg 2018 classification |
| 6 | Predictors / input | ✅ | 96³ HU-windowed CT volume |
| 7 | Sample size | ⏳ | Target: 200-300 TCGA-OV with both imaging + labels, 50+ external |
| 8 | Missing data | ✅ | Patients without matched imaging + labels excluded; scanner metadata imputed |
| 9 | Model development | ✅ | Med3D or MONAI DenseNet121 transfer; freeze + fine-tune last blocks |
| 10 | Model calibration | ⏳ | Platt scaling on held-out fold + isotonic backup if miscalibrated |
| 11 | Training / validation split | ✅ | 5-fold CV stratified on (hrd_class × scanner_manufacturer) |
| 12 | Performance measures | ✅ | AUROC, AUPRC, balanced accuracy, per-scanner AUROC, calibration plot |
| 13 | Model evaluation | ⏳ | Held-out TCIA cohort + Hartwig external |
| 14 | Interpretability | ⏳ | Grad-CAM3D saliency maps per-patient in reports/saliency/ |
| 15 | Fairness / subgroup | ⏳ | Per-scanner, per-hospital, and ancestry where demographics allow |
| 16 | Reproducibility | ✅ | Snakemake + fixed seeds + pinned dep versions + shared preprocess |
| 17 | Risk of bias | ⏳ | PROBAST+AI filled on first release (reports/probast_ai_v0.md) |
| 18 | Clinical use statement | ✅ | Research tier only. Not FDA-cleared. Not for treatment decisions. |
| 19 | Funding + conflicts | ✅ | Self-funded; no conflicts |
| 20 | Registration | ❌ | Not registered; research prototype |
| 21 | Data availability | ✅ | TCGA-OV + TCIA public; Hartwig DAR-restricted |
| 22 | Code availability | ✅ | MIT on GitHub |
| 23 | Model availability | ⏳ | HF Hub release post-training |
| 24 | Supplementary | ⏳ | Per-fold AUROC CSV, calibration PDFs, Grad-CAM overlays per release |
| 25 | Predetermined Change Control Plan | ⏳ | For any future FDA SaMD path |
| 26 | CLAIM 2.0 (imaging-specific) | ⏳ | See reports/claim_v0.md |
| 27 | Limitations | ✅ | ~300 patient pairs small; scanner variability risk; indel-pathogenic HRD mislabelling |
