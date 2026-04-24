"""External-cohort evaluation: run the best CV checkpoint on a held-out set.

Reads the per-fold checkpoints produced by `rule train_cv`, picks the one
with the best val_auroc, and evaluates it on an external manifest at
`data/external_manifest.parquet` (same schema as the internal one). This
is the honest generalisation test — the only metric that actually
matters for publishing.

External manifest sources (any one is acceptable):
    - a held-out TCIA cohort (e.g. TCGA-CESC or a separate TCGA-OV split)
    - Hartwig Medical Foundation data (DAR-restricted, 3-6 month access)
    - institutional data you already have rights to

If the external manifest is missing, emit a zero-sample report noting
the gap rather than crashing the pipeline — we want `snakemake all` to
succeed for anyone inspecting the scaffolding.
"""


import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from radiogenomics.dataset import VolumeDataset
from radiogenomics.train import _build_model

logger = logging.getLogger("evaluate_external")

def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")

    summary_path = Path(sm.input.summary)
    model_dir = summary_path.parent
    out_path = Path(sm.output.report)
    external_manifest = Path("data/external_manifest.parquet")

    if not external_manifest.exists():
        logger.warning(
            "no external manifest at %s; emitting an empty report so the "
            "pipeline still completes. Drop an external manifest in place "
            "and re-run `snakemake eval_external`.",
            external_manifest,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"n": 0, "note": "no external manifest"}, indent=2))
        return

    # Pick the best fold by val_auroc.
    summary = json.loads(summary_path.read_text())
    best_fold = max(summary["folds"], key=lambda f: f["val_auroc"])
    ckpt_path = model_dir / f"fold{best_fold['fold']}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt["model_card"].get("backbone", "monai_densenet")
    model = _build_model(backbone)
    model.load_state_dict(ckpt["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    ext_df = pd.read_parquet(external_manifest)
    loader = DataLoader(VolumeDataset(ext_df), batch_size=4, shuffle=False)

    all_y: list[int] = []
    all_p: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel().tolist()
            all_y.extend(y.tolist())
            all_p.extend(p)

    metrics = {
        "n": len(all_y),
        "backbone": backbone,
        "best_internal_fold": best_fold["fold"],
        "best_internal_auroc": best_fold["val_auroc"],
        "external_auroc": float(roc_auc_score(all_y, all_p)),
        "external_auprc": float(average_precision_score(all_y, all_p)),
        "auroc_delta": float(roc_auc_score(all_y, all_p) - best_fold["val_auroc"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("external metrics: %s", metrics)

if __name__ == "__main__":
    main()
