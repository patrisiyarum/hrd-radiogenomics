"""Training loop with 5-fold stratified CV + MONAI Trainer idioms.

Trains either Med3D or MONAI DenseNet121 (selected via config) for binary
HRD classification on the preprocessed 96³ volumes. Stratifies on
(hrd_class, scanner_manufacturer) so no single scanner type dominates
one fold — critical for an honest external-validation story.

CLI:
    python -m radiogenomics.train \\
        --manifest data/manifest.parquet \\
        --backbone monai_densenet \\
        --output models/radiogen_v0/ \\
        --epochs 50 --batch-size 4
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from radiogenomics.backbones import load_med3d, load_monai_densenet

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    manifest: Path
    backbone: str = "monai_densenet"     # or "med3d"
    output: Path = Path("models/radiogen_v0")
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    n_folds: int = 5
    seed: int = 0


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    manifest = pd.read_parquet(cfg.manifest)
    # Stratify on scanner manufacturer × HRD class so no single scanner type
    # dominates one fold — critical for honest external validation. The
    # dataset itself loads hrd_class → {0, 1} labels inside VolumeDataset.
    strata = manifest["scanner_manufacturer"].fillna("unknown") + "_" + manifest["hrd_class"]
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_metrics: list[dict] = []
    cfg.output.mkdir(parents=True, exist_ok=True)

    for fold, (train_i, val_i) in enumerate(skf.split(manifest, strata)):
        logger.info("fold %d: %d train / %d val", fold, len(train_i), len(val_i))
        model = _build_model(cfg.backbone).to(device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = _make_loader(manifest.iloc[train_i], cfg.batch_size, shuffle=True)
        val_loader = _make_loader(manifest.iloc[val_i], cfg.batch_size, shuffle=False)

        best_auroc = 0.0
        for epoch in range(cfg.epochs):
            _train_epoch(model, train_loader, opt, loss_fn, device)
            auroc = _eval_auroc(model, val_loader, device)
            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_card": asdict(cfg) | {"fold": fold, "epoch": epoch, "val_auroc": auroc},
                    },
                    cfg.output / f"fold{fold}.pt",
                )
        fold_metrics.append({"fold": fold, "val_auroc": best_auroc})
        logger.info("fold %d best AUROC=%.3f", fold, best_auroc)

    summary = {"folds": fold_metrics, "mean_auroc": float(np.mean([f["val_auroc"] for f in fold_metrics]))}
    (cfg.output / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("CV mean AUROC=%.3f", summary["mean_auroc"])


def _build_model(backbone: str) -> nn.Module:
    if backbone == "monai_densenet":
        return load_monai_densenet(freeze_features=True)
    if backbone == "med3d":
        return load_med3d(freeze_until=3)
    raise ValueError(f"unknown backbone {backbone!r}")


def _make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool):
    # Deferred import so the scaffolding is inspectable without a MONAI install.
    from torch.utils.data import DataLoader

    from radiogenomics.dataset import VolumeDataset

    return DataLoader(VolumeDataset(df), batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, opt, loss_fn, device) -> None:
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


def _eval_auroc(model, loader, device) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_y: list[int] = []
    all_p: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel().tolist()
            all_y.extend(y.tolist())
            all_p.extend(p)
    return float(roc_auc_score(all_y, all_p))


def main() -> None:
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def run(
        manifest: Path,
        backbone: str = "monai_densenet",
        output: Path = Path("models/radiogen_v0"),
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        logging.basicConfig(level=logging.INFO)
        train(TrainConfig(
            manifest=manifest, backbone=backbone, output=output,
            epochs=epochs, batch_size=batch_size, lr=lr,
        ))

    app()


if __name__ == "__main__":
    main()
