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
    # When True, swap the light v1 augmentation for the v2 stack — adds
    # Gaussian noise / blur / gamma (scanner-style perturbations) and
    # MixUp pairing between batches. Targets cross-fold variance and
    # scanner-generalization, the two main weaknesses of v1.
    strong_augment: bool = False
    # Probability of applying MixUp on each training batch. 0.0 disables.
    # 0.3 with alpha=0.2 is a gentle setting good for small cohorts.
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.2


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
        # Class-weighted BCE so the 71/64 HRD-positive / non-HRD split
        # doesn't let the model cheat by always predicting the majority class.
        train_df = manifest.iloc[train_i]
        n_pos = int((train_df["hrd_class"] == "HRD").sum())
        n_neg = int((train_df["hrd_class"] == "non-HRD").sum())
        pos_weight = torch.tensor(
            [max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float32, device=device,
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader = _make_loader(
            train_df, cfg.batch_size, shuffle=True, augment=True,
            strong_augment=cfg.strong_augment,
        )
        val_loader = _make_loader(manifest.iloc[val_i], cfg.batch_size, shuffle=False, augment=False)

        best_auroc = 0.0
        for epoch in range(cfg.epochs):
            _train_epoch(
                model, train_loader, opt, loss_fn, device,
                mixup_prob=cfg.mixup_prob, mixup_alpha=cfg.mixup_alpha,
            )
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
        # freeze_features=False is correct for the 3D DenseNet — MONAI's
        # 3D DenseNet121 has no PyTorch-Hub pretrained weights, so the
        # trunk is randomly initialised and must train end-to-end.
        return load_monai_densenet(freeze_features=False)
    if backbone == "med3d":
        # freeze_until=2 with the Tencent MedicalNet 23-dataset pretrained
        # weights: freeze the stem + ResNet stages 1-2 (which encode generic
        # CT features that transfer cleanly to ovarian imaging), fine-tune
        # stages 3-4 + classifier on the 108 ovarian patients. With a true
        # held-out test set this is the standard medical-imaging transfer
        # recipe; freeze_until=3 was too aggressive for fine-tuning.
        return load_med3d(freeze_until=2)
    raise ValueError(f"unknown backbone {backbone!r}")


def _make_loader(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    augment: bool = False,
    strong_augment: bool = False,
):
    # Deferred import so the scaffolding is inspectable without a MONAI install.
    from torch.utils.data import DataLoader

    from radiogenomics.dataset import VolumeDataset

    return DataLoader(
        VolumeDataset(df, augment=augment, strong_augment=strong_augment),
        batch_size=batch_size, shuffle=shuffle,
    )


def _train_epoch(
    model, loader, opt, loss_fn, device,
    mixup_prob: float = 0.0, mixup_alpha: float = 0.2,
) -> None:
    """One pass over the loader. When mixup_prob > 0, randomly blend each
    batch with a permuted copy of itself — interpolates both inputs and
    targets by `lam ~ Beta(alpha, alpha)`. Strong regularizer for small
    cohorts; gentle setting (prob=0.3, alpha=0.2) is enough.
    """
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        opt.zero_grad()

        if mixup_prob > 0.0 and float(np.random.rand()) < mixup_prob:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=device)
            x = lam * x + (1.0 - lam) * x[perm]
            y_a, y_b = y, y[perm]
            logits = model(x)
            loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
        else:
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
        strong_augment: bool = False,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.2,
    ) -> None:
        logging.basicConfig(level=logging.INFO)
        train(TrainConfig(
            manifest=manifest, backbone=backbone, output=output,
            epochs=epochs, batch_size=batch_size, lr=lr,
            strong_augment=strong_augment,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
        ))

    app()


if __name__ == "__main__":
    main()
