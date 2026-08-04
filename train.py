"""Hydra-driven fold training for both legacy and pancreas AnyMC3D models."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch

from training_callbacks import EpochMetricsHistory


def metric_mode(metric: str) -> str:
    """Infer checkpoint direction without hardcoding one particular metric."""

    return "min" if "loss" in metric.lower() else "max"


def snapshot_checkpoint_summary(metrics, callbacks) -> dict[str, dict]:
    """Freeze best paths/scores before checkpoint-based validation restores state."""

    summary = {}
    for metric, callback in zip(metrics, callbacks):
        score = callback.best_model_score
        summary[str(metric)] = {
            "path": str(callback.best_model_path),
            "score": None if score is None else float(score),
        }
    return summary


def resolve_folds(cfg: DictConfig) -> list[int]:
    fold_value = cfg.data.get("fold", cfg.data.module.get("fold", 0))
    if isinstance(fold_value, (list, ListConfig)):
        return [int(value) for value in fold_value]
    return [int(fold_value)]


def get_datamodule(cfg: DictConfig, fold: int):
    return instantiate(cfg.data.module, fold=fold)


def attach_legacy_augmentation(datamodule, cfg: DictConfig):
    """Retain the old Dataset behavior without affecting the new DataModule."""

    if hasattr(datamodule, "train_transform") and "augment" in cfg.data:
        from data_modules.data_augmentation import apply_augmentation

        apply_augmentation(
            datamodule, augment_train=bool(cfg.data.augment)
        )
    return datamodule


def _model_value(cfg: DictConfig, key: str, default):
    return cfg.model.get(key, default)


def compute_pos_weight(records) -> tuple[float, int, int]:
    """Compute N_negative/N_positive from training patients only."""

    positives = int((records["label"] == 1).sum())
    negatives = int((records["label"] == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Cannot compute pos_weight with positive={positives}, "
            f"negative={negatives}"
        )
    return negatives / positives, negatives, positives


def train_one_fold(cfg: DictConfig, fold: int) -> Path:
    run_name = str(_model_value(cfg, "run_name", "anymc3d"))
    checkpoint_dir = Path("checkpoints") / run_name / f"fold_{fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, checkpoint_dir / "config.yaml")

    L.seed_everything(
        int(_model_value(cfg, "seed", 0)), workers=True
    )
    datamodule = attach_legacy_augmentation(
        get_datamodule(cfg, fold), cfg
    )
    model_overrides = {"fold": fold} if "fold" in cfg.model else {}
    configured_pos_weight = cfg.model.get("pos_weight")
    if configured_pos_weight == "auto":
        if not hasattr(datamodule, "train_records"):
            raise TypeError("pos_weight=auto requires a fold-aware DataModule")
        datamodule.setup(stage="fit")
        pos_weight, negatives, positives = compute_pos_weight(
            datamodule.train_records
        )
        model_overrides["pos_weight"] = pos_weight
        balance = {
            "fold": fold,
            "negative_training_patients": negatives,
            "positive_training_patients": positives,
            "pos_weight": pos_weight,
        }
        (checkpoint_dir / "class_balance.json").write_text(
            json.dumps(balance, indent=2) + "\n"
        )
        print(f"Fold {fold} class balance: {balance}")
    model = instantiate(cfg.model, **model_overrides)
    if hasattr(model, "model") and hasattr(model.model, "parameter_report"):
        report = model.model.parameter_report()
        print(f"Parameter report: {report}")
        print("LoRA modules:")
        for name in model.model.lora_report.all_modules:
            print(f"  {name}")

    checkpoint_metric = str(
        _model_value(cfg, "checkpoint_metric", "val/AUROC")
    )
    additional_metrics = list(
        _model_value(cfg, "additional_checkpoint_metrics", [])
    )
    checkpoint_metrics = list(
        dict.fromkeys([checkpoint_metric, *additional_metrics])
    )
    checkpoint_callbacks = []
    for metric in checkpoint_metrics:
        metric_label = metric.removeprefix("val_").replace("/", "_")
        checkpoint_callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=metric,
                mode=metric_mode(metric),
                save_top_k=int(_model_value(cfg, "save_top_k", 1)),
                filename=f"best-{metric_label}-epoch={{epoch:03d}}",
                auto_insert_metric_name=False,
            )
        )
    early_metric = str(
        _model_value(
            cfg,
            "early_stopping_metric",
            "val_loss" if "_" in checkpoint_metric else "val/loss",
        )
    )
    callbacks = [
        *checkpoint_callbacks,
        EarlyStopping(
            monitor=early_metric,
            mode=metric_mode(early_metric),
            patience=int(
                _model_value(cfg, "early_stopping_patience", 30)
            ),
            min_delta=float(
                _model_value(cfg, "early_stopping_min_delta", 0.0)
            ),
            check_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EpochMetricsHistory(
            checkpoint_dir / "epoch_metrics.csv", fold=fold
        ),
    ]
    logger = CSVLogger(
        save_dir=str(checkpoint_dir), name="lightning_logs"
    )
    trainer = L.Trainer(
        max_epochs=int(_model_value(cfg, "max_epochs", 100)),
        accelerator=_model_value(cfg, "accelerator", "auto"),
        devices=_model_value(cfg, "devices", 1),
        strategy=_model_value(cfg, "strategy", "auto"),
        precision=_model_value(cfg, "precision", "bf16-mixed"),
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        gradient_clip_val=float(
            _model_value(cfg, "gradient_clip_val", 0.0)
        ),
        accumulate_grad_batches=int(
            _model_value(cfg, "accumulate_grad_batches", 1)
        ),
        log_every_n_steps=int(
            _model_value(cfg, "log_every_n_steps", 10)
        ),
        enable_progress_bar=bool(
            _model_value(cfg, "enable_progress_bar", False)
        ),
    )
    trainer.fit(model, datamodule=datamodule)
    # Freeze every callback's best state before loading the primary checkpoint:
    # Lightning restores callback state during validate(), which can otherwise
    # erase later callbacks' best paths from the final JSON summary.
    checkpoint_summary = snapshot_checkpoint_summary(
        checkpoint_metrics, checkpoint_callbacks
    )
    # Re-export predictions/attention from the selected checkpoint, not merely
    # the final epoch.
    trainer.validate(
        model=None,
        datamodule=datamodule,
        ckpt_path=checkpoint_summary[checkpoint_metric]["path"],
    )
    for metric, summary in checkpoint_summary.items():
        print(f"Best {metric} checkpoint: {summary['path']}")
    (checkpoint_dir / "best_checkpoints.json").write_text(
        json.dumps(checkpoint_summary, indent=2) + "\n"
    )
    return Path(checkpoint_callbacks[0].best_model_path)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    folds = resolve_folds(cfg)
    for fold in folds:
        train_one_fold(cfg, fold)


if __name__ == "__main__":
    main()
