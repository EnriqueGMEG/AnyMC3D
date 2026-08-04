"""Callbacks for readable, persistent epoch-level training metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.trainer.states import TrainerFn


METRIC_COLUMNS = (
    "train_loss",
    "train_accuracy",
    "train_auroc",
    "train_pr_auc",
    "val_loss",
    "val_auroc",
    "val_pr_auc",
)


def _scalar(metrics: dict, name: str) -> float:
    if name == "train_loss":
        value = metrics.get("train_loss_epoch")
        if value is None:
            value = metrics.get(name)
    else:
        value = metrics.get(name)
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


class EpochMetricsHistory(L.Callback):
    """Append one stable CSV/log line after every validation epoch."""

    def __init__(self, output_path: str | Path, *, fold: int) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.fold = int(fold)

    def _early_stopping_state(
        self, trainer: L.Trainer
    ) -> tuple[float, int]:
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                best = callback.best_score
                if isinstance(best, torch.Tensor):
                    best = best.detach().cpu().item()
                return float(best), int(callback.wait_count)
        return float("nan"), 0

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        train_metrics = getattr(pl_module, "latest_train_metrics", {})
        row = {"epoch": int(trainer.current_epoch)}
        for name in METRIC_COLUMNS:
            if name in train_metrics:
                row[name] = float(train_metrics[name])
            else:
                row[name] = _scalar(trainer.callback_metrics, name)
        best, wait_count = self._early_stopping_state(trainer)
        row["best_early_stopping_metric"] = best
        row["epochs_without_improvement"] = wait_count

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists()
        with self.output_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()

        values = " ".join(
            f"{name}={row[name]:.4f}" for name in METRIC_COLUMNS
        )
        print(
            f"[fold={self.fold} epoch={row['epoch']:03d}] {values} "
            f"best={best:.4f} wait={wait_count}",
            flush=True,
        )
