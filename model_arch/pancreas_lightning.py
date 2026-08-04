"""Lightning training module for binary patient-level metastasis prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from evaluation import compute_binary_metrics, finite_metric, optimize_threshold
from .anymc3d_dinov3 import AnyMC3DDINOv3


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Standard alpha-balanced binary focal loss operating on logits."""

    targets = targets.float().view_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


class PancreasMetastasisLightningModule(L.LightningModule):
    """One logit and one loss value per patient."""

    def __init__(
        self,
        *,
        backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        slice_chunk_size: int = 8,
        gradient_checkpointing: bool = False,
        head_dropout: float = 0.0,
        loss: str = "focal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        pos_weight: float | None = None,
        lora_lr: float = 1e-4,
        lora_weight_decay: float = 1e-5,
        head_lr: float = 1e-3,
        head_weight_decay: float = 1e-4,
        threshold: float = 0.5,
        threshold_mode: str = "fixed",
        fold: int = 0,
        output_dir: str = "outputs/pancreas_metastasis",
        # Trainer-owned fields are accepted so one Hydra model group remains
        # self-contained; they are not used directly by this module.
        max_epochs: int = 100,
        precision: str = "bf16-mixed",
        seed: int = 0,
        early_stopping_patience: int = 30,
        early_stopping_metric: str = "val_pr_auc",
        early_stopping_min_delta: float = 0.0,
        checkpoint_metric: str = "val_pr_auc",
        additional_checkpoint_metrics: list[str] | None = None,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        devices: int | list[int] | str = 1,
        accelerator: str = "auto",
        strategy: str = "auto",
        log_every_n_steps: int = 10,
        enable_progress_bar: bool = False,
        save_top_k: int = 1,
        run_name: str = "anymc3d-dinov3",
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        if loss not in {"focal", "bce"}:
            raise ValueError(f"loss must be focal or bce, got {loss}")
        if threshold_mode not in {"fixed", "validation_youden", "validation_f1"}:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        self.model = AnyMC3DDINOv3(
            backbone_name=backbone_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            slice_chunk_size=slice_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            head_dropout=head_dropout,
            backbone=backbone,
        )
        self.register_buffer(
            "selected_threshold", torch.tensor(float(threshold))
        )
        self.training_rows: list[tuple[int, float]] = []
        self.training_loss_sum = 0.0
        self.training_patient_count = 0
        self.latest_train_metrics: dict[str, float] = {}
        self.validation_rows: list[dict[str, Any]] = []
        self.validation_attention_rows: list[dict[str, Any]] = []
        self.validation_geometry_rows: list[dict[str, Any]] = []

    def forward(
        self, volume: torch.Tensor, slice_mask: torch.Tensor
    ):
        return self.model(volume, slice_mask)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1, 1).float()
        if self.hparams.loss == "focal":
            return binary_focal_loss_with_logits(
                logits,
                labels,
                gamma=float(self.hparams.focal_gamma),
                alpha=float(self.hparams.focal_alpha),
            )
        pos_weight = (
            None
            if self.hparams.pos_weight is None
            else torch.tensor(
                float(self.hparams.pos_weight),
                device=logits.device,
                dtype=logits.dtype,
            )
        )
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )

    def on_train_epoch_start(self) -> None:
        self.training_rows.clear()
        self.training_loss_sum = 0.0
        self.training_patient_count = 0

    def _finalize_train_metrics(self) -> None:
        if not self.training_rows:
            return
        labels = np.asarray([row[0] for row in self.training_rows], dtype=int)
        probabilities = np.asarray(
            [row[1] for row in self.training_rows], dtype=float
        )
        metrics = compute_binary_metrics(labels, probabilities, threshold=0.5)
        if self.training_patient_count != len(self.training_rows):
            raise RuntimeError("Training loss and prediction counts differ")
        self.latest_train_metrics = {
            "train_loss": self.training_loss_sum / self.training_patient_count
        }
        self.latest_train_metrics.update(
            {
                f"train_{name}": finite_metric(metrics[name])
                for name in ("accuracy", "auroc", "pr_auc")
            }
        )
        for name, value in self.latest_train_metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        output = self(batch["volume"], batch["slice_mask"])
        loss = self._loss(output.logits, batch["label"])
        probabilities = torch.sigmoid(output.logits[:, 0].detach()).float().cpu()
        labels = batch["label"].detach().long().cpu()
        patient_count = int(labels.numel())
        self.training_loss_sum += float(loss.detach().float().cpu()) * patient_count
        self.training_patient_count += patient_count
        self.training_rows.extend(
            (int(label), float(probability))
            for label, probability in zip(labels, probabilities)
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["label"]),
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._finalize_train_metrics()
        self.validation_rows.clear()
        self.validation_attention_rows.clear()
        self.validation_geometry_rows.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        output = self(batch["volume"], batch["slice_mask"])
        loss = self._loss(output.logits, batch["label"])
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["label"]),
        )
        probabilities = torch.sigmoid(output.logits[:, 0])
        for index, patient_id in enumerate(batch["patient_id"]):
            label = int(batch["label"][index].item())
            logit = float(output.logits[index, 0].detach().cpu())
            probability = float(probabilities[index].detach().cpu())
            self.validation_rows.append(
                {
                    "patient_id": str(patient_id),
                    "fold": int(batch["fold"][index].item()),
                    "label": label,
                    "logit": logit,
                    "probability": probability,
                }
            )
            valid_count = int(batch["slice_mask"][index].sum().item())
            attention = output.attention_weights[index, :valid_count].detach().cpu()
            positions = batch["slice_positions_mm"][index, :valid_count].cpu()
            original = batch["original_slice_indices"][
                index, :valid_count
            ].cpu()
            for slice_index in range(valid_count):
                self.validation_attention_rows.append(
                    {
                        "patient_id": str(patient_id),
                        "fold": int(batch["fold"][index].item()),
                        "slice_index": slice_index,
                        "original_slice_index": int(original[slice_index]),
                        "z_position_mm": float(positions[slice_index]),
                        "attention_weight": float(attention[slice_index]),
                        "is_valid_slice": True,
                    }
                )
            geometry = dict(batch["geometry"][index])
            geometry["fold"] = int(batch["fold"][index].item())
            self.validation_geometry_rows.append(geometry)
        return loss

    def _validation_threshold(
        self, labels: np.ndarray, probabilities: np.ndarray
    ) -> float:
        mode = str(self.hparams.threshold_mode)
        if mode == "fixed":
            return float(self.hparams.threshold)
        method = "youden" if mode == "validation_youden" else "f1"
        return optimize_threshold(labels, probabilities, method=method)

    def on_validation_epoch_end(self) -> None:
        if not self.validation_rows:
            return
        frame = pd.DataFrame(self.validation_rows)
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy()
        threshold = self._validation_threshold(labels, probabilities)
        self.selected_threshold.fill_(threshold)
        frame["prediction"] = (probabilities >= threshold).astype(int)
        metrics = compute_binary_metrics(
            labels, probabilities, threshold=threshold
        )
        for name in (
            "auroc",
            "pr_auc",
            "brier",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
        ):
            self.log(
                f"val_{name}",
                finite_metric(metrics[name]),
                prog_bar=name in {"auroc", "pr_auc"},
                sync_dist=False,
            )
        self.log("val_threshold", threshold)

        if getattr(self.trainer, "sanity_checking", False):
            return
        output_dir = (
            Path(self.hparams.output_dir) / f"fold_{int(self.hparams.fold)}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "predictions.csv", index=False)
        pd.DataFrame(self.validation_attention_rows).to_csv(
            output_dir / "slice_attention.csv", index=False
        )
        pd.DataFrame(self.validation_geometry_rows).drop_duplicates(
            subset=["patient_id"]
        ).to_csv(output_dir / "patient_geometry.csv", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )

    def configure_optimizers(self):
        lora_parameters = []
        head_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".lora_A" in name or ".lora_B" in name:
                lora_parameters.append(parameter)
            else:
                head_parameters.append(parameter)
        if not lora_parameters:
            raise RuntimeError("No trainable LoRA parameters found")
        if not head_parameters:
            raise RuntimeError("No trainable task query/head parameters found")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_parameters,
                    "lr": float(self.hparams.lora_lr),
                    "weight_decay": float(self.hparams.lora_weight_decay),
                    "name": "lora",
                },
                {
                    "params": head_parameters,
                    "lr": float(self.hparams.head_lr),
                    "weight_decay": float(self.hparams.head_weight_decay),
                    "name": "query_and_head",
                },
            ]
        )
        return optimizer
