from __future__ import annotations

import json

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from conftest import write_nifti_pair
from data_modules.collate_variable_slices import collate_variable_slices
from data_modules.manifest import load_and_validate_manifest, split_manifest
from data_modules.pancreas_metastasis_dataset import (
    PancreasMetastasisDataModule,
    PancreasMetastasisDataset,
)
from model_arch.pancreas_lightning import (
    PancreasMetastasisLightningModule,
    binary_focal_loss_with_logits,
)
from preprocessing.pipeline import preprocess_case
from training_callbacks import EpochMetricsHistory


def _tiny_backbone() -> DINOv3ViTModel:
    return DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            image_size=32,
            patch_size=16,
            num_register_tokens=2,
            attention_dropout=0.0,
            drop_path_rate=0.0,
        )
    )


def _make_manifest_and_artifacts(tmp_path):
    raw_dir = tmp_path / "raw"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    rows = []
    shapes = [(14, 16, 8), (16, 18, 10), (15, 17, 9), (18, 20, 11)]
    labels = [0, 1, 0, 1]
    folds = [0, 0, 1, 1]
    for index, (shape, label, fold) in enumerate(
        zip(shapes, labels, folds)
    ):
        patient_id = f"patient_{index}"
        ct_path, mask_path = write_nifti_pair(
            raw_dir,
            patient_id,
            shape=shape,
            spacing=(1.0, 1.0, 2.0),
            ct_value=float(index * 20),
            mask_bounds=(
                slice(4, min(10, shape[0] - 1)),
                slice(5, min(12, shape[1] - 1)),
                slice(2, shape[2] - 2),
            ),
        )
        case = preprocess_case(
            patient_id=patient_id,
            ct_path=ct_path,
            pancreas_mask_path=mask_path,
            target_spacing_mm=(1.0, 1.0, 2.0),
            canvas_hw=(32, 32),
            crop_margin_mm=(4.0, 4.0, 4.0),
        )
        np.savez_compressed(
            artifact_dir / f"{patient_id}.npz",
            volume=case.volume,
            slice_positions_mm=case.slice_positions_mm,
            original_slice_indices=case.original_slice_indices,
            geometry_json=np.asarray(json.dumps(case.geometry)),
        )
        rows.append(
            {
                "patient_id": patient_id,
                "ct_path": str(ct_path),
                "pancreas_mask_path": str(mask_path),
                "label": label,
                "fold": fold,
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, artifact_dir


def test_manifest_validation_split_and_leakage_detection(tmp_path):
    manifest_path, _ = _make_manifest_and_artifacts(tmp_path)
    frame = load_and_validate_manifest(manifest_path)
    train, validation = split_manifest(frame, validation_fold=0)
    assert set(train.patient_id).isdisjoint(validation.patient_id)
    assert set(validation.fold) == {0}

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_and_validate_manifest(manifest_path, check_nifti_geometry=False)

    invalid_fold = frame.copy()
    invalid_fold["fold"] = invalid_fold["fold"].astype(float)
    invalid_fold.loc[0, "fold"] = 0.5
    invalid_fold.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="non-negative integers"):
        load_and_validate_manifest(manifest_path, check_nifti_geometry=False)


def test_dataset_variable_slices_collate_and_mask(tmp_path):
    manifest_path, artifact_dir = _make_manifest_and_artifacts(tmp_path)
    frame = load_and_validate_manifest(manifest_path)
    dataset = PancreasMetastasisDataset(
        frame, preprocessed_root=artifact_dir, augmentation_profile="none"
    )
    first, second = dataset[0], dataset[1]
    assert first["volume"].shape[0] != second["volume"].shape[0]
    batch = collate_variable_slices([first, second])
    assert batch["volume"].shape == (2, 10, 1, 32, 32)
    assert batch["slice_mask"].dtype == torch.bool
    assert batch["slice_mask"][0].sum() == 8
    assert batch["slice_mask"][1].sum() == 10
    assert torch.isnan(batch["slice_positions_mm"][0, 8:]).all()
    assert torch.equal(
        batch["original_slice_indices"][0, 8:],
        torch.full((2,), -1),
    )


def test_focal_loss_uses_one_patient_logit():
    logits = torch.tensor([[0.2], [-0.4]], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    loss = binary_focal_loss_with_logits(logits, labels)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_synthetic_end_to_end_training_validation_and_exports(tmp_path):
    manifest_path, artifact_dir = _make_manifest_and_artifacts(tmp_path)
    datamodule = PancreasMetastasisDataModule(
        manifest_path=str(manifest_path),
        preprocessed_root=str(artifact_dir),
        fold=0,
        batch_size=2,
        num_workers=0,
        augmentation_profile="none",
        validate_nifti_on_setup=True,
    )
    output_dir = tmp_path / "outputs"
    model = PancreasMetastasisLightningModule(
        backbone=_tiny_backbone(),
        backbone_name="synthetic-dinov3",
        lora_rank=2,
        lora_alpha=4,
        slice_chunk_size=3,
        loss="focal",
        threshold_mode="fixed",
        fold=0,
        output_dir=str(output_dir),
        precision="32-true",
    )
    history_path = tmp_path / "epoch_metrics.csv"
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        callbacks=[EpochMetricsHistory(history_path, fold=0)],
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule)

    fold_dir = output_dir / "fold_0"
    predictions = pd.read_csv(fold_dir / "predictions.csv")
    attention = pd.read_csv(fold_dir / "slice_attention.csv")
    geometry = pd.read_csv(fold_dir / "patient_geometry.csv")
    metrics = json.loads((fold_dir / "metrics.json").read_text())

    assert list(predictions.columns) == [
        "patient_id",
        "fold",
        "label",
        "logit",
        "probability",
        "prediction",
    ]
    assert len(predictions) == 2
    assert set(metrics) >= {
        "auroc",
        "pr_auc",
        "brier",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    assert len(attention) == int(geometry["num_slices"].sum())
    assert attention["is_valid_slice"].all()
    assert np.allclose(
        attention.groupby("patient_id")["attention_weight"].sum(), 1.0
    )
    assert set(geometry["patient_id"]) == set(predictions["patient_id"])
    assert set(trainer.callback_metrics) >= {
        "train_loss_epoch",
        "train_accuracy",
        "train_auroc",
        "train_pr_auc",
        "val_loss",
        "val_auroc",
        "val_pr_auc",
    }
    history = pd.read_csv(history_path)
    assert len(history) == 1
    assert history.loc[0, "train_loss"] == pytest.approx(
        float(trainer.callback_metrics["train_loss_epoch"])
    )
    assert history[[
        "train_loss",
        "train_accuracy",
        "train_auroc",
        "train_pr_auc",
        "val_loss",
        "val_auroc",
        "val_pr_auc",
    ]].notna().all().all()
    assert list(history.columns) == [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_auroc",
        "train_pr_auc",
        "val_loss",
        "val_auroc",
        "val_pr_auc",
        "best_early_stopping_metric",
        "epochs_without_improvement",
    ]
