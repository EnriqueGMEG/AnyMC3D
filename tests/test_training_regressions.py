import json

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from data_modules.pancreas_metastasis_dataset import (
    PancreasMetastasisDataset,
    VolumeAugmenter,
)


def test_hydra_model_accepts_trainer_owned_checkpoint_fields(tiny_dinov3):
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="train",
            overrides=["data=pmpd_v2", "model=anymc3d_dinov3_vitb_regularized"],
        )

    model = instantiate(
        config.model,
        backbone=tiny_dinov3,
        pos_weight=1.5,
    )

    assert model.hparams.early_stopping_metric == "val_pr_auc"
    assert model.hparams.early_stopping_patience == 30
    assert model.hparams.early_stopping_min_delta == 0.0
    assert model.hparams.enable_progress_bar is False
    assert list(model.hparams.additional_checkpoint_metrics) == ["val_auroc"]
    assert model.hparams.pos_weight == 1.5
    assert model.hparams.lora_lr == 3.0e-5
    assert model.hparams.head_lr == 3.0e-4
    assert model.hparams.lora_weight_decay == 1.0e-4
    assert model.hparams.head_weight_decay == 1.0e-3
    assert model.model.classification_dropout.p == 0.3


def test_spatial_augmentation_is_slice_coherent(monkeypatch):
    augmenter = VolumeAugmenter("size_preserving")
    base_slice = torch.zeros(1, 16, 16)
    base_slice[:, 4:12, 6:10] = 1.0
    volume = base_slice.unsqueeze(0).repeat(4, 1, 1, 1)
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.tensor(0.0)
    )

    transformed = augmenter.spatial(
        volume, translation_limit_hw=(4.0, 4.0)
    )

    assert transformed.shape == volume.shape
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0
    assert torch.allclose(transformed[0], transformed[1])
    assert torch.allclose(transformed[1], transformed[2])
    assert torch.allclose(transformed[2], transformed[3])


def test_size_preserving_augmentation_keeps_canvas_padding_zero(
    tmp_path, monkeypatch
):
    patient_id = "case_1"
    volume = np.zeros((2, 1, 8, 10), dtype=np.float32)
    volume[:, :, 2:7, 3:9] = 0.5
    geometry = {
        "padding": {"top": 2, "bottom": 1, "left": 3, "right": 1}
    }
    np.savez_compressed(
        tmp_path / f"{patient_id}.npz",
        volume=volume,
        slice_positions_mm=np.asarray([0.0, 2.0], dtype=np.float32),
        original_slice_indices=np.asarray([0, 1], dtype=np.int64),
        geometry_json=np.asarray(json.dumps(geometry)),
    )
    records = pd.DataFrame(
        [{"patient_id": patient_id, "label": 1, "fold": 1}]
    )
    dataset = PancreasMetastasisDataset(
        records,
        preprocessed_root=tmp_path,
        augmentation_profile="size_preserving",
    )
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        dataset.augment, "spatial", lambda volume, **kwargs: volume
    )

    augmented = dataset[0]["volume"]

    assert torch.count_nonzero(augmented[:, :, :2, :]) == 0
    assert torch.count_nonzero(augmented[:, :, 7:, :]) == 0
    assert torch.count_nonzero(augmented[:, :, :, :3]) == 0
    assert torch.count_nonzero(augmented[:, :, :, 9:]) == 0
    assert torch.count_nonzero(augmented[:, :, 2:7, 3:9]) > 0
