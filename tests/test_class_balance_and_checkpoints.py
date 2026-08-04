from types import SimpleNamespace

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from train import compute_pos_weight, metric_mode, snapshot_checkpoint_summary


def test_pos_weight_uses_negative_over_positive_training_counts():
    records = pd.DataFrame({"label": [0, 0, 0, 1, 1]})

    pos_weight, negatives, positives = compute_pos_weight(records)

    assert negatives == 3
    assert positives == 2
    assert pos_weight == pytest.approx(1.5)


def test_pos_weight_rejects_single_class_training_fold():
    with pytest.raises(ValueError, match="positive=0"):
        compute_pos_weight(pd.DataFrame({"label": [0, 0]}))


def test_checkpoint_modes_and_vitl_hydra_config():
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        config = compose(
            config_name="train",
            overrides=["data=pmpd_v2", "model=anymc3d_dinov3_vitl"],
        )

    assert config.model.loss == "bce"
    assert config.model.pos_weight == "auto"
    assert config.model.early_stopping_metric == "val_pr_auc"
    assert config.model.early_stopping_patience == 30
    assert config.model.early_stopping_min_delta == 0.0
    assert config.model.enable_progress_bar is False
    assert config.model.checkpoint_metric == "val_pr_auc"
    assert list(config.model.additional_checkpoint_metrics) == ["val_auroc"]
    assert metric_mode("val_loss") == "min"
    assert metric_mode("val_pr_auc") == "max"
    assert metric_mode("val_auroc") == "max"



def test_checkpoint_summary_is_frozen_before_validation_restores_callbacks():
    callbacks = [
        SimpleNamespace(best_model_path="best-pr.ckpt", best_model_score=0.61),
        SimpleNamespace(best_model_path="best-roc.ckpt", best_model_score=0.72),
    ]

    summary = snapshot_checkpoint_summary(
        ["val_pr_auc", "val_auroc"], callbacks
    )
    callbacks[1].best_model_path = ""
    callbacks[1].best_model_score = None

    assert summary == {
        "val_pr_auc": {"path": "best-pr.ckpt", "score": pytest.approx(0.61)},
        "val_auroc": {"path": "best-roc.ckpt", "score": pytest.approx(0.72)},
    }
