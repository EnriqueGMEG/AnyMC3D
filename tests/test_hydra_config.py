from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_requested_training_command_composes_flat_model_config():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        config = compose(
            config_name="train",
            overrides=[
                "data=pancreas_metastasis",
                "model=anymc3d_dinov3",
                "data.fold=0",
            ],
        )
    assert (
        config.model._target_
        == "model_arch.pancreas_lightning.PancreasMetastasisLightningModule"
    )
    assert "model" not in config.model
    assert config.model.fold == 0
    assert config.data.module.fold == 0
