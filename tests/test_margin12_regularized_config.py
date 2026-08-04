from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_margin12_data_and_preprocessing_configs():
    repo_root = Path(__file__).resolve().parents[1]
    preprocessing = OmegaConf.load(
        repo_root
        / "configs/preprocessing/preserve_physical_size_margin12.yaml"
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "configs")
    ):
        config = compose(
            config_name="train",
            overrides=[
                "data=pmpd_v2_margin12",
                "model=anymc3d_dinov3_vitb_regularized",
            ],
        )

    assert list(preprocessing.crop_margin_mm) == [12.0, 12.0, 12.0]
    assert (
        config.data.module.preprocessed_root
        == "data/pmpd_v2_preprocessed_margin12"
    )
    assert config.model.head_dropout == 0.3
    assert config.model.early_stopping_patience == 30
    assert config.model.early_stopping_min_delta == 0.0
