from pathlib import Path

from hydra import compose, initialize_config_dir


def test_pmpd_v2_training_command_preserves_one_based_fold():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        config = compose(
            config_name="train",
            overrides=[
                "data=pmpd_v2",
                "model=anymc3d_dinov3_vitb",
                "data.fold=5",
            ],
        )

    assert config.data.fold == 5
    assert config.data.module.fold == 5
    assert config.data.module.manifest_path == "data/pmpd_v2_manifest.csv"
    assert (
        config.data.module.preprocessed_root == "data/pmpd_v2_preprocessed"
    )
