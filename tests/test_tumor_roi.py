from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from preprocessing.inference_contract import preprocess_with_saved_contract
from preprocessing.pancreas_crop import (
    crop_from_mask_roi,
    select_crop_roi,
)
from preprocessing.pipeline import preprocess_case


TUMOR_POLICY = {
    "mode": "preferred_label_with_fallback",
    "preferred_label": 2,
    "preferred_name": "tumor",
    "min_preferred_volume_mm3": 100.0,
    "component_policy": "largest",
    "connectivity": 26,
    "fallback_mode": "all_foreground",
    "fallback_name": "pancreas_fallback",
}


def _images(mask: np.ndarray, spacing=(1.0, 1.0, 2.0)):
    affine = np.diag([*spacing, 1.0])
    ct = nib.Nifti1Image(
        np.full(mask.shape, 50.0, dtype=np.float32), affine
    )
    segmentation = nib.Nifti1Image(mask.astype(np.uint8), affine)
    return ct, segmentation


def _write_images(tmp_path, mask):
    ct, segmentation = _images(mask)
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(ct, ct_path)
    nib.save(segmentation, mask_path)
    return ct_path, mask_path


def test_largest_valid_tumor_component_defines_crop():
    mask = np.zeros((24, 28, 16), dtype=np.uint8)
    mask[2:22, 3:25, 2:14] = 1
    mask[7:11, 9:14, 5:8] = 2  # 60 voxels * 2 mm3 = 120 mm3
    mask[18:20, 20:22, 11:12] = 2  # smaller disconnected component
    ct, segmentation = _images(mask)

    crop = crop_from_mask_roi(
        ct,
        segmentation,
        margin_mm=(6.0, 6.0, 6.0),
        roi_policy=TUMOR_POLICY,
    )

    assert crop.roi_source == "tumor"
    assert crop.roi_label == 2
    assert crop.fallback_reason is None
    assert crop.requested_roi_component_count == 2
    assert crop.requested_roi_total_voxel_count == 64
    assert crop.mask_voxel_count == 60
    assert crop.roi_volume_mm3 == 120.0
    assert crop.bbox_original == ((7, 11), (9, 14), (5, 8))
    assert crop.margin_voxels == (6, 6, 3)
    assert crop.bbox_expanded == ((1, 17), (3, 20), (2, 11))


def test_absent_tumor_falls_back_to_all_pancreas_foreground():
    mask = np.zeros((20, 24, 12), dtype=np.uint8)
    mask[4:16, 5:20, 2:10] = 1

    selection = select_crop_roi(
        mask,
        spacing_mm=(1.0, 1.0, 2.0),
        roi_policy=TUMOR_POLICY,
    )

    assert selection.roi_source == "pancreas_fallback"
    assert selection.roi_label is None
    assert selection.fallback_reason == "preferred_label_absent"
    assert selection.roi_voxel_count == int(np.count_nonzero(mask > 0))


def test_tumor_below_100_mm3_falls_back_but_exact_threshold_is_valid():
    mask = np.zeros((20, 24, 12), dtype=np.uint8)
    mask[2:18, 3:21, 1:11] = 1
    mask[6:11, 8:12, 4:6] = 2  # 40 voxels * 2 mm3 = 80 mm3
    below = select_crop_roi(
        mask,
        spacing_mm=(1.0, 1.0, 2.0),
        roi_policy=TUMOR_POLICY,
    )
    assert below.roi_source == "pancreas_fallback"
    assert below.fallback_reason == "largest_component_below_min_volume"
    assert below.requested_roi_largest_component_volume_mm3 == 80.0
    assert below.roi_voxel_count == int(np.count_nonzero(mask > 0))

    mask[6:11, 8:13, 4:6] = 2  # 50 voxels * 2 mm3 = 100 mm3
    exact = select_crop_roi(
        mask,
        spacing_mm=(1.0, 1.0, 2.0),
        roi_policy=TUMOR_POLICY,
    )
    assert exact.roi_source == "tumor"
    assert exact.fallback_reason is None
    assert exact.roi_volume_mm3 == 100.0


def test_saved_contract_reproduces_tumor_preprocessing(tmp_path):
    mask = np.zeros((24, 28, 16), dtype=np.uint8)
    mask[2:22, 3:25, 2:14] = 1
    mask[7:11, 9:14, 5:8] = 2
    ct_path, mask_path = _write_images(tmp_path, mask)
    contract = {
        "target_spacing_mm": {"x": 1.0, "y": 1.0, "z": 2.0},
        "canvas_hw": [32, 32],
        "crop_margin_mm": [6.0, 6.0, 6.0],
        "overflow_policy": "error",
        "patch_size": 16,
        "roi_policy": TUMOR_POLICY,
    }
    config = OmegaConf.create(
        {
            "hu_min": -150.0,
            "hu_max": 250.0,
            "spacing_tolerance_mm": 0.01,
            "alignment": {"resample_mask_to_ct": False},
        }
    )
    training = preprocess_case(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        target_spacing_mm=(1.0, 1.0, 2.0),
        canvas_hw=(32, 32),
        crop_margin_mm=(6.0, 6.0, 6.0),
        roi_policy=TUMOR_POLICY,
    )
    inference = preprocess_with_saved_contract(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        geometry_contract=contract,
        preprocessing_config=config,
    )

    assert np.array_equal(training.volume, inference.volume)
    assert training.geometry == inference.geometry
    assert training.geometry["roi_source"] == "tumor"
    assert training.geometry["roi_label"] == 2
    assert training.geometry["roi_fallback_reason"] is None


def test_tumor_experiment_configs_compose():
    repo_root = Path(__file__).resolve().parents[1]
    preprocessing = OmegaConf.load(
        repo_root
        / "configs/preprocessing/tumor_or_pancreas_fallback_margin6.yaml"
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(repo_root / "configs")
    ):
        config = compose(
            config_name="train",
            overrides=[
                "data=pmpd_v2_tumor_margin6",
                "model=anymc3d_dinov3_vitb_regularized",
            ],
        )

    assert list(preprocessing.crop_margin_mm) == [6.0, 6.0, 6.0]
    assert preprocessing.roi.preferred_label == 2
    assert preprocessing.roi.min_preferred_volume_mm3 == 100.0
    assert preprocessing.roi.component_policy == "largest"
    assert (
        config.data.module.preprocessed_root
        == "data/pmpd_v2_preprocessed_tumor_margin6"
    )
    assert config.model.head_dropout == 0.3
    assert config.model.early_stopping_patience == 30
    assert config.model.early_stopping_min_delta == 0.0

    launcher = (repo_root / "train_pmpd_v2_tumor_two_gpus.sh").read_text()
    assert "model.max_epochs=150" in launcher
    assert "model.early_stopping_patience=150" in launcher
