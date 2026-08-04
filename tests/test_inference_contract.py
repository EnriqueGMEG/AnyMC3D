from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from conftest import write_nifti_pair
from preprocessing.inference_contract import preprocess_with_saved_contract
from preprocessing.pipeline import preprocess_case


def test_inference_reproduces_training_preprocessing_from_saved_contract(
    tmp_path,
):
    ct_path, mask_path = write_nifti_pair(tmp_path, "case")
    contract = {
        "target_spacing_mm": {"x": 0.8, "y": 0.8, "z": 2.0},
        "canvas_hw": [32, 48],
        "crop_margin_mm": [4.0, 4.0, 4.0],
        "overflow_policy": "error",
        "patch_size": 16,
    }
    config = OmegaConf.create(
        {
            "hu_min": -150.0,
            "hu_max": 250.0,
            "spacing_tolerance_mm": 0.01,
            "alignment": {"resample_mask_to_ct": False},
        }
    )
    training_case = preprocess_case(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        target_spacing_mm=(0.8, 0.8, 2.0),
        canvas_hw=(32, 48),
        crop_margin_mm=(4.0, 4.0, 4.0),
        hu_min=-150.0,
        hu_max=250.0,
        spacing_tolerance_mm=0.01,
        overflow_policy="error",
        resample_mask_to_ct=False,
    )
    inference_case = preprocess_with_saved_contract(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        geometry_contract=contract,
        preprocessing_config=config,
    )
    assert np.array_equal(training_case.volume, inference_case.volume)
    assert np.array_equal(
        training_case.original_slice_indices,
        inference_case.original_slice_indices,
    )
    assert np.array_equal(
        training_case.slice_positions_mm,
        inference_case.slice_positions_mm,
    )
    assert training_case.geometry == inference_case.geometry


def test_inference_refuses_incomplete_geometry_contract(tmp_path):
    ct_path, mask_path = write_nifti_pair(tmp_path, "case")
    config = OmegaConf.create(
        {
            "hu_min": -150.0,
            "hu_max": 250.0,
            "spacing_tolerance_mm": 0.01,
            "alignment": {"resample_mask_to_ct": False},
        }
    )
    with pytest.raises(ValueError, match="missing keys"):
        preprocess_with_saved_contract(
            patient_id="case",
            ct_path=ct_path,
            pancreas_mask_path=mask_path,
            geometry_contract={},
            preprocessing_config=config,
        )
