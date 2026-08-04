from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import write_nifti_pair
from prepare_dpcg_external_manifest import build_dpcg_manifest
from preprocessing.pipeline import preprocess_case
from validate_external_ensemble import (
    aggregate_majority_vote,
    discover_pr_auc_checkpoints,
)


def test_majority_vote_uses_three_hard_votes_not_mean_probability():
    probabilities = np.asarray(
        [
            [0.51, 0.51, 0.51, 0.00, 0.00],
            [0.99, 0.99, 0.49, 0.49, 0.49],
        ]
    )

    result = aggregate_majority_vote(probabilities, [0.5] * 5)

    assert result["positive_votes"].tolist() == [3, 2]
    assert result["majority_prediction"].tolist() == [1, 0]
    assert result["soft_prediction"].tolist() == [0, 1]
    assert result["vote_fraction"].tolist() == pytest.approx([0.6, 0.4])


def test_best_pr_checkpoint_discovery_is_fold_strict(tmp_path):
    root = tmp_path / "checkpoints"
    for fold, score in ((1, 0.61), (2, 0.72)):
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        checkpoint = fold_dir / f"best-pr_auc-epoch=00{fold}.ckpt"
        checkpoint.touch()
        (fold_dir / "best_checkpoints.json").write_text(
            json.dumps(
                {
                    "val_pr_auc": {
                        "path": str(checkpoint),
                        "score": score,
                    }
                }
            )
        )

    selected = discover_pr_auc_checkpoints(root, [1, 2])

    assert [entry["fold"] for entry in selected] == [1, 2]
    assert [entry["internal_validation_score"] for entry in selected] == [
        0.61,
        0.72,
    ]


def test_dpcg_manifest_uses_metastasis3_and_checks_pairing(tmp_path):
    data_root = tmp_path / "DPCG"
    image_dir = data_root / "images"
    mask_dir = data_root / "nnunet_predicted_masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    for patient_id in ("DPCG_001", "DPCG_002"):
        (image_dir / f"{patient_id}.nii.gz").touch()
        (mask_dir / f"{patient_id}.nii.gz").touch()
    metadata_path = tmp_path / "Meta_dpcg.csv"
    pd.DataFrame(
        {
            "Name": ["DPCG_001", "DPCG_002"],
            "metastasis3": ["No", "Yes"],
        }
    ).to_csv(metadata_path, index=False)
    training_path = tmp_path / "training.csv"
    pd.DataFrame({"patient_id": ["RUM:RUM_001"]}).to_csv(
        training_path, index=False
    )

    manifest = build_dpcg_manifest(
        data_root=data_root,
        metadata_csv=metadata_path,
        training_manifest=training_path,
    )

    assert manifest["patient_id"].tolist() == ["DPCG_001", "DPCG_002"]
    assert manifest["label"].tolist() == [0, 1]
    assert manifest["fold"].tolist() == [0, 0]


def test_preprocess_records_auditable_center_crop_overflow(tmp_path):
    ct_path, mask_path = write_nifti_pair(
        tmp_path,
        "overflow",
        shape=(20, 24, 12),
        spacing=(1.0, 1.0, 2.0),
        mask_bounds=(slice(6, 14), slice(4, 20), slice(3, 8)),
    )
    kwargs = {
        "patient_id": "overflow",
        "ct_path": ct_path,
        "pancreas_mask_path": mask_path,
        "target_spacing_mm": (1.0, 1.0, 2.0),
        "canvas_hw": (16, 32),
        "crop_margin_mm": (4.0, 4.0, 4.0),
    }
    with pytest.raises(ValueError, match="exceeds canvas"):
        preprocess_case(**kwargs, overflow_policy="error")

    case = preprocess_case(**kwargs, overflow_policy="center_crop")

    assert case.volume.shape[1:] == (1, 16, 32)
    assert case.geometry["shape_before_overflow_crop_shw"] == [9, 24, 16]
    assert case.geometry["overflow_center_crop"] == {
        "top": 4,
        "bottom": 4,
        "left": 0,
        "right": 0,
    }
    assert case.geometry["padding_fraction_in_plane"] >= 0.0
    assert case.log["overflow_center_crop"] == case.geometry[
        "overflow_center_crop"
    ]
