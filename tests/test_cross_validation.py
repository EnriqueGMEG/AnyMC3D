from __future__ import annotations

import json

import pandas as pd

from train_cv import aggregate


def test_cross_validation_aggregation_is_patient_level(tmp_path):
    for fold, patient_id, label, probability, num_slices in (
        (0, "p0", 0, 0.2, 3),
        (1, "p1", 1, 0.8, 4),
    ):
        fold_dir = tmp_path / f"fold_{fold}"
        fold_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "label": label,
                    "logit": -1.4 if label == 0 else 1.4,
                    "probability": probability,
                    "prediction": label,
                }
            ]
        ).to_csv(fold_dir / "predictions.csv", index=False)
        pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "fold": fold,
                    "slice_index": index,
                    "original_slice_index": index,
                    "z_position_mm": index * 2.0,
                    "attention_weight": 1.0 / num_slices,
                    "is_valid_slice": True,
                }
                for index in range(num_slices)
            ]
        ).to_csv(fold_dir / "slice_attention.csv", index=False)
        pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "num_slices": num_slices,
                    "S": num_slices,
                    "H": 32,
                    "W": 32,
                }
            ]
        ).to_csv(fold_dir / "patient_geometry.csv", index=False)
        (fold_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "auroc": 1.0,
                    "pr_auc": 1.0,
                    "brier": 0.04,
                    "accuracy": 1.0,
                    "sensitivity": float(label == 1),
                    "specificity": float(label == 0),
                    "precision": float(label == 1),
                    "recall": float(label == 1),
                    "f1": float(label == 1),
                    "threshold": 0.5,
                }
            )
        )

    aggregate([0, 1], tmp_path)
    oof = pd.read_csv(tmp_path / "oof_predictions.csv")
    geometry = pd.read_csv(tmp_path / "oof_patient_geometry.csv")
    summary = json.loads((tmp_path / "cv_summary.json").read_text())
    assert len(oof) == len(geometry) == 2
    assert summary["class_counts"] == {"negative": 1, "positive": 1}
    assert summary["num_slices_per_patient"]["min"] == 3
    assert "attention_weight_distribution" in summary
    assert summary["oof_global"]["confusion_matrix"] == [[1, 0], [0, 1]]
