#!/usr/bin/env python3
"""Train configured folds and aggregate patient-level OOF predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import compute_binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--data-config",
        default="pancreas_metastasis",
        help="Hydra data config name (for example, pmpd_v2).",
    )
    parser.add_argument(
        "--model-config",
        default="anymc3d_dinov3_vitb",
        help="Hydra model config name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pancreas_metastasis"),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides passed to every train.py call.",
    )
    return parser.parse_args()


def aggregate(folds: list[int], output_dir: Path) -> None:
    prediction_frames = []
    fold_metrics = []
    attention_frames = []
    geometry_frames = []
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold}"
        predictions_path = fold_dir / "predictions.csv"
        metrics_path = fold_dir / "metrics.json"
        attention_path = fold_dir / "slice_attention.csv"
        geometry_path = fold_dir / "patient_geometry.csv"
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} outputs are incomplete in {fold_dir}"
            )
        if not attention_path.is_file() or not geometry_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} attention/geometry outputs are incomplete in {fold_dir}"
            )
        prediction_frames.append(pd.read_csv(predictions_path))
        fold_metrics.append(
            {"fold": fold, **json.loads(metrics_path.read_text())}
        )
        attention_frames.append(pd.read_csv(attention_path))
        geometry_frames.append(pd.read_csv(geometry_path))

    oof = pd.concat(prediction_frames, ignore_index=True)
    duplicates = oof.loc[
        oof["patient_id"].duplicated(keep=False), "patient_id"
    ].tolist()
    if duplicates:
        raise ValueError(
            f"OOF patient leakage/duplicates: {duplicates[:20]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    attention = pd.concat(attention_frames, ignore_index=True)
    attention.to_csv(
        output_dir / "oof_slice_attention.csv", index=False
    )
    geometry = pd.concat(geometry_frames, ignore_index=True)
    duplicate_geometry = geometry.loc[
        geometry["patient_id"].duplicated(keep=False), "patient_id"
    ].tolist()
    if duplicate_geometry:
        raise ValueError(
            f"OOF geometry patient duplicates: {duplicate_geometry[:20]}"
        )
    if set(geometry["patient_id"]) != set(oof["patient_id"]):
        raise ValueError("OOF geometry and prediction patient sets differ")
    geometry.to_csv(
        output_dir / "oof_patient_geometry.csv", index=False
    )
    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    numeric_columns = [
        column
        for column in (
            "auroc",
            "pr_auc",
            "brier",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "recall",
            "f1",
        )
        if column in metrics_frame
    ]
    threshold = 0.5
    if "threshold" in metrics_frame:
        threshold = float(metrics_frame["threshold"].median())
    global_metrics = compute_binary_metrics(
        oof["label"], oof["probability"], threshold=threshold
    )
    summary = {
        "folds": folds,
        "per_fold": fold_metrics,
        "mean": {
            column: float(np.nanmean(metrics_frame[column]))
            for column in numeric_columns
        },
        "std": {
            column: float(np.nanstd(metrics_frame[column], ddof=1))
            if len(metrics_frame) > 1
            else 0.0
            for column in numeric_columns
        },
        "oof_global": global_metrics,
        "class_counts": {
            "negative": int((oof["label"] == 0).sum()),
            "positive": int((oof["label"] == 1).sum()),
        },
        "num_slices_per_patient": {
            "min": int(geometry["num_slices"].min()),
            "median": float(geometry["num_slices"].median()),
            "p95": float(geometry["num_slices"].quantile(0.95)),
            "max": int(geometry["num_slices"].max()),
        },
        "attention_weight_distribution": {
            "min": float(attention["attention_weight"].min()),
            "median": float(attention["attention_weight"].median()),
            "p95": float(attention["attention_weight"].quantile(0.95)),
            "p99": float(attention["attention_weight"].quantile(0.99)),
            "max": float(attention["attention_weight"].max()),
        },
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        command = [
            sys.executable,
            "train.py",
            f"data={args.data_config}",
            f"model={args.model_config}",
            f"data.fold={fold}",
            *args.overrides,
            f"model.output_dir={args.output_dir}",
        ]
        subprocess.run(command, check=True)
    aggregate(args.folds, args.output_dir)


if __name__ == "__main__":
    main()
