#!/usr/bin/env python3
"""Evaluate five fold checkpoints on an untouched external cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_modules.collate_variable_slices import collate_variable_slices
from data_modules.manifest import load_and_validate_manifest
from data_modules.pancreas_metastasis_dataset import (
    PancreasMetastasisDataset,
)
from evaluation import compute_binary_metrics
from model_arch.pancreas_lightning import (
    PancreasMetastasisLightningModule,
)


def discover_pr_auc_checkpoints(
    checkpoint_root: Path, folds: Sequence[int]
) -> list[dict[str, object]]:
    """Resolve exactly the persisted best-PR checkpoint for every fold."""

    resolved = []
    for fold in folds:
        summary_path = checkpoint_root / f"fold_{fold}" / "best_checkpoints.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Checkpoint summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if "val_pr_auc" not in summary:
            raise ValueError(f"{summary_path} has no val_pr_auc selection")
        selected = summary["val_pr_auc"]
        checkpoint = Path(str(selected["path"]))
        if not checkpoint.is_absolute():
            checkpoint = (summary_path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Fold {fold} best PR-AUC checkpoint not found: {checkpoint}"
            )
        if checkpoint.parent.name != f"fold_{fold}":
            raise ValueError(
                f"Fold {fold} checkpoint points outside its fold: {checkpoint}"
            )
        resolved.append(
            {
                "fold": int(fold),
                "selection_metric": "val_pr_auc",
                "internal_validation_score": float(selected["score"]),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    return resolved


def aggregate_majority_vote(
    probabilities: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, np.ndarray]:
    """Return hard majority and secondary soft-voting ensemble outputs."""

    values = np.asarray(probabilities, dtype=float)
    limits = np.asarray(thresholds, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"probabilities must have shape [patients, models], got {values.shape}"
        )
    if values.shape[1] != len(limits):
        raise ValueError("One threshold is required per model")
    if values.shape[1] % 2 == 0:
        raise ValueError("Hard majority voting requires an odd model count")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Model probabilities must be finite values in [0, 1]")
    votes = values >= limits[np.newaxis, :]
    required = values.shape[1] // 2 + 1
    positive_votes = votes.sum(axis=1).astype(int)
    return {
        "votes": votes.astype(int),
        "positive_votes": positive_votes,
        "vote_fraction": positive_votes.astype(float) / values.shape[1],
        "majority_prediction": (positive_votes >= required).astype(int),
        "mean_probability": values.mean(axis=1),
        "soft_prediction": (values.mean(axis=1) >= 0.5).astype(int),
    }


def bootstrap_rank_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Patient-level percentile bootstrap intervals for AUROC and PR-AUC."""

    rng = np.random.default_rng(seed)
    aurocs = []
    pr_aucs = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_scores = scores[indices]
        aurocs.append(roc_auc_score(sampled_labels, sampled_scores))
        pr_aucs.append(
            average_precision_score(sampled_labels, sampled_scores)
        )
    if not aurocs:
        raise ValueError("Bootstrap produced no two-class samples")
    return {
        "auroc_95_ci": [
            float(value) for value in np.percentile(aurocs, [2.5, 97.5])
        ],
        "pr_auc_95_ci": [
            float(value) for value in np.percentile(pr_aucs, [2.5, 97.5])
        ],
    }


def plot_external_curves(
    labels: np.ndarray,
    mean_probability: np.ndarray,
    vote_fraction: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for score, label, color in (
        (mean_probability, "Media de probabilidades", "tab:blue"),
        (vote_fraction, "Fracción de votos", "tab:orange"),
    ):
        fpr, tpr, _ = roc_curve(labels, score)
        precision, recall, _ = precision_recall_curve(labels, score)
        axes[0].plot(
            fpr,
            tpr,
            label=f"{label} (AUC={roc_auc_score(labels, score):.3f})",
            color=color,
        )
        axes[1].plot(
            recall,
            precision,
            label=(
                f"{label} "
                f"(AP={average_precision_score(labels, score):.3f})"
            ),
            color=color,
        )
    prevalence = float(np.mean(labels))
    axes[0].plot([0, 1], [0, 1], ":", color="gray")
    axes[1].axhline(prevalence, linestyle=":", color="gray")
    axes[0].set(
        xlabel="1 - especificidad",
        ylabel="Sensibilidad",
        title="ROC externa DPCG",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precisión",
        title="Precision-recall externa DPCG",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def infer_one_checkpoint(
    *,
    checkpoint: Path,
    fold: int,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Run one fold model and return patient probabilities by ID."""

    model = PancreasMetastasisLightningModule.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    if int(model.hparams.fold) != int(fold):
        raise ValueError(
            f"Checkpoint fold mismatch: expected {fold}, got {model.hparams.fold}"
        )
    model.eval().to(device)
    threshold = float(model.selected_threshold.detach().cpu())
    probabilities: dict[str, float] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Checkpoint fold {fold}", leave=False):
            volume = batch["volume"].to(device, non_blocking=True)
            slice_mask = batch["slice_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(volume, slice_mask)
            batch_probabilities = (
                torch.sigmoid(output.logits[:, 0]).float().cpu().numpy()
            )
            for patient_id, probability in zip(
                batch["patient_id"], batch_probabilities
            ):
                if patient_id in probabilities:
                    raise ValueError(f"Duplicate inference ID: {patient_id}")
                probabilities[str(patient_id)] = float(probability)
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.output_dir / "external_predictions.csv"
    if predictions_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{predictions_path} exists; pass --overwrite explicitly"
        )
    if len(args.folds) != 5 or len(set(args.folds)) != 5:
        raise ValueError("External majority voting requires five unique folds")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    records = load_and_validate_manifest(
        args.manifest, check_nifti_geometry=False
    )
    dataset = PancreasMetastasisDataset(
        records,
        preprocessed_root=args.preprocessed_root,
        augmentation_profile="none",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_slices,
    )
    checkpoints = discover_pr_auc_checkpoints(
        args.checkpoint_root, args.folds
    )
    probabilities = []
    thresholds = []
    for entry in checkpoints:
        fold = int(entry["fold"])
        values, threshold = infer_one_checkpoint(
            checkpoint=Path(str(entry["checkpoint"])),
            fold=fold,
            loader=loader,
            device=device,
        )
        missing = set(records["patient_id"]) - set(values)
        extra = set(values) - set(records["patient_id"])
        if missing or extra:
            raise ValueError(
                f"Fold {fold} prediction IDs mismatch: "
                f"missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}"
            )
        probabilities.append(
            np.asarray([values[patient_id] for patient_id in records.patient_id])
        )
        thresholds.append(threshold)
        entry["decision_threshold"] = threshold

    probability_matrix = np.column_stack(probabilities)
    ensemble = aggregate_majority_vote(probability_matrix, thresholds)
    labels = records["label"].to_numpy(dtype=int)
    result = records[["patient_id", "label"]].copy()
    for index, fold in enumerate(args.folds):
        result[f"probability_fold_{fold}"] = probability_matrix[:, index]
        result[f"threshold_fold_{fold}"] = thresholds[index]
        result[f"vote_fold_{fold}"] = ensemble["votes"][:, index]
    result["positive_votes"] = ensemble["positive_votes"]
    result["vote_fraction"] = ensemble["vote_fraction"]
    result["majority_prediction"] = ensemble["majority_prediction"]
    result["mean_probability"] = ensemble["mean_probability"]
    result["soft_prediction"] = ensemble["soft_prediction"]
    if not np.array_equal(
        result["majority_prediction"].to_numpy(),
        (result["positive_votes"].to_numpy() >= 3).astype(int),
    ):
        raise RuntimeError("Majority prediction is inconsistent with votes")

    per_checkpoint = {}
    for index, fold in enumerate(args.folds):
        per_checkpoint[f"fold_{fold}"] = compute_binary_metrics(
            labels,
            probability_matrix[:, index],
            threshold=float(thresholds[index]),
        )
    majority_metrics = compute_binary_metrics(
        labels, ensemble["vote_fraction"], threshold=3.0 / 5.0
    )
    soft_metrics = compute_binary_metrics(
        labels, ensemble["mean_probability"], threshold=0.5
    )
    majority_metrics.update(
        bootstrap_rank_intervals(
            labels,
            ensemble["vote_fraction"],
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    soft_metrics.update(
        bootstrap_rank_intervals(
            labels,
            ensemble["mean_probability"],
            samples=args.bootstrap_samples,
            seed=args.seed + 1,
        )
    )
    metrics = {
        "cohort": "DPCG",
        "external_validation": True,
        "target": "metastasis3",
        "checkpoint_selection": "best internal-validation PR-AUC per fold",
        "num_models": 5,
        "hard_majority_rule": "positive when at least 3 of 5 checkpoints vote positive",
        "no_external_threshold_tuning": True,
        "prevalence": float(labels.mean()),
        "strict_majority_voting": majority_metrics,
        "secondary_soft_voting": soft_metrics,
        "per_checkpoint": per_checkpoint,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(predictions_path, index=False)
    (args.output_dir / "external_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (args.output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoints, indent=2) + "\n"
    )
    plot_external_curves(
        labels,
        ensemble["mean_probability"],
        ensemble["vote_fraction"],
        args.output_dir / "external_roc_pr_curves.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
