"""Patient-level binary metrics and threshold selection."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def optimize_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    method: str,
) -> float:
    """Select a threshold using validation labels only."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Threshold optimization requires both classes")
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        finite = np.isfinite(thresholds)
        index = np.argmax((tpr - fpr)[finite])
        return float(thresholds[finite][index])
    if method == "f1":
        candidates = np.unique(np.r_[0.0, p, 1.0])
        scores = [
            f1_score(y, p >= threshold, zero_division=0)
            for threshold in candidates
        ]
        return float(candidates[int(np.argmax(scores))])
    raise ValueError(f"Unknown threshold optimization method: {method}")


def compute_binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute all required metrics over one row per patient."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("labels and probabilities must be equal-length vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only 0/1")
    predictions = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    auroc = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    pr_auc = (
        float(average_precision_score(y, p))
        if np.any(y == 1)
        else float("nan")
    )
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y)),
        "n_negative": int(np.sum(y == 0)),
        "n_positive": int(np.sum(y == 1)),
    }


def finite_metric(value: object, fallback: float = 0.0) -> float:
    """Convert NaN/inf metric values into a safe logging fallback."""

    result = float(value)
    return result if math.isfinite(result) else float(fallback)
