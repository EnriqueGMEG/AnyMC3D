#!/usr/bin/env python3
"""Plot clean per-epoch validation curves from the five-fold training run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/anymc3d-dinov3-vitb-pmpd-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pmpd_v2_cv"),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=range(1, 6))
    return parser.parse_args()


def _latest_metrics_csv(fold_root: Path) -> Path:
    versions = sorted(
        (fold_root / "lightning_logs").glob("version_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"No Lightning logs found in {fold_root}")
    metrics_path = versions[-1] / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
    return metrics_path


def _best_checkpoint_epoch(fold_root: Path) -> int:
    primary = list(fold_root.glob("best-pr_auc-epoch=*.ckpt"))
    checkpoints = primary or list(fold_root.glob("epoch=*.ckpt"))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected one primary PR-AUC checkpoint in {fold_root}, "
            f"found {len(checkpoints)}"
        )
    return int(checkpoints[0].stem.split("=")[-1])


def load_fold_curve(
    checkpoint_root: Path, fold: int
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Return fit-only validation rows, excluding validate(best) after fit."""

    fold_root = checkpoint_root / f"fold_{fold}"
    metrics_path = _latest_metrics_csv(fold_root)
    metrics = pd.read_csv(metrics_path)
    metric_columns = ["val_auroc", "val_pr_auc", "val_loss"]
    validation = metrics.loc[
        metrics[metric_columns].notna().any(axis=1),
        ["epoch", "step", *metric_columns],
    ].copy()
    if len(validation) < 2:
        raise ValueError(f"Too few validation rows in {metrics_path}")

    # train.py calls trainer.validate(ckpt_path="best") after fit. Lightning
    # appends that revalidation as one extra apparent epoch. It is not a
    # training epoch and must not be drawn as the final point of the curve.
    fit_validation = validation.iloc[:-1].copy()
    final_revalidation = validation.iloc[-1]
    fit_validation["epoch"] = fit_validation["epoch"].astype(int)
    best_checkpoint_epoch = _best_checkpoint_epoch(fold_root)
    checkpoint_row = fit_validation.loc[
        fit_validation["epoch"] == best_checkpoint_epoch
    ]
    if len(checkpoint_row) != 1:
        raise ValueError(
            f"Checkpoint epoch {best_checkpoint_epoch} is missing from "
            f"{metrics_path}"
        )

    best_pr_row = fit_validation.loc[fit_validation["val_pr_auc"].idxmax()]
    best_roc_row = fit_validation.loc[fit_validation["val_auroc"].idxmax()]
    best_loss_row = fit_validation.loc[fit_validation["val_loss"].idxmin()]
    summary = {
        "fold": fold,
        "trained_epochs": len(fit_validation),
        "stop_after_epoch": int(fit_validation["epoch"].iloc[-1]),
        "checkpoint_epoch": best_checkpoint_epoch,
        "checkpoint_val_pr_auc": float(
            checkpoint_row["val_pr_auc"].iloc[0]
        ),
        "checkpoint_val_auroc": float(
            checkpoint_row["val_auroc"].iloc[0]
        ),
        "best_pr_auc_epoch": int(best_pr_row["epoch"]),
        "best_pr_auc": float(best_pr_row["val_pr_auc"]),
        "best_roc_auc_epoch": int(best_roc_row["epoch"]),
        "best_roc_auc": float(best_roc_row["val_auroc"]),
        "best_val_loss_epoch": int(best_loss_row["epoch"]),
        "best_val_loss": float(best_loss_row["val_loss"]),
        "last_fit_pr_auc": float(fit_validation["val_pr_auc"].iloc[-1]),
        "last_fit_roc_auc": float(fit_validation["val_auroc"].iloc[-1]),
        "last_fit_val_loss": float(fit_validation["val_loss"].iloc[-1]),
        "excluded_final_revalidation_epoch_field": int(
            final_revalidation["epoch"]
        ),
        "metrics_csv": str(metrics_path.resolve()),
    }
    fit_validation.insert(0, "fold", fold)
    return fit_validation, summary


def plot_auc_curves(
    curves: list[pd.DataFrame],
    summaries: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = plt.get_cmap("tab10")
    metric_specs = (
        ("val_auroc", "ROC-AUC de validación", 0.5),
        ("val_pr_auc", "PR-AUC de validación", 188 / 479),
    )
    for axis, (column, title, baseline) in zip(axes, metric_specs):
        for index, curve in enumerate(curves):
            fold = int(curve["fold"].iloc[0])
            color = colors(index)
            axis.plot(
                curve["epoch"],
                curve[column],
                marker="o",
                markersize=3,
                linewidth=1.7,
                color=color,
                label=f"Fold {fold}",
            )
            summary = summaries.loc[summaries["fold"] == fold].iloc[0]
            checkpoint_epoch = int(summary["checkpoint_epoch"])
            checkpoint_value = float(
                curve.loc[curve["epoch"] == checkpoint_epoch, column].iloc[0]
            )
            axis.scatter(
                checkpoint_epoch,
                checkpoint_value,
                marker="*",
                s=150,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                int(curve["epoch"].iloc[-1]),
                float(curve[column].iloc[-1]),
                marker="X",
                s=65,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )
        axis.axhline(
            baseline,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Referencia aleatoria",
        )
        axis.set_title(title)
        axis.set_ylabel(column.replace("val_", "").upper())
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Época")
    axes[0].legend(ncol=3, loc="best")
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="silver",
            markeredgecolor="black",
            markersize=13,
            label="Checkpoint elegido por PR-AUC",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="silver",
            markeredgecolor="black",
            markersize=9,
            label="Última época entrenada",
        ),
    ]
    axes[1].legend(handles=marker_legend, loc="best")
    figure.suptitle(
        "PMPD-v2: curvas AUC por fold (sin la revalidación final artificial)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curves(
    curves: list[pd.DataFrame],
    summaries: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    colors = plt.get_cmap("tab10")
    for index, curve in enumerate(curves):
        fold = int(curve["fold"].iloc[0])
        color = colors(index)
        axis.plot(
            curve["epoch"],
            curve["val_loss"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=color,
            label=f"Fold {fold}",
        )
        summary = summaries.loc[summaries["fold"] == fold].iloc[0]
        best_epoch = int(summary["best_val_loss_epoch"])
        best_value = float(
            curve.loc[curve["epoch"] == best_epoch, "val_loss"].iloc[0]
        )
        axis.scatter(
            best_epoch,
            best_value,
            marker="v",
            s=80,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
    axis.set_title(
        "PMPD-v2: val_loss por fold (triángulo = mínimo que reinicia paciencia)"
    )
    axis.set_xlabel("Época")
    axis.set_ylabel("Focal loss de validación")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = []
    summaries = []
    for fold in args.folds:
        curve, summary = load_fold_curve(args.checkpoint_root, fold)
        curves.append(curve)
        summaries.append(summary)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        args.output_dir / "training_curve_summary.csv", index=False
    )
    pd.concat(curves, ignore_index=True).to_csv(
        args.output_dir / "training_curves_clean.csv", index=False
    )
    plot_auc_curves(
        curves, summary_frame, args.output_dir / "auc_learning_curves.png"
    )
    plot_loss_curves(
        curves, summary_frame, args.output_dir / "val_loss_curves.png"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
