"""Strict patient manifest validation shared by preprocessing and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.pancreas_crop import (
    canonicalize_ct,
    canonicalize_pair,
    compute_mask_bbox,
)

MANIFEST_COLUMNS = (
    "patient_id",
    "ct_path",
    "pancreas_mask_path",
    "label",
    "fold",
)


def load_and_validate_manifest(
    manifest_path: str | Path,
    *,
    check_nifti_geometry: bool = True,
    resample_mask_to_ct: bool = False,
    require_mask: bool = True,
) -> pd.DataFrame:
    """Load the configured CSV and fail on every invalid patient."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    required_columns = tuple(
        column for column in MANIFEST_COLUMNS if require_mask or column != "pancreas_mask_path"
    )
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Manifest must contain {required_columns}; missing {sorted(missing)}"
        )
    frame = frame.loc[:, required_columns].copy()
    if frame.empty:
        raise ValueError("Manifest contains no patients")
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id cannot be empty")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    if (frame["patient_id"] == "").any():
        raise ValueError("patient_id cannot be blank")
    if frame["patient_id"].str.contains(r"[/\\]", regex=True).any():
        raise ValueError("patient_id cannot contain path separators")
    duplicates = frame.loc[
        frame["patient_id"].duplicated(keep=False), "patient_id"
    ].tolist()
    if duplicates:
        raise ValueError(f"Duplicate patient_id values: {duplicates[:20]}")

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    invalid_labels = ~numeric_labels.isin([0, 1])
    if invalid_labels.any():
        rows = frame.loc[invalid_labels, ["patient_id", "label"]].to_dict(
            "records"
        )
        raise ValueError(f"Labels must be binary 0/1: {rows[:20]}")
    frame["label"] = numeric_labels.astype(int)
    numeric_folds = pd.to_numeric(frame["fold"], errors="coerce")
    invalid_folds = (
        numeric_folds.isna()
        | (numeric_folds < 0)
        | ~np.isclose(numeric_folds, np.round(numeric_folds))
    )
    if invalid_folds.any():
        rows = frame.loc[invalid_folds, ["patient_id", "fold"]].to_dict(
            "records"
        )
        raise ValueError(f"Folds must be non-negative integers: {rows[:20]}")
    frame["fold"] = numeric_folds.astype(int)

    base = path.parent
    path_columns = ("ct_path", "pancreas_mask_path") if require_mask else ("ct_path",)
    for column in path_columns:
        resolved = []
        for raw in frame[column]:
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved.append(str(candidate.resolve()))
        frame[column] = resolved

    for row in frame.itertuples(index=False):
        if not Path(row.ct_path).is_file():
            raise FileNotFoundError(
                f"patient_id={row.patient_id}: CT not found: {row.ct_path}"
            )
        if require_mask and not Path(row.pancreas_mask_path).is_file():
            raise FileNotFoundError(
                f"patient_id={row.patient_id}: mask not found: "
                f"{row.pancreas_mask_path}"
            )
        if check_nifti_geometry:
            try:
                if require_mask:
                    _, mask, _, _, _ = canonicalize_pair(
                        row.ct_path,
                        row.pancreas_mask_path,
                        resample_mask_to_ct=resample_mask_to_ct,
                    )
                    compute_mask_bbox(mask.get_fdata(dtype=np.float32))
                else:
                    canonicalize_ct(row.ct_path)
            except Exception as exc:
                scope = "CT/mask pair" if require_mask else "CT"
                raise ValueError(
                    f"patient_id={row.patient_id}: invalid {scope}: {exc}"
                ) from exc
    return frame


def split_manifest(
    frame: pd.DataFrame, validation_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split exclusively by patient_id using the provided fold column."""

    validation = frame.loc[frame["fold"] == int(validation_fold)].copy()
    train = frame.loc[frame["fold"] != int(validation_fold)].copy()
    if train.empty or validation.empty:
        raise ValueError(
            f"Fold {validation_fold} yields train={len(train)}, "
            f"validation={len(validation)}"
        )
    leakage = set(train["patient_id"]) & set(validation["patient_id"])
    if leakage:
        raise ValueError(
            f"Patient leakage between train and validation: {sorted(leakage)[:20]}"
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True)
