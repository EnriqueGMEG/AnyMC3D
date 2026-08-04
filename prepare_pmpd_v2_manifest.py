#!/usr/bin/env python3
"""Build the training manifest from the official PMPD-v2 five-fold split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_modules.manifest import load_and_validate_manifest

SOURCE_DIRECTORIES = {
    "PANGENE_OG": "Pangen_OG",
    "RUM": "RUM",
    "ZZU": "ZZU",
}
LABELS = {"No": 0, "Yes": 1}
REQUIRED_SPLIT_COLUMNS = {
    "fold",
    "patient_key",
    "name",
    "source_dataset",
    "metastasis3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Defaults to <data-root>/5fold_cv_stratified_by_center_location_metastasis.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pmpd_v2_manifest.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _single_nifti(directory: Path, case_name: str) -> Path:
    matches = sorted(directory.glob(f"{case_name}.nii*"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one NIfTI for {case_name} in {directory}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def build_manifest(data_root: Path, split_csv: Path) -> pd.DataFrame:
    """Map each split row to one CT/mask pair without creating new folds."""

    if not data_root.is_dir():
        raise NotADirectoryError(f"PMPD-v2 data root not found: {data_root}")
    if not split_csv.is_file():
        raise FileNotFoundError(f"Five-fold split not found: {split_csv}")

    split = pd.read_csv(split_csv)
    missing = REQUIRED_SPLIT_COLUMNS - set(split.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
    if split.empty:
        raise ValueError("Split CSV contains no patients")
    if split["patient_key"].duplicated().any():
        duplicates = split.loc[
            split["patient_key"].duplicated(keep=False), "patient_key"
        ].tolist()
        raise ValueError(f"Duplicate patient_key values: {duplicates[:20]}")

    sources = set(split["source_dataset"])
    unknown_sources = sources - set(SOURCE_DIRECTORIES)
    if unknown_sources:
        raise ValueError(
            f"Unsupported source_dataset values: {sorted(unknown_sources)}"
        )
    label_values = set(split["metastasis3"])
    unknown_labels = label_values - set(LABELS)
    if unknown_labels:
        raise ValueError(
            f"Unsupported metastasis3 values: {sorted(unknown_labels)}"
        )

    numeric_folds = pd.to_numeric(split["fold"], errors="coerce")
    if numeric_folds.isna().any() or not numeric_folds.isin(range(1, 6)).all():
        raise ValueError("The PMPD-v2 split must contain only folds 1, 2, 3, 4, 5")
    if set(numeric_folds.astype(int)) != set(range(1, 6)):
        raise ValueError("The PMPD-v2 split must contain all five folds")

    records = []
    for row in split.itertuples(index=False):
        cohort_root = data_root / SOURCE_DIRECTORIES[row.source_dataset]
        ct_path = _single_nifti(cohort_root / "images", str(row.name))
        mask_path = _single_nifti(
            cohort_root / "nnunet_predicted_masks", str(row.name)
        )
        records.append(
            {
                "patient_id": str(row.patient_key),
                "ct_path": str(ct_path),
                "pancreas_mask_path": str(mask_path),
                "label": LABELS[row.metastasis3],
                "fold": int(row.fold),
            }
        )

    manifest = pd.DataFrame(records)
    expected_counts = split.groupby("fold").size().sort_index().to_dict()
    actual_counts = manifest.groupby("fold").size().sort_index().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Fold assignments changed: expected {expected_counts}, got {actual_counts}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    split_csv = (
        args.split_csv.resolve()
        if args.split_csv
        else data_root
        / "5fold_cv_stratified_by_center_location_metastasis.csv"
    )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite explicitly"
        )

    manifest = build_manifest(data_root, split_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    validated = load_and_validate_manifest(args.output)
    counts = validated.groupby(["fold", "label"]).size().unstack(fill_value=0)
    print(f"Wrote {len(validated)} patients to {args.output.resolve()}")
    print(counts.rename(columns={0: "negative", 1: "positive"}).to_string())


if __name__ == "__main__":
    main()
