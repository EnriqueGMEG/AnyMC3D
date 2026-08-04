#!/usr/bin/env python3
"""Build the independent DPCG external-validation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_modules.manifest import load_and_validate_manifest

LABELS = {"No": 0, "Yes": 1}
REQUIRED_METADATA_COLUMNS = {"Name", "metastasis3"}


def nifti_case_id(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def indexed_niftis(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"NIfTI directory not found: {directory}")
    paths = sorted(directory.glob("*.nii*"))
    result = {nifti_case_id(path): path.resolve() for path in paths}
    if len(result) != len(paths):
        raise ValueError(f"Duplicate NIfTI case IDs in {directory}")
    return result


def build_dpcg_manifest(
    *,
    data_root: Path,
    metadata_csv: Path,
    training_manifest: Path | None = None,
) -> pd.DataFrame:
    """Pair all DPCG cases and map the training target metastasis3 to 0/1."""

    metadata = pd.read_csv(metadata_csv)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(
            f"DPCG metadata is missing columns: {sorted(missing)}"
        )
    if metadata.empty:
        raise ValueError("DPCG metadata contains no cases")
    metadata = metadata.copy()
    metadata["Name"] = metadata["Name"].astype(str).str.strip()
    if metadata["Name"].duplicated().any():
        duplicates = metadata.loc[
            metadata["Name"].duplicated(keep=False), "Name"
        ].tolist()
        raise ValueError(f"Duplicate DPCG Names: {duplicates[:20]}")
    unknown_labels = set(metadata["metastasis3"]) - set(LABELS)
    if unknown_labels:
        raise ValueError(
            f"Unsupported DPCG metastasis3 values: {sorted(unknown_labels)}"
        )

    images = indexed_niftis(data_root / "images")
    masks = indexed_niftis(data_root / "nnunet_predicted_masks")
    names = set(metadata["Name"])
    problems = {
        "metadata_without_image": sorted(names - set(images)),
        "image_without_metadata": sorted(set(images) - names),
        "metadata_without_mask": sorted(names - set(masks)),
        "mask_without_metadata": sorted(set(masks) - names),
    }
    if any(problems.values()):
        raise ValueError(f"DPCG pairing mismatch: {problems}")

    manifest = pd.DataFrame(
        {
            "patient_id": metadata["Name"],
            "ct_path": [str(images[name]) for name in metadata["Name"]],
            "pancreas_mask_path": [
                str(masks[name]) for name in metadata["Name"]
            ],
            "label": metadata["metastasis3"].map(LABELS).astype(int),
            "fold": 0,
        }
    )
    if training_manifest is not None:
        training = pd.read_csv(training_manifest)
        training_ids = set(training["patient_id"].astype(str))
        overlap = training_ids & set(manifest["patient_id"])
        if overlap:
            raise ValueError(
                "Patient leakage between training and external DPCG: "
                f"{sorted(overlap)[:20]}"
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/local/radiomics/PMPD_v2_data/DPCG"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/local/radiomics/PMPD_v2_data/Meta_dpcg.csv"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/pmpd_v2_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dpcg_external_manifest.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists; pass --overwrite explicitly"
        )
    manifest = build_dpcg_manifest(
        data_root=args.data_root.resolve(),
        metadata_csv=args.metadata_csv.resolve(),
        training_manifest=args.training_manifest.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    validated = load_and_validate_manifest(args.output)
    provenance = {
        "cohort": "DPCG",
        "external_validation": True,
        "data_root": str(args.data_root.resolve()),
        "metadata_csv": str(args.metadata_csv.resolve()),
        "training_manifest": str(args.training_manifest.resolve()),
        "target_column": "metastasis3",
        "label_mapping": LABELS,
        "num_patients": int(len(validated)),
        "class_counts": {
            "negative": int((validated["label"] == 0).sum()),
            "positive": int((validated["label"] == 1).sum()),
        },
        "fold_value_is_placeholder_only": 0,
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
