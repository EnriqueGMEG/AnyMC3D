#!/usr/bin/env python3
"""Audit pancreas-crop geometry and persist a shared model canvas."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from preprocessing.geometry_analysis import (
    audit_case_geometry,
    build_geometry_config,
    resolve_target_spacing,
    write_geometry_outputs,
)

LOGGER = logging.getLogger("geometry_audit")
REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "ct_path",
    "pancreas_mask_path",
}


def _nifti_id(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def records_from_directories(ct_dir: Path, mask_dir: Path) -> pd.DataFrame:
    """Pair NIfTI files by exact filename without creating labels/folds."""

    if not ct_dir.is_dir():
        raise NotADirectoryError(f"CT directory not found: {ct_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")
    cts = {p.name: p.resolve() for p in ct_dir.glob("*.nii*")}
    masks = {p.name: p.resolve() for p in mask_dir.glob("*.nii*")}
    missing_masks = sorted(cts.keys() - masks.keys())
    missing_cts = sorted(masks.keys() - cts.keys())
    if missing_masks or missing_cts:
        raise ValueError(
            "Unpaired files: "
            f"CT without mask={missing_masks[:10]}, "
            f"mask without CT={missing_cts[:10]}"
        )
    if not cts:
        raise ValueError("No NIfTI pairs found")
    return pd.DataFrame(
        [
            {
                "patient_id": _nifti_id(Path(name)),
                "ct_path": str(cts[name]),
                "pancreas_mask_path": str(masks[name]),
            }
            for name in sorted(cts)
        ]
    )


def load_records(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest:
        records = pd.read_csv(args.manifest)
        missing = REQUIRED_MANIFEST_COLUMNS - set(records.columns)
        if missing:
            raise ValueError(
                f"Manifest missing required geometry columns: {sorted(missing)}"
            )
        return records
    if not args.ct_dir or not args.mask_dir:
        raise ValueError("Provide --manifest or both --ct-dir and --mask-dir")
    return records_from_directories(Path(args.ct_dir), Path(args.mask_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--ct-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/preserve_physical_size.yaml"),
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("geometry_config.json")
    )
    parser.add_argument(
        "--cases-csv", type=Path, default=Path("geometry_audit_cases.csv")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing geometry contract.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_json} already exists. Reuse it, or pass --overwrite "
            "to make recalculation explicit."
        )
    cfg = OmegaConf.load(args.config)
    roi_policy = (
        OmegaConf.to_container(cfg.roi, resolve=True)
        if "roi" in cfg
        else None
    )
    if roi_policy is not None and not isinstance(roi_policy, dict):
        raise ValueError("preprocessing roi configuration must be a mapping")
    intensity_config = (
        OmegaConf.to_container(cfg.intensity, resolve=True)
        if "intensity" in cfg
        else {"mode": "hu_window"}
    )
    if not isinstance(intensity_config, dict):
        raise ValueError("preprocessing intensity configuration must be a mapping")
    records = load_records(args)
    if records["patient_id"].astype(str).duplicated().any():
        duplicates = records.loc[
            records["patient_id"].astype(str).duplicated(keep=False), "patient_id"
        ].tolist()
        raise ValueError(f"Duplicate patient_id values: {duplicates[:20]}")

    target = resolve_target_spacing(
        records,
        x=cfg.target_spacing_mm.x,
        y=cfg.target_spacing_mm.y,
        z=cfg.target_spacing_mm.z,
        resample_mask_to_ct=bool(cfg.alignment.resample_mask_to_ct),
        roi_policy=roi_policy,
    )
    LOGGER.info("Resolved target spacing (X,Y,Z): %s", target)
    cases = []
    for row in tqdm(
        records.itertuples(index=False),
        total=len(records),
        desc="Auditing geometry",
    ):
        try:
            cases.append(
                audit_case_geometry(
                    patient_id=str(row.patient_id),
                    ct_path=str(row.ct_path),
                    pancreas_mask_path=(
                        str(row.pancreas_mask_path)
                        if roi_policy is None
                        or roi_policy.get("mode") not in ("full_volume", "body")
                        else None
                    ),
                    target_spacing_mm=target,
                    crop_margin_mm=tuple(cfg.crop_margin_mm),
                    resample_mask_to_ct=bool(
                        cfg.alignment.resample_mask_to_ct
                    ),
                    spacing_tolerance_mm=float(cfg.spacing_tolerance_mm),
                    roi_policy=roi_policy,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Geometry audit failed for patient_id={row.patient_id}: {exc}"
            ) from exc

    geometry_config = build_geometry_config(
        cases,
        target_spacing_mm=target,
        crop_margin_mm=tuple(cfg.crop_margin_mm),
        canvas_policy=str(cfg.canvas_policy),
        canvas_percentile=float(cfg.canvas_percentile),
        overflow_policy=str(cfg.overflow_policy),
        patch_size=int(cfg.patch_size),
        roi_policy=roi_policy,
        intensity_config=intensity_config,
    )
    write_geometry_outputs(
        config=geometry_config,
        cases=cases,
        output_json=args.output_json,
        cases_csv=args.cases_csv,
    )
    LOGGER.info("Geometry config: %s", args.output_json.resolve())
    LOGGER.info("Per-case audit: %s", args.cases_csv.resolve())
    LOGGER.info("Selected canvas HxW: %s", geometry_config["canvas_hw"])


if __name__ == "__main__":
    main()
