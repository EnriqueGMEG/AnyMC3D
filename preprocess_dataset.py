#!/usr/bin/env python3
"""Preprocess a validated manifest into variable-slice `.npz` artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from data_modules.manifest import load_and_validate_manifest
from preprocessing.pipeline import preprocess_case

LOGGER = logging.getLogger("preprocess_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/preserve_physical_size.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overflow-policy-override",
        choices=("error", "center_crop"),
        default=None,
        help=(
            "Explicit external-cohort override; the saved contract remains "
            "unchanged and the effective policy is recorded in the run manifest."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_npz(path: Path, case) -> None:
    np.savez_compressed(
        path,
        volume=case.volume.astype(np.float32, copy=False),
        slice_positions_mm=case.slice_positions_mm.astype(np.float32),
        original_slice_indices=case.original_slice_indices.astype(np.int64),
        geometry_json=np.asarray(json.dumps(case.geometry)),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    geometry = json.loads(args.geometry_config.read_text())
    cfg = OmegaConf.load(args.config)
    canvas = geometry["canvas_hw"]
    patch_size = int(geometry["patch_size"])
    if canvas[0] % patch_size or canvas[1] % patch_size:
        raise ValueError(
            f"Persisted canvas {canvas} is not divisible by patch size {patch_size}"
        )
    target = geometry["target_spacing_mm"]
    target_spacing = (target["x"], target["y"], target["z"])
    effective_overflow_policy = (
        args.overflow_policy_override
        if args.overflow_policy_override is not None
        else str(geometry["overflow_policy"])
    )
    intensity = dict(geometry.get("intensity", {"mode": "hu_window"}))
    require_mask = bool(geometry.get("mask_required", True))
    prewindowed_min = float(intensity.get("input_min", 0.0))
    prewindowed_max = float(intensity.get("input_max", 255.0))
    intensity_tolerance = float(intensity.get("range_tolerance", 1.0e-3))
    records = load_and_validate_manifest(
        args.manifest,
        check_nifti_geometry=True,
        resample_mask_to_ct=bool(cfg.alignment.resample_mask_to_ct),
        require_mask=require_mask,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "preprocessing_log.jsonl"
    if log_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{log_path} already exists; pass --overwrite explicitly"
        )

    case_logs = []
    patient_geometry = []
    for row in tqdm(
        records.itertuples(index=False),
        total=len(records),
        desc="Preprocessing",
    ):
        output = args.output_dir / f"{row.patient_id}.npz"
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"patient_id={row.patient_id}: output exists: {output}"
            )
        try:
            case = preprocess_case(
                patient_id=str(row.patient_id),
                ct_path=row.ct_path,
                pancreas_mask_path=getattr(row, "pancreas_mask_path", None),
                target_spacing_mm=target_spacing,
                canvas_hw=canvas,
                crop_margin_mm=geometry["crop_margin_mm"],
                hu_min=float(cfg.hu_min),
                hu_max=float(cfg.hu_max),
                spacing_tolerance_mm=float(cfg.spacing_tolerance_mm),
                overflow_policy=effective_overflow_policy,
                resample_mask_to_ct=bool(
                    cfg.alignment.resample_mask_to_ct
                ),
                roi_policy=geometry.get("roi_policy"),
                intensity_mode=str(intensity.get("mode", "hu_window")),
                prewindowed_min=prewindowed_min,
                prewindowed_max=prewindowed_max,
                intensity_range_tolerance=intensity_tolerance,
            )
            _write_npz(output, case)
            case_logs.append(case.log)
            patient_geometry.append(case.geometry)
        except Exception as exc:
            failure = {
                "patient_id": str(row.patient_id),
                "status": "error",
                "errors": [str(exc)],
            }
            case_logs.append(failure)
            log_path.write_text(
                "".join(json.dumps(entry) + "\n" for entry in case_logs)
            )
            raise RuntimeError(
                f"Preprocessing failed for patient_id={row.patient_id}: {exc}"
            ) from exc

    log_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in case_logs)
    )
    pd.DataFrame(patient_geometry).to_csv(
        args.output_dir / "patient_geometry.csv", index=False
    )
    run_manifest = {
        "schema_version": 2,
        "source_manifest": str(args.manifest.resolve()),
        "geometry_config": str(args.geometry_config.resolve()),
        "num_patients": len(records),
        "intensity": intensity,
        "mask_required": require_mask,
        "mask_used": require_mask,
        "roi_policy": geometry.get("roi_policy"),
        "saved_overflow_policy": str(geometry["overflow_policy"]),
        "effective_overflow_policy": effective_overflow_policy,
        "overflow_policy_was_overridden": (
            args.overflow_policy_override is not None
        ),
        "artifacts": "{patient_id}.npz",
    }
    (args.output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    LOGGER.info("Preprocessed %d patients into %s", len(records), args.output_dir)


if __name__ == "__main__":
    main()
