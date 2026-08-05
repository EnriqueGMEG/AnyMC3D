"""Inference adapter for the immutable training-time geometry contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig

from .pipeline import PreprocessedCase, preprocess_case


def preprocess_with_saved_contract(
    *,
    patient_id: str,
    ct_path: str | Path,
    pancreas_mask_path: str | Path | None,
    geometry_contract: Mapping[str, Any],
    preprocessing_config: DictConfig,
) -> PreprocessedCase:
    """Run exactly the shared pipeline using a persisted geometry contract."""

    required = {
        "target_spacing_mm",
        "canvas_hw",
        "crop_margin_mm",
        "overflow_policy",
        "patch_size",
    }
    missing = required - set(geometry_contract)
    if missing:
        raise ValueError(
            f"Saved geometry contract is missing keys: {sorted(missing)}"
        )
    target = geometry_contract["target_spacing_mm"]
    canvas = tuple(int(value) for value in geometry_contract["canvas_hw"])
    patch_size = int(geometry_contract["patch_size"])
    if canvas[0] % patch_size or canvas[1] % patch_size:
        raise ValueError(
            f"Saved canvas {canvas} is not divisible by patch size {patch_size}"
        )
    intensity = dict(geometry_contract.get("intensity", {"mode": "hu_window"}))

    return preprocess_case(
        patient_id=patient_id,
        ct_path=ct_path,
        pancreas_mask_path=pancreas_mask_path,
        target_spacing_mm=(
            float(target["x"]),
            float(target["y"]),
            float(target["z"]),
        ),
        canvas_hw=canvas,
        crop_margin_mm=geometry_contract["crop_margin_mm"],
        hu_min=float(preprocessing_config.hu_min),
        hu_max=float(preprocessing_config.hu_max),
        spacing_tolerance_mm=float(
            preprocessing_config.spacing_tolerance_mm
        ),
        overflow_policy=str(geometry_contract["overflow_policy"]),
        resample_mask_to_ct=bool(
            preprocessing_config.alignment.resample_mask_to_ct
        ),
        roi_policy=geometry_contract.get("roi_policy"),
        intensity_mode=str(intensity.get("mode", "hu_window")),
        prewindowed_min=float(intensity.get("input_min", 0.0)),
        prewindowed_max=float(intensity.get("input_max", 255.0)),
        intensity_range_tolerance=float(intensity.get("range_tolerance", 1.0e-3)),
    )
