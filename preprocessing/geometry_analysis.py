"""Dataset-level physical geometry audit and canvas selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .pancreas_crop import (
    body_region,
    canonicalize_ct,
    canonicalize_pair,
    crop_from_mask_roi,
    full_volume_region,
    inspect_image,
    normalize_roi_policy,
)
from .resampling import compute_resampled_shape, physical_span_mm


@dataclass(frozen=True)
class GeometryCase:
    """Per-patient geometry after canonicalization, crop, and resampling."""

    patient_id: str
    ct_path: str
    pancreas_mask_path: str
    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    original_orientation: tuple[str, str, str]
    canonical_shape: tuple[int, int, int]
    canonical_spacing_mm: tuple[float, float, float]
    bbox_original: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    crop_margin_mm: tuple[float, float, float]
    margin_voxels: tuple[int, int, int]
    bbox_expanded: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    crop_shape_before_resampling: tuple[int, int, int]
    target_spacing_mm: tuple[float, float, float]
    spacing_after_resampling_mm: tuple[float, float, float]
    was_resampled: bool
    resampled_shape_xyz: tuple[int, int, int]
    crop_extent_before_mm: tuple[float, float, float]
    crop_extent_after_mm: tuple[float, float, float]
    mask_voxel_count: int
    roi_source: str
    roi_label: int | None
    roi_volume_mm3: float
    requested_roi_label: int | None
    requested_roi_total_voxel_count: int
    requested_roi_total_volume_mm3: float
    requested_roi_component_count: int
    requested_roi_largest_component_voxel_count: int
    requested_roi_largest_component_volume_mm3: float
    roi_fallback_reason: str | None
    mask_was_resampled_to_ct: bool
    warnings: tuple[str, ...] = ()

    @property
    def model_shape_shw(self) -> tuple[int, int, int]:
        x, y, z = self.resampled_shape_xyz
        return z, y, x


def ceil_to_multiple(value: float, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return int(math.ceil(float(value) / divisor) * divisor)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_5": float(np.percentile(array, 99.5)),
        "max": float(np.max(array)),
    }


def resolve_target_spacing(
    records: pd.DataFrame,
    *,
    x: float | str = "auto",
    y: float | str = "auto",
    z: float | str = 2.0,
    resample_mask_to_ct: bool = False,
    roi_policy: Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Resolve auto X/Y spacing to medians without using labels."""

    policy = normalize_roi_policy(roi_policy)
    spacings: list[tuple[float, float, float]] = []
    for row in records.itertuples(index=False):
        if policy["mode"] in ("full_volume", "body"):
            ct, _ = canonicalize_ct(row.ct_path)
        else:
            ct, _, _, _, _ = canonicalize_pair(
                row.ct_path,
                row.pancreas_mask_path,
                resample_mask_to_ct=resample_mask_to_ct,
            )
        spacings.append(tuple(float(v) for v in inspect_image(ct).spacing_mm))
    values = np.asarray(spacings, dtype=float)

    def resolve(value: float | str, axis: int) -> float:
        if isinstance(value, str) and value.lower() == "auto":
            return float(np.median(values[:, axis]))
        result = float(value)
        if result <= 0:
            raise ValueError(f"Target spacing must be positive, got {value}")
        return result

    return resolve(x, 0), resolve(y, 1), resolve(z, 2)


def audit_case_geometry(
    *,
    patient_id: str,
    ct_path: str,
    pancreas_mask_path: str | None,
    target_spacing_mm: Sequence[float],
    crop_margin_mm: Sequence[float],
    resample_mask_to_ct: bool = False,
    spacing_tolerance_mm: float = 0.01,
    roi_policy: Mapping[str, Any] | None = None,
) -> GeometryCase:
    """Audit one patient without reading its classification label."""

    policy = normalize_roi_policy(roi_policy)
    if policy["mode"] in ("full_volume", "body"):
        ct, original_ct = canonicalize_ct(ct_path)
        mask_resampled = False
        if policy["mode"] == "body":
            crop = body_region(
                ct,
                threshold=policy["threshold"],
                margin_mm=crop_margin_mm,
                drop_degenerate_slices=policy["drop_degenerate_slices"],
            )
        else:
            crop = full_volume_region(ct)
        mask_path_value = ""
    else:
        if pancreas_mask_path is None:
            raise ValueError(f"ROI mode {policy['mode']!r} requires a mask")
        ct, mask, original_ct, _, mask_resampled = canonicalize_pair(
            ct_path,
            pancreas_mask_path,
            resample_mask_to_ct=resample_mask_to_ct,
        )
        crop = crop_from_mask_roi(
            ct,
            mask,
            margin_mm=crop_margin_mm,
            roi_policy=policy,
        )
        mask_path_value = str(Path(pancreas_mask_path).resolve())
    canonical = inspect_image(ct)
    target = tuple(float(v) for v in target_spacing_mm)
    spacing_delta = np.abs(np.asarray(crop.spacing_mm) - np.asarray(target))
    was_resampled = not np.all(spacing_delta <= float(spacing_tolerance_mm))
    if was_resampled:
        resampled_shape = compute_resampled_shape(
            crop.shape, crop.spacing_mm, target
        )
        spacing_after = target
    else:
        resampled_shape = crop.shape
        spacing_after = crop.spacing_mm
    before = physical_span_mm(crop.shape, crop.spacing_mm)
    after = physical_span_mm(resampled_shape, spacing_after)
    warnings: list[str] = []
    delta = np.abs(np.asarray(before) - np.asarray(after))
    if np.any(delta > np.asarray(target) / 2 + 1e-6):
        warnings.append(f"physical_span_delta_mm={delta.tolist()}")

    return GeometryCase(
        patient_id=str(patient_id),
        ct_path=str(Path(ct_path).resolve()),
        pancreas_mask_path=mask_path_value,
        original_shape=original_ct.shape,
        original_spacing_mm=original_ct.spacing_mm,
        original_orientation=original_ct.orientation,
        canonical_shape=canonical.shape,
        canonical_spacing_mm=canonical.spacing_mm,
        bbox_original=crop.bbox_original,
        crop_margin_mm=crop.margin_mm,
        margin_voxels=crop.margin_voxels,
        bbox_expanded=crop.bbox_expanded,
        crop_shape_before_resampling=crop.shape,
        target_spacing_mm=target,
        spacing_after_resampling_mm=spacing_after,
        was_resampled=was_resampled,
        resampled_shape_xyz=resampled_shape,
        crop_extent_before_mm=before,
        crop_extent_after_mm=after,
        mask_voxel_count=crop.mask_voxel_count,
        roi_source=crop.roi_source,
        roi_label=crop.roi_label,
        roi_volume_mm3=crop.roi_volume_mm3,
        requested_roi_label=crop.requested_roi_label,
        requested_roi_total_voxel_count=(
            crop.requested_roi_total_voxel_count
        ),
        requested_roi_total_volume_mm3=(
            crop.requested_roi_total_volume_mm3
        ),
        requested_roi_component_count=crop.requested_roi_component_count,
        requested_roi_largest_component_voxel_count=(
            crop.requested_roi_largest_component_voxel_count
        ),
        requested_roi_largest_component_volume_mm3=(
            crop.requested_roi_largest_component_volume_mm3
        ),
        roi_fallback_reason=crop.fallback_reason,
        mask_was_resampled_to_ct=mask_resampled,
        warnings=tuple(warnings),
    )


def canvas_options(
    cases: Sequence[GeometryCase], *, patch_size: int = 16
) -> dict[str, dict[str, object]]:
    """Return percentile/max canvas candidates and memory estimates."""

    h_values = np.asarray([case.model_shape_shw[1] for case in cases])
    w_values = np.asarray([case.model_shape_shw[2] for case in cases])
    s_values = np.asarray([case.model_shape_shw[0] for case in cases])
    options: dict[str, dict[str, object]] = {}
    for name, percentile in (
        ("p95", 95.0),
        ("p99", 99.0),
        ("p99_5", 99.5),
        ("dataset_max", 100.0),
    ):
        raw_h = float(np.max(h_values) if percentile == 100 else np.percentile(h_values, percentile))
        raw_w = float(np.max(w_values) if percentile == 100 else np.percentile(w_values, percentile))
        canvas_h = ceil_to_multiple(raw_h, patch_size)
        canvas_w = ceil_to_multiple(raw_w, patch_size)

        def mib(slices: float, channels: int) -> float:
            return float(slices * canvas_h * canvas_w * channels * 4 / 2**20)

        options[name] = {
            "percentile": percentile,
            "canvas_hw": [canvas_h, canvas_w],
            "patients_exceeding_canvas": int(
                np.sum((h_values > canvas_h) | (w_values > canvas_w))
            ),
            "float32_volume_memory_mib": {
                "median_slices": mib(float(np.median(s_values)), 1),
                "p95_slices": mib(float(np.percentile(s_values, 95)), 1),
                "max_slices": mib(float(np.max(s_values)), 1),
            },
            "float32_rgb_input_memory_mib": {
                "median_slices": mib(float(np.median(s_values)), 3),
                "p95_slices": mib(float(np.percentile(s_values, 95)), 3),
                "max_slices": mib(float(np.max(s_values)), 3),
            },
        }
    return options


def choose_canvas(
    options: dict[str, dict[str, object]],
    *,
    policy: str,
    percentile: float,
) -> tuple[int, int]:
    """Select one persisted canvas from audited options."""

    if policy == "dataset_max":
        selected = options["dataset_max"]["canvas_hw"]
    elif policy == "percentile":
        key = {95.0: "p95", 99.0: "p99", 99.5: "p99_5"}.get(float(percentile))
        if key is None:
            raise ValueError(
                "canvas_percentile must be one of 95, 99, or 99.5"
            )
        selected = options[key]["canvas_hw"]
    else:
        raise ValueError(f"Unknown canvas_policy: {policy}")
    return int(selected[0]), int(selected[1])  # type: ignore[index]


def build_geometry_config(
    cases: Sequence[GeometryCase],
    *,
    target_spacing_mm: Sequence[float],
    crop_margin_mm: Sequence[float],
    canvas_policy: str = "dataset_max",
    canvas_percentile: float = 99.0,
    overflow_policy: str = "error",
    patch_size: int = 16,
    roi_policy: Mapping[str, Any] | None = None,
    intensity_config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Build the JSON-serializable, label-independent geometry contract."""

    if not cases:
        raise ValueError("No valid cases were audited")
    options = canvas_options(cases, patch_size=patch_size)
    canvas = choose_canvas(
        options, policy=canvas_policy, percentile=canvas_percentile
    )
    shapes = [case.model_shape_shw for case in cases]
    policy = normalize_roi_policy(roi_policy)
    return {
        "schema_version": 3,
        "coordinate_convention": {
            "nifti_array": "X,Y,Z canonical RAS",
            "model_volume": "S,1,H,W = Z,1,Y,X",
        },
        "geometry_uses_classification_labels": False,
        "roi_uses_segmentation_labels": policy["mode"]
        not in ("full_volume", "body"),
        "mask_required": policy["mode"] not in ("full_volume", "body"),
        "roi_policy": policy,
        "intensity": dict(intensity_config or {"mode": "hu_window"}),
        "num_cases": len(cases),
        "spacing_mode": "resample_to_common_spacing",
        "target_spacing_mm": {
            "x": float(target_spacing_mm[0]),
            "y": float(target_spacing_mm[1]),
            "z": float(target_spacing_mm[2]),
        },
        "crop_margin_mm": [float(v) for v in crop_margin_mm],
        "patch_size": int(patch_size),
        "canvas_policy": canvas_policy,
        "canvas_percentile": float(canvas_percentile),
        "overflow_policy": overflow_policy,
        "canvas_hw": list(canvas),
        "distributions": {
            "H": _distribution([shape[1] for shape in shapes]),
            "W": _distribution([shape[2] for shape in shapes]),
            "S": _distribution([shape[0] for shape in shapes]),
            "spacing_x_original": _distribution(
                [case.canonical_spacing_mm[0] for case in cases]
            ),
            "spacing_y_original": _distribution(
                [case.canonical_spacing_mm[1] for case in cases]
            ),
            "spacing_z_original": _distribution(
                [case.canonical_spacing_mm[2] for case in cases]
            ),
        },
        "canvas_options": options,
        "implementation_decisions": [
            "Unlike the paper's fixed 432x240x70 T5 input, crops are not resized.",
            "All real post-crop slices are retained after common-spacing resampling.",
            "A shared rectangular in-plane canvas is filled by constant padding.",
        ],
    }


def geometry_cases_to_frame(cases: Iterable[GeometryCase]) -> pd.DataFrame:
    """Flatten dataclass records into a readable CSV table."""

    rows = []
    for case in cases:
        row = asdict(case)
        s, h, w = case.model_shape_shw
        row.update({"S": s, "H": h, "W": w})
        rows.append(row)
    return pd.DataFrame(rows)


def write_geometry_outputs(
    *,
    config: dict[str, object],
    cases: Sequence[GeometryCase],
    output_json: str | Path,
    cases_csv: str | Path,
) -> None:
    """Persist the immutable geometry contract and per-case audit."""

    output_json = Path(output_json)
    cases_csv = Path(cases_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cases_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(config, indent=2) + "\n")
    geometry_cases_to_frame(cases).to_csv(cases_csv, index=False)
