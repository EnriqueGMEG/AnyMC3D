"""Shared training/inference preprocessing for one CT and pancreas mask."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from .pancreas_crop import (
    canonicalize_ct,
    canonicalize_pair,
    crop_from_mask_roi,
    full_volume_region,
    inspect_image,
    normalize_roi_policy,
)
from .resampling import physical_span_mm, resample_image_to_spacing


@dataclass(frozen=True)
class Padding:
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class InPlaneCenterCrop:
    """Pixels removed before padding when a crop exceeds the saved canvas."""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


@dataclass
class PreprocessedCase:
    """Model input and fully traceable physical metadata for one patient."""

    patient_id: str
    volume: np.ndarray
    slice_positions_mm: np.ndarray
    original_slice_indices: np.ndarray
    geometry: dict[str, Any]
    log: dict[str, Any]


def window_ct(
    data_hu: np.ndarray, hu_min: float = -150.0, hu_max: float = 250.0
) -> np.ndarray:
    """Clip HU and linearly map to float32 [0, 1]."""

    if hu_max <= hu_min:
        raise ValueError(f"hu_max must exceed hu_min, got {hu_min}, {hu_max}")
    clipped = np.clip(np.asarray(data_hu, dtype=np.float32), hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)


def normalize_prewindowed_ct(
    data: np.ndarray,
    *,
    input_min: float = 0.0,
    input_max: float = 255.0,
    range_tolerance: float = 1.0e-3,
) -> np.ndarray:
    """Validate an already-windowed CT and scale it once to float32 [0, 1]."""

    if input_max <= input_min:
        raise ValueError(
            f"input_max must exceed input_min, got {input_min}, {input_max}"
        )
    if range_tolerance < 0:
        raise ValueError("range_tolerance cannot be negative")
    values = np.asarray(data, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Prewindowed CT contains NaN or infinite values")
    observed_min = float(values.min())
    observed_max = float(values.max())
    if (
        observed_min < input_min - range_tolerance
        or observed_max > input_max + range_tolerance
    ):
        raise ValueError(
            "Prewindowed CT is outside the declared input range "
            f"[{input_min}, {input_max}]: observed "
            f"[{observed_min}, {observed_max}]"
        )
    clipped = np.clip(values, input_min, input_max)
    return ((clipped - input_min) / (input_max - input_min)).astype(np.float32)


def normalize_ct_intensity(
    data: np.ndarray,
    *,
    mode: str = "hu_window",
    hu_min: float = -150.0,
    hu_max: float = 250.0,
    prewindowed_min: float = 0.0,
    prewindowed_max: float = 255.0,
    range_tolerance: float = 1.0e-3,
) -> np.ndarray:
    """Apply exactly one configured intensity conversion to [0, 1]."""

    if mode == "hu_window":
        return window_ct(data, hu_min=hu_min, hu_max=hu_max)
    if mode == "prewindowed_0_255":
        return normalize_prewindowed_ct(
            data,
            input_min=prewindowed_min,
            input_max=prewindowed_max,
            range_tolerance=range_tolerance,
        )
    raise ValueError(f"Unknown intensity mode: {mode}")


def compute_symmetric_padding(
    height: int, width: int, canvas_hw: Sequence[int]
) -> Padding:
    """Center an HxW rectangle deterministically (extra pixel bottom/right)."""

    canvas_h, canvas_w = (int(canvas_hw[0]), int(canvas_hw[1]))
    if height > canvas_h or width > canvas_w:
        raise ValueError(
            f"Crop {(height, width)} exceeds canvas {(canvas_h, canvas_w)}"
        )
    dh, dw = canvas_h - height, canvas_w - width
    return Padding(
        top=dh // 2,
        bottom=dh - dh // 2,
        left=dw // 2,
        right=dw - dw // 2,
    )


def compute_symmetric_center_crop(
    height: int, width: int, canvas_hw: Sequence[int]
) -> InPlaneCenterCrop:
    """Return deterministic center-crop widths needed to fit a canvas."""

    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    excess_h = max(0, int(height) - canvas_h)
    excess_w = max(0, int(width) - canvas_w)
    return InPlaneCenterCrop(
        top=excess_h // 2,
        bottom=excess_h - excess_h // 2,
        left=excess_w // 2,
        right=excess_w - excess_w // 2,
    )


def apply_in_plane_center_crop(
    volume_s1hw: np.ndarray,
    center_crop: InPlaneCenterCrop,
) -> np.ndarray:
    """Apply a previously calculated in-plane center crop."""

    height_end = (
        volume_s1hw.shape[-2] - center_crop.bottom
        if center_crop.bottom
        else volume_s1hw.shape[-2]
    )
    width_end = (
        volume_s1hw.shape[-1] - center_crop.right
        if center_crop.right
        else volume_s1hw.shape[-1]
    )
    return volume_s1hw[
        :,
        :,
        center_crop.top:height_end,
        center_crop.left:width_end,
    ]


def pad_volume_in_plane(
    volume_s1hw: np.ndarray,
    canvas_hw: Sequence[int],
    *,
    value: float = 0.0,
    overflow_policy: str = "error",
) -> tuple[np.ndarray, Padding]:
    """Constant-pad a variable HxW crop; never resize it."""

    if volume_s1hw.ndim != 4 or volume_s1hw.shape[1] != 1:
        raise ValueError(
            f"Expected volume [S,1,H,W], got {volume_s1hw.shape}"
        )
    _, _, height, width = volume_s1hw.shape
    try:
        padding = compute_symmetric_padding(height, width, canvas_hw)
    except ValueError:
        if overflow_policy != "center_crop":
            raise
        canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
        start_h = max(0, (height - canvas_h) // 2)
        start_w = max(0, (width - canvas_w) // 2)
        volume_s1hw = volume_s1hw[
            :, :, start_h : start_h + canvas_h, start_w : start_w + canvas_w
        ]
        _, _, height, width = volume_s1hw.shape
        padding = compute_symmetric_padding(height, width, canvas_hw)
    padded = np.pad(
        volume_s1hw,
        (
            (0, 0),
            (0, 0),
            (padding.top, padding.bottom),
            (padding.left, padding.right),
        ),
        mode="constant",
        constant_values=float(value),
    )
    return padded.astype(np.float32, copy=False), padding


def _slice_world_positions(
    image: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-Z positions and representative world points per slice."""

    x_center = (image.shape[0] - 1) / 2.0
    y_center = (image.shape[1] - 1) / 2.0
    voxels = np.column_stack(
        (
            np.full(image.shape[2], x_center),
            np.full(image.shape[2], y_center),
            np.arange(image.shape[2], dtype=float),
        )
    )
    world = nib.affines.apply_affine(image.affine, voxels)
    return world[:, 2].astype(np.float32), world


def preprocess_case(
    *,
    patient_id: str,
    ct_path: str | Path,
    pancreas_mask_path: str | Path | None,
    target_spacing_mm: Sequence[float],
    canvas_hw: Sequence[int],
    crop_margin_mm: Sequence[float] = (4.0, 4.0, 4.0),
    hu_min: float = -150.0,
    hu_max: float = 250.0,
    spacing_tolerance_mm: float = 0.01,
    overflow_policy: str = "error",
    resample_mask_to_ct: bool = False,
    roi_policy: Mapping[str, Any] | None = None,
    intensity_mode: str = "hu_window",
    prewindowed_min: float = 0.0,
    prewindowed_max: float = 255.0,
    intensity_range_tolerance: float = 1.0e-3,
) -> PreprocessedCase:
    """Apply the complete physical-size-preserving pipeline to one patient."""

    policy = normalize_roi_policy(roi_policy)
    if policy["mode"] == "full_volume":
        ct, original_ct = canonicalize_ct(ct_path)
        original_mask = None
        mask_resampled = False
        crop = full_volume_region(ct)
    else:
        if pancreas_mask_path is None:
            raise ValueError(
                f"ROI mode {policy['mode']!r} requires pancreas_mask_path"
            )
        ct, mask, original_ct, original_mask, mask_resampled = canonicalize_pair(
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
    canonical_ct = inspect_image(ct)
    crop_before_shape = crop.shape
    crop_before_spacing = crop.spacing_mm
    crop_before_span = physical_span_mm(
        crop_before_shape, crop_before_spacing
    )
    resampled, was_resampled = resample_image_to_spacing(
        crop.image,
        target_spacing_mm,
        tolerance_mm=spacing_tolerance_mm,
        cval=(
            float(prewindowed_min)
            if intensity_mode == "prewindowed_0_255"
            else float(hu_min)
        ),
    )
    resampled_spacing = tuple(
        float(v) for v in nib.affines.voxel_sizes(resampled.affine)
    )
    resampled_shape = tuple(int(v) for v in resampled.shape[:3])
    resampled_span = physical_span_mm(resampled_shape, resampled_spacing)
    data_xyz = resampled.get_fdata(dtype=np.float32)
    observed_intensity_range = [float(data_xyz.min()), float(data_xyz.max())]
    data_01_xyz = normalize_ct_intensity(
        data_xyz,
        mode=intensity_mode,
        hu_min=hu_min,
        hu_max=hu_max,
        prewindowed_min=prewindowed_min,
        prewindowed_max=prewindowed_max,
        range_tolerance=intensity_range_tolerance,
    )
    # NIfTI X,Y,Z -> model S,1,H,W = Z,1,Y,X.
    unpadded = np.transpose(data_01_xyz, (2, 1, 0))[:, np.newaxis, :, :]
    shape_before_overflow_crop = (
        int(unpadded.shape[0]),
        int(unpadded.shape[2]),
        int(unpadded.shape[3]),
    )
    center_crop = compute_symmetric_center_crop(
        unpadded.shape[-2], unpadded.shape[-1], canvas_hw
    )
    if any(asdict(center_crop).values()):
        if overflow_policy != "center_crop":
            raise ValueError(
                f"Crop {tuple(unpadded.shape[-2:])} exceeds canvas "
                f"{tuple(int(value) for value in canvas_hw)}"
            )
        unpadded = apply_in_plane_center_crop(unpadded, center_crop)
    padded, padding = pad_volume_in_plane(
        unpadded,
        canvas_hw,
        value=0.0,
        overflow_policy="error",
    )

    z_positions, world_points = _slice_world_positions(resampled)
    # Map world coordinates back to the raw, pre-canonicalization grid.
    original_raw_inverse = np.linalg.inv(np.asarray(original_ct.affine))
    original_voxels = nib.affines.apply_affine(
        original_raw_inverse, world_points
    )
    original_indices = np.rint(original_voxels[:, 2]).astype(np.int64)
    original_indices = np.clip(original_indices, 0, original_ct.shape[2] - 1)

    _, _, crop_h, crop_w = unpadded.shape
    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    real_fraction = float(crop_h * crop_w / (canvas_h * canvas_w))
    bbox_dims = [
        (hi - lo) * spacing
        for (lo, hi), spacing in zip(
            crop.bbox_original, canonical_ct.spacing_mm
        )
    ]
    bbox_volume = float(np.prod(bbox_dims))
    mask_used = policy["mode"] != "full_volume"
    geometry = {
        "patient_id": str(patient_id),
        "input_region_mode": policy["mode"],
        "mask_used": mask_used,
        "intensity_mode": intensity_mode,
        "physical_crop_dimensions_mm_xyz": [
            float(n * spacing)
            for n, spacing in zip(resampled_shape, resampled_spacing)
        ],
        "model_field_of_view_dimensions_mm_xyz": [
            float(crop_w * resampled_spacing[0]),
            float(crop_h * resampled_spacing[1]),
            float(resampled_shape[2] * resampled_spacing[2]),
        ],
        "resampled_shape_xyz": list(resampled_shape),
        "shape_before_overflow_crop_shw": list(shape_before_overflow_crop),
        "overflow_center_crop": asdict(center_crop),
        "S": int(padded.shape[0]),
        "H": int(padded.shape[2]),
        "W": int(padded.shape[3]),
        "roi_bbox_volume_mm3": bbox_volume,
        "roi_source": crop.roi_source,
        "roi_label": crop.roi_label,
        "roi_voxel_count": crop.mask_voxel_count,
        "roi_volume_mm3": crop.roi_volume_mm3,
        "requested_roi_label": crop.requested_roi_label,
        "requested_roi_total_voxel_count": (
            crop.requested_roi_total_voxel_count
        ),
        "requested_roi_total_volume_mm3": (
            crop.requested_roi_total_volume_mm3
        ),
        "requested_roi_component_count": (
            crop.requested_roi_component_count
        ),
        "requested_roi_largest_component_voxel_count": (
            crop.requested_roi_largest_component_voxel_count
        ),
        "requested_roi_largest_component_volume_mm3": (
            crop.requested_roi_largest_component_volume_mm3
        ),
        "roi_fallback_reason": crop.fallback_reason,
        "real_data_fraction_in_plane": real_fraction,
        "padding_fraction_in_plane": 1.0 - real_fraction,
        "num_slices": int(padded.shape[0]),
        "z_extent_mm": float(
            0.0
            if len(z_positions) < 2
            else abs(float(z_positions[-1] - z_positions[0]))
        ),
        "padding": asdict(padding),
    }
    log = {
        "patient_id": str(patient_id),
        "status": "ok",
        "ct_path": str(Path(ct_path).resolve()),
        "pancreas_mask_path": (
            str(Path(pancreas_mask_path).resolve())
            if pancreas_mask_path is not None and mask_used
            else None
        ),
        "original_ct": asdict(original_ct),
        "original_mask": asdict(original_mask) if original_mask is not None else None,
        "canonical_ct": asdict(canonical_ct),
        "bbox_original": crop.bbox_original,
        "roi_source": crop.roi_source,
        "roi_label": crop.roi_label,
        "roi_voxel_count": crop.mask_voxel_count,
        "roi_volume_mm3": crop.roi_volume_mm3,
        "requested_roi_label": crop.requested_roi_label,
        "requested_roi_total_voxel_count": (
            crop.requested_roi_total_voxel_count
        ),
        "requested_roi_total_volume_mm3": (
            crop.requested_roi_total_volume_mm3
        ),
        "requested_roi_component_count": (
            crop.requested_roi_component_count
        ),
        "requested_roi_largest_component_voxel_count": (
            crop.requested_roi_largest_component_voxel_count
        ),
        "requested_roi_largest_component_volume_mm3": (
            crop.requested_roi_largest_component_volume_mm3
        ),
        "roi_fallback_reason": crop.fallback_reason,
        "crop_margin_mm": crop.margin_mm,
        "margin_voxels": crop.margin_voxels,
        "bbox_expanded": crop.bbox_expanded,
        "crop_shape_before_resampling": crop_before_shape,
        "crop_spacing_before_mm": crop_before_spacing,
        "crop_span_before_mm": crop_before_span,
        "target_spacing_mm": tuple(float(v) for v in target_spacing_mm),
        "was_resampled": was_resampled,
        "mask_was_resampled_to_ct": mask_resampled,
        "shape_after_resampling": resampled_shape,
        "spacing_after_resampling_mm": resampled_spacing,
        "crop_span_after_mm": resampled_span,
        "shape_before_overflow_crop_shw": shape_before_overflow_crop,
        "overflow_center_crop": asdict(center_crop),
        "shape_before_padding_shw": (
            int(unpadded.shape[0]),
            int(unpadded.shape[2]),
            int(unpadded.shape[3]),
        ),
        "padding": asdict(padding),
        "canvas_hw": (canvas_h, canvas_w),
        "real_data_fraction_in_plane": real_fraction,
        "padding_fraction_in_plane": 1.0 - real_fraction,
        "intensity": {
            "mode": intensity_mode,
            "observed_input_range_after_resampling": observed_intensity_range,
            "output_range": [float(data_01_xyz.min()), float(data_01_xyz.max())],
            "hu_window": (
                [float(hu_min), float(hu_max)]
                if intensity_mode == "hu_window"
                else None
            ),
            "prewindowed_range": [float(prewindowed_min), float(prewindowed_max)],
            "range_tolerance": float(intensity_range_tolerance),
        },
        "warnings": [],
        "errors": [],
    }
    return PreprocessedCase(
        patient_id=str(patient_id),
        volume=padded,
        slice_positions_mm=z_positions,
        original_slice_indices=original_indices,
        geometry=geometry,
        log=log,
    )
