"""NIfTI validation and rectangular anatomical ROI crop construction.

The segmentation is used only to locate a rectangular bounding box. It is
never multiplied with CT intensities or returned as model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage


class AlignmentError(ValueError):
    """Raised when CT and mask do not share the same physical grid."""


@dataclass(frozen=True)
class ImageMetadata:
    """Geometry metadata captured without changing image intensities."""

    path: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]
    affine: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CropResult:
    """A rectangular CT crop and the geometry used to construct it."""

    image: nib.spatialimages.SpatialImage
    bbox_original: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    margin_mm: tuple[float, float, float]
    margin_voxels: tuple[int, int, int]
    bbox_expanded: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    mask_voxel_count: int
    roi_source: str = "all_foreground"
    roi_label: int | None = None
    roi_volume_mm3: float = 0.0
    requested_roi_label: int | None = None
    requested_roi_total_voxel_count: int = 0
    requested_roi_total_volume_mm3: float = 0.0
    requested_roi_component_count: int = 0
    requested_roi_largest_component_voxel_count: int = 0
    requested_roi_largest_component_volume_mm3: float = 0.0
    fallback_reason: str | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.image.shape[:3])

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.image.header.get_zooms()[:3])

    @property
    def physical_extent_mm(self) -> tuple[float, float, float]:
        return tuple(
            float(n) * float(spacing)
            for n, spacing in zip(self.shape, self.spacing_mm)
        )


@dataclass(frozen=True)
class ROISelection:
    """Binary crop locator plus auditable selection metadata."""

    mask: np.ndarray
    roi_source: str
    roi_label: int | None
    roi_voxel_count: int
    roi_volume_mm3: float
    requested_roi_label: int | None
    requested_roi_total_voxel_count: int
    requested_roi_total_volume_mm3: float
    requested_roi_component_count: int
    requested_roi_largest_component_voxel_count: int
    requested_roi_largest_component_volume_mm3: float
    fallback_reason: str | None


def normalize_roi_policy(
    roi_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize the persisted rectangular-ROI policy."""

    if roi_policy is None:
        return {
            "mode": "all_foreground",
            "source_name": "all_foreground",
        }
    policy = dict(roi_policy)
    mode = str(policy.get("mode", "all_foreground"))
    if mode == "all_foreground":
        return {
            "mode": mode,
            "source_name": str(policy.get("source_name", "all_foreground")),
        }
    if mode != "preferred_label_with_fallback":
        raise ValueError(f"Unknown ROI mode: {mode}")

    preferred_label = int(policy.get("preferred_label", 2))
    min_volume = float(policy.get("min_preferred_volume_mm3", 100.0))
    if min_volume < 0:
        raise ValueError(
            "min_preferred_volume_mm3 cannot be negative: "
            f"{min_volume}"
        )
    component_policy = str(policy.get("component_policy", "largest"))
    if component_policy != "largest":
        raise ValueError(
            f"Unsupported component_policy: {component_policy}"
        )
    connectivity = int(policy.get("connectivity", 26))
    if connectivity != 26:
        raise ValueError(
            "Only 26-neighbour 3D connectivity is supported, got "
            f"{connectivity}"
        )
    fallback_mode = str(policy.get("fallback_mode", "all_foreground"))
    if fallback_mode != "all_foreground":
        raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")
    return {
        "mode": mode,
        "preferred_label": preferred_label,
        "preferred_name": str(policy.get("preferred_name", "tumor")),
        "min_preferred_volume_mm3": min_volume,
        "component_policy": component_policy,
        "connectivity": connectivity,
        "fallback_mode": fallback_mode,
        "fallback_name": str(
            policy.get("fallback_name", "pancreas_fallback")
        ),
    }


def select_crop_roi(
    mask_data: np.ndarray,
    *,
    spacing_mm: Sequence[float],
    roi_policy: Mapping[str, Any] | None = None,
) -> ROISelection:
    """Select the binary locator used to construct the rectangular CT crop."""

    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got {mask_data.shape}")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing: {spacing}")
    voxel_volume_mm3 = float(np.prod(spacing))
    policy = normalize_roi_policy(roi_policy)

    if policy["mode"] == "all_foreground":
        selected = np.asarray(mask_data > 0, dtype=bool)
        voxel_count = int(np.count_nonzero(selected))
        if voxel_count == 0:
            raise ValueError("Pancreas mask is empty")
        return ROISelection(
            mask=selected,
            roi_source=str(policy["source_name"]),
            roi_label=None,
            roi_voxel_count=voxel_count,
            roi_volume_mm3=voxel_count * voxel_volume_mm3,
            requested_roi_label=None,
            requested_roi_total_voxel_count=0,
            requested_roi_total_volume_mm3=0.0,
            requested_roi_component_count=0,
            requested_roi_largest_component_voxel_count=0,
            requested_roi_largest_component_volume_mm3=0.0,
            fallback_reason=None,
        )

    requested_label = int(policy["preferred_label"])
    requested = np.isclose(mask_data, requested_label)
    requested_total_count = int(np.count_nonzero(requested))
    structure = np.ones((3, 3, 3), dtype=bool)
    components, component_count = ndimage.label(
        requested, structure=structure
    )
    largest_count = 0
    largest_mask = np.zeros(mask_data.shape, dtype=bool)
    if component_count:
        counts = np.bincount(components.ravel())
        counts[0] = 0
        largest_index = int(np.argmax(counts))
        largest_count = int(counts[largest_index])
        largest_mask = components == largest_index

    largest_volume = largest_count * voxel_volume_mm3
    minimum = float(policy["min_preferred_volume_mm3"])
    fallback_reason: str | None
    if requested_total_count == 0:
        fallback_reason = "preferred_label_absent"
    elif largest_volume < minimum:
        fallback_reason = "largest_component_below_min_volume"
    else:
        fallback_reason = None

    if fallback_reason is None:
        selected = largest_mask
        source = str(policy["preferred_name"])
        selected_label: int | None = requested_label
    else:
        selected = np.asarray(mask_data > 0, dtype=bool)
        source = str(policy["fallback_name"])
        selected_label = None
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        raise ValueError(
            "Segmentation is empty, so neither preferred ROI nor fallback "
            "foreground can define a crop"
        )
    return ROISelection(
        mask=selected,
        roi_source=source,
        roi_label=selected_label,
        roi_voxel_count=selected_count,
        roi_volume_mm3=selected_count * voxel_volume_mm3,
        requested_roi_label=requested_label,
        requested_roi_total_voxel_count=requested_total_count,
        requested_roi_total_volume_mm3=(
            requested_total_count * voxel_volume_mm3
        ),
        requested_roi_component_count=int(component_count),
        requested_roi_largest_component_voxel_count=largest_count,
        requested_roi_largest_component_volume_mm3=largest_volume,
        fallback_reason=fallback_reason,
    )


def _shape3(image: nib.spatialimages.SpatialImage) -> tuple[int, int, int]:
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {image.shape}")
    return tuple(int(v) for v in image.shape)


def inspect_image(
    image_or_path: nib.spatialimages.SpatialImage | str | Path,
) -> ImageMetadata:
    """Return shape, spacing, orientation, and affine for a NIfTI image."""

    image = (
        nib.load(str(image_or_path))
        if isinstance(image_or_path, (str, Path))
        else image_or_path
    )
    path = str(image_or_path) if isinstance(image_or_path, (str, Path)) else ""
    shape = _shape3(image)
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(image.affine))
    orientation = tuple(str(v) for v in nib.aff2axcodes(image.affine))
    affine = tuple(tuple(float(v) for v in row) for row in image.affine)
    return ImageMetadata(path, shape, spacing, orientation, affine)


def validate_pair_alignment(
    ct: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    *,
    affine_atol: float = 1e-4,
    spacing_atol: float = 1e-4,
) -> None:
    """Validate that CT and mask occupy exactly the same voxel grid."""

    ct_meta = inspect_image(ct)
    mask_meta = inspect_image(mask)
    errors: list[str] = []
    if ct_meta.shape != mask_meta.shape:
        errors.append(f"shape CT={ct_meta.shape}, mask={mask_meta.shape}")
    if ct_meta.orientation != mask_meta.orientation:
        errors.append(
            f"orientation CT={ct_meta.orientation}, mask={mask_meta.orientation}"
        )
    if not np.allclose(
        ct_meta.spacing_mm, mask_meta.spacing_mm, atol=spacing_atol, rtol=0
    ):
        errors.append(
            f"spacing CT={ct_meta.spacing_mm}, mask={mask_meta.spacing_mm}"
        )
    if not np.allclose(ct.affine, mask.affine, atol=affine_atol, rtol=0):
        max_delta = float(np.max(np.abs(ct.affine - mask.affine)))
        errors.append(f"affine max_abs_delta={max_delta:.6g}")
    if errors:
        raise AlignmentError("CT/mask grid mismatch: " + "; ".join(errors))


def canonicalize_pair(
    ct_path: str | Path,
    mask_path: str | Path,
    *,
    resample_mask_to_ct: bool = False,
    affine_atol: float = 1e-4,
    spacing_atol: float = 1e-4,
) -> tuple[
    nib.spatialimages.SpatialImage,
    nib.spatialimages.SpatialImage,
    ImageMetadata,
    ImageMetadata,
    bool,
]:
    """Load, canonicalize to RAS, and strictly align a CT/mask pair.

    When ``resample_mask_to_ct`` is explicitly enabled, a misaligned mask is
    resampled onto the CT grid using nearest-neighbor interpolation.
    """

    ct_path = Path(ct_path)
    mask_path = Path(mask_path)
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT not found: {ct_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Pancreas mask not found: {mask_path}")

    ct_raw = nib.load(str(ct_path))
    mask_raw = nib.load(str(mask_path))
    ct_original = inspect_image(ct_raw)
    mask_original = inspect_image(mask_raw)
    _shape3(ct_raw)
    _shape3(mask_raw)

    try:
        validate_pair_alignment(
            ct_raw, mask_raw, affine_atol=affine_atol, spacing_atol=spacing_atol
        )
    except AlignmentError as exc:
        if not resample_mask_to_ct:
            raise AlignmentError(f"CT/mask mismatch before canonicalization: {exc}") from exc

    ct = nib.as_closest_canonical(ct_raw)
    mask = nib.as_closest_canonical(mask_raw)
    mask_was_resampled = False
    try:
        validate_pair_alignment(
            ct, mask, affine_atol=affine_atol, spacing_atol=spacing_atol
        )
    except AlignmentError:
        if not resample_mask_to_ct:
            raise
        mask = resample_from_to(mask, (ct.shape, ct.affine), order=0, mode="constant")
        mask_was_resampled = True
        validate_pair_alignment(
            ct, mask, affine_atol=affine_atol, spacing_atol=spacing_atol
        )

    return ct, mask, ct_original, mask_original, mask_was_resampled


def compute_mask_bbox(
    mask_data: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    int,
]:
    """Return an inclusive-exclusive 3D bbox around all mask values > 0."""

    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got {mask_data.shape}")
    foreground = np.argwhere(mask_data > 0)
    if foreground.size == 0:
        raise ValueError("Pancreas mask is empty")
    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0) + 1
    bbox = tuple((int(lo), int(hi)) for lo, hi in zip(lower, upper))
    return bbox, int(foreground.shape[0])


def margin_mm_to_voxels(
    margin_mm: Sequence[float],
    spacing_mm: Sequence[float],
) -> tuple[int, int, int]:
    """Convert independent physical margins to voxels using ceil."""

    if len(margin_mm) != 3:
        raise ValueError(f"crop_margin_mm must have three values, got {margin_mm}")
    values = tuple(float(v) for v in margin_mm)
    if any(v < 0 for v in values):
        raise ValueError(f"crop_margin_mm cannot be negative: {values}")
    return tuple(
        int(np.ceil(margin / spacing))
        for margin, spacing in zip(values, spacing_mm)
    )


def expand_bbox(
    bbox: Sequence[Sequence[int]],
    margin_voxels: Sequence[int],
    shape: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Expand an inclusive-exclusive bbox and clamp it to the image."""

    expanded = []
    for (lo, hi), margin, size in zip(bbox, margin_voxels, shape):
        expanded.append((max(0, int(lo) - int(margin)), min(int(size), int(hi) + int(margin))))
    return tuple(expanded)  # type: ignore[return-value]


def crop_from_pancreas_mask(
    ct: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    *,
    margin_mm: Sequence[float] = (4.0, 4.0, 4.0),
) -> CropResult:
    """Crop unmasked CT to the expanded rectangular pancreas bbox."""

    return crop_from_mask_roi(
        ct,
        mask,
        margin_mm=margin_mm,
        roi_policy=None,
    )


def crop_from_mask_roi(
    ct: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    *,
    margin_mm: Sequence[float] = (4.0, 4.0, 4.0),
    roi_policy: Mapping[str, Any] | None = None,
) -> CropResult:
    """Crop unmasked CT around a selected segmentation-derived ROI."""

    validate_pair_alignment(ct, mask)
    mask_data = np.asanyarray(mask.dataobj)
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(ct.affine))
    selection = select_crop_roi(
        mask_data,
        spacing_mm=spacing,
        roi_policy=roi_policy,
    )
    bbox, mask_voxel_count = compute_mask_bbox(selection.mask)
    margin_tuple = tuple(float(v) for v in margin_mm)
    margin_voxels = margin_mm_to_voxels(margin_tuple, spacing)
    expanded = expand_bbox(bbox, margin_voxels, ct.shape[:3])
    slices = tuple(slice(lo, hi) for lo, hi in expanded)
    cropped = ct.slicer[slices]
    return CropResult(
        image=cropped,
        bbox_original=bbox,
        margin_mm=margin_tuple,
        margin_voxels=margin_voxels,
        bbox_expanded=expanded,
        mask_voxel_count=mask_voxel_count,
        roi_source=selection.roi_source,
        roi_label=selection.roi_label,
        roi_volume_mm3=selection.roi_volume_mm3,
        requested_roi_label=selection.requested_roi_label,
        requested_roi_total_voxel_count=(
            selection.requested_roi_total_voxel_count
        ),
        requested_roi_total_volume_mm3=(
            selection.requested_roi_total_volume_mm3
        ),
        requested_roi_component_count=(
            selection.requested_roi_component_count
        ),
        requested_roi_largest_component_voxel_count=(
            selection.requested_roi_largest_component_voxel_count
        ),
        requested_roi_largest_component_volume_mm3=(
            selection.requested_roi_largest_component_volume_mm3
        ),
        fallback_reason=selection.fallback_reason,
    )
