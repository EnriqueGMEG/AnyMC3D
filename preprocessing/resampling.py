"""Affine-aware NIfTI resampling helpers."""

from __future__ import annotations

from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def compute_resampled_shape(
    shape: Sequence[int],
    source_spacing_mm: Sequence[float],
    target_spacing_mm: Sequence[float],
) -> tuple[int, int, int]:
    """Compute a shape preserving the center-to-center physical span."""

    if not (len(shape) == len(source_spacing_mm) == len(target_spacing_mm) == 3):
        raise ValueError("shape and spacings must all contain three values")
    result = []
    for n, source, target in zip(shape, source_spacing_mm, target_spacing_mm):
        if int(n) <= 0 or float(source) <= 0 or float(target) <= 0:
            raise ValueError(
                f"Invalid resampling geometry: shape={shape}, "
                f"source={source_spacing_mm}, target={target_spacing_mm}"
            )
        result.append(max(1, int(round((int(n) - 1) * float(source) / float(target))) + 1))
    return tuple(result)  # type: ignore[return-value]


def _target_grid(
    image: nib.spatialimages.SpatialImage,
    target_spacing_mm: Sequence[float],
) -> tuple[tuple[int, int, int], np.ndarray]:
    source_spacing = nib.affines.voxel_sizes(image.affine)
    shape = compute_resampled_shape(
        image.shape[:3], source_spacing, target_spacing_mm
    )
    direction = image.affine[:3, :3] / source_spacing[np.newaxis, :]
    target_affine = np.array(image.affine, dtype=np.float64, copy=True)
    target_affine[:3, :3] = direction * np.asarray(target_spacing_mm)[np.newaxis, :]
    return shape, target_affine


def resample_image_to_spacing(
    image: nib.spatialimages.SpatialImage,
    target_spacing_mm: Sequence[float],
    *,
    tolerance_mm: float = 0.01,
    cval: float = -150.0,
) -> tuple[nib.spatialimages.SpatialImage, bool]:
    """Linearly resample an image while retaining its direction and origin."""

    current = np.asarray(nib.affines.voxel_sizes(image.affine), dtype=float)
    target = np.asarray(tuple(float(v) for v in target_spacing_mm), dtype=float)
    if target.shape != (3,) or np.any(target <= 0):
        raise ValueError(f"Invalid target spacing: {target_spacing_mm}")
    if np.all(np.abs(current - target) <= float(tolerance_mm)):
        return image, False
    grid = _target_grid(image, target)
    result = resample_from_to(
        image, grid, order=1, mode="constant", cval=float(cval)
    )
    return result, True


def resample_mask_to_reference(
    mask: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> nib.spatialimages.SpatialImage:
    """Resample a discrete label image with nearest-neighbor only."""

    return resample_from_to(
        mask,
        (reference.shape[:3], reference.affine),
        order=0,
        mode="constant",
        cval=0,
    )


def physical_span_mm(
    shape: Sequence[int], spacing_mm: Sequence[float]
) -> tuple[float, float, float]:
    """Return center-to-center extent, the invariant used for resampling."""

    return tuple(
        max(0, int(n) - 1) * float(spacing)
        for n, spacing in zip(shape, spacing_mm)
    )
