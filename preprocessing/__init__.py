"""Physical-space preprocessing for pancreas-focused CT classification."""

from .pancreas_crop import (
    AlignmentError,
    CropResult,
    ImageMetadata,
    body_region,
    canonicalize_ct,
    canonicalize_pair,
    compute_mask_bbox,
    crop_from_mask_roi,
    crop_from_pancreas_mask,
    find_degenerate_slices,
    full_volume_region,
    inspect_image,
    normalize_roi_policy,
    select_crop_roi,
    validate_pair_alignment,
)
from .resampling import (
    compute_resampled_shape,
    resample_image_to_spacing,
    resample_mask_to_reference,
)

__all__ = [
    "AlignmentError",
    "CropResult",
    "ImageMetadata",
    "body_region",
    "canonicalize_ct",
    "canonicalize_pair",
    "compute_mask_bbox",
    "compute_resampled_shape",
    "crop_from_mask_roi",
    "crop_from_pancreas_mask",
    "find_degenerate_slices",
    "full_volume_region",
    "inspect_image",
    "normalize_roi_policy",
    "select_crop_roi",
    "resample_image_to_spacing",
    "resample_mask_to_reference",
    "validate_pair_alignment",
]
