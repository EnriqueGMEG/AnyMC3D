from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from conftest import write_nifti_pair
from preprocessing.pancreas_crop import (
    AlignmentError,
    canonicalize_pair,
    crop_from_pancreas_mask,
)
from preprocessing.pipeline import (
    compute_symmetric_padding,
    pad_volume_in_plane,
    preprocess_case,
    window_ct,
)
from preprocessing.resampling import (
    physical_span_mm,
    resample_image_to_spacing,
)


def test_load_reorient_alignment_crop_and_four_mm_margin(tmp_path):
    ct_path, mask_path = write_nifti_pair(
        tmp_path, "case", orientation="LPS"
    )
    ct, mask, original_ct, _, _ = canonicalize_pair(ct_path, mask_path)
    assert original_ct.orientation == ("L", "P", "S")
    assert nib.aff2axcodes(ct.affine) == ("R", "A", "S")
    crop = crop_from_pancreas_mask(ct, mask, margin_mm=(4, 4, 4))
    assert crop.margin_voxels == (4, 4, 2)
    # Mask occupies z=[3,8); 4 mm at 2 mm/slice adds 2 per side.
    assert crop.bbox_original[2] == (3, 8)
    assert crop.bbox_expanded[2] == (1, 10)


def test_alignment_error_and_explicit_nearest_mask_resampling(tmp_path):
    ct_path, mask_path = write_nifti_pair(tmp_path, "case")
    mask = nib.load(mask_path)
    shifted = np.array(mask.affine)
    shifted[0, 3] += 0.5
    nib.save(nib.Nifti1Image(np.asanyarray(mask.dataobj), shifted), mask_path)
    with pytest.raises(AlignmentError):
        canonicalize_pair(ct_path, mask_path)
    ct, resampled_mask, _, _, changed = canonicalize_pair(
        ct_path, mask_path, resample_mask_to_ct=True
    )
    assert changed
    assert np.array_equal(
        np.unique(np.asanyarray(resampled_mask.dataobj)), [0, 1]
    )
    assert np.allclose(ct.affine, resampled_mask.affine)


def test_empty_mask_fails(tmp_path):
    ct_path, mask_path = write_nifti_pair(tmp_path, "case")
    mask = nib.load(mask_path)
    nib.save(nib.Nifti1Image(np.zeros(mask.shape), mask.affine), mask_path)
    ct, mask, *_ = canonicalize_pair(ct_path, mask_path)
    with pytest.raises(ValueError, match="empty"):
        crop_from_pancreas_mask(ct, mask)


def test_hu_window_and_mask_is_not_model_input(tmp_path):
    values = np.array([-200, -150, 50, 250, 300], dtype=np.float32)
    assert np.allclose(window_ct(values), [0, 0, 0.5, 1, 1])
    ct_path, mask_path = write_nifti_pair(
        tmp_path,
        "case",
        shape=(20, 24, 12),
        ct_value=50,
        mask_bounds=(slice(9, 11), slice(10, 12), slice(5, 7)),
    )
    case = preprocess_case(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        target_spacing_mm=(1, 1, 2),
        canvas_hw=(32, 32),
    )
    assert case.volume.dtype == np.float32
    assert case.volume.min() == 0  # rectangular canvas padding
    # All real rectangular crop pixels remain CT=50 HU => 0.5, including
    # pixels outside the tiny pancreas silhouette.
    pad = case.geometry["padding"]
    real = case.volume[
        :,
        :,
        pad["top"] : 32 - pad["bottom"],
        pad["left"] : 32 - pad["right"],
    ]
    assert np.allclose(real, 0.5)
    assert set(case.__dict__) == {
        "patient_id",
        "volume",
        "slice_positions_mm",
        "original_slice_indices",
        "geometry",
        "log",
    }


def test_resampling_preserves_physical_span():
    data = np.zeros((11, 21, 9), dtype=np.float32)
    image = nib.Nifti1Image(data, np.diag([1.0, 1.5, 2.5, 1.0]))
    result, changed = resample_image_to_spacing(
        image, (0.5, 1.0, 2.0), tolerance_mm=0.001
    )
    assert changed
    before = physical_span_mm(image.shape, (1.0, 1.5, 2.5))
    after = physical_span_mm(
        result.shape, nib.affines.voxel_sizes(result.affine)
    )
    assert np.allclose(before, after, atol=1.0)


def test_padding_no_resize_divisibility_and_overflow():
    volume = np.ones((7, 1, 11, 15), dtype=np.float32)
    padded, padding = pad_volume_in_plane(volume, (16, 32))
    assert padded.shape == (7, 1, 16, 32)
    assert padding == compute_symmetric_padding(11, 15, (16, 32))
    assert (padding.top, padding.bottom) == (2, 3)
    assert (padding.left, padding.right) == (8, 9)
    assert np.array_equal(
        padded[:, :, 2 : 2 + 11, 8 : 8 + 15], volume
    )
    assert padded.shape[-1] % 16 == padded.shape[-2] % 16 == 0
    with pytest.raises(ValueError, match="exceeds"):
        pad_volume_in_plane(volume, (8, 16))


def test_preprocessing_is_reproducible_for_inference(tmp_path):
    ct_path, mask_path = write_nifti_pair(tmp_path, "case")
    kwargs = dict(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=mask_path,
        target_spacing_mm=(0.8, 0.8, 2.0),
        canvas_hw=(32, 48),
    )
    first = preprocess_case(**kwargs)
    second = preprocess_case(**kwargs)
    assert np.array_equal(first.volume, second.volume)
    assert np.array_equal(
        first.original_slice_indices, second.original_slice_indices
    )
    assert np.allclose(first.slice_positions_mm, second.slice_positions_mm)
