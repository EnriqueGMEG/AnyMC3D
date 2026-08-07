from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from conftest import write_nifti_pair
from preprocessing.pancreas_crop import (
    AlignmentError,
    body_region,
    canonicalize_ct,
    canonicalize_pair,
    crop_from_pancreas_mask,
    find_degenerate_slices,
    normalize_roi_policy,
)
from preprocessing.pipeline import (
    compute_symmetric_padding,
    normalize_prewindowed_ct,
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


def _body_ct(tmp_path, name="body", filler=None, filler_index=-1, shape=(40, 40, 10)):
    """A centered cube of tissue in air, optionally with a constant filler slice."""

    data = np.zeros(shape, dtype=np.float32)
    data[12:28, 10:30, :] = 120.0
    if filler is not None:
        data[:, :, filler_index] = filler
    path = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0])), path)
    return path


def test_find_degenerate_slices_flags_only_constant_planes():
    data = np.random.default_rng(0).random((8, 8, 5)).astype(np.float32)
    data[:, :, 0] = 0.0
    data[:, :, 3] = 50.0
    assert find_degenerate_slices(data) == (0, 3)
    with pytest.raises(ValueError, match="3D volume"):
        find_degenerate_slices(data[..., 0])


def test_body_region_crops_air_and_keeps_every_slice(tmp_path):
    ct, _ = canonicalize_ct(_body_ct(tmp_path))
    crop = body_region(ct)
    assert crop.bbox_original == ((12, 28), (10, 30), (0, 10))
    assert crop.shape == (16, 20, 10)
    assert crop.roi_source == "body"
    assert crop.degenerate_slice_indices == ()
    # 16x20x10 tissue voxels at 1x1x2 mm, counted rather than approximated
    # from the projected footprint.
    assert crop.mask_voxel_count == 16 * 20 * 10
    assert crop.roi_volume_mm3 == pytest.approx(16 * 20 * 10 * 2.0)


def test_body_region_trims_trailing_constant_filler_slice(tmp_path):
    # Reproduces the 10 PMPD_v2 cases whose last slice is a uniform value.
    ct, _ = canonicalize_ct(_body_ct(tmp_path, filler=50.0, filler_index=-1))
    crop = body_region(ct)
    assert crop.degenerate_slice_indices == (9,)
    assert crop.bbox_original[2] == (0, 9)
    assert crop.shape == (16, 20, 9)


def test_body_region_keeps_filler_slice_when_disabled(tmp_path):
    ct, _ = canonicalize_ct(_body_ct(tmp_path, filler=50.0))
    crop = body_region(ct, drop_degenerate_slices=False)
    assert crop.degenerate_slice_indices == ()
    assert crop.shape[2] == 10
    # The filler spans the whole plane, so it inflates the in-plane bbox.
    assert crop.shape[:2] == (40, 40)


def test_body_region_rejects_interior_constant_slice(tmp_path):
    ct, _ = canonicalize_ct(_body_ct(tmp_path, filler=50.0, filler_index=5))
    with pytest.raises(ValueError, match="inside the retained axial range"):
        body_region(ct)


def test_body_region_margin_is_in_plane_only(tmp_path):
    ct, _ = canonicalize_ct(_body_ct(tmp_path, filler=50.0, filler_index=-1))
    crop = body_region(ct, margin_mm=(3.0, 3.0, 8.0))
    # 3 mm at 1 mm spacing widens H/W by three voxels a side...
    assert crop.shape[:2] == (22, 26)
    # ...while the axial range stays trimmed, not re-expanded onto the filler.
    assert crop.shape[2] == 9


def test_body_roi_policy_defaults_and_rejects_unknown_mode():
    policy = normalize_roi_policy({"mode": "body"})
    assert policy["threshold"] == 0.0
    assert policy["drop_degenerate_slices"] is True
    assert policy["source_name"] == "body"
    with pytest.raises(ValueError, match="Unknown ROI mode"):
        normalize_roi_policy({"mode": "torso"})


def test_body_mode_end_to_end_needs_no_mask(tmp_path):
    ct_path = _body_ct(tmp_path, shape=(40, 40, 6))
    case = preprocess_case(
        patient_id="case",
        ct_path=ct_path,
        pancreas_mask_path=None,
        target_spacing_mm=(1, 1, 2),
        canvas_hw=(32, 32),
        crop_margin_mm=(0.0, 0.0, 0.0),
        roi_policy={"mode": "body", "threshold": 0.0},
        intensity_mode="prewindowed_0_255",
        prewindowed_min=0.0,
        prewindowed_max=255.0,
    )
    assert case.volume.shape == (6, 1, 32, 32)
    assert case.log["roi_source"] == "body"
    assert case.log["body_threshold"] == 0.0
    assert case.log["bbox_original"] == ((12, 28), (10, 30), (0, 6))


def test_prewindowed_in_range_is_rescaled_without_clipping():
    values = np.array([0, 51, 255], dtype=np.float32)
    rescaled, stats = normalize_prewindowed_ct(values)
    assert np.allclose(rescaled, [0.0, 0.2, 1.0])
    assert stats["was_clipped"] is False
    assert stats["out_of_range_voxel_fraction"] == 0.0


def test_prewindowed_out_of_range_fails_under_error_policy():
    values = np.array([-51, 128, 306], dtype=np.float32)
    with pytest.raises(ValueError, match="outside the declared input range"):
        normalize_prewindowed_ct(values)


def test_prewindowed_out_of_range_saturates_within_budgets():
    values = np.full(1000, 128.0, dtype=np.float32)
    values[0] = -51.0
    values[1] = 306.0
    rescaled, stats = normalize_prewindowed_ct(
        values,
        out_of_range_policy="clip",
        max_out_of_range_fraction=0.05,
        max_out_of_range_magnitude=64.0,
    )
    assert stats["was_clipped"] is True
    assert stats["out_of_range_voxel_fraction"] == pytest.approx(0.002)
    assert stats["max_excursion"] == pytest.approx(51.0, abs=1e-2)
    assert rescaled.min() == 0.0 and rescaled.max() == 1.0


def test_prewindowed_clip_rejects_excess_voxel_fraction():
    values = np.full(100, 300.0, dtype=np.float32)
    with pytest.raises(ValueError, match="voxel fraction"):
        normalize_prewindowed_ct(
            values,
            out_of_range_policy="clip",
            max_out_of_range_fraction=0.05,
            max_out_of_range_magnitude=64.0,
        )


def test_prewindowed_clip_rejects_raw_hu_volume():
    values = np.full(1000, 128.0, dtype=np.float32)
    values[0] = -1024.0
    with pytest.raises(ValueError, match="excursion"):
        normalize_prewindowed_ct(
            values,
            out_of_range_policy="clip",
            max_out_of_range_fraction=0.05,
            max_out_of_range_magnitude=64.0,
        )


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
