from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from transformers import DINOv3ViTConfig, DINOv3ViTModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tiny_dinov3() -> DINOv3ViTModel:
    config = DINOv3ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=64,
        patch_size=16,
        num_register_tokens=4,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    return DINOv3ViTModel(config)


def write_nifti_pair(
    directory: Path,
    patient_id: str,
    *,
    shape: tuple[int, int, int] = (20, 24, 12),
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    orientation: str = "RAS",
    ct_value: float = 50.0,
    mask_bounds: tuple[slice, slice, slice] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    ct = np.full(shape, ct_value, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    if mask_bounds is None:
        mask_bounds = (
            slice(6, 14),
            slice(8, 17),
            slice(3, 8),
        )
    mask[mask_bounds] = 1
    signs = {
        "RAS": (1.0, 1.0, 1.0),
        "LPS": (-1.0, -1.0, 1.0),
    }[orientation]
    affine = np.diag(
        [
            signs[0] * spacing[0],
            signs[1] * spacing[1],
            signs[2] * spacing[2],
            1.0,
        ]
    )
    ct_path = directory / f"{patient_id}_ct.nii.gz"
    mask_path = directory / f"{patient_id}_mask.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), ct_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    return ct_path, mask_path
