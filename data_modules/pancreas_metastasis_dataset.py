"""Dataset and Lightning DataModule for preprocessed pancreas CT crops."""

from __future__ import annotations

from math import cos, radians, sin
import json
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .collate_variable_slices import collate_variable_slices
from .manifest import load_and_validate_manifest, split_manifest


class VolumeAugmenter:
    """Volume-coherent augmentation profiles with no per-slice randomness."""

    def __init__(self, profile: str = "size_preserving") -> None:
        if profile not in {"none", "size_preserving", "paper_like"}:
            raise ValueError(f"Unknown augmentation profile: {profile}")
        self.profile = profile

    def _gaussian_blur(self, volume: torch.Tensor) -> torch.Tensor:
        sigma = float(torch.empty(()).uniform_(0.5, 1.0))
        coordinates = torch.arange(
            -2, 3, dtype=volume.dtype, device=volume.device
        )
        kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        return F.conv2d(volume, kernel_2d, padding=2)

    def _size_preserving(self, volume: torch.Tensor) -> torch.Tensor:
        result = volume
        if torch.rand(()) < 0.25:
            result = result + torch.randn_like(result) * 0.015
        if torch.rand(()) < 0.15:
            result = self._gaussian_blur(result)
        if torch.rand(()) < 0.15:
            result = result * torch.empty(()).uniform_(0.9, 1.1)
        if torch.rand(()) < 0.15:
            mean = result.mean()
            result = (result - mean) * torch.empty(()).uniform_(0.9, 1.1) + mean
        if torch.rand(()) < 0.25:
            gamma = torch.empty(()).uniform_(0.85, 1.15)
            result = result.clamp(0, 1).pow(gamma)
        return result.clamp(0, 1)

    def _paper_like(self, volume: torch.Tensor) -> torch.Tensor:
        from .data_augmentation import build_train_transforms

        # S,1,H,W -> 1,H,W,S for one coherent 3D MONAI transform.
        chw_s = volume[:, 0].permute(1, 2, 0).unsqueeze(0)
        transformed = build_train_transforms()({"image": chw_s})["image"]
        return transformed[0].permute(2, 0, 1).unsqueeze(1)

    def spatial(
        self,
        volume: torch.Tensor,
        *,
        translation_limit_hw: tuple[float, float],
    ) -> torch.Tensor:
        if self.profile != "size_preserving" or torch.rand(()) >= 0.30:
            return volume
        angle = radians(float(torch.empty(()).uniform_(-7.0, 7.0)))
        limit_h, limit_w = translation_limit_hw
        translate_h = float(torch.empty(()).uniform_(-limit_h, limit_h))
        translate_w = float(torch.empty(()).uniform_(-limit_w, limit_w))
        theta = torch.zeros((2, 3), dtype=volume.dtype, device=volume.device)
        theta[0, 0] = cos(angle)
        theta[0, 1] = -sin(angle)
        theta[1, 0] = sin(angle)
        theta[1, 1] = cos(angle)
        theta[0, 2] = 2.0 * translate_w / float(volume.shape[-1])
        theta[1, 2] = 2.0 * translate_h / float(volume.shape[-2])
        grid = F.affine_grid(
            theta.unsqueeze(0).expand(volume.shape[0], -1, -1),
            volume.shape,
            align_corners=False,
        )
        return F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).clamp(0, 1)

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if self.profile == "none":
            return volume
        if self.profile == "size_preserving":
            return self._size_preserving(volume)
        return self._paper_like(volume)


class PancreasMetastasisDataset(Dataset):
    """Load `.npz` artifacts without any spatial resize."""

    def __init__(
        self,
        records,
        *,
        preprocessed_root: str | Path,
        augmentation_profile: str = "none",
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.preprocessed_root = Path(preprocessed_root)
        self.augment = VolumeAugmenter(augmentation_profile)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        path = self.preprocessed_root / f"{row.patient_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"patient_id={row.patient_id}: artifact not found: {path}"
            )
        with np.load(path, allow_pickle=False) as artifact:
            volume = torch.from_numpy(artifact["volume"]).float()
            positions = torch.from_numpy(
                artifact["slice_positions_mm"]
            ).float()
            indices = torch.from_numpy(
                artifact["original_slice_indices"]
            ).long()
            geometry = json.loads(str(artifact["geometry_json"].item()))
        if volume.ndim != 4 or volume.shape[1] != 1:
            raise ValueError(
                f"patient_id={row.patient_id}: invalid volume shape {volume.shape}"
            )
        if not (len(volume) == len(positions) == len(indices)):
            raise ValueError(
                f"patient_id={row.patient_id}: slice metadata length mismatch"
            )
        padding = geometry.get("padding", {})
        top = int(padding.get("top", 0))
        bottom = int(padding.get("bottom", 0))
        left = int(padding.get("left", 0))
        right = int(padding.get("right", 0))
        height_end = volume.shape[-2] - bottom if bottom else volume.shape[-2]
        width_end = volume.shape[-1] - right if right else volume.shape[-1]
        if not (0 <= top < height_end <= volume.shape[-2]):
            raise ValueError(f"patient_id={row.patient_id}: invalid vertical padding {padding}")
        if not (0 <= left < width_end <= volume.shape[-1]):
            raise ValueError(f"patient_id={row.patient_id}: invalid horizontal padding {padding}")
        unpadded = volume[..., top:height_end, left:width_end]
        augmented = self.augment(unpadded)
        if any((top, bottom, left, right)):
            volume = torch.zeros_like(volume)
            volume[..., top:height_end, left:width_end] = augmented
        else:
            volume = augmented
        volume = self.augment.spatial(
            volume,
            translation_limit_hw=(
                float(min(6, top, bottom)),
                float(min(6, left, right)),
            ),
        )
        return {
            "volume": volume,
            "label": torch.tensor(float(row.label), dtype=torch.float32),
            "patient_id": str(row.patient_id),
            "fold": int(row.fold),
            "slice_positions_mm": positions,
            "original_slice_indices": indices,
            "geometry": geometry,
        }


class PancreasMetastasisDataModule(L.LightningDataModule):
    """Strict fold-aware DataModule using the manifest's existing folds."""

    def __init__(
        self,
        *,
        manifest_path: str,
        preprocessed_root: str,
        fold: int = 0,
        batch_size: int = 2,
        num_workers: int = 4,
        augmentation_profile: str = "size_preserving",
        validate_nifti_on_setup: bool = True,
        require_mask: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.preprocessed_root = preprocessed_root
        self.fold = int(fold)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.augmentation_profile = augmentation_profile
        self.validate_nifti_on_setup = bool(validate_nifti_on_setup)
        self.require_mask = bool(require_mask)
        self.train_records = None
        self.validation_records = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_records is not None and self.validation_records is not None:
            return
        manifest = load_and_validate_manifest(
            self.manifest_path,
            check_nifti_geometry=self.validate_nifti_on_setup,
            require_mask=self.require_mask,
        )
        self.train_records, self.validation_records = split_manifest(
            manifest, self.fold
        )

    def _loader(self, records, *, train: bool) -> DataLoader:
        if records is None:
            raise RuntimeError("Call setup() before requesting a DataLoader")
        dataset = PancreasMetastasisDataset(
            records,
            preprocessed_root=self.preprocessed_root,
            augmentation_profile=(
                self.augmentation_profile if train else "none"
            ),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_variable_slices,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_records, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_records, train=False)
