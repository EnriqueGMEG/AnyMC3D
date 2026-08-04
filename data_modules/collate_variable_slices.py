"""Batch collation for a variable number of real axial slices."""

from __future__ import annotations

from typing import Any

import torch


def collate_variable_slices(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad only the slice axis and retain physical metadata."""

    if not items:
        raise ValueError("Cannot collate an empty batch")
    heights = {int(item["volume"].shape[-2]) for item in items}
    widths = {int(item["volume"].shape[-1]) for item in items}
    channels = {int(item["volume"].shape[1]) for item in items}
    if len(heights) != 1 or len(widths) != 1 or channels != {1}:
        raise ValueError(
            "All cases must already share canvas HxW and have one channel"
        )
    max_slices = max(int(item["volume"].shape[0]) for item in items)
    batch_size = len(items)
    height, width = heights.pop(), widths.pop()
    volume = torch.zeros(
        batch_size, max_slices, 1, height, width, dtype=torch.float32
    )
    slice_mask = torch.zeros(batch_size, max_slices, dtype=torch.bool)
    positions = torch.full(
        (batch_size, max_slices), float("nan"), dtype=torch.float32
    )
    original_indices = torch.full(
        (batch_size, max_slices), -1, dtype=torch.long
    )
    for batch_index, item in enumerate(items):
        count = int(item["volume"].shape[0])
        if count == 0:
            raise ValueError(f"patient_id={item['patient_id']} has no slices")
        volume[batch_index, :count] = item["volume"]
        slice_mask[batch_index, :count] = True
        positions[batch_index, :count] = item["slice_positions_mm"]
        original_indices[batch_index, :count] = item[
            "original_slice_indices"
        ]
    return {
        "volume": volume,
        "label": torch.stack([item["label"] for item in items]).float(),
        "patient_id": [str(item["patient_id"]) for item in items],
        "fold": torch.tensor([int(item["fold"]) for item in items]),
        "slice_mask": slice_mask,
        "slice_positions_mm": positions,
        "original_slice_indices": original_indices,
        "geometry": [item["geometry"] for item in items],
    }
