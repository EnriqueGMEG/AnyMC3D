#!/usr/bin/env python3
"""Reproducible single-patient inference from CT and an optional ROI mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from model_arch.pancreas_lightning import (
    PancreasMetastasisLightningModule,
)
from preprocessing.inference_contract import preprocess_with_saved_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--pancreas-mask", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/preprocessing/preserve_physical_size.yaml"),
    )
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_contract = json.loads(args.geometry_config.read_text())
    preprocessing = OmegaConf.load(args.preprocessing_config)
    patient_id = args.patient_id or (
        args.ct.name[:-7]
        if args.ct.name.endswith(".nii.gz")
        else args.ct.stem
    )
    case = preprocess_with_saved_contract(
        patient_id=patient_id,
        ct_path=args.ct,
        pancreas_mask_path=args.pancreas_mask,
        geometry_contract=geometry_contract,
        preprocessing_config=preprocessing,
    )
    requested_device = (
        args.device if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(requested_device)
    model = PancreasMetastasisLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.eval().to(device)
    volume = torch.from_numpy(case.volume).unsqueeze(0).to(device)
    slice_mask = torch.ones(
        1, case.volume.shape[0], dtype=torch.bool, device=device
    )
    with torch.inference_mode():
        output = model(volume, slice_mask)
        logit = float(output.logits[0, 0].cpu())
        probability = float(torch.sigmoid(output.logits[0, 0]).cpu())
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(model.selected_threshold.cpu())
    )
    prediction = int(probability >= threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "patient_id": patient_id,
        "logit": logit,
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "geometry": case.geometry,
        "geometry_config": str(args.geometry_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    (args.output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    attention = output.attention_weights[0].cpu().numpy()
    pd.DataFrame(
        {
            "patient_id": patient_id,
            "slice_index": range(len(attention)),
            "original_slice_index": case.original_slice_indices,
            "z_position_mm": case.slice_positions_mm,
            "attention_weight": attention,
            "is_valid_slice": True,
        }
    ).to_csv(args.output_dir / "slice_attention.csv", index=False)
    (args.output_dir / "geometry.json").write_text(
        json.dumps(case.geometry, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
