from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from prepare_pmpd_v2_manifest import build_manifest


def _write_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.eye(4)),
        path,
    )


def test_build_manifest_preserves_source_folds_and_maps_labels(tmp_path):
    rows = []
    cohorts = {
        "PANGENE_OG": "Pangen_OG",
        "RUM": "RUM",
        "ZZU": "ZZU",
    }
    for fold in range(1, 6):
        source = tuple(cohorts)[(fold - 1) % len(cohorts)]
        name = f"case_{fold}"
        cohort = tmp_path / cohorts[source]
        _write_nifti(cohort / "images" / f"{name}.nii.gz")
        _write_nifti(
            cohort / "nnunet_predicted_masks" / f"{name}.nii.gz"
        )
        rows.append(
            {
                "fold": fold,
                "patient_key": f"{source}:{name}",
                "name": name,
                "source_dataset": source,
                "metastasis3": "Yes" if fold % 2 else "No",
            }
        )

    split_csv = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    manifest = build_manifest(tmp_path, split_csv)

    assert manifest["fold"].tolist() == [1, 2, 3, 4, 5]
    assert manifest["label"].tolist() == [1, 0, 1, 0, 1]
    assert manifest["patient_id"].tolist() == [
        row["patient_key"] for row in rows
    ]
    assert all(Path(path).is_file() for path in manifest["ct_path"])
    assert all(
        Path(path).is_file() for path in manifest["pancreas_mask_path"]
    )
