#!/usr/bin/env bash
# Body-cropped five-fold CV on a single GPU.
#
# GPU 1 hosts an unrelated long-running service, so folds cannot be split across
# both cards. They run sequentially on GPU 0; override GPU_ID to move them.
set -uo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="$REPO_DIR/training_logs/pmpd_v2_vitb_pancreas_base_body_cv"
OUTPUT_DIR="outputs/pmpd_v2_vitb_pancreas_base_body_cv"
RUN_NAME="anymc3d-dinov3-vitb-pmpd-v2-pancreas-base-body-cropped"

mkdir -p "$LOG_DIR"

for fold in 1 2 3 4 5; do
    echo "$(date -Is) Starting fold $fold on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
        "$PYTHON" train.py \
        data=pmpd_v2_body \
        model=anymc3d_dinov3_vitb_pancreas_base_body \
        "data.fold=[$fold]" \
        "model.output_dir=$OUTPUT_DIR" \
        "model.run_name=$RUN_NAME" \
        >"$LOG_DIR/fold_${fold}.log" 2>&1
    status=$?
    if [[ "$status" -ne 0 ]]; then
        echo "$(date -Is) Fold $fold failed (exit $status); OOF aggregation was not run."
        exit 1
    fi
    echo "$(date -Is) Finished fold $fold"
done

"$PYTHON" -c \
    'from pathlib import Path; from train_cv import aggregate; aggregate([1, 2, 3, 4, 5], Path("outputs/pmpd_v2_vitb_pancreas_base_body_cv"))'
echo "$(date -Is) Full-volume five-fold training and OOF aggregation completed"
