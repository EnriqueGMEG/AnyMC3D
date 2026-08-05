#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/training_logs/pmpd_v2_vitb_pancreas_base_full_volume_cv"
OUTPUT_DIR="outputs/pmpd_v2_vitb_pancreas_base_full_volume_cv"
RUN_NAME="anymc3d-dinov3-vitb-pmpd-v2-pancreas-base-full-volume"

mkdir -p "$LOG_DIR"

run_fold_group() {
    local gpu_id="$1"
    local folds="$2"
    local log_path="$3"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
        exec "$PYTHON" train.py \
        data=pmpd_v2_full_volume \
        model=anymc3d_dinov3_vitb_pancreas_base_full_volume \
        "data.fold=$folds" \
        "model.output_dir=$OUTPUT_DIR" \
        "model.run_name=$RUN_NAME" \
        >"$log_path" 2>&1
}

echo "$(date -Is) Starting GPU 0 folds [1,3,5]"
run_fold_group 0 '[1,3,5]' "$LOG_DIR/gpu0_folds_1_3_5.log" &
PID_GPU0=$!

echo "$(date -Is) Starting GPU 1 folds [2,4]"
run_fold_group 1 '[2,4]' "$LOG_DIR/gpu1_folds_2_4.log" &
PID_GPU1=$!

trap 'kill "$PID_GPU0" "$PID_GPU1" 2>/dev/null || true' INT TERM

wait "$PID_GPU0"
STATUS_GPU0=$?
wait "$PID_GPU1"
STATUS_GPU1=$?

if [[ "$STATUS_GPU0" -ne 0 || "$STATUS_GPU1" -ne 0 ]]; then
    echo "At least one worker failed; OOF aggregation was not run."
    exit 1
fi

"$PYTHON" -c \
    'from pathlib import Path; from train_cv import aggregate; aggregate([1, 2, 3, 4, 5], Path("outputs/pmpd_v2_vitb_pancreas_base_full_volume_cv"))'
echo "$(date -Is) Full-volume five-fold training and OOF aggregation completed"
