#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/training_logs/pmpd_v2_vitb_tumor_margin6_cv"
OUTPUT_DIR="outputs/pmpd_v2_vitb_tumor_margin6_cv"
RUN_NAME="anymc3d-dinov3-vitb-pmpd-v2-tumor-margin6"

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
        data=pmpd_v2_tumor_margin6 \
        model=anymc3d_dinov3_vitb_regularized \
        "data.fold=$folds" \
        data.module.batch_size=4 \
        model.slice_chunk_size=32 \
        model.max_epochs=150 \
        model.early_stopping_patience=150 \
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

echo "GPU 0 worker PID: $PID_GPU0"
echo "GPU 1 worker PID: $PID_GPU1"

trap 'kill "$PID_GPU0" "$PID_GPU1" 2>/dev/null || true' INT TERM

wait "$PID_GPU0"
STATUS_GPU0=$?
wait "$PID_GPU1"
STATUS_GPU1=$?

echo "$(date -Is) GPU 0 worker exit status: $STATUS_GPU0"
echo "$(date -Is) GPU 1 worker exit status: $STATUS_GPU1"

if [[ "$STATUS_GPU0" -ne 0 || "$STATUS_GPU1" -ne 0 ]]; then
    echo "At least one worker failed; OOF aggregation was not run."
    exit 1
fi

echo "$(date -Is) Aggregating folds 1-5"
"$PYTHON" -c \
    'from pathlib import Path; from train_cv import aggregate; aggregate([1, 2, 3, 4, 5], Path("outputs/pmpd_v2_vitb_tumor_margin6_cv"))'
echo "$(date -Is) PMPD-v2 tumor-margin-6 five-fold training and OOF aggregation completed"
