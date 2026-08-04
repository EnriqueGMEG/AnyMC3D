#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${1:-anymc3d-dinov3-vitb-pmpd-v2-regularized-margin12-full}"
CHECKPOINT_ROOT="$REPO_DIR/checkpoints/$RUN_NAME"

shopt -s nullglob
metric_files=("$CHECKPOINT_ROOT"/fold_*/epoch_metrics.csv)

if [[ ${#metric_files[@]} -eq 0 ]]; then
    echo "No epoch metrics found yet under: $CHECKPOINT_ROOT"
    exit 0
fi

for metric_file in "${metric_files[@]}"; do
    fold_dir="$(basename -- "$(dirname -- "$metric_file")")"
    echo
    echo "=== $fold_dir ==="
    if command -v column >/dev/null 2>&1; then
        column -s, -t < "$metric_file"
    else
        sed -n '1,200p' "$metric_file"
    fi
done
