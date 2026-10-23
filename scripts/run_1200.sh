#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

COMPILE_FLAG=(--compile)
if [[ "${NO_COMPILE:-0}" == "1" ]]; then
    COMPILE_FLAG=()
fi

# 1200 * 32 * 2 * 512 = 39.3M tokens per model.
CAPACITY="${CAPACITY:-0.5}"
COMMON=(
    --max-steps 1200
    --set model.mod_capacity="$CAPACITY"
    --set train.eval_interval=200
    --set train.log_interval=20
    --set train.save_interval=600
)

echo "=== 1200-step dense run ==="
python -u -m src.train \
    --config configs/dense_tiny.yaml \
    --set train.out_dir=runs/dense_1200 \
    "${COMMON[@]}" \
    "${COMPILE_FLAG[@]}" \
    "$@" 2>&1 | tee runs/dense_1200.log

echo
echo "=== 1200-step LiteMoD run ==="
python -u -m src.train \
    --config configs/fast_tiny.yaml \
    --set train.out_dir=runs/fast_1200 \
    "${COMMON[@]}" \
    "${COMPILE_FLAG[@]}" \
    "$@" 2>&1 | tee runs/fast_1200.log

echo
echo "=== comparison ==="
python -m src.compare_runs \
    --dense runs/dense_1200 \
    --fast runs/fast_1200 \
    --out runs/compare_1200.md \
    --json runs/compare_1200.json

echo
echo "=== plot ==="
python analysis/plot.py \
    --dense runs/dense_1200/metrics.csv \
    --fast runs/fast_1200/metrics.csv \
    --out runs/compare_1200.png

echo
echo "Done:"
echo "  runs/compare_1200.md"
echo "  runs/compare_1200.json"
echo "  runs/compare_1200.png"
