#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

COMPILE_FLAG=(--compile)
if [[ "${NO_COMPILE:-0}" == "1" ]]; then
    COMPILE_FLAG=()
fi

echo "Training dense baseline ..."
python -u -m src.train \
    --config configs/dense_tiny.yaml \
    "${COMPILE_FLAG[@]}" \
    "$@" 2>&1 | tee runs/dense.log

echo
echo "Training LiteMoD ..."
python -u -m src.train \
    --config configs/fast_tiny.yaml \
    "${COMPILE_FLAG[@]}" \
    "$@" 2>&1 | tee runs/fast.log

echo
echo "Done. Plot with:"
echo "  python analysis/plot.py --dense runs/dense_tiny/metrics.csv --fast runs/fast_tiny/metrics.csv --out runs/compare.png"
