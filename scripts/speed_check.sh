#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

COMPILE_FLAG=(--compile)
if [[ "${NO_COMPILE:-0}" == "1" ]]; then
    COMPILE_FLAG=()
fi

python -u -m src.bench \
    --dense configs/dense_tiny.yaml \
    --fast configs/fast_tiny.yaml \
    --batch-sizes 8,16,32 \
    --seq-lens 256,512 \
    --warmup 5 \
    --iters 20 \
    "${COMPILE_FLAG[@]}" \
    "$@"
