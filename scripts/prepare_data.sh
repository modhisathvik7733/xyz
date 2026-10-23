#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -u -m src.prepare_data \
    --dataset roneneldan/TinyStories \
    --out-dir data \
    --vocab-size 16384 \
    --tokenizer-sample 200000 \
    "$@"
