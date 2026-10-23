#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m pip install -r chat_lora/requirements.txt
python -u -m chat_lora.train_lora \
    --config chat_lora/configs/qwen05b.yaml \
    --device cuda \
    "$@"

echo
echo "Chat with:"
echo "  python -m chat_lora.chat --adapter chat_lora/runs/qwen05b/best --device cuda"
