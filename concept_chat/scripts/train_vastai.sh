#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m pip install -r concept_chat/requirements.txt
python -u -m concept_chat.train \
    --config concept_chat/configs/small.yaml \
    --device cuda \
    "$@"
