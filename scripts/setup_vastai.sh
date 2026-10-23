#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== LiteMoD Vast.ai setup =="
python -m pip install --upgrade pip

if ! python -c "import torch" >/dev/null 2>&1; then
    echo
    echo "ERROR: PyTorch is not installed in this Python environment."
    echo "Use a Vast.ai PyTorch template (CUDA 12.1+, torch >= 2.3), or install"
    echo "the CUDA wheel for your driver before re-running this script."
    exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"

python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("bf16 supported:", torch.cuda.is_bf16_supported())
props = torch.cuda.get_device_properties(0)
print(f"GPU memory: {props.total_memory / 1e9:.1f} GB")
PY

echo
echo "Setup complete. Next:"
echo "  python -m src.prepare_data --out-dir data --vocab-size 16384"
echo "  bash scripts/speed_check.sh"
