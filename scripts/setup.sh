#!/usr/bin/env bash
# Environment bootstrap. Assumes `uv` and a CUDA GPU with >=20GB free
# (8B model in bf16 is ~16GB; the SAE adds ~0.5GB).
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

# torch first, from the CUDA index matching the host driver. cu130 wheels are what
# the GB10/Blackwell box needs; on older x86 boxes swap to cu124.
uv pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
uv pip install -e .

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo
echo "Next: python -m sot.validate_hook     # MUST pass before any sweep"
