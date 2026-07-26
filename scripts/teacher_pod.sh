#!/usr/bin/env bash
# Generate the paper's SFT priming data on a 2x A100 pod, with the paper's teacher.
#
#   bash scripts/teacher_pod.sh
#
# Expects /workspace/pool.json to already exist (built off-GPU by rl.pool_build and
# scp'd in -- the pool build is CPU-only, so it must not burn GPU time). Falls back to
# building it here only if it is missing.
#
# Everything about fidelity lives in rl/paper_spec.py; this script only supplies compute.
set -uo pipefail
REPO="${REPO:-/workspace/societies-of-thought}"
POOL="${POOL:-/workspace/pool.json}"
OUT="${OUT:-/workspace/ood}"
ATTEMPT="${ATTEMPT:-2500}"
TP="${TP:-2}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PYTHONUNBUFFERED=1
mkdir -p /workspace/logs "$HF_HOME"

echo "$(date -Is) setup"
[ -d "$REPO/.git" ] || git clone -q https://github.com/m9h/societies-of-thought.git "$REPO"
cd "$REPO" && git pull -q || true

python -c "import vllm" 2>/dev/null || pip install -q vllm
python -c "import datasets" 2>/dev/null || pip install -q datasets
python -c "import pandas, pyarrow" 2>/dev/null || pip install -q pandas pyarrow
pip install -q trackio 2>/dev/null || echo "trackio unavailable (JSONL metrics still written)"

# Fidelity gate FIRST: never spend GPU on a config that is not the paper's.
python -m pytest tests/test_paper_fidelity.py -q \
  || { echo "FIDELITY GATE FAILED -- not spending GPU" >&2; exit 1; }

if [ ! -f "$POOL" ]; then
  echo "$(date -Is) pool.json missing; building on-pod (should have been scp'd)"
  python -m rl.pool_build --out "$POOL" || exit 1
fi
python - "$POOL" <<'PY' || exit 1
import json, sys
from rl import paper_spec as S
pool = json.load(open(sys.argv[1]))
assert len(pool) == S.POOL_TOTAL, f"pool is {len(pool)}, paper specifies {S.POOL_TOTAL}"
grad = sum(1 for r in pool if r["gradable"])
assert grad == 7738, f"gradable is {grad}; the paper uses 7,738 after excluding IFEval"
assert not any("countdown" in r["source"] for r in pool), "pool must be out-of-domain"
print(f"pool OK: {len(pool)} problems, {grad} gradable, no Countdown")
PY

echo "$(date -Is) generating priming traces with the paper's teacher"
setsid nohup python -m rl.generate_sft_ood \
  --pool "$POOL" --out "$OUT" --attempt "$ATTEMPT" --tp "$TP" \
  > /workspace/logs/generate_ood.log 2>&1 < /dev/null &
echo "$(date -Is) launched (pid $!). tail -f /workspace/logs/generate_ood.log"
