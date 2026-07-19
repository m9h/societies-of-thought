#!/usr/bin/env bash
#SBATCH --job-name=sot-layer-sweep
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#
# Layer sweep on the DGX Spark, under Slurm with a HARD memory cap.
#
# WHY SLURM AND NOT nohup. On GB10 the GPU shares unified memory with the host,
# so a GPU OOM is a SYSTEM OOM -- it does not kill the process, it wedges the
# box. A hard --mem is the only thing standing between a bad batch size and a
# hung machine. Do not run this with nohup. Do not run it bare.
#
# WHAT IT RUNS. The grid comes from sot.registry.layer_sweep_grid (5 10 15 20 25
# 30) -- see docs/why_layers.md for why those layers and not a grid centred on
# the paper's layer 15. Never paste a literal list here; tests/test_sweep_grid.py
# fails if one appears in run_stages.sh, and the same reasoning applies.
#
# RESUMPTION. run_sweep re-reads the output JSONL and skips completed rows, so
# requeueing after a timeout is safe and cheap. The 572 rows already at layer 5
# in results/steering/layers.jsonl are picked up rather than redone -- point
# --out at that file to reuse them.
#
#   sbatch scripts/sbatch_layer_sweep.sh
#   squeue -u "$USER"
#   tail -f results/slurm-<jobid>.out
#
# COST. ~2900 generations per layer x 6 layers, max_new_tokens 8192. Steered
# traces frequently never emit EOS, so budget full length per batch rather than
# the average. This is the expensive run -- check the grid and the batch size
# before firing.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# PREFLIGHT: Slurm's accounting is not sufficient on this box.
#
# Two gaps, both observed on 2026-07-19:
#   1. Other projects and agents run OUTSIDE Slurm (the queue was empty while a
#      cell-tracking job held 22GB and a wwj benchmark held ~10GB). Slurm counts
#      that memory as free and will happily admit this job on top of it.
#   2. GB10 unified memory means a GPU allocation is a HOST allocation, but it
#      shows up in neither Slurm's accounting nor RSS -- that same job read as
#      22GB in nvidia-smi and 2.9GB in ps.
#
# So --mem is necessary but not sufficient. Check the real numbers and refuse to
# start if the box is already loaded, because a GPU OOM here is a system OOM: it
# wedges the host rather than killing this process.
NEED_GB="${NEED_GB:-60}"
avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
gpu_used_gb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
              | awk '{s+=$1} END {print int(s/1024)}')
gpu_used_gb=${gpu_used_gb:-0}

echo "preflight   : ${avail_gb}GB host available, ${gpu_used_gb}GB GPU already in use"
if [ "${avail_gb:-0}" -lt "$NEED_GB" ]; then
    echo "ABORT: only ${avail_gb}GB available, need ${NEED_GB}GB." >&2
    echo "Something outside Slurm is using the box. Check:" >&2
    ps -eo pid,etime,rss,args --sort=-rss --no-headers 2>/dev/null | head -5 >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
fi
if [ "$gpu_used_gb" -gt 8 ]; then
    echo "ABORT: ${gpu_used_gb}GB of GPU memory already allocated by another process." >&2
    echo "On GB10 that is host memory too. Refusing to contend." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
fi

# HF CACHE. ~/.cache/huggingface/hub is owned by root (created Jul 1 by
# something running as root), so model downloads there fail with a
# PermissionError that transformers misreports as a stale lock. Rather than
# needing sudo, point HF_HOME at /mnt/t9, which is user-owned with 2.6T free and
# is the local staging disk anyway -- NFS is the wrong place for a 16GB
# checkpoint. The auth token still lives in the old cache and is readable, so
# carry it across explicitly; without it the new HF_HOME has no token and
# transformers reports a public repo as "not a valid model identifier".
export HF_HOME="${HF_HOME:-/mnt/t9/hf-cache}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
mkdir -p "$HF_HOME"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python

LAYERS="$($PY -m sot.sweep_grid)"
OUT="${OUT:-results/steering/layers.jsonl}"

echo "host        : $(hostname)"
echo "layers      : $LAYERS"
echo "out         : $OUT  ($( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 ) rows already present)"
echo "mem cap     : ${SLURM_MEM_PER_NODE:-unset} MB"
echo "HF_HOME     : $HF_HOME  ($(df -h "$HF_HOME" 2>/dev/null | tail -1 | awk '{print $4}') free)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

$PY -m sot.run_sweep \
  --tasks gpqa math_hard \
  --layers $LAYERS \
  --mixture mixed \
  --n-candidates 3 --n-controls 3 --select-method neighbors \
  --alphas -2 2 \
  --n-problems 100 \
  --max-new-tokens 8192 \
  --batch-size 16 \
  --out "$OUT"

echo
echo "sweep finished; analysing"
$PY -m sot.analyze --results "$OUT"
