#!/usr/bin/env bash
#SBATCH --job-name=sot-layer-sweep
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --mem=96G
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

PY=.venv/bin/python
[ -x "$PY" ] || PY=python

LAYERS="$($PY -m sot.sweep_grid)"
OUT="${OUT:-results/steering/layers.jsonl}"

echo "host        : $(hostname)"
echo "layers      : $LAYERS"
echo "out         : $OUT  ($( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 ) rows already present)"
echo "mem cap     : ${SLURM_MEM_PER_NODE:-unset} MB"
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
