#!/usr/bin/env bash
#SBATCH --job-name=sot-rl-smoke
#SBATCH --output=results/rl-smoke-%j.out
#SBATCH --error=results/rl-smoke-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# SMOKE TEST for Claim A / Claim B (rl/train_grpo.py).
#
# WHY A SMOKE TEST FIRST. This harness has NEVER run end to end. Four bugs have
# already been found and fixed in it (completions-vs-prompts batching, 35GB
# checkpoints, OOM -> LoRA, HF-generate -> vLLM). A fifth is more likely than
# not, and discovering it 8 hours into a 9-run fan-out is the expensive way to
# find out. This run is deliberately tiny: 5 steps, 32 train problems, 8 eval
# problems. It is not an experiment -- it answers "does the loop turn over".
#
# WHAT IT DOES NOT TEST. With --steps 5 nothing will have learned. Do not read
# the accuracy. The pass condition is: SFT (n/a for baseline) -> rollout ->
# reward -> optimiser step -> eval, without crashing, with a checkpoint written.
#
# --no-vllm ON PURPOSE. vLLM is not installed on this box and may not have an
# aarch64+CUDA build. HF generate is roughly an order of magnitude slower, which
# is irrelevant for 5 steps and lets the smoke test run today rather than after
# a dependency fight. The real A/B needs vLLM -- see the harness's own note that
# it is "the difference between a 1-day and a 10-day A/B (3 arms x 3 seeds)".
#
# GB10 NOTE. Unified memory means a GPU OOM is a SYSTEM OOM: it wedges the host
# rather than killing this process. Hence sbatch with a hard --mem, never nohup.
# Slurm cannot see jobs run outside it, so the preflight below checks reality.
#
#   sbatch scripts/sbatch_rl_smoke.sh
#   tail -f results/rl-smoke-<jobid>.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export HF_HOME="${HF_HOME:-/mnt/t9/hf-cache}"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"; export HF_TOKEN
fi
mkdir -p "$HF_HOME" results

NEED_GB="${NEED_GB:-40}"
avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
gpu_used_gb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
              | awk '{s+=$1} END {print int(s/1024)}')
gpu_used_gb=${gpu_used_gb:-0}
echo "preflight   : ${avail_gb}GB host available, ${gpu_used_gb}GB GPU in use"
if [ "${avail_gb:-0}" -lt "$NEED_GB" ] || [ "$gpu_used_gb" -gt 8 ]; then
    echo "ABORT: box is loaded. Other projects run outside Slurm here." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    exit 1
fi

PY=.venv/bin/python
ARM="${ARM:-baseline}"     # baseline = Claim A (RL from base, nothing rewards dialogue)

echo "arm         : $ARM"
echo "model       : Qwen/Qwen2.5-3B"
echo

$PY -m rl.train_grpo \
  --arm "$ARM" \
  --seed 0 \
  --steps 5 \
  --eval-every 5 \
  --eval-n 8 \
  --train-n 32 \
  --num-generations 4 \
  --batch-size 8 \
  --grad-accum 2 \
  --no-vllm \
  --out results/rl_smoke

echo
echo "smoke finished. Check that:"
echo "  - a reward curve was printed (values, not NaN)"
echo "  - an eval ran at step 5"
echo "  - results/rl_smoke/ contains a checkpoint"
echo "Then scale: --steps 250, --train-n 2000, drop --no-vllm, 3 arms x 3 seeds."
