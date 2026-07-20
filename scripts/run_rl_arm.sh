#!/usr/bin/env bash
# Run one ARM of the Claim B A/B across all seeds, sequentially.
#
# GATED ON PREFLIGHT. rl.preflight runs on CPU in milliseconds and refuses to
# proceed if the experiment cannot produce a valid result -- mismatched problem
# sets, an arm the eval grader cannot parse, or a split between the eval and
# reward grading paths. On 2026-07-19 the absence of this gate cost ~$40: three
# pods trained on code where the dialogue arm reported 0% while scoring real
# reward, and nothing had to pass before the money flowed. Now something does.
#
# Detached (setsid nohup) so an ssh drop does not kill the run. Resumable: an arm
# whose curve.json exists is skipped, so a relaunch after a pod dies costs only
# the unfinished seeds.
#
#   setsid nohup scripts/run_rl_arm.sh baseline 150 > /tmp/driver.out 2>&1 < /dev/null &
set -uo pipefail
ARM="${1:?arm (baseline|dialogue|monologue)}"; STEPS="${2:-150}"
cd "$(dirname "$0")/.." 2>/dev/null || true

PY="${PY:-python}"

# --- the gate. No GPU is touched until this passes. --------------------------
echo "$(date -Is) preflight for ${ARM} ..."
if ! $PY -m rl.preflight; then
  echo "$(date -Is) ABORT ${ARM}: preflight failed -- not spending GPU on a broken experiment" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/rl_ab logs

for SEED in 0 1 2; do
  if [ -f "results/rl_ab/${ARM}_seed${SEED}/curve.json" ]; then
    echo "$(date -Is) SKIP ${ARM} seed ${SEED} (curve.json exists)" | tee -a logs/driver.log
    continue
  fi
  echo "$(date -Is) START ${ARM} seed ${SEED}" | tee -a logs/driver.log
  $PY -m rl.train_grpo --arm "$ARM" --seed "$SEED" \
    --steps "$STEPS" --eval-every 10 --eval-n 128 --train-n 2000 \
    --num-generations 8 --batch-size 8 --grad-accum 48 \
    --out results/rl_ab > "logs/${ARM}_seed${SEED}.log" 2>&1
  echo "$(date -Is) END ${ARM} seed ${SEED} exit=$?" | tee -a logs/driver.log
done
echo "$(date -Is) ALL SEEDS DONE for ${ARM}" | tee -a logs/driver.log
