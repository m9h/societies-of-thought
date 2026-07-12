#!/usr/bin/env bash
# Staged experiment. Each stage is a gate: if it fails, the next stage is not
# worth running. Run them in order.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Keep the HF cache project-local and user-owned. The shared ~/.cache/huggingface/hub
# can end up root-owned after a container run, which fails downloads with a bare
# PermissionError; this sidesteps it without touching root-owned paths.
export HF_HOME="${HF_HOME:-$PWD/.hf}"
mkdir -p "$HF_HOME"

STAGE="${1:-help}"

case "$STAGE" in

# ---------------------------------------------------------------------------
# Stage 0. Where does the SAE actually attach, and is our scaling right?
# The SAE config says resid_post; Neuronpedia's metadata says resid_pre. These
# differ by a layer. Settle it against the published max activation (5.906)
# before steering anything.
# ---------------------------------------------------------------------------
hook)
  python -m sot.validate_hook --layer 15 --mixture slimpj --feature 30939
  ;;

# ---------------------------------------------------------------------------
# Stage 0b. Calibrate feature activation scales in OUR units.
# Neuronpedia's published max activations are on a different scale than the SAE's
# (~3.2x here). Steering strength is a multiple of max activation, so using their
# number would make every intervention ~3x weaker than intended -- silently.
# ---------------------------------------------------------------------------
calibrate)
  python -m sot.calibrate --layer 15 --mixture slimpj --n-docs 2000
  ;;

# ---------------------------------------------------------------------------
# Stage 1. POSITIVE CONTROL: reproduce the paper on its own task.
# Target: 27.1% at s=0 -> 54.8% at s=+10 on Countdown, feature 30939.
# Uses the paper's raw strengths, not our alpha units, so the numbers are
# directly comparable. If this does not roughly reproduce, the harness is wrong
# and every later result is uninterpretable. ~5k generations.
# ---------------------------------------------------------------------------
control)
  python -m sot.run_sweep \
    --tasks countdown --layers 15 --mixture slimpj \
    --features 30939 \
    --raw-strengths -10 -5 5 10 \
    --n-problems 1024 --max-new-tokens 1024 \
    --batch-size 32 --save-traces \
    --out results/stage1_countdown_control.jsonl
  python -m sot.analyze --results results/stage1_countdown_control.jsonl
  ;;

# ---------------------------------------------------------------------------
# Stage 2. THE ACTUAL QUESTION: does it survive off Countdown?
# Layer 15, the paper's SAE and its anchor feature, plus conversational
# candidates and sparsity/magnitude-matched controls, on GPQA-Diamond and
# MATH-Hard. Reasoning traces here are long, hence the larger token budget.
# ~11 features x 4 strengths x 2 tasks x 100 problems.
# ---------------------------------------------------------------------------
main)
  python -m sot.run_sweep \
    --tasks gpqa math_hard --layers 15 --mixture slimpj \
    --n-candidates 5 --n-controls 5 --select-method neighbors \
    --alphas -2 -1 1 2 \
    --n-problems 100 --samples 2 --max-new-tokens 8192 \
    --batch-size 16 --save-traces \
    --out results/stage2_main.jsonl
  python -m sot.analyze --results results/stage2_main.jsonl
  ;;

# ---------------------------------------------------------------------------
# Stage 3. LAYER SWEEP. Only worth running if stage 2 shows an effect at all.
# Uses the "mixed" SAE suite because it is the only one published for every
# layer. Layer 15 is in both suites, so it doubles as a check that the effect
# is not an artifact of the SAE's training mixture.
# ---------------------------------------------------------------------------
layers)
  python -m sot.run_sweep \
    --tasks gpqa math_hard --layers 9 12 15 18 21 24 --mixture mixed \
    --n-candidates 3 --n-controls 3 --select-method neighbors \
    --alphas -2 2 \
    --n-problems 100 --max-new-tokens 8192 \
    --batch-size 16 \
    --out results/stage3_layers.jsonl
  python -m sot.analyze --results results/stage3_layers.jsonl
  ;;

# Fast end-to-end wiring check: 8 problems, one feature. Minutes, not hours.
smoke)
  python -m sot.run_sweep \
    --tasks gpqa --layers 15 --mixture slimpj \
    --features 30939 --alphas 2 \
    --n-problems 8 --max-new-tokens 1024 --batch-size 8 \
    --save-traces --out results/smoke.jsonl
  ;;

*)
  echo "usage: $0 {hook|calibrate|smoke|control|main|layers}"
  echo
  echo "  hook      resolve the resid_pre/resid_post ambiguity  (required first)"
  echo "  calibrate measure feature max-acts in our units       (required second)"
  echo "  smoke     8-problem wiring check"
  echo "  control   reproduce the paper on Countdown            (gate)"
  echo "  main      GPQA + MATH-Hard steering sweep             (the experiment)"
  echo "  layers    layer sweep, only if 'main' shows an effect"
  exit 1
  ;;
esac
