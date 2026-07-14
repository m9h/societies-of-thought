#!/usr/bin/env bash
# THE GATE, done properly: a dose-response ladder on Countdown.
#
# Two corrections over the first attempt.
#
# (1) TOKEN BUDGET. At 1024 max_new_tokens the baseline parsed only 17% of traces
#     and scored 3.1% -- against the paper's 27.1%. R1-distill simply thinks longer
#     than that on Countdown, so we were measuring the token ceiling, not accuracy.
#     Generation stops at EOS, so a higher cap costs little on traces that finish
#     early; it only rescues the ones that were being guillotined.
#
# (2) STEERING UNITS ARE AMBIGUOUS IN THE PAPER. It steers at s = +-10, but never
#     says which activation scale s lives in:
#         - Neuronpedia's scale (max-act 5.906)      => s=10 is alpha 1.693
#         - the SAE's own scale (our max-act 14.754) => s=10 is alpha 0.678
#     That is a 2.5x difference in how hard we hit the residual stream. At
#     alpha=1.693 the added vector's norm is ~8.6 against a residual norm of ~12 --
#     a 71% perturbation, and the model degenerates into "oh, wait, no, wait"
#     babble. At alpha=0.678 it is ~30%.
#
#     So do not pick. Sweep the ladder and let accuracy-vs-strength answer it. Both
#     candidate readings of the paper (0.678 and 1.693) are on the ladder.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HOME="$PWD/.hf"
source .venv/bin/activate

SCOPE="${SCOPE:-generated}"   # set by the scope test; see results/scope_*.jsonl

python -m sot.run_sweep \
  --tasks countdown --layers 15 --mixture slimpj \
  --features 30939 \
  --alphas 0.25 0.5 0.678 1.0 1.693 \
  --scope "$SCOPE" \
  --n-problems 200 --max-new-tokens 4096 --batch-size 32 \
  --save-traces \
  --out "results/gate_dose_${SCOPE}.jsonl"

python -m sot.analyze --results "results/gate_dose_${SCOPE}.jsonl"
