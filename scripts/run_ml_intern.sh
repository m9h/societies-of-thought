#!/usr/bin/env bash
# Launch HF's ml-intern on the RL half of the paper (the SFT-scaffolding replication).
#
# ml-intern is billed per LLM call. Guard rails:
#   --max-iterations   caps the agentic loop (default in ml-intern is 300)
#   headless prompt    points it at a written brief rather than an open-ended goal
#
# The brief (briefs/rl_replication.md) carries the gates. The agent is told to stop
# and report on a gate failure rather than press on -- that is the whole point of
# handing it a spec instead of a wish.
set -euo pipefail

REPO="$HOME/Workspace/societies-of-thought"
BRIEF="$REPO/briefs/rl_replication.md"
WORKDIR="$HOME/Workspace/sot-rl"          # agent works here, NOT in the steering repo
MODEL="${ML_INTERN_MODEL:-openrouter/z-ai/glm-4.6}"
MAX_ITER="${ML_INTERN_MAX_ITER:-120}"

# OpenRouter creds. ~/.bashrc early-returns for non-interactive shells, so the
# export there is invisible over ssh; the key is mirrored to this file instead.
set -a; . "$HOME/.openrouter.env"; set +a
export LOCAL_LLM_BASE_URL="https://openrouter.ai/api/v1"
export LOCAL_LLM_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_KEY="$OPENROUTER_API_KEY"     # some paths look for this name
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Keep HF cache project-local: the shared ~/.cache/huggingface/hub is root-owned.
export HF_HOME="$WORKDIR/.hf"
mkdir -p "$WORKDIR" "$HF_HOME"

[ -f "$BRIEF" ] || { echo "missing brief: $BRIEF" >&2; exit 1; }

cd "$WORKDIR"
cp "$BRIEF" ./BRIEF.md

export PATH="$HOME/.local/bin:$PATH"

echo "ml-intern  model=$MODEL  max-iter=$MAX_ITER  workdir=$WORKDIR"
echo "brief: $(wc -l < BRIEF.md) lines"
echo

exec ml-intern --model "$MODEL" --max-iterations "$MAX_ITER" "$(cat <<'PROMPT'
Read BRIEF.md in the current directory in full before doing anything. It is a
research specification with hard gates, written by a collaborator who has already
built the interpretability half of this project.

Execute it. Key points, which the brief expands on:

- The priority is Claim B: does SFT on multi-agent DIALOGUE traces make subsequent
  PPO on Countdown learn faster than SFT on MONOLOGUE traces, over identical
  problems with identical correct answers?
- The gates are not optional. Gate 2 in particular: if baseline PPO on
  un-fine-tuned Qwen-2.5-3B does not learn Countdown (paper: ~0% -> ~58% by step
  250), your RL harness is broken. STOP and report. Do not proceed to the A/B, and
  do not report a between-condition difference produced by a broken harness.
- The paper appears to report single runs. Run >= 3 seeds per condition and report
  the across-seed spread. If seed variance swamps the condition gap, that IS the
  finding. Report it plainly; do not bury it.
- Train on this machine's GPU (NVIDIA GB10, aarch64). Disk is tight -- check free
  space before downloading and do not hoard checkpoints. verl may not build on
  aarch64/Blackwell; if it fights you, use TRL's PPO/GRPO trainer instead, hold it
  constant across all conditions, and say so.
- A reference Countdown grader with passing tests is at
  ~/Workspace/societies-of-thought/sot/grade.py (_grade_countdown). Check yourself
  against it. An unparseable answer scores WRONG; it is never dropped.

Deliver results/curves.json, a seed-shaded accuracy-vs-step plot, and REPORT.md.
Every claim in REPORT.md must be traceable to a file in results/.
PROMPT
)"
