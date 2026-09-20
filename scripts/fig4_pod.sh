#!/usr/bin/env bash
# Fig. 4 (C4) -- does conversational behaviour EMERGE under accuracy-only RL?
# One SEED, self-contained on a 2x A100 pod. Run from the pod:
#
#   SEED=0 bash scripts/fig4_pod.sh
#
# THE EXPERIMENT. The paper trains Qwen-2.5-3B -- the *base* model, no instruction
# tuning, no priming -- with PPO on Countdown rewarding only 0.9*accuracy +
# 0.1*format, and reports that conversational behaviours rise across training
# (Fig. 4b, LLM-as-judge). This is that arm and nothing else. It is deliberately
# the same PPO as scripts/claimB_pod.sh's `baseline`, so the two are comparable.
#
# WHY A SEED KNOB AT ALL. This verl vintage exposes no seed (grep the resolved
# config in any of our logs: there is no `seed` key). A co-author has told us the
# RL results are seed-sensitive and that they are running multi-seed for the
# revision. To say anything useful about that we need runs that provably differ,
# so the seed is the training-set permutation -- reproducible, and visible in the
# parquet itself. See rl/run_seed.py.
#
# WHAT THE ARTIFACT IS. For C5 the result was the val curve and the log was
# scaffolding. Here it is the reverse: the model's generated traces are the
# measurement, and they exist ONLY in this log. The last time a run ended, its raw
# logs died with the pod. So the log is mirrored off-pod on a timer, and the run is
# worthless without it -- if the mirror is not configured, this script says so
# loudly rather than spending fourteen hours producing something unrecoverable.
set -uo pipefail

SEED="${SEED:?SEED required, e.g. SEED=0}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
MODEL_TAG="${MODEL_TAG:-qwen}"
REPO_DIR="${REPO_DIR:-/workspace/societies-of-thought}"
TZ="${TZ:-/workspace/TinyZero}"
DATA="${DATA:-/workspace/data/fig4}"
LOGD="${LOGD:-/workspace/logs/fig4}"
RUN="${MODEL_TAG}-s${SEED}"
LOG="$LOGD/ppo_${RUN}.log"
PY="${PY:-$(command -v python || command -v python3)}"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export N_GPUS=2 ROLLOUT_TP_SIZE=2 VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DATA" "$LOGD"

# --- 0. refuse to spend without a way to get the traces back --------------------
MIRROR_REPO="${MIRROR_REPO:-mhough/sot-fig4-logs}"
if [ -z "${HF_TOKEN:-}" ] && [ "${ALLOW_NO_MIRROR:-0}" != "1" ]; then
  echo "REFUSING TO START: no HF_TOKEN, so the traces -- which ARE this experiment's" >&2
  echo "result -- would exist only on this pod and die with it. Set HF_TOKEN, or set" >&2
  echo "ALLOW_NO_MIRROR=1 if you are pulling the log by scp yourself." >&2
  exit 2
fi

# --- 1. code + deps -------------------------------------------------------------
[ -d "$REPO_DIR/.git" ] || git clone -q https://github.com/m9h/societies-of-thought.git "$REPO_DIR"
[ -d "$TZ/.git" ]       || git clone -q https://github.com/Jiayi-Pan/TinyZero.git "$TZ"
cd "$REPO_DIR" && git fetch -q origin && git checkout -q "${BRANCH:-analysis/ddm-pilot}" && git pull -q || true
# TinyZero's verl pins `vllm<=0.6.3`, and resolving that from scratch sends pip
# backtracking through every uvicorn release down to 0.5.2 -- versions whose setup.py
# opens a README.md that is not in the sdist, so metadata generation dies. Two variants
# of that failure have already cost this pod two boots, and both surfaced as a missing
# parquet rather than as a failed install.
#
# So do not let the resolver search. Pin the chain, install it in order, and install verl
# itself with --no-deps once its requirements are already satisfied.
if ! "$PY" -c "import verl" 2>/dev/null; then
  echo "$(date -Is) installing verl dependency chain (pinned, no resolver search) ..."
  # uvicorn first: it is the package the backtracking is about. 0.30.6 has the
  # `standard` extra AND valid metadata, which the pre-0.9 releases do not.
  "$PY" -m pip install -q "uvicorn[standard]==0.30.6" || true
  "$PY" -m pip install -q "vllm==0.6.3" \
    || { echo "vllm==0.6.3 install FAILED -- verl cannot run" >&2; exit 1; }
  "$PY" -m pip install -q accelerate codetiming datasets dill hydra-core numpy pandas \
    pyarrow pybind11 ray "tensordict<0.6" "transformers<4.48" wandb \
    || { echo "verl requirement install FAILED" >&2; exit 1; }
  (cd "$TZ" && "$PY" -m pip install -q -e . --no-deps)
fi
# FAIL HERE, not three steps later. Without this the run continued into a data build
# that could not work, and the first symptom was a missing parquet -- which reads like a
# data bug rather than a failed dependency install.
"$PY" -c "import verl" 2>/dev/null \
  || { echo "verl did NOT install -- PPO cannot run. Tail of the pip output above." >&2; exit 1; }
# `datasets` is what TinyZero's countdown preprocessor imports. It arrives as a verl
# dependency when that install succeeds, and its absence is how a failed verl install
# first became visible. Name it explicitly so the dependency is not implicit.
"$PY" -c "import pandas, pyarrow, datasets" 2>/dev/null \
  || { echo "pandas/pyarrow/datasets missing after the dependency install" >&2; exit 1; }
if ! "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "$(date -Is) installing flash-attn (required by verl PPO) ..."
  "$PY" -m pip install -q ninja
  "$PY" -m pip install -q flash-attn --no-build-isolation \
    || { echo "flash-attn install FAILED -- verl PPO cannot run" >&2; exit 1; }
fi
cd "$REPO_DIR"

# --- 2. data: the stock Countdown set, our prompt, permuted by SEED -------------
if [ ! -f "$DATA/test.parquet" ]; then
  echo "$(date -Is) building data ..."
  "$PY" "$TZ/examples/data_preprocess/countdown.py" --local_dir "$DATA/_tz"
  "$PY" - "$DATA" <<'PYEOF'
import sys
from pathlib import Path
from rl.claimB_data import rewrite_ppo_prompt
d = Path(sys.argv[1])
for split in ("train", "test"):
    n = rewrite_ppo_prompt(d / "_tz" / f"{split}.parquet", d / f"_base_{split}.parquet")
    print(f"  rewrote {n} {split} prompts")
PYEOF
  cp "$DATA/_base_test.parquet" "$DATA/test.parquet"   # the eval set is shared, unpermuted
fi
for f in _base_train.parquet test.parquet; do
  [ -s "$DATA/$f" ] || { echo "data build produced no $f -- see the errors above" >&2; exit 1; }
done
"$PY" -m rl.run_seed "$DATA/_base_train.parquet" "$DATA/train_s${SEED}.parquet" --seed "$SEED"

# --- 3. smoke gate: fail cheap ---------------------------------------------------
"$PY" - "$DATA" "$SEED" <<'PYEOF' || { echo "SMOKE FAILED -- not spending GPU" >&2; exit 1; }
import sys
import pandas as pd
d, seed = sys.argv[1], sys.argv[2]
tr = pd.read_parquet(f"{d}/train_s{seed}.parquet")
p0 = tr.iloc[0]["prompt"][0]["content"]
# The scorer returns None (score 0) unless it finds a response marker. A prompt without
# it silently zeroes every rollout -- this cost a 170-step run once.
assert ("Assistant:" in p0) or ("<|im_start|>assistant" in p0), \
    "PPO prompt lacks the marker verl's Countdown scorer splits on -- all scores would be 0"
assert "create an equation that equals" in p0, "paper's instruction text missing"
assert "<answer>" in p0, "prompt does not request the answer container"
assert tr.iloc[0]["data_source"] == "countdown", "stock scorer key lost"
# The whole point of the seed: this order must not be the canonical one.
base = pd.read_parquet(f"{d}/_base_train.parquet")
assert len(base) == len(tr), "permutation changed the row count"
assert not base["prompt"].astype(str).equals(tr["prompt"].astype(str)), \
    f"seed {seed} produced the unpermuted order -- the run would be unseeded"
print(f"smoke OK: seed {seed}, {len(tr)} train rows")
PYEOF

# --- 4. mirror the log off-pod on a timer ----------------------------------------
# The traces are the result. Losing them to a terminated pod is the one failure this
# experiment cannot absorb, and it has happened here before.
if [ -n "${HF_TOKEN:-}" ]; then
  "$PY" -m pip install -q huggingface_hub 2>/dev/null
  (
    while true; do
      sleep "${MIRROR_EVERY:-900}"
      [ -s "$LOG" ] || continue
      "$PY" - "$LOG" "$MIRROR_REPO" "$RUN" <<'PYEOF' 2>/dev/null || true
import os, sys
from huggingface_hub import HfApi
log, repo, run = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=log, path_in_repo=f"{run}/ppo.log",
                repo_id=repo, repo_type="dataset")
PYEOF
    done
  ) & MIRROR_PID=$!
  echo "$(date -Is) mirroring $LOG -> hf://$MIRROR_REPO/$RUN every ${MIRROR_EVERY:-900}s (pid $MIRROR_PID)"
fi

# --- 5. PPO, un-primed, accuracy-only reward -------------------------------------
# SAVE_FREQ=-1: each verl checkpoint is ~16GB and three arms once filled a 200GB disk
# mid-save. The result here is the log, not the weights.
echo "$(date -Is) PPO fig4 seed=$SEED from $MODEL"
setsid nohup "$PY" -m verl.trainer.main_ppo \
  data.train_files="$DATA/train_s${SEED}.parquet" data.val_files="$DATA/test.parquet" \
  data.train_batch_size=128 data.val_batch_size=640 \
  data.max_prompt_length=1024 data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.optim.lr=1e-5 critic.model.path="$MODEL" \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['console'] +trainer.val_before_train=True \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:--1}" trainer.test_freq=10 \
  trainer.project_name=TinyZero trainer.experiment_name="countdown-fig4-$RUN" \
  trainer.total_epochs=15 trainer.total_training_steps="${STEPS:-250}" \
  > "$LOG" 2>&1 < /dev/null &

PPO_PID=$!
# A crash-on-import otherwise leaves the pod billing for hours against a dead process.
sleep 45
if ! kill -0 "$PPO_PID" 2>/dev/null; then
  echo "PPO DIED WITHIN 45s -- log tail:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
echo "$(date -Is) launched PPO fig4 seed=$SEED (pid $PPO_PID), alive at 45s."
echo "  tail -f $LOG"
