#!/usr/bin/env bash
# Claim B (C5), one ARM, self-contained on a 2x A100 pod. Run from the pod:
#
#   bash scripts/claimB_pod.sh baseline    # RL from base Qwen2.5-3B, shared prompt
#   bash scripts/claimB_pod.sh dialogue    # SFT-prime on dialogue, then the SAME RL
#   bash scripts/claimB_pod.sh monologue   # SFT-prime on monologue, then the SAME RL
#
# THE DESIGN. Every arm runs the IDENTICAL PPO recovered from Tier-0/Claim A
# (2x A100, TP=2, grad-checkpoint actor+critic, lr 1e-6/1e-5, kl 0.001,
# total_epochs 15). The ONLY differences across arms are (a) the starting weights
# -- base vs dialogue-primed vs monologue-primed -- and nothing else. The PPO set
# is TinyZero's own countdown parquet with only the prompt string swapped to the
# shared "Assistant:"-terminated prompt (rl.claimB_data.rewrite_ppo_prompt), so it
# is Claim A's problem set exactly, changed in one controlled way.
#
# Self-contained per arm so three of these can run on three pods in parallel, or
# sequentially on one. Each arm does its own priming; the baseline does none.
set -uo pipefail
ARM="${1:?arm: baseline | dialogue | monologue}"
case "$ARM" in baseline|dialogue|monologue) ;; *) echo "bad arm: $ARM" >&2; exit 2;; esac

REPO="${REPO:-/workspace/societies-of-thought}"
TZ="${TZ:-/workspace/TinyZero}"
DATA="${DATA:-/workspace/data/claimB}"
CKPT="${CKPT:-/workspace/ckpt/$ARM}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export N_GPUS=2 ROLLOUT_TP_SIZE=2 VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DATA" "$(dirname "$CKPT")" /workspace/logs

# --- 1. code + deps ------------------------------------------------------------
[ -d "$REPO/.git" ] || git clone -q https://github.com/m9h/societies-of-thought.git "$REPO"
[ -d "$TZ/.git" ]   || git clone -q https://github.com/Jiayi-Pan/TinyZero.git "$TZ"
cd "$REPO" && git pull -q || true
python -c "import verl" 2>/dev/null || (cd "$TZ" && pip install -q -e .)
python -c "import bitsandbytes, flash_attn" 2>/dev/null \
  || pip install -q bitsandbytes flash-attn --no-build-isolation

# --- 2. data: SFT parquets + the shared-prompt PPO parquet ----------------------
if [ ! -f "$DATA/train.parquet" ]; then
  echo "$(date -Is) building data ..."
  python -m rl.claimB_data --data rl/data --out "$DATA"                    # SFT parquets
  python "$TZ/examples/data_preprocess/countdown.py" --local_dir "$DATA/_tz"  # stock PPO set
  python - "$DATA" <<'PY'                                                   # swap in our prompt
import sys; from pathlib import Path; from rl.claimB_data import rewrite_ppo_prompt
d = Path(sys.argv[1])
for split in ("train", "test"):
    n = rewrite_ppo_prompt(d/"_tz"/f"{split}.parquet", d/f"{split}.parquet")
    print(f"  rewrote {n} {split} prompts -> {d}/{split}.parquet")
PY
fi

# --- 3. on-pod smoke gate: fail cheap, before any long run ----------------------
python - "$DATA" "$ARM" <<'PY' || { echo "SMOKE FAILED -- not spending GPU" >&2; exit 1; }
import sys; import pandas as pd
d, arm = sys.argv[1], sys.argv[2]
tr = pd.read_parquet(f"{d}/train.parquet")
p0 = tr.iloc[0]["prompt"][0]["content"]
assert p0.rstrip().endswith("Assistant:"), "PPO prompt not the shared template"
assert "<think>" not in p0.rsplit("Assistant", 1)[-1], "PPO prompt pre-opens <think>"
assert tr.iloc[0]["data_source"] == "countdown", "stock scorer key lost"
if arm != "baseline":
    s = pd.read_parquet(f"{d}/sft_{arm}_train.parquet")
    assert {"prompt", "response"} <= set(s.columns), "SFT schema missing prompt/response"
    sp = s.iloc[0]["prompt"]
    # same TEMPLATE as the PPO prompt (different problem, so not byte-equal): both must
    # end at "Assistant:" and share the instruction preamble, or priming is OOD under PPO.
    assert sp.rstrip().endswith("Assistant:"), "SFT prompt not the shared template"
    assert sp[:80] == p0[:80], "SFT and PPO prompt preambles diverge -- priming would be OOD"
    r0 = s.iloc[0]["response"]
    assert "<answer>" in r0 and r0.rstrip().endswith("</answer>"), "SFT response not gradable"
print("smoke OK:", arm)
PY

# --- 4. prime (skip for baseline) ----------------------------------------------
if [ "$ARM" != "baseline" ] && [ ! -f "$CKPT/config.json" ]; then
  echo "$(date -Is) SFT priming $ARM ..."
  python -m rl.sft_prime --train "$DATA/sft_${ARM}_train.parquet" \
    --model Qwen/Qwen2.5-3B --out "$CKPT" --epochs 3 --lr 1e-5 \
    2>&1 | tee "/workspace/logs/sft_${ARM}.log"
  [ -f "$CKPT/config.json" ] || { echo "priming produced no checkpoint" >&2; exit 1; }
fi

# --- 5. the identical PPO, from this arm's starting weights ---------------------
BASE_MODEL="Qwen/Qwen2.5-3B"; [ "$ARM" != "baseline" ] && BASE_MODEL="$CKPT"
echo "$(date -Is) PPO $ARM from $BASE_MODEL"
setsid nohup python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/test.parquet" \
  data.train_batch_size=256 data.val_batch_size=1312 \
  data.max_prompt_length=256 data.max_response_length=1024 \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.optim.lr=1e-5 critic.model.path="$BASE_MODEL" \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['console'] +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.save_freq=100 trainer.test_freq=25 \
  trainer.project_name=TinyZero trainer.experiment_name="countdown-claimB-$ARM" \
  trainer.total_epochs=15 > "/workspace/logs/ppo_${ARM}.log" 2>&1 < /dev/null &

echo "$(date -Is) launched PPO $ARM (pid $!). tail -f /workspace/logs/ppo_${ARM}.log"
