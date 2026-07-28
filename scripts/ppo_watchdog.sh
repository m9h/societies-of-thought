#!/usr/bin/env bash
# Restart a stalled/dead PPO arm, and stop the pod billing for a corpse.
#
# The monologue arm died silently at step 14 and was discovered 14 hours later, having
# billed ~$42 for an idle GPU. verl can exit without a traceback, so "process alive" and
# "log growing" are both required -- and neither was being checked.
#
#   setsid nohup bash scripts/ppo_watchdog.sh monologue qwen > /workspace/logs/wd.log 2>&1 &
set -u
ARM="${1:?arm}"; TAG="${2:-qwen}"
LOG="/workspace/logs/$TAG/ppo_${ARM}.log"
STALL_S="${STALL_S:-1800}"          # no log growth for 30 min => stalled
MAX_RESTARTS="${MAX_RESTARTS:-3}"
restarts=0

target_reached() {                   # 250 steps done?
  [ -f "$LOG" ] || return 1
  local s
  s=$(grep -oE "step:[0-9]+ - global_seqlen" "$LOG" 2>/dev/null \
      | grep -oE "[0-9]+" | sort -n | tail -1)
  [ "${s:-0}" -ge "${STEPS:-250}" ]
}

while true; do
  sleep 300
  if target_reached; then
    echo "$(date -Is) $ARM reached ${STEPS:-250} steps -- watchdog exiting"; exit 0
  fi

  # pgrep -f matches its own command line; -a + grep -v guards the self-match that has
  # produced false "alive" readings in this project more than once.
  alive=$(pgrep -af "verl.trainer.main_ppo" | grep -v watchdog | grep -vc pgrep || true)
  age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))

  if [ "${alive:-0}" -ge 1 ] && [ "$age" -lt "$STALL_S" ]; then continue; fi

  echo "$(date -Is) $ARM UNHEALTHY: procs=$alive log_age=${age}s"
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "$(date -Is) $ARM exhausted $MAX_RESTARTS restarts -- leaving dead so the"
    echo "  pod can be reclaimed rather than billing indefinitely."
    exit 1
  fi
  restarts=$((restarts + 1))
  echo "$(date -Is) restart $restarts/$MAX_RESTARTS of $ARM"
  pkill -f verl.trainer.main_ppo >/dev/null 2>&1; sleep 20
  cd /workspace/societies-of-thought || exit 1
  setsid nohup env MODEL="${MODEL:-Qwen/Qwen2.5-3B}" MODEL_TAG="$TAG" \
    bash scripts/claimB_pod.sh "$ARM" \
    > "/workspace/logs/restart_${ARM}_${restarts}.log" 2>&1 < /dev/null &
  sleep 240
done
