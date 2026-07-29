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
STALL_S="${STALL_S:-5400}"          # measured cadence is ~165 s/step; a validation pass
                                    # is far longer. A 1800s threshold killed a HEALTHY
                                    # run at step 41 and cost it from scratch. 90 min is
                                    # ~30x the step time -- generous enough that only a
                                    # real hang trips it.
MAX_RESTARTS="${MAX_RESTARTS:-3}"
restarts=0

target_reached() {
  # Completion must be detected by MORE than the training-step counter. verl writes its
  # last `step:N - global_seqlen` line at N-1 (249 for a 250-step run), so a bare
  # `>= 250` never fires -- and this watchdog restarted a COMPLETED 250-step run,
  # wiping its log. The result survived only because rl.metrics_bridge had already
  # written the JSONL. Three independent signals now, any one of which counts:
  [ -f "$LOG" ] || return 1
  # 1. verl's own end-of-run marker
  grep -q "Final validation metrics" "$LOG" 2>/dev/null && return 0
  # 2. a validation line at or past the target step
  local v
  v=$(grep -oE "step:[0-9]+ - .*val/test_score" "$LOG" 2>/dev/null \
      | grep -oE "step:[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
  [ "${v:-0}" -ge "${STEPS:-250}" ] && return 0
  # 3. the training-step counter, allowing for verl's off-by-one
  local s
  s=$(grep -oE "step:[0-9]+ - global_seqlen" "$LOG" 2>/dev/null \
      | grep -oE "[0-9]+" | sort -n | tail -1)
  [ "${s:-0}" -ge $(( ${STEPS:-250} - 1 )) ]
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
