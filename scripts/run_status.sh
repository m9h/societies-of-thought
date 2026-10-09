#!/usr/bin/env bash
# One-line status for a verl PPO log: run_status.sh LOG [STEPS=250] [PID]
#
#   STATUS=not_started|running|completed|crashed|incomplete STEP=n VAL=v LOG_AGE_S=s
#
# Replaces the ad-hoc session monitor used on the Fig. 4 run, which had two bugs:
#  - it anchored `^step:` and so read step 0 on Ray-prefixed lines when the log said 250;
#  - it used `kill -0` alone, which cannot tell a normal exit from a crash, and reported
#    a successful 250-step completion as FAILURE.
# Completion is decided from the log by the same three signals as ppo_watchdog.sh; the
# pid only distinguishes running from crashed once the log says the run is NOT complete.
set -u
LOG="${1:?log}"; STEPS="${2:-250}"; PID="${3:-}"

if [ ! -f "$LOG" ]; then
  echo "STATUS=not_started STEP=0 VAL=- LOG_AGE_S=-"; exit 0
fi

# No line anchor: verl's lines arrive as "(main_task pid=N) step:K - ..." with ANSI colour.
step=$(grep -oE "step:[0-9]+ - " "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1)
step="${step:-0}"
val=$(grep -oE "val/test_score/[A-Za-z0-9_]+:[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
val="${val:--}"
age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))

completed=0
grep -q "Final validation metrics" "$LOG" && completed=1
vstep=$(grep -oE "step:[0-9]+ - .*val/test_score" "$LOG" | grep -oE "step:[0-9]+" \
        | grep -oE "[0-9]+" | sort -n | tail -1)
[ "${vstep:-0}" -ge "$STEPS" ] && completed=1
# verl writes its last training-step line at N-1
[ "$step" -ge $(( STEPS - 1 )) ] && completed=1

if [ "$completed" -eq 1 ]; then
  status=completed
elif [ -n "$PID" ]; then
  if kill -0 "$PID" 2>/dev/null; then status=running; else status=crashed; fi
else
  status=incomplete
fi
echo "STATUS=$status STEP=$step VAL=$val LOG_AGE_S=$age"
