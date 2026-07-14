#!/usr/bin/env bash
# Wait for any in-flight sweep to release the GPU, then run the Countdown control
# gate. ~5.7h on the GB10. This is THE gate: it must reproduce the paper's
# 27.1% -> 54.8% before any GPQA/MATH number is worth reading.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "sot.run_sweep" > /dev/null; do sleep 20; done

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.hf"
exec ./scripts/run_stages.sh control
