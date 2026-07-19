"""Print the layer sweep grid, so shell scripts can interpolate it.

    python -m sot.sweep_grid                                  -> 5 10 15 20 25 30
    python -m sot.sweep_grid --model google/gemma-3-27b-it    -> 16 31 40 41 53

Exists so scripts/run_stages.sh does not carry its own copy of the grid. It
used to, and the copy drifted: the script asked for 9/12/15/18/21/24 while
calibration on disk covered 5/10/15/20/25/30 and the only completed rows were
at layer 5. Running the script would have recalibrated five layers and
discarded 572 rows already paid for.
"""

from __future__ import annotations

import argparse

from sot.registry import layer_sweep_grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    args = ap.parse_args()
    print(" ".join(str(layer) for layer in layer_sweep_grid(args.model)))


if __name__ == "__main__":
    main()
