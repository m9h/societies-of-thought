"""Re-score saved traces with the current grader. No GPU, no regeneration.

Generation is the expensive part and the grader is the part most likely to be wrong,
so `--save-traces` exists precisely to let the two be decoupled. When the grader
changes, this replays it over traces already on disk.

It also recomputes `truncated`, which the sweep recorded incorrectly: it compared the
*padded batch* length against max_new_tokens, so a single long sequence marked every
sequence in its batch as truncated. The per-sequence token count (n_tokens) was always
right, so truncation can be recovered exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grade import grade


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="the budget the traces were generated under")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.results}")
    if "trace" not in rows[0]:
        raise SystemExit("traces were not saved; re-run the sweep with --save-traces")

    changed = 0
    for r in rows:
        old_correct, old_parsed = r["correct"], r["parsed"]
        g = grade(r["task"], r["trace"], r["gold"])
        r["correct"], r["parsed"], r["pred"] = bool(g.correct), bool(g.parsed), g.pred
        # true truncation: this sequence itself ran out of budget
        r["truncated"] = r["n_tokens"] >= args.max_new_tokens - 1
        if (old_correct, old_parsed) != (r["correct"], r["parsed"]):
            changed += 1

    out = args.out or args.results.with_name(args.results.stem + "_regraded.jsonl")
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"re-graded {len(rows)} rows, {changed} changed ({changed/len(rows):.1%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
