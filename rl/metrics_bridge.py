"""Turn verl's console log into a durable metrics artifact (and a Trackio run).

Why this exists
---------------
The first Claim B runs logged to console only. Every curve in the write-up was then
recovered by grepping 22MB of text with a regex, partial curves were stranded on pods
that had to be scp'd before termination, and a disk-full crash nearly lost the Llama
data outright. Console logs are not an experiment record.

This bridge tails a verl log and emits:

  * `<run>.jsonl`  -- one JSON object per training step, every metric verl printed.
    This is the durable artifact: greppable by nobody, loadable by everybody.
  * a Trackio run    -- live, shareable curves. Trackio is wandb-API-compatible and
    runs locally / on a HF Space, so there is no account or key to manage on a pod.

Deliberately decoupled from verl internals: it parses stdout rather than hooking the
trainer, so it survives verl version changes and works on a run already in flight.

    python -m rl.metrics_bridge --log /workspace/logs/qwen/ppo_dialogue.log \
        --out /workspace/metrics/qwen_dialogue.jsonl \
        --project societies-of-thought --run qwen-dialogue --follow
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# verl console format: "step:12 - critic/kl:0.055 - val/test_score/countdown:0.42 - ..."
_STEP = re.compile(r"\bstep:(\d+)\b")
_PAIR = re.compile(r"([A-Za-z_][\w/.\-]*):(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_line(line: str) -> dict | None:
    """Parse one verl metrics line into {step, **metrics}, or None if not one."""
    line = _ANSI.sub("", line).replace("\r", "")
    m = _STEP.search(line)
    if not m:
        return None
    step = int(m.group(1))
    metrics = {}
    for key, val in _PAIR.findall(line):
        if key == "step":
            continue
        metrics[key] = float(val)
    if not metrics:
        return None
    return {"step": step, **metrics}


def _open_tracker(project: str, run: str, config: dict | None):
    """Return a Trackio run handle, or None if trackio is unavailable."""
    try:
        import trackio
    except ImportError:
        print("trackio not installed; writing JSONL only "
              "(pip install trackio to get live curves)")
        return None
    trackio.init(project=project, name=run, config=config or {})
    return trackio


def bridge(log: Path, out: Path, project: str, run: str, follow: bool,
           poll: float = 5.0, config: dict | None = None,
           trackio: bool = False) -> int:
    """Stream `log` into `out` (JSONL), and optionally into Trackio.

    JSONL is the durable artifact and is opened FIRST. Trackio is opt-in via
    `trackio=True`: `trackio.init()` can block trying to stand up a dashboard, and when
    it did, the bridge produced no file at all while its process looked alive -- three
    training arms ran with zero metrics recorded. Nothing optional gets to sit in front
    of the thing we actually need.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    if out.exists():  # resume without duplicating
        for ln in out.read_text().splitlines():
            try:
                seen.add(json.loads(ln)["step"])
            except Exception:
                pass

    written = 0
    pos = 0
    sink = out.open("a")          # create the artifact before anything can block
    sink.write("")
    sink.flush()
    tracker = _open_tracker(project, run, config) if trackio else None
    try:
        while True:
            if log.exists():
                with log.open(errors="replace") as fh:
                    fh.seek(pos)
                    for line in fh:
                        rec = parse_line(line)
                        if rec is None or rec["step"] in seen:
                            continue
                        seen.add(rec["step"])
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()
                        written += 1
                        if tracker is not None:
                            payload = {k: v for k, v in rec.items() if k != "step"}
                            tracker.log(payload, step=rec["step"])
                    pos = fh.tell()
            if not follow:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception:
                pass
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="verl console log -> JSONL + Trackio.")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--project", default="societies-of-thought")
    ap.add_argument("--run", required=True)
    ap.add_argument("--follow", action="store_true", help="tail a live run")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--config", type=str, default=None,
                    help="JSON blob recorded with the run (arm, model, seed, ...)")
    ap.add_argument("--trackio", action="store_true",
                    help="also stream to Trackio (opt-in: init can block)")
    args = ap.parse_args()

    cfg = json.loads(args.config) if args.config else None
    n = bridge(args.log, args.out, args.project, args.run, args.follow, args.poll, cfg,
               trackio=args.trackio)
    print(f"{n} steps -> {args.out}")


if __name__ == "__main__":
    main()
