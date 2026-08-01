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


class _TrackerThread:
    """Own the tracker on a single background thread and talk to it through a queue.

    Trackio's init state is thread-affine: initialising on one thread and logging from
    another raises "Call trackio.init() before trackio.log()". So the thread that inits
    must also be the thread that logs -- a naive timeout wrapper around init alone is
    both broken and useless, which our tests caught immediately.

    Everything the tracker does happens here, and the caller only ever enqueues. A slow,
    hanging or exploding tracker therefore costs a background thread and nothing else;
    the JSONL sink in `bridge` is untouched. Trackio was measured initialising in ~0.6s,
    so this should never engage -- it exists because when it did hang, three training
    arms recorded zero metrics while the process looked perfectly alive.
    """

    def __init__(self, project: str, run: str, config: dict | None, timeout: float):
        import queue
        import threading

        self._q: "queue.Queue" = queue.Queue(maxsize=10_000)
        self._ready = threading.Event()
        self._ok = False
        self._thread = threading.Thread(
            target=self._serve, args=(project, run, config), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=timeout)
        if not self._ok:
            print("tracker not ready; JSONL only")

    def _serve(self, project, run, config):
        tracker = _open_tracker(project, run, config)
        self._ok = tracker is not None
        self._ready.set()
        if tracker is None:
            return
        while True:
            item = self._q.get()
            if item is None:
                try:
                    tracker.finish()
                except Exception:
                    pass
                return
            payload, step = item
            try:
                tracker.log(payload, step=step)
            except Exception:
                pass          # a broken tracker must never interrupt the run

    def log(self, payload: dict, step: int) -> None:
        if not self._ok:
            return
        try:
            self._q.put_nowait((payload, step))
        except Exception:
            pass              # full queue: drop the point, keep the run

    def finish(self) -> None:
        if self._ok:
            try:
                self._q.put_nowait(None)
            except Exception:
                pass


def bridge(log: Path, out: Path, project: str, run: str, follow: bool,
           poll: float = 5.0, config: dict | None = None,
           trackio: bool = True, tracker_timeout: float = 60.0) -> int:
    """Stream `log` into `out` (JSONL), and optionally into Trackio.

    JSONL is the durable artifact and is opened FIRST; Trackio is on by default but
    time-boxed, so a slow or broken tracker degrades to JSONL rather than stalling the
    run. Trackio init was measured at ~0.6s, and the guard exists because when it did
    hang, three training arms recorded zero metrics while the process looked alive.
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
    tracker = (_TrackerThread(project, run, config, tracker_timeout)
               if trackio else None)
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
                            tracker.log({k: v for k, v in rec.items() if k != "step"},
                                        rec["step"])
                    pos = fh.tell()
            if not follow:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        if tracker is not None:
            tracker.finish()
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
    ap.add_argument("--no-trackio", action="store_true",
                    help="write JSONL only; Trackio is on by default and time-boxed")
    args = ap.parse_args()

    cfg = json.loads(args.config) if args.config else None
    n = bridge(args.log, args.out, args.project, args.run, args.follow, args.poll, cfg,
               trackio=not args.no_trackio)
    print(f"{n} steps -> {args.out}")


if __name__ == "__main__":
    main()
