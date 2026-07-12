"""The sweep: does conversational-feature steering help on GPQA and MATH-Hard?

The paper's causal claim rests on one feature, at one layer, in one 8B distill,
on one task (Countdown), where steering took accuracy from 27.1% to 54.8%.
Countdown is a search puzzle with a weak baseline and enormous headroom, and its
reward is exactly what the model was RL'd on in the paper's other experiments. A
doubling there is compatible with "steering induces a society of thought that
explores solution space better" -- and equally compatible with "jostling the
residual stream makes an 8B model enumerate more candidates before committing."

GPQA-Diamond and MATH Level-5 discriminate between those. Both are in the paper's
own benchmark suite, neither has Countdown's headroom, and neither rewards blind
enumeration. If the effect is a general reasoning mechanism it should survive. If
it is a Countdown artifact it should vanish -- and the matched controls should
look just like the candidates.

Every condition, including the unsteered baseline, runs through the same code
path with the same seeds, so accuracy differences cannot come from batching or
sampling differences.

Resumable: completed (task, layer, feature, alpha, pid, sample) cells are skipped
on restart, so a long grid can be interrupted freely.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_problems
from .features import ANCHOR_FEATURE, Feature, load_features, select
from .grade import grade
from .sae import load_sae, neuronpedia_source_id
from .steering import steer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Cheap lexical proxies for the paper's LLM-judged "conversational behaviours".
# NOT the paper's measure -- we do not run a Gemini judge here. These let us check
# the mechanism claim (does steering actually make traces more dialogic?) without
# an API dependency, and let the analysis test whether any accuracy change is
# accompanied by the behaviour change the paper's story requires.
MARKERS = {
    "self_interrupt": re.compile(r"\b(wait|hold on|hmm+|oh)\b", re.I),
    "alternative": re.compile(r"\b(alternatively|another (?:idea|approach|way)|but what if|on the other hand)\b", re.I),
    "contradiction": re.compile(r"\b(but|however|actually|no,|that'?s (?:not|wrong))\b", re.I),
    "question": re.compile(r"\?"),
    "first_person_plural": re.compile(r"\b(we|us|let'?s|our)\b", re.I),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["gpqa", "math_hard"],
                    choices=["gpqa", "math_hard", "countdown"])
    ap.add_argument("--layers", nargs="+", type=int, default=[15])
    ap.add_argument("--mixture", default="slimpj",
                    help="slimpj = the paper's SAE (layer 15 only). mixed = all 32 layers.")
    ap.add_argument("--features", nargs="+", type=int, default=None,
                    help="explicit feature ids; default = auto-select candidates + matched controls")
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--select-method", default="neighbors", choices=["neighbors", "lexicon"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                    help="steering strength in units of the feature's max activation. "
                         "The paper's s=+-10 on feature 30939 (max act 5.906) is alpha ~ +-1.7.")
    ap.add_argument("--raw-strengths", nargs="+", type=float, default=None,
                    help="override alphas with raw SAE-space strengths (the paper's units)")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--samples", type=int, default=1, help="samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)  # paper's setting
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scope", default="all", choices=["all", "generated"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/sweep.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("results/npcache"))
    ap.add_argument("--save-traces", action="store_true")
    ap.add_argument("--hook-point", type=Path, default=Path("results/hook_point.json"))
    args = ap.parse_args()

    _check_math_verify(args.tasks)

    hook_layer_offset = _resolve_hook(args.hook_point)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(args.out)
    if done:
        print(f"resuming: {len(done)} attempts already recorded in {args.out}")

    for layer in args.layers:
        sae = load_sae(layer, args.mixture, device=device)
        source = neuronpedia_source_id(layer, args.mixture)
        feats = load_features(source, args.cache)
        _apply_calibration(feats, layer, args.mixture)

        if args.features:
            chosen = [feats[i] for i in args.features]
            for f in chosen:
                f.role = f.role or "explicit"
        else:
            chosen = select(
                feats,
                n_candidates=args.n_candidates,
                n_controls=args.n_controls,
                method=args.select_method,
                anchor=ANCHOR_FEATURE if layer == 15 else _layer_anchor(feats),
                seed=args.seed,
            )

        print(f"\n=== layer {layer} ({source}) ===")
        for f in chosen:
            print(f"  [{f.role:16s}] f{f.index:<6d} maxact={f.max_act:5.2f} "
                  f"freq={f.frac_nonzero:.5f} conv={f.conv_score:+.3f}  {f.description[:60]}")

        for task in args.tasks:
            problems = load_problems(task, n=args.n_problems, seed=args.seed)
            print(f"\n--- {task}: {len(problems)} problems ---")

            # Baseline once per (task, layer): unsteered, but identical code path.
            _run_condition(
                model, tok, sae, problems, task, layer,
                feature=None, role="baseline", alpha=0.0, strength=0.0,
                args=args, done=done, hook_layer_offset=hook_layer_offset,
            )

            for f in chosen:
                strengths = (
                    [(None, s) for s in args.raw_strengths]
                    if args.raw_strengths
                    else [(a, a * f.max_act) for a in args.alphas]
                )
                for alpha, strength in strengths:
                    if abs(strength) < 1e-9:
                        continue  # alpha=0 is the baseline, already run
                    _run_condition(
                        model, tok, sae, problems, task, layer,
                        feature=f, role=f.role, alpha=alpha, strength=strength,
                        args=args, done=done, hook_layer_offset=hook_layer_offset,
                    )

    print(f"\ndone -> {args.out}")


def _run_condition(model, tok, sae, problems, task, layer, feature, role, alpha,
                   strength, args, done, hook_layer_offset):
    fid = feature.index if feature is not None else -1
    key_prefix = (task, layer, fid, round(strength, 6))

    todo = [
        (p, s) for p in problems for s in range(args.samples)
        if (*key_prefix, p.pid, s) not in done
    ]
    if not todo:
        print(f"  [skip, complete] f{fid} strength={strength:+.2f}")
        return

    delta = None
    if feature is not None:
        delta = sae.steering_vector(feature.index, strength)

    label = "baseline" if feature is None else f"f{fid} ({role}) a={alpha} s={strength:+.2f}"
    t0 = time.time()
    n_correct = n_parsed = 0

    hook_layer = layer + hook_layer_offset

    with args.out.open("a") as fh:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i : i + args.batch_size]
            prompts = [_chat(tok, p.prompt) for p, _ in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)

            torch.manual_seed(args.seed + i)  # same draws across conditions
            with steer(model, hook_layer, delta, scope=args.scope):
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tok.pad_token_id,
                    )

            gen = out[:, enc["input_ids"].shape[1] :]
            for (p, sample), row in zip(chunk, gen):
                text = tok.decode(row, skip_special_tokens=True)
                g = grade(task, text, p.answer)
                n_correct += g.correct
                n_parsed += g.parsed
                rec = {
                    "task": task, "pid": p.pid, "sample": sample,
                    "layer": layer, "hook_layer": hook_layer,
                    "feature": fid, "role": role,
                    "alpha": alpha, "strength": strength,
                    "correct": bool(g.correct), "parsed": bool(g.parsed),
                    "pred": g.pred, "gold": p.answer,
                    "n_tokens": int((row != tok.pad_token_id).sum()),
                    "truncated": len(row) >= args.max_new_tokens,
                    "markers": {k: len(r.findall(text)) for k, r in MARKERS.items()},
                }
                if args.save_traces:
                    rec["trace"] = text
                fh.write(json.dumps(rec) + "\n")
            fh.flush()

    n = len(todo)
    print(f"  {label:44s} acc={n_correct/n:6.1%}  parsed={n_parsed/n:6.1%}  "
          f"({n} gens, {time.time()-t0:.0f}s)")


def _chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def _completed(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a killed run
            done.add((r["task"], r["layer"], r["feature"], round(r["strength"], 6),
                      r["pid"], r["sample"]))
    return done


def _resolve_hook(path: Path) -> int:
    """Offset added to the SAE's layer index to get the layer whose output we hook."""
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run `python -m sot.validate_hook` first -- it determines "
            "whether this SAE attaches to resid_post (layer L's output) or resid_pre "
            "(layer L-1's output). The two sources disagree, and steering the wrong "
            "tensor invalidates the whole sweep."
        )
    info = json.loads(path.read_text())
    offset = info["hook_layer"] - info["layer"]
    print(f"hook point: {info['hook_point']} (offset {offset:+d} from SAE layer)")
    return offset


def _apply_calibration(feats: dict[int, Feature], layer: int, mixture: str) -> None:
    """Replace Neuronpedia's max activations with ones measured in our own units.

    Neuronpedia's activation scale differs from the SAE's by a constant factor
    (~3.2x here). Feature SELECTION is unaffected -- it only compares features to
    each other, and the factor is common to all of them. But steering STRENGTH is
    absolute: alpha is a multiple of max activation, and using Neuronpedia's number
    would make every intervention ~3x weaker than intended, with nothing to signal
    that anything went wrong.
    """
    path = Path(f"results/feature_stats_L{layer}_{mixture}.json")
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run:\n"
            f"  python -m sot.calibrate --layer {layer} --mixture {mixture}\n"
            "It measures each feature's max activation in the same units the steering "
            "hook uses. Neuronpedia's published values are on a different scale and "
            "would silently mis-size every intervention."
        )
    stats = json.loads(path.read_text())
    max_act = stats["max_act"]
    frac = stats["frac_nonzero"]
    for i, f in feats.items():
        if i < len(max_act):
            f.max_act = float(max_act[i])
            f.frac_nonzero = float(frac[i])
    print(f"calibrated max activations from {path} ({stats['n_tokens']} tokens)")


def _layer_anchor(feats: dict[int, Feature]) -> int:
    """For layers other than 15 there is no published anchor, so use the most
    conversational feature by lexicon as the anchor for neighbour search."""
    from .features import score_conversationality

    score_conversationality(feats, "lexicon")
    return max(feats.values(), key=lambda f: f.conv_score).index


def _check_math_verify(tasks) -> None:
    if "math_hard" not in tasks:
        return
    try:
        import math_verify  # noqa: F401
    except ImportError:
        print("WARNING: math_verify not installed. MATH grading will fall back to "
              "normalized string match, which under-counts correct answers "
              "(equally across conditions, so it biases toward finding no effect). "
              "Install with: uv pip install math-verify")


if __name__ == "__main__":
    main()
