# Does reasoning work by simulating a "society of thought"?

A replication and stress-test of **"Reasoning Models Generate Societies of Thought"**
(Kim, Lai, Scherrer, Agüera y Arcas & Evans, Google / UChicago / Santa Fe Institute,
[arXiv:2601.10825](https://arxiv.org/abs/2601.10825), Jan 2026).

The paper ships **no code and no data**. Everything here is rebuilt from public artifacts.

---

## The claim we're testing

When a reasoning model like DeepSeek-R1 "thinks", its chain of thought reads like a
*conversation* — it asks itself questions, changes its mind, argues, and reconciles.
Instruction-tuned models (the same base model, different post-training) don't do this;
they produce one-sided monologue.

The paper says this dialogue **is the mechanism**: reasoning works *because* the model
simulates diverse internal voices that debate. Its evidence comes in two parts.

**Part A — descriptive (we don't dispute this).** Across 8,262 problems, R1's traces
contain far more question-asking, perspective-shifting, and disagreement than V3's, even
controlling for how long the traces are. The effect is large and holds at every model size.

**Part B — causal (this is what we test).** They find a single feature inside the model
that fires on conversational surprise markers — the "Oh!" feature — and *turn it up*.
Accuracy on a puzzle task **doubles**, 27.1% → 54.8%. They read this as: induce more
society-of-thought, get more reasoning.

**The gap:** Part B was only ever run on **one task** — Countdown, an arithmetic puzzle.
The benchmarks that make it look like a general result (GPQA, MATH) were only *observed*,
never intervened on.

---

## What we did

### Experiment 1 — Steering (the main result). ✅ Complete

Turn the "Oh!" feature up and down inside DeepSeek-R1-Distill-Llama-8B and measure
accuracy. Three things the paper didn't do:

1. **Sweep the strength** instead of using one setting. (The paper's setting is
   ambiguous enough to mean two different doses; both are on our ladder.)
2. **Run it on the paper's own benchmarks** — GPQA-Diamond and MATH-Hard — not just
   Countdown.
3. **Compare against matched control features.** The paper's controls weren't matched on
   how often a feature fires or how strongly, so "this is just a bigger poke" was never
   ruled out. Ours are matched on both.

**Result: the effect is a Countdown artifact.**

| | Countdown (paper's task) | MATH-Hard (paper's benchmark) |
|---|---|---|
| baseline | 24.0% *(paper: 27.1% — we reproduce)* | 62.0% |
| the paper's feature, turned up | **+10.0 pts** | **−22.0 pts** |

The same feature, at the same strength, **helps on Countdown and wrecks MATH**. Turn it up
further and the model degenerates into literal babble (*"cyclochoh! Wait, no, wait, no,
wait..."*).

And the paper's own mechanism story fails: steering **does** make the traces more dialogic
(self-interruptions +36%, contradictions and questions up) — and accuracy falls anyway.
**You can induce the society of thought and get a dumber model.**

Why? Countdown is a *search* puzzle — combine 3–4 numbers to hit a target. The way to win
is to try more candidate expressions. On a task like that, "poke the model into trying more
things" and "make the model reason better" are indistinguishable. On MATH you can't
brute-force your way to the answer, and the poke just makes it ramble.

→ **[Full findings, with all the numbers](results/steering/FINDINGS.md)**

### Experiment 2 — Reinforcement learning (Claim B). 🔄 In progress

The other half of the paper, and the only part that touches *real* RL rather than a
pre-trained model. Take a base model (Qwen-2.5-3B), fine-tune it two ways over **identical
problems with identical correct answers**:

- **dialogue** — traces written as several experts talking it through
- **monologue** — the same solutions, written as one voice

Then run identical RL on both. The paper says the dialogue-primed model learns faster. If
true, the social account survives our steering result — the mechanism would be real but
*mislocated*: it emerges in training rather than living in a steerable feature.

We also do what the paper didn't: **≥3 random seeds per arm.** Their headline is an
early-training gap, which is exactly where seed noise is largest, and they appear to report
single runs.

---

## The bugs worth knowing about

Six silent failures, none of which raised an error, each of which would have produced a
confident wrong answer. They're documented because **anyone replicating this paper will hit
them**:

1. **Neuronpedia lists the wrong hook point** for this SAE (`resid_pre`; it's actually
   `resid_post`). Replicate from the published metadata and you steer the wrong layer.
2. **The first token is an attention sink** (norm 466 vs 11 for everything else). Include
   it and every measurement is garbage.
3. **The activation function is JumpReLU, not ReLU.**
4. **Neuronpedia's activation scale is ~2.5× off the SAE's own** — size your steering from
   it and every intervention is 2.5× too weak.
5. **The model answers in LaTeX `\boxed{}`, not the `<answer>` tag the prompt demands.**
   Grading only the tag scored 74% of *correct* traces as unparseable and put our baseline
   at 5.5% instead of 24% — which looks exactly like "the paper doesn't reproduce."
6. **`truncated` measured the padded batch, not the sequence** — 96% "truncation" that was
   really 12%.

The lesson, and the reason for the test suite: *an unparseable answer must score **wrong**,
never be dropped.* Otherwise a degenerating model "improves" by shrinking its own
denominator.

---

## Running it

```bash
./scripts/setup.sh                  # uv venv + torch + deps

./scripts/run_stages.sh hook        # REQUIRED: resolve the hook point by reconstruction
./scripts/run_stages.sh calibrate   # REQUIRED: measure activation scales in OUR units
./scripts/run_stages.sh control     # the Countdown dose-response
./scripts/run_stages.sh main        # GPQA + MATH-Hard with matched controls

python -m rl.generate_sft           # dialogue/monologue SFT data (verified, matched)
python -m rl.train_grpo --arm baseline --seed 42     # Gate 2: does RL learn at all?

pytest tests/                       # 53 tests
```

Stages are **gates**. If the Countdown control doesn't reproduce ~24%, the harness is
wrong and nothing downstream means anything — fix that first.

Needs one GPU with ≥20GB. **Don't run this on a unified-memory box** (DGX Spark): a GPU
over-allocation there starves the OS and takes the whole machine down, rather than just
killing your job. We learned that twice.

## Layout

```
sot/            steering: SAE loading, the hook, calibration, grading, the sweep
rl/             the RL half: SFT data generation, GRPO training, provenance checking
tests/          53 tests — the graders, the SAE maths, the hook, the arm-matching
scripts/        staged runners + RunPod provisioning
results/        raw traces (5,664 attempts) and findings
```

## Provenance of the artifacts

- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (MIT)
- SAEs: `OpenMOSS-Team/Llama-Scope-R1-Distill` (Apache-2.0) — the paper's is the
  `800M-Slimpajama-0-OpenR1-Math-220k/L15R` subdirectory
- Feature labels + firing stats: Neuronpedia's S3 export (GPT-4o-mini autointerp)
- GPQA: `fingertap/GPQA-Diamond` (the official one is gated) · MATH: `lighteval/MATH-Hard`
  · Countdown: `Jiayi-Pan/Countdown-Tasks-3to4`

## The caveat that applies to us *and* to the paper

`DeepSeek-R1-Distill-Llama-8B` **was never RL'd**. It's Llama-3.1-8B fine-tuned to *imitate*
R1's outputs. It's the only reasoning model with a public SAE, so it's what the paper used
and what we used.

Which means the field's mechanistic understanding of reasoning models currently rests on a
model that is an *impersonation* of one. Training an SAE on a genuinely RL'd reasoner
(QwQ-32B) is the obvious next step, and nobody has done it.
