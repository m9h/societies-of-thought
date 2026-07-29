# Claim B / C5 — faithful replication, completed

*Qwen2.5-3B, three arms, 250 steps, completed 2026-07-29. Data:
`curves_paper_faithful.json`, per-step metrics in `metrics_paper/*.jsonl`. Fidelity
spec: `rl/paper_spec.py`. Deviation ledger: `docs/paper_fidelity_audit.md`.*

## The claim under test

> fine-tuning models with conversational scaffolding **substantially accelerates**
> reasoning improvement compared to base models and models fine-tuned with monologue-like
> reasoning — *abstract*

> both conditions are trained on identical problems and correct answers, yet
> conversation-fine-tuned models consistently improve faster and **reach higher asymptotic
> accuracy** — *main text*

> Models initially fine-tuned with multi-agent dialogues (red) reach high accuracy faster
> than models fine-tuned with monologue-style reasoning (blue), **though both eventually
> converge** — *Extended Data Fig. 8 caption*

The main text and the figure caption disagree. This run adjudicates.

## Result

| step 250 | reward |
|---|---|
| baseline (no priming) | 0.661 |
| **dialogue-primed** | **0.653** |
| **monologue-primed** | **0.671** |

Windowed means across the run:

| window | dialogue − monologue | dialogue − baseline | monologue − baseline |
|---|---|---|---|
| **early (10–60)** | **+0.043** | +0.058 | +0.015 |
| **mid (70–180)** | −0.006 | +0.034 | **+0.040** |
| **late (190–250)** | −0.011 | +0.003 | +0.014 |

**The dialogue advantage is real, early, and transient.** Dialogue leads monologue by
+0.043 through step 60 while monologue tracks the un-primed baseline. By step 70 monologue
has caught up entirely; from there the two primed arms are inseparable (mean gap −0.006),
and dialogue finishes **last of the three**.

The variable that survives to the end is **priming, not dialogue**. Both primed arms sit
~+0.04 over baseline through the middle of training; which *form* the priming took stops
mattering after step 60.

## We reproduce the paper's number, and reject the inference

The paper states exactly one per-arm Qwen comparison, at step 40:

| step 40 | paper | ours |
|---|---|---|
| dialogue | 38% | **37.7%** |
| monologue | 28% | **30.4%** |
| gap | +10.0 | **+7.3** |

Their number holds. So does their baseline: they report a PPO reward of **0.5665** at 250
steps; ours finished at **0.661** on the same metric and task.

What does not hold is the generalisation from that snapshot. Their figure caption — *"both
eventually converge"* — is what our curve shows. Their abstract and main text are not.

## Why this run settles what earlier ones could not

Three previous runs (in-domain Qwen, in-domain Llama, and this design) all showed the same
decay, but only this one can attribute it, because only this one primes **out of domain**:

- Earlier runs primed on **Countdown itself**, so both arms received task knowledge and
  convergence was over-determined. That tested something the paper did not claim.
- This run primes on **BBH/GPQA/MATH-Hard/MMLU-Pro/MUSR** and RLs on Countdown. Nothing
  about Countdown is in the priming data. Only *form* can transfer — which is the paper's
  actual claim.

Also matched here for the first time: the paper's **teacher** (Qwen2.5-32B-Instruct, not
the 72B we substituted), the **verbatim generation prompts**, `rollout.n=4` (was 1),
SFT 5 epochs / batch 64 / ctx 2048 / 10% warmup (was 3/32/1536/3%), PPO batch 128 / val 640
/ max_prompt 1024, and eval **every 10 steps** (was 25 — too coarse to even resolve their
step-40 claim). A fidelity gate (`tests/test_paper_fidelity.py`) ran **on the pod** before
any GPU spend.

## Interpretation — what priming is doing

Priming here is weight-level SFT on **500 out-of-domain examples** (~100k tokens) before RL
begins, with the prompt masked from the loss, initialising both actor and critic. It cannot
be transferring task knowledge — wrong domain, and far too small.

The mechanism most consistent with the data is that **priming installs the output contract,
which fixes exploration**. Measured directly on this base model: Qwen2.5-3B emits an
`<answer>` tag in only **48/64** completions under the scaffolded prompt (16/64 without it).
Under the stock scorer a rollout with no `<answer>` scores exactly **0** — no partial credit
and no gradient direction. So an un-primed policy spends early RL with many rollouts
carrying no learning signal, while a primed policy is gradable from step 1. That predicts
precisely what we see: a real early advantage that decays as RL teaches the format anyway.

A second mechanism is not separable in this design: the primed weights also initialise the
**critic**, so a primed run starts with lower-variance advantages independent of policy
format.

On this account "dialogue vs monologue" is the wrong axis. Both formats teach a bounded,
terminated, answer-containing response; dialogue does it slightly faster, perhaps because
its traces are 1.75× longer and contain more instances of the contract per example.

### Independent support

[arXiv 2607.22925](https://arxiv.org/abs/2607.22925) (Baherwani, Goldstein & Panda) finds
frontier models gain up to ~11 points from **semantically irrelevant filler tokens**, and
speculates the filler span acts as *"a workspace over which different latent computations
can be conditioned"* — an internal search mechanism, reached with no dialogic content at
all. Their Appendix G reports that **SFT fails to transfer** a filler benefit, because "the
tokens themselves carry no transferable signal." Claim B *is* an SFT-then-RL design, and our
curve has the shape their result predicts. See `docs/related_work_2026.md` §3.

## Limits — read before using any of this

- **n = 1 per arm.** No error bars. The late-window gaps (|Δ| ≤ 0.02) are within plausible
  run-to-run noise; the early dialogue lead (+0.043 over 6 checkpoints) is the one
  difference large and sustained enough to lean on.
- **The length confound is unresolved.** Dialogue traces are **1.75×** longer (252 vs 144
  median words). The early advantage is confounded with tokens spent. The decisive test —
  replacing dialogue content with length-matched random tokens, per the trace ablation in
  2607.22925 — has not been run.
- **The monologue arm was interrupted twice**: once by a genuine silent death at step 14,
  once by an over-aggressive watchdog at step 41. Both restarts began from scratch, so the
  reported curve is a single uninterrupted 250-step run, but the arm consumed more wall
  clock than the others.
- **Llama-3.2-3B has not been re-run** under this faithful configuration. The paper's Llama
  result (dialogue 40% vs monologue ~18% plateau at step 150) is its strongest, and it
  length-matches the Llama dialogue condition — a control we now implement
  (`concatenate_personas`) but have not yet used in a run.
- **Declared deviations**: pool problem identity, GPQA via an open mirror, and the RL prompt
  scaffold. See `rl/paper_spec.DEVIATIONS`.

## What would move this next, in order

1. **Format-compliance probe on the primed checkpoints** — measure `<answer>` emission rate
   for base vs dialogue-primed vs monologue-primed. If priming takes it to ~100% while both
   forms land together, the whole effect localises to gradability rather than reasoning.
   Minutes of GPU; requires re-priming (~$3) since the checkpoints were terminated with the pod.
2. **The length-matched trace ablation** — replace persona content with random tokens of
   matched length at evaluation. Distinguishes "content is load-bearing" from "length is".
3. **Llama-3.2-3B, faithful, with persona concatenation** — their strongest claim, with
   their own length control.
4. **Seeds.** Three per arm on Qwen to put error bars on the early gap.
