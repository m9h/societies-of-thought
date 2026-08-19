# HANDOFF — societies-of-thought

*State as of 2026-08-18. Branch `analysis/ddm-pilot`, **5 commits ahead of `main`**
(head `3564f18`). 331 tests pass, 3 skipped. RunPod balance **$234.45**, nothing running,
no pods. Read "The decision waiting for you" last — everything else is context for it.*

---

## 1. What this project is

An adversarial replication of **"Reasoning Models Generate Societies of Thought"**
(Kim, Lai, Scherrer, Agüera y Arcas & Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)).
The paper ships **no code and no data**; everything here is rebuilt from public artifacts,
and every claim is run against the control that could kill it.

It is the adversarial-replication arm of a five-repo programme (the same table is in each
repo's README): `spinning-up-in-mech-interp` (curriculum), `jacobian-lens` (research),
`tri-lens` (instrument agreement), **`societies-of-thought`** (this),
`controls-and-trajectories` (published datasets).

---

## 2. Results, with the numbers

### C1 — steering (done, and **corrected**)
Steering feature 30939 gives **+10 pts on Countdown** and **−22 on MATH-Hard**, an
inverted-U in dose (24% → 34% at α=1.0 → **3.5%** at α=1.693).

> ⚠ **A published-in-repo result was RETRACTED here.** We claimed the induced society is
> *redundant* (diversity falls 0.236 → 0.190). It **rises**, 0.124 → 0.182. The old
> analysis dropped traces with too few segments, and since steering *creates* shifts that
> discarded **47.5% of baseline vs 4% at top dose**. See
> `results/steering/RECHECK_length_and_filtering.md`. The argument survived and improved:
> diversity and accuracy are **decoupled** — max diversity coincides with near-total failure.

### C2/C3 — diversity and mediation (done, **null at every level**)
| test | result |
|---|---|
| QwQ math, unadjusted | correct **less** diverse ✱ |
| QwQ math, length-matched | **no difference** (99% shrinkage) |
| non-math pooled, length-matched | +0.0066 ✱ (small) |
| GPQA, length-matched | +0.0110 ✱ (1,003 pairs) |
| **GPQA, within-problem** | **+0.0023 [−0.0032, +0.0078]** — no difference |

The two significant results point in **opposite directions** and each dissolves under its
own control (length; problem identity). `results/qwq/FINDINGS.md`,
`results/qwq/GPQA_within_problem.md`.

**Bonus finding:** QwQ per-problem accuracy is **bimodal, not binomial** — 22% of GPQA
problems mixed where independence predicts 94%. It either knows a problem or doesn't.

### C5 — Claim B, faithful replication (done, **n=1**)
Qwen2.5-3B, 3 arms, 250 steps, the paper's teacher/prompts/out-of-domain pool,
`rollout.n=4`, eval every 10 steps.

| step 250 | baseline | dialogue | monologue |
|---|---|---|---|
| reward | 0.661 | **0.653** | **0.671** |

Dialogue leads monologue +0.043 through step 60, caught by step 70, **finishes last**.
We reproduce their only per-arm number (step 40: theirs 38/28, ours 37.7/30.4) and their
baseline (0.5665 vs our 0.661). **The surviving variable is priming, not dialogue.**
`results/claimB/RESULT_faithful.md`.

### DDM pilot (external critique, on this branch) + our MATH extension
A collaborating agent fit drift-diffusion models to our data (`analysis/ddm_pilot/`).
It **independently reproduced our between-problem-leakage result** on a different
instrument (grand-mean +0.100 P=1.000 → within-problem +0.0065 P=0.61), and found that
what survives within-problem is a **boundary** effect: more diverse traces *stop earlier*.

Its §1 pre-registered a discriminating test. **We ran it today:**

| MATH-Hard fit | Δ boundary | Δ drift | P(v↑) |
|---|---|---|---|
| flat | +0.156 (P=0.936) | **−0.336** | **0.001** |
| near-ceiling dropped | +0.049 (P=0.686) | **−0.282** | **0.010** |
| hierarchical (1\|pid) | +0.271 (P=0.969) | **−0.460** | **0.000** |

Pre-registration: *exhaust story → drift DOWN; paper's story → drift UP*. **Drift falls.**
Survives the censoring sensitivity that would have rescued the paper. **Δa is NOT
established** (swings across fits) — quote drift only. `analysis/ddm_pilot/RESULTS_math.md`.

**Matched controls (also today):** none of six sparsity/max-act-matched features reaches
significance; paper's feature sits **4.4 control-SD** below the control mean (control Δv
mean −0.071 sd 0.061; control acc 57.5% vs 62% baseline vs **40%** for 30939). So the
lever is **real and specific** — and on MATH it points the wrong way.

### C6 — transfer (**built, never run**)
LIAR2 (`chengxuphd/liar2`, 22,962 claims vs the paper's 23,299), 6→3 recoding verbatim,
reward `0.9·acc + 0.1·format`. Data built: 18,369 train / 1,024 test. **25 tests.**
Majority-class floor is **58%** (always "false").
Two details that took re-reading the paper: C6 primes on **Countdown** dialogues (not the
OOD pool C5 uses), and the paper runs **only 2 arms** — no monologue. That missing arm is
our contribution: it separates *dialogue transfers* from *any priming transfers*.

---

## 3. The critique's issues — audited, not assumed

| item | status |
|---|---|
| §1 MATH discriminating fit | ✅ **DONE today** |
| §2 censored-likelihood DDM | ❌ **not done** — the important one |
| §3 dose-dependent lapse | ❌ not done |
| §3 BBH within-problem | ❌ **NOT RUNNABLE** — BBH is 1 sample/problem, 0 mixed-outcome |
| §3 Claim-B checkpoints as LaRT rows | ❌ not done |
| §4 `pid` reconstructed from row order | ✅ **FIXED today** |
| §4 `hse_recheck` vs `_rg` | documented |
| §4 GPQA non-convergence | understood (79% separation) |

**§2 matters most.** Censoring is 15%→31% on MATH and their Countdown sensitivity showed
it moves real credit between *a* and *v* — it is why our Δa is unstable. Route: custom WFPT
survivor-function likelihood in HSSM/PyMC.

---

## 4. The decision waiting for you

**Full replication with seeds costs ~$1,001. Balance is $234.45. Shortfall $766.**

Every RL arm is 13.9 h × $2.38 (measured, not estimated). Unrun: C4 entirely, C5 on Llama,
C6 entirely; C5 Qwen is n=1. Three seeds × 3 arms × 2 models is nine runs per claim.

Two options fit the balance — **not both**:

- **C5 Qwen +2 seeds (~$200)** — puts error bars on the result we lead with. Dialogue −
  monologue is −0.006 and currently unbounded. *My recommendation:* an n=1 headline is the
  weakest thing in the writeup.
- **C6 Qwen 3 arms, n=1 (~$100)** — closes the last untested claim. But it would add a
  second unbounded result.

I cannot deliver "fully replicated with sufficient seeds" on this balance and will not
quietly run one arm and call it seeded.

---

## 5. Sharp edges — do not rediscover these

- **`main` is 5 commits behind this branch.** Merge or cherry-pick before publishing.
- **Answer extraction has broken 5 times.** Markdown (`**Answer:** (F)`), truncation,
  letter-vs-option-text, `\boxed{}`, prose prefixes. `rl/pool_build.is_correct` now handles
  all; tests use *verbatim observed strings*.
- **Truncation is not neutral.** A truncated trace is max-length AND graded wrong —
  the length/correctness confound in pure form. Always exclude via `finish_reason`.
- **Single-voice traces score 0, never dropped** (the paper's own rule). Dropping them
  inverted *two* results.
- **verl's Countdown scorer needs `"Assistant:"`** in the prompt or every rollout scores 0.
  This cost a 170-step run that looked like a model that couldn't learn.
- **Robustness ≠ validity.** We ran sampling/size/embedder robustness and reported "SIGN
  STABLE across 3 runs". It meant nothing — none touched length.
- **Modal, not hand-rolled pods** (`feedback_use_modal_not_runpod_pods`). If a pod needs
  SSH it MUST get a `PUBLIC_KEY` env var.
- **Chunked persistence**: a Modal timeout at 92% cost ~16 GPU-hours for zero output.
  `rl/chunking.py` + `retries=0`.
- **Primed checkpoints now publish to HF before PPO.** The first run's were lost with the
  pods; `SAVE_FREQ=-1` is correct for *PPO* checkpoints only.
- **Trackio init is thread-affine** — init and log must share a thread. On by default,
  time-boxed behind a queue.

## 6. Method that actually worked

Write the analysis **before** the data exists. `within_problem.py` was red across 9 tests
first, and that is why the difficulty confound was caught rather than published. Three
results were retracted or corrected this cycle — every one produced by building the
analysis after seeing the numbers.

## 7. Cross-project

`NOTE_FROM_SOT_AGENT.md` in `~/Workspace/jacobian-lens` carries the retraction, three
transferable failure mechanisms, the within-item control, and infrastructure lessons to the
GWT agent. Our open instrument-agreement question (HSE vs the paper's LLM judge on shared
inputs) belongs with `tri-lens`.
