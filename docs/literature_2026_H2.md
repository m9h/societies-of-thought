# Literature review, 2026 H2 — and what it does to this project

*Written 2026-10-06. Every paper below **post-dates** arXiv:2601.10825 (15 Jan 2026), which
is still at **v1** with no code, no data and no journal reference. A co-author told us in
September that code ships with the revised version. That has not happened, so our window is
still open.*

---

## 1. The field formalised our central claim, at scale, after the paper came out

Our strongest result is methodological: Fig. 4's answer depends on which instrument you
use, and the instrument is an unpublished LLM-judge prompt. Two papers now establish that
as a general property of LLM judges, with far more data than we have.

**Norman, Rivera & Hughes, "Reliability without Validity" ([2606.19544](https://arxiv.org/abs/2606.19544), 17 Jun 2026).**
21 judge models, 3 benchmarks, ~541,000 judgments.

- **Kappa deflation of 33–41 percentage points** between exact-match agreement and Cohen's
  κ on MT-Bench. Exact match "does not correct for chance and systematically overstates
  discriminative ability."
- **Judge rankings shift by up to 14 positions** across benchmarks.
- A **consistency–bias paradox**: production judges with test–retest reliability > 0.95
  *and* position bias > 0.10. Reliable, and not valid.
- They propose a "Minimum Viable Validation Protocol."

**Yang, Hou & Yang, "When the Judge Changes, So Does the Measurement" ([2607.08535](https://arxiv.org/abs/2607.08535), 9 Jul 2026).**
The abstract is the finding: *"An LLM-as-judge score can move even when the candidate
responses stay fixed, simply because the evaluator has changed."* Judge *upgrades* are not
monotone — only Qwen3 1.7B→4B gave reliable adjacent improvement; MiniMax adjacent releases
were poorly interchangeable. Recommended reporting standard: dataset slices, bias probes,
error-dependence estimates, protocol audit trails.

Also relevant: **"A Judge Should Know What Changed: Construct Validity for LLM-as-a-Judge"**
([2608.24419](https://arxiv.org/abs/2608.24419)).

### What this does to the SoT paper

Their judge validation — ICC(3,1) ≈ .85 vs GPT-5.2, ≈ .76 vs human raters, plus the
Intelligence Squared Debates corpus (N = 1,196) — was reasonable for January 2026. By the
second half of 2026 it is **insufficient rather than wrong**: Norman et al. show precisely
that high inter-judge agreement coexists with severe bias, so ICC ≈ .85 does not license
treating a judge's persona count as a measurement. Fig. 4b and 4e rest entirely on that
instrument, and its prompt is unreleased.

### What this does to *us*

It raises the bar on our own result symmetrically. **Our null currently rests on one judge
model family.** Cross-judge replication is now the expected standard and we have only a
partial version of it (the second judge agreed on `n_personas = 1.00` in every bin, but
failed on 74% of traces before the token-budget fix). This is the largest hole in our own
work and it is cheap to close.

---

## 2. Trace-narrative decline under RL is now a documented regularity

Our faithful run shows every conversational behaviour declining across 250 steps. That is
no longer a surprising isolated observation:

- RL fine-tuning **reduces chain-of-thought faithfulness over the course of training**,
  with some work reporting a rise that then falls
  ([2602.12506](https://arxiv.org/abs/2602.12506),
  [2605.24286](https://arxiv.org/abs/2605.24286)).
- **"Two Regimes of Chain-of-Thought Unfaithfulness"** ([2607.23458](https://arxiv.org/abs/2607.23458))
  — metric-based detection fails exactly where models are wrong, which is a sharper version
  of our own "the instrument fails where the signal is" lesson.
- **Gandhi et al.** ([2503.01307](https://arxiv.org/abs/2503.01307)), the paper the
  co-author pointed us at, remains the direct prior for Fig. 8: RL "selectively amplifies
  empirically useful behaviors while suppressing others," and Qwen moves from explicit
  verification language to implicit checking on this exact task.

So our declining curve sits inside an established pattern. **That is good for the authors,
not bad**: a Fig. 4 whose markers rise early and fall later is consistent with the field.
What it does not license is the paper's unqualified "base models increase conversational
behaviors."

---

## 3. The paper that most threatens our interpretation — and hands us the best next experiment

**Boppana, Ma, Loeffler, Sarfati, Bigelow, Geiger, Lewis & Merullo, "Reasoning Theater:
Disentangling Model Beliefs from Chain-of-Thought"** ([2603.05488](https://arxiv.org/abs/2603.05488), 5 Mar 2026).
Activation probing + early forced answering + CoT monitoring, on DeepSeek-R1 671B and
GPT-OSS 120B.

Their headline supports a performative reading: the model becomes confident early and
"continues generating tokens without revealing its internal belief." Probe-guided early
exit cuts tokens up to 80% on MMLU at similar accuracy.

**But the finding that matters for us cuts the other way:**

> inflection points (e.g., backtracking, "aha" moments) occur almost exclusively in
> responses where probes show large belief shifts

Discourse markers **do** track genuine internal state. That refutes the strong "markers are
exhaust" reading this project has flirted with, and it is a point in the SoT paper's favour
that we should concede in writing.

It also defines the best experiment left in this programme, and it needs no judge at all:

> **Do judge-identified persona boundaries and conflict events coincide with
> probe-detected belief shifts?**

If they do, the SoT paper is measuring something real and our null is an instrument
artifact. If they don't — if markers track belief shifts (Boppana) but *persona structure*
does not — then the dialogic framing is the part that fails, while the marker-level claim
survives. Either outcome is publishable, neither depends on an LLM judge, and it is the
natural bridge to `jacobian-lens`.

---

## 4. Adjacent: what post-training actually does

**Krishnamurthy, Huang & Rajaraman, "Select and Improve: Understanding the Mechanics of
Post-Training for Reasoning"** ([2606.13125](https://arxiv.org/abs/2606.13125), 11 Jun 2026)
names two mechanisms: **strategy selection**, activated by SFT on diverse reasoning
strategies, and **strategy improvement**, enabled by increasing RL difficulty.

Our C5 conclusion — priming installs the output contract, which fixes early exploration, so
it buys *speed* and not asymptote — is a concrete instance of strategy selection. Worth
citing: it gives our mechanism account a theory-side name from a strong group, and it is
independent of any claim about dialogue.

---

## 5. Reception: the paper is being built on, not tested

- It is now listed in [Google Research's publications](https://research.google/pubs/reasoning-models-generate-societies-of-thought/).
- Jason McEwen's ["Reasoning Models Don't Think Alone"](https://inductivebias.substack.com/p/reasoning-models-dont-think-alone)
  synthesises it into a "Bayesian modular minds" thesis. It is sympathetic commentary, and
  it explicitly does **not** scrutinise judge reliability, or whether measured persona
  diversity reflects cognitive specialisation versus annotation artifact.
- Searching for replications, critiques, or independent persona-emergence tests returns
  nothing.

**Nine months after publication the paper has commentary and zero adversarial replication.**
That is the gap this repository occupies, and it is still unoccupied.

---

## 6. Standards we should now meet, and currently don't

Taking Norman et al. and Yang et al. as the bar:

| standard | our status |
|---|---|
| cross-judge replication across model families | ⚠ partial — one family, second judge only on 100 traces |
| judge–human agreement on a sample | ❌ never done |
| positive control that the instrument recovers a known answer | ✅ done (2.87 personas on known-dialogue, 1.00 on known-monologue) |
| chance-corrected agreement (κ / ICC) rather than exact match | ❌ not computed |
| multiple seeds | ❌ n = 1 |
| protocol audit trail, prompt and cache published | ✅ `rl/judge.py`, cache keyed by model + prompt version |

The positive control is the thing we have that most of the field doesn't, and the two
missing cheap items — cross-judge and judge–human — are what a reviewer would ask for first.
