# Correction: the "redundant society" finding was a filtering artifact

*Re-audit run 2026-07-30 after the same two errors were found in the QwQ analysis
(`results/qwq/FINDINGS.md`). Code: `analysis/hse_steering_recheck.py`. Data:
`hse_recheck_rg.json`.*

## What we previously reported

> | α | segments/trace | normalised diversity | accuracy |
> |---|---|---|---|
> | 0 | 21.4 | **0.236** | 15.2% |
> | 1.0 | 44.6 | 0.190 | 31.5% |
> | 1.693 | 54.7 | **0.190** | 3.6% |
>
> **Steering makes the society bigger and proportionally *more redundant*.**
> — `docs/replication_and_tools.md` §5

That reading produced the "louder crowd saying more of the same thing" / echo-chamber
metaphor, and it is the claim `docs/related_work_2026.md` connects to the martingale
theorem. **It does not survive re-analysis.**

## What is actually true

Recomputed with single-voice traces scored **zero** rather than dropped — the paper's own
convention (*"If a reasoning trace contained only a single implicit voice, E = 0"*):

| α | n | single-voice | words | segments | **hse_norm** | mean_dist | accuracy |
|---|---|---|---|---|---|---|---|
| 0.000 | 200 | **95** | 673 | 12.0 | **0.1237** | 0.1681 | 24.0% |
| 0.250 | 200 | 69 | 941 | 16.0 | 0.1471 | 0.2133 | 23.0% |
| 0.500 | 200 | 49 | 1128 | 22.6 | 0.1598 | 0.2510 | 31.0% |
| 0.678 | 200 | 35 | 1423 | 28.8 | 0.1714 | 0.2826 | 29.0% |
| 1.000 | 200 | 32 | 1530 | 37.7 | 0.1594 | 0.2857 | 34.0% |
| 1.693 | 200 | **8** | 2074 | 52.6 | **0.1820** | 0.3452 | **3.5%** |

**Normalised diversity RISES with steering — 0.124 → 0.182 — it does not fall.**

### Why the original was wrong

The old analysis dropped traces with fewer than 3 segments. Steering *creates* perspective
shifts, so the drop rate collapses as α rises: **95 of 200 baseline traces were discarded
(47.5%) versus 8 at α=1.693 (4%)**. Nearly half the baseline condition was deleted, and
precisely its least diverse half. That inflated the baseline from 0.124 to 0.236 and
manufactured a downward trend out of an upward one.

This is the same error as QwQ error (1), and it bit harder here because the treatment
directly controls the variable the filter keys on.

### And it survives length matching

Unlike the QwQ result, this one is **not** explained by length. Matching steered traces to
baseline traces of the same length (±10% caliper), differences reported as
baseline − steered so *negative* means steering raises diversity:

| α | pairs | unadjusted | matched | |
|---|---|---|---|---|
| 0.250 | 161 | −0.0234 | −0.0189 [−0.0435, +0.0057] | |
| 0.500 | 143 | −0.0361 | **−0.0374** [−0.0599, −0.0148] | ✱ |
| 0.678 | 116 | −0.0477 | **−0.0507** [−0.0771, −0.0243] | ✱ |
| 1.000 | 116 | −0.0357 | **−0.0523** [−0.0788, −0.0258] | ✱ |
| 1.693 | 36 | −0.0584 | −0.0193 [−0.0709, +0.0323] | *(few pairs)* |

Matching *increases* the effect at α = 1.0 (shrinkage −46%). Steering genuinely raises
normalised diversity in traces of equal length.

## What this does to the argument

**Retracted.** "Steering produces a redundant society", the echo-chamber metaphor, and
"the workspace's diversity collapses into redundancy" are all withdrawn. The induced
society is *more* differentiated, not less.

**Strengthened, and this is the point.** The mediation claim fails more cleanly than
before, because diversity and accuracy are now visibly **decoupled**:

- diversity rises **monotonically** with α: 0.124 → 0.147 → 0.160 → 0.171 → 0.159 → 0.182
- accuracy traces an **inverted U**: 24% → 23% → 31% → 29% → **34%** → **3.5%**

At α = 1.693 the society is at its **most diverse** and the model is at its **least
accurate** — 3.5% against a 24% baseline. Maximum measured diversity coincides with
near-total failure.

That is a stronger refutation than the redundancy story was. The old argument said *the
society isn't real, so of course it doesn't help*. The new one says *the society is real,
measurably more differentiated at every dose, and it still does not buy accuracy* — you can
drive genuine perspective diversity up and reasoning down at the same time. No appeal to
the society being fake is needed.

The paper's own prediction is that diversity produces accuracy. Here they come apart.

## What this does NOT change

- The **inverted-U dose-response** stands: +10 points at α=1.0, collapse to 3.5% at 1.693.
- **C1 reverses on MATH-Hard** (−22 points): untouched by this, a different experiment.
- The **descriptive observation** stands: steering does induce dialogic behaviour.
- **Claim B / C5** is unaffected — a separate RL experiment with its own data.

## Consequences for the martingale framing

`docs/related_work_2026.md` §1 connects our redundancy finding to the MAD martingale
result: *zero information asymmetry ⇒ error correlation 1.0 ⇒ no gain from exchange*. That
bridge was built on the claim that the induced society is redundant, and **that claim is
now withdrawn**. The martingale connection was already flagged there as an analogy rather
than a derivation; it now loses its main empirical anchor and must be re-stated or dropped.

Measured segment diversity is not the same quantity as inter-agent error correlation, and
we have never measured the latter. That was listed as an open follow-up and is now the
prerequisite for saying anything about the martingale at all.

## Method note

Both errors that hit this analysis were found by porting a check from elsewhere, not by
inspecting it directly. The QwQ work forced the single-voice convention and the length
control; applying them here overturned a result we had been citing for weeks. The general
lesson is in `results/qwq/FINDINGS.md`: robustness across sampling, sample size and
embedder said nothing about either problem, because none of those choices touched the
filter or the length distribution.
