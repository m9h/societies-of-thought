# Within QwQ, perspective diversity does not predict correctness

*Judge-free test of SoT's mediation claim on 10,000 QwQ traces. Code:
`analysis/hse_qwq.py`, cross-run comparison in `analysis/hse_qwq_compare.py`. Canonical
data: `hse_shuffled_minilm.json`; `hse_qwq.json` is retained as the cautionary
drop-handling comparison.*

## The claim

SoT argues that reasoning models outperform instruction-tuned models because they simulate
a *diverse* internal society, and that this diversity **accounts for** the accuracy
advantage. Diversity is measured by an LLM judge that infers personas and scores their
spread, over 8,262 traces.

The cross-model version of that comparison cannot isolate diversity: DeepSeek-R1 differs
from DeepSeek-V3 in many ways at once, so "the reasoning model is both more diverse and
more accurate" is compatible with diversity being incidental. The **within-model** version
is decidable. Hold the model, the decoding, and the task distribution fixed; vary only the
outcome. If diversity mediates accuracy, correct traces must be more diverse than
incorrect ones.

`PrimeIntellect/NuminaMath-QwQ-CoT-5M` (MIT) makes this cheap: 5,138,102 **QwQ** traces
with a per-trace `correct` label. QwQ-32B is one of the two models SoT builds its
descriptive and diversity claims on.

## Result

Canonical run: n = 5,000 correct and 5,000 incorrect, shuffled sample, nothing discarded.
Instrument reused unchanged from
`analysis/hse.py` (segmentation at the paper's own perspective-shift cues, then Balch's
Hierarchic Social Entropy).

| metric | correct | incorrect | difference (95% CI) |
|---|---|---|---|
| **hse_norm** (diversity, count-normalised) | 0.2746 | 0.2932 | **−0.0186** [−0.0237, −0.0135] ✱ |
| mean pairwise distance | 0.3826 | 0.4819 | **−0.0992** [−0.1065, −0.0920] ✱ |
| hse (raw) | 0.8127 | 1.1985 | **−0.3858** [−0.4060, −0.3656] ✱ |
| segments per trace | 10.9 | 30.2 | −19.3 ✱ |
| words per trace | 924 | 2,100 | −1,176 ✱ |

✱ = 95% CI on the difference excludes zero.

**Every diversity measure is *lower* in the traces that reach the correct answer.** The
mediation claim requires the opposite sign.

The mechanism is legible in the last two rows. When QwQ gets a problem wrong it flails:
2.8× more perspective shifts, 2.3× more words. Shift markers — *wait*, *but*, *actually*,
*however* — track **struggle**, not success. On this evidence the "society" is denser
precisely where the reasoning fails.

That is the same conclusion our steering experiment reached by intervention rather than
observation: pushing the conversational feature up made traces measurably more dialogic and
accuracy *fell* (`results/steering/FINDINGS.md`). Two independent methods, one direction.

### Effect sizes, stated honestly

`hse_norm` differs by −0.0186 against a base of ~0.29 — about **6% relative**. That is a
small effect, precisely estimated. The large differences (`hse` −0.40, `mean_dist` −0.10)
are substantially **length artifacts**: more segments mechanically raise raw entropy and
pairwise spread. `hse_norm` divides out log2(N) and is the measure to quote. So the
defensible statement is *diversity does not predict correctness in the direction the claim
requires, and if anything anti-predicts it*, not *diversity strongly causes failure*.

## The methodological point that decides the sign

Our first run **inverted this result**, and the cause is worth recording because it is a
one-line decision.

Traces with fewer than 3 perspective shifts cannot support a dendrogram. Our first pass
**dropped** them. But QwQ's correct traces have far fewer shifts, so dropping removed
**314 correct vs 143 incorrect** traces — 2.2× more from the group with less diversity,
i.e. it deleted the low-diversity tail of the correct group specifically.

| single-voice traces | hse_norm difference | reads as |
|---|---|---|
| **dropped** | **+0.0134** [+0.0075, +0.0193] ✱ | correct traces MORE diverse |
| **scored 0** | **−0.0149** [−0.0230, −0.0068] ✱ | correct traces LESS diverse |

Same data, same instrument, opposite conclusions, both with CIs excluding zero.

**The paper settles which is right.** Its own diversity metric assigns zero, not exclusion:

> If a reasoning trace contained only a single implicit voice, **P_j = 0**
> If a reasoning trace contained only a single implicit voice, **E = 0**

A trace with one voice has no diversity — that is a measurement, not a missing value.
Scoring zero is the paper's convention and is now the default (`--degenerate zero`), with
tests pinning it and the `drop` path documented as the trap it is.

Had we not checked, we would have published a result in the paper's favour, produced by our
own filter.

## Robustness — the three checks the first version owed

All three follow-ups listed in the first version have now been run. Every arbitrary
choice was varied and the sign did not move.

| run | embedder | sampling | n/class | **hse_norm difference** |
|---|---|---|---|---|
| *(the trap)* single-voice **dropped** | MiniLM-L6 | prefix | 2,000 | **+0.0134** [+0.0075, +0.0193] |
| single-voice **zero** | MiniLM-L6 | prefix | 2,000 | **−0.0149** [−0.0230, −0.0068] |
| single-voice zero | MiniLM-L6 | **shuffled** (100k) | **5,000** | **−0.0186** [−0.0237, −0.0135] |
| single-voice zero | **mpnet-base** | shuffled (100k) | 2,000 | **−0.0214** [−0.0289, −0.0139] |

`analysis/hse_qwq_compare.py` judges sign stability across runs and reports:
**SIGN STABLE across 3 paper-convention runs — correct traces LESS diverse**, on all three
of `hse_norm`, `mean_dist` and `hse`.

Three specifics worth noting:

1. **The prefix concern is closed.** A seeded 100k reservoir over the stream, at 2.5× the
   sample size, moved `hse_norm` from −0.0149 to −0.0186 — same direction, slightly
   larger. `mean_dist` is essentially unchanged across sampling (−0.0991 → −0.0992).
2. **It is not an artifact of a weak embedder.** `all-mpnet-base-v2` gives a *larger*
   effect than `all-MiniLM-L6-v2` (−0.0214 vs −0.0186), so the finding does not depend on
   the cheaper model's limitations.
3. **Only one choice ever flipped the sign**: dropping single-voice traces instead of
   scoring them zero. Sampling and embedder did not. That isolates the earlier inversion
   as a filtering error, not measurement noise.

Canonical run: `hse_shuffled_minilm.json` (n = 5,000/class, shuffled, zero convention).
Its single-voice counts are 707 correct vs 365 incorrect — the same 2:1 asymmetry that
made the drop-handling error so consequential.

## Limits

- ~~Streamed prefix~~ — **resolved.** A seeded 100k shuffle reservoir is now the default
  and does not change the direction.
- **Math only.** NuminaMath, whereas SoT's pool spans BBH/GPQA/MATH/MMLU-Pro/MUSR. This
  tests the claim within one domain.
- **Segmentation is a heuristic** — a regex over the paper's cue words, not a semantic
  parse. It is the same heuristic used in our steering analysis, so the two are
  comparable, but it is not the paper's LLM judge and does not claim to be.
- ~~One embedding model~~ — **resolved.** Replicated with `all-mpnet-base-v2`, larger effect.
- **Correlational, and length is the live confound.** Incorrect traces are 2.3× longer.
  `hse_norm` divides out segment *count*, but not everything length brings with it. The
  matched-length subsample (Next §6) is the check that would close this.
- **Not the paper's instrument.** If the authors' judge-based measure disagrees with HSE on
  the same traces, that is a finding about the measures. Running both on this corpus would
  settle it and has not been done.

## Next

1. ~~Shuffle-buffer resample~~ — **done**, sign unchanged.
2. ~~Embedder robustness~~ — **done**, effect larger with mpnet.
3. ~~Scale up~~ — **done** at 5,000/class.
4. **Run the paper's own judge-based measure on these same traces.** If an LLM judge
   disagrees with HSE here, that is a finding about the instruments and the most
   informative thing left. Our result is only as strong as the claim that HSE measures
   what their judge measures, and that has never been checked on shared inputs.
5. **The same test on DeepSeek-R1 traces**, the paper's other subject model, if an
   openly-licensed corpus with correctness labels exists.
6. **Length-matched subsample.** Compare correct and incorrect traces at matched word
   count. It will shrink the sample hard, but it separates "diversity anti-predicts
   correctness" from "long traces are both more diverse and more often wrong".
