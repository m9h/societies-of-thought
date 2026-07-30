# Within QwQ, perspective diversity does not predict correctness

*Judge-free test of SoT's mediation claim on 4,000 QwQ traces. Code:
`analysis/hse_qwq.py`. Data: `hse_qwq_paper_convention.json` (canonical) and
`hse_qwq.json` (the drop-handling run, kept as the cautionary comparison).*

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

n = 2,000 correct and 2,000 incorrect, nothing discarded. Instrument reused unchanged from
`analysis/hse.py` (segmentation at the paper's own perspective-shift cues, then Balch's
Hierarchic Social Entropy).

| metric | correct | incorrect | difference (95% CI) |
|---|---|---|---|
| **hse_norm** (diversity, count-normalised) | 0.2698 | 0.2847 | **−0.0149** [−0.0230, −0.0068] ✱ |
| mean pairwise distance | 0.3810 | 0.4801 | **−0.0991** [−0.1109, −0.0873] ✱ |
| hse (raw) | 0.8178 | 1.2155 | **−0.3977** [−0.4308, −0.3646] ✱ |
| segments per trace | 13.5 | 38.2 | −24.7 ✱ |
| words per trace | 1,065 | 2,489 | −1,424 ✱ |

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

`hse_norm` differs by −0.0149 against a base of ~0.28 — about **5% relative**. That is a
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

## Limits

- **Streamed prefix, not a random sample.** `load_balanced` takes the first matching rows
  from the stream. If the corpus is ordered (by problem, difficulty, or generation batch),
  our 4,000 traces are a prefix rather than a random draw from 5.1M. This is the largest
  open weakness; a shuffle buffer would fix it and should be run before publication.
- **Math only.** NuminaMath, whereas SoT's pool spans BBH/GPQA/MATH/MMLU-Pro/MUSR. This
  tests the claim within one domain.
- **Segmentation is a heuristic** — a regex over the paper's cue words, not a semantic
  parse. It is the same heuristic used in our steering analysis, so the two are
  comparable, but it is not the paper's LLM judge and does not claim to be.
- **One embedding model** (`all-MiniLM-L6-v2`). Robustness to the embedder is unchecked.
- **Correlational.** This shows diversity does not track correctness within a model; it
  does not establish what does.
- **Not the paper's instrument.** If the authors' judge-based measure disagrees with HSE on
  the same traces, that is a finding about the measures. Running both on this corpus would
  settle it and has not been done.

## Next

1. **Shuffle-buffer resample** to remove the prefix concern — cheap, and it is the one
   limitation that could change the numbers rather than their interpretation.
2. **Embedder robustness** — repeat with a second embedding model.
3. **Scale to 50k/class** — the effect is small enough that a tighter estimate is worth
   the CPU, and the corpus supports it for free.
4. **The same test on DeepSeek-R1 traces**, the paper's other subject model, if an
   openly-licensed corpus with correctness labels exists.
