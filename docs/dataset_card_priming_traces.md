---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- multi-agent
- chain-of-thought
- replication
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
---

# Paired dialogue/monologue priming traces (SoT Claim B reconstruction)

600 reasoning problems, each solved **twice** by the same teacher: once as a
**multi-agent dialogue** between 2–4 named personas, once as a **single-voice
monologue**. Both traces reach the correct answer on the *same* problem.

This is a **reconstruction** of the supervised fine-tuning corpus described in
*Reasoning Models Generate Societies of Thought*
([arXiv:2601.10825](https://arxiv.org/abs/2601.10825)). **It is not the authors' data** —
that paper ships no code or data. Everything here was regenerated from the paper's
published specification. Differences from the authors' corpus are unknowable in detail;
what is reproducible is the *procedure*, documented below.

## Why this exists

The paper's central causal experiment (its Claim B / C5) primes a base model on dialogue
vs monologue traces and then runs identical RL, asking whether conversational structure
*learned on unrelated problems* accelerates learning on a new task. Reproducing it
requires the priming corpus. There isn't one to download — so we rebuilt it, and we are
releasing it so the next person doesn't have to spend the GPU hours again.

## How it was built

| | |
|---|---|
| **Teacher** | `Qwen/Qwen2.5-32B-Instruct` — the model the paper specifies |
| **Problem pool** | The paper's 8,262-problem pool, reconstructed from Supplementary Table 9 |
| **Prompts** | Verbatim from the paper's Supplementary Methods |
| **Selection** | Kept only problems where **both** arms reach the correct answer |
| **Split** | 500 train / 100 validation, arms verified to cover identical `pid`s |

The pool is drawn from **BigBench Hard, GPQA, MATH-Hard, MMLU-Pro, MUSR and IFEval**, and
its per-subtask counts reconstruct the paper's 8,262 exactly. A useful independent check:
excluding IFEval (which the paper says it drops as ungradable) leaves **7,738** gradable
problems — the same figure the paper reports, and one we did not target.

**The priming problems are deliberately out-of-domain** relative to the downstream RL task
(Countdown arithmetic). That relationship is the entire point of the experiment: it tests
whether conversational *structure* transfers, not whether task knowledge does.

## Fields

| field | description |
|---|---|
| `pid` | stable problem id, e.g. `bbh/logical_deduction_three_objects/10` |
| `source` | `bbh` \| `gpqa` \| `musr` \| `math_hard` \| `mmlu_pro` |
| `subtask` | benchmark subtask |
| `task` | the problem text shown to the teacher |
| `answer` | gold answer |
| `dialogue` | `<cast_of_characters>` … `<conversation>` … `<group_solution>` |
| `monologue` | `<think>` … `</think>` then the answer |

Composition (train): bbh 342, gpqa 52, musr 43, math_hard 33, mmlu_pro 30.

## The length confound — please read before using

Dialogue traces are **~1.75× longer** than monologue traces (median 252 vs 144 words).
Any dialogue advantage measured on this corpus is confounded with tokens spent unless you
control for it.

The paper controls for this on Llama-3.2-3B — "reasoning content from multiple personas
was concatenated into a single block (`<think> </think>`) to ensure comparable sequence
lengths" — and **does not** do so for Qwen. If you use this data, decide deliberately
which of those you are reproducing. Helper:
[`rl.claimB_data.concatenate_personas`](https://github.com/m9h/societies-of-thought).

## Known deviations from the paper

1. **Problem identity.** Table 9 gives per-subtask *counts*, not *which* problems were
   sampled where a benchmark is larger (MMLU-Pro 432 of ~12k, GPQA main 380 of 448). We
   match the counts with a fixed seed.
2. **GPQA source.** The canonical `Idavidrein/gpqa` is gated; we fall back to the open
   mirror `Wanfq/gpqa`, which carries identical per-config counts (198/448/546) and schema.
3. **Sampling.** Teacher decoding at `temperature=0.7, top_p=0.95, seed=0`. The paper does
   not report its decoding parameters.

## Grading honesty

Answer grading went through three revisions, and the failures are instructive:

- A **letter-only** matcher silently excluded GPQA, MMLU-Pro and MUSR *entirely* — every
  answer stated as option text scored wrong — leaving an 81%-BBH corpus at 11.5% yield.
- Adding option-text matching raised yield to 44.4% and restored all five benchmarks, but
  a substring rule then **inverted** some answers: `(B) tanh(J/T) …` against gold `D`
  scored *correct* because D's text appeared later in the sentence.
- The final rule is priority-based: a prediction that declares a choice letter is taken at
  its word and never falls through to text matching. 22 of 2,218 matched pairs were wrong
  under the loose rule and were dropped.

Yield was 44.4% of attempted problems; the teacher simply gets many of these wrong.

## Citation

Cite the original paper for the method. For this reconstruction, link the repository:
<https://github.com/m9h/societies-of-thought>.
