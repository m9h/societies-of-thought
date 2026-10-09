---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- reinforcement-learning
- chain-of-thought
- llm-as-judge
- replication
- countdown
size_categories:
- 1K<n<10K
configs:
- config_name: rollouts
  data_files:
  - split: train
    path: rollouts.jsonl
- config_name: judgments
  data_files:
  - split: train
    path: judgments.jsonl
---

# Fig. 4 corpus: every training rollout of a faithful PPO run on Countdown, with LLM-judge verdicts

Part of the **Controls & Trajectories** programme. This is the complete rollout record of one
250-step PPO run reproducing the configuration behind Fig. 4 of *Reasoning Models Generate
Societies of Thought* (Kim, Lai, Scherrer, Agüera y Arcas & Evans, [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)),
plus every verdict the paper's own LLM-judge instrument produced on a stratified subsample.
The paper ships no code and no traces; this is the artifact a replicator needs to check the
Fig. 4b/4e claim (that conversational behaviours and persona count rise under accuracy-only RL)
against something other than the figure.

## What was run

| | value |
|---|---|
| policy | `Qwen/Qwen2.5-3B` (base, **no** priming, no SFT) |
| task | Countdown (TinyZero data build) |
| algorithm | PPO, verl/TinyZero, full fine-tuning |
| steps | 250 (the paper's horizon), `rollout.n=4`, train batch 128, mini-batch 64 |
| actor / critic lr | 1e-6 / 1e-5, kl 0.001, temperature 1.0, response length 1024 |
| reward | TinyZero Countdown scorer: 1.0 correct, 0.1 format only, 0 otherwise |
| hardware | 2×A100 80GB (RunPod), ~654 s/step |
| validation reward | 0.104 at step 0 → **0.702** at step 250 (peak 0.707) |
| seed | one; the seed knob is a permutation of the training set (`rl/run_seed.py`) |

Full configuration audit against the paper: `docs/paper_fidelity_audit.md` and
`tests/test_fig4_fidelity.py` in [m9h/societies-of-thought](https://github.com/m9h/societies-of-thought).

## Files

**`rollouts.jsonl`** — one row per generation printed to the verl log, 2,555 rows: 2,102
on-policy training rollouts (temperature 1.0) and 453 held-out validation generations
(greedy, printed at every 10th step). The paper's Fig. 4b is measured on validation
generations, so the `source` field matters: the two series are different distributions.

| field | meaning |
|---|---|
| `idx` | position in the log |
| `step` | the training step whose gradient this rollout contributed to, or the checkpoint a validation generation was sampled from (verl prints both *before* the `step:N` line) |
| `source` | `train` (on-policy, temperature 1.0) or `validation` (greedy, held-out set) |
| `prompt` | the Countdown prompt as shown to the policy |
| `response` | the policy's generation, cut at its end-of-text token, with the prompt's `Assistant:` and the scorer's debug echo stripped (see below) |
| `words` | whitespace token count of `response` |
| `sha256` | hash of `response`, the join key for `judgments.jsonl` |

**`judgments.jsonl`** — one row per (trace, judge). Every row is a *parsed* verdict; a judge
response that failed to parse was never written as zero.

| field | meaning |
|---|---|
| `idx`, `step`, `sha256` | the rollout |
| `judge_model` | the judge that produced the row |
| `prompt_version` | `paper-v1` = the paper's two prompts verbatim (`judge_prompt_behaviours.txt`, `judge_prompt_persona.txt`); `v1` = this project's earlier paraphrase, one call |
| `source` | as in `rollouts.jsonl` |
| `question_answering`, `perspective_shift`, `conflict_of_perspectives`, `reconciliation` | integer counts, the paper's four behaviours and definitions |
| `n_personas` | `n_perspectives` from the paper's persona prompt (under `paper-v1`), or the single-call count (under `v1`) |
| `domain_expertise` | the paper's per-perspective expertise profiles, when the prompt produced them |

**`judge_prompt_behaviours.txt`, `judge_prompt_persona.txt`** — the paper's prompts
(arXiv 2601.10825v1, Supplementary Methods: LLM-as-Judge prompts), with `<TRACE>` where the
trace goes. Note that the persona prompt lists transitional markers ("however," "but,"
"wait," "actually") as indicators of a perspective boundary.

## Known caveats

- **Two series.** The paper judges held-out validation generations at each checkpoint. The
  `validation` rows here are that (greedy, ~17 per checkpoint); the `train` rows are the
  on-policy samples PPO trained on. Our first analysis mixed them.
- **Scorer-debug stripping.** TinyZero's reward function echoes `Target:` / `Extracted equation:`
  / `Solution string: A conversation between User and Assistant...` before each graded response
  and its verdict (`Invalid equation`, `No equation found`, ...) after it. Both are stripped;
  the response is cut at `<|endoftext|>`.
- **Judge identity.** The paper judges with Gemini-2.5-Pro. The verdicts here are from three
  Anthropic models. Agreement between them is reported in the repository's
  `results/emergence/cross_judge_*.json`, chance-corrected (κ, ICC(3,1)), not exact-match.
- **One seed.**

## Headline, for orientation only

On this run the judge found `n_personas = 1` on every one of 861 traces across all
subsamples, and conflict of perspectives at zero from step 96 onward, while the run learned
normally. The same judge scores the paper's own dialogue-primed corpus at 2.9 personas on
average (>1 on 93% of traces) and a monologue corpus at 1.0. Read `results/emergence/FINDINGS.md`
before quoting; §5 lists the limits.

## Citation

```
@misc{hough2026sotfig4,
  title  = {Fig. 4 corpus: PPO rollouts on Countdown with LLM-judge verdicts},
  author = {Hough, Morgan},
  year   = {2026},
  note   = {Controls \& Trajectories. https://github.com/m9h/societies-of-thought}
}
```
