# Roadmap — as of 2026-10-08 (status pass after Tier 0–1 execution)

*Grounded in `docs/literature_2026_H2.md`. Ordered by value per dollar, which is not the
same as ordered by cost. Balance $291.02, nothing billing.*

## The state of the question

**Revised 2026-10-08.** Cross-judging (item 4) found three errors of ours and reversed the
headline: the paper's judge prompts *are* published (v1 supplement), and under them,
verbatim, **Fig. 4b/4e reproduce** on our run with two judges — while the judge-free series
shows the traces becoming a single-voice, question-free enumeration that the paper's
definitions score as rising conflict between an increasing number of perspectives. The
result is now a content-validity finding, not a non-replication. See
`results/emergence/FINDINGS.md` §4.10–4.11. The paper is still v1 with no code.

The field has since formalised our central methodological claim (judge-dependence) at
scale, and simultaneously raised the bar our own null must meet. Boppana et al. have also
shown that discourse markers *do* track probe-measured belief shifts — which both softens
our "exhaust" reading and hands us a judge-free test.

## Tier 0 — free, this week

1. **Send `docs/note_to_authors.md`.** ⏳ *Rewritten 2026-10-08 (§4 and Q7–9) around the
   verbatim-instrument result and our corrections; ready to send — the user's call.* The
   two asks that resolve things: Gemini verdicts on a dozen of our traces, and the
   expertise profiles of their step-120 personas.
2. **Publish the corpus.** ⏳ *Built and verified locally (`python -m
   scripts.export_fig4_corpus --repo mhough/sot-fig4-countdown-ppo-rollouts --public`;
   2,555 rollouts with `source`, 1,031 paper-prompt verdicts, both prompts, card). The
   public upload needs the user to run that one command.*
3. **Fold the literature into the writeup.** ✅ FINDINGS §7, note §4 addendum.

## Tier 1 — cheap, and what a reviewer asks for first

4. **Cross-judge replication.** ✅ *within one vendor* (Sonnet 5, Haiku 4.5; Opus 5
   refused every call), on the paper's prompts, both series, κ/ICC reported
   (`analysis/cross_judge.py`, `results/emergence/cross_judge_*.json`). ⏳ **Across
   families still open**: no Gemini/OpenAI key here (NIM returns 403). `make_backend
   ("gemini/gemini-2.5-pro")` runs the moment `GEMINI_API_KEY` exists — Gemini-2.5-Pro is
   the paper's judge, so this is the single most valuable remaining call.
5. **Judge–human agreement.** ⏳ Sheet built on the validation series with the paper's
   definitions (`results/emergence/human/sheet.md`, 50 traces, blind, shuffled). **Needs a
   person to fill `ratings_template.csv`**, then `python -m analysis.human_annotation score`.
6. **Fix the watchdog.** ✅ `scripts/run_status.sh`, 9 tests.

## Tier 2 — the strongest science left

7. **The Boppana test: do persona boundaries coincide with belief shifts?** *Sharper now:*
   the paper's persona boundaries on our traces are the `<think>`/`<answer>` split and
   `(not N)` rejection lines. The probe test asks whether those carry a belief shift at
   all. Train a linear probe for answer-confidence on hidden states over our own rollouts;
   locate probe-detected belief shifts; test whether judge-identified conflict events and
   persona boundaries align with them. **Needs checkpoints**: the FINAL run saved none
   (`save_freq=-1`), so this requires a re-run with `SAVE_FREQ=50` (~$143) or probing the
   base model only. Judge-free, mechanism-level, adjudicates between
   "markers are exhaust" and "markers track state." Needs Qwen-2.5-3B forward passes with
   hidden-state capture on ~1,000 traces — Modal, a few GPU-hours. Natural bridge to
   `jacobian-lens`.
8. **Seeds.** Two more at ~$140. The field standard says conclusions must not depend on a
   single seed, and the authors asked. Demoted below 4–7 because the open uncertainty is
   the instrument, not the seed — but it should be done before any public writeup.

## Tier 3 — backlog, unchanged

9. Censored-likelihood DDM (critique §2).
10. C6 transfer, 3 arms.
11. Publish the QwQ corpus.
12. Propagate the appendix correction to `docs/related_work_2026.md`.

## What I would not do

- More marker-proxy *curves*. The judge-free token shares in §4.10 ("we", "I", "?", the
  marker set) are reported because they are transparent and because they explain the
  judge's counts; marker *rates* as a substitute for the judge are not.
- Report any Fig. 4 number without saying which prompt produced it. The same traces give
  1.00 personas under ours and 1.8 under theirs.
- Buy seeds before 4 and 5. A seeded result on an unvalidated instrument is a more precise
  version of the same uncertainty.
