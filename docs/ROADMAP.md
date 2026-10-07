# Roadmap — as of 2026-10-06

*Grounded in `docs/literature_2026_H2.md`. Ordered by value per dollar, which is not the
same as ordered by cost. Balance $291.02, nothing billing.*

## The state of the question

Fig. 4 has been answered on the paper's own configuration: 861 traces, 0 failures, every
one at a single persona; all four behaviours decline; conflict of perspectives zero from
step 96. The run learned normally (0.104 → 0.702). The qualitative result has now been
reached three times with three instruments, and the paper is still v1 with no code.

The field has since formalised our central methodological claim (judge-dependence) at
scale, and simultaneously raised the bar our own null must meet. Boppana et al. have also
shown that discourse markers *do* track probe-measured belief shifts — which both softens
our "exhaust" reading and hands us a judge-free test.

## Tier 0 — free, this week

1. **Send `docs/note_to_authors.md`.** It now cites the judge-validity literature and asks
   the one question that resolves everything: does Fig. 4e's persona count exceed 1 on
   Countdown? Earlier is worth more than more complete; they are mid-revision.
2. **Publish the corpus** to `controls-and-trajectories`: 2,555 rollouts from a complete
   faithful run, 861 judged verdicts, the judge prompt, the cache. Nobody else has it.
3. **Fold the literature into the writeup.** Recast the methodological claim in Norman et
   al.'s reliability/validity vocabulary; cite Boppana et al. as the point conceded to the
   paper; cite Krishnamurthy et al. for the C5 mechanism.

## Tier 1 — cheap, and what a reviewer asks for first

4. **Cross-judge replication across model families.** Our null rests on one family. Re-run
   `rl/judge.py` with two other families on the same 861 traces; report chance-corrected
   agreement (κ, ICC), not exact match. API cost only; the cache makes it incremental.
   *This is the largest hole in our own result and the cheapest to close.*
5. **Judge–human agreement.** 50 traces, stratified by step, read by a person against the
   same four definitions. Norman et al. make this the validity anchor; we have never done
   it. Free.
6. **Fix the watchdog.** It reported a successful completion as FAILURE and read step 0
   when the log said 250. Both bugs still in `scripts/ppo_watchdog.sh`-era logic and the
   session monitor. Cheap, and a watchdog that cries wolf on success is ignored on failure.

## Tier 2 — the strongest science left

7. **The Boppana test: do persona boundaries coincide with belief shifts?** Train a linear
   probe for answer-confidence on hidden states over our own rollouts; locate
   probe-detected belief shifts; test whether judge-identified conflict events and
   persona boundaries align with them. Judge-free, mechanism-level, adjudicates between
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

- More marker-proxy analysis. It has disagreed with the judge on every configuration, in
  both directions, by up to two orders of magnitude. `pattern_dominance` stays as a
  diagnostic; the rates do not get reported.
- Buy seeds before 4 and 5. A seeded result on an unvalidated instrument is a more precise
  version of the same uncertainty.
