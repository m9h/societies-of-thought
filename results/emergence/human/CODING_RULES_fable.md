# Coding rules used for `ratings_fable.csv`

**Rater:** Claude Fable 5.1, reading `sheet.md` in a Claude Code session on 2026-10-22,
blind to `answer_key.json` until the ratings were written. This is **not a human rating**
and it is **not the paper's API prompt**: the rater read the sheet as a person would and
applied the paper's verbatim definitions with the rules below. Treat it as a third rater
from a different model and a different reading mode, not as the human anchor the paper
reports (.756 ICC). The human anchor is still open.

The rules were fixed before any trace was scored and are recorded so that anyone can
re-score the sheet the same way or dispute a rule rather than a number.

## Question and Answering

Counted only when a question is actually posed: an interrogative, or an explicit inquiry
("what if…", "how do we…", "is this…"). A trial of the form "Let's try X = Y, which is
not Z" is a proposal and a result, not a question and an answer. The paper's own example
("Let's try X…? This gives us Y") carries a question mark; none of the 50 traces does.
Under this rule every trace scores 0. The lenient alternative (each proposal–result line
counts) is the judge's behaviour and is already reported mechanically in FINDINGS §4.10
(count of rejection lines).

## Perspective Shift

Counted at an explicit change of approach or idea: "Instead", "another approach",
"However … if we", "But if we", a move to a different subproblem. A new combination under
the same brute-force strategy is **not** a shift, otherwise every enumeration line is one.
Sequencing words alone ("Next, let's try", "Now, let's try", "Finally") do not count when
the strategy is unchanged.

## Conflict of Perspectives

Counted when the text expresses correction of, or tension with, a prior claim or plan:
"However, this does not equal 29", "but we need to adjust", "It seems I missed something",
"I need to re-evaluate". A verification result on a trial that was never asserted to be
right ("(not 65)", "which is too high", "close but not exact") is a check, not a
disagreement, and is **not** counted.

## Reconciliation

Counted when a prior tension is explicitly resolved into a conclusion ("Ah, I see it now:
… So, the equation is"). Merely stating the answer after enumeration is not reconciliation.

## n personas

1 for a single continuous voice. 2 when a distinct corrective/reconsidering role
interrupts the working ("It seems I missed something … Ah, I see it now"; "However … Instead").
No trace showed more than these two roles. The `<think>`/`<answer>` split is a format,
not a perspective, and was never counted as a second persona.

## Verbatim loops

Where a trace degenerates into a verbatim repeated cycle (T13, T20, T28, T31), each
distinct cycle is counted once; verbatim repeats are not re-counted.

## Result of applying the rules

45 of 50 traces: all four behaviours 0, one persona. The five traces with any conflict
(T05, T13, T31, T38, T40) are all prose-style "we can … However … Instead" traces, i.e.
the early-training style; no enumeration-style trace scored above zero on anything.
