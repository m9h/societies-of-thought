"""Hierarchic Social Entropy, checked against cases where the answer is known.

HSE is the one metric in this project that could quietly manufacture the paper's
conclusion. It rises mechanically with the number of agents -- H(0) = log2(N) -- and
steering produces more discourse markers, hence more segments. So a naive "HSE went up"
would be an artifact of counting, not a finding about diversity.

These tests pin the properties that make the interpretation safe.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.hse import hierarchic_social_entropy, segment


def _dist(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return d


def test_identical_agents_have_zero_diversity():
    """A perfectly redundant society: every voice the same. HSE must be 0."""
    pts = np.zeros((6, 3))
    hse, hse_n, md = hierarchic_social_entropy(_dist(pts))
    assert hse == pytest.approx(0.0, abs=1e-9)
    assert md == pytest.approx(0.0, abs=1e-9)


def test_more_agents_saying_the_same_thing_does_not_raise_distance():
    """THE CONFOUND, made explicit.

    Twelve identical voices vs three identical voices: raw HSE is still 0 here (all
    distances vanish), but the real point is that mean pairwise distance -- the measure
    we lean on -- cannot be inflated by simply adding redundant voices.
    """
    small = _dist(np.zeros((3, 4)))
    big = _dist(np.zeros((12, 4)))
    _, _, md_small = hierarchic_social_entropy(small)
    _, _, md_big = hierarchic_social_entropy(big)
    assert md_small == pytest.approx(md_big, abs=1e-9) == pytest.approx(0.0, abs=1e-9)


def test_exact_duplicates_do_not_inflate_hse():
    """A useful property, discovered while writing these tests.

    Under single linkage, exact copies merge at height ~0 and contribute nothing to the
    integral. So HSE cannot be inflated by simply repeating a voice -- which is exactly
    the robustness we need against a model that says the same thing many times.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 8))
    duped = np.vstack([a, a + 1e-9, a, a])          # 16 agents, but only 4 distinct

    hse_a, _, _ = hierarchic_social_entropy(_dist(a))
    hse_d, _, _ = hierarchic_social_entropy(_dist(duped))
    assert hse_d == pytest.approx(hse_a, rel=1e-6), (
        "duplicating voices must not raise HSE -- redundancy is not diversity"
    )


def test_raw_hse_depends_on_count_but_mean_distance_does_not():
    """Why the analysis leans on mean_dist and controls for segment count.

    Measured, drawing from the SAME distribution at different sizes:

        n= 5  HSE= 6.14   mean_dist=3.06
        n=10  HSE=10.41   mean_dist=4.06
        n=20  HSE=12.17   mean_dist=4.08
        n=40  HSE=12.19   mean_dist=3.81

    Raw HSE is count-dependent and NON-monotonic -- it saturates, because adding points
    brings nearest neighbours closer, shrinking single-linkage merge heights and
    offsetting the log2(N) gain. (The naive fear that "more segments mechanically means
    more HSE" is therefore wrong, but the count dependence is real and messy.)

    Mean pairwise distance is stable. It is the metric that answers the actual question --
    ARE THE VOICES DIFFERENT? -- so it carries the argument, and segment count is reported
    alongside so any count effect is visible rather than hidden.
    """
    rng = np.random.default_rng(0)
    md = {}
    hse = {}
    for n in (5, 10, 20, 40):
        h, _, m = hierarchic_social_entropy(_dist(rng.normal(size=(n, 8))))
        hse[n], md[n] = h, m

    spread = max(md.values()) - min(md.values())
    assert spread < 1.5, "mean pairwise distance must be roughly count-stable"
    assert hse[40] > hse[5], "raw HSE is count-dependent -- never interpret it alone"


def test_genuinely_differentiated_society_scores_above_a_redundant_one():
    """The measurement that carries the argument: spread-out voices beat clustered ones."""
    rng = np.random.default_rng(1)
    redundant = rng.normal(scale=0.01, size=(8, 6))     # all saying the same thing
    diverse = rng.normal(scale=1.00, size=(8, 6))       # genuinely different

    _, _, md_red = hierarchic_social_entropy(_dist(redundant))
    _, _, md_div = hierarchic_social_entropy(_dist(diverse))
    assert md_div > md_red * 10, "mean pairwise distance must separate real from redundant"


def test_hse_matches_a_hand_computable_case():
    """Two tight pairs, far apart. Verify the integral against the dendrogram by hand.

    Points at 0, 0 (pair A) and 10, 10 (pair B), in 1-D.
      merge heights: 0 (A joins), 0 (B joins), 10 (A joins B)
      H(h) for h in [0, 10):  2 clusters of 2  -> H = -2*(0.5 log2 0.5) = 1.0
      H(h) for h >= 10:       1 cluster        -> H = 0
      => S = 1.0 * 10 = 10.0
    """
    pts = np.array([[0.0], [0.0], [10.0], [10.0]])
    hse, _, _ = hierarchic_social_entropy(_dist(pts))
    assert hse == pytest.approx(10.0, rel=1e-6)


def test_segmentation_cuts_at_perspective_shifts():
    """Segmentation uses the PAPER'S OWN conversational cues, not an LLM's judgement."""
    trace = (
        "First I will compute the product of the two leading terms carefully. "
        "But wait, that leaves the remainder unaccounted for in the expansion. "
        "Actually the cleaner route is to factor it before substituting anything. "
        "Alternatively I could just expand the whole thing and check numerically."
    )
    segs = segment(trace)
    assert len(segs) >= 3, f"expected several voices, got {len(segs)}"
    assert any(s.lower().lstrip().startswith(("but", "actually", "alternatively")) for s in segs)


def test_degenerate_babble_is_not_mistaken_for_a_society():
    """The steered failure mode: many markers, no content. Segments are filtered by length,
    so 'wait, no, wait, no, wait' must not read as a rich society of voices."""
    babble = "wait. no. wait. oh. wait. no. hmm. oh. wait."
    assert len(segment(babble)) < 3, "marker spam must not become a society"
