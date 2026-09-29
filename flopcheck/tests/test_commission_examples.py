"""Reproduce the Commission's own figures, and show what they rest on."""

from __future__ import annotations

import math

import pytest

from flopcheck import commission_examples as ce
from flopcheck import estimate, peaks


@pytest.mark.parametrize("example", ce.EXAMPLES, ids=lambda e: e.label)
def test_reproduces_published_figure(example):
    """Every Annex A.3 figure is recoverable to within its own rounding."""
    recomputed = example.compute()
    assert example.agrees(), (
        f"Model {example.label}: recomputed {recomputed:.4g} rounds to "
        f"{estimate.round_sigfigs(recomputed, example.published_sigfigs):.4g}, "
        f"Annex published {example.published:.4g}"
    )


def test_only_the_hardware_examples_rest_on_assumptions():
    """Six examples follow from published facts; two do not."""
    dependent = {e.label for e in ce.assumption_dependent()}
    assert dependent == {"B", "H"}


@pytest.mark.parametrize("label", ["B", "H"])
def test_hardware_examples_fail_without_the_50_percent_assumption(label):
    """Drop U=0.5 and the Annex's published figure is unrecoverable.

    Substituting a measured mid-range MFU for the Annex's assumed 50%
    moves the answer outside even a generous margin -- which is the point:
    the number is an artefact of the assumption, not of the disclosure.
    """
    example = next(e for e in ce.EXAMPLES if e.label == label)
    realistic_mfu = 0.35
    with_mfu = estimate.hardware_based_from_unit_hours(
        example.given["unit_hours"],
        example.assumed["peak_flops"],
        realistic_mfu,
    )
    assert not estimate.within_margin(with_mfu, example.published), (
        f"Model {label}: U=0.35 gives {with_mfu:.3g} against a published "
        f"{example.published:.3g}; expected this to fall outside the 30% margin"
    )


def test_h100_candidate_peaks_span_a_factor_of_eight():
    """Paper §4.2: 'peak theoretical performance' admits an 8x range."""
    h100 = peaks.get("H100")
    # 8.0004, not exactly 8: NVIDIA's own datasheet rounds 494.7 / 989.4 /
    # 1978.9 inconsistently. The inconsistency is real and worth reporting.
    assert h100.peak_spread() == pytest.approx(8.0, rel=1e-3)
    candidates = h100.candidate_peaks()
    assert min(candidates.values()) == pytest.approx(494.7e12)
    assert max(candidates.values()) == pytest.approx(3957.8e12)


def test_commission_used_dense_bf16_without_saying_so():
    """Model H's 989 TFLOP/s is the dense BF16 figure, one of eight candidates."""
    h100 = peaks.get("H100")
    assert h100.peak(precision="bf16", sparse=False) == pytest.approx(989.4e12)
    example = next(e for e in ce.EXAMPLES if e.label == "H")
    assert example.assumed["peak_flops"] == pytest.approx(989e12, rel=1e-3)


def test_one_significant_figure_consumes_a_large_share_of_the_margin():
    """Paper §6.1: the Form's rounding alone eats up to 40% of the budget."""
    lo, hi = estimate.significant_figure_interval(4e5, sigfigs=1)
    assert (lo, hi) == pytest.approx((3.5e5, 4.5e5))

    err = estimate.significant_figure_error(4e5, sigfigs=1)
    assert err == pytest.approx(0.125)

    share = estimate.fraction_of_margin(err)
    assert share > 0.4, "expected >40% of the 30% margin consumed by rounding"


def test_utilisation_ambiguity_can_decide_threshold_classification():
    """Paper §4.1: one undefined word moves a model across Article 51(2).

    6 000 H100s for 30 days -- all of it public, undisputed fact. At this
    scale the entire plausible occupancy band sits above the Article 51(2)
    threshold and the entire plausible MFU band sits below it. The model is
    or is not of systemic risk according to which quantity the reader takes
    Annex A.2.1's undefined ``U`` to denote.
    """
    n_units = 6_000
    duration_s = 30 * estimate.SECONDS_PER_DAY
    peak = peaks.get("H100").peak("bf16")

    spread = estimate.utilisation_spread(n_units, duration_s, peak)

    occupancy_low, occupancy_high = spread["nvidia-smi occupancy"]
    mfu_low, mfu_high = spread["model FLOP utilisation (MFU)"]

    # Every occupancy reading is above the threshold ...
    assert occupancy_low > estimate.THRESHOLD
    assert occupancy_high > estimate.THRESHOLD
    # ... and every MFU reading is below it.
    assert mfu_high < estimate.THRESHOLD
    assert mfu_low < estimate.THRESHOLD
    # The gap between the two readings of the same run:
    assert occupancy_low / mfu_high > 1.7

    flat = {k: v for k, (v, _) in spread.items()}
    assert estimate.crosses_threshold(flat)


def test_reconciliation_backs_out_implied_utilisation():
    """Where the two approaches disagree, the residual is the utilisation."""
    n_units = 1_000
    duration_s = 10 * estimate.SECONDS_PER_DAY
    peak = peaks.get("H100").peak("bf16")
    arch = estimate.architecture_based_6pd(params=70e9, tokens=2e12)

    rec = estimate.reconcile(
        n_units=n_units,
        duration_s=duration_s,
        peak_flops=peak,
        utilisation=0.5,
        architecture_estimate=arch,
    )

    ceiling = n_units * duration_s * peak
    assert rec.implied_utilisation == pytest.approx(arch / ceiling)
    assert 0 < rec.implied_utilisation < 1


def test_margin_is_a_safe_harbour():
    """A documented estimate below the threshold tolerates a true value above it."""
    reported = 7.7e24
    lo, hi = estimate.margin_interval(reported)
    assert lo < estimate.THRESHOLD < hi
    assert hi == pytest.approx(1.001e25)


def test_6pd_understates_at_long_context():
    """The attention term the Annex drops grows with context length."""
    common = dict(
        non_embedding_params=7e9,
        tokens=1e12,
        n_layers=32,
        d_model=4096,
    )
    short = estimate.architecture_based_transformer(**common, context_length=2048)
    long = estimate.architecture_based_transformer(**common, context_length=131072)
    plain = estimate.architecture_based_6pd(7e9, 1e12)

    assert short == pytest.approx(plain, rel=0.25)
    assert long > 2 * plain, "6 * P * D should badly understate at 128k context"


def test_unverified_entries_are_flagged():
    """Nothing marked unverified should reach a publication unchecked."""
    names = {a.name for a in peaks.unverified()}
    assert "NVIDIA B200" in names
    for acc in peaks.all_accelerators():
        assert acc.note, f"{acc.name} needs a provenance note"


def test_weighted_peak_matches_annex_rule():
    """Heterogeneous fleets use a unit-count-weighted mean peak."""
    a100 = peaks.get("A100").peak("bf16")
    h100 = peaks.get("H100").peak("bf16")
    result = estimate.weighted_peak([(400_000, a100), (200_000, h100)])
    expected = (400_000 * a100 + 200_000 * h100) / 600_000
    assert result == pytest.approx(expected)
    assert math.isfinite(result)
