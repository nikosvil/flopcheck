"""The two estimation approaches of Annex A.2, and what they leave undefined.

Annex A.2 of the Commission's Guidelines on the scope of the obligations for
general-purpose AI models (18 July 2025) permits any estimation method
"so long as the estimated amount is, in the providers' best judgement,
accurate within an overall error margin of 30% of the reported estimate",
and describes two:

    A.2.1  hardware-based      C = N * L * H * U  (paragraph 124)
    A.2.2  architecture-based  C = passes * operations-per-pass, approximated
                               for dense transformers as C = 6 * P * D
                               (paragraph 129)

The letters are the Annex's own: N hardware units, L seconds of use, H peak
FLOP/s per unit, U utilisation, P parameters, D training tokens.

This module implements both, plus the analyses that quantify what the
Annex leaves open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import peaks

MARGIN = 0.30
"""The error margin bound by Annex A.2, paragraph 120."""

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


# ---------------------------------------------------------------------------
# A.2.1 -- hardware-based
# ---------------------------------------------------------------------------


def hardware_based(
    n_units: float,
    duration_s: float,
    peak_flops: float,
    utilisation: float,
) -> float:
    """Annex A.2.1: ``C = N * L * H * U``.

    Parameters
    ----------
    n_units:
        Number of accelerators (``N``).
    duration_s:
        Total duration of use in seconds (``L``).
    peak_flops:
        Peak theoretical performance per unit in FLOP/s (``H``). Which peak
        is left undefined by the Annex; see :mod:`flopcheck.peaks`.
    utilisation:
        Average percentage of GPU utilisation as a fraction (``U``). Which
        quantity this denotes is left undefined by the Annex; see
        :data:`UTILISATION_METRICS`.
    """
    if not 0 < utilisation <= 1:
        raise ValueError(f"utilisation must be in (0, 1], got {utilisation}")
    return n_units * duration_s * peak_flops * utilisation


def hardware_based_from_unit_hours(
    unit_hours: float,
    peak_flops: float,
    utilisation: float,
) -> float:
    """A.2.1 where the disclosure is in accelerator-hours rather than N and L.

    This is the form the Model Documentation Form actually elicits: the
    "hardware days" field asks for e.g. "4x10^5 Nvidia A100 days", which
    fixes only the product ``N * L``.
    """
    return hardware_based(1.0, unit_hours * SECONDS_PER_HOUR, peak_flops, utilisation)


def hardware_based_piecewise(
    segments: list[tuple[float, float, float, float]],
) -> float:
    """A.2.1 summed over periods of constant unit count.

    The Annex permits this where "different numbers of hardware units are
    used for different periods of time".

    Parameters
    ----------
    segments:
        ``(n_units, duration_s, peak_flops, utilisation)`` per period.
    """
    return sum(hardware_based(*seg) for seg in segments)


def weighted_peak(units: list[tuple[int, float]]) -> float:
    """Unit-count-weighted mean peak, per A.2.1.

    The Annex requires a weighted average "where the weights are determined
    by the number of hardware units of each type".

    Parameters
    ----------
    units:
        ``(count, peak_flops)`` per accelerator type.
    """
    total = sum(count for count, _ in units)
    if total == 0:
        raise ValueError("no hardware units given")
    return sum(count * peak for count, peak in units) / total


# ---------------------------------------------------------------------------
# A.2.2 -- architecture-based
# ---------------------------------------------------------------------------


def architecture_based_6pd(params: float, tokens: float) -> float:
    """Annex A.2.2: ``C = 6 * P * D``.

    Valid for dense transformers where, per the Annex's footnote 14, the
    parameter count is "significantly larger than one twelfth of the number
    of tokens in the input context".
    """
    return 6.0 * params * tokens


def architecture_based_transformer(
    *,
    non_embedding_params: float,
    tokens: float,
    n_layers: int,
    d_model: int,
    context_length: int,
) -> float:
    """``6 * P * D`` with the attention term made explicit.

    The ``6 * P * D`` approximation drops the context-dependent cost of attention.
    That term grows with the square of context length, so the approximation
    increasingly understates compute as context grows -- the regime the
    Annex's own footnote 14 warns about but does not quantify.

    The attention contribution per token is approximately
    ``12 * n_layers * d_model * context_length`` FLOP, counting forward and
    backward passes.
    """
    dense = 6.0 * non_embedding_params * tokens
    attention = 12.0 * n_layers * d_model * context_length * tokens
    return dense + attention


def architecture_based_moe(
    *,
    active_params: float,
    tokens: float,
) -> float:
    """``6 * P * D`` over *active* parameters, for mixture-of-experts models.

    Using total parameters overstates compute by up to the sparsity factor;
    using active parameters understates it by omitting routing and
    all-to-all cost. The Annex addresses neither. Report both bounds.
    """
    return 6.0 * active_params * tokens


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reconciliation:
    """Result of comparing an A.2.1 estimate against an A.2.2 estimate."""

    hardware_estimate: float
    architecture_estimate: float
    implied_utilisation: float
    ratio: float
    within_margin: bool

    def __str__(self) -> str:
        return (
            f"A.2.1 = {self.hardware_estimate:.3g} FLOP, "
            f"A.2.2 = {self.architecture_estimate:.3g} FLOP, "
            f"ratio = {self.ratio:.2f}x, "
            f"implied U = {self.implied_utilisation:.3f}, "
            f"{'within' if self.within_margin else 'OUTSIDE'} the 30% margin"
        )


def reconcile(
    *,
    n_units: float,
    duration_s: float,
    peak_flops: float,
    utilisation: float,
    architecture_estimate: float,
) -> Reconciliation:
    """Compare the two approaches and back out the utilisation they imply.

    The Guidelines offer both approaches and no tie-breaker. Where they
    disagree, the residual *is* the achieved-utilisation term: the
    architecture-based estimate divided by the hardware ceiling gives the
    utilisation that would have been required for the two to agree.
    """
    hw = hardware_based(n_units, duration_s, peak_flops, utilisation)
    ceiling = n_units * duration_s * peak_flops
    implied = architecture_estimate / ceiling
    ratio = max(hw, architecture_estimate) / min(hw, architecture_estimate)
    return Reconciliation(
        hardware_estimate=hw,
        architecture_estimate=architecture_estimate,
        implied_utilisation=implied,
        ratio=ratio,
        within_margin=within_margin(hw, architecture_estimate),
    )


# ---------------------------------------------------------------------------
# The margin
# ---------------------------------------------------------------------------


def within_margin(estimate: float, reference: float, margin: float = MARGIN) -> bool:
    """Is ``estimate`` within ``margin`` of ``reference``?

    Annex A.2 paragraph 120 requires accuracy "within an overall error
    margin of 30% of the reported estimate" -- the denominator is the
    reported estimate, not the true value.
    """
    return abs(estimate - reference) / estimate <= margin


def margin_interval(estimate: float, margin: float = MARGIN) -> tuple[float, float]:
    """True values compatible with a reported ``estimate``.

    A report of 7.7e24 FLOP with a +/-30% margin is consistent with a true
    value up to 1e25. The Guidelines do not say how a margin that crosses
    1e25 should be treated (see Section 6.3 of the paper).
    """
    return estimate * (1 - margin), estimate * (1 + margin)


def round_sigfigs(value: float, sigfigs: int) -> float:
    """Round ``value`` to ``sigfigs`` significant figures.

    Needed to compare against the Annex's own published figures, which are
    themselves rounded -- Models A and C-H to two significant figures,
    Model B to one.
    """
    if value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value))) - (sigfigs - 1)
    ulp = 10.0**exponent
    return round(value / ulp) * ulp


def significant_figure_interval(value: float, sigfigs: int) -> tuple[float, float]:
    """Interval of true values consistent with a rounded disclosure.

    The Model Documentation Form requires hardware days "recorded with at
    least one significant figure". A disclosure of ``4e5 A100 days`` is
    therefore consistent with anything in [3.5e5, 4.5e5) -- a +/-12.5%
    quantisation of the ``H * T`` term, before any estimation assumption is
    made, against a total permitted margin of 30%.
    """
    if value <= 0:
        raise ValueError("value must be positive")
    exponent = math.floor(math.log10(abs(value))) - (sigfigs - 1)
    ulp = 10.0**exponent
    rounded = round(value / ulp) * ulp
    return rounded - ulp / 2, rounded + ulp / 2


def significant_figure_error(value: float, sigfigs: int) -> float:
    """Worst-case relative error from rounding to ``sigfigs``.

    Returns the fraction of the 30% budget this alone can consume via
    :func:`fraction_of_margin`.
    """
    lo, hi = significant_figure_interval(value, sigfigs)
    mid = (lo + hi) / 2
    return (hi - mid) / mid


def fraction_of_margin(relative_error: float, margin: float = MARGIN) -> float:
    """What share of the permitted margin a given relative error consumes."""
    return relative_error / margin


# ---------------------------------------------------------------------------
# Paper §4.1 -- what "utilisation" can mean
# ---------------------------------------------------------------------------

UTILISATION_METRICS: dict[str, tuple[float, float, str]] = {
    "nvidia-smi occupancy": (
        0.90,
        0.99,
        "utilization.gpu: fraction of sampling intervals in which at least "
        "one kernel was resident. Says nothing about work done. Reads ~100% "
        "for a job doing pure memory traffic and zero arithmetic.",
    ),
    "DCGM tensor-pipe activity": (
        0.30,
        0.70,
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: duty cycle of the tensor pipes. "
        "A kernel running at 5% of peak FLOP/s still scores fully active.",
    ),
    "model FLOP utilisation (MFU)": (
        0.20,
        0.50,
        "Achieved model FLOP/s divided by peak FLOP/s. The only one of the "
        "three that measures useful arithmetic.",
    ),
}
"""Quantities a good-faith reader could take Annex A.2.1's ``U`` to denote.

Ranges are typical for large-scale LLM pre-training and are documented in
the systems literature; they are priors for the desk analysis, to be
replaced by measured values from the accompanying harness.
"""


def utilisation_spread(
    n_units: float,
    duration_s: float,
    peak_flops: float,
    metrics: dict[str, tuple[float, float, str]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Reported compute under each candidate reading of ``U``.

    This is paper §4.1 in arithmetic form: one run, one set of undisputed
    public facts, and a range of lawful answers.
    """
    metrics = metrics or UTILISATION_METRICS
    ceiling = n_units * duration_s * peak_flops
    return {
        name: (ceiling * lo, ceiling * hi) for name, (lo, hi, _) in metrics.items()
    }


def peak_spread(
    n_units: float,
    duration_s: float,
    accelerator: str,
    utilisation: float,
) -> dict[str, float]:
    """Reported compute under each candidate reading of ``P``.

    This is paper §4.2 in arithmetic form.
    """
    acc = peaks.get(accelerator)
    return {
        label: hardware_based(n_units, duration_s, value, utilisation)
        for label, value in acc.candidate_peaks().items()
    }


THRESHOLD = 1e25
"""The Article 51(2) presumption threshold, in FLOP."""


def crosses_threshold(estimates: dict[str, float], threshold: float = THRESHOLD) -> bool:
    """Do the candidate estimates straddle the threshold?

    ``True`` means the classification of the model under Article 51(2)
    depends on which reading of an undefined term the reader adopts.
    """
    values = list(estimates.values())
    return min(values) <= threshold < max(values)
