"""Generate the paper's tables and figure data from the analyses.

Run as ``python -m flopcheck.report``.
"""

from __future__ import annotations

from . import commission_examples as ce
from . import estimate, peaks


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def table_1() -> str:
    """Annex A.3 reproduced, with the assumptions exposed."""
    lines = [
        "Table 1 -- The Commission's eight worked examples (Annex A.3), reproduced",
        "",
        f"{'Model':<6} {'Approach':<13} {'Published':<11} {'Recomputed':<11} "
        f"{'OK':<4} Assumptions supplied by the Annex",
        "-" * 100,
    ]
    for row in ce.table():
        lines.append(
            f"{row['label']:<6} {row['approach']:<13} "
            f"{_fmt(row['published']):<11} {_fmt(row['recomputed']):<11} "
            f"{'yes' if row['agrees'] else 'NO':<4} {row['assumptions']}"
        )
    dependent = [e.label for e in ce.assumption_dependent()]
    lines += [
        "",
        f"Six of eight follow from published facts alone. Models {' and '.join(dependent)} "
        "do not:",
        "they are recoverable only by supplying a peak performance and a GPU",
        "utilisation that the Annex asserts without derivation or citation.",
    ]
    return "\n".join(lines)


def table_2() -> str:
    """Candidate peaks admitted by Annex A.2.1's undefined ``P``."""
    lines = [
        "Table 2 -- 'Peak theoretical performance' is precision-ambiguous",
        "",
        f"{'Accelerator':<30} {'Min TFLOP/s':>12} {'Max TFLOP/s':>12} "
        f"{'Spread':>8}  Status",
        "-" * 90,
    ]
    for acc in peaks.all_accelerators():
        candidates = acc.candidate_peaks()
        lines.append(
            f"{acc.name:<30} {min(candidates.values())/1e12:>12.1f} "
            f"{max(candidates.values())/1e12:>12.1f} "
            f"{acc.peak_spread():>7.1f}x  {acc.status}"
        )
    lines += [
        "",
        "Annex A.2.1 requires 'the peak theoretical performance ... measured in",
        "FLOP/second' without naming a precision or a sparsity basis, while the",
        "same section requires that 'all operations should be counted equally,",
        "independently of floating-point precision'. The numerator of the",
        "utilisation term is thus precision-independent and its denominator is",
        "not; the Annex does not reconcile them.",
    ]
    return "\n".join(lines)


def figure_1_data(n_units: int = 6_000, days: int = 30) -> str:
    """One run, three readings of ``U``, and the Article 51(2) threshold."""
    duration_s = days * estimate.SECONDS_PER_DAY
    peak = peaks.get("H100").peak("bf16")
    ceiling = n_units * duration_s * peak
    spread = estimate.utilisation_spread(n_units, duration_s, peak)

    lines = [
        f"Figure 1 -- {n_units:,} NVIDIA H100 SXM for {days} days",
        "",
        f"Disclosed facts: {n_units:,} accelerators, {days} days, "
        f"peak {peak/1e12:.1f} TFLOP/s (BF16 dense).",
        f"Ceiling at U=1: {_fmt(ceiling)} FLOP.",
        f"Article 51(2) threshold: {_fmt(estimate.THRESHOLD)} FLOP.",
        "",
        f"{'Reading of U':<34} {'Range':<10} {'Reported compute':<26} Verdict",
        "-" * 92,
    ]
    for name, (lo_c, hi_c) in spread.items():
        lo_u, hi_u, _ = estimate.UTILISATION_METRICS[name]
        if lo_c > estimate.THRESHOLD:
            verdict = "SYSTEMIC RISK"
        elif hi_c < estimate.THRESHOLD:
            verdict = "not systemic risk"
        else:
            verdict = "straddles the threshold"
        lines.append(
            f"{name:<34} {lo_u:.2f}-{hi_u:.2f}  "
            f"{_fmt(lo_c)} - {_fmt(hi_c)} FLOP{'':<4} {verdict}"
        )

    occ_lo = spread["nvidia-smi occupancy"][0]
    mfu_hi = spread["model FLOP utilisation (MFU)"][1]
    lines += [
        "",
        f"Same run, same undisputed facts. The occupancy reading exceeds the",
        f"threshold by {occ_lo/estimate.THRESHOLD:.1f}x; the MFU reading falls short of it",
        f"by {estimate.THRESHOLD/mfu_hi:.1f}x. The classification of the model under",
        f"Article 51(2) is decided by which quantity the reader takes an",
        f"undefined term to denote.",
    ]
    return "\n".join(lines)


def table_3() -> str:
    """Paper §6.1: what the Form's rounding costs before any estimation."""
    lines = [
        "Table 3 -- Disclosure precision consumes the error budget",
        "",
        "The Model Documentation Form requires hardware days 'recorded with at",
        "least one significant figure'; the compute figure to national competent",
        "authorities is required only 'up to its order of magnitude'.",
        "",
        f"{'Disclosure':<40} {'Sig figs':>9} {'Implied +/-':>12} {'Share of 30% margin':>21}",
        "-" * 86,
    ]
    cases = [
        ("Hardware days to the AI Office (4e5)", 4e5, 1),
        ("Compute to the AI Office (2.4e25)", 2.4e25, 2),
        ("Compute to NCAs, read as 1 sig fig", 1e25, 1),
    ]
    for label, value, sf in cases:
        err = estimate.significant_figure_error(value, sf)
        share = estimate.fraction_of_margin(err)
        lines.append(f"{label:<40} {sf:>9} {err:>11.1%} {share:>20.0%}")

    # "Order of magnitude" admits a second, wider reading: the exponent alone,
    # so 1e25 covers [10^24.5, 10^25.5]. Both readings exceed the margin.
    lines.append(
        f"{'Compute to NCAs, read as the exponent':<40} {'-':>9} "
        f"{'x3.2':>11} {'>10x':>20}"
    )
    lines += [
        "",
        "The one-significant-figure hardware-days field alone consumes 42% of the",
        "entire permitted margin before any estimation assumption is made.",
        "",
        "The compute figure disclosed to national competent authorities is worse",
        "still. Read narrowly as one significant figure it carries +/-50%, i.e.",
        "1.67 times the margin the provider is bound to; read as the exponent",
        "alone it spans a factor of 3.2 either way. On either reading the",
        "disclosure is less precise than the accuracy the Guidelines require of",
        "the estimate behind it, so an NCA cannot in principle check compliance",
        "with the 30% margin from the figure it is given.",
    ]
    return "\n".join(lines)


def table_1b() -> str:
    """Hardware-based examples, draft and final, against the models they match."""
    lines = [
        "Table 1b -- Hardware-based examples checked against the models they match",
        "",
        f"{'Example':<24} {'Matches (inferred)':<18} {'U used':>6}  {'A.2.1':<9} {'6PD':<9} "
        f"{'Ratio':>7} {'Implied U':>10}",
        "-" * 92,
    ]
    for c in ce.CROSS_CHECKS:
        ex = c.example
        u = {**ex.given, **ex.assumed}["utilisation"]
        tag = "draft " if ex.version == ce.DRAFT else "final "
        bound = c.tokens_is_upper_bound
        ratio = f"{'>=' if bound else ''}{c.ratio:.2f}"
        implied = f"{'<=' if bound else ''}{c.implied_utilisation:.2f}"
        lines.append(
            f"{tag + ex.label:<24} {c.model:<18} {u:>6.2f}  {_fmt(c.hardware_estimate):<9} "
            f"{_fmt(c.architecture_estimate):<9} {ratio:>7} {implied:>10}"
        )
    lines += [
        "",
        "Only the example whose utilisation is the model's own measured MFU reconciles.",
        "A borrowed utilisation misses by 1.6x; an assumed one by an order of magnitude.",
        "Matches rest on exact GPU-hour figures; see commission_examples.CROSS_CHECKS.",
    ]
    return "\n".join(lines)


def main() -> None:
    for section in (table_1(), table_1b(), table_2(), figure_1_data(), table_3()):
        print(section)
        print("\n")


if __name__ == "__main__":
    main()
