"""Peak theoretical performance of AI accelerators.

Annex A.2.1 of the Commission's GPAI Guidelines (18 July 2025) requires
"the peak theoretical performance ... of the GPUs or other hardware units
used ... (measured in FLOP/second)".

It does not say *which* peak. A modern accelerator has a different peak for
every supported number format, and vendors additionally quote a doubled
figure obtained with 2:1 structured sparsity. For an H100 SXM the candidate
values span 494.7 -> 3957.8 TFLOP/s, a factor of exactly 8.

This module makes that spread explicit and machine-readable. Every value
carries a provenance tag so a reader can check it against a datasheet.

Note the tension inside Annex A.2: the same paragraph that requires a peak
figure also states that "all operations should be counted equally,
independently of floating-point precision". The numerator of the utilisation
term is therefore precision-independent while its denominator is not, and
the Guidelines never reconcile the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

TFLOPS = 1e12

#: Precision labels used throughout. "tf32", "bf16", "fp16", "fp8", "fp4".
Precision = str


class Verification(str):
    """How far a peak figure has been checked."""


#: Value taken from a vendor datasheet and cross-checked. Safe to cite.
CONFIRMED = Verification("confirmed")
#: Value believed correct but not yet cross-checked against a primary
#: datasheet. MUST be verified before publication.
UNVERIFIED = Verification("unverified")


@dataclass(frozen=True)
class Accelerator:
    """Peak theoretical performance, by precision, for one accelerator.

    Attributes
    ----------
    name:
        Vendor's product name, as it would appear in a Model Documentation
        Form "hardware days" entry.
    dense:
        Mapping of precision -> peak FLOP/s without sparsity.
    sparse_multiplier:
        Factor the vendor applies when quoting structured-sparsity figures.
        ``None`` where the vendor does not quote a sparse number.
    status:
        ``CONFIRMED`` or ``UNVERIFIED``.
    note:
        Free text; typically the datasheet or the reason for doubt.
    """

    name: str
    dense: dict[Precision, float]
    sparse_multiplier: float | None = 2.0
    status: Verification = UNVERIFIED
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def peak(self, precision: Precision, *, sparse: bool = False) -> float:
        """Peak FLOP/s for ``precision``.

        Raises
        ------
        KeyError
            If the accelerator has no published peak for that precision.
        ValueError
            If ``sparse`` is requested but the vendor quotes no sparse figure.
        """
        if precision not in self.dense:
            raise KeyError(
                f"{self.name} has no recorded peak for precision {precision!r}. "
                f"Known: {sorted(self.dense)}"
            )
        value = self.dense[precision]
        if not sparse:
            return value
        if self.sparse_multiplier is None:
            raise ValueError(f"{self.name} has no vendor-quoted sparse peak.")
        return value * self.sparse_multiplier

    def candidate_peaks(self) -> dict[str, float]:
        """Every value a good-faith reader could take as "the" peak.

        This is the raw material for Finding 2: the set of numbers that
        Annex A.2.1's undefined term admits.
        """
        out: dict[str, float] = {}
        for precision, value in self.dense.items():
            out[f"{precision}-dense"] = value
            if self.sparse_multiplier is not None:
                out[f"{precision}-sparse"] = value * self.sparse_multiplier
        return out

    def peak_spread(self) -> float:
        """Ratio of the largest to smallest candidate peak.

        For an H100 SXM this is 8.0.
        """
        values = self.candidate_peaks().values()
        return max(values) / min(values)


# ---------------------------------------------------------------------------
# The table.
#
# CONFIRMED entries are those cross-checked against vendor datasheets and,
# where possible, against the figures the Commission itself used in Annex A.3.
# UNVERIFIED entries must be checked before they appear in a publication.
# ---------------------------------------------------------------------------

_ACCELERATORS: tuple[Accelerator, ...] = (
    Accelerator(
        name="NVIDIA A100 SXM 80GB",
        dense={"tf32": 156 * TFLOPS, "bf16": 312 * TFLOPS, "fp16": 312 * TFLOPS},
        sparse_multiplier=2.0,
        status=CONFIRMED,
        note=(
            "NVIDIA A100 datasheet. The Commission's Annex A.3 Model B uses "
            "300 TFLOP/s for a 40GB A100, i.e. approximately the BF16/FP16 "
            "dense figure, without saying so."
        ),
        aliases=("A100", "A100 80GB", "Nvidia A100"),
    ),
    Accelerator(
        name="NVIDIA A100 SXM 40GB",
        dense={"tf32": 156 * TFLOPS, "bf16": 312 * TFLOPS, "fp16": 312 * TFLOPS},
        sparse_multiplier=2.0,
        status=CONFIRMED,
        note="Same compute as the 80GB part; memory capacity differs.",
        aliases=("A100 40GB",),
    ),
    Accelerator(
        name="NVIDIA H100 SXM",
        dense={
            "tf32": 494.7 * TFLOPS,
            "bf16": 989.4 * TFLOPS,
            "fp16": 989.4 * TFLOPS,
            "fp8": 1978.9 * TFLOPS,
        },
        sparse_multiplier=2.0,
        status=CONFIRMED,
        note=(
            "NVIDIA H100 datasheet. The Commission's Annex A.3 Model H uses "
            "989 TFLOP/s, i.e. the BF16 dense figure. Candidate peaks span "
            "494.7 to 3957.8 TFLOP/s: a factor of 8."
        ),
        aliases=("H100", "H100 80GB", "Nvidia H100"),
    ),
    Accelerator(
        name="NVIDIA H100 PCIe",
        dense={"tf32": 378.2 * TFLOPS, "bf16": 756.45 * TFLOPS, "fp16": 756.45 * TFLOPS, "fp8": 1512.9 * TFLOPS},
        sparse_multiplier=2.0,
        status=UNVERIFIED,
        note=(
            "Derived as for the SXM's 989.4 (132 SMs x 4096 FLOPs/cycle x 1830 MHz): "
            "114 SMs (NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.01) x "
            "4096 FLOPs/cycle (Pedersen et al. 2026) x a 1620 MHz tensor-core boost "
            "clock (NVIDIA developer forum) = 756.45 TFLOP/s BF16 dense. The same "
            "whitepaper rounds PCIe to 800 and SXM5 to 1000. The Model Documentation "
            "Form's example 'Nvidia H100 days' does not distinguish PCIe from SXM, "
            "whose peaks differ by a factor of 1.31."
        ),
        aliases=("H100 PCIe",),
    ),
    Accelerator(
        name="NVIDIA H200 SXM",
        dense={
            "tf32": 494.7 * TFLOPS,
            "bf16": 989.4 * TFLOPS,
            "fp16": 989.4 * TFLOPS,
            "fp8": 1978.9 * TFLOPS,
        },
        sparse_multiplier=2.0,
        status=CONFIRMED,
        note="Same compute silicon as H100 SXM; HBM capacity and bandwidth differ.",
        aliases=("H200",),
    ),
    Accelerator(
        name="NVIDIA GH200 Grace Hopper",
        dense={
            "tf32": 494.7 * TFLOPS,
            "bf16": 989.4 * TFLOPS,
            "fp16": 989.4 * TFLOPS,
            "fp8": 1978.9 * TFLOPS,
        },
        sparse_multiplier=2.0,
        status=UNVERIFIED,
        note=(
            "Hopper-class GPU paired with a Grace CPU. Several SKUs exist; "
            "confirm the exact part before citing. This is the accelerator in "
            "DAEDALUS (GRNET / EuroHPC, Lavrion) and is the same generation as "
            "the Commission's Model H example."
        ),
        aliases=("GH200",),
    ),
    Accelerator(
        name="NVIDIA B200",
        dense={"bf16": 2250 * TFLOPS, "fp8": 4500 * TFLOPS, "fp4": 9000 * TFLOPS},
        sparse_multiplier=2.0,
        status=UNVERIFIED,
        note=(
            "Blackwell. Figures widely reported but NOT cross-checked here. "
            "Verify against the datasheet before publication. Note that FP4 "
            "widens the Annex A.2.1 ambiguity from a factor of 8 to a factor "
            "of roughly 16."
        ),
        aliases=("B200",),
    ),
    Accelerator(
        name="NVIDIA V100 SXM2",
        dense={"fp16": 125 * TFLOPS},
        sparse_multiplier=None,
        status=CONFIRMED,
        note="Volta. No structured sparsity. Included for historical estimates.",
        aliases=("V100",),
    ),
    Accelerator(
        name="AMD Instinct MI250X",
        dense={"fp16": 383 * TFLOPS, "bf16": 383 * TFLOPS},
        sparse_multiplier=None,
        status=UNVERIFIED,
        note=(
            "AMD quotes no structured-sparsity figure, which by itself removes "
            "one axis of the Annex A.2.1 ambiguity. This is the accelerator in "
            "LUMI and is the cheapest route to a cross-vendor result."
        ),
        aliases=("MI250X",),
    ),
    Accelerator(
        name="AMD Instinct MI300X",
        dense={"fp16": 1307.4 * TFLOPS, "bf16": 1307.4 * TFLOPS, "fp8": 2614.9 * TFLOPS},
        sparse_multiplier=None,
        status=UNVERIFIED,
        note="Verify against the AMD datasheet before publication.",
        aliases=("MI300X",),
    ),
)

_BY_NAME: dict[str, Accelerator] = {}
for _acc in _ACCELERATORS:
    _BY_NAME[_acc.name.lower()] = _acc
    for _alias in _acc.aliases:
        _BY_NAME[_alias.lower()] = _acc


def get(name: str) -> Accelerator:
    """Look up an accelerator by name or alias, case-insensitively."""
    try:
        return _BY_NAME[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown accelerator {name!r}. Known: "
            f"{sorted({a.name for a in _ACCELERATORS})}"
        ) from None


def all_accelerators() -> Iterator[Accelerator]:
    """Iterate the table."""
    return iter(_ACCELERATORS)


def unverified() -> list[Accelerator]:
    """Entries that must be checked against a datasheet before publication."""
    return [a for a in _ACCELERATORS if a.status is UNVERIFIED]
