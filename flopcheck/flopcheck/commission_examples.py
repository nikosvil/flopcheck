"""The worked examples of the Commission's GPAI Guidelines, draft and final.

The Commission's Guidelines on the scope of the obligations for
general-purpose AI models exist in two published versions:

  * a consultation draft (June 2025, "Targeted consultation in preparation of
    the Commission guidelines ..."), whose Annex A.1 works four examples; and
  * the final Guidelines (18 July 2025), whose Annex A.3 works eight.

Both illustrate the two estimation approaches of Annex A.2. The
architecture-based examples are fully determined by published facts. The
hardware-based examples need a peak performance and a GPU utilisation, and
the two versions source these differently:

  * the draft takes its utilisation (56%) from one model's own documentation
    -- a figure that documentation describes as *model FLOPs utilisation* --
    states the peak basis (bfloat16, no sparsity), and then reuses both for
    other models;
  * the final text assumes 50%, cites nothing, drops the peak-basis footnote,
    and reuses the figure with the words "GPU utilisation as previously".

This module encodes every example with its inputs split into ``given``
(stated or sourced by the Commission) and ``assumed`` (asserted without a
source, or borrowed from a different model). It then cross-checks the
hardware-based examples against the architecture-based estimate for the
publicly documented model each one matches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import estimate

TFLOPS = 1e12

FINAL = "final (18 July 2025)"
DRAFT = "draft (June 2025 consultation)"


@dataclass(frozen=True)
class Example:
    """One worked example from the Guidelines."""

    label: str
    description: str
    approach: str
    published: float
    #: Significant figures to which the Commission reports this figure.
    published_sigfigs: int = 2
    #: Inputs the Commission states or sources.
    given: dict[str, float] = field(default_factory=dict)
    #: Inputs the Commission asserts without a source, or borrows from
    #: another model.
    assumed: dict[str, float] = field(default_factory=dict)
    #: Where the utilisation figure came from, in the Commission's words.
    utilisation_source: str = ""
    version: str = FINAL
    note: str = ""

    @property
    def rests_on_assumption(self) -> bool:
        return bool(self.assumed)

    def _input(self, key: str) -> float:
        return {**self.given, **self.assumed}[key]

    def compute(self) -> float:
        """Recompute the Commission's figure from its own inputs."""
        if self.approach == "architecture":
            return estimate.architecture_based_6pd(
                self._input("params"), self._input("tokens")
            )
        if self.approach == "hardware":
            return estimate.hardware_based_from_unit_hours(
                self._input("unit_hours"),
                self._input("peak_flops"),
                self._input("utilisation"),
            )
        raise ValueError(f"unknown approach {self.approach!r}")

    def ceiling(self) -> float:
        """Hardware-based compute at 100% utilisation."""
        if self.approach != "hardware":
            raise ValueError("ceiling is defined for hardware-based examples only")
        return (
            self._input("unit_hours")
            * estimate.SECONDS_PER_HOUR
            * self._input("peak_flops")
        )

    def agrees(self) -> bool:
        """Does the recomputed value round to the published figure?

        The Commission reports most figures to two significant figures and
        some to one, so agreement at its own stated precision is the right
        test.
        """
        rounded = estimate.round_sigfigs(self.compute(), self.published_sigfigs)
        return math.isclose(rounded, self.published, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Final Guidelines, Annex A.3
# ---------------------------------------------------------------------------

EXAMPLES: tuple[Example, ...] = (
    Example(
        label="A",
        description="Language model, transformer decoder, 3.8e9 parameters",
        approach="architecture",
        given={"params": 3.8e9, "tokens": 3.3e12},
        published=7.5e22,
    ),
    Example(
        label="B",
        description="Image diffusion model, 200 000 A100-40GB GPU-hours",
        approach="hardware",
        given={"unit_hours": 200_000},
        assumed={"peak_flops": 300 * TFLOPS, "utilisation": 0.5},
        published=1e23,
        published_sigfigs=1,
        utilisation_source="'Assuming ... GPU utilisation of 50%' -- no source",
        note=(
            "300 TFLOP/s is approximately the A100 dense BF16/FP16 figure "
            "(312); the TF32 dense figure is 156 and the sparse figure 624. "
            "The draft stated the basis ('bfloat16 training with no "
            "sparsity'); the final text does not."
        ),
    ),
    Example(
        label="C",
        description="Language model, transformer decoder, 600e6 parameters",
        approach="architecture",
        given={"params": 0.6e9, "tokens": 36e12},
        published=1.3e23,
    ),
    Example(
        label="D",
        description="Language model, transformer decoder, 1.54e9 parameters",
        approach="architecture",
        given={"params": 1.54e9, "tokens": 18e12},
        published=1.7e23,
    ),
    Example(
        label="E",
        description="Language model, transformer decoder, 1.7e9 parameters",
        approach="architecture",
        given={"params": 1.7e9, "tokens": 36e12},
        published=3.7e23,
    ),
    Example(
        label="F",
        description="Language model, transformer decoder, 2.5e9 parameters",
        approach="architecture",
        given={"params": 2.5e9, "tokens": 12e12},
        published=1.8e23,
    ),
    Example(
        label="G",
        description=(
            "Language model, 2.6e9 parameters, 61e9 tokens x 164 epochs "
            "~ 10e12 total training tokens"
        ),
        approach="architecture",
        given={"params": 2.6e9, "tokens": 10e12},
        published=1.6e23,
    ),
    Example(
        label="H",
        description="Language model, 370 000 H100-80GB GPU-hours",
        approach="hardware",
        given={"unit_hours": 370_000},
        assumed={"peak_flops": 989 * TFLOPS, "utilisation": 0.5},
        published=6.6e23,
        utilisation_source="'GPU utilisation as previously' -- inherited from Model B",
        note=(
            "989 TFLOP/s is the H100 SXM dense BF16 figure; the candidate "
            "peaks for that part span 494.7 to 3957.8, a factor of 8."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Consultation draft, Annex A.1
# ---------------------------------------------------------------------------

DRAFT_EXAMPLES: tuple[Example, ...] = (
    Example(
        label="A (hardware)",
        description=(
            "Language model, 1.1e9 parameters; 3 456 A100 GPU-hours per 300e9 "
            "tokens, scaled to 3e12 tokens as ~35 000 GPU-hours"
        ),
        approach="hardware",
        given={"unit_hours": 35_000, "peak_flops": 300 * TFLOPS, "utilisation": 0.56},
        published=2.1e22,
        utilisation_source="'56% as indicated in model's documentation'",
        version=DRAFT,
        note=(
            "The peak is sourced to the A100 datasheet, 'assuming bfloat16 "
            "training with no sparsity'. The 56% is the model's own reported "
            "figure, which its documentation calls model FLOPs utilisation. "
            "The only hardware-based example, in either version, whose "
            "inputs are all sourced."
        ),
    ),
    Example(
        label="A (architecture)",
        description="Same model, 6*P*D with 1.1e9 parameters and 3e12 tokens",
        approach="architecture",
        given={"params": 1.1e9, "tokens": 3e12},
        published=1.98e22,
        published_sigfigs=3,
        version=DRAFT,
        note="The draft computes Model A both ways and notes that they agree.",
    ),
    Example(
        label="B",
        description="Language model, 1e9 parameters, 4 830 A100-40GB GPU-hours",
        approach="hardware",
        given={"unit_hours": 4_830, "peak_flops": 300 * TFLOPS},
        assumed={"utilisation": 0.56},
        published=3e21,
        published_sigfigs=1,
        utilisation_source="'Assuming the same numbers ... as for Model A' -- borrowed",
        version=DRAFT,
    ),
    Example(
        label="C",
        description="Language model, transformer decoder, 3.8e9 parameters",
        approach="architecture",
        given={"params": 3.8e9, "tokens": 3.3e12},
        published=7.5e22,
        version=DRAFT,
    ),
    Example(
        label="D",
        description="Image diffusion model, 200 000 A100-40GB GPU-hours",
        approach="hardware",
        given={"unit_hours": 200_000, "peak_flops": 300 * TFLOPS},
        assumed={"utilisation": 0.56},
        published=1.2e23,
        utilisation_source="'Assuming the same numbers ... as previously' -- borrowed",
        version=DRAFT,
        note="The same model as final Model B, where utilisation became 50%.",
    ),
)


# ---------------------------------------------------------------------------
# Cross-checks against publicly documented models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossCheck:
    """A hardware-based example set against the model it matches.

    The Guidelines anonymise their examples. Each match here rests on exact
    agreement between the published GPU-hours and a public model card or
    paper, and is an inference, not a statement by the Commission.
    """

    example: Example
    model: str
    params: float
    tokens: float
    evidence: str
    #: True where the public token count is an upper bound ("up to").
    tokens_is_upper_bound: bool = False

    @property
    def architecture_estimate(self) -> float:
        return estimate.architecture_based_6pd(self.params, self.tokens)

    @property
    def hardware_estimate(self) -> float:
        return self.example.compute()

    @property
    def ratio(self) -> float:
        """Hardware-based over architecture-based.

        With an upper-bound token count this ratio is a lower bound.
        """
        return self.hardware_estimate / self.architecture_estimate

    @property
    def implied_utilisation(self) -> float:
        """Utilisation at which the two approaches would agree."""
        return self.architecture_estimate / self.example.ceiling()

    @property
    def within_margin(self) -> bool:
        return estimate.within_margin(self.hardware_estimate, self.architecture_estimate)


def _find(examples: tuple[Example, ...], label: str) -> Example:
    return next(e for e in examples if e.label == label)


CROSS_CHECKS: tuple[CrossCheck, ...] = (
    CrossCheck(
        example=_find(DRAFT_EXAMPLES, "A (hardware)"),
        model="TinyLlama 1.1B",
        params=1.1e9,
        tokens=3e12,
        evidence=(
            "TinyLlama paper (arXiv:2401.02385): 1.1B parameters, 3T tokens, "
            "3 456 A100-40G GPU-hours per 300B tokens. README: 24k tokens/s "
            "per A100-40G, '56% model flops utilization'."
        ),
    ),
    CrossCheck(
        example=_find(DRAFT_EXAMPLES, "B"),
        model="Pythia-1B",
        params=1.0118e9,
        tokens=2.99892736e11,
        evidence=(
            "TinyLlama paper, Table 2: Pythia-1.0B at 4 830 GPU-hours for "
            "300B tokens. Pythia README: 299 892 736 000 training tokens; "
            "16 layers, d_model 2048. Parameter count derived from that "
            "architecture with untied 50 304-token embeddings (~1.01B)."
        ),
    ),
    CrossCheck(
        example=_find(EXAMPLES, "H"),
        model="Llama 3.2 1B",
        params=1.23e9,
        tokens=9e12,
        tokens_is_upper_bound=True,
        evidence=(
            "Llama 3.2 model card: 370k H100-80GB GPU-hours for the 1B model; "
            "1.23B parameters; 'up to 9T tokens'; logits from Llama 3.1 8B "
            "and 70B used as token-level targets during pre-training, with "
            "pruning and knowledge distillation."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table(examples: tuple[Example, ...] = EXAMPLES) -> list[dict[str, object]]:
    """Every example, recomputed, with its assumptions exposed."""
    rows = []
    for ex in examples:
        rows.append(
            {
                "label": ex.label,
                "version": ex.version,
                "approach": ex.approach,
                "published": ex.published,
                "recomputed": ex.compute(),
                "agrees": ex.agrees(),
                "assumptions": ", ".join(f"{k}={v:.3g}" for k, v in ex.assumed.items())
                or "none",
                "utilisation_source": ex.utilisation_source,
                "rests_on_assumption": ex.rests_on_assumption,
            }
        )
    return rows


def assumption_dependent(examples: tuple[Example, ...] = EXAMPLES) -> list[Example]:
    """The examples that cannot be reproduced from sourced inputs alone."""
    return [ex for ex in examples if ex.rests_on_assumption]
