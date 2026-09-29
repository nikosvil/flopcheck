"""The consultation draft's examples, and cross-checks against public models."""

from __future__ import annotations

import pytest

from flopcheck import commission_examples as ce
from flopcheck import estimate


@pytest.mark.parametrize("example", ce.DRAFT_EXAMPLES, ids=lambda e: e.label)
def test_reproduces_draft_figure(example):
    """Every figure in the draft's Annex A.1 is recoverable at its own precision."""
    assert example.agrees(), (
        f"Draft {example.label}: recomputed {example.compute():.4g}, "
        f"published {example.published:.4g}"
    )


def test_only_draft_model_a_uses_fully_sourced_inputs():
    """Draft Model A sources both peak and utilisation; B and D borrow them."""
    dependent = {e.label for e in ce.assumption_dependent(ce.DRAFT_EXAMPLES)}
    assert dependent == {"B", "D"}


def test_draft_model_a_agrees_across_the_two_approaches():
    """The draft's own cross-check: hardware and 6*P*D agree for Model A."""
    hw = next(e for e in ce.DRAFT_EXAMPLES if e.label == "A (hardware)").compute()
    arch = next(e for e in ce.DRAFT_EXAMPLES if e.label == "A (architecture)").compute()
    assert estimate.within_margin(hw, arch, margin=0.10)


def test_diffusion_example_changed_utilisation_between_versions():
    """Same model, same GPU-hours: 56% in the draft, 50% in the final."""
    draft = next(e for e in ce.DRAFT_EXAMPLES if e.label == "D")
    final = next(e for e in ce.EXAMPLES if e.label == "B")
    assert draft.given["unit_hours"] == final.given["unit_hours"] == 200_000
    assert draft.assumed["utilisation"] == pytest.approx(0.56)
    assert final.assumed["utilisation"] == pytest.approx(0.50)


def _check(model: str) -> ce.CrossCheck:
    return next(c for c in ce.CROSS_CHECKS if c.model == model)


def test_sourced_utilisation_reconciles():
    """TinyLlama: utilisation from the model's own documentation -> agreement."""
    c = _check("TinyLlama 1.1B")
    assert c.within_margin
    assert c.ratio == pytest.approx(1.07, abs=0.02)


def test_borrowed_utilisation_does_not_reconcile():
    """Pythia-1B: TinyLlama's 56% carried over -> 1.6x, outside the margin."""
    c = _check("Pythia-1B")
    assert not c.within_margin
    assert 1.5 < c.ratio < 1.7
    assert 0.33 < c.implied_utilisation < 0.37


def test_assumed_utilisation_is_an_order_of_magnitude_off():
    """Llama 3.2 1B: the two approved approaches differ by ~10x.

    The public token count is an upper bound, so the ratio is a lower bound
    and the implied utilisation an upper bound.
    """
    c = _check("Llama 3.2 1B")
    assert c.tokens_is_upper_bound
    assert c.ratio > 9
    assert c.implied_utilisation < 0.06


def test_every_cross_check_documents_its_evidence():
    for c in ce.CROSS_CHECKS:
        assert c.evidence and c.example.approach == "hardware"
