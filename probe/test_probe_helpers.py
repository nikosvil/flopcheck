"""Local tests for the probe's pure helpers -- no GPU, no torch needed."""

import pytest

import h100_probe as hp


def test_tinyllama_parameter_count_matches_published_size():
    c = hp.param_counts(hp.TINYLLAMA)
    assert c["total"] == 1_100_048_384          # "1.1B"
    assert c["matmul"] == 1_034_420_224         # excludes the input-embedding lookup


def test_model_flops_per_token_at_2048():
    f = hp.model_flops_per_token(hp.TINYLLAMA, 2048)
    assert f == pytest.approx(6 * 1_034_420_224 + 12 * 22 * 2048 * 2048)
    assert f == pytest.approx(7.31e9, rel=1e-3)


def test_attention_convention_matters_at_long_context():
    """Halving the causal attention term moves MFU by ~21% at 8k context."""
    full = hp.model_flops_per_token(hp.TINYLLAMA, 8192)
    half = hp.model_flops_per_token(hp.TINYLLAMA, 8192, causal_half=True)
    assert 1.15 < full / half < 1.30


@pytest.mark.parametrize(
    "name,label,peak,mhz",
    [
        ("NVIDIA H100 80GB HBM3", "H100 SXM", 989.4e12, 1830),
        ("NVIDIA H100 PCIe", "H100 PCIe", 756.45e12, 1620),
        ("NVIDIA A100-SXM4-80GB", None, None, None),
    ],
)
def test_detect_peak(name, label, peak, mhz):
    assert hp.detect_peak(name) == (label, peak, mhz)


def test_detect_peak_override():
    assert hp.detect_peak("anything", tflops=835.0, tc_mhz=1785) == ("override", 835e12, 1785)


def test_parse_dcgm_dmon_skips_headers_and_na():
    text = """#Entity   SMACT  TENSO  SMCLK  POWER  GPUTL
ID
GPU 0     0.912  0.521  1980   612.4  100
GPU 0     N/A    N/A    1980   610.0  100
GPU 0     0.905  0.518  1965   615.1  100
"""
    rows = hp.parse_dcgm_dmon(text, 5)
    assert rows == [[0.912, 0.521, 1980.0, 612.4, 100.0], [0.905, 0.518, 1965.0, 615.1, 100.0]]


def test_ofu_matches_pedersen_definition():
    """OFU = tensor-pipe activity x SM clock / 1830 MHz (H100)."""
    assert hp.ofu_from_samples([0.5, 0.5], [1830, 1830], 1830) == pytest.approx(0.5)
    assert hp.ofu_from_samples([0.6], [1525], 1830) == pytest.approx(0.5)


def test_summarise():
    s = hp.summarise([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert s["n"] == 10 and s["mean"] == pytest.approx(5.5) and s["median"] == pytest.approx(5.5)
    assert hp.summarise([]) == {"n": 0}


def test_sweep_covers_the_axes_the_paper_needs():
    seqs = {c.seq for c in hp.SWEEP}
    assert {2048, 8192} <= seqs
    assert any(c.ckpt for c in hp.SWEEP) and any(not c.ckpt for c in hp.SWEEP)
    # checkpointed and non-checkpointed at identical shape, for the HFU/MFU gap
    assert hp.Config(2048, 8, False) in hp.SWEEP and hp.Config(2048, 8, True) in hp.SWEEP
