"""The sampler must recover the stated quantiles. This is the contract that
makes 'built to the stated figures' a checkable claim instead of a vibe."""
import numpy as np
import pytest

from traffic_replay import profile as prof

SPEC = prof.Profile(
    name="t", provenance="test",
    input_tokens={"p50": 10_000, "p95": 24_000},
    output_tokens={"p50": 40, "p95": 90},
    cache_fraction={"p50": 0.60, "p95": 0.87},
)


def test_quantile_recovery_within_2pct():
    d = prof.sample(SPEC, 60_000, seed=3)
    r = prof.quantile_report(d)
    assert abs(r["input_tokens"]["p50"] / 10_000 - 1) < 0.02
    assert abs(r["input_tokens"]["p95"] / 24_000 - 1) < 0.02
    assert abs(r["output_tokens"]["p50"] / 40 - 1) < 0.05
    assert abs(r["cache_fraction"]["p50"] - 0.60) < 0.01
    assert abs(r["cache_fraction"]["p95"] - 0.87) < 0.01


def test_prefix_plus_suffix_equals_input():
    d = prof.sample(SPEC, 5_000, seed=5)
    assert (d["prefix_tokens"] + d["suffix_tokens"] == d["input_tokens"]).all()
    assert (d["prefix_tokens"] >= 0).all()
    assert (d["suffix_tokens"] >= 0).all()


def test_reproducible_by_seed():
    a = prof.sample(SPEC, 1_000, seed=11)
    b = prof.sample(SPEC, 1_000, seed=11)
    assert np.array_equal(a["input_tokens"], b["input_tokens"])
    assert np.array_equal(a["cache_target_fraction"], b["cache_target_fraction"])


def test_bad_quantiles_rejected():
    with pytest.raises(ValueError):
        prof.lognormal_from_quantiles(100, 100)
    with pytest.raises(ValueError):
        prof.logitnormal_from_quantiles(0.9, 0.6)
    with pytest.raises(ValueError):
        prof.logitnormal_from_quantiles(0.5, 1.2)


def test_clipping_respected():
    d = prof.sample(SPEC, 20_000, seed=7, min_input=256, max_input=30_000)
    assert d["input_tokens"].min() >= 256
    assert d["input_tokens"].max() <= 30_000
