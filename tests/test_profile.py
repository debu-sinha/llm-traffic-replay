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
        prof.lognormal_from_quantiles(100, 99)
    with pytest.raises(ValueError):
        prof.logitnormal_from_quantiles(0.9, 0.6)
    with pytest.raises(ValueError):
        prof.logitnormal_from_quantiles(0.5, 1.2)


def test_constant_token_and_zero_cache_profiles_are_legitimate():
    p = prof.Profile(
        name="constant", input_tokens={"p50": 512, "p95": 512},
        output_tokens={"p50": 32, "p95": 32},
        cache_fraction={"p50": 0.0, "p95": 0.0})
    d = prof.sample(p, 100, seed=4)
    assert (d["input_tokens"] == 512).all()
    assert (d["output_tokens"] == 32).all()
    assert (d["cache_target_fraction"] == 0.0).all()
    assert (d["prefix_tokens"] == 0).all()


def test_constant_full_cache_profile_is_supported():
    p = prof.Profile(
        name="all-cache", input_tokens={"p50": 128, "p95": 128},
        output_tokens={"p50": 8, "p95": 8},
        cache_fraction={"p50": 1.0, "p95": 1.0})
    d = prof.sample(p, 10)
    assert (d["cache_target_fraction"] == 1.0).all()
    assert np.array_equal(d["prefix_tokens"], d["input_tokens"])


def test_nonconstant_boundary_cache_distribution_recovers_quantiles():
    p = prof.Profile(name="boundary", input_tokens={"p50": 10, "p95": 20},
                     output_tokens={"p50": 1, "p95": 2},
                     cache_fraction={"p50": 0.0, "p95": 0.5})
    d = prof.sample(p, 60_000, seed=8)
    assert np.percentile(d["cache_target_fraction"], 50) < 0.01
    assert abs(np.percentile(d["cache_target_fraction"], 95) - 0.5) < 0.02
    assert d["params"]["cache_family"] == "clipped_normal"


def test_clipping_respected():
    d = prof.sample(SPEC, 20_000, seed=7, min_input=256, max_input=30_000)
    assert d["input_tokens"].min() >= 256
    assert d["input_tokens"].max() <= 30_000


def test_profile_schema_is_validated_when_loaded(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"name":"missing-shape"}')
    with pytest.raises(ValueError, match="missing required"):
        prof.Profile.from_json(p)
