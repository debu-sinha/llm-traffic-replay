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


def test_schema_v1_draws_remain_bitwise_compatible_with_legacy_sampler():
    n, seed = 2000, 123
    draw = prof.sample(SPEC, n, seed=seed)
    rng = np.random.default_rng(seed)
    mu_i, sigma_i = prof.lognormal_from_quantiles(**SPEC.input_tokens)
    mu_o, sigma_o = prof.lognormal_from_quantiles(**SPEC.output_tokens)
    mu_c, sigma_c = prof.logitnormal_from_quantiles(**SPEC.cache_fraction)
    expected_input = np.clip(
        rng.lognormal(mu_i, sigma_i, n).round(), 1, 200_000).astype(int)
    expected_output = np.clip(
        rng.lognormal(mu_o, sigma_o, n).round(), 1, 8_192).astype(int)
    latent = np.clip(rng.normal(mu_c, sigma_c, n), -709.0, 709.0)
    expected_cache = 1.0 / (1.0 + np.exp(-latent))
    expected_prefix = np.round(expected_input * expected_cache).astype(int)
    assert SPEC.schema_version == 1
    assert SPEC.sampling is None
    assert np.array_equal(draw["input_tokens"], expected_input)
    assert np.array_equal(draw["output_tokens"], expected_output)
    assert np.array_equal(draw["cache_target_fraction"], expected_cache)
    assert np.array_equal(draw["prefix_tokens"], expected_prefix)


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


def test_profile_json_duplicate_keys_are_rejected_at_every_depth(tmp_path):
    p = tmp_path / "duplicate.json"
    p.write_text(
        '{"name":"duplicate","input_tokens":{"p50":5,"p50":6,'
        '"p95":10},"output_tokens":{"p50":5,"p95":10},'
        '"cache_fraction":{"p50":0,"p95":0}}')
    with pytest.raises(ValueError, match="duplicate key 'p50'"):
        prof.Profile.from_json(p)


@pytest.mark.parametrize("field,value", [
    ("input_tokens", {"p50": True, "p95": 10}),
    ("output_tokens", {"p50": "5", "p95": 10}),
    ("cache_fraction", {"p50": False, "p95": 0.5}),
])
def test_profile_quantiles_require_real_json_numbers(field, value):
    kwargs = {
        "name": "strict",
        "input_tokens": {"p50": 5, "p95": 10},
        "output_tokens": {"p50": 5, "p95": 10},
        "cache_fraction": {"p50": 0.1, "p95": 0.5},
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="must be numbers"):
        prof.Profile(**kwargs)


def test_profile_embedded_acceptance_policy_is_validated():
    with pytest.raises(ValueError, match="p101"):
        prof.Profile(
            name="bad-policy",
            input_tokens={"p50": 5, "p95": 10},
            output_tokens={"p50": 5, "p95": 10},
            cache_fraction={"p50": 0.1, "p95": 0.5},
            extra={"acceptance_targets": {"ttft_ms": {"p101": 10}}},
        )


@pytest.mark.parametrize("kwargs", [
    {"n": True}, {"seed": True}, {"seed": -1},
    {"min_input": 1.5}, {"max_output": True},
])
def test_sampler_integer_controls_are_strict(kwargs):
    base = {"n": 10}
    base.update(kwargs)
    with pytest.raises(ValueError):
        prof.sample(SPEC, **base)


def test_empty_draw_has_no_quantiles_to_report():
    with pytest.raises(ValueError, match="empty draw"):
        prof.quantile_report(prof.sample(SPEC, 0))


def test_every_shipped_profile_loads_and_samples():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs").glob("profile_*.json")):
        profile = prof.Profile.from_json(path)
        draw = prof.sample(profile, 10, seed=1)
        assert len(draw["input_tokens"]) == 10, path


def _cdf_profile(**sampling_updates):
    sampling = {
        "mode": "quantile_cdf",
        "probabilities": [0.1, 0.5, 0.9, 0.95, 0.99],
        "input_tokens": [25, 100, 400, 800, 1600],
        "output_tokens": [4, 10, 40, 80, 160],
        "cache_fraction": [0.0, 0.2, 0.6, 0.8, 1.0],
    }
    sampling.update(sampling_updates)
    return prof.Profile(
        schema_version=2,
        name="cdf",
        input_tokens={"p50": 100, "p95": 800},
        output_tokens={"p50": 10, "p95": 80},
        cache_fraction={"p50": 0.2, "p95": 0.8},
        sampling=sampling,
    )


def _joint_profile(rows=None):
    if rows is None:
        rows = [
            {"input_tokens": 100, "output_tokens": 10,
             "cache_fraction": 0.0, "weight": 1},
            {"input_tokens": 1000, "output_tokens": 100,
             "cache_fraction": 1.0, "weight": 3},
        ]
    return prof.Profile(
        schema_version=2,
        name="joint",
        input_tokens={"p50": 1000, "p95": 1000},
        output_tokens={"p50": 100, "p95": 100},
        cache_fraction={"p50": 1.0, "p95": 1.0},
        sampling={"mode": "empirical_joint", "rows": rows},
    )


def test_quantile_cdf_exact_knots_log_interpolation_and_clamped_tails():
    probabilities = np.asarray([0.1, 0.5, 0.9])
    values = np.asarray([25.0, 100.0, 400.0])
    ranks = np.asarray([0.0, 0.1, 0.5, 0.7, 0.9, 1.0])
    recovered = prof._interpolate_quantile_cdf(
        probabilities, values, ranks, logarithmic=True)
    assert np.array_equal(recovered[[0, 1, 2, 4, 5]],
                          [25.0, 25.0, 100.0, 400.0, 400.0])
    assert recovered[3] == pytest.approx(200.0)


def test_quantile_cdf_recovers_every_ladder_knot_without_invented_dependence():
    draw = prof.sample(_cdf_profile(), 300_000, seed=718)
    expected = {
        "input_tokens": (100, 400, 800, 1600),
        "output_tokens": (10, 40, 80, 160),
        "cache_target_fraction": (0.2, 0.6, 0.8, 1.0),
    }
    for field_name, knots in expected.items():
        actual = np.percentile(draw[field_name], [50, 90, 95, 99])
        assert np.allclose(actual, knots, rtol=0.025, atol=0.01), field_name
    assert draw["params"]["dependence"] == "independent_marginals"
    assert draw["params"]["rank_sampling"] == \
        "independently_shuffled_stratified"
    assert draw["params"]["tail_policy"] == "clamp_to_end_knots"
    assert abs(np.corrcoef(
        draw["input_tokens"], draw["output_tokens"])[0, 1]) < 0.02


def test_quantile_cdf_stratification_bounds_finite_run_knot_drift():
    draw = prof.sample(_cdf_profile(), 1000, seed=912)
    for field_name, expected in (
        ("input_tokens", [100, 400, 800, 1600]),
        ("output_tokens", [10, 40, 80, 160]),
        ("cache_target_fraction", [0.2, 0.6, 0.8, 1.0]),
    ):
        actual = np.percentile(draw[field_name], [50, 90, 95, 99])
        assert np.allclose(actual, expected, rtol=0.015, atol=0.01), field_name


def test_bundled_blended_profile_samples_its_authoritative_full_ladder():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    profile = prof.Profile.from_json(
        root / "configs" / "profile_agent_blended.json")
    assert profile.schema_version == 2
    assert profile.sampling == {
        "mode": "quantile_cdf",
        "probabilities": [0.5, 0.9, 0.95, 0.99],
        "input_tokens": [10000.0, 13000.0, 24000.0, 25000.0],
        "output_tokens": [40.0, 70.0, 90.0, 165.0],
        "cache_fraction": [0.6, 0.75, 0.87, 0.98],
    }
    draw = prof.sample(profile, 300_000, seed=81)
    for field_name, expected in (
        ("input_tokens", [10000, 13000, 24000, 25000]),
        ("output_tokens", [40, 70, 90, 165]),
        ("cache_target_fraction", [0.6, 0.75, 0.87, 0.98]),
    ):
        assert np.allclose(
            np.percentile(draw[field_name], [50, 90, 95, 99]),
            expected, rtol=0.025, atol=0.01), field_name
    assert list(prof.quantile_report(draw)["input_tokens"]) == [
        "p50", "p90", "p95", "p99"]


def test_quantile_cdf_requires_exact_legacy_anchor_knots():
    with pytest.raises(ValueError, match="exactly match"):
        _cdf_profile(input_tokens=[25, 101, 400, 800, 1600])


@pytest.mark.parametrize("updates,match", [
    ({"unexpected": 1}, "unknown key"),
    ({"mode": "spline"}, "mode"),
    ({"probabilities": [0.1, 0.5, 0.95, 0.9, 0.99]}, "increasing"),
    ({"probabilities": [0.1, 0.5, 0.9, 0.99, 1.0]}, "between 0 and 1"),
    ({"probabilities": [0.1, 0.5, 0.9, 0.94, 0.99]}, "0.95"),
    ({"input_tokens": [25, 100, 99, 800, 1600]}, "nondecreasing"),
    ({"output_tokens": [4, 10, True, 80, 160]}, "must be a number"),
    ({"cache_fraction": [0, 0.2, 0.6, 0.8, 1.1]}, "between 0 and 1"),
])
def test_quantile_cdf_schema_is_closed_and_strict(updates, match):
    with pytest.raises(ValueError, match=match):
        _cdf_profile(**updates)


def test_empirical_joint_exact_frequencies_correlation_and_combinations():
    draw = prof.sample(_joint_profile(), 400, seed=919)
    triples = list(zip(
        draw["input_tokens"], draw["output_tokens"],
        draw["cache_target_fraction"]))
    assert set(triples) == {(100, 10, 0.0), (1000, 100, 1.0)}
    assert triples.count((100, 10, 0.0)) == 100
    assert triples.count((1000, 100, 1.0)) == 300
    assert np.corrcoef(
        draw["input_tokens"], draw["output_tokens"])[0, 1] == 1.0
    assert draw["params"]["dependence"] == "observed_joint_triples"
    assert draw["params"]["sampling"] == "balanced_weighted_cycles"


def test_empirical_joint_partial_cycle_drift_is_bounded_by_one_observation():
    draw = prof.sample(_joint_profile(), 403, seed=61)
    small = int(np.sum(draw["input_tokens"] == 100))
    assert abs(small - 403 / 4) < 1


def test_empirical_joint_fixed_seed_is_deterministic_and_order_canonical():
    rows = [
        {"input_tokens": 100, "output_tokens": 10,
         "cache_fraction": 0.0, "weight": 1},
        {"input_tokens": 1000, "output_tokens": 100,
         "cache_fraction": 1.0, "weight": 3},
    ]
    first = prof.sample(_joint_profile(rows), 41, seed=77)
    second = prof.sample(_joint_profile(list(reversed(rows))), 41, seed=77)
    assert np.array_equal(first["input_tokens"], second["input_tokens"])
    assert np.array_equal(first["output_tokens"], second["output_tokens"])
    assert np.array_equal(
        first["cache_target_fraction"], second["cache_target_fraction"])


@pytest.mark.parametrize("rows,match", [
    ([{"input_tokens": 100, "output_tokens": 10,
       "cache_fraction": 0.0, "weight": 1, "raw_prompt": "secret"}],
     "unknown key"),
    ([{"input_tokens": 100.0, "output_tokens": 10,
       "cache_fraction": 0.0, "weight": 1}], "must be an integer"),
    ([{"input_tokens": 100, "output_tokens": 10,
       "cache_fraction": 0.0, "weight": 0}], "positive integer"),
    ([{"input_tokens": 100, "output_tokens": 10,
       "cache_fraction": 0.0, "weight": True}], "must be an integer"),
    ([{"input_tokens": 100, "output_tokens": 10,
       "cache_fraction": -0.1, "weight": 1}], "between 0 and 1"),
])
def test_empirical_joint_rows_are_content_free_and_strict(rows, match):
    with pytest.raises(ValueError, match=match):
        _joint_profile(rows)


def test_empirical_joint_duplicate_triples_must_be_combined_as_weights():
    row = {"input_tokens": 1000, "output_tokens": 100,
           "cache_fraction": 1.0, "weight": 2}
    with pytest.raises(ValueError, match="duplicates"):
        _joint_profile([row, dict(row)])


def test_empirical_joint_anchors_cannot_drift_from_weighted_rows():
    with pytest.raises(ValueError, match="inverted-CDF anchors"):
        prof.Profile(
            schema_version=2, name="drift",
            input_tokens={"p50": 100, "p95": 1000},
            output_tokens={"p50": 100, "p95": 100},
            cache_fraction={"p50": 1.0, "p95": 1.0},
            sampling=_joint_profile().sampling,
        )


def test_schema_versions_and_sampling_integer_bounds_are_strict():
    with pytest.raises(ValueError, match="schema_version"):
        prof.Profile(
            schema_version=True, name="bad",
            input_tokens={"p50": 1, "p95": 1},
            output_tokens={"p50": 1, "p95": 1},
            cache_fraction={"p50": 0, "p95": 0})
    rows = [{"input_tokens": 2 ** 63, "output_tokens": 1,
             "cache_fraction": 0, "weight": 1}]
    with pytest.raises(ValueError, match="signed 64-bit"):
        _joint_profile(rows)
