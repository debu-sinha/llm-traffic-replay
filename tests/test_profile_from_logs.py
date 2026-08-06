"""Real-log profile extraction preserves measured boundaries and fails loud."""
from __future__ import annotations

import math

import pytest

from scripts.profile_from_logs import (_load_records, build_profile)
from traffic_replay.profile import Profile, sample


def _build(records, fraction_field=None):
    return build_profile(
        records, "real", "input_tokens", "output_tokens", "cached_tokens",
        fraction_field)


def test_constant_zero_cache_data_is_not_artificially_perturbed():
    records = [
        {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 0}
        for _ in range(20)
    ]
    raw = _build(records)
    assert raw["input_tokens"] == {"p50": 100, "p95": 100}
    assert raw["output_tokens"] == {"p50": 20, "p95": 20}
    assert raw["cache_fraction"] == {"p50": 0.0, "p95": 0.0}
    profile = Profile(
        name=raw["name"], input_tokens=raw["input_tokens"],
        output_tokens=raw["output_tokens"],
        cache_fraction=raw["cache_fraction"])
    draw = sample(profile, 10)
    assert set(draw["input_tokens"]) == {100}
    assert set(draw["cache_target_fraction"]) == {0.0}


def test_full_cache_boundary_is_preserved():
    records = [
        {"input_tokens": 100, "output_tokens": 20, "cache_fraction": 1.0}
        for _ in range(10)
    ]
    raw = _build(records, "cache_fraction")
    assert raw["cache_fraction"] == {"p50": 1.0, "p95": 1.0}


@pytest.mark.parametrize("records,match", [
    ([{"input_tokens": 100, "output_tokens": 20,
       "cached_tokens": 101}], "cannot exceed"),
    ([{"input_tokens": 100, "output_tokens": 20,
       "cached_tokens": -1}], "non-negative"),
    ([{"input_tokens": 100, "output_tokens": 20,
       "cache_fraction": 1.1}], "between 0 and 1"),
    ([{"input_tokens": math.nan, "output_tokens": 20,
       "cached_tokens": 0}], "finite"),
    ([{"input_tokens": 100.5, "output_tokens": 20,
       "cached_tokens": 0}], "integer count"),
])
def test_invalid_log_numbers_are_rejected_not_clipped(records, match):
    fraction = "cache_fraction" if "cache_fraction" in records[0] else None
    with pytest.raises(ValueError, match=match):
        _build(records, fraction)


def test_zero_output_median_cannot_be_sold_as_a_generation_profile():
    records = [
        {"input_tokens": 100, "output_tokens": 0, "cached_tokens": 0}
        for _ in range(10)
    ]
    with pytest.raises(ValueError, match="one or more tokens"):
        _build(records)


def test_custom_input_field_still_requires_positive_token_counts():
    records = [{"prompt_tokens": 0, "output_tokens": 20,
                "cached_tokens": 0}]
    with pytest.raises(ValueError, match="positive"):
        build_profile(
            records, "real", "prompt_tokens", "output_tokens",
            "cached_tokens", None)


def test_jsonl_errors_include_filename_and_line(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"input_tokens": 1}\n{bad}\n')
    with pytest.raises(ValueError, match=r"logs\.jsonl:2"):
        _load_records(path)


def test_jsonl_records_must_be_objects(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="must be an object"):
        _load_records(path)
