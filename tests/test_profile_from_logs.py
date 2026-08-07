"""Real-log profile extraction preserves measured boundaries and fails loud."""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from scripts.profile_from_logs import (_load_records, build_profile, main)
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


def test_jsonl_duplicate_keys_are_rejected_with_location(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"input_tokens":1,"input_tokens":2}\n')
    with pytest.raises(ValueError, match=r"logs\.jsonl:1.*duplicate key"):
        _load_records(path)


@pytest.mark.parametrize("content,match", [
    ("input_tokens,input_tokens,output_tokens\n1,2,3\n", "unique"),
    ("input_tokens,output_tokens\n1,2,3\n", "more values"),
])
def test_csv_ambiguous_columns_are_rejected(tmp_path, content, match):
    path = tmp_path / "logs.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        _load_records(path)


def test_legacy_extraction_explicitly_counts_every_incomplete_signal():
    records = [
        {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 50},
        {"input_tokens": 200, "output_tokens": 30},
        {"input_tokens": 300, "cached_tokens": 0},
        {"output_tokens": 40, "cached_tokens": 0},
    ]
    raw = _build(records)
    assert raw["extraction"] == {
        "total_records": 4,
        "usable_input_records": 3,
        "dropped_input_records": 1,
        "usable_output_records": 3,
        "dropped_output_records": 1,
        "usable_cache_records": 2,
        "dropped_cache_records": 2,
        "complete_joint_records": 1,
        "dropped_incomplete_joint_records": 3,
    }


def test_empirical_joint_deduplicates_only_content_free_complete_triples():
    records = [
        {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 0,
         "prompt": "customer secret alpha", "trace_id": "arbitrary-a"},
        {"input_tokens": 100, "output_tokens": 10, "cached_tokens": 0,
         "prompt": "customer secret beta", "trace_id": "arbitrary-b"},
        {"input_tokens": 1000, "output_tokens": 100,
         "cached_tokens": 1000, "messages": [{"content": "do not copy"}]},
        {"input_tokens": 99, "cached_tokens": 0,
         "prompt": "incomplete secret"},
    ]
    digest = "a" * 64
    raw = build_profile(
        records, "joint", "input_tokens", "output_tokens", "cached_tokens",
        None, mode="empirical-joint", source_sha256=digest)
    assert raw["schema_version"] == 2
    assert raw["sampling"] == {
        "mode": "empirical_joint",
        "rows": [
            {"input_tokens": 100, "output_tokens": 10,
             "cache_fraction": 0.0, "weight": 2},
            {"input_tokens": 1000, "output_tokens": 100,
             "cache_fraction": 1.0, "weight": 1},
        ],
    }
    assert raw["extraction"] == {
        "total_records": 4,
        "complete_joint_records": 3,
        "dropped_incomplete_joint_records": 1,
        "records_missing_input": 0,
        "records_missing_output": 1,
        "records_missing_cache": 0,
        "unique_joint_rows": 2,
    }
    assert raw["source"] == {
        "digest_algorithm": "sha256", "sha256": digest}
    serialized = json.dumps(raw)
    for forbidden in (
            "customer secret", "incomplete secret", "trace_id", "messages"):
        assert forbidden not in serialized

    profile = Profile(
        schema_version=raw["schema_version"], name=raw["name"],
        input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"],
        cache_fraction=raw["cache_fraction"], sampling=raw["sampling"],
        extra={"extraction": raw["extraction"], "source": raw["source"]})
    draw = sample(profile, 30, seed=9)
    triples = set(zip(
        draw["input_tokens"], draw["output_tokens"],
        draw["cache_target_fraction"]))
    assert triples == {(100, 10, 0.0), (1000, 100, 1.0)}


def test_empirical_joint_cli_hashes_exact_source_bytes(tmp_path, capsys):
    source = tmp_path / "logs.jsonl"
    source_bytes = (
        b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0,'
        b'"prompt":"never emit me"}\n')
    source.write_bytes(source_bytes)
    assert main([
        "--input", str(source), "--name", "joint",
        "--mode", "empirical-joint",
    ]) == 0
    raw = json.loads(capsys.readouterr().out)
    digest = hashlib.sha256(source_bytes).hexdigest()
    assert raw["source"]["sha256"] == digest
    assert digest in raw["provenance"]
    assert "never emit me" not in json.dumps(raw)


@pytest.mark.parametrize("kwargs,match", [
    ({"mode": "unknown"}, "mode must"),
    ({"mode": "empirical-joint", "source_sha256": "ABC"},
     "64 lowercase"),
])
def test_profile_extractor_controls_are_strict(kwargs, match):
    records = [{"input_tokens": 100, "output_tokens": 10,
                "cached_tokens": 0}]
    with pytest.raises(ValueError, match=match):
        build_profile(
            records, "joint", "input_tokens", "output_tokens",
            "cached_tokens", None, **kwargs)


def test_empirical_joint_rejects_zero_output_rows_instead_of_emitting_them():
    records = [{"input_tokens": 100, "output_tokens": 0,
                "cached_tokens": 0}]
    with pytest.raises(ValueError, match="positive"):
        build_profile(
            records, "joint", "input_tokens", "output_tokens",
            "cached_tokens", None, mode="empirical-joint")
