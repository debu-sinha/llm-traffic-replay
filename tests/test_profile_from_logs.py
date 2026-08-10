"""Real-log profile extraction preserves measured boundaries and fails loud."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.profile_from_logs as profile_from_logs
from scripts.profile_from_logs import (
    _InputLimits, _profile_from_path, build_profile, main)
from traffic_replay.profile import Profile, sample


def _build(records, fraction_field=None):
    return build_profile(
        records, "real", "input_tokens", "output_tokens", "cached_tokens",
        fraction_field)


def _build_from_path(path, *, fraction_field=None, mode="quantiles",
                     limits=None):
    return _profile_from_path(
        path, "real", "input_tokens", "output_tokens", "cached_tokens",
        fraction_field, mode=mode, limits=limits or _InputLimits())


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


@pytest.mark.parametrize("value", [
    9_007_199_254_740_992,
    "9007199254740993",
    "1e999999999",
])
def test_token_counts_that_cannot_roundtrip_exactly_are_rejected(value):
    records = [{"input_tokens": value, "output_tokens": 20,
                "cached_tokens": 0}]
    with pytest.raises(ValueError, match="exact token-count limit"):
        _build(records)


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
        _build_from_path(path)


def test_jsonl_records_must_be_objects(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="must be an object"):
        _build_from_path(path)


def test_jsonl_duplicate_keys_are_rejected_with_location(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"input_tokens":1,"input_tokens":2}\n')
    with pytest.raises(ValueError, match=r"logs\.jsonl:1.*duplicate key"):
        _build_from_path(path)


@pytest.mark.parametrize("content,match", [
    (b'{"input_tokens":NaN}\n', "non-finite"),
    (b'{"input_tokens":1e999}\n', "non-finite"),
    (b'{"input_tokens":"\xff"}\n', "not UTF-8"),
])
def test_jsonl_uses_strict_json_for_numeric_and_encoding_ambiguity(
        tmp_path, content, match):
    path = tmp_path / "logs.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        _build_from_path(path)


def test_jsonl_duplicate_secret_key_is_not_echoed_in_error(tmp_path):
    secret = "Bearer " + "dapi" + ("x" * 40)
    path = tmp_path / "logs.jsonl"
    path.write_text(json.dumps({secret: 1})[:-1] + f',"{secret}":2}}\n')
    with pytest.raises(ValueError) as caught:
        _build_from_path(path)
    assert secret not in str(caught.value)
    assert "sha256=" in str(caught.value)


@pytest.mark.parametrize("content,match", [
    ("input_tokens,input_tokens,output_tokens\n1,2,3\n", "unique"),
    ("input_tokens,output_tokens\n1,2,3\n", "more values"),
])
def test_csv_ambiguous_columns_are_rejected(tmp_path, content, match):
    path = tmp_path / "logs.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        _build_from_path(path)


@pytest.mark.parametrize("content", [
    "",
    "input_tokens,output_tokens,cached_tokens\n",
])
def test_empty_or_header_only_csv_reports_no_records(tmp_path, content):
    path = tmp_path / "empty.csv"
    path.write_text(content)
    with pytest.raises(SystemExit, match="no records"):
        _build_from_path(path)


def test_csv_invalid_utf8_reports_offset_without_payload(tmp_path):
    path = tmp_path / "logs.csv"
    path.write_bytes(
        b"input_tokens,output_tokens,cached_tokens,prompt\n"
        b'100,10,0,"private-\xff-value"\n')
    with pytest.raises(ValueError, match=r"not UTF-8 at byte offset") as caught:
        _build_from_path(path)
    assert "private-" not in str(caught.value)


def test_malformed_csv_error_does_not_echo_source_record(tmp_path):
    path = tmp_path / "logs.csv"
    secret = "customer-secret-that-must-not-appear"
    path.write_text(
        "input_tokens,output_tokens,cached_tokens,prompt\n"
        f'100,10,0,"{secret}"unexpected\n')
    with pytest.raises(ValueError, match="malformed CSV") as caught:
        _build_from_path(path)
    assert secret not in str(caught.value)


def test_multiline_unselected_csv_field_is_supported_and_discarded(tmp_path):
    path = tmp_path / "logs.csv"
    source = (
        "input_tokens,output_tokens,cached_tokens,prompt\n"
        '100,10,0,"private first line\nprivate second line"\n')
    path.write_text(source)
    raw = _build_from_path(path)
    assert raw["extraction"]["total_records"] == 1
    assert "private" not in json.dumps(raw)


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
    assert raw["source"]["bytes"] == len(source_bytes)
    assert digest in raw["provenance"]
    assert f"bytes: {len(source_bytes)}" in raw["provenance"]
    assert "never emit me" not in json.dumps(raw)


def test_cli_streams_without_whole_file_path_read(tmp_path, capsys,
                                                  monkeypatch):
    source = tmp_path / "streamed.jsonl"
    source_bytes = (
        b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
        b'{"input_tokens":200,"output_tokens":20,"cached_tokens":100}\n')
    source.write_bytes(source_bytes)

    def whole_file_read_is_forbidden(_path):
        raise AssertionError("whole-file read attempted")

    monkeypatch.setattr(Path, "read_bytes", whole_file_read_is_forbidden)
    assert main(["--input", str(source), "--name", "streamed"]) == 0

    raw = json.loads(capsys.readouterr().out)
    assert raw["source"] == {
        "digest_algorithm": "sha256",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "bytes": len(source_bytes),
    }


def test_cli_refuses_to_overwrite_input_with_output(tmp_path, capsys):
    source = tmp_path / "customer.jsonl"
    source_bytes = (
        b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0,'
        b'"prompt":"retain this source"}\n')
    source.write_bytes(source_bytes)
    with pytest.raises(SystemExit) as stopped:
        main(["--input", str(source), "--out", str(source)])
    assert stopped.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err
    assert source.read_bytes() == source_bytes


def test_cli_atomically_rejects_symlink_output(tmp_path, capsys):
    source = tmp_path / "customer.jsonl"
    source.write_text(
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n')
    target = tmp_path / "unrelated.txt"
    target.write_text("do not replace\n")
    output = tmp_path / "profile.json"
    output.symlink_to(target)
    with pytest.raises(SystemExit) as stopped:
        main(["--input", str(source), "--out", str(output)])
    assert stopped.value.code == 2
    assert "symbolic link" in capsys.readouterr().err
    assert target.read_text() == "do not replace\n"


def test_cli_output_is_valid_private_json(tmp_path, capsys):
    source = tmp_path / "customer.jsonl"
    source.write_text(
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n')
    output = tmp_path / "profile.json"
    assert main([
        "--input", str(source), "--out", str(output), "--name", "safe",
    ]) == 0
    assert json.loads(output.read_text())["name"] == "safe"
    assert stat.S_IMODE(output.stat().st_mode) & 0o077 == 0
    assert "wrote" in capsys.readouterr().err


def test_input_mutation_during_stream_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "changing.jsonl"
    first = b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
    source.write_bytes(first)
    strict_loads = profile_from_logs.loads_strict
    calls = 0

    def mutate_after_parse(raw):
        nonlocal calls
        value = strict_loads(raw)
        calls += 1
        if calls == 1:
            with source.open("ab") as handle:
                handle.write(
                    b'{"input_tokens":200,"output_tokens":20,'
                    b'"cached_tokens":0}\n')
        return value

    monkeypatch.setattr(profile_from_logs, "loads_strict", mutate_after_parse)
    with pytest.raises(ValueError, match="changed while it was read"):
        _build_from_path(source)


@pytest.mark.parametrize("kwargs,match", [
    ({"mode": "unknown"}, "mode must"),
    ({"mode": "empirical-joint", "source_sha256": "ABC"},
     "64 lowercase"),
    ({"source_sha256": "a" * 64, "source_byte_count": True},
     "non-negative integer"),
    ({"source_byte_count": 1}, "requires source_sha256"),
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


def test_streaming_parser_does_not_retain_complete_json_objects(
        tmp_path, capsys, monkeypatch):
    source = tmp_path / "many.jsonl"
    source.write_bytes(b"{}\n" * 2_000)
    live_refs = []
    peak_live = 0

    class TrackedRecord(dict):
        pass

    def tracked_loads(_raw):
        nonlocal peak_live
        live_refs[:] = [ref for ref in live_refs if ref() is not None]
        record = TrackedRecord(
            input_tokens=100, output_tokens=10, cached_tokens=0,
            prompt="customer payload must not persist")
        live_refs.append(weakref.ref(record))
        peak_live = max(peak_live, sum(ref() is not None for ref in live_refs))
        return record

    monkeypatch.setattr(profile_from_logs, "loads_strict", tracked_loads)
    assert main(["--input", str(source)]) == 0
    output = capsys.readouterr().out
    assert "customer payload" not in output
    assert peak_live == 1
    assert all(ref() is None for ref in live_refs)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "device"])
def test_input_must_be_a_nonsymlink_regular_file(tmp_path, kind):
    regular = tmp_path / "regular.jsonl"
    regular.write_text(
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n')
    if kind == "symlink":
        candidate = tmp_path / "linked.jsonl"
        candidate.symlink_to(regular)
        match = "symbolic link"
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("mkfifo is unavailable")
        candidate = tmp_path / "pipe.jsonl"
        os.mkfifo(candidate)
        match = "regular file"
    elif kind == "directory":
        candidate = tmp_path
        match = "regular file"
    else:
        candidate = Path("/dev/null")
        if not candidate.exists():
            pytest.skip("no portable device path")
        match = "regular file"
    with pytest.raises(ValueError, match=match):
        _build_from_path(candidate)


def test_exact_file_byte_limit_passes_and_one_byte_less_fails(tmp_path):
    source = tmp_path / "bounded.jsonl"
    raw = b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
    source.write_bytes(raw)
    exact = replace(_InputLimits(), max_bytes=len(raw))
    assert _build_from_path(source, limits=exact)["source"]["bytes"] == len(raw)
    too_small = replace(_InputLimits(), max_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match=r"--max-bytes"):
        _build_from_path(source, limits=too_small)


def test_physical_line_limit_is_bounded_before_json_parsing(tmp_path):
    source = tmp_path / "long.jsonl"
    secret = "private-prompt-" + ("x" * 300)
    raw = json.dumps({
        "input_tokens": 100, "output_tokens": 10, "cached_tokens": 0,
        "prompt": secret,
    }).encode() + b"\n"
    source.write_bytes(raw)
    limits = replace(_InputLimits(), max_line_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match=r"--max-line-bytes") as caught:
        _build_from_path(source, limits=limits)
    assert secret not in str(caught.value)


def test_logical_json_record_limit_is_explicit(tmp_path):
    source = tmp_path / "record.jsonl"
    raw = b'{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
    source.write_bytes(raw)
    limits = replace(_InputLimits(), max_record_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match=r"--max-record-bytes"):
        _build_from_path(source, limits=limits)


def test_physical_line_count_limit_includes_blank_lines(tmp_path):
    source = tmp_path / "lines.jsonl"
    source.write_text(
        "\n" +
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n')
    limits = replace(_InputLimits(), max_lines=1)
    with pytest.raises(ValueError, match=r"--max-lines"):
        _build_from_path(source, limits=limits)


def test_request_record_count_limit_excludes_blank_jsonl_lines(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n"
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
        "\n"
        '{"input_tokens":200,"output_tokens":20,"cached_tokens":0}\n')
    limits = replace(_InputLimits(), max_records=1)
    with pytest.raises(ValueError, match=r"--max-records"):
        _build_from_path(source, limits=limits)


def test_multiline_csv_logical_record_is_bounded_without_echoing_it(tmp_path):
    source = tmp_path / "multiline.csv"
    secret = "confidential-customer-prompt-" + ("x" * 200)
    source.write_text(
        "input_tokens,output_tokens,cached_tokens,prompt\n"
        f'100,10,0,"first line\n{secret}"\n')
    # The header fits, while the multiline logical data record does not.
    limits = replace(_InputLimits(), max_record_bytes=80)
    with pytest.raises(ValueError, match=r"--max-record-bytes") as caught:
        _build_from_path(source, limits=limits)
    assert secret not in str(caught.value)


def test_empirical_unique_triple_limit_bounds_retained_state(tmp_path):
    source = tmp_path / "unique.jsonl"
    source.write_text(
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n'
        '{"input_tokens":200,"output_tokens":20,"cached_tokens":0}\n')
    limits = replace(_InputLimits(), max_unique_triples=1)
    with pytest.raises(ValueError, match=r"--max-unique-triples"):
        _build_from_path(source, mode="empirical-joint", limits=limits)


def test_cli_limit_errors_are_concise_and_do_not_echo_source_fields(
        tmp_path, capsys):
    source = tmp_path / "private.jsonl"
    secret = "do-not-print-this-customer-prompt"
    source.write_text(json.dumps({
        "input_tokens": 100,
        "output_tokens": 10,
        "cached_tokens": 0,
        "prompt": secret,
    }) + "\n")
    with pytest.raises(SystemExit) as stopped:
        main([
            "--input", str(source),
            "--max-line-bytes", "10",
        ])
    assert stopped.value.code == 2
    stderr = capsys.readouterr().err
    assert "--max-line-bytes=10" in stderr
    assert secret not in stderr


@pytest.mark.parametrize("option", [
    "--max-bytes", "--max-line-bytes", "--max-record-bytes",
    "--max-lines", "--max-records", "--max-unique-triples",
])
@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_cli_limits_require_positive_integers(tmp_path, option, value):
    source = tmp_path / "valid.jsonl"
    source.write_text(
        '{"input_tokens":100,"output_tokens":10,"cached_tokens":0}\n')
    with pytest.raises(SystemExit) as stopped:
        main(["--input", str(source), option, value])
    assert stopped.value.code == 2


def test_csv_stream_keeps_only_selected_numeric_columns(tmp_path):
    source = tmp_path / "private.csv"
    source_bytes = (
        b"input_tokens,output_tokens,cached_tokens,prompt,authorization\n"
        b'100,10,0,"customer prompt one","Bearer private-one"\n'
        b'200,20,100,"customer prompt two","Bearer private-two"\n')
    source.write_bytes(source_bytes)
    raw = _build_from_path(source)
    serialized = json.dumps(raw)
    assert raw["input_tokens"] == {"p50": 150, "p95": 195}
    assert raw["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    for forbidden in (
            "customer prompt", "Bearer private", "authorization", "prompt"):
        assert forbidden not in serialized
