"""Adversarial tests for external run verification receipts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

import pytest

from traffic_replay import run_verification as verification_module
from traffic_replay.artifacts import strict_json_dumps
from traffic_replay.cli import main
from traffic_replay.metrics import render_html, render_markdown, summarize
from traffic_replay.run_verification import (
    _validate_preflight_gate_consistency,
    create_run_verification_receipt,
    verify_run_output,
    verify_run_receipt,
)


def test_preflight_pass_cannot_be_bound_to_non_200_response_rows(tmp_path):
    gate = {
        "skipped": False, "attempted": 2, "reachable": 2,
        "readable": 2, "reasoning_probe_requests": 0,
        "outcome": "preflight_passed", "force_requested": False,
        "gate_satisfied": True,
    }
    summary = {"run": {"preflight_gate": gate}}
    start = {"preflight_gate": gate}
    evidence = {
        "phases": {"preflight": 2},
        "preflight_rows_judged": 2,
        "preflight_acceptable_outcomes": 0,
        # Both adversarial rows can claim visible, complete answers, but HTTP
        # 503 is never a reachable/passing production preflight response.
        "preflight_http_200": 0,
    }

    with pytest.raises(ValueError, match="preflight gate counts disagree"):
        _validate_preflight_gate_consistency(
            tmp_path, summary, start, evidence)


@pytest.fixture(autouse=True)
def _clean_external_verifier_source(monkeypatch):
    monkeypatch.setattr(
        verification_module,
        "snapshot_source_state",
        lambda _path: _source_state(),
    )


def _source_state(*, dirty=False, commit="a" * 40, tree="b" * 64):
    return {
        "captured_at_unix": 1_800_000_000.0,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status_sha256": "c" * 64,
        "source_tree_sha256": tree,
        "source_files": [{
            "path": "runner.py", "sha256": "d" * 64, "bytes": 12,
        }],
    }


def _request_row() -> dict:
    return {
        "phase": "replay",
        "request_id": "r0",
        "global_index": 0,
        "ok": True,
        "status": 200,
        "visible_content_seen": True,
        "reasoning_seen": False,
        "valid_tool_calls": 0,
        "stream_complete": True,
        "parse_errors": 0,
        "request_attempts": 1,
        "first_send_unix": 1_800_000_001.0,
        "t_send_unix": 1_800_000_001.0,
        "finished_unix": 1_800_000_001.2,
        "t_completed_unix": 1_800_000_001.2,
        "scheduled_s": 0.0,
        "queue_wait_ms": 0.0,
        "dispatch_lag_ms": 0.0,
        "ttfb_ms": 50.0,
        "ttft_ms": 100.0,
        "e2e_ms": 200.0,
        "interchunk_max_ms": 20.0,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cached_tokens": 0,
        "cached_tokens_source": "test",
        "intended_input_tokens": 100,
        "intended_output_tokens": 20,
        "intended_cache_fraction": 0.0,
        "finish_reason": "stop",
        "retries": 0,
        "response_model": "fixture-model",
        "response_object": "chat.completion.chunk",
        "response_id_sha256": "e" * 64,
        "system_fingerprint": "fixture-fingerprint",
    }


def _summary(row=None) -> dict:
    row = _request_row() if row is None else row
    summary = summarize(
        [row],
        schedule_meta={
            "requests": 1, "seconds": 1, "source": "unit test",
            "rate_min": 1.0, "rate_p50": 1.0,
            "rate_p95": 1.0, "rate_max": 1.0,
        },
        run_meta={
            "title": "verified fixture",
            "input_mode": "profile",
            "endpoint_path": "/serving-endpoints/fixture/invocations",
            "endpoint_model": "fixture-model",
            "artifact_id": "artifact-fixture",
            "aggregation_valid": True,
            "transport": {
                "connection_policy_id":
                    "fresh_http1_per_physical_attempt",
                "production_connection_policy_declared":
                    "fresh_http1_per_physical_attempt",
                "production_connection_policy_match": True,
                "production_connection_policy_assurance":
                    "operator asserted an exact production policy match",
                "production_comparability_warning": None,
            },
        },
        rate_limit_results=[row],
    )
    # Report rendering needs a complete current summary. Tail adequacy is not
    # the subject of this one-row artifact fixture, so isolate it from the
    # receipt lifecycle assertions below.
    summary["sample"] = {
        "n": 1_000,
        "supports": ["p50", "p90", "p95", "p99"],
        "indicative_only": [],
        "warning": None,
    }
    summary["drift"] = {"drift_kind": "stable"}
    summary["rate_limits"] = {
        "binding": {"binding_complete": True},
        "configured": {},
        "comparisons": {},
        "external_usage_warning": (
            "No external usage is included in this fixture."),
        "warning": None,
    }
    return summary


def _json_bytes(value: object) -> bytes:
    return (strict_json_dumps(value, indent=2) + "\n").encode("utf-8")


def _file_metadata(raw: bytes, *, rows: int | None = None) -> dict:
    value = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if rows is not None:
        value["row_count"] = rows
    return value


def _seal_run(base: Path, *, source=None, summary=None, row=None) -> Path:
    d = base
    d.mkdir()
    source = _source_state() if source is None else source
    row = _request_row() if row is None else row
    summary = _summary(row) if summary is None else summary
    start = {
        "run_started_at_unix": 1_800_000_000.0,
        "source": source,
        "effective_config": {"title": "verified fixture"},
    }
    files = {
        "requests.jsonl": (strict_json_dumps(row) + "\n").encode("utf-8"),
        "summary.json": _json_bytes(summary),
        "report.md": b"# Manifest-bound report\n",
        "report.html": b"<!doctype html><title>Manifest-bound report</title>\n",
        "start.json": _json_bytes(start),
    }
    for name, raw in files.items():
        (d / name).write_bytes(raw)
    timestamps = struct.pack("<d", 0.0)
    indices = struct.pack("<q", 0)
    manifest = {
        "manifest_schema_version": 3,
        "artifact_created_at_utc": "2027-01-15T08:00:00+00:00",
        "run_id": "logical-fixture",
        "logical_run_id": "logical-fixture",
        "workload_id": "workload-fixture",
        "execution_id": "execution-fixture",
        "artifact_id": "artifact-fixture",
        "harness_version": "0.5.1",
        "git_commit": source.get("git_commit"),
        "git_dirty": source.get("git_dirty"),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "source": source,
        "shard": "1/1",
        "schedule": {"requests": 1, "seconds": 1, "shard": "1/1"},
        "schedule_identity": {
            "encoding": "float64-le-seconds-from-run-start",
            "global_timestamps_sha256": hashlib.sha256(timestamps).hexdigest(),
            "shard_timestamps_sha256": hashlib.sha256(timestamps).hexdigest(),
            "global_count": 1,
            "shard_count": 1,
            "global_min_s": 0.0,
            "global_max_s": 0.0,
            "shard_min_s": 0.0,
            "shard_max_s": 0.0,
        },
        "index_identity": {
            "encoding": "int64-le",
            "global_indices_sha256": hashlib.sha256(indices).hexdigest(),
            "count": 1,
            "global_count": 1,
            "shard_index": 0,
            "shard_total": 1,
            "partition": "unsharded",
            "min": 0,
            "max": 0,
        },
        "artifacts": {
            name: _file_metadata(
                raw, rows=1 if name == "requests.jsonl" else None)
            for name, raw in files.items()
        },
    }
    manifest_raw = _json_bytes(manifest)
    (d / "manifest.json").write_bytes(manifest_raw)
    completion = {
        "artifact_id": manifest["artifact_id"],
        "status": "complete",
        "completed_at_unix": 1_800_000_002.0,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_bytes": len(manifest_raw),
        "request_rows": 1,
    }
    (d / ".traffic-replay-complete").write_bytes(_json_bytes(completion))
    return d


def _tree_snapshot(d: Path) -> dict:
    snapshot = {}
    for path in sorted(d.rglob("*")):
        relative = path.relative_to(d).as_posix()
        info = path.lstat()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            raw = path.read_bytes()
            snapshot[relative] = (
                "file", info.st_mode, hashlib.sha256(raw).hexdigest(), len(raw))
        else:
            snapshot[relative] = ("directory", info.st_mode)
    return snapshot


def _reseal_manifest(d: Path) -> None:
    manifest = json.loads((d / "manifest.json").read_text())
    for name in manifest["artifacts"]:
        raw = (d / name).read_bytes()
        rows = raw.count(b"\n") if name == "requests.jsonl" else None
        manifest["artifacts"][name] = _file_metadata(raw, rows=rows)
    raw = _json_bytes(manifest)
    (d / "manifest.json").write_bytes(raw)
    completion = json.loads((d / ".traffic-replay-complete").read_text())
    completion["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    completion["manifest_bytes"] = len(raw)
    completion["request_rows"] = manifest["artifacts"][
        "requests.jsonl"]["row_count"]
    (d / ".traffic-replay-complete").write_bytes(_json_bytes(completion))


def _reseal_receipt(d: Path) -> None:
    manifest = json.loads((d / "manifest.json").read_text())
    for name in manifest["artifacts"]:
        manifest["artifacts"][name] = _file_metadata(
            (d / name).read_bytes())
    raw = _json_bytes(manifest)
    (d / "manifest.json").write_bytes(raw)
    completion = json.loads((d / ".traffic-replay-complete").read_text())
    completion["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    completion["manifest_bytes"] = len(raw)
    (d / ".traffic-replay-complete").write_bytes(_json_bytes(completion))


def test_receipt_is_external_self_sealed_and_source_is_unchanged(tmp_path):
    run = _seal_run(tmp_path / "run")
    before = _tree_snapshot(run)

    receipt = create_run_verification_receipt(run, tmp_path / "receipt")

    assert _tree_snapshot(run) == before
    assert receipt.parent == run.parent
    assert (receipt / ".traffic-replay-complete").is_file()
    assert not (receipt / ".traffic-replay-writing").exists()
    payload = verify_run_receipt(receipt)
    assert payload["verified"] is True
    assert payload["digital_signature"] is False
    assert "not a digital signature" in payload["assurance"]
    assert payload["source_run"]["manifest"]["sha256"] == hashlib.sha256(
        (run / "manifest.json").read_bytes()).hexdigest()
    assert payload["source_run"]["completion"]["sha256"] == hashlib.sha256(
        (run / ".traffic-replay-complete").read_bytes()).hexdigest()
    assert payload["source_run"]["summary"] == payload[
        "source_run"]["artifacts"]["summary.json"]
    assert payload["decision"]["evidence_integrity"]["code"] == "VERIFIED"
    assert payload["decision"]["endpoint_capacity"]["code"] == \
        "HELD_AT_TESTED_LOAD"
    manifest = json.loads((receipt / "manifest.json").read_text())
    for name in (
            "verification.json", "verified-report.md",
            "verified-report.html"):
        assert manifest["artifacts"][name] == _file_metadata(
            (receipt / name).read_bytes())
    source_manifest_sha = hashlib.sha256(
        (run / "manifest.json").read_bytes()).hexdigest()
    markdown = (receipt / "verified-report.md").read_text()
    html = (receipt / "verified-report.html").read_text()
    for rendered in (markdown, html):
        assert "EXTERNAL VERIFIED VIEW" in rendered
        assert "artifact-fixture" in rendered
        assert source_manifest_sha in rendered
        assert payload["verifier_version"] in rendered
        assert payload["created_at_utc"] in rendered
        assert "not a digital signature" in rendered
        assert "Source reproducibility" in rendered
        assert "Verifier reproducibility" in rendered
    assert "Integrity: **VERIFIED**" in markdown
    assert "Source reproducibility: **PASS**" in markdown
    assert "Verifier reproducibility: **PASS**" in markdown
    assert html.count("status-pass") >= 3
    assert "PRINT/PDF DERIVATIVE" in html
    assert "EXTERNAL VERIFIED VIEW" not in (run / "report.md").read_text()
    assert "EXTERNAL VERIFIED VIEW" not in (run / "report.html").read_text()


def test_default_report_renderers_remain_unverified_and_context_is_keyword_only():
    summary = _summary()

    markdown = render_markdown(summary, "fixture")
    html = render_html(summary, "fixture")

    for rendered in (markdown, html):
        assert "EXTERNAL VERIFIED VIEW" not in rendered
        assert "Verification required" in rendered
    assert "UNSEALED PRINT/PDF DERIVATIVE" in html
    with pytest.raises(TypeError):
        render_html(summary, "fixture", {})
    with pytest.raises(TypeError):
        render_markdown(summary, "fixture", {})


@pytest.mark.parametrize("field,value,reason_code", [
    ("git_dirty", True, "GIT_STATE_DIRTY_OR_UNKNOWN"),
    ("git_commit", "not-a-commit", "GIT_COMMIT_DIGEST_INVALID"),
    ("git_commit", "0" * 40, "GIT_COMMIT_DIGEST_INVALID"),
    ("source_tree_sha256", "short", "SOURCE_TREE_DIGEST_INVALID"),
    ("source_tree_sha256", "0" * 64, "SOURCE_TREE_DIGEST_INVALID"),
])
def test_unreconstructible_source_never_gets_held_capacity(
        tmp_path, field, value, reason_code):
    source = _source_state()
    source[field] = value
    run = _seal_run(tmp_path / "run", source=source)

    verified = verify_run_output(run)

    reconstructibility = verified["source_reconstructibility"]
    assert reconstructibility["reconstructible"] is False
    assert reason_code in reconstructibility["reason_codes"]
    capacity = verified["decision"]["endpoint_capacity"]
    assert capacity["code"] == "INCONCLUSIVE"
    assert "SOURCE_NOT_RECONSTRUCTIBLE" in capacity["reason_codes"]
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    payload = verify_run_receipt(receipt)
    assert payload["decision"]["endpoint_capacity"]["code"] == \
        "INCONCLUSIVE"
    markdown = (receipt / "verified-report.md").read_text()
    html = (receipt / "verified-report.html").read_text()
    assert "Source reproducibility: **FAILED**" in markdown
    assert reason_code.replace("_", r"\_") in markdown
    assert "repro-warning" in html
    assert "status-failed" in html
    assert reason_code in html


def test_dirty_external_verifier_never_issues_held_capacity(
        tmp_path, monkeypatch):
    run = _seal_run(tmp_path / "run")
    assert verify_run_output(run)["decision"]["endpoint_capacity"][
        "code"] == "HELD_AT_TESTED_LOAD"
    monkeypatch.setattr(
        verification_module,
        "snapshot_source_state",
        lambda _path: _source_state(dirty=True),
    )

    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    payload = verify_run_receipt(receipt)

    assert payload["source_reconstructibility"]["reconstructible"] is True
    assert payload["verifier_source_reconstructibility"][
        "reconstructible"] is False
    capacity = payload["decision"]["endpoint_capacity"]
    assert capacity["code"] == "INCONCLUSIVE"
    assert "VERIFIER_SOURCE_NOT_RECONSTRUCTIBLE" in capacity["reason_codes"]
    markdown = (receipt / "verified-report.md").read_text()
    html = (receipt / "verified-report.html").read_text()
    assert "Source reproducibility: **PASS**" in markdown
    assert "Verifier reproducibility: **FAILED**" in markdown
    assert r"VERIFIER\_GIT\_STATE\_DIRTY\_OR\_UNKNOWN" in markdown
    assert "repro-warning" in html
    assert "VERIFIER_GIT_STATE_DIRTY_OR_UNKNOWN" in html


def test_self_consistent_verifier_reproducibility_upgrade_is_rejected(
        tmp_path, monkeypatch):
    run = _seal_run(tmp_path / "run")
    monkeypatch.setattr(
        verification_module,
        "snapshot_source_state",
        lambda _path: _source_state(dirty=True),
    )
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    payload_path = receipt / "verification.json"
    payload = json.loads(payload_path.read_text())
    payload["verifier_source_reconstructibility"] = \
        verification_module._generator_reconstructibility(_source_state())
    payload_path.write_bytes(_json_bytes(payload))
    manifest = json.loads((receipt / "manifest.json").read_text())
    manifest["verifier_source_reconstructible"] = True
    (receipt / "manifest.json").write_bytes(_json_bytes(manifest))
    _reseal_receipt(receipt)

    with pytest.raises(
            ValueError,
            match="disagrees with recorded verifier source"):
        verify_run_receipt(receipt)


def test_renderer_rejects_held_capacity_with_failed_reproducibility(tmp_path):
    source = _source_state(dirty=True)
    run = _seal_run(tmp_path / "run", source=source)
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    payload = verify_run_receipt(receipt)
    context = verification_module._verified_report_context(payload)
    context["decision"] = verification_module.build_report_decision(
        _summary(),
        verification_module.IntegrityContext(
            "verified", "internal consistency fixture"),
    )

    with pytest.raises(ValueError, match="cannot claim held capacity"):
        render_html(_summary(), "fixture", verification_context=context)


@pytest.mark.parametrize("name", [
    "requests.jsonl", "summary.json", "report.md", "report.html",
    "start.json", "manifest.json", ".traffic-replay-complete",
])
def test_tamper_in_any_canonical_chain_file_is_rejected(tmp_path, name):
    run = _seal_run(tmp_path / "run")
    path = run / name
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="mismatch|invalid"):
        create_run_verification_receipt(run, tmp_path / "receipt")
    assert not (tmp_path / "receipt").exists()


@pytest.mark.parametrize("name", [
    "requests.jsonl", "summary.json", "report.md", "report.html", "start.json",
    "manifest.json", ".traffic-replay-complete",
])
def test_missing_canonical_artifact_is_rejected_before_receipt(tmp_path, name):
    run = _seal_run(tmp_path / "run")
    (run / name).unlink()

    with pytest.raises(ValueError, match="missing|cannot read"):
        create_run_verification_receipt(run, tmp_path / "receipt")
    assert not (tmp_path / "receipt").exists()


def test_source_directory_and_artifact_symlinks_are_rejected(tmp_path):
    run = _seal_run(tmp_path / "run")
    alias = tmp_path / "run-alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(ValueError, match="not a regular directory"):
        verify_run_output(alias)

    original = run / "report.md"
    saved = tmp_path / "saved-report.md"
    saved.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(saved)
    with pytest.raises(ValueError, match="not a regular file"):
        verify_run_output(run)


def test_completion_symlink_and_writing_marker_are_rejected(tmp_path):
    run = _seal_run(tmp_path / "run")
    completion = run / ".traffic-replay-complete"
    saved = tmp_path / "completion.json"
    saved.write_bytes(completion.read_bytes())
    completion.unlink()
    completion.symlink_to(saved)
    with pytest.raises(ValueError, match="not a regular file"):
        verify_run_output(run)

    completion.unlink()
    completion.write_bytes(saved.read_bytes())
    (run / ".traffic-replay-writing").write_text("still writing\n")
    with pytest.raises(ValueError, match="still being written"):
        create_run_verification_receipt(run, tmp_path / "receipt")
    assert not (tmp_path / "receipt").exists()


@pytest.mark.parametrize("name,bad", [
    ("summary.json", b'{"requests_total":1,"requests_total":2}\n'),
    ("start.json", b'{"source":{"git_dirty":NaN}}\n'),
    ("requests.jsonl", b'{"phase":"replay","phase":"probe"}\n'),
])
def test_manifest_bound_malformed_json_is_still_rejected(tmp_path, name, bad):
    run = _seal_run(tmp_path / "run")
    (run / name).write_bytes(bad)
    _reseal_manifest(run)

    with pytest.raises(ValueError, match="duplicate key|non-finite"):
        verify_run_output(run)


def test_source_identity_disagreement_blocks_held_capacity(tmp_path):
    run = _seal_run(tmp_path / "run")
    start = json.loads((run / "start.json").read_text())
    start["source"]["git_commit"] = "e" * 40
    (run / "start.json").write_bytes(_json_bytes(start))
    _reseal_manifest(run)

    result = verify_run_output(run)

    assert result["source_reconstructibility"]["reconstructible"] is False
    assert "SOURCE_IDENTITY_INCONSISTENT" in result[
        "source_reconstructibility"]["reason_codes"]
    assert result["decision"]["endpoint_capacity"]["code"] == "INCONCLUSIVE"


@pytest.mark.parametrize("case,match", [
    ("request_totals", "requests_ok disagrees"),
    ("answer_counts", "answers.acceptable_outcomes disagrees"),
    ("http_429_count", "http_429_count disagrees"),
    ("replay_phase", "index_identity SHA-256 disagrees|replay row count disagrees"),
    ("request_outcome", "requests_ok disagrees"),
    ("scheduled_time", "schedule_identity SHA-256 disagrees"),
    ("global_index", "index_identity SHA-256 disagrees"),
    ("request_id", "no valid replay request_id"),
])
def test_manifest_bound_summary_and_request_log_must_agree(
        tmp_path, case, match):
    run = _seal_run(tmp_path / "run")
    summary = json.loads((run / "summary.json").read_text())
    row = json.loads((run / "requests.jsonl").read_text())
    if case == "request_totals":
        summary["requests_ok"] = 0
        (run / "summary.json").write_bytes(_json_bytes(summary))
    elif case == "answer_counts":
        summary["answers"]["acceptable_outcomes"] = 0
        (run / "summary.json").write_bytes(_json_bytes(summary))
    elif case == "http_429_count":
        row["status"] = 429
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    elif case == "replay_phase":
        row["phase"] = "probe"
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    elif case == "request_outcome":
        row["ok"] = False
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    elif case == "scheduled_time":
        row["scheduled_s"] = 0.5
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    elif case == "global_index":
        row["global_index"] = 1
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    else:
        row["request_id"] = ""
        (run / "requests.jsonl").write_text(strict_json_dumps(row) + "\n")
    _reseal_manifest(run)

    with pytest.raises(ValueError, match=match):
        verify_run_output(run)


def test_external_verifier_rederives_visible_refusal_as_unacceptable(
        tmp_path):
    row = _request_row()
    row["refusal_seen"] = True
    summary = _summary(row)
    assert summary["answers"]["judged"] == 1
    assert summary["answers"]["acceptable_outcomes"] == 0

    verified = verify_run_output(
        _seal_run(tmp_path / "run", row=row, summary=summary))

    evidence = verified["binding"]["request_evidence"]
    assert evidence["answer_rows_judged"] == 1
    assert evidence["acceptable_outcomes"] == 0
    assert verified["summary"]["answers"]["acceptable_outcomes"] == 0


def test_external_verifier_rejects_non_boolean_refusal_evidence(tmp_path):
    canonical = _request_row()
    canonical["refusal_seen"] = True
    malformed = dict(canonical, refusal_seen=1)
    run = _seal_run(
        tmp_path / "run", row=malformed, summary=_summary(canonical))

    with pytest.raises(ValueError, match="invalid answer-outcome fields"):
        verify_run_output(run)


def test_collision_and_concurrent_receipts_are_unique(tmp_path):
    run = _seal_run(tmp_path / "run")
    requested = tmp_path / "receipt"
    requested.mkdir()
    sentinel = requested / "belongs-to-user.txt"
    sentinel.write_text("untouched")

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(
            lambda _index: create_run_verification_receipt(run, requested),
            range(8),
        ))

    assert len(set(receipts)) == 8
    assert all(path != requested for path in receipts)
    assert sentinel.read_text() == "untouched"
    assert all(verify_run_receipt(path)["verified"] for path in receipts)


def test_existing_output_symlink_is_never_followed(tmp_path):
    run = _seal_run(tmp_path / "run")
    target = tmp_path / "user-owned"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("untouched")
    requested = tmp_path / "receipt"
    requested.symlink_to(target, target_is_directory=True)

    receipt = create_run_verification_receipt(run, requested)

    assert receipt != requested
    assert receipt.parent == requested.parent
    assert sentinel.read_text() == "untouched"
    assert requested.is_symlink()
    assert verify_run_receipt(receipt)["verified"] is True


def test_source_change_between_receipt_passes_leaves_no_complete_receipt(
        tmp_path, monkeypatch):
    run = _seal_run(tmp_path / "run")
    before = _tree_snapshot(run)
    original = verification_module.verify_run_output
    calls = 0

    def mutate_before_second(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            report = run / "report.md"
            report.write_bytes(report.read_bytes() + b"concurrent change\n")
        return original(path)

    monkeypatch.setattr(
        verification_module, "verify_run_output", mutate_before_second)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_run_verification_receipt(run, tmp_path / "receipt")

    assert calls == 2
    assert (tmp_path / "receipt" / ".traffic-replay-writing").is_file()
    assert not (tmp_path / "receipt" / ".traffic-replay-complete").exists()
    assert _tree_snapshot(run) != before  # only the simulated external writer


def test_source_change_during_view_render_leaves_no_complete_receipt(
        tmp_path, monkeypatch):
    run = _seal_run(tmp_path / "run")
    original = verification_module.verify_run_output
    calls = 0

    def mutate_before_post_render_check(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            report = run / "report.html"
            report.write_bytes(report.read_bytes() + b"changed during render\n")
        return original(path)

    monkeypatch.setattr(
        verification_module, "verify_run_output",
        mutate_before_post_render_check)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_run_verification_receipt(run, tmp_path / "receipt")

    assert calls == 3
    assert (tmp_path / "receipt" / ".traffic-replay-writing").is_file()
    assert not (tmp_path / "receipt" / ".traffic-replay-complete").exists()


def test_write_failure_never_promotes_a_green_receipt_and_source_is_unchanged(
        tmp_path, monkeypatch):
    run = _seal_run(tmp_path / "run")
    before = _tree_snapshot(run)
    original = verification_module._atomic_text

    def fail_manifest(fd, name, value):
        if name == "manifest.json":
            raise OSError("simulated full disk")
        return original(fd, name, value)

    monkeypatch.setattr(verification_module, "_atomic_text", fail_manifest)
    with pytest.raises(OSError, match="simulated full disk"):
        create_run_verification_receipt(run, tmp_path / "receipt")

    assert _tree_snapshot(run) == before
    assert (tmp_path / "receipt" / ".traffic-replay-writing").is_file()
    assert not (tmp_path / "receipt" / ".traffic-replay-complete").exists()


def test_receipt_tamper_and_later_source_mutation_are_detected(tmp_path):
    run = _seal_run(tmp_path / "run")
    first = create_run_verification_receipt(run, tmp_path / "receipt-a")
    payload = first / "verification.json"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_run_receipt(first)

    second = create_run_verification_receipt(run, tmp_path / "receipt-b")
    (run / "report.html").write_bytes(
        (run / "report.html").read_bytes() + b"later mutation")
    with pytest.raises(ValueError, match="SHA-256 mismatch|no longer matches"):
        verify_run_receipt(second)


@pytest.mark.parametrize("name", [
    "verification.json", "verified-report.md", "verified-report.html",
])
def test_tamper_in_any_receipt_artifact_is_rejected(tmp_path, name):
    run = _seal_run(tmp_path / "run")
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    path = receipt / name
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_run_receipt(receipt)


@pytest.mark.parametrize("name", [
    ".traffic-replay-complete", "manifest.json", "verification.json",
    "verified-report.md", "verified-report.html",
])
def test_missing_receipt_chain_file_is_rejected(tmp_path, name):
    run = _seal_run(tmp_path / "run")
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    (receipt / name).unlink()

    with pytest.raises(ValueError, match="missing"):
        verify_run_receipt(receipt)


def test_receipt_artifact_symlink_is_rejected(tmp_path):
    run = _seal_run(tmp_path / "run")
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    report = receipt / "verified-report.html"
    saved = tmp_path / "saved-verified-report.html"
    saved.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(saved)

    with pytest.raises(ValueError, match="not a regular file"):
        verify_run_receipt(receipt)


@pytest.mark.parametrize("name", ["verified-report.md", "verified-report.html"])
def test_self_consistent_noncanonical_verified_view_is_rejected(
        tmp_path, name):
    run = _seal_run(tmp_path / "run")
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    path = receipt / name
    path.write_bytes(path.read_bytes() + b"self-consistent tamper")
    _reseal_receipt(receipt)

    with pytest.raises(ValueError, match="not the canonical external"):
        verify_run_receipt(receipt)


def test_explicit_source_override_allows_portable_receipt(tmp_path):
    run = _seal_run(tmp_path / "run")
    receipt = create_run_verification_receipt(run, tmp_path / "receipt")
    moved_copy = tmp_path / "copied-run"
    shutil.copytree(run, moved_copy)

    payload = verify_run_receipt(receipt, source_run=moved_copy)

    assert payload["verified"] is True
    assert "source_run_path" not in payload["source_run"]
    assert payload["source_locator"] == {
        "kind": "sibling_directory", "directory_name": "run",
    }


def test_receipt_output_must_be_a_true_sibling(tmp_path):
    run = _seal_run(tmp_path / "run")
    elsewhere = tmp_path / "receipts" / "receipt"
    with pytest.raises(ValueError, match="must be a sibling"):
        create_run_verification_receipt(run, elsewhere)
    assert not elsewhere.parent.exists()


def test_output_inside_source_is_rejected_without_mutating_source(tmp_path):
    run = _seal_run(tmp_path / "run")
    before = _tree_snapshot(run)
    with pytest.raises(ValueError, match="outside the immutable source run"):
        create_run_verification_receipt(run, run / "receipt")
    assert _tree_snapshot(run) == before


def test_cli_success_and_failure_contract(tmp_path, capsys):
    run = _seal_run(tmp_path / "run")
    code = main([
        "verify-run", str(run), "--out", str(tmp_path / "receipt"),
        "--format", "json",
    ])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["verified"] is True
    assert output["digital_signature"] is False
    assert Path(output["receipt_dir"]).is_dir()

    (run / "summary.json").write_text("{}\n")
    code = main([
        "verify-run", str(run), "--out", str(tmp_path / "failed"),
        "--format", "json",
    ])
    output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert output["verified"] is False
    assert not (tmp_path / "failed").exists()
