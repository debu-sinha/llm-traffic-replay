"""Production evidence lifecycle, identity, and redaction regressions."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from traffic_replay import artifacts as artifact_module
from traffic_replay.artifacts import (
    COMPLETE_MARKER,
    PARTIAL_REQUESTS,
    WRITING_MARKER,
    ArtifactError,
    RunArtifacts,
    redact_secrets,
    sanitize_display_text,
    sanitize_title,
    strict_json_dumps,
)
from traffic_replay.client import RequestResult
from traffic_replay.metrics import render_markdown, summarize, write_outputs
from traffic_replay.runner import RunConfig, run


def _profile(path: Path, *, acceptance=None) -> bytes:
    extra = {}
    if acceptance is not None:
        extra["acceptance_targets"] = acceptance
    raw = (json.dumps({
        "name": "integrity-shape",
        "input_tokens": {"p50": 12, "p95": 20},
        "output_tokens": {"p50": 4, "p95": 6},
        "cache_fraction": {"p50": 0.0, "p95": 0.0},
        **extra,
    }, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _schedule(n=3):
    return {
        "rates": np.asarray([float(n)]),
        "counts": np.asarray([n]),
        "timestamps": np.zeros(n, dtype=float),
    }


def _config(base: Path, profile: Path | None = None, **overrides) -> RunConfig:
    if profile is None:
        profile = base / "profile.json"
        _profile(profile)
    values = {
        "endpoint": {
            "base_url": "http://example.invalid",
            "path": "/invocations",
            "auth_token_env": "TRAFFIC_REPLAY_TEST_TOKEN_DO_NOT_SET",
        },
        "profile_path": str(profile),
        "duration_s": 1,
        "qps_base": 3.0,
        "qps_burst": 3.0,
        "qps_min": 3.0,
        "qps_max": 3.0,
        "calibrate_n": 0,
        "max_concurrency": 1,
        "capture_endpoint_metadata": False,
        "measure_network_path": False,
        "out_dir": str(base / "results"),
    }
    values.update(overrides)
    return RunConfig(**values)


class _DeterministicClient:
    seen_messages: list[list[dict]] = []
    scheduled_targets: list[float | None] = []

    def __init__(self, *args, **kwargs):
        pass

    def send(self, messages, max_tokens, request_id, scheduled_s,
             dispatch_lag_ms, intended, chars_sent, *,
             scheduled_monotonic=None):
        type(self).seen_messages.append(json.loads(json.dumps(messages)))
        type(self).scheduled_targets.append(scheduled_monotonic)
        now = time.time()
        return RequestResult(
            request_id=request_id,
            scheduled_s=scheduled_s,
            dispatch_lag_ms=dispatch_lag_ms,
            t_send_unix=now,
            first_send_unix=now,
            ttfb_ms=1.0,
            ttft_ms=2.0,
            ttfr_ms=None,
            ttfv_ms=2.0,
            e2e_ms=3.0,
            status=200,
            ok=True,
            error=None,
            content_chunks=1,
            interchunk_max_ms=None,
            finish_reason="stop",
            prompt_tokens=max(intended[0], 1),
            completion_tokens=1,
            cached_tokens=0,
            cached_tokens_source="test",
            intended_input_tokens=intended[0],
            intended_output_tokens=intended[1],
            intended_cache_fraction=intended[2],
            doc_id=intended[3],
            chars_sent=chars_sent,
            stream_complete=True,
            visible_content_seen=True,
            max_tokens_requested=max_tokens,
            queue_wait_ms=0.5,
            caller_ttfb_ms=1.5,
            caller_ttft_ms=2.5,
            caller_ttfv_ms=2.5,
            caller_e2e_ms=3.5,
        )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (path / "requests.jsonl").read_text().splitlines()]


def test_repeated_runs_separate_execution_ids_but_reproduce_bodies(
        tmp_path, monkeypatch):
    monkeypatch.setattr("traffic_replay.runner.EndpointClient",
                        _DeterministicClient)
    monkeypatch.setattr("traffic_replay.runner.make_schedule",
                        lambda **kwargs: _schedule(3))
    _DeterministicClient.seen_messages = []
    _DeterministicClient.scheduled_targets = []
    cfg = _config(tmp_path)

    first = run(cfg, quiet=True)
    second = run(cfg, quiet=True)
    out1, out2 = Path(first["out_dir"]), Path(second["out_dir"])
    m1 = json.loads((out1 / "manifest.json").read_text())
    m2 = json.loads((out2 / "manifest.json").read_text())

    assert out1 != out2
    assert m1["manifest_schema_version"] == 3
    assert m1["workload_id"] == m2["workload_id"]
    for field in ("logical_run_id", "execution_id", "artifact_id"):
        assert m1[field] != m2[field]
    replay1 = sorted((r for r in _rows(out1) if r["phase"] == "replay"),
                     key=lambda r: r["global_index"])
    replay2 = sorted((r for r in _rows(out2) if r["phase"] == "replay"),
                     key=lambda r: r["global_index"])
    assert [r["request_id"] for r in replay1] != [
        r["request_id"] for r in replay2]
    assert [r["body_request_id"] for r in replay1] == [
        r["body_request_id"] for r in replay2]
    assert [r["request_body_sha256"] for r in replay1] == [
        r["request_body_sha256"] for r in replay2]
    assert all(value is not None
               for value in _DeterministicClient.scheduled_targets)


def test_sealed_evidence_does_not_persist_absolute_local_paths(
        tmp_path, monkeypatch):
    private_dir = tmp_path / "customer-local-directory"
    private_dir.mkdir()
    profile = private_dir / "profile.json"
    _profile(profile)
    monkeypatch.setattr("traffic_replay.runner.EndpointClient",
                        _DeterministicClient)
    monkeypatch.setattr("traffic_replay.runner.make_schedule",
                        lambda **kwargs: _schedule(2))

    result = run(_config(
        private_dir, profile=profile,
        out_dir=str(private_dir / "private-results")), quiet=True)
    out = Path(result["out_dir"])
    evidence = "\n".join(
        (out / name).read_text()
        for name in ("start.json", "summary.json", "manifest.json"))
    assert str(tmp_path) not in evidence

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"]["profile"]["name"] == "profile.json"
    assert manifest["profile_path"] == "profile.json"
    assert manifest["effective_config"]["out_dir"] == "private-results"


def test_workload_uses_private_prompt_snapshot_when_original_mutates(
        tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompts.jsonl"
    original = (b'{"prompt":"original zero"}\n'
                b'{"prompt":"original one"}\n')
    prompt_path.write_bytes(original)

    class MutatingClient(_DeterministicClient):
        seen_messages = []
        scheduled_targets = []
        mutated = False

        def send(self, *args, **kwargs):
            if not type(self).mutated:
                prompt_path.write_text('{"prompt":"CHANGED"}\n')
                type(self).mutated = True
            return super().send(*args, **kwargs)

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", MutatingClient)
    monkeypatch.setattr("traffic_replay.runner.make_schedule",
                        lambda **kwargs: _schedule(4))
    cfg = _config(
        tmp_path, profile=None, profile_path=None,
        prompts_file=str(prompt_path), max_pending_requests=10)
    result = run(cfg, quiet=True)
    manifest = json.loads(
        (Path(result["out_dir"]) / "manifest.json").read_text())

    assert manifest["inputs"]["prompts"]["sha256"] == \
        hashlib.sha256(original).hexdigest()
    assert manifest["inputs"]["prompts"]["bytes"] == len(original)
    observed = [m[0]["content"] for m in MutatingClient.seen_messages]
    assert observed == ["original zero", "original one",
                        "original zero", "original one"]


def test_unusable_output_destination_fails_before_auth_or_client(
        tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    cfg = _config(tmp_path, out_dir=str(blocker))
    called = {"token": 0, "client": 0}

    def token(*args, **kwargs):
        called["token"] += 1
        return None

    class Client:
        def __init__(self, *args, **kwargs):
            called["client"] += 1

    monkeypatch.setattr("traffic_replay.runner._token", token)
    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    with pytest.raises(OSError):
        run(cfg, quiet=True)
    assert called == {"token": 0, "client": 0}


def test_killed_process_leaves_parseable_incremental_journal(tmp_path):
    target = tmp_path / "killed"
    code = "\n".join((
        "import sys,time",
        "from traffic_replay.artifacts import RunArtifacts",
        "a=RunArtifacts.claim(sys.argv[1], {'case':'kill'}, sync_every_rows=1)",
        "i=0",
        "while True:",
        " a.append({'sequence':i,'phase':'replay','ok':True})",
        " i+=1",
        " time.sleep(0.01)",
    ))
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(target)],
        cwd=Path(__file__).parents[1], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 5
        partial = target / PARTIAL_REQUESTS
        while time.time() < deadline:
            if partial.exists() and partial.read_bytes().count(b"\n") >= 5:
                break
            time.sleep(0.01)
        else:
            pytest.fail("subprocess did not persist request rows")
        proc.kill()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    raw_lines = partial.read_bytes().splitlines(keepends=True)
    assert len(raw_lines) >= 5
    assert all(line.endswith(b"\n") for line in raw_lines)
    recovered = [json.loads(line) for line in raw_lines]
    assert [r["sequence"] for r in recovered] == list(range(len(recovered)))
    assert (target / WRITING_MARKER).exists()
    assert (target / "start.json").exists()
    assert not (target / COMPLETE_MARKER).exists()
    assert not (target / "manifest.json").exists()


def test_artifact_claim_refuses_symlink_leaf(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        RunArtifacts.claim(link, {"case": "symlink"})
    assert not list(target.iterdir())


def test_new_artifact_directory_entry_is_fsynced_before_child_files(
        tmp_path, monkeypatch):
    target = tmp_path / "durable-claim"
    original = artifact_module._fsync_directory_path
    observed = []

    def inspect_parent(path):
        observed.append(Path(path))
        if len(observed) == 1:
            assert target.is_dir()
            assert list(target.iterdir()) == []
        return original(path)

    monkeypatch.setattr(
        artifact_module, "_fsync_directory_path", inspect_parent)
    artifacts = RunArtifacts.claim(target, {"case": "parent-fsync"})
    try:
        assert observed[0] == tmp_path
        assert (target / WRITING_MARKER).is_file()
        assert (target / "start.json").is_file()
    finally:
        artifacts.abort()


def test_artifact_claim_supports_symlinked_parent_but_refuses_leaf_alias(
        tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "parent-alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    target = alias_parent / "run"

    artifacts = RunArtifacts.claim(target, {"case": "parent-alias"})
    try:
        assert artifacts.path == target
        assert target.resolve().parent == real_parent.resolve()
        assert (target / WRITING_MARKER).is_file()
    finally:
        artifacts.abort()


def test_parent_fsync_failure_removes_new_claim_directory(
        tmp_path, monkeypatch):
    target = tmp_path / "parent-fsync-failure"
    calls = []

    def fail_parent_fsync(path):
        calls.append(Path(path))
        raise ArtifactError("forced parent fsync failure")

    monkeypatch.setattr(
        artifact_module, "_fsync_directory_path", fail_parent_fsync)
    with pytest.raises(ArtifactError, match="forced parent fsync failure"):
        RunArtifacts.claim(target, {"case": "parent-fsync-failure"})

    assert calls
    assert not target.exists()


def test_claim_initialization_failure_cleans_only_a_new_directory(
        tmp_path, monkeypatch):
    new_target = tmp_path / "new-initialization-failure"

    def fail_start(_self, name, _value):
        if name == "start.json":
            raise OSError("forced start persistence failure")

    monkeypatch.setattr(RunArtifacts, "_atomic_json", fail_start)
    with pytest.raises(OSError, match="forced start persistence failure"):
        RunArtifacts.claim(new_target, {"case": "new-failure"})
    assert not new_target.exists()

    existing_target = tmp_path / "caller-owned-empty-directory"
    existing_target.mkdir()
    with pytest.raises(OSError, match="forced start persistence failure"):
        RunArtifacts.claim(existing_target, {"case": "existing-failure"})
    assert existing_target.is_dir()
    assert list(existing_target.iterdir()) == []


def test_redaction_covers_credentials_without_hiding_token_controls():
    pat = "dapi" + "0123456789" + "supersecret"
    def encode(raw):
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    jwt = ".".join((encode(b'{"alg":"HS256"}'),
                    encode(b'{"sub":"synthetic-test"}'),
                    encode(b"synthetic-signature")))
    value = {
        "headers": [f"Authorization: Bearer {pat}",
                    "Authorization: Basic dXNlcjpwYXNz"],
        "client_assertion": jwt,
        "endpoint": ("https://user:password@example.test/invoke?"
                     "sv=1&sig=azure-secret&max_tokens=64"),
        "min_tokens": 8,
        "max_tokens": 64,
        "output_token_limit": 128,
        "api_token": "opaque-api-value",
        "service_token": "opaque-service-value",
        "custom_headers": {
            "Content-Type": "application/json",
            "X-Custom-Auth": "opaque-header-value",
            "X-Numeric": 123,
        },
        "auth_profile": "customer-workspace-profile",
        "note": "basic benchmark methodology",
    }
    safe = redact_secrets(value)
    persisted = strict_json_dumps(safe)
    for secret in (pat, jwt, "dXNlcjpwYXNz", "azure-secret", "password",
                   "opaque-api-value", "opaque-service-value",
                   "opaque-header-value", "application/json"):
        assert secret not in persisted
    assert safe["min_tokens"] == 8
    assert safe["max_tokens"] == 64
    assert safe["output_token_limit"] == 128
    assert safe["api_token"] == "<redacted>"
    assert safe["service_token"] == "<redacted>"
    assert set(safe["custom_headers"].values()) == {"<redacted>"}
    assert safe["auth_profile"] == "<redacted>"
    assert safe["note"] == "basic benchmark methodology"
    title = sanitize_title(f"report\nAuthorization: Bearer {pat}")
    assert "\n" not in title and pat not in title


def test_display_text_removes_direction_spoofing_and_controls():
    hostile = "trusted\u202eLIAF\u2066\nnext\x00value"

    assert sanitize_display_text(hostile) == "trustedLIAF next value"
    assert sanitize_title(hostile) == "trustedLIAF next value"


def test_strict_json_rejects_nonfinite_numbers():
    with pytest.raises(ValueError):
        strict_json_dumps({"latency_ms": float("nan")})


def test_durable_request_journal_rejects_duplicate_keys(tmp_path):
    artifacts = RunArtifacts.claim(
        tmp_path / "duplicate-journal", {"status": "starting"})
    try:
        (artifacts.path / PARTIAL_REQUESTS).write_text(
            '{"request_id":"first","request_id":"second"}\n')
        with pytest.raises(
                ArtifactError,
                match=(r"invalid durable JSON row 1 .*requests\.jsonl\.partial: "
                       r"JSON contains duplicate key 'request_id'")):
            list(artifacts.read_rows())
    finally:
        artifacts.abort()


@pytest.mark.parametrize("row,diagnostic", [
    ('{"latency_ms":NaN}\n', "non-finite number"),
    ('{"latency_ms":1e999}\n', "non-finite number"),
    ('{"value":' + "[" * 10_000 + "0" + "]" * 10_000 + '}\n',
     "safe nesting depth"),
])
def test_durable_request_journal_reports_safe_strict_json_reason(
        tmp_path, row, diagnostic):
    artifacts = RunArtifacts.claim(
        tmp_path / f"strict-journal-{hashlib.sha256(row.encode()).hexdigest()[:8]}",
        {"status": "starting"})
    try:
        (artifacts.path / PARTIAL_REQUESTS).write_text(row)
        with pytest.raises(ArtifactError) as caught:
            list(artifacts.read_rows())
        message = str(caught.value)
        assert "row 1" in message
        assert str(artifacts.path / PARTIAL_REQUESTS) in message
        assert diagnostic in message
    finally:
        artifacts.abort()


def test_durable_duplicate_key_diagnostic_does_not_echo_private_key(
        tmp_path):
    artifacts = RunArtifacts.claim(
        tmp_path / "private-key-journal", {"status": "starting"})
    # Keep the source payload free of a contiguous credential-shaped literal;
    # the notebook packer must reject those even when they appear in tests.
    private_key = "Bearer " + "dapi0123456789" + "-private-customer-material"
    encoded_key = json.dumps(private_key)
    try:
        (artifacts.path / PARTIAL_REQUESTS).write_text(
            f'{{{encoded_key}:1,{encoded_key}:2}}\n')
        with pytest.raises(ArtifactError) as caught:
            list(artifacts.read_rows())
        message = str(caught.value)
        assert private_key not in message
        assert "duplicate key <redacted; bytes=" in message
        assert "sha256=" in message
    finally:
        artifacts.abort()


def test_manifest_binds_every_final_artifact_and_detects_tamper(tmp_path):
    row = {
        "phase": "replay", "ok": True, "ttft_ms": 1.0,
        "e2e_ms": 2.0, "t_send_unix": 10.0,
        "prompt_tokens": 2, "completion_tokens": 1,
    }
    summary = summarize([row], run_meta={"title": "integrity"})
    out = write_outputs([row], summary, tmp_path / "bound", "integrity")
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        raw = (out / name).read_bytes()
        assert expected["bytes"] == len(raw)
        assert expected["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["artifacts"]["requests.jsonl"]["row_count"] == 1
    complete = json.loads((out / COMPLETE_MARKER).read_text())
    manifest_raw = (out / "manifest.json").read_bytes()
    assert complete["manifest_sha256"] == \
        hashlib.sha256(manifest_raw).hexdigest()
    assert not list(out.glob("*.tmp"))

    (out / "summary.json").write_text("{}\n")
    assert hashlib.sha256((out / "summary.json").read_bytes()).hexdigest() \
        != manifest["artifacts"]["summary.json"]["sha256"]


@pytest.mark.parametrize("override,match", [
    ({"acceptance_targets": {"ttft_ms": {"p101": 1}}},
     "unknown field"),
    ({"pricing": {"mode": "per_token", "input_dbu_per_m": 1}},
     "missing required"),
    ({"capture_endpoint_metadata": 1}, "must be boolean"),
    ({"measure_network_path": "false"}, "must be boolean"),
    ({"endpoint": {"base_url": "https://example.test/path",
                   "path": "/invoke"}}, "must be an origin"),
    ({"endpoint": {"base_url": "https://example.test",
                   "path": "/invoke", "unknown": True}},
     "invalid endpoint configuration"),
])
def test_run_config_delegates_policy_and_endpoint_validation(
        tmp_path, override, match):
    profile = tmp_path / "profile.json"
    _profile(profile)
    values = {
        "endpoint": {"base_url": "https://example.test", "path": "/invoke"},
        "profile_path": str(profile),
    }
    values.update(override)
    with pytest.raises(ValueError, match=match):
        RunConfig(**values)


def test_invalid_profile_policy_fails_before_auth_or_endpoint(
        tmp_path, monkeypatch):
    profile = tmp_path / "bad-profile.json"
    _profile(profile, acceptance={"ttft_ms": {"p101": 1}})
    cfg = _config(tmp_path, profile=profile)
    called = {"token": 0, "client": 0}

    def token(*args, **kwargs):
        called["token"] += 1

    class Client:
        def __init__(self, *args, **kwargs):
            called["client"] += 1

    monkeypatch.setattr("traffic_replay.runner._token", token)
    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    with pytest.raises(ValueError, match="profile.acceptance_targets"):
        run(cfg, quiet=True)
    assert called == {"token": 0, "client": 0}
    incomplete = list((tmp_path / "results").glob("*/failure.json"))
    # Endpoint-free input validation now precedes artifact creation. An
    # invalid profile must leave no run-shaped directory that could be
    # mistaken for a started benchmark.
    assert incomplete == []


def test_valid_tool_call_only_stream_is_an_acceptable_timed_outcome():
    row = {
        "ok": True,
        "status": 200,
        "visible_content_seen": False,
        "valid_tool_calls": 1,
        "tool_call_seen": True,
        "stream_complete": True,
        "parse_errors": 0,
        "ttft_ms": None,
        "ttf_tool_call_ms": 42.0,
        "e2e_ms": 60.0,
        "t_send_unix": 100.0,
        "first_send_unix": 100.0,
    }
    summary = summarize([row], acceptance={"success_rate": 0.99})
    answers = summary["answers"]
    assert answers["answered"] == 1
    assert answers["tool_call_only_outcomes"] == 1
    assert answers["no_acceptable_outcome"] == 0
    assert answers["answer_rate"] == 1.0
    assert summary["ttf_tool_call_ms"]["p50"] == 42.0
    assert summary["e2e_ms"]["n"] == 1
    assert summary["sla"]["success_rate"]["met"] is True
    assert "valid tool call" in render_markdown(summary, "tool")
