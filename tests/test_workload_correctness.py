"""Regression coverage for workload identity and control-plane safety."""
from __future__ import annotations

import hashlib
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from traffic_replay.client import EndpointConfig, RequestResult
from traffic_replay.runner import (
    RunConfig, _PreparedWorkload, _payload_hash, _representative_plans,
    _resolved_run_id, _stable_request_id, prevalidate_run_inputs, run,
)


PROFILE = "configs/profile_validation_small.json"


def _endpoint():
    return {"base_url": "http://example.invalid", "path": "/invocations",
            "auth_token_env": "UNUSED"}


def _cfg(tmp_path: Path, **overrides) -> RunConfig:
    values = dict(
        endpoint=_endpoint(), profile_path=PROFILE,
        duration_s=1, qps_base=4.0, qps_burst=4.0,
        qps_min=4.0, qps_max=4.0, calibrate_n=0,
        max_concurrency=2, max_output_tokens_cap=24,
        measure_network_path=False, capture_endpoint_metadata=False,
        out_dir=str(tmp_path), run_id="shared-run")
    values.update(overrides)
    return RunConfig(**values)


def _result(request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent):
    now = time.time()
    return RequestResult(
        request_id=request_id, scheduled_s=scheduled_s,
        dispatch_lag_ms=dispatch_lag_ms, t_send_unix=now,
        ttfb_ms=1.0, ttft_ms=1.0, ttfr_ms=None, ttfv_ms=1.0,
        e2e_ms=2.0, status=200, ok=True, error=None,
        content_chunks=1, interchunk_max_ms=None, finish_reason="stop",
        prompt_tokens=max(1, intended[0]), completion_tokens=1,
        cached_tokens=0, cached_tokens_source="test",
        intended_input_tokens=intended[0], intended_output_tokens=intended[1],
        intended_cache_fraction=intended[2], doc_id=intended[3],
        chars_sent=chars_sent, stream_complete=True,
        visible_content_seen=True, first_send_unix=now,
        max_tokens_requested=1)


def test_partial_calibration_keeps_original_cpt_and_is_disclosed(
        tmp_path, monkeypatch):
    class Client:
        calls = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, _messages, _max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent, **_kwargs):
            type(self).calls += 1
            row = _result(
                request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent)
            if type(self).calls == 1:
                row.ok = False
                row.stream_complete = False
                row.error = "incomplete calibration stream"
            return row

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    rc = _cfg(
        tmp_path / "partial-calibration", duration_s=2,
        qps_base=1.0, qps_burst=1.0, qps_min=1.0, qps_max=1.0,
        calibrate_n=2, max_concurrency=1)

    out = run(rc, quiet=True)
    start = json.loads((Path(out["out_dir"]) / "start.json").read_text())
    calibration = start["calibration"]
    assert calibration["eligible_clean_usage_requests"] == 1
    assert calibration["status"] == "incomplete_cpt_unchanged"
    assert calibration["cpt_final"] == calibration["cpt_initial"] == rc.cpt


def test_dispatch_lag_includes_submit_preparation_work(tmp_path, monkeypatch):
    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def send(self, _messages, _max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent, **_kwargs):
            return _result(
                request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent)

    original_plan = _PreparedWorkload.plan

    def slow_plan(self, global_index, request_id):
        time.sleep(0.03)
        return original_plan(self, global_index, request_id)

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", Client)
    monkeypatch.setattr(_PreparedWorkload, "plan", slow_plan)
    out = run(_cfg(
        tmp_path / "dispatch-preparation", duration_s=1,
        max_concurrency=1), quiet=True)
    rows = [json.loads(line) for line in (
        Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    replay = [row for row in rows if row.get("phase") == "replay"]

    assert replay
    assert min(row["dispatch_lag_ms"] for row in replay) >= 25.0


def test_preflight_profile_uses_concrete_p50_p95_shape_and_budgets(tmp_path):
    rc = _cfg(tmp_path, max_output_tokens_cap=20)
    plans = _representative_plans(rc)
    profile = json.loads(Path(PROFILE).read_text())
    assert [p["representative"] for p in plans] == ["p50", "p95"]
    assert [p["intended"][0] for p in plans] == [
        profile["input_tokens"]["p50"], profile["input_tokens"]["p95"]]
    assert [p["max_output"] for p in plans] == [12, 20]
    assert all(p["construction"]["error_chars"] == 0 for p in plans)


def test_preflight_prompt_mode_uses_real_prompts_and_configured_cap(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"prompt":"first real prompt"}\n{"prompt":"second real prompt"}\n')
    rc = _cfg(tmp_path, profile_path=None, prompts_file=str(prompt_file),
              max_output_tokens_cap=73)
    plans = _representative_plans(rc)
    assert [p["messages"][0]["content"] for p in plans] == [
        "first real prompt", "second real prompt"]
    assert [p["max_output"] for p in plans] == [73, 73]


def test_readable_preflight_gate_does_not_depend_on_reasoning_schema(
        tmp_path, monkeypatch):
    from traffic_replay.cli import _preflight

    class Empty200Client:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, messages, max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent):
            row = _result(request_id, scheduled_s, dispatch_lag_ms,
                          intended, chars_sent)
            row.ok = False
            row.error = "stream ended with no content delta"
            row.visible_content_seen = False
            row.ttfv_ms = None
            row.reasoning_seen = False
            row.reasoning_chunks = 0
            return row

    monkeypatch.setattr("traffic_replay.client.EndpointClient", Empty200Client)
    cfg = vars(_cfg(tmp_path)).copy()
    result = _preflight(cfg)
    assert result["reachable"] == 2
    assert result["readable"] == 0
    assert result["reasoning"] is False
    assert result["visible"] is False
    assert result["budget"] == max(result["budgets"])
    assert result["failed_probe_index"] == result["budgets"].index(
        max(result["budgets"]))


def test_preflight_rejects_mixed_visible_refusal_outcomes(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import _check_preflight, _preflight

    class MixedRefusalClient:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, messages, max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent):
            row = _result(request_id, scheduled_s, dispatch_lag_ms,
                          intended, chars_sent)
            # Some APIs include explanatory visible text alongside a
            # structured refusal marker. The refusal marker is authoritative.
            row.refusal_seen = True
            return row

    monkeypatch.setattr(
        "traffic_replay.client.EndpointClient", MixedRefusalClient)
    cfg = vars(_cfg(tmp_path)).copy()
    result = _preflight(cfg)

    assert result["reachable"] == result["attempted"] == 2
    assert result["visible"] is True
    assert result["readable"] == 0
    assert result["budget"] == max(result["budgets"])
    assert all(row["refusal_seen"] for row in result["_request_rows"])

    class Args:
        force = False
        probe_extra_body = []

    args = Args()
    assert _check_preflight(cfg, args) == 3
    assert args._preflight_evidence["readable"] == 0
    assert "non-refusal visible content" in capsys.readouterr().out


def test_preflight_probe_budget_is_largest_failure_not_largest_success(
        tmp_path, monkeypatch):
    from traffic_replay.cli import _preflight

    class MixedClient:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, messages, max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent):
            row = _result(request_id, scheduled_s, dispatch_lag_ms,
                          intended, chars_sent)
            if max_tokens < 20:
                row.ok = False
                row.error = "stream ended before a completed answer"
                row.visible_content_seen = False
                row.ttfv_ms = None
            return row

    monkeypatch.setattr("traffic_replay.client.EndpointClient", MixedClient)
    cfg = vars(_cfg(tmp_path, max_output_tokens_cap=20)).copy()
    result = _preflight(cfg)

    assert result["budgets"] == [12, 20]
    assert result["readable"] == 1
    assert result["failed_probe_index"] == 0
    assert result["budget"] == 12


def test_reasoning_probe_label_is_stable_and_describes_supplied_json():
    from traffic_replay.cli import _probe_label
    control = {"thinking": {"type": "disabled"}}
    assert _probe_label(control, 2) == "candidate 2 (thinking)"


def test_global_profile_bodies_are_identical_before_and_after_sharding(tmp_path):
    rc = _cfg(tmp_path, seed=41, run_id="one-logical-run")
    ecfg = EndpointConfig(**rc.endpoint)
    full = _PreparedWorkload(rc, 17)
    expected = {}
    for i in range(17):
        rid = _stable_request_id(_resolved_run_id(rc), i)
        plan = full.plan(i, rid)
        expected[i] = _payload_hash(ecfg, plan["messages"], plan["max_output"])

    observed = {}
    for shard_index in range(4):
        # A separate materializer/pool per process must still reproduce the
        # same globally indexed request body.
        worker = _PreparedWorkload(rc, 17)
        for i in range(shard_index, 17, 4):
            rid = _stable_request_id(_resolved_run_id(rc), i)
            plan = worker.plan(i, rid)
            observed[i] = _payload_hash(
                ecfg, plan["messages"], plan["max_output"])
    assert observed == expected


def test_prompt_indices_are_global_not_restarted_per_shard(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text("\n".join(
        json.dumps({"prompt": f"prompt-{i}"}) for i in range(5)) + "\n")
    rc = _cfg(tmp_path, profile_path=None, prompts_file=str(prompt_file))
    workload = _PreparedWorkload(rc, 13)
    per_shard = {}
    for shard_index in range(3):
        for global_index in range(shard_index, 13, 3):
            rid = _stable_request_id("shared", global_index)
            per_shard[global_index] = workload.plan(
                global_index, rid)["prompt_index"]
    assert per_shard == {i: i % 5 for i in range(13)}


def test_shards_reject_independent_unloaded_sizing(tmp_path):
    with pytest.raises(ValueError, match="cannot size independently"):
        _cfg(tmp_path, sizing_concurrency=2, shard_total=5, shard_index=0,
             run_id="shared", start_at_unix=time.time() + 60)


def test_timestamp_trace_rejects_unused_paid_sizing_pass(tmp_path):
    trace = tmp_path / "arrivals.txt"
    trace.write_text("0\n1\n")

    with pytest.raises(ValueError, match=(
            "sizing_concurrency cannot be combined with timestamps_file")):
        _cfg(tmp_path, sizing_concurrency=2, timestamps_file=str(trace))


def test_shards_require_shared_identity_and_future_start(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        _cfg(tmp_path, shard_total=2, shard_index=0, run_id=None)
    with pytest.raises(ValueError, match="start_at_unix"):
        _cfg(tmp_path, shard_total=2, shard_index=0, run_id="shared")
    stale = _cfg(tmp_path, shard_total=2, shard_index=0,
                 run_id="shared", start_at_unix=time.time() - 10)
    with pytest.raises(ValueError, match="stale"):
        run(stale, quiet=True)


def test_obvious_run_config_errors_are_refused_early(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        RunConfig(endpoint=_endpoint())
    with pytest.raises(ValueError, match="duration_s"):
        _cfg(tmp_path, duration_s=0)
    with pytest.raises(ValueError, match="qps_base"):
        _cfg(tmp_path, qps_base=9.0)
    with pytest.raises(ValueError, match="max_output_tokens_cap"):
        _cfg(tmp_path, max_output_tokens_cap=0)
    for field, value, match in (
            ("max_concurrency", 4097, "max_concurrency cannot exceed"),
            ("max_pending_requests", 100_001,
             "max_pending_requests cannot exceed"),
            ("pool_docs_per_bucket", 10_001,
             "pool_docs_per_bucket cannot exceed"),
            ("calibrate_n", 10_001, "calibrate_n cannot exceed")):
        with pytest.raises(ValueError, match=match):
            _cfg(tmp_path, **{field: value})
    with pytest.raises(ValueError, match="exact scheduler limit"):
        _cfg(tmp_path, duration_s=300, qps_base=1_000_000,
             qps_burst=1_000_000, qps_min=1_000_000,
             qps_max=1_000_000)


def _forbid_endpoint_access(monkeypatch):
    contacted = []

    def forbidden(name):
        def call(*args, **kwargs):
            contacted.append((name, args, kwargs))
            raise AssertionError(
                f"{name} occurred before local input prevalidation")
        return call

    monkeypatch.setattr("traffic_replay.runner._token", forbidden("token"))
    monkeypatch.setattr(
        "traffic_replay.runner.EndpointClient", forbidden("client"))
    monkeypatch.setattr(
        "traffic_replay.netpath.measure_network_path",
        forbidden("network-path"))
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata",
        forbidden("control-plane"))
    return contacted


def _workspace_enabled_cfg(tmp_path, **overrides):
    return _cfg(
        tmp_path, measure_network_path=True,
        capture_endpoint_metadata=True, **overrides)


def test_invalid_trace_fails_before_auth_or_workspace_access(
        tmp_path, monkeypatch):
    trace = tmp_path / "bad.trace"
    trace.write_text('{"t": true}\n')
    contacted = _forbid_endpoint_access(monkeypatch)

    with pytest.raises(ValueError, match=r"bad\.trace:1"):
        run(_workspace_enabled_cfg(
            tmp_path / "invalid-trace", timestamps_file=str(trace)), quiet=True)
    assert contacted == []


def test_unsampleable_profile_fails_before_paid_sizing(
        tmp_path, monkeypatch):
    profile = tmp_path / "too-large.json"
    profile.write_text(json.dumps({
        "name": "too-large",
        "input_tokens": {"p50": 300_000, "p95": 300_000},
        "output_tokens": {"p50": 1, "p95": 1},
        "cache_fraction": {"p50": 0, "p95": 0},
    }))
    contacted = _forbid_endpoint_access(monkeypatch)

    with pytest.raises(ValueError, match="outside sampler bounds"):
        run(_workspace_enabled_cfg(
            tmp_path / "invalid-profile", profile_path=str(profile),
            sizing_concurrency=2), quiet=True)
    assert contacted == []


def test_invalid_prompts_fail_before_auth_control_plane_or_sizing(
        tmp_path, monkeypatch):
    prompts = tmp_path / "bad.jsonl"
    prompts.write_text('{"messages": []}\n')
    contacted = _forbid_endpoint_access(monkeypatch)

    with pytest.raises(ValueError, match="messages.*non-empty"):
        run(_workspace_enabled_cfg(
            tmp_path / "invalid-prompts", profile_path=None,
            prompts_file=str(prompts), sizing_concurrency=2), quiet=True)
    assert contacted == []


def test_zero_arrival_schedule_fails_before_any_workspace_access(
        tmp_path, monkeypatch):
    contacted = _forbid_endpoint_access(monkeypatch)
    monkeypatch.setattr(
        "traffic_replay.runner.make_schedule",
        lambda **_kwargs: {
            "rates": np.asarray([0.0]), "counts": np.asarray([0]),
            "timestamps": np.asarray([], dtype=float),
        })

    with pytest.raises(RuntimeError, match="zero arrivals"):
        run(_workspace_enabled_cfg(tmp_path / "zero-schedule"), quiet=True)
    assert contacted == []


def test_workload_construction_failure_precedes_all_workspace_access(
        tmp_path, monkeypatch):
    contacted = _forbid_endpoint_access(monkeypatch)

    def invalid_workload_plan(self, global_index, request_id):
        raise ValueError("deterministic workload construction failed")

    monkeypatch.setattr(
        "traffic_replay.runner._PreparedWorkload.plan",
        invalid_workload_plan)

    with pytest.raises(ValueError, match="workload construction failed"):
        run(_workspace_enabled_cfg(tmp_path / "invalid-workload"), quiet=True)
    assert contacted == []


def test_shared_prevalidation_returns_reusable_exact_inputs(tmp_path):
    rc = _cfg(tmp_path, calibrate_n=2)

    checked = prevalidate_run_inputs(rc)

    assert checked.schedule_kind == "deterministic_synthetic"
    assert checked.full_schedule is not None
    assert len(checked.full_schedule["timestamps"]) > 0
    assert checked.workload is not None
    assert checked.workload.total_n == len(
        checked.full_schedule["timestamps"])
    assert checked.profile is not None
    assert checked.prompts is None
    assert [item["representative"]
            for item in checked.representative_plans] == ["p50", "p95"]


def test_prevalidation_reads_profile_once_and_reuses_it_across_sweep_rungs(
        tmp_path, monkeypatch):
    profile = tmp_path / "shape.json"
    profile.write_bytes(Path(PROFILE).read_bytes())
    target = profile.resolve()
    reads = 0
    real_read_text = Path.read_text

    def counted(path, *args, **kwargs):
        nonlocal reads
        if path.resolve() == target:
            reads += 1
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    first_rc = _cfg(
        tmp_path / "first", profile_path=str(profile),
        qps_base=4.0, qps_burst=4.0, qps_min=4.0, qps_max=4.0)
    first = prevalidate_run_inputs(first_rc)
    second_rc = dataclasses.replace(
        first_rc, qps_base=8.0, qps_burst=8.0,
        qps_min=8.0, qps_max=8.0)
    second = prevalidate_run_inputs(second_rc, reuse_source=first)

    assert reads == 1
    assert second.profile is first.profile
    assert second.representative_plans is first.representative_plans
    assert second.workload is not first.workload


def test_saved_input_expectation_refuses_changed_bytes_before_workspace_access(
        tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    original = b'{"prompt":"original"}\n'
    prompts.write_bytes(original)
    rc = _workspace_enabled_cfg(
        tmp_path / "changed-input", profile_path=None,
        prompts_file=str(prompts), input_expectations={
            "prompts": {
                "sha256": hashlib.sha256(original).hexdigest(),
                "bytes": len(original),
            }})
    prompts.write_text('{"prompt":"different"}\n')
    contacted = _forbid_endpoint_access(monkeypatch)

    with pytest.raises(ValueError, match="input bytes changed"):
        run(rc, quiet=True)
    assert contacted == []


def test_input_expectations_are_closed_and_match_configured_sources(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"one"}\n')
    digest = "0" * 64

    with pytest.raises(ValueError, match="exactly match"):
        _cfg(
            tmp_path / "wrong-key", profile_path=None,
            prompts_file=str(prompts), input_expectations={
                "profile": {"sha256": digest, "bytes": 1}})
    with pytest.raises(ValueError, match="exactly sha256 and bytes"):
        _cfg(
            tmp_path / "unknown-field", profile_path=None,
            prompts_file=str(prompts), input_expectations={
                "prompts": {
                    "sha256": digest, "bytes": 1, "path": "secret"}})
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _cfg(
            tmp_path / "bad-digest", profile_path=None,
            prompts_file=str(prompts), input_expectations={
                "prompts": {"sha256": "G" * 64, "bytes": 1}})


def _fixed_schedule(n=4):
    return {"rates": np.asarray([float(n)]), "counts": np.asarray([n]),
            "timestamps": np.zeros(n, dtype=float)}


def test_unexpected_worker_exceptions_become_persisted_error_rows(
        tmp_path, monkeypatch):
    class RaisingClient:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, *args, **kwargs):
            raise RuntimeError("worker exploded")

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", RaisingClient)
    monkeypatch.setattr("traffic_replay.runner.make_schedule",
                        lambda **kwargs: _fixed_schedule(4))
    out = run(_cfg(tmp_path / "raising"), quiet=True)
    rows = [json.loads(x) for x in
            (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    replay = [r for r in rows if r["phase"] == "replay"]
    assert len(replay) == 4
    assert all(not r["ok"] for r in replay)
    assert all("unexpected worker exception" in r["error"] for r in replay)
    assert sorted(r["global_index"] for r in replay) == [0, 1, 2, 3]
    assert all(r["request_body_sha256"] for r in replay)


def test_pending_future_bound_rejects_instead_of_growing_unbounded(
        tmp_path, monkeypatch):
    class SlowClient:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, messages, max_tokens, request_id, scheduled_s,
                 dispatch_lag_ms, intended, chars_sent):
            time.sleep(0.05)
            return _result(request_id, scheduled_s, dispatch_lag_ms,
                           intended, chars_sent)

    monkeypatch.setattr("traffic_replay.runner.EndpointClient", SlowClient)
    monkeypatch.setattr("traffic_replay.runner.make_schedule",
                        lambda **kwargs: _fixed_schedule(6))
    out = run(_cfg(tmp_path / "bounded", max_concurrency=1,
                   max_pending_requests=1), quiet=True)
    rows = [json.loads(x) for x in
            (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    replay = [r for r in rows if r["phase"] == "replay"]
    assert len(replay) == 6
    rejected = [r for r in replay if "pending limit" in (r["error"] or "")]
    assert rejected
    assert out["summary"]["requests_total"] == 6
