"""Regression coverage for workload identity and control-plane safety."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_replay.client import EndpointConfig, RequestResult
from traffic_replay.runner import (
    RunConfig, _PreparedWorkload, _payload_hash, _representative_plans,
    _resolved_run_id, _shard_concurrency, _stable_request_id, run,
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
    assert result["budget"] in result["budgets"]


def test_known_reasoning_controls_include_nested_chat_template_flag():
    from traffic_replay.cli import _REASONING_LEVERS
    controls = dict(_REASONING_LEVERS)
    assert controls["chat_template_kwargs.enable_thinking=false"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


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


def test_sizing_share_uses_exact_quotient_remainder_and_allows_zero(tmp_path):
    shares = []
    future = time.time() + 60
    for shard_index in range(5):
        rc = _cfg(tmp_path, sizing_concurrency=2, shard_total=5,
                  shard_index=shard_index, start_at_unix=future)
        shares.append(_shard_concurrency(rc))
    assert shares == [1, 1, 0, 0, 0]
    assert sum(shares) == 2


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
