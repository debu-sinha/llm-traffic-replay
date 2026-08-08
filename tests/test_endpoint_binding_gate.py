"""Quota-aware commands bind endpoint identity before paid inference."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pytest

from traffic_replay.client import RequestResult
from traffic_replay.endpoint_meta import _summarize, rate_limit_endpoint_binding
from traffic_replay.quota_planner import QuotaPlanError
from traffic_replay.runner import RunConfig, run


def _limits() -> dict:
    return {
        "input_tokens_per_minute": 200_000,
        "output_tokens_per_minute": 20_000,
        "queries_per_hour": 7_200,
        "warning_utilization": 0.8,
        "source": ("https://docs.databricks.com/aws/en/machine-learning/"
                   "foundation-model-apis/limits"),
        "as_of": "2026-08-03",
        "verified_at": date.today().isoformat(),
        "max_age_days": 7,
        "scope": "Enterprise workspace pay-per-token traffic",
        "provider": "databricks",
        "deployment_mode": "pay_per_token",
        "workspace_tier": "Enterprise",
        "model": "databricks-glm-5-2",
        "accounting_model": "databricks_fmapi_pay_per_token",
    }


def _metadata(*, name: str = "databricks-glm-5-2",
              provisioned: bool = False,
              foundation_model_name: str | None = (
                  "system.ai.databricks-glm-5-2")) -> dict:
    entity = {"name": name}
    if foundation_model_name is not None:
        entity["foundation_model"] = {"name": foundation_model_name}
    if provisioned:
        entity.update(workload_type="GPU_LARGE", workload_size="Medium")
    return {
        "name": name,
        "task": "llm/v1/chat",
        "route_optimized": False,
        "ready": "READY",
        "served_entities": [entity],
    }


LIVE_GLM_PAY_PER_TOKEN_RESPONSE = {
    "name": "databricks-glm-5-2",
    "task": "llm/v1/chat",
    "route_optimized": False,
    "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
    "config": {"served_entities": [{
        "name": "databricks-glm-5-2",
        "foundation_model": {"name": "system.ai.databricks-glm-5-2"},
    }]},
}


def _profile(path: Path) -> Path:
    path.write_text(json.dumps({
        "name": "binding-gate-fixture",
        "input_tokens": {"p50": 1_000, "p95": 1_000},
        "output_tokens": {"p50": 10, "p95": 10},
        "cache_fraction": {"p50": 0.5, "p95": 0.5},
        "provenance": "test fixture",
        "label": "test fixture",
    }))
    return path


def _run_config(tmp_path: Path) -> RunConfig:
    trace = tmp_path / "timestamps.txt"
    trace.write_text("0\n")
    return RunConfig(
        endpoint={
            "base_url": "https://unit-test.cloud.databricks.com",
            "path": "/serving-endpoints/databricks-glm-5-2/invocations",
            "max_retries": 0,
        },
        profile_path=str(_profile(tmp_path / "profile.json")),
        timestamps_file=str(trace),
        duration_s=1,
        qps_base=0.05,
        qps_burst=0.05,
        qps_min=0.05,
        qps_max=0.05,
        calibrate_n=0,
        max_concurrency=1,
        max_output_tokens_cap=10,
        measure_network_path=False,
        out_dir=str(tmp_path / "runs"),
        rate_limits=_limits(),
    )


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        (None, "metadata was not captured"),
        (_metadata(name="another-model"), "endpoint name does not match"),
        (_metadata(provisioned=True), "provisioned-throughput entity fields"),
        (_metadata(foundation_model_name=None),
         "missing foundation_model.name evidence"),
        (_metadata(foundation_model_name="system.ai.another-model"),
         "foundation_model.name does not match expected"),
    ],
)
def test_shared_binding_fails_closed_without_claiming_workspace_tier(
        metadata, reason):
    binding = rate_limit_endpoint_binding(
        _limits(), metadata,
        "/serving-endpoints/databricks-glm-5-2/invocations")

    assert binding["status"] == "refused"
    assert binding["binding_complete"] is False
    assert any(reason in item for item in binding["reasons"])
    assert binding["workspace_tier_is_configured_assertion"] is True
    assert binding["workspace_tier_verified"] is False


def test_shared_binding_accepts_captured_pay_per_token_shape():
    metadata = _summarize(LIVE_GLM_PAY_PER_TOKEN_RESPONSE)
    binding = rate_limit_endpoint_binding(
        _limits(), metadata,
        "/serving-endpoints/databricks-glm-5-2/invocations")

    assert binding["status"] == "verified"
    assert binding["endpoint_model_verified"] is True
    assert binding["expected_foundation_model_name"] == \
        "system.ai.databricks-glm-5-2"
    assert binding["observed_foundation_model_names"] == [
        "system.ai.databricks-glm-5-2"]
    assert binding["foundation_model_names_verified"] is True
    assert binding["deployment_mode_verified"] is True
    assert binding["binding_complete"] is True
    assert binding["reasons"] == []
    assert binding["workspace_tier_verified"] is False
    assert "system.ai.<rate_limits.model>" in binding["note"]


def test_shared_binding_rejects_mixed_active_foundation_models():
    metadata = _metadata()
    metadata["served_entities"].append({
        "name": "databricks-glm-5-2",
        "foundation_model": {"name": "system.ai.another-model"},
    })

    binding = rate_limit_endpoint_binding(
        _limits(), metadata,
        "/serving-endpoints/databricks-glm-5-2/invocations")

    assert binding["status"] == "refused"
    assert binding["foundation_model_names_verified"] is False
    assert binding["observed_foundation_model_names"] == [
        "system.ai.databricks-glm-5-2", "system.ai.another-model"]
    assert any("foundation_model.name does not match expected" in reason
               for reason in binding["reasons"])


@pytest.mark.parametrize("route_optimized", [None, True])
def test_shared_binding_rejects_missing_or_optimized_route_mode(
        route_optimized):
    metadata = _metadata()
    metadata["route_optimized"] = route_optimized

    binding = rate_limit_endpoint_binding(
        _limits(), metadata,
        "/serving-endpoints/databricks-glm-5-2/invocations")

    assert binding["binding_complete"] is False
    assert binding["route_mode_verified"] is False
    assert any("route_optimized" in reason for reason in binding["reasons"])


@pytest.mark.parametrize("ready", [None, "NOT_READY", "UPDATE_FAILED"])
def test_shared_binding_rejects_endpoint_that_is_not_exact_ready(ready):
    metadata = _metadata()
    metadata["ready"] = ready

    binding = rate_limit_endpoint_binding(
        _limits(), metadata,
        "/serving-endpoints/databricks-glm-5-2/invocations")

    assert binding["binding_complete"] is False
    assert binding["endpoint_ready_verified"] is False
    assert any("not exact READY" in reason for reason in binding["reasons"])


@pytest.mark.parametrize("path", [
    "/serving-endpoints/databricks-glm-5-2",
    "/serving-endpoints/databricks-glm-5-2/chat/completions",
    "/serving-endpoints/databricks-glm-5-2/invocations/extra",
    "/serving-endpoints/databricks-glm-5-2/invocations?x=1",
])
def test_shared_binding_rejects_noncanonical_request_route(path):
    binding = rate_limit_endpoint_binding(_limits(), _metadata(), path)

    assert binding["binding_complete"] is False
    assert binding["configured_route_endpoint_name"] is None
    assert any("request route endpoint does not match" in reason
               for reason in binding["reasons"])


def test_runner_seals_binding_refusal_before_any_inference(
        tmp_path, monkeypatch):
    inference_calls = []
    monkeypatch.setattr("traffic_replay.runner._token", lambda _cfg: "token")
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata",
        lambda *_args, **_kwargs: _metadata(provisioned=True))

    def must_not_send(*args, **kwargs):
        inference_calls.append((args, kwargs))
        raise AssertionError("paid inference was reached")

    monkeypatch.setattr(
        "traffic_replay.client.EndpointClient.send", must_not_send)
    rc = _run_config(tmp_path)

    with pytest.raises(QuotaPlanError, match="before paid inference traffic"):
        run(rc, quiet=True)

    assert inference_calls == []
    artifacts = list((tmp_path / "runs").iterdir())
    assert len(artifacts) == 1
    start = json.loads((artifacts[0] / "start.json").read_text())
    plan = start["quota_plan"]
    assert start["status"] == "quota-binding-refused"
    assert start["endpoint_binding"]["status"] == "refused"
    assert plan["status"] == "refused"
    assert plan["refusal_stage"] == "endpoint_binding"
    assert plan["endpoint_binding"]["binding_complete"] is False
    assert any("provisioned-throughput" in reason
               for reason in plan["refusal_reasons"])


def _successful_result(request_id, scheduled_s, dispatch_lag_ms, intended,
                       chars_sent) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        scheduled_s=scheduled_s,
        dispatch_lag_ms=dispatch_lag_ms,
        t_send_unix=now,
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
        prompt_tokens=intended[0],
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
        max_tokens_requested=10,
        first_send_unix=now,
        first_attempt_unix=now,
        finished_unix=now + 0.003,
        connection_attempts=1,
        request_attempts=1,
    )


def test_runner_matching_fixture_reaches_inference_and_persists_binding(
        tmp_path, monkeypatch):
    inference_calls = []
    monkeypatch.setattr("traffic_replay.runner._token", lambda _cfg: "token")
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata",
        lambda *_args, **_kwargs: _metadata())

    def send(_self, _messages, _max_tokens, request_id, scheduled_s,
             dispatch_lag_ms, intended, chars_sent, **_kwargs):
        inference_calls.append(request_id)
        return _successful_result(
            request_id, scheduled_s, dispatch_lag_ms, intended, chars_sent)

    monkeypatch.setattr("traffic_replay.client.EndpointClient.send", send)
    out = run(_run_config(tmp_path), quiet=True)

    assert len(inference_calls) == 1
    start = json.loads((Path(out["out_dir"]) / "start.json").read_text())
    assert start["endpoint_binding"]["binding_complete"] is True
    assert start["quota_plan"]["status"] == "ready_for_paid_inference"
    assert start["quota_plan"]["may_start"] is True


def _benchmark_args(tmp_path: Path, limits_path: Path) -> list[str]:
    return [
        "benchmark",
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--fixed-rate", "1",
        "--duration", "2",
        "--input-tokens", "1000,1000",
        "--output-tokens", "10,10",
        "--rate-limits", str(limits_path),
        "--out-dir", str(tmp_path / "cli-out"),
    ]


def test_cli_binding_refusal_never_reaches_preflight_or_runner(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import main

    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(_limits()))
    monkeypatch.setattr("traffic_replay.runner._token", lambda _cfg: "token")
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata",
        lambda *_args, **_kwargs: None)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("paid inference path was reached")

    monkeypatch.setattr("traffic_replay.cli._preflight", must_not_run)
    monkeypatch.setattr("traffic_replay.runner.run", must_not_run)

    args = _benchmark_args(tmp_path, limits_path) + ["--format", "json"]
    assert main(args) == 3
    refusal = json.loads(capsys.readouterr().out)
    plan = refusal["quota_plan"]
    assert refusal["stage"] == "quota_plan"
    assert plan["status"] == "refused"
    assert plan["refusal_stage"] == "endpoint_binding"
    assert plan["endpoint_binding"]["status"] == "refused"
    assert any("metadata was not captured" in reason
               for reason in plan["refusal_reasons"])


def test_cli_matching_fixture_passes_gate_and_invokes_runner(
        tmp_path, monkeypatch):
    from traffic_replay.cli import main

    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(_limits()))
    monkeypatch.setattr("traffic_replay.runner._token", lambda _cfg: "token")
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata",
        lambda *_args, **_kwargs: _metadata())
    calls = []

    def fake_run(rc, quiet=False):
        calls.append(rc)
        return {"out_dir": rc.out_dir, "summary": {}}

    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr(
        "traffic_replay.cli._finish",
        lambda _out, _fail_on="miss", _fmt="text": 0)
    args = _benchmark_args(tmp_path, limits_path) + ["--skip-preflight"]

    assert main(args) == 0
    assert len(calls) == 1
