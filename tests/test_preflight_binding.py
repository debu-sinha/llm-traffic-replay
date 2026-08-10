"""Adversarial coverage for command-setup authorization bindings."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from traffic_replay.adapters import get_endpoint_adapter
from traffic_replay.client import EndpointConfig, serialize_request_body
from traffic_replay.runner import (
    RunConfig,
    _SETUP_ARTIFACT_REFERENCE_SCHEMA,
    _attach_setup_request_binding,
    _preflight_binding_for_rows,
    _rescope_representative_plans,
    prevalidate_run_inputs,
    run,
)


def _config(tmp_path: Path) -> RunConfig:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"exact customer representative"}\n')
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n")
    expectations = {
        "prompts": {
            "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
            "bytes": len(prompts.read_bytes()),
        },
        "timestamps": {
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "bytes": len(trace.read_bytes()),
        },
    }
    return RunConfig(
        prompts_file=str(prompts), timestamps_file=str(trace), duration_s=1,
        endpoint={
            "base_url": "https://endpoint-a.example",
            "path": "/serving-endpoints/frontier/invocations",
            "model": "frontier-model",
            "temperature": 0.0,
            "include_usage": True,
            "extra_body": {"reasoning_effort": "none"},
        },
        qps_base=1, qps_burst=1, qps_min=1, qps_max=1,
        max_concurrency=1, max_pending_requests=1, calibrate_n=0,
        max_output_tokens_cap=8, capture_endpoint_metadata=False,
        measure_network_path=False, input_expectations=expectations,
        out_dir=str(tmp_path / "results"),
    )


def _authorization(rc: RunConfig, scope: str = "execution-aaaaaaaa"):
    prevalidated = prevalidate_run_inputs(rc)
    plans = _rescope_representative_plans(
        prevalidated.representative_plans, scope)
    ecfg = EndpointConfig(**rc.endpoint)
    rows = []
    for position, plan in enumerate(plans, start=1):
        physical = hashlib.sha256(serialize_request_body(
            ecfg, plan["messages"], plan["max_output"],
            include_usage=True)).hexdigest()
        row = {
            "phase": "preflight", "request_id": plan["request_id"],
            "global_index": plan["global_index"],
            "sample_index": plan["sample_index"],
            "prompt_index": plan["prompt_index"],
            "body_request_id": plan["body_request_id"],
            "request_attempts": 1, "connection_attempts": 1,
            "retries": 0, "retry_reasons": [],
            "first_attempt_unix": 1_800_000_000.0 + position,
            "first_send_unix": 1_800_000_000.1 + position,
            "t_send_unix": 1_800_000_000.1 + position,
            "finished_unix": 1_800_000_000.2 + position,
            "status": 200, "ok": True, "stream_complete": True,
            "visible_content_seen": True, "reasoning_seen": False,
            "valid_tool_calls": 0, "refusal_seen": False,
            "parse_errors": 0, "max_tokens_requested": plan["max_output"],
            "request_body_sha256": hashlib.sha256(serialize_request_body(
                ecfg, plan["messages"], plan["max_output"],
                include_usage=False)).hexdigest(),
            "physical_request_body_sha256s": [physical],
            "endpoint_adapter": ecfg.adapter,
            "response_mode": "streaming",
        }
        _attach_setup_request_binding(
            row, endpoint=rc.endpoint,
            max_output_tokens_cap=rc.max_output_tokens_cap,
            plan=plan, phase="preflight", position=position)
        rows.append(row)
    binding, digest = _preflight_binding_for_rows(rc, plans, rows)
    gate = {
        "skipped": False, "attempted": 2, "reachable": 2, "readable": 2,
        "reasoning_probe_requests": 0, "outcome": "preflight_passed",
        "force_requested": False, "gate_satisfied": True,
        "evidence_mode": "carried_setup_rows",
        "binding": binding, "binding_sha256": digest,
    }
    reference = {
        "schema_version": _SETUP_ARTIFACT_REFERENCE_SCHEMA,
        "artifact_id": "artifact-setup",
        "execution_id": scope,
        "workload_id": "workload-setup",
        "manifest_sha256": "a" * 64,
        "manifest_bytes": 123,
        "preflight_binding_sha256": digest,
    }
    return plans, rows, gate, reference


@pytest.mark.parametrize("mutation", [
    lambda rc: rc.endpoint.update(base_url="https://endpoint-b.example"),
    lambda rc: rc.endpoint.update(path="/serving-endpoints/other/invocations"),
    lambda rc: rc.endpoint.update(model="other-model"),
    lambda rc: rc.endpoint.update(temperature=None),
    lambda rc: rc.endpoint.update(include_usage=False),
    lambda rc: rc.endpoint.update(extra_body={"reasoning_effort": "high"}),
    lambda rc: setattr(rc, "max_output_tokens_cap", 9),
])
def test_carried_gate_refuses_endpoint_or_control_mutation_before_credentials(
        tmp_path, monkeypatch, mutation):
    original = _config(tmp_path)
    _plans, rows, gate, reference = _authorization(original)
    changed = copy.deepcopy(original)
    mutation(changed)
    credential_calls = []
    monkeypatch.setattr(
        "traffic_replay.runner._token",
        lambda _cfg: credential_calls.append(True))

    with pytest.raises(ValueError, match="does not match this exact"):
        run(changed, quiet=True, prior_request_rows=rows,
            preflight_gate=gate, setup_artifact_reference=reference)

    assert credential_calls == []
    assert not Path(changed.out_dir).exists()


def test_carried_gate_refuses_workload_mutation_before_credentials(
        tmp_path, monkeypatch):
    rc = _config(tmp_path)
    _plans, rows, gate, reference = _authorization(rc)
    Path(rc.prompts_file).write_text('{"prompt":"changed workload"}\n')
    credential_calls = []
    monkeypatch.setattr(
        "traffic_replay.runner._token",
        lambda _cfg: credential_calls.append(True))

    with pytest.raises(ValueError, match="input bytes changed"):
        run(rc, quiet=True, prior_request_rows=rows,
            preflight_gate=gate, setup_artifact_reference=reference)

    assert credential_calls == []
    assert not Path(rc.out_dir).exists()


def test_execution_trace_ids_are_unique_but_body_identity_is_reproducible(
        tmp_path):
    rc = _config(tmp_path)
    base = prevalidate_run_inputs(rc).representative_plans
    first = _rescope_representative_plans(base, "execution-11111111")
    second = _rescope_representative_plans(base, "execution-22222222")
    adapter = get_endpoint_adapter(rc.endpoint["adapter"] if
                                   "adapter" in rc.endpoint else
                                   "openai.chat_completions.sse/v1")

    assert [item["request_id"] for item in first] != [
        item["request_id"] for item in second]
    assert [item["body_request_id"] for item in first] == [
        item["body_request_id"] for item in second]
    assert all(len(item["request_id"]) == 16 and
               set(item["request_id"]) <= set("0123456789abcdef")
               for item in first + second)
    for item in first + second:
        assert adapter.request_headers(item["request_id"])["X-Request-Id"] \
            == item["request_id"]


def test_legacy_carried_gate_and_rows_fail_closed(tmp_path):
    rc = _config(tmp_path)
    legacy = [{
        "phase": "preflight", "request_id": "old-row",
        "endpoint_adapter": "openai.chat_completions.sse/v1",
        "response_mode": "streaming",
    }]
    with pytest.raises(ValueError, match="legacy carried rows fail closed"):
        run(rc, quiet=True, prior_request_rows=legacy)
