"""Quota planning must stop unsafe paid traffic before it starts."""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_replay.client import EndpointConfig, serialize_request_body
from traffic_replay.quota_planner import (
    _CALIBRATED_CPT_HARD_MAX,
    _CHAT_FRAMING_TOKEN_ALLOWANCE,
    _synthetic_json_escape_overhead,
    _workload_values,
    _rolling_peak,
    QuotaPlanError,
    plan_run_quota,
    plan_sweep_quota,
    render_quota_plan,
)
from traffic_replay.runner import RunConfig, run


_ROOT = Path(__file__).resolve().parents[1]


def _limits(**overrides) -> dict:
    value = {
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
    for name, item in overrides.items():
        if item is None:
            value.pop(name, None)
        else:
            value[name] = item
    return value


def _profile(path: Path, *, input_tokens: int = 10_000,
             output_tokens: int = 200,
             cache_fraction: float = 0.5) -> Path:
    path.write_text(json.dumps({
        "name": "quota-test",
        "input_tokens": {"p50": input_tokens, "p95": input_tokens},
        "output_tokens": {"p50": output_tokens, "p95": output_tokens},
        "cache_fraction": {"p50": cache_fraction,
                           "p95": cache_fraction},
        "provenance": "test fixture",
        "label": "test fixture",
    }))
    return path


def _rc(tmp_path: Path, *, rate: float = 0.05,
        duration: int = 300, limits: dict | None = None,
        profile: Path | None = None, endpoint: dict | None = None,
        **overrides) -> RunConfig:
    values = {
        "profile_path": str(profile or _profile(tmp_path / "profile.json")),
        "endpoint": endpoint or {
            "base_url": "https://unit-test.cloud.databricks.com",
            "path": "/serving-endpoints/databricks-glm-5-2/invocations",
            "max_retries": 0,
        },
        "duration_s": duration,
        "qps_base": rate,
        "qps_burst": rate,
        "qps_min": rate,
        "qps_max": rate,
        "rate_scale": 1.0,
        "calibrate_n": 0,
        "max_concurrency": 16,
        "out_dir": str(tmp_path / "out"),
        "max_output_tokens_cap": 200,
        "rate_limits": limits or _limits(),
    }
    values.update(overrides)
    return RunConfig(**values)


def _exact_wire_input_bound(endpoint: dict, messages: list[dict],
                            max_output: int) -> int:
    """Match the submitted body and add only provider-owned chat framing."""
    ecfg = EndpointConfig(**endpoint)
    body = serialize_request_body(
        ecfg, messages, max_output, ecfg.include_usage)
    return len(body) + _CHAT_FRAMING_TOKEN_ALLOWANCE * (len(messages) + 1)


def _clean_prior_row(**overrides) -> dict:
    row = {
        "phase": "preflight",
        "request_id": "prior-clean",
        "request_attempts": 2,
        "connection_attempts": 2,
        "retries": 1,
        "retry_reasons": ["transport retry"],
        "first_send_unix": 1.0,
        "t_send_unix": 1.5,
        "finished_unix": 2.0,
        "status": 200,
        "ok": True,
        "stream_complete": True,
        "parse_errors": 0,
        "prompt_tokens": 123,
        "completion_tokens": 20,
        "cached_tokens": 23,
        "reasoning_tokens": 5,
        "max_tokens_requested": 25,
    }
    row.update(overrides)
    return row


def test_low_rate_glm_shape_passes_only_as_a_harness_budget(tmp_path):
    plan = plan_run_quota(_rc(
        tmp_path,
        limits=_limits(input_tokens_per_minute=10_000_000)))

    assert plan["may_start"] is True
    assert plan["status"] == "within_configured_harness_warning_budget"
    assert plan["provider_headroom_proven"] is False
    assert plan["workspace_external_traffic_included"] is False
    assert plan["rate_limit_snapshot_freshness"]["status"] == "fresh"
    assert plan["physical_attempts_per_logical_worst_case"] == 3
    assert plan["windows"]["input_tokens_per_minute"][
        "ratio_to_configured_limit"] < 0.8
    rendered = render_quota_plan(plan)
    assert "rate-limit snapshot: FRESH" in rendered
    assert f"verified={date.today().isoformat()}" in rendered


def test_rolling_peak_keeps_exact_boundary_event_conservatively():
    events = [
        {"t": 0.0, "queries": 2},
        {"t": 60.0, "queries": 3},
        {"t": 60.000001, "queries": 4},
    ]

    assert _rolling_peak(events[:2], "queries", 60.0) == 5
    assert _rolling_peak(events, "queries", 60.0) == 7


def test_quota_plan_reuses_prevalidated_schedule_and_workload(
        tmp_path, monkeypatch):
    from traffic_replay.runner import prevalidate_run_inputs

    rc = _rc(tmp_path)
    checked = prevalidate_run_inputs(rc)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("quota planner rebuilt prevalidated local inputs")

    monkeypatch.setattr("traffic_replay.quota_planner._schedule", unexpected)
    monkeypatch.setattr(
        "traffic_replay.runner._PreparedWorkload", unexpected)

    plan = plan_run_quota(rc, prevalidated=checked)

    assert plan["logical_replay_requests"] == len(
        checked.full_schedule["timestamps"])


def test_missing_snapshot_freshness_refuses_before_paid_traffic(tmp_path):
    plan = plan_run_quota(_rc(
        tmp_path,
        limits=_limits(verified_at=None, max_age_days=None)))

    assert plan["may_start"] is False
    assert plan["rate_limit_snapshot_freshness"]["status"] == "missing"
    assert any("no verified_at/max_age_days" in reason
               for reason in plan["refusal_reasons"])


def test_stale_snapshot_refuses_before_paid_traffic(tmp_path):
    stale = (date.today() - timedelta(days=8)).isoformat()

    plan = plan_run_quota(_rc(
        tmp_path, limits=_limits(verified_at=stale, max_age_days=7)))

    assert plan["may_start"] is False
    freshness = plan["rate_limit_snapshot_freshness"]
    assert freshness["status"] == "stale"
    assert freshness["age_days"] == 8
    assert any("snapshot is stale" in reason
               for reason in plan["refusal_reasons"])


def test_default_sized_glm_shape_is_refused_before_traffic(tmp_path):
    plan = plan_run_quota(_rc(tmp_path, rate=1.0, duration=120))

    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"][
        "ratio_to_configured_limit"] >= 0.8
    assert any("input_tokens_per_minute" in reason
               for reason in plan["refusal_reasons"])


def test_planner_counts_transport_and_protocol_retries(tmp_path):
    endpoint = {
        "base_url": "https://unit-test.cloud.databricks.com",
        "path": "/serving-endpoints/databricks-glm-5-2/invocations",
        "max_retries": 2,
    }
    plan = plan_run_quota(_rc(
        tmp_path, rate=0.1, duration=60, endpoint=endpoint,
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000)))

    assert plan["physical_attempts_per_logical_worst_case"] == 5
    assert plan["planned_physical_attempts_worst_case"] == \
        plan["logical_replay_requests"] * 5


def test_real_prompt_input_quota_uses_utf8_bytes_not_character_count(
        tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompt = "é" * 50
    prompts.write_text(prompt + "\n")
    trace = tmp_path / "trace.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1,
        limits=_limits(input_tokens_per_minute=600,
                       output_tokens_per_minute=10_000,
                       queries_per_hour=10_000),
        profile=_profile(tmp_path / "unused.json"),
        prompts_file=str(prompts), profile_path=None,
        timestamps_file=str(trace))
    messages = [{"role": "user", "content": prompt}]
    per_attempt = _exact_wire_input_bound(
        rc.endpoint, messages, rc.max_output_tokens_cap)

    plan = plan_run_quota(rc)

    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == per_attempt * 3
    assert not plan["unknowns"]
    assert any("input_tokens_per_minute" in reason
               for reason in plan["refusal_reasons"])


def test_query_only_policy_can_plan_real_prompts(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one real prompt\nanother real prompt\n")
    rc = _rc(
        tmp_path, rate=0.2, duration=60,
        limits=_limits(input_tokens_per_minute=None,
                       output_tokens_per_minute=None,
                       queries_per_hour=100_000),
        prompts_file=str(prompts), profile_path=None)

    plan = plan_run_quota(rc)

    assert plan["may_start"] is True
    assert "queries_per_hour" in plan["windows"]
    assert not plan["unknowns"]


def test_synthetic_replay_reserves_hard_maximum_post_calibration_cpt(
        tmp_path):
    from traffic_replay.runner import prevalidate_run_inputs

    trace = tmp_path / "one-request.txt"
    trace.write_text("0\n")
    profile = _profile(
        tmp_path / "calibration-growth.json",
        input_tokens=100, output_tokens=1)
    rc = _rc(
        tmp_path, duration=1, profile=profile, timestamps_file=str(trace),
        cpt=4.0,
        limits=_limits(input_tokens_per_minute=4_000,
                       output_tokens_per_minute=10_000,
                       queries_per_hour=10_000))

    checked = prevalidate_run_inputs(rc)
    plan = plan_run_quota(rc, prevalidated=checked)

    endpoint = EndpointConfig(**rc.endpoint)
    empty_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    content_chars = math.ceil(100 * _CALIBRATED_CPT_HARD_MAX)
    per_attempt = (
        len(serialize_request_body(
            endpoint, empty_messages, 1, endpoint.include_usage))
        + _CHAT_FRAMING_TOKEN_ALLOWANCE * 3
        + content_chars
        + _synthetic_json_escape_overhead(content_chars)
    )
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == per_attempt * 3
    # Intended-token planning would reserve only 300 and incorrectly pass the
    # 3,200-token warning budget.  The post-calibration byte bound refuses.
    assert plan["may_start"] is False
    assert per_attempt * 3 >= 4_000 * 0.8

    checked.workload.set_cpt(_CALIBRATED_CPT_HARD_MAX)
    concrete = checked.workload.plan(0, "after-calibration")
    concrete_bound = _exact_wire_input_bound(
        rc.endpoint, concrete["messages"], concrete["max_output"])
    assert concrete_bound <= per_attempt


def test_preflight_uses_concrete_bytes_when_intended_tokens_underestimate_3x(
        tmp_path):
    trace = tmp_path / "one-replay.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1,
        profile=_profile(tmp_path / "tiny-replay.json", input_tokens=1,
                         output_tokens=1),
        timestamps_file=str(trace),
        limits=_limits(input_tokens_per_minute=1_300,
                       output_tokens_per_minute=10_000,
                       queries_per_hour=10_000))
    setup = [{
        "messages": [{"role": "user", "content": "x" * 300}],
        "intended": (100, 1, 0.0, -1),
        "max_output": 1,
    }]

    baseline = plan_run_quota(rc)
    plan = plan_run_quota(rc, setup_plans=setup)

    setup_bound = _exact_wire_input_bound(
        rc.endpoint, setup[0]["messages"], setup[0]["max_output"]) * 3
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == baseline["windows"][
            "input_tokens_per_minute"]["planned_peak"] + setup_bound
    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] > 100 * 3


def test_prior_request_without_provider_usage_never_uses_intended_fallback(
        tmp_path):
    trace = tmp_path / "one-prior-replay.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1,
        profile=_profile(tmp_path / "prior.json", input_tokens=1,
                         output_tokens=1),
        timestamps_file=str(trace),
        limits=_limits(input_tokens_per_minute=1_000_000,
                       output_tokens_per_minute=1_000_000,
                       queries_per_hour=100_000))
    prior = [_clean_prior_row(
        prompt_tokens=None, completion_tokens=0, cached_tokens=None,
        reasoning_tokens=None, intended_input_tokens=1,
        max_tokens_requested=1)]

    plan = plan_run_quota(rc, prior_rows=prior)

    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"]["planned_peak"] is None
    assert any("prior_request input tokens are unknown" in unknown
               for unknown in plan["unknowns"])


def test_chat_framing_bound_scales_with_message_count(tmp_path):
    prompts = tmp_path / "conversation.jsonl"
    messages = [{"role": "user", "content": "x"} for _ in range(100)]
    prompts.write_text(json.dumps({"messages": messages}) + "\n")
    trace = tmp_path / "one-conversation.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, prompts_file=str(prompts), profile_path=None,
        timestamps_file=str(trace),
        limits=_limits(input_tokens_per_minute=1_000_000,
                       output_tokens_per_minute=1_000_000,
                       queries_per_hour=100_000))

    plan = plan_run_quota(rc)

    per_attempt = _exact_wire_input_bound(
        rc.endpoint, messages, rc.max_output_tokens_cap)
    assert plan["windows"]["input_tokens_per_minute"]["planned_peak"] == \
        per_attempt * 3


def test_complete_replay_body_counts_huge_noncontent_fields(tmp_path):
    """Roles, metadata, model, tools, and controls must never evade quota."""
    trace = tmp_path / "one-huge-body.txt"
    trace.write_text("0\n")
    messages = [{
        "role": "custom-" + "r" * 100_000,
        "content": "tiny",
        "name": "actor-" + "n" * 100_000,
        "metadata": {"opaque": "m" * 100_000},
    }]
    prompts = tmp_path / "huge-body.jsonl"
    prompts.write_text(
        json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    endpoint = {
        "base_url": "https://unit-test.cloud.databricks.com",
        "path": "/serving-endpoints/databricks-glm-5-2/invocations",
        "model": "deployment-" + "d" * 100_000,
        "max_retries": 0,
        "extra_body": {
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "description": "s" * 100_000,
                    },
                },
            }],
            "thinking": {"type": "enabled", "budget": 512},
            "routing": {
                "provider": "standard", "control": "p" * 100_000,
            },
        },
    }
    ecfg = EndpointConfig(**endpoint)
    body = serialize_request_body(ecfg, messages, 17, ecfg.include_usage)
    expected_payload = {
        **endpoint["extra_body"],
        "messages": messages,
        "max_tokens": 17,
        "temperature": 0.0,
        "stream": True,
        "model": endpoint["model"],
        "stream_options": {"include_usage": True},
    }
    expected_body = json.dumps(
        expected_payload, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")
    assert body == expected_body
    decoded = json.loads(body)
    assert decoded["messages"][0]["role"] == messages[0]["role"]
    assert decoded["messages"][0]["name"] == messages[0]["name"]
    assert decoded["messages"][0]["metadata"] == messages[0]["metadata"]
    assert decoded["model"] == endpoint["model"]
    assert decoded["tools"] == endpoint["extra_body"]["tools"]
    assert decoded["thinking"] == endpoint["extra_body"]["thinking"]
    assert decoded["routing"] == endpoint["extra_body"]["routing"]
    per_attempt = len(expected_body) \
        + _CHAT_FRAMING_TOKEN_ALLOWANCE * 2
    planned_peak = per_attempt * 3
    rc = _rc(
        tmp_path, duration=1, prompts_file=str(prompts), profile_path=None,
        timestamps_file=str(trace), endpoint=endpoint,
        max_output_tokens_cap=17,
        limits=_limits(input_tokens_per_minute=planned_peak,
                       output_tokens_per_minute=100_000_000,
                       queries_per_hour=100_000))

    plan = plan_run_quota(rc)

    peak = plan["windows"]["input_tokens_per_minute"]["planned_peak"]
    assert peak == planned_peak
    assert per_attempt > 600_000
    assert per_attempt > len(messages[0]["content"].encode("utf-8")) * 10_000
    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"][
        "ratio_to_configured_limit"] == 1.0
    assert any("input_tokens_per_minute" in reason
               for reason in plan["refusal_reasons"])


def test_wire_escaping_and_utf8_are_counted_exactly(tmp_path):
    trace = tmp_path / "one-escaped-body.txt"
    trace.write_text("0\n")
    messages = [{
        "role": "odd\"role\\line\n🤖",
        "content": "line one\nline two \\\\ \"quoted\" 💡 雪",
        "metadata": {
            "nested": ["slash\\", "quote\"", "newline\n", "emoji🚀"],
        },
    }]
    prompts = tmp_path / "escaped.jsonl"
    prompts.write_text(
        json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    endpoint = {
        "base_url": "https://unit-test.cloud.databricks.com",
        "path": "/serving-endpoints/databricks-glm-5-2/invocations",
        "model": "model\"\\\n🌍",
        "max_retries": 0,
        "extra_body": {
            "thinking": {"control": "a\nb\\c\"d🌈"},
            "tools": [{"description": "\"\\\n漢字"}],
        },
    }
    rc = _rc(
        tmp_path, duration=1, prompts_file=str(prompts), profile_path=None,
        timestamps_file=str(trace), endpoint=endpoint,
        max_output_tokens_cap=11,
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))
    ecfg = EndpointConfig(**endpoint)
    body = serialize_request_body(ecfg, messages, 11, ecfg.include_usage)
    expected_payload = {
        **endpoint["extra_body"],
        "messages": messages,
        "max_tokens": 11,
        "temperature": 0.0,
        "stream": True,
        "model": endpoint["model"],
        "stream_options": {"include_usage": True},
    }
    expected_body = json.dumps(
        expected_payload, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode("utf-8")
    assert body == expected_body

    plan = plan_run_quota(rc)

    per_attempt = len(body) + _CHAT_FRAMING_TOKEN_ALLOWANCE * 2
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == per_attempt * 3
    assert b"\\n" in body
    assert b"\\\\" in body
    assert b'\\"' in body
    assert "🤖".encode("utf-8") in body
    assert b"\\ud83e" not in body


def test_reasoning_probe_quota_uses_deep_merged_candidate_body(tmp_path):
    from traffic_replay.cli import _quota_setup_plans

    trace = tmp_path / "one-probe-replay.txt"
    trace.write_text("0\n")
    endpoint = {
        "base_url": "https://unit-test.cloud.databricks.com",
        "path": "/serving-endpoints/databricks-glm-5-2/invocations",
        "max_retries": 0,
        "extra_body": {
            "thinking": {"type": "enabled", "budget": 128},
            "routing": {"region": "us-east", "sticky": True},
        },
    }
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace), endpoint=endpoint,
        profile=_profile(tmp_path / "probe-profile.json", input_tokens=1,
                         output_tokens=1),
        limits=_limits(input_tokens_per_minute=100_000_000,
                       output_tokens_per_minute=100_000_000,
                       queries_per_hour=100_000))
    representatives = [{
        "messages": [{"role": "user", "content": "small"}],
        "intended": (10, 3, 0.0, -1),
        "max_output": 3,
    }, {
        "messages": [{"role": "user", "content": "largest"}],
        "intended": (20, 7, 0.0, -1),
        "max_output": 7,
    }]
    probe = {
        "thinking": {
            "type": "disabled",
            "evidence": "probe-control-" + "x" * 20_000,
        },
        "tools": [{"description": "tool-schema-" + "y" * 20_000}],
    }
    setup = _quota_setup_plans(
        dict(rc.__dict__),
        SimpleNamespace(skip_preflight=False, probe_extra_body=[probe]),
        representative_plans=representatives)

    assert len(setup) == 3
    probe_plan = setup[-1]
    merged = probe_plan["_quota_extra_body"]
    assert merged["thinking"] == {
        "type": "disabled", "budget": 128,
        "evidence": probe["thinking"]["evidence"],
    }
    assert merged["routing"] == endpoint["extra_body"]["routing"]
    assert merged["tools"] == probe["tools"]
    probe_endpoint = dict(endpoint)
    probe_endpoint["extra_body"] = merged
    probe_ecfg = EndpointConfig(**probe_endpoint)
    submitted_probe = json.loads(serialize_request_body(
        probe_ecfg, probe_plan["messages"], probe_plan["max_output"],
        probe_ecfg.include_usage))
    assert submitted_probe["thinking"] == merged["thinking"]
    assert submitted_probe["routing"] == merged["routing"]
    assert submitted_probe["tools"] == merged["tools"]

    baseline = plan_run_quota(rc)
    planned = plan_run_quota(rc, setup_plans=setup)
    base_setup = sum(
        _exact_wire_input_bound(
            endpoint, item["messages"], item["max_output"])
        for item in representatives
    )
    candidate_setup = _exact_wire_input_bound(
        probe_endpoint, probe_plan["messages"], probe_plan["max_output"])
    expected_delta = (base_setup + candidate_setup) * 3
    assert planned["windows"]["input_tokens_per_minute"][
        "planned_peak"] == baseline["windows"][
            "input_tokens_per_minute"]["planned_peak"] + expected_delta


def test_clean_prior_usage_counts_observed_attempts_without_multiplier(
        tmp_path):
    trace = tmp_path / "one-clean-prior.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace),
        profile=_profile(tmp_path / "clean-prior.json", input_tokens=1,
                         output_tokens=1),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))
    baseline = plan_run_quota(rc)

    plan = plan_run_quota(rc, prior_rows=[_clean_prior_row()])

    assert plan["unknowns"] == []
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] == baseline["windows"][
            "input_tokens_per_minute"]["planned_peak"] + 123 * 2
    assert plan["windows"]["output_tokens_per_minute"][
        "planned_peak"] == baseline["windows"][
            "output_tokens_per_minute"]["planned_peak"] + 25 * 2
    assert plan["windows"]["queries_per_hour"][
        "planned_peak"] == baseline["windows"][
            "queries_per_hour"]["planned_peak"] + 2


@pytest.mark.parametrize("dirty", [
    {"status": 201},
    {"ok": False},
    {"stream_complete": False},
    {"parse_errors": 1},
    {"parse_errors": True},
    {"prompt_tokens": 0},
    {"prompt_tokens": True},
    {"completion_tokens": -1},
    {"completion_tokens": True},
    {"cached_tokens": -1},
    {"cached_tokens": 124},
    {"cached_tokens": True},
    {"reasoning_tokens": -1},
    {"reasoning_tokens": 21},
    {"reasoning_tokens": True},
], ids=[
    "non-200", "not-ok", "partial-stream", "parse-error",
    "boolean-parse-errors", "zero-prompt", "boolean-prompt",
    "negative-completion", "boolean-completion", "negative-cached",
    "cached-over-prompt", "boolean-cached", "negative-reasoning",
    "reasoning-over-completion", "boolean-reasoning",
])
def test_dirty_prior_usage_fails_input_quota_closed(tmp_path, dirty):
    trace = tmp_path / "one-dirty-prior.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace),
        profile=_profile(tmp_path / "dirty-prior.json", input_tokens=1,
                         output_tokens=1),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))

    plan = plan_run_quota(
        rc, prior_rows=[_clean_prior_row(**dirty)])

    assert plan["may_start"] is False
    assert plan["windows"]["input_tokens_per_minute"][
        "planned_peak"] is None
    assert any("prior_request input tokens are unknown" in item
               for item in plan["unknowns"])
    assert any("input_tokens_per_minute cannot be bounded" in item
               for item in plan["refusal_reasons"])


@pytest.mark.parametrize("contradiction", [
    {"request_attempts": None},
    {"request_attempts": True},
    {"request_attempts": -1},
    {"connection_attempts": None},
    {"connection_attempts": True},
    {"request_attempts": 2, "connection_attempts": 1},
    {"first_send_unix": None},
    {"first_send_unix": True},
    {"first_send_unix": -1},
    {"first_send_unix": float("nan")},
    {"first_send_unix": float("inf")},
    {"t_send_unix": None},
    {"t_send_unix": True},
    {"t_send_unix": 0.5},
    {"retries": -1},
    {"retry_reasons": None},
    {"retry_reasons": [""]},
    {"retries": 1, "retry_reasons": []},
    {"request_attempts": 0, "connection_attempts": 0,
     "retries": 0, "retry_reasons": []},
], ids=[
    "missing-attempts", "boolean-attempts", "negative-attempts",
    "missing-connections", "boolean-connections", "attempts-over-connections",
    "missing-first-send", "boolean-first-send", "negative-first-send",
    "nan-first-send", "infinite-first-send", "missing-last-send",
    "boolean-last-send", "last-before-first", "negative-retries",
    "missing-retry-reasons", "empty-retry-reason", "retry-count-mismatch",
    "zero-attempts-with-sent-evidence",
])
def test_contradictory_prior_attempt_evidence_fails_closed(
        tmp_path, contradiction):
    trace = tmp_path / "one-attempt-contradiction.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace),
        profile=_profile(tmp_path / "attempt-contradiction.json",
                         input_tokens=1, output_tokens=1),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))

    plan = plan_run_quota(
        rc, prior_rows=[_clean_prior_row(**contradiction)])

    assert plan["may_start"] is False
    assert plan["planned_physical_attempts_worst_case"] is None
    assert any("unknown provider attempts" in item
               for item in plan["unknowns"])
    assert any("provider-attempt count is unknown" in item
               for item in plan["refusal_reasons"])


@pytest.mark.parametrize("positive_evidence", [
    {"ok": True},
    {"stream_complete": True},
])
def test_zero_attempts_rejects_positive_protocol_evidence(
        tmp_path, positive_evidence):
    trace = tmp_path / "one-zero-attempt-contradiction.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace),
        profile=_profile(tmp_path / "zero-attempt-contradiction.json",
                         input_tokens=1, output_tokens=1),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))
    row = _clean_prior_row(
        request_attempts=0, connection_attempts=0,
        retries=0, retry_reasons=[],
        first_send_unix=None, t_send_unix=None, finished_unix=None,
        status=None, prompt_tokens=None, completion_tokens=None,
        cached_tokens=None, reasoning_tokens=None,
        ok=False, stream_complete=False)
    row.update(positive_evidence)

    plan = plan_run_quota(rc, prior_rows=[row])

    assert plan["may_start"] is False
    assert plan["planned_physical_attempts_worst_case"] is None
    assert any("unknown provider attempts" in item
               for item in plan["unknowns"])


def test_clean_zero_attempt_row_is_ignored_without_unknowns(tmp_path):
    trace = tmp_path / "one-unsent-prior.txt"
    trace.write_text("0\n")
    rc = _rc(
        tmp_path, duration=1, timestamps_file=str(trace),
        profile=_profile(tmp_path / "unsent-prior.json", input_tokens=1,
                         output_tokens=1),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000))
    baseline = plan_run_quota(rc)
    row = _clean_prior_row(
        request_attempts=0, connection_attempts=1,
        retries=0, retry_reasons=[],
        first_send_unix=None, t_send_unix=None, finished_unix=None,
        status=None, prompt_tokens=None, completion_tokens=None,
        cached_tokens=None, reasoning_tokens=None,
        max_tokens_requested=None, ok=False, stream_complete=False,
        parse_errors=0)

    plan = plan_run_quota(rc, prior_rows=[row])

    assert plan["unknowns"] == []
    assert plan["planned_physical_attempts_worst_case"] == baseline[
        "planned_physical_attempts_worst_case"]
    assert plan["windows"] == baseline["windows"]


@pytest.mark.parametrize("cpt", [1.5, 4.0, 12.0, 17.25])
@pytest.mark.parametrize("cache_fraction", [0.0, 0.5, 1.0])
def test_synthetic_analytical_bound_covers_exact_serialized_wire_body(
        tmp_path, cpt, cache_fraction):
    from traffic_replay.runner import prevalidate_run_inputs

    trace = tmp_path / "one-synthetic-bound.txt"
    trace.write_text("0\n")
    for seed in (0, 1, 2_147_483_647):
        for input_tokens in (1, 101, 4_096):
            profile = _profile(
                tmp_path / f"synthetic-{seed}-{input_tokens}.json",
                input_tokens=input_tokens, output_tokens=7,
                cache_fraction=cache_fraction)
            endpoint = {
                "base_url": "https://unit-test.cloud.databricks.com",
                "path": ("/serving-endpoints/databricks-glm-5-2/"
                         "invocations"),
                "model": "model-\n-🤖",
                "max_retries": 0,
                "extra_body": {
                    "thinking": {"control": "a\nb\\c\"d"},
                    "tools": [{"description": "schema-\n-\\-\"-🌍"}],
                },
            }
            rc = _rc(
                tmp_path, duration=1, timestamps_file=str(trace),
                profile=profile, endpoint=endpoint, cpt=cpt, seed=seed,
                max_output_tokens_cap=7,
                limits=_limits(input_tokens_per_minute=100_000_000,
                               output_tokens_per_minute=100_000_000,
                               queries_per_hour=100_000))
            checked = prevalidate_run_inputs(rc)
            ecfg = EndpointConfig(**endpoint)
            for post_calibration in (False, True):
                planned_input, planned_output = _workload_values(
                    ecfg, checked.workload, 0,
                    post_calibration=post_calibration)
                exact_cpt = (max(cpt, _CALIBRATED_CPT_HARD_MAX)
                             if post_calibration else cpt)
                checked.workload.set_cpt(exact_cpt)
                concrete = checked.workload.plan(0, "quota-plan-prompt-0")
                exact_input = _exact_wire_input_bound(
                    endpoint, concrete["messages"], concrete["max_output"])
                assert planned_output == concrete["max_output"] == 7
                assert planned_input >= exact_input, (
                    cpt, cache_fraction, seed, input_tokens,
                    post_calibration, planned_input, exact_input)


def test_sizing_concurrency_cannot_claim_a_pretraffic_quota_plan(tmp_path):
    rc = _rc(tmp_path, sizing_concurrency=1)

    plan = plan_run_quota(rc)

    assert plan["may_start"] is False
    assert any("fixed rate" in reason for reason in plan["refusal_reasons"])


def test_whole_default_sweep_is_refused_on_cumulative_qph(tmp_path):
    base = _rc(
        tmp_path,
        limits=_limits(input_tokens_per_minute=100_000_000,
                       output_tokens_per_minute=100_000_000)).__dict__
    base = dict(base)
    base["calibrate_n"] = 0
    rates = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    plan = plan_sweep_quota(
        base, rates, duration_s=120, cooldown_s=60)

    assert plan["may_start"] is False
    assert plan["logical_replay_requests"] > 7_200
    assert plan["windows"]["queries_per_hour"][
        "ratio_to_configured_limit"] >= 0.8


def test_sweep_cooldown_never_manufactures_a_quota_reset(tmp_path):
    base = dict(_rc(
        tmp_path, rate=10.0, duration=1,
        profile=_profile(tmp_path / "small.json", input_tokens=100,
                         output_tokens=10),
        limits=_limits(input_tokens_per_minute=10_000_000,
                       output_tokens_per_minute=10_000_000,
                       queries_per_hour=100_000)).__dict__)
    no_setup = plan_sweep_quota(
        base, [10.0], duration_s=1, cooldown_s=60)
    content = "preflight"
    setup = [{
        "messages": [{"role": "user", "content": content}],
        "intended": (100, 10, 0.0, -1),
        "max_output": 10,
    }]
    with_setup = plan_sweep_quota(
        base, [10.0], duration_s=1, cooldown_s=60, setup_plans=setup)

    setup_bound = _exact_wire_input_bound(
        base["endpoint"], setup[0]["messages"], setup[0]["max_output"])
    assert with_setup["windows"]["input_tokens_per_minute"][
        "planned_peak"] == no_setup["windows"][
                "input_tokens_per_minute"]["planned_peak"] + setup_bound * 3
    assert with_setup["windows"]["queries_per_hour"][
        "planned_peak"] == no_setup["windows"][
            "queries_per_hour"]["planned_peak"] + 3


def test_runner_refuses_before_auth_or_network_lookup(tmp_path, monkeypatch):
    rc = _rc(tmp_path, rate=1.0, duration=120)
    contacted = []

    def should_not_run(*args, **kwargs):
        contacted.append((args, kwargs))
        raise AssertionError("network/auth path was reached")

    monkeypatch.setattr("traffic_replay.runner._token", should_not_run)

    with pytest.raises(QuotaPlanError, match="refused before endpoint traffic"):
        run(rc, quiet=True)
    assert contacted == []
    assert not (tmp_path / "out").exists()


def test_cli_refuses_before_preflight_and_loads_a_dated_snapshot(
        tmp_path, monkeypatch):
    from traffic_replay.cli import main

    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(_limits()))

    def should_not_preflight(_cfg):
        raise AssertionError("preflight sent traffic after quota refusal")

    monkeypatch.setattr("traffic_replay.cli._preflight", should_not_preflight)
    code = main([
        "benchmark",
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--fixed-rate", "1",
        "--duration", "120",
        "--input-tokens", "10000,10000",
        "--output-tokens", "200,200",
        "--rate-limits", str(limits_path),
        "--out-dir", str(tmp_path / "cli-out"),
    ])

    assert code == 3


def test_cli_empty_schedule_is_a_clean_pretraffic_refusal(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import main

    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(_limits()))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("empty schedule reached auth, metadata, or preflight")

    monkeypatch.setattr("traffic_replay.cli._preflight", unexpected)
    monkeypatch.setattr(
        "traffic_replay.endpoint_meta.fetch_endpoint_metadata", unexpected)

    code = main([
        "benchmark",
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--fixed-rate", "0.000000001",
        "--duration", "1",
        "--input-tokens", "10",
        "--output-tokens", "10",
        "--rate-limits", str(limits_path),
        "--out-dir", str(tmp_path / "cli-empty"),
    ])

    assert code == 2
    output = capsys.readouterr().out
    assert "REFUSED before endpoint traffic" in output
    assert "schedule produced zero arrivals" in output


def _forbid_cli_endpoint_stages(monkeypatch):
    contacted = []

    def forbidden(name):
        def call(*args, **kwargs):
            contacted.append((name, args, kwargs))
            raise AssertionError(f"invalid local input reached {name}")
        return call

    for target, name in (
            ("traffic_replay.cli._quota_gate", "quota"),
            ("traffic_replay.cli._check_preflight", "preflight"),
            ("traffic_replay.runner._token", "token"),
            ("traffic_replay.runner.EndpointClient", "client"),
            ("traffic_replay.netpath.measure_network_path", "network"),
            ("traffic_replay.endpoint_meta.fetch_endpoint_metadata",
             "control-plane"),
            ("traffic_replay.sweep_artifacts.SweepArtifacts.claim",
             "sweep-claim"),
            ("traffic_replay.runner.run", "runner")):
        monkeypatch.setattr(target, forbidden(name))
    return contacted


def _unsampleable_profile(path: Path) -> Path:
    path.write_text(json.dumps({
        "name": "outside-runner-bounds",
        "input_tokens": {"p50": 300_000, "p95": 300_000},
        "output_tokens": {"p50": 1, "p95": 1},
        "cache_fraction": {"p50": 0, "p95": 0},
    }))
    return path


@pytest.mark.parametrize("command", ["benchmark", "sweep"])
def test_cli_unsampleable_profile_never_reaches_quota_or_endpoint_stages(
        tmp_path, monkeypatch, command):
    from traffic_replay.cli import main

    contacted = _forbid_cli_endpoint_stages(monkeypatch)
    argv = [
        command,
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--profile", str(_unsampleable_profile(tmp_path / "large.json")),
        "--duration", "1",
        "--out-dir", str(tmp_path / command),
    ]
    if command == "benchmark":
        argv.extend(["--fixed-rate", "1"])
    else:
        argv.extend(["--rate", "1"])

    assert main(argv) == 2
    assert contacted == []


@pytest.mark.parametrize("command", ["benchmark", "sweep"])
def test_cli_zero_arrival_never_reaches_quota_or_endpoint_stages(
        tmp_path, monkeypatch, command):
    from traffic_replay.cli import main

    contacted = _forbid_cli_endpoint_stages(monkeypatch)
    argv = [
        command,
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--duration", "1",
        "--input-tokens", "10,10",
        "--output-tokens", "1,1",
        "--out-dir", str(tmp_path / command),
    ]
    if command == "benchmark":
        argv.extend(["--fixed-rate", "0.000000001"])
    else:
        argv.extend(["--rate", "0.000000001"])

    assert main(argv) == 2
    assert contacted == []


def test_sweep_prevalidates_every_exact_rung_before_quota(
        tmp_path, monkeypatch):
    from traffic_replay.cli import main
    import traffic_replay.runner as runner

    events = []
    real_prevalidate = runner.prevalidate_run_inputs

    def observe(rc, **kwargs):
        result = real_prevalidate(rc, **kwargs)
        events.append(("prevalidate", rc.qps_base, rc.qps_burst,
                       rc.qps_min, rc.qps_max, rc.rate_scale,
                       rc.duration_s, rc.calibrate_n,
                       rc.sizing_concurrency))
        return result

    def stop_at_quota(_cfg, _args, *, rates=None,
                      prevalidated_rungs=None, **_kwargs):
        events.append(("quota", list(rates or []),
                       len(prevalidated_rungs or [])))
        return 3

    monkeypatch.setattr(
        "traffic_replay.runner.prevalidate_run_inputs", observe)
    monkeypatch.setattr("traffic_replay.cli._quota_gate", stop_at_quota)

    code = main([
        "sweep",
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--rate", "100,200,300", "--duration", "1",
        "--input-tokens", "10,10", "--output-tokens", "1,1",
        "--out-dir", str(tmp_path / "ordered"),
    ])

    assert code == 3
    assert [event[0] for event in events] == [
        "prevalidate", "prevalidate", "prevalidate", "quota"]
    for event, rate in zip(events[:3], (100.0, 200.0, 300.0)):
        assert event[1:6] == (rate, rate, rate, rate, 1.0)
        assert event[6:] == (1, 0, None)
    assert events[-1] == ("quota", [100.0, 200.0, 300.0], 3)


def test_long_low_rate_sweep_base_never_inherits_bursty_defaults(
        tmp_path, monkeypatch):
    from traffic_replay.cli import main

    seen = {}

    def stop_at_quota(cfg, _args, *, rates=None,
                      prevalidated_rungs=None, **_kwargs):
        seen.update(
            rates=list(rates or []),
            qps=(cfg["qps_base"], cfg["qps_burst"],
                 cfg["qps_min"], cfg["qps_max"]),
            validated=len(prevalidated_rungs or []),
        )
        return 3

    monkeypatch.setattr("traffic_replay.cli._quota_gate", stop_at_quota)
    code = main([
        "sweep",
        "--host", "https://unit-test.cloud.databricks.com",
        "--endpoint", "databricks-glm-5-2",
        "--rate", "1", "--duration", "3000",
        "--input-tokens", "10,10", "--output-tokens", "1,1",
        "--out-dir", str(tmp_path / "long-low-rate"),
    ])

    assert code == 3
    assert seen == {
        "rates": [1.0], "qps": (1.0, 1.0, 1.0, 1.0), "validated": 1}
