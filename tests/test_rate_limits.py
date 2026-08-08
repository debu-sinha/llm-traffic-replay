"""Rolling quota evidence must expose bursts without claiming provider state."""
from __future__ import annotations

from datetime import date

import pytest

from traffic_replay.config_validation import validate_rate_limits
from traffic_replay.metrics import (_rate_limit_evidence, _rolling_peak,
                                    _verdict, render_html, render_markdown,
                                    summarize)
from traffic_replay.runner import RunConfig, _prepare_prior_request_rows


def _row(stamp: float, prompt: int | None, *, output: int = 10,
         reserved: int = 20, attempts: int = 1) -> dict:
    return {
        "request_id": f"r-{stamp}",
        "phase": "replay",
        "ok": True,
        "status": 200,
        "first_send_unix": stamp,
        "finished_unix": stamp + 0.5,
        "queue_wait_ms": 0.0,
        "caller_ttft_ms": 10.0,
        "caller_e2e_ms": 20.0,
        "ttft_ms": 10.0,
        "e2e_ms": 20.0,
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "max_tokens_requested": reserved,
        "request_attempts": attempts,
        "visible_content_seen": True,
        "valid_tool_calls": 0,
        "stream_complete": True,
        "parse_errors": 0,
        "finish_reason": "stop",
        "intended_input_tokens": prompt,
        "intended_output_tokens": output,
        "intended_cache_fraction": None,
        "cached_tokens": None,
        "retries": max(attempts - 1, 0),
    }


def _limits(**overrides) -> dict:
    value = {
        "input_tokens_per_minute": 1_000,
        "output_tokens_per_minute": 1_000,
        "queries_per_hour": 1_000,
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


def _meta() -> dict:
    return {
        "endpoint_metadata": {
            "name": "databricks-glm-5-2",
            "ready": "READY",
            "route_optimized": False,
            "served_entities": [{
                "name": "databricks-glm-5-2",
                "foundation_model": {
                    "name": "system.ai.databricks-glm-5-2",
                },
            }],
        }
    }


def test_rolling_peak_uses_a_true_half_open_sliding_window():
    peak = _rolling_peak([
        (100.0, 100.0),
        (159.9, 200.0),
        (160.0, 400.0),
        (220.0, 50.0),
    ], 60.0)

    # At t=160 the event at t=100 is exactly 60 seconds old and is excluded;
    # the 159.9 and 160.0 events remain together.
    assert peak["max"] == 600
    assert peak["events_in_peak"] == 2
    assert peak["window_end_unix"] == 160.0
    assert peak["window_start_unix"] == 100.0


def test_summary_reports_burst_and_compares_as_of_limits():
    rows = [_row(100.0, 100), _row(159.9, 200), _row(160.0, 400)]
    summary = summarize(
        rows, run_meta=_meta(),
        rate_limits=_limits(queries_per_hour=None))

    windows = summary["observed_rate_windows"]
    assert windows["input_tokens_by_first_send"]["max"] == 600
    assert windows["input_tokens_by_first_send"]["coverage"] == 1.0
    assert windows[
        "offered_output_token_reservation_demand_by_first_send"]["max"] == 40
    comparison = summary["rate_limits"]["comparisons"][
        "input_tokens_per_minute"]
    assert comparison["utilization"] == 0.6
    assert comparison["status"] == \
        "run_evidence_below_warning_threshold"
    assert comparison["provider_headroom_established"] is False
    assert summary["rate_limits"]["configured"]["as_of"] == "2026-08-03"
    assert summary["rate_limits"]["warning"] is None

    markdown = render_markdown(summary, "quota evidence")
    html = render_html(summary, "quota evidence")
    assert "rolling rate-window evidence" in markdown
    assert "observed 600 / configured 1000.0" in markdown
    assert "ratio 60.0%" in markdown
    assert "operator reverified" in markdown
    assert date.today().isoformat() in markdown
    assert "Rolling rate windows" in html
    assert "60.0%" in html
    assert "operator reverified" in html


def test_near_limit_and_incomplete_usage_cannot_be_silent():
    near = summarize(
        [_row(0.0, 0), _row(61.0, 450), _row(62.0, 450)],
        run_meta=_meta(),
        rate_limits=_limits(queries_per_hour=None))
    comparison = near["rate_limits"]["comparisons"][
        "input_tokens_per_minute"]
    assert comparison["status"] == \
        "run_evidence_warning_threshold_reached"
    assert "90.0%" in near["rate_limits"]["warning"]

    incomplete = summarize(
        [_row(0.0, 0), _row(61.0, 450), _row(62.0, None)],
        run_meta=_meta(),
        rate_limits=_limits(queries_per_hour=None))
    comparison = incomplete["rate_limits"]["comparisons"][
        "input_tokens_per_minute"]
    assert comparison["status"] == "incomplete_run_evidence"
    assert comparison["comparison_is_complete"] is False
    assert "cannot establish headroom" in incomplete["rate_limits"]["warning"]


def test_protocol_corrupt_usage_cannot_complete_itpm_evidence():
    rows = [_row(0.0, 400), _row(61.0, 400)]
    rows[1]["parse_errors"] = 1

    summary = summarize(
        rows, run_meta=_meta(),
        rate_limits=_limits(output_tokens_per_minute=None,
                            queries_per_hour=None))

    observed = summary["observed_rate_windows"][
        "input_tokens_by_first_send"]
    comparison = summary["rate_limits"]["comparisons"][
        "input_tokens_per_minute"]
    assert observed["coverage"] == 0.5
    assert comparison["status"] == "incomplete_run_evidence"
    assert comparison["comparison_is_complete"] is False


def test_unexpected_returned_priority_tier_invalidates_standard_quota_model():
    row = _row(1.0, 100)
    row["service_tier"] = "priority"
    meta = _meta() | {
        "request_params": {"extra_body": {"service_tier": "default"}},
    }

    summary = summarize([row], run_meta=meta, rate_limits=_limits())

    tier = summary["observed_rate_windows"]["service_tier"]
    assert tier["configured"] == "default"
    assert tier["observed"] == ["priority"]
    assert tier["consistent_with_standard_pay_per_token"] is False
    assert all(
        item["status"] == "incomplete_run_evidence"
        for item in summary["rate_limits"]["comparisons"].values())
    assert "service tier was not exact default" in (
        summary["rate_limits"]["warning"] or "")


def test_retries_make_physical_attempt_timing_incomplete():
    summary = summarize(
        [_row(1.0, 100, attempts=2)], run_meta=_meta(),
        rate_limits=_limits())

    for name in ("input_tokens_per_minute", "output_tokens_per_minute",
                 "queries_per_hour"):
        assert summary["rate_limits"]["comparisons"][name]["status"] \
            == "incomplete_run_evidence"
    assert summary["observed_rate_windows"][
        "offered_output_token_reservation_demand_by_first_send"]["max"] == 40


def test_legacy_retry_count_is_grouped_but_never_declared_exact():
    row = _row(1.0, 100, reserved=200)
    row.pop("request_attempts")
    row["retries"] = 2

    summary = summarize([row], run_meta=_meta(), rate_limits=_limits())
    windows = summary["observed_rate_windows"]

    assert windows["physical_queries_by_first_send"]["max"] == 3
    assert windows[
        "offered_output_token_reservation_demand_by_first_send"]["max"] == 600
    assert windows["traffic_scope"]["attempt_count_unknown_rows"] == 1
    for comparison in summary["rate_limits"]["comparisons"].values():
        assert comparison["status"] == "incomplete_run_evidence"


def test_rate_limited_request_is_offered_demand_not_consumed_reservation():
    row = _row(1.0, None, reserved=1_000)
    row.update(ok=False, status=429, completion_tokens=None,
               visible_content_seen=False, stream_complete=False)
    limits = _limits(output_tokens_per_minute=500)

    summary = summarize([row], run_meta=_meta(), rate_limits=limits)
    windows = summary["observed_rate_windows"]

    assert windows[
        "offered_output_token_reservation_demand_by_first_send"]["max"] \
        == 1_000
    assert windows["actual_output_tokens_by_completion"]["max"] == 0
    comparison = summary["rate_limits"]["comparisons"][
        "output_tokens_per_minute"]
    assert comparison["status"] == \
        "run_evidence_at_or_above_nominal_limit"
    assert "offered to pre-admission" in comparison["qualifier"]
    rendered = render_markdown(summary, "rejected")
    assert "pre-admission demand, not observed provider consumption" in rendered


def test_missing_window_evidence_never_renders_as_zero():
    row = _row(1.0, 100)
    row["completion_tokens"] = None
    row["max_tokens_requested"] = None

    markdown = render_markdown(summarize([row]), "missing")
    html = render_html(summarize([row]), "missing")

    assert "offered output reservation demand: NOT REPORTED" in markdown
    assert "actual output tokens: NOT REPORTED" in markdown
    assert "offered output reservation demand: 0" not in markdown
    assert "n/a tok; pre-admission demand" in html


def test_quota_windows_can_include_setup_phases_without_polluting_sla():
    replay = _row(4.0, 100)
    replay["phase"] = "replay"
    prior = []
    for stamp, phase in ((1.0, "preflight"), (2.0, "sizing"),
                         (3.0, "calibration")):
        row = _row(stamp, 100)
        row["phase"] = phase
        prior.append(row)

    summary = summarize(
        [replay], run_meta=_meta(), rate_limits=_limits(),
        rate_limit_results=prior + [replay])

    assert summary["requests_total"] == 1
    windows = summary["observed_rate_windows"]
    assert windows["input_tokens_by_first_send"]["max"] == 400
    assert set(windows["traffic_scope"]["phases"]) == {
        "preflight", "sizing", "calibration", "replay"}


def test_short_qph_window_projects_sustained_demand_and_cannot_go_green():
    rows = [_row(index / 10.0, 10) for index in range(3_000)]
    _observed, block = _rate_limit_evidence(
        rows,
        _limits(input_tokens_per_minute=None,
                output_tokens_per_minute=None,
                queries_per_hour=7_200), _meta(),
    )

    comparison = block["comparisons"]["queries_per_hour"]
    assert comparison["observed_max"] == 3_000
    assert comparison["steady_state_projection"] > 35_000
    assert comparison["ratio_to_nominal_limit"] > 4.9
    assert comparison["status"] == \
        "short_observation_projection_at_or_above_warning"
    assert comparison["comparison_is_complete"] is False
    assert "cannot establish sustained quota headroom" in block["warning"]


def test_short_tpm_window_is_never_a_clean_headroom_conclusion():
    rows = [_row(0.0, 100), _row(10.0, 100)]

    _observed, block = _rate_limit_evidence(
        rows, _limits(output_tokens_per_minute=None,
                      queries_per_hour=None), _meta())

    comparison = block["comparisons"]["input_tokens_per_minute"]
    assert comparison["status"].startswith("short_observation")
    assert comparison["comparison_is_complete"] is False
    assert comparison["steady_state_projection"] == 1_200


def test_unknown_preflight_outcome_forces_every_comparison_incomplete():
    unknown = {
        "phase": "preflight", "request_id": "unknown",
        "first_send_unix": None, "request_attempts": None,
        "connection_attempts": None,
    }
    replay = _row(10.0, 100)
    replay["phase"] = "replay"

    observed, block = _rate_limit_evidence(
        [unknown, replay], _limits(), _meta())

    assert observed["traffic_scope"]["unknown_outcome_rows"] == 1
    assert observed["traffic_scope"]["phases"]["preflight"][
        "unknown_outcome_rows"] == 1
    assert all(item["status"] == "incomplete_run_evidence"
               for item in block["comparisons"].values())


def test_missing_endpoint_binding_forces_comparisons_incomplete():
    _observed, block = _rate_limit_evidence(
        [_row(1.0, 100)], _limits())

    assert block["binding"]["binding_complete"] is False
    assert all(item["status"] == "incomplete_run_evidence"
               for item in block["comparisons"].values())
    assert "could not be bound" in block["warning"]


def test_query_only_and_unmeasured_rate_warnings_render_in_html():
    limits = _limits(input_tokens_per_minute=None,
                     output_tokens_per_minute=None)
    summary = summarize(
        [], run_meta=_meta(), rate_limits=limits,
        rate_limit_results=[{
            "phase": "preflight", "request_id": "unknown",
            "first_send_unix": None, "request_attempts": None,
        }])

    html = render_html(summary, "query-only")

    assert "Rolling rate windows" in html
    assert "queries_per_hour" in html
    assert "NOT VERIFIED" not in html
    assert "could not be measured" in html


def test_rate_limit_metadata_cannot_inject_markdown_blocks():
    limits = _limits(
        scope="safe scope\n# FORGED GREEN VERDICT\n| bad | table |",
        note="note\n## forged")
    summary = summarize(
        [_row(1.0, 100)], run_meta=_meta(), rate_limits=limits)

    markdown = render_markdown(summary, "safe")

    assert "\n# FORGED GREEN VERDICT" not in markdown
    assert "safe scope # FORGED GREEN VERDICT" in markdown


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf"), True])
def test_rolling_peak_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="rolling window"):
        _rolling_peak([(1.0, 1.0)], window)


@pytest.mark.parametrize("stamp", [-1, "bad", True, 10 ** 400])
def test_prior_request_rows_reject_invalid_timestamps(stamp):
    with pytest.raises(ValueError, match="first_send_unix"):
        _prepare_prior_request_rows([{
            "phase": "preflight", "request_id": "bad-time",
            "first_send_unix": stamp,
        }])


def test_prior_request_rows_reject_nested_or_unknown_payload_fields():
    with pytest.raises(ValueError, match="unknown metadata field"):
        _prepare_prior_request_rows([{
            "phase": "preflight", "request_id": "private",
            "meta": {"prompt": "PRIVATE CUSTOMER PROMPT"},
        }])


@pytest.mark.parametrize("overrides,match", [
    ({"request_attempts": 0, "connection_attempts": 1,
      "first_send_unix": 1.0}, "zero request_attempts"),
    ({"request_attempts": 0, "connection_attempts": 1,
      "status": 200}, "zero request_attempts"),
    ({"request_attempts": 0, "connection_attempts": 1,
      "prompt_tokens": 5}, "zero request_attempts"),
    ({"request_attempts": 0, "connection_attempts": 1,
      "ok": True}, "zero request_attempts"),
    ({"request_attempts": 0, "connection_attempts": 1,
      "stream_complete": True}, "zero request_attempts"),
    ({"request_attempts": 2, "connection_attempts": 1},
     "cannot exceed connection_attempts"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 1.0}, "must include first_send_unix and t_send_unix"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 2.0, "t_send_unix": 1.0},
     "t_send_unix cannot precede"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 1.0, "t_send_unix": 2.0,
      "finished_unix": 1.5}, "finished_unix cannot precede t_send_unix"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 1.0, "t_send_unix": 1.0,
      "retries": 1, "retry_reasons": []},
     "retries must equal"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 1.0, "t_send_unix": 1.0,
      "cached_tokens": 6, "prompt_tokens": 5},
     "cached_tokens cannot exceed"),
    ({"request_attempts": 1, "connection_attempts": 1,
      "first_send_unix": 1.0, "t_send_unix": 1.0,
      "reasoning_tokens": 6, "completion_tokens": 5},
     "reasoning_tokens cannot exceed"),
])
def test_prior_request_rows_reject_cross_field_contradictions(
        overrides, match):
    row = {
        "phase": "preflight", "request_id": "contradiction",
        "max_tokens_requested": 10, "retries": 0, "retry_reasons": [],
        "first_attempt_unix": 0.5,
    }
    row.update(overrides)

    with pytest.raises(ValueError, match=match):
        _prepare_prior_request_rows([row])


def test_rate_limit_warning_downgrades_an_otherwise_green_verdict():
    summary = {
        "sla": {
            "ttft_definition": "first_content",
            "ttft_vs_target": [{
                "quantile": "p50", "target_ms": 100,
                "actual_ms": 10, "met": True,
            }],
            "ttfg_vs_target": [],
        },
        "ttft_ms": {"n": 20},
        "sample": {"n": 20, "indicative_only": ["p90", "p95", "p99"]},
        "drift": {"drift_kind": "stable"},
        "error_rate": 0.0,
        "rate_limits": {"warning": "input token warning threshold reached"},
    }

    kind, text = _verdict(summary)

    assert kind == "caution"
    assert "input token warning threshold reached" in text


@pytest.mark.parametrize("value,match", [
    ({}, "needs input_tokens_per_minute"),
    ({name: item for name, item in _limits().items() if name != "source"},
     "source must be a non-empty string"),
    (_limits(as_of="08/07/2026"), "as_of must be YYYY-MM-DD"),
    (_limits(as_of="9999-12-31"), "cannot be in the future"),
    (_limits(verified_at=None),
     "verified_at is required when snapshot freshness is set"),
    (_limits(max_age_days=None),
     "max_age_days is required when snapshot freshness is set"),
    (_limits(verified_at="08/07/2026"),
     "verified_at must be YYYY-MM-DD"),
    (_limits(verified_at="9999-12-31"),
     "verified_at cannot be in the future"),
    (_limits(max_age_days=0), "max_age_days must be a positive integer"),
    (_limits(max_age_days=1.5), "max_age_days must be a positive integer"),
    (_limits(max_age_days=True), "max_age_days must be a positive integer"),
    (_limits(warning_utilization=1.1),
     "warning_utilization must be at most 1"),
    (_limits(input_tokens_per_minute=True),
     "input_tokens_per_minute must be a number"),
    (_limits(input_tokens_per_minute=10 ** 400), "finite number"),
    (_limits(source="quota page"), "source must be an https URL"),
    (_limits(provider="openai"), "provider must be 'databricks'"),
    (_limits(deployment_mode="provisioned"),
     "deployment_mode must be 'pay_per_token'"),
    (_limits(accounting_model="guess"), "accounting_model must be"),
    (_limits(typo=1), "unknown field"),
])
def test_rate_limit_config_is_strict(value, match):
    with pytest.raises(ValueError, match=match):
        validate_rate_limits(value)


def test_run_config_preserves_valid_rate_limit_snapshot(tmp_path):
    config = RunConfig(
        endpoint={
            "base_url": "https://unit-test.cloud.databricks.com",
            "path": ("/serving-endpoints/databricks-glm-5-2/"
                     "invocations"),
        },
        prompts_file=str(tmp_path / "prompts.jsonl"),
        rate_limits=_limits(),
    )

    assert config.rate_limits["input_tokens_per_minute"] == 1_000
    assert config.rate_limits["provider"] == "databricks"


@pytest.mark.parametrize("extra_body", [None, {}, {"service_tier": "default"}])
def test_standard_rate_limits_allow_only_absent_or_default_tier(
        tmp_path, extra_body):
    endpoint = {
        "base_url": "https://unit-test.cloud.databricks.com",
        "path": "/serving-endpoints/databricks-glm-5-2/invocations",
    }
    if extra_body is not None:
        endpoint["extra_body"] = extra_body
    config = RunConfig(
        endpoint=endpoint,
        prompts_file=str(tmp_path / "prompts.jsonl"),
        rate_limits=_limits())
    assert (config.endpoint.get("extra_body") or {}).get(
        "service_tier", "default") == "default"


@pytest.mark.parametrize("tier", ["priority", "auto", "DEFAULT", 1, True,
                                   None, {"name": "default"}])
def test_standard_rate_limits_reject_nondefault_service_tier_before_io(
        tmp_path, tier):
    with pytest.raises(ValueError, match="service_tier must be absent or"):
        RunConfig(
            endpoint={
                "base_url": "https://unit-test.cloud.databricks.com",
                "path": "/serving-endpoints/databricks-glm-5-2/invocations",
                "extra_body": {"service_tier": tier},
            },
            prompts_file=str(tmp_path / "prompts.jsonl"),
            rate_limits=_limits())


@pytest.mark.parametrize("endpoint,pricing,capture,match", [
    ({"base_url": "https://api.openai.com",
      "path": "/serving-endpoints/databricks-glm-5-2/invocations"},
     None, True, "Databricks workspace host"),
    ({"base_url": "https://unit-test.cloud.databricks.com",
      "path": "/serving-endpoints/other-model/invocations"},
     None, True, "must match the serving endpoint name"),
    ({"base_url": "https://unit-test.cloud.databricks.com",
      "path": "/serving-endpoints/databricks-glm-5-2/invocations"},
     {"mode": "provisioned", "dbu_per_hour": 1}, True,
     "cannot be combined with provisioned pricing"),
    ({"base_url": "https://unit-test.cloud.databricks.com",
      "path": "/serving-endpoints/databricks-glm-5-2/invocations"},
     None, False, "capture_endpoint_metadata=true"),
])
def test_rate_limit_snapshot_is_bound_to_target_mode(
        tmp_path, endpoint, pricing, capture, match):
    with pytest.raises(ValueError, match=match):
        RunConfig(
            endpoint=endpoint, prompts_file=str(tmp_path / "prompts.jsonl"),
            pricing=pricing, capture_endpoint_metadata=capture,
            rate_limits=_limits())


def test_new_rate_limits_argument_does_not_break_positional_concurrency():
    rows = [_row(float(index), 10) for index in range(20)]

    summary = summarize(
        rows, None, None, None, "first_content", None, 7)

    assert summary["concurrency"]["sizing_concurrency_requested"] == 7
    assert "rate_limits" not in summary
