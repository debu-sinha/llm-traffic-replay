"""Diagnostic arithmetic from clean usage and unverified supplied rates."""
from __future__ import annotations

import pytest

from traffic_replay.metrics import (
    _cost_block, render_html, render_markdown, summarize,
)


def _rows(pt, ct, comp, n=1):
    return [{"ok": True, "prompt_tokens": pt, "cached_tokens": ct,
             "completion_tokens": comp, "stream_complete": True,
             "parse_errors": 0, "connection_attempts": 1,
             "request_attempts": 1, "retries": 0,
             "retry_reasons": []} for _ in range(n)]


def test_per_token_dbu_math():
    ok = _rows(10000, 6000, 100)
    c = _cost_block(ok, dur=60, in_tok=10000, out_tok=100, cached_tok=6000,
                    pricing={"mode": "per_token", "input_dbu_per_m": 20.0,
                             "output_dbu_per_m": 62.857,
                             "cache_read_dbu_per_m": 2.0, "usd_per_dbu": 0.07})
    # 4000 uncached*20/M + 6000 cached*2/M + 100 out*62.857/M
    expect = 4000 / 1e6 * 20 + 6000 / 1e6 * 2 + 100 / 1e6 * 62.857
    assert abs(c["dbu_total"] - expect) < 1e-9
    assert abs(c["cache_dbu_saved"] - 6000 / 1e6 * (20 - 2)) < 1e-9
    assert abs(c["usd_total"] - expect * 0.07) < 1e-9
    assert c["rates_dbu_per_m"]["cache_read"] == 2.0
    assert c["provenance_verified"] is False
    assert "not a current Databricks price" in c["applicability_warning"]


def test_cache_read_defaults_to_input_rate():
    ok = _rows(1000, 400, 0)
    c = _cost_block(ok, dur=60, in_tok=1000, out_tok=0, cached_tok=400,
                    pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                             "output_dbu_per_m": 30.0})
    # no cache rate -> cached billed at input rate -> all 1000 at 10/M
    assert abs(c["dbu_total"] - 1000 / 1e6 * 10) < 1e-9
    assert c["cache_dbu_saved"] == 0.0


def test_provisioned_effective_rate():
    rows = _rows(18000, 0, 150)
    c = _cost_block(rows, dur=3600, in_tok=18000, out_tok=150, cached_tok=0,
                    pricing={"mode": "provisioned", "dbu_per_hour": 85.714,
                             "usd_per_dbu": 0.07})
    # 18150 tokens in 1 hour -> eff = 85.714 / (18150/1e6)
    assert abs(c["effective_dbu_per_1m_tokens"] - 85.714 / (18150 / 1e6)) < 1e-6
    assert abs(c["effective_usd_per_1m_tokens"]
               - c["effective_dbu_per_1m_tokens"] * 0.07) < 1e-6
    assert c["complete"] is True
    assert c["coverage_warning"] is None
    assert c["tokens_measured"] == 18150


@pytest.mark.parametrize(
    ("attempts", "retry_reasons", "ambiguity_field"),
    [
        (2, [], "ambiguous_retry_rows"),
        (1, ["transport_error_after_post"], "ambiguous_retry_rows"),
        (0, ["transport_error_after_post"], "ambiguous_retry_rows"),
        (None, [], "unknown_attempt_rows"),
        (True, [], "unknown_attempt_rows"),
        (-1, [], "unknown_attempt_rows"),
    ],
)
def test_provisioned_withholds_effective_rate_when_attempts_are_ambiguous(
        attempts, retry_reasons, ambiguity_field):
    rows = _rows(18000, 0, 150)
    rows[0]["request_attempts"] = attempts
    rows[0]["retry_reasons"] = retry_reasons
    rows[0]["retries"] = len(retry_reasons)
    if isinstance(attempts, int) and not isinstance(attempts, bool) \
            and attempts >= 0:
        rows[0]["connection_attempts"] = max(attempts, 1)

    c = _cost_block(
        rows, dur=3600, in_tok=18000, out_tok=150, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 85.714,
                 "usd_per_dbu": 0.07})

    # Clean final-response usage is not evidence for an earlier physical POST.
    assert c["usage_coverage"] == 1.0
    assert c["complete"] is False
    assert c["coverage"] == 0.0
    assert c[ambiguity_field] == 1
    assert c["tokens_measured"] is None
    assert c["tokens_measured_subset"] == 18150
    assert c["effective_dbu_per_1m_tokens"] is None
    assert c["effective_usd_per_1m_tokens"] is None
    assert "token-throughput denominator" in c["coverage_warning"]


def test_provisioned_known_unsent_row_keeps_exact_denominator():
    rows = _rows(18000, 0, 150)
    rows.append({
        "ok": False, "error": "cancelled before HTTP POST",
        "connection_attempts": 0, "request_attempts": 0,
        "retries": 0, "retry_reasons": [],
        "prompt_tokens": None, "completion_tokens": None,
    })

    c = _cost_block(
        rows, dur=3600, in_tok=18000, out_tok=150, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 85.714})

    assert c["complete"] is True
    assert c["coverage"] == 1.0
    assert c["known_unsent_rows"] == 1
    assert c["tokens_measured"] == 18150
    assert c["effective_dbu_per_1m_tokens"] is not None


def test_summarize_cost_covers_known_unsent_scheduled_tail():
    rows = _rows(1000, 0, 100, n=1)
    rows[0].update({
        "scheduled_s": 0.0,
        "first_send_unix": 1_700_000_000.0,
        "finished_unix": 1_700_000_001.0,
    })
    for i in range(1, 100):
        rows.append({
            "ok": False,
            "error": "cancelled before HTTP POST",
            "scheduled_s": i / 10.0,
            "connection_attempts": 0,
            "request_attempts": 0,
            "retries": 0,
            "retry_reasons": [],
            "prompt_tokens": None,
            "completion_tokens": None,
        })

    summary = summarize(
        rows,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0},
    )
    cost = summary["cost"]
    assert cost["complete"] is True
    assert cost["known_unsent_rows"] == 99
    assert cost["observation_seconds"] == pytest.approx(9.9)
    assert cost["duration_basis"] == (
        "max(logical_schedule_span,response_drain)")
    # The old one-second sent-prefix duration under-reported effective cost
    # by almost tenfold.
    expected = 10.0 / ((1100 / 9.9 * 3600.0) / 1e6)
    assert cost["effective_dbu_per_1m_tokens"] == pytest.approx(expected)


def test_pre_post_connection_retry_keeps_exact_cost_accounting():
    rows = _rows(1000, 400, 50)
    rows[0].update({
        "connection_attempts": 2,
        "request_attempts": 1,
        "retries": 1,
        "retry_reasons": ["connection_error_before_post"],
    })
    pricing = {
        "mode": "per_token",
        "input_dbu_per_m": 10.0,
        "output_dbu_per_m": 30.0,
        "cache_read_dbu_per_m": 2.0,
    }
    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=400,
        pricing=pricing,
    )

    expected = 600 / 1e6 * 10 + 400 / 1e6 * 2 + 50 / 1e6 * 30
    assert c["ambiguous_retry_rows"] == 0
    assert c["unknown_attempt_rows"] == 0
    assert c["coverage"] == 1.0
    assert c["complete"] is True
    assert c["dbu_total"] == pytest.approx(expected)


def test_pre_post_connection_retry_with_zero_posts_is_known_unsent():
    rows = [{
        "ok": False,
        "error": "connection failed before HTTP POST",
        "connection_attempts": 2,
        "request_attempts": 0,
        "retries": 1,
        "retry_reasons": ["connection_error_before_post"],
        "prompt_tokens": None,
        "completion_tokens": None,
    }]
    c = _cost_block(
        rows, dur=60, in_tok=0, out_tok=0, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0},
    )

    assert c["known_unsent_rows"] == 1
    assert c["ambiguous_retry_rows"] == 0
    assert c["unknown_attempt_rows"] == 0
    assert c["coverage"] == 1.0
    assert c["complete"] is True


def test_zero_attempts_with_response_evidence_is_not_treated_as_unsent():
    rows = _rows(1000, 0, 50)
    rows[0]["request_attempts"] = 0

    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["known_unsent_rows"] == 0
    assert c["unknown_attempt_rows"] == 1
    assert c["complete"] is False
    assert c["effective_dbu_per_1m_tokens"] is None


def test_zero_attempts_with_contradictory_retry_count_is_unknown_not_unsent():
    rows = [{
        "ok": False, "error": "cancelled before HTTP POST",
        "connection_attempts": 1, "request_attempts": 0,
        "retries": 1, "retry_reasons": [],
        "prompt_tokens": None, "completion_tokens": None,
    }]

    c = _cost_block(
        rows, dur=60, in_tok=0, out_tok=0, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["known_unsent_rows"] == 0
    assert c["unknown_attempt_rows"] == 1
    assert c["ambiguous_retry_rows"] == 0
    assert c["complete"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"retries": 1, "retry_reasons": []},
        {"retries": 0, "retry_reasons": ["transport_error_after_post"]},
        {"retries": True, "retry_reasons": []},
        {"retries": -1, "retry_reasons": []},
        {"retries": 0, "retry_reasons": "transport_error_after_post"},
        {"retries": 0, "retry_reasons": [""]},
    ],
)
def test_malformed_or_mismatched_retry_metadata_is_unknown(metadata):
    rows = _rows(1000, 0, 50)
    rows[0].update(metadata)

    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["unknown_attempt_rows"] == 1
    assert c["ambiguous_retry_rows"] == 0
    assert c["complete"] is False
    assert c["effective_dbu_per_1m_tokens"] is None


@pytest.mark.parametrize("missing", ["retries", "retry_reasons"])
def test_partial_retry_metadata_is_unknown(missing):
    rows = _rows(1000, 0, 50)
    rows[0].pop(missing)

    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["unknown_attempt_rows"] == 1
    assert c["complete"] is False
    assert c["effective_dbu_per_1m_tokens"] is None


@pytest.mark.parametrize("connections", [0, -1, True, 1.5, None])
def test_invalid_or_too_small_connection_attempt_count_is_unknown(connections):
    rows = _rows(1000, 0, 50)
    rows[0]["connection_attempts"] = connections

    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["unknown_attempt_rows"] == 1
    assert c["complete"] is False
    assert c["effective_dbu_per_1m_tokens"] is None


def test_provisioned_missing_usage_withholds_token_denominator():
    rows = _rows(1000, 0, 50)
    rows[0]["prompt_tokens"] = None

    c = _cost_block(
        rows, dur=60, in_tok=0, out_tok=50, cached_tok=0,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    assert c["usage_coverage"] == 0.0
    assert c["exact_single_usage_rows"] == 0
    assert c["complete"] is False
    assert c["tokens_measured"] is None
    assert c["effective_dbu_per_1m_tokens"] is None
    assert "single-POST row" in c["coverage_warning"]


def test_provisioned_reports_do_not_render_effective_rate_on_ambiguous_retry():
    rows = _rows(1000, 0, 50)
    rows[0].update({
        "connection_attempts": 2,
        "request_attempts": 2,
        "retries": 1,
        "retry_reasons": ["transport_error_after_post"],
        "t_send_unix": 1_700_000_000.0,
        "first_send_unix": 1_700_000_000.0,
        "finished_unix": 1_700_000_060.0,
    })
    summary = summarize(
        rows,
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0,
                 "usd_per_dbu": 0.07})

    markdown = render_markdown(summary, "ambiguous provisioned cost")
    html = render_html(summary, "ambiguous provisioned cost")

    assert "effective cost per 1M tokens unavailable" in markdown
    assert "at the measured throughput" not in markdown
    assert "Effective cost per 1M tokens is unavailable" in html
    assert "effective cost per 1M tokens</th>" not in html
    assert "Cost coverage" in html


def test_cost_errors_are_reported_not_raised():
    assert "error" in _cost_block([], 60, 0, 0, 0, {"mode": "per_token"})
    assert "error" in _cost_block([], 60, 0, 0, 0, {"mode": "provisioned"})


def test_stream_counted_reasoning_fallback():
    # usage reports NO reasoning_tokens, but the stream had reasoning deltas
    ok = [{"ok": True, "t_send_unix": 0.0, "prompt_tokens": 100,
           "completion_tokens": 10, "reasoning_chunks": 12,
           "reasoning_tokens": None, "dispatch_lag_ms": 0.0},
          {"ok": True, "t_send_unix": 1.0, "prompt_tokens": 100,
           "completion_tokens": 10, "reasoning_chunks": 8,
           "reasoning_tokens": None, "dispatch_lag_ms": 0.0}]
    s = summarize(ok)
    assert "reasoning_tokens_total" not in s
    assert "reasoning_tokens_per_min" not in s["throughput"]
    assert s["reasoning_stream_deltas_total"] == 20
    assert "not token counts" in s["reasoning_stream_deltas_source"]
    assert "reasoning_stream_deltas_per_min" not in s["throughput"]
    assert "completion time" in s["throughput"]["coverage_warning"]
    report = render_html(s, "reasoning chunks")
    assert "Reasoning stream deltas" in report
    assert "These are SSE chunks, not tokens" in report


def test_missing_usage_makes_full_run_cost_unavailable_not_zero():
    rows = _rows(1000, 400, 50, n=2)
    rows.append({"ok": True, "prompt_tokens": None, "cached_tokens": None,
                 "completion_tokens": None, "stream_complete": True,
                 "parse_errors": 0, "connection_attempts": 1,
                 "request_attempts": 1, "retries": 0,
                 "retry_reasons": []})
    c = _cost_block(
        rows, dur=60, in_tok=2000, out_tok=100, cached_tok=800,
        pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                 "output_dbu_per_m": 30.0, "cache_read_dbu_per_m": 2.0})
    assert c["coverage"] == 2 / 3
    assert c["dbu_total"] is None
    assert c["dbu_per_1k_requests"] is None
    assert c["dbu_per_min"] is None
    assert c["coverage_warning"]
    assert c["dbu_total_measured_subset"] > 0


def test_cached_tokens_above_prompt_tokens_invalidate_full_cost():
    rows = _rows(1000, 400, 50, n=1)
    rows.append({"ok": True, "prompt_tokens": 100,
                 "cached_tokens": 101, "completion_tokens": 5})
    c = _cost_block(
        rows, dur=60, in_tok=1100, out_tok=55, cached_tok=501,
        pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                 "output_dbu_per_m": 30.0,
                 "cache_read_dbu_per_m": 2.0})
    assert c["priced_rows"] == 1
    assert c["dbu_total"] is None
    assert c["coverage_warning"]


def test_cost_card_in_html():
    ok = [{"ok": True, "t_send_unix": 0.0, "prompt_tokens": 1000,
           "cached_tokens": 0, "completion_tokens": 100,
           "dispatch_lag_ms": 0.0, "stream_complete": True,
           "parse_errors": 0, "connection_attempts": 1,
           "request_attempts": 1, "retries": 0,
           "retry_reasons": []}]
    s = summarize(ok, pricing={"mode": "per_token", "input_dbu_per_m": 20.0,
                               "output_dbu_per_m": 60.0, "usd_per_dbu": 0.07})
    h = render_html(s, "cost run")
    assert "Unverified user-supplied rate arithmetic" in h
    assert "DBU per request" in h
    assert "cache DBUs saved" in h
    assert "$" in h  # usd shown when usd_per_dbu given
    assert "not a current Databricks price" in h


def test_cost_renders_when_all_requests_failed():
    # a load tester will be pointed at dead/misauthed endpoints; with pricing
    # set, the report must still render, not crash on the empty cost figures
    from traffic_replay.metrics import render_markdown, render_html
    failed = [{"ok": False, "error": "http 500", "t_send_unix": 0.0,
               "dispatch_lag_ms": 0.0},
              {"ok": False, "error": "http 500", "t_send_unix": 1.0,
               "dispatch_lag_ms": 0.0}]
    s = summarize(failed, pricing={"mode": "per_token", "input_dbu_per_m": 20.0,
                                   "output_dbu_per_m": 60.0, "usd_per_dbu": 0.07})
    md = render_markdown(s, "all failed")
    h = render_html(s, "all failed")
    assert "aggregate replay total unavailable" in md
    assert "Aggregate replay total is unavailable" in h
    assert h.startswith("<!doctype html>")


def test_after_post_retry_withholds_aggregate_cost_even_with_final_usage():
    rows = _rows(1000, 400, 50)
    rows[0]["request_attempts"] = 2
    rows[0]["connection_attempts"] = 2
    rows[0]["retries"] = 1
    rows[0]["retry_reasons"] = ["transport_error_after_post"]
    c = _cost_block(
        rows, dur=60, in_tok=1000, out_tok=50, cached_tok=400,
        pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                 "output_dbu_per_m": 30.0, "cache_read_dbu_per_m": 2.0})
    assert c["complete"] is False
    assert c["dbu_total"] is None
    assert c["dbu_per_min"] is None
    assert c["ambiguous_retry_rows"] == 1
    assert "earlier billed usage is not observed" in c["coverage_warning"]


def test_corrupt_or_incomplete_usage_is_diagnostic_only():
    rows = _rows(1000, 400, 50, n=2)
    rows[1]["parse_errors"] = 1
    rows[1]["stream_complete"] = False
    c = _cost_block(
        rows, dur=60, in_tok=2000, out_tok=100, cached_tok=800,
        pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                 "output_dbu_per_m": 30.0, "cache_read_dbu_per_m": 2.0})
    assert c["priced_rows"] == 1
    assert c["successful_rows"] == 1
    assert c["coverage"] == 0.5
    assert c["dbu_total"] is None


def test_summarize_excludes_corrupt_usage_from_throughput_and_reasoning():
    rows = _rows(100, 0, 10, n=100)
    for i, row in enumerate(rows):
        row.update({"first_send_unix": float(i),
                    "finished_unix": float(i) + 1.0,
                    "reasoning_tokens": 5,
                    "reasoning_tokens_source": "usage.output_token_details"})
    rows[-1]["parse_errors"] = 1
    summary = summarize(rows)
    assert summary["throughput"]["usage_coverage"] == 0.99
    assert summary["reasoning_tokens_total"] == 99 * 5
    assert "99 of 100 attempted requests" in (
        summary["throughput"]["coverage_warning"] or "")
