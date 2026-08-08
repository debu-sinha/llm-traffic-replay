"""HTTP failures need stable aggregates and capacity-safe 429 policy."""
from __future__ import annotations

from traffic_replay.metrics import (_verdict, render_html, render_markdown,
                                    summarize)


def _row(i: int, *, status: int | None = 200, ok: bool = True,
         phase: str = "replay", error: str | None = None) -> dict:
    stamp = 1_700_000_000.0 + i
    return {
        "request_id": f"request-{i}",
        "phase": phase,
        "scheduled_s": float(i),
        "dispatch_lag_ms": 0.0,
        "first_send_unix": stamp,
        "t_send_unix": stamp,
        "finished_unix": stamp + 0.1,
        "queue_wait_ms": 0.0,
        "status": status,
        "ok": ok,
        "error": error,
        "ttfb_ms": 5.0 if ok else None,
        "ttft_ms": 10.0 if ok else None,
        "e2e_ms": 100.0 if ok else None,
        "caller_ttft_ms": 10.0 if ok else None,
        "caller_e2e_ms": 100.0 if ok else None,
        "prompt_tokens": 100 if ok else None,
        "completion_tokens": 10 if ok else None,
        "max_tokens_requested": 20,
        "request_attempts": 1,
        "retries": 0,
        "visible_content_seen": ok,
        "valid_tool_calls": 0,
        "stream_complete": ok,
        "parse_errors": 0,
        "finish_reason": "stop" if ok else None,
        "intended_input_tokens": 100,
        "intended_output_tokens": 10,
        "intended_cache_fraction": None,
        "cached_tokens": None,
    }


def test_http_status_counts_are_stable_across_varying_429_body_digests():
    rows = [
        _row(0),
        _row(1, status=429, ok=False,
             error="http 429 (body sample bytes=31, sha256=aaaaaaaaaaaaaaaa)"),
        _row(2, status=429, ok=False,
             error="http 429 (body sample bytes=48, sha256=bbbbbbbbbbbbbbbb)"),
        _row(3, status=500, ok=False,
             error="http 500 (body sample bytes=8, sha256=cccccccccccccccc)"),
        _row(4, status=None, ok=False, error="TimeoutError"),
    ]

    summary = summarize(rows)

    assert summary["failures_by_http_status"] == {"429": 2, "500": 1}
    assert summary["failures_by_error"]["http 429 (rate limited)"] == 2
    assert not any("aaaaaaaa" in key or "bbbbbbbb" in key
                   for key in summary["failures_by_error"])
    assert summary["http_429_count"] == 2
    assert summary["http_429_rate"] == 0.4
    assert summary["http_429"]["http_status_observed_for"] == 4
    assert summary["http_429"]["phases"] == {"replay": 2}
    assert summary["quota_limited"] is True


def test_one_429_invalidates_an_otherwise_green_capacity_interpretation():
    summary = {
        "sla": {
            "ttft_vs_target": [{
                "quantile": "p50", "target_ms": 100,
                "actual_ms": 10, "met": True,
            }],
            "ttfg_vs_target": [],
        },
        "ttft_ms": {"n": 200},
        "sample": {"n": 200, "indicative_only": ["p99"]},
        "drift": {"drift_kind": "stable"},
    }
    assert _verdict(summary) == ("ok", "meets every acceptance target")

    summary.update({
        "http_429_count": 1,
        "http_429": {"request_rows_examined": 200},
    })
    kind, text = _verdict(summary)

    assert kind == "invalid"
    assert "quota-limited" in text
    assert "no endpoint-capacity conclusion" in text
    assert "provider telemetry" in text


def test_setup_phase_429_cannot_be_hidden_by_a_clean_replay():
    replay = [_row(i) for i in range(3)]
    preflight_429 = _row(
        10, status=429, ok=False, phase="preflight",
        error="http 429 (body sample bytes=1, sha256=dddddddddddddddd)")

    summary = summarize(
        replay, rate_limit_results=[preflight_429, *replay])

    # The replay error counters retain their documented replay population,
    # while the capacity gate covers all supplied request phases.
    assert summary["failures_by_http_status"] == {}
    assert summary["requests_failed"] == 0
    assert summary["http_429_count"] == 1
    assert summary["http_429_rate"] == 0.25
    assert summary["http_429"]["scope"] == "all supplied request phases"
    assert summary["http_429"]["phases"] == {"preflight": 1}
    assert _verdict(summary)[0] == "invalid"


def test_both_reports_put_quota_limiting_and_status_counts_in_plain_view():
    rows = [
        _row(0),
        _row(1, status=429, ok=False,
             error="http 429 (body sample bytes=4, sha256=eeeeeeeeeeeeeeee)"),
    ]
    summary = summarize(rows)

    markdown = render_markdown(summary, "rate limited")
    html = render_html(summary, "rate limited")

    for rendered in (markdown, html):
        lowered = rendered.lower()
        assert "quota-limited" in lowered
        assert "no endpoint-capacity conclusion" in lowered
        assert "http 429" in lowered
        assert "failed requests by http status" in lowered
        assert "provider telemetry" in lowered
    assert '{"429": 1}' in markdown
    assert "429&quot;: 1" in html


def test_loose_status_values_cannot_forge_http_429_evidence():
    rows = [
        _row(0, status="429", ok=False, error="untyped status"),  # type: ignore[arg-type]
        _row(1, status=True, ok=False, error="boolean status"),  # type: ignore[arg-type]
    ]

    summary = summarize(rows)

    assert summary["failures_by_http_status"] == {}
    assert summary["http_429_count"] == 0
    assert summary["quota_limited"] is False
