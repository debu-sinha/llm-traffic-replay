"""SLA scorecard: targets from the profile config are scored against
measured percentiles, hard timeouts count as failures, and the report
renders the verdicts."""
from traffic_replay.metrics import render_markdown, summarize


def _row(i, ttft, e2e, ok=True, prompt=1000, comp=50, inter=5.0):
    return {
        "request_id": f"r{i}", "scheduled_s": float(i),
        "dispatch_lag_ms": 1.0, "t_send_unix": 1000.0 + i,
        "ttfb_ms": ttft - 5 if ttft else None, "ttft_ms": ttft,
        "e2e_ms": e2e, "status": 200 if ok else 500, "ok": ok,
        "error": None if ok else "http 500", "content_chunks": comp,
        "interchunk_max_ms": inter, "finish_reason": "stop" if ok else None,
        "prompt_tokens": prompt if ok else None,
        "completion_tokens": comp if ok else None,
        "cached_tokens": None, "cached_tokens_source": None,
        "intended_input_tokens": prompt, "intended_output_tokens": comp,
        "intended_cache_fraction": 0.6, "doc_id": 1, "chars_sent": 4000,
        "retries": 0, "phase": "replay",
    }


ACCEPT = {
    "ttft_ms": {"p50": 500, "p95": 900},
    "ttfg_ms": {"p50": 700, "p95": 1500},
    "hard_timeouts": {"ttft_s": 15, "ttfg_s": 45},
    "success_rate": 0.99,
}


def test_targets_met_and_missed_are_scored():
    # 100 requests: ttft 400ms flat (meets 500/900), e2e 2000ms flat
    # (misses both 700 and 1500)
    rows = [_row(i, 400.0, 2000.0) for i in range(100)]
    s = summarize(rows, acceptance=ACCEPT)
    ttft = {r["quantile"]: r for r in s["sla"]["ttft_vs_target"]}
    ttfg = {r["quantile"]: r for r in s["sla"]["ttfg_vs_target"]}
    assert ttft["p50"]["met"] is True and ttft["p95"]["met"] is True
    assert ttfg["p50"]["met"] is False and ttfg["p95"]["met"] is False
    report = render_markdown(s, "t")
    assert "SLA scorecard" in report
    assert "| TTFG | p50 | 700 | 2000.0 | NO |" in report


def test_hard_timeout_counts_against_success_rate():
    rows = [_row(i, 400.0, 800.0) for i in range(99)]
    rows.append(_row(99, 16_000.0, 20_000.0))  # ttft over the 15s hard cap
    s = summarize(rows, acceptance=ACCEPT)
    assert s["sla"]["hard_timeout_breaches"] == 1
    sr = s["sla"]["success_rate"]
    assert sr["actual"] == 0.99 and sr["met"] is True
    # one more breach pushes below the 0.99 bar
    rows.append(_row(100, 16_000.0, 20_000.0))
    s2 = summarize(rows, acceptance=ACCEPT)
    assert s2["sla"]["success_rate"]["met"] is False


def test_interchunk_and_throughput_present():
    rows = [_row(i, 400.0, 800.0, inter=7.5) for i in range(50)]
    s = summarize(rows)
    assert s["interchunk_max_ms"]["n"] == 50
    assert abs(s["interchunk_max_ms"]["p50"] - 7.5) < 1e-9
    assert s["throughput"]["input_tokens_per_min"] > 0
    report = render_markdown(s, "t")
    assert "interchunk max" in report and "tokens/min" in report


def test_no_acceptance_no_sla_section():
    rows = [_row(i, 400.0, 800.0) for i in range(10)]
    s = summarize(rows)
    assert "sla" not in s
    assert "SLA scorecard" not in render_markdown(s, "t")
