"""DBU cost from endpoint-reported tokens and user-supplied rates, plus the
stream-counted reasoning fallback. Rates are never fetched, so the math is
what gets tested, against the Databricks pricing model (per-token DBU/M and
provisioned DBU/hour)."""
from __future__ import annotations

from traffic_replay.metrics import _cost_block, render_html, summarize


def _rows(pt, ct, comp, n=1):
    return [{"ok": True, "prompt_tokens": pt, "cached_tokens": ct,
             "completion_tokens": comp} for _ in range(n)]


def test_per_token_dbu_math():
    ok = [{"prompt_tokens": 10000, "cached_tokens": 6000,
           "completion_tokens": 100}]
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


def test_cache_read_defaults_to_input_rate():
    ok = [{"prompt_tokens": 1000, "cached_tokens": 400, "completion_tokens": 0}]
    c = _cost_block(ok, dur=60, in_tok=1000, out_tok=0, cached_tok=400,
                    pricing={"mode": "per_token", "input_dbu_per_m": 10.0,
                             "output_dbu_per_m": 30.0})
    # no cache rate -> cached billed at input rate -> all 1000 at 10/M
    assert abs(c["dbu_total"] - 1000 / 1e6 * 10) < 1e-9
    assert c["cache_dbu_saved"] == 0.0


def test_provisioned_effective_rate():
    c = _cost_block([], dur=3600, in_tok=18000, out_tok=150, cached_tok=0,
                    pricing={"mode": "provisioned", "dbu_per_hour": 85.714,
                             "usd_per_dbu": 0.07})
    # 18150 tokens in 1 hour -> eff = 85.714 / (18150/1e6)
    assert abs(c["effective_dbu_per_1m_tokens"] - 85.714 / (18150 / 1e6)) < 1e-6
    assert abs(c["effective_usd_per_1m_tokens"]
               - c["effective_dbu_per_1m_tokens"] * 0.07) < 1e-6


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
    assert s["throughput"]["reasoning_stream_deltas_per_min"] > 0
    report = render_html(s, "reasoning chunks")
    assert "Reasoning stream deltas" in report
    assert "These are SSE chunks, not tokens" in report


def test_missing_usage_makes_full_run_cost_unavailable_not_zero():
    rows = _rows(1000, 400, 50, n=2)
    rows.append({"ok": True, "prompt_tokens": None, "cached_tokens": None,
                 "completion_tokens": None})
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


def test_cost_card_in_html():
    ok = [{"ok": True, "t_send_unix": 0.0, "prompt_tokens": 1000,
           "cached_tokens": 0, "completion_tokens": 100,
           "dispatch_lag_ms": 0.0}]
    s = summarize(ok, pricing={"mode": "per_token", "input_dbu_per_m": 20.0,
                               "output_dbu_per_m": 60.0, "usd_per_dbu": 0.07})
    h = render_html(s, "cost run")
    assert "Cost (Databricks DBUs)" in h
    assert "DBU per request" in h
    assert "cache DBUs saved" in h
    assert "$" in h  # usd shown when usd_per_dbu given


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
    assert "no successful requests to price" in md
    assert "no successful requests to price" in h
    assert h.startswith("<!doctype html>")
