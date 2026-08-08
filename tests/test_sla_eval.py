"""Acceptance scorecard: targets from the profile config are scored against
measured percentiles, hard timeouts count as failures, and the report
renders the verdicts."""
from traffic_replay.metrics import _verdict, render_markdown, summarize


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
    assert "Acceptance scorecard" in report
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


def test_hard_ttft_timeout_uses_the_configured_first_visible_definition():
    rows = [_row(i, 100.0, 21_000.0) for i in range(20)]
    for row in rows:
        row.update({"ttfv_ms": 20_000.0, "visible_content_seen": True,
                    "stream_complete": True, "parse_errors": 0})
    acceptance = {"hard_timeouts": {"ttft_s": 15}}
    visible = summarize(rows, acceptance=acceptance,
                        ttft_definition="first_visible")
    content = summarize(rows, acceptance=acceptance,
                        ttft_definition="first_content")
    assert visible["sla"]["hard_timeout_breaches"] == 20
    assert visible["sla"]["hard_timeout_basis"]["ttft_metric"] == "ttfv_ms"
    assert content["sla"]["hard_timeout_breaches"] == 0


def test_missing_first_visible_event_breaches_a_first_visible_hard_cap():
    rows = [_row(i, 100.0, 1000.0) for i in range(10)]
    for row in rows:
        row.update({"ttfv_ms": None, "visible_content_seen": False,
                    "stream_complete": True, "parse_errors": 0})
    s = summarize(rows, acceptance={"hard_timeouts": {"ttft_s": 15}},
                  ttft_definition="first_visible")
    assert s["sla"]["hard_timeout_breaches"] == 10


def test_hard_caps_include_client_queue_wait():
    rows = [_row(i, 100.0, 200.0) for i in range(20)]
    for i, row in enumerate(rows):
        # First half establishes the schedule-to-send offset; the second half
        # waits two seconds inside the generator before a fast endpoint call.
        row["t_send_unix"] += 0.0 if i < 10 else 2.0
    s = summarize(rows, acceptance={"hard_timeouts": {"ttfg_s": 1}})
    assert s["e2e_ms"]["p95"] == 200.0
    assert s["sla"]["hard_timeout_breaches"] == 10
    assert s["sla"]["hard_timeout_basis"]["includes_client_queue_wait"] is True


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
    assert "Acceptance scorecard" not in render_markdown(s, "t")


def test_interchunk_threshold_counts_as_breach():
    # 40 clean (interchunk 5ms), 10 stalled (interchunk 50ms) vs a 20ms cap
    rows = [_row(i, 400.0, 800.0, inter=5.0) for i in range(40)]
    rows += [_row(i, 400.0, 800.0, inter=50.0) for i in range(40, 50)]
    accept = {"interchunk_ms": 20, "success_rate": 0.95}
    s = summarize(rows, acceptance=accept)
    assert s["sla"]["interchunk_breaches"] == 10
    sr = s["sla"]["success_rate"]
    assert sr["actual"] == 0.80 and sr["met"] is False
    assert "interchunk breaches" in render_markdown(s, "t")


def test_no_interchunk_target_no_breach_field():
    rows = [_row(i, 400.0, 800.0, inter=99.0) for i in range(10)]
    s = summarize(rows, acceptance={"success_rate": 0.99})
    assert "interchunk_breaches" not in s["sla"]


def test_output_token_targeting_reports_ratio_and_finish_reasons():
    rows = [_row(i, 400.0, 800.0, comp=40) for i in range(30)]   # stop, ratio 1.0
    for i in range(30, 40):
        r = _row(i, 400.0, 800.0, comp=40)
        r["finish_reason"] = "length"
        r["completion_tokens"] = 100                              # ran to the cap
        rows.append(r)
    s = summarize(rows)
    tt = s["token_targeting"]
    assert tt["output_reported_over_intended_p50"] is not None
    assert tt["finish_reasons"]["stop"] == 30
    assert tt["finish_reasons"]["length"] == 10
    assert "output tokens" in render_markdown(s, "t")


def _stable_target_rows(*, prompt_actual=1000, prompt_intended=1000,
                        output_actual=100, output_intended=100):
    rows = []
    for i in range(600):
        row = _row(i, 400.0, 600.0, prompt=prompt_actual,
                   comp=output_actual)
        row["intended_input_tokens"] = prompt_intended
        row["intended_output_tokens"] = output_intended
        row["intended_cache_fraction"] = None
        row["first_send_unix"] = row["t_send_unix"]
        row["finished_unix"] = row["t_send_unix"] + 0.6
        rows.append(row)
    return rows


def test_input_workload_mismatch_blocks_an_otherwise_green_verdict():
    rows = _stable_target_rows(prompt_actual=100, prompt_intended=1000)
    s = summarize(rows, acceptance=ACCEPT)
    kind, text = _verdict(s)
    assert kind == "caution"
    assert "input tokens did not reproduce" in text
    tt = s["token_targeting"]
    assert tt["input_coverage"] == 1.0
    assert tt["input_abs_relative_error_pct"]["p95"] == 90.0
    assert "CAUTION (workload token fidelity)" in render_markdown(s, "t")


def test_output_workload_mismatch_blocks_an_otherwise_green_verdict():
    rows = _stable_target_rows(output_actual=1, output_intended=100)
    s = summarize(rows, acceptance=ACCEPT)
    kind, text = _verdict(s)
    assert kind == "caution"
    assert "output tokens did not reproduce" in text
    assert s["token_targeting"]["output_abs_relative_error_pct"]["p95"] == 99.0


def test_matching_workload_token_shape_can_reach_green():
    s = summarize(_stable_target_rows(), acceptance=ACCEPT)
    assert _verdict(s) == ("ok", "meets every acceptance target")
    assert s["token_targeting"]["status"] == "verified"


def test_failed_profile_rows_make_token_fidelity_incomplete():
    rows = _stable_target_rows()
    for row in rows[300:]:
        row.update({
            "ok": False,
            "status": 500,
            "error": "http 500",
            "prompt_tokens": None,
            "completion_tokens": None,
        })

    summary = summarize(rows, acceptance=ACCEPT)
    targeting = summary["token_targeting"]
    assert targeting["input_intended_requests"] == 600
    assert targeting["input_eligible_successes"] == 300
    assert targeting["input_coverage"] == 0.5
    assert targeting["output_intended_requests"] == 600
    assert targeting["output_coverage"] == 0.5
    assert targeting["status"] == "mismatch"
    assert "300 of 600 captured profile requests" in targeting["warning"]


def test_parse_corrupt_usage_cannot_verify_token_or_cache_fidelity():
    rows = _stable_target_rows()
    for row in rows[300:]:
        row.update({
            "parse_errors": 1,
            "cached_tokens": 0,
            "cached_tokens_source":
                "prompt_tokens_details.cached_tokens",
            "intended_cache_fraction": 0.0,
        })
    for row in rows[:300]:
        row.update({
            "parse_errors": 0,
            "cached_tokens": 0,
            "cached_tokens_source":
                "prompt_tokens_details.cached_tokens",
            "intended_cache_fraction": 0.0,
        })

    summary = summarize(rows, acceptance=ACCEPT)
    assert summary["token_targeting"]["input_coverage"] == 0.5
    assert summary["token_targeting"]["output_coverage"] == 0.5
    assert summary["cache_fidelity"]["coverage"] == 0.5
    assert summary["cache_fidelity"]["status"] == "unverified"
    assert "300 of 600 captured profile requests" in \
        summary["cache_fidelity"]["warning"]


def test_illustrative_targets_can_never_produce_an_unqualified_green():
    targets = {
        **ACCEPT,
        "note": "illustrative targets; replace with customer requirements",
    }
    s = summarize(_stable_target_rows(), acceptance=targets)
    assert s["sla"]["targets_warning"]
    kind, text = _verdict(s)
    assert kind == "caution"
    assert "illustrative" in text
