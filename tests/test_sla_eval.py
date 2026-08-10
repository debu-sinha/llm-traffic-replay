"""Acceptance scorecard: targets from the profile config are scored against
measured percentiles, hard timeouts count as failures, and the report
renders the verdicts."""
from traffic_replay.metrics import (
    _verdict, _wilson_lower_95, render_html, render_markdown, summarize,
)
from traffic_replay.report_decision import (
    IntegrityContext,
    build_report_decision,
)


def _decision(summary):
    return build_report_decision(
        summary, IntegrityContext(status="verified", reason="test evidence"))


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
    assert "| TTFG | p50 | 700 | 2,000 | NO |" in report


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


def test_model_refusals_cannot_be_reported_as_successful_answers():
    rows = [_row(i, 10.0, 20.0) for i in range(300)]
    for row in rows:
        row.update({
            "visible_content_seen": True,
            "reasoning_seen": False,
            "refusal_seen": True,
            "valid_tool_calls": 0,
            "stream_complete": True,
            "parse_errors": 0,
        })

    summary = summarize(rows, acceptance=ACCEPT)

    assert summary["answers"]["model_refusal_outcomes"] == 300
    assert summary["answers"]["model_refusal_rate"] == 1.0
    assert summary["answers"]["acceptable_outcomes"] == 0
    assert summary["answers"]["answer_rate"] == 0.0
    assert summary["sla"]["success_rate"]["successes"] == 0
    assert summary["sla"]["success_rate"]["met"] is False
    assert summary["drift"]["drift_kind"] == "failing"
    assert "same acceptable-answer population" in \
        summary["drift"]["outcome_population"]
    assert _verdict(summary)[0] in {"invalid", "miss"}
    report = render_markdown(summary, "refusals")
    assert "model refusals (unacceptable by default): 300" in report


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


def test_tool_call_only_output_cannot_pass_a_first_content_hard_cap():
    rows = [_row(i, None, 1000.0, comp=0) for i in range(100)]
    for row in rows:
        row.update({
            "caller_ttft_ms": None,
            "ttf_tool_call_ms": 100.0,
            "caller_ttf_tool_call_ms": 100.0,
            "visible_content_seen": False,
            "valid_tool_calls": 1,
            "stream_complete": True,
            "parse_errors": 0,
        })

    summary = summarize(
        rows,
        acceptance={"hard_timeouts": {"ttft_s": 0.05}},
        ttft_definition="first_content",
    )

    assert summary["sla"]["hard_timeout_breaches"] == 100
    assert summary["sla"]["hard_timeout_unmeasured"] == 0
    assert _decision(summary)["customer_sla"]["code"] == "MISS"


def test_missing_exact_caller_clock_makes_hard_cap_inconclusive():
    rows = [_row(i, 10.0, 20.0) for i in range(20)]
    for row in rows:
        row.update({
            "caller_ttft_ms": None,
            "caller_e2e_ms": None,
            "stream_complete": True,
            "parse_errors": 0,
        })

    summary = summarize(
        rows,
        acceptance={"hard_timeouts": {"ttft_s": 1, "ttfg_s": 2}},
    )

    assert summary["sla"]["hard_timeout_breaches"] == 0
    assert summary["sla"]["hard_timeout_unmeasured"] == 20
    assert _decision(summary)["customer_sla"]["code"] == "INCONCLUSIVE"
    assert "INCONCLUSIVE" in render_markdown(summary, "missing clocks")


def test_failed_request_over_the_hard_deadline_is_not_omitted():
    rows = [_row(i, 100.0, 200.0) for i in range(999)]
    failed = _row(999, None, 60_000.0, ok=False)
    failed.update({
        "caller_ttft_ms": None,
        "caller_e2e_ms": 60_000.0,
        "request_attempts": 1,
        "stream_complete": False,
        "parse_errors": 0,
    })
    rows.append(failed)

    summary = summarize(
        rows,
        acceptance={"hard_timeouts": {"ttfg_s": 45}},
    )

    assert summary["requests_failed"] == 1
    assert summary["sla"]["hard_timeout_breaches"] == 1
    assert summary["sla"][
        "hard_timeout_breaches_among_protocol_clean_successes"] == 0
    assert _decision(summary)["customer_sla"]["code"] == "MISS"


def test_early_failed_request_is_a_success_failure_not_a_hard_timeout():
    rows = [_row(i, 100.0, 200.0) for i in range(99)]
    failed = _row(99, None, 200.0, ok=False)
    failed.update({"caller_ttft_ms": None, "caller_e2e_ms": 200.0,
                   "request_attempts": 1})
    rows.append(failed)

    summary = summarize(
        rows,
        acceptance={
            "hard_timeouts": {"ttft_s": 1, "ttfg_s": 2},
            "success_rate": 0.999,
        },
    )

    assert summary["sla"]["hard_timeout_breaches"] == 0
    assert summary["sla"]["hard_timeout_unmeasured"] == 0
    assert summary["sla"]["success_rate"]["actual"] == 0.99


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


def test_interchunk_target_with_no_gap_measurements_is_inconclusive():
    rows = [_row(i, 100.0, 200.0, inter=None) for i in range(100)]
    summary = summarize(rows, acceptance={"interchunk_ms": 20})

    assert summary["sla"]["interchunk_breaches"] == 0
    assert summary["sla"]["interchunk_measured"] == 0
    assert summary["sla"]["interchunk_unmeasured"] == 100
    assert _decision(summary)["customer_sla"]["code"] == "INCONCLUSIVE"
    assert "INCONCLUSIVE" in render_markdown(summary, "no gaps")


def test_no_interchunk_target_no_breach_field():
    rows = [_row(i, 400.0, 800.0, inter=99.0) for i in range(10)]
    s = summarize(rows, acceptance={"success_rate": 0.99})
    assert "interchunk_breaches" not in s["sla"]


def test_percentile_target_counts_missing_events_as_non_meeting():
    rows = []
    for i in range(1000):
        if i < 920:
            row = _row(i, 100.0, 200.0)
        elif i < 960:
            row = _row(i, 10_000.0, 10_100.0)
        else:
            row = _row(i, None, 200.0, comp=0)
            row.update({"valid_tool_calls": 1,
                        "ttf_tool_call_ms": 50.0})
        row.update({"visible_content_seen": i < 960,
                    "stream_complete": True, "parse_errors": 0})
        rows.append(row)

    summary = summarize(
        rows, acceptance={"ttft_ms": {"p95": 500}})
    scored = summary["sla"]["ttft_vs_target"][0]

    # The descriptive event-bearing survivor percentile is fast, but the
    # acceptance nearest-rank p95 over all outcomes lands in the slow group.
    assert scored["descriptive_event_only_percentile_ms"] == 100.0
    assert scored["actual_ms"] == 10_000.0
    assert scored["meeting_outcomes"] == 920
    assert scored["eligible_outcomes"] == 1000
    assert scored["observed_meeting_fraction"] == 0.92
    assert scored["required_meeting_fraction"] == 0.95
    assert scored["met"] is False
    assert _decision(summary)["customer_sla"]["code"] == "MISS"


def test_acceptance_actual_and_result_use_the_same_nearest_rank_estimator():
    rows = [_row(i, 100.0, 200.0) for i in range(190)]
    rows += [_row(i, 1000.0, 1100.0) for i in range(190, 200)]

    summary = summarize(
        rows, acceptance={"ttft_ms": {"p95": 120}})
    scored = summary["sla"]["ttft_vs_target"][0]

    # NumPy's descriptive linear p95 is 145 ms for this boundary sample. The
    # acceptance contract uses nearest-rank consistently: observation 190 is
    # 100 ms, so actual <= target and PASS cannot contradict one another.
    assert scored["descriptive_event_only_percentile_ms"] == 145.0
    assert scored["actual_estimator"] == "nearest_rank"
    assert scored["actual_ms"] == 100.0
    assert scored["met"] is True


def test_latency_boundary_persists_and_renders_the_unrounded_scored_value():
    rows = [_row(i, 100.04, 200.0) for i in range(20)]

    summary = summarize(rows, acceptance={"ttft_ms": {"p50": 100.0}})
    scored = summary["sla"]["ttft_vs_target"][0]

    assert scored["actual_ms"] == 100.04
    assert scored["target_ms"] == 100.0
    assert scored["met"] is False
    markdown = render_markdown(summary, "precision boundary")
    html = render_html(summary, "precision boundary")
    assert "| TTFT | p50 | 100.00 | 100.04 | NO |" in markdown
    assert ">100.00</td><td>100.04</td><td class='no'>NO</td>" in html


def test_wilson_boundary_persists_precision_and_never_renders_false_equality():
    rows = [_row(i, 100.0, 200.0) for i in range(2702)]

    summary = summarize(rows, acceptance={"success_rate": 0.999})
    scored = summary["sla"]["success_rate"]

    assert scored["actual"] == 1.0
    assert scored["one_sided_95pct_wilson_lower"] == \
        _wilson_lower_95(2702, 2702)
    assert scored["one_sided_95pct_wilson_lower"] < scored["target"]
    assert scored["statistically_demonstrated"] is False
    markdown = render_markdown(summary, "Wilson precision boundary")
    html = render_html(summary, "Wilson precision boundary")
    assert "lower bound 0.9989997" in markdown
    assert ">0.9990000</td><td>0.9989997</td>" in html


def test_html_success_rate_boundary_does_not_round_a_miss_to_equality():
    rows = [_row(i, 100.0, 200.0) for i in range(20_000)]
    rows[-21:] = [
        _row(i, 100.0, 200.0, ok=False)
        for i in range(20_000 - 21, 20_000)
    ]

    summary = summarize(rows, acceptance={"success_rate": 0.999})
    scored = summary["sla"]["success_rate"]
    assert scored["actual"] == 0.99895
    assert scored["met"] is False
    html = render_html(summary, "point precision boundary")
    assert ">0.99900</td><td>0.99895</td><td class='no'>NO</td>" in html


def test_answer_rate_boundary_is_not_rounded_up_to_the_implicit_floor():
    judged = 20_099
    answered = 19_898
    rows = [_row(i, 100.0, 200.0) for i in range(judged)]
    for index, row in enumerate(rows):
        row.update(
            visible_content_seen=index < answered, valid_tool_calls=0,
            refusal_seen=False, stream_complete=True, parse_errors=0)

    summary = summarize(rows)

    assert summary["answers"]["answer_rate"] == answered / judged
    assert summary["answers"]["answer_rate"] < 0.99
    kind, text = _verdict(summary)
    assert kind == "miss"
    assert "did not produce a readable answer" in text


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
    kind, text = _verdict(s)
    assert kind == "caution"
    assert "response model was reported" in text
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
