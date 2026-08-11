"""Small-N gate, drift-over-time, network floor (connect), and endpoint
metadata in the report. These are the confidence features: they make a short
or misleading run say so, and they record what was actually tested."""
from __future__ import annotations

import hashlib
import json
import random

from traffic_replay import __version__
from traffic_replay.metrics import (_concurrency_block, _drift_block,
                                    render_html, render_markdown, summarize)


def _rows(n, base_ttft=100.0, t0=0.0, dt=1.0):
    return [{"ok": True, "t_send_unix": t0 + i * dt, "ttft_ms": base_ttft,
             "ttfb_ms": 1.0, "e2e_ms": base_ttft * 2, "connect_ms": 8.0,
             "dispatch_lag_ms": 0.0, "prompt_tokens": 100,
             "completion_tokens": 10} for i in range(n)]


def test_first_parsed_stream_event_is_a_separate_diagnostic_metric():
    rows = _rows(4)
    for index, row in enumerate(rows):
        row.update(
            ttse_ms=2.0 + index,
            caller_ttse_ms=12.0 + index,
            caller_ttfb_ms=11.0 + index,
            caller_ttft_ms=110.0 + index,
            caller_e2e_ms=210.0 + index,
        )
    summary = summarize(rows)

    assert summary["ttse_ms"]["n"] == 4
    assert summary["ttse_corrected_ms"]["n"] == 4
    assert summary["stream_event_definition"]["excludes_claims"] == [
        "model_token", "reasoning_content", "visible_content",
        "successful_response",
    ]
    markdown = render_markdown(summary, "TTSE test")
    html = render_html(summary, "TTSE test")
    assert "TTSE (first parsed stream event; diagnostic)" in markdown
    assert "TTSE (first parsed stream event; diagnostic)" in html
    assert "first token" not in summary["stream_event_definition"]["meaning"]


def _reasoning_probe(candidate, *, index=1, disposition="accepted"):
    canonical = json.dumps(
        candidate, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"))
    rejected = disposition == "rejected"
    return {
        "schema_version": "reasoning-control-probe-evidence/v1",
        "candidate_index": index,
        "candidate_redacted": candidate,
        "candidate_canonical_sha256": hashlib.sha256(
            canonical.encode("utf-8")).hexdigest(),
        "disposition": disposition,
        "evidence_method": (
            "request_validation_response" if rejected else
            "single_request_behavior_observation"),
        "effective_status": (
            "not_applied_request_rejected" if rejected else "unknown"),
        "effective_value": None,
        "request_id": f"probe-{index:02d}",
        "logical_request_body_sha256": "a" * 64,
        "physical_request_body_sha256s": ["b" * 64],
    }


def test_the_sample_gate_names_which_quantiles_it_supports():
    """A quantile needs roughly ten observations past it to be an estimate.
    At n=100 there is a 37 percent chance of drawing nothing at all beyond
    the true p99, so the old "100 is enough for p99" rule was not
    defensible."""
    tiny = summarize(_rows(10))["sample"]
    assert tiny["supports"] == []
    assert "p99" in tiny["indicative_only"]

    mid = summarize(_rows(150))["sample"]
    assert mid["supports"] == ["p50", "p90"]
    assert mid["indicative_only"] == ["p95", "p99"]
    assert "p95, p99 are indicative only" in mid["warning"]

    big = summarize(_rows(1200))["sample"]
    assert big["supports"] == ["p50", "p90", "p95", "p99"]
    assert big["warning"] is None


def test_a_target_on_an_unsupportable_quantile_is_not_a_pass():
    """Scoring a p99 target on 150 requests and calling it met would be a
    verdict the sample cannot carry."""
    s = summarize(_rows(150), acceptance={"ttft_ms": {"p99": 100000}})
    assert s["sla"]["ttft_vs_target"][0]["met"] is True
    assert "Meets every acceptance target" not in render_html(s, "x")
    md = [x for x in render_markdown(s, "x").splitlines()
          if x.startswith("verdict:")][0]
    assert "p99" in md and "cannot support" in md


def test_drift_flag_rises_with_a_rising_tail():
    # window 0 (0-60s) fast, window 2 (120-180s) slow -> drift
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    late = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(early + late)
    assert len(d["windows"]) >= 2
    assert d["drift_flag"] is True
    assert d["ttft_p95_drift_ratio"] > 1.3


def test_drift_needs_two_windows():
    d = _drift_block(_rows(30, t0=0.0, dt=1.0))  # all within 60s
    assert d["windows"] == []
    assert "two" in d["note"]


def test_drift_uses_one_logical_schedule_clock_not_final_retry_send_time():
    """A retry may finish its final POST in a later minute. Stability cohorts
    belong to when the logical request was scheduled, not whichever physical
    retry produced the terminal result."""
    rows = []
    for i in range(25):
        row = _rows(1, base_ttft=100.0)[0]
        row.update({
            "scheduled_s": float(i),
            "first_send_unix": 1_000_000.0 + i,
            "t_send_unix": 1_000_061.0 + i,
            "retries": 1,
        })
        rows.append(row)
    for i in range(25):
        row = _rows(1, base_ttft=400.0)[0]
        row.update({
            "scheduled_s": 70.0 + i,
            "first_send_unix": 1_000_070.0 + i,
            "t_send_unix": 1_000_070.0 + i,
        })
        rows.append(row)

    drift = _drift_block(rows)

    assert drift["window_clock"] == "scheduled_s"
    assert drift["window_clock_n"] == drift["window_clock_of"] == 50
    assert [window["n"] for window in drift["windows"]] == [25, 25]
    assert drift["drift_kind"] == "variable"


def test_connect_and_endpoint_render_in_html():
    s = summarize(_rows(120), run_meta={
        "input_mode": "profile", "endpoint_path": "/e",
        "endpoint_metadata": {"name": "acme-glm-prod-42", "task": "llm/v1/chat",
                              "route_optimized": True, "ready": "READY",
                              "served_entities": [{"name": "e",
                                                   "workload_type": "GPU_LARGE"}]}})
    h = render_html(s, "extras")
    assert "Connection setup" in h              # connect line
    assert "excluded" in h                      # states it is not in TTFT
    assert "8" in h                             # connect ms value
    assert "fresh-connection setup diagnostic" in h
    assert "upper bound on network distance" not in h
    assert "Endpoint under test" in h           # endpoint metadata card
    assert "acme-glm-prod-42" in h            # custom name shown
    assert "GPU_LARGE" in h                     # served entity workload


def test_report_distinguishes_physical_post_attempts_from_retry_markers():
    rows = _rows(3)
    rows[0].update({
        "request_attempts": 3,
        "connection_attempts": 3,
        "retries": 2,
        "retry_reasons": [
            "stream_options_rejected", "transport_error_after_post"],
    })
    rows[1].update({
        "request_attempts": 1,
        "connection_attempts": 1,
        "retries": 0,
        "retry_reasons": [],
    })
    rows[2].update({"retries": 1})  # legacy row: physical count unknown

    summary = summarize(rows)
    evidence = summary["physical_post_attempts"]
    assert evidence["logical_rows_with_additional_attempts"] == 1
    assert evidence["additional_attempts"] == 2
    assert evidence["recorded_retry_triggers"] == {
        "stream_options_rejected": 1,
        "transport_error_after_post": 1,
    }
    assert evidence["legacy_retry_marked_rows_without_attempt_count"] == 1

    markdown = render_markdown(summary, "attempt evidence")
    report_html = render_html(summary, "attempt evidence")
    for rendered in (markdown, report_html):
        assert "logical rows with additional physical post attempts" \
            in rendered.lower()
        assert "stream_options_rejected" in rendered
        assert "transport_error_after_post" in rendered
        assert "connection retry" not in rendered.lower()


def test_report_does_not_project_unobserved_answer_length_from_percentiles():
    summary = summarize(_rows(30))
    markdown = render_markdown(summary, "observed TPOT")

    assert summary["tpot_ms"]["n"] == 30
    assert summary["completion_tpot_ms"] == summary["tpot_ms"]
    assert summary["tpot_scope"] == "all_endpoint_reported_completion_tokens"
    assert "(e2e - ttft) / (completion_tokens - 1)" in markdown
    assert "all-completion TPOT" in markdown
    assert "not visible-output TPOT" in markdown
    assert "500-token" not in markdown
    assert "unobserved answer length" in markdown


def test_reasoning_completion_tokens_are_not_visible_output_metrics():
    rows = _rows(30)
    for row in rows:
        row.update(reasoning_seen=True, reasoning_tokens=8,
                   reasoning_chunks=3, visible_content_seen=True,
                   stream_complete=True)
    summary = summarize(rows)
    throughput = summary["throughput"]

    assert throughput["completion_tokens_per_min"] == \
        throughput["output_tokens_per_min"]
    assert throughput["output_tokens_per_min_legacy_alias_of"] == \
        "completion_tokens_per_min"
    assert "visible_output_tokens_per_min" not in throughput
    assert throughput["visible_output_token_accounting"]["status"] == \
        "unavailable"
    assert "visible_tpot_ms" not in summary

    markdown = render_markdown(summary, "reasoning completion accounting")
    html = render_html(summary, "reasoning completion accounting")
    for rendered in (markdown, html):
        assert "all-completion" in rendered
        assert "completion token" in rendered
    assert "time per output token (TPOT)" not in markdown
    assert "output throughput" not in html


def test_visible_output_metrics_require_exact_source_labeled_accounting():
    rows = _rows(30)
    for row in rows:
        row.update(
            visible_output_tokens=4,
            visible_output_tokens_source="provider.usage.visible_tokens",
            ttfv_ms=125.0,
            visible_content_seen=True,
            stream_complete=True,
        )
    summary = summarize(rows)

    throughput = summary["throughput"]
    assert throughput["visible_output_token_accounting"]["status"] == \
        "available"
    assert throughput["visible_output_tokens_per_min"] > 0
    assert summary["visible_tpot_ms"]["n"] == 30
    assert "explicitly accounted visible output token" in \
        render_markdown(summary, "visible accounting")


def test_reasoning_probe_section_is_safe_linked_and_non_inferential():
    candidate = {
        "reasoning_effort": "none",
        "display_marker": "</td><script>alert(1)</script>|`probe`",
    }
    probe = _reasoning_probe(candidate)
    summary = summarize(_rows(30), run_meta={
        "preflight_gate": {
            "reasoning_probe_requests": 1,
            "reasoning_control_probes": [probe],
        },
    })

    markdown = render_markdown(summary, "probe evidence")
    html = render_html(summary, "probe evidence")
    digest = probe["candidate_canonical_sha256"]

    assert "## Reasoning-control probes" in markdown
    assert digest in markdown and digest in html
    assert "reasoning\\_effort" in markdown and "reasoning_effort" in html
    assert "does not prove the provider applied" in markdown.lower()
    assert "Accepted does not" in html
    assert "unknown" in markdown.lower() and "unknown" in html.lower()
    assert "[requests.jsonl](requests.jsonl)" in markdown
    assert "href='requests.jsonl'" in html
    assert probe["request_id"] in markdown and probe["request_id"] in html
    assert probe["logical_request_body_sha256"] in markdown
    assert probe["physical_request_body_sha256s"][0] in html
    assert "<script>alert(1)</script>" not in markdown
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in markdown
    assert "&lt;script&gt;" in html


def test_reasoning_probe_effect_is_only_known_for_request_rejection():
    rejected = _reasoning_probe(
        {"reasoning_effort": "unsupported"}, disposition="rejected")
    summary = summarize(_rows(30), run_meta={
        "preflight_gate": {
            "reasoning_probe_requests": 1,
            "reasoning_control_probes": [rejected],
        },
    })

    markdown = render_markdown(summary, "rejected probe")
    html = render_html(summary, "rejected probe")
    assert "not applied - request rejected" in markdown
    assert "not applied - request rejected" in html


def test_reasoning_probe_report_withholds_unsupported_effect_claim():
    probe = _reasoning_probe({"reasoning_effort": "none"})
    probe["effective_status"] = "thinking_disabled"
    probe["effective_value"] = False
    summary = summarize(_rows(30), run_meta={
        "preflight_gate": {
            "reasoning_probe_requests": 1,
            "reasoning_control_probes": [probe],
        },
    })

    markdown = render_markdown(summary, "bad effective claim")
    html = render_html(summary, "bad effective claim")
    for rendered in (markdown, html):
        assert "unsupported effective-behavior claim withheld" in rendered
        assert "unknown" in rendered.lower()
        assert "thinking_disabled" not in rendered


def test_missing_visible_content_does_not_invent_a_finish_reason():
    rows = _rows(3)
    for row in rows:
        row["ttfr_ms"] = 80.0
    rows[0]["ttfv_ms"] = 150.0

    markdown = render_markdown(summarize(rows), "reasoning visibility")

    assert "remaining requests had no observed visible-content event" \
        in markdown
    assert "artifact does not establish why" in markdown
    assert "ran out of output tokens still reasoning" not in markdown


def test_exact_caller_display_is_not_labeled_as_a_correction():
    rows = _rows(30)
    for row in rows:
        row["caller_ttft_ms"] = 125.0
        row["caller_e2e_ms"] = 225.0
    summary = summarize(rows)

    markdown = render_markdown(summary, "exact caller")
    report_html = render_html(summary, "exact caller")
    for rendered in (markdown, report_html):
        assert "Exact caller TTFT" in rendered
        assert "corrected (configured" not in rendered


def test_stability_card_present_for_long_run():
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    late = _rows(25, base_ttft=110.0, t0=140.0, dt=1.0)
    h = render_html(summarize(early + late), "stability")
    assert "Stability over time" in h


def test_warmup_is_not_reported_as_stable():
    """Window 0 is 15x slower than the last window.

    Comparing only first to last calls that an improvement and passes it as
    stable, which would let a caller quote a blended p95 from a run whose
    measured path changed materially. Client evidence must not assign the
    cause to a cold endpoint without correlated backend telemetry.
    """
    cold = _rows(25, base_ttft=31000.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=3500.0, t0=70.0, dt=1.0)
    warm = _rows(25, base_ttft=2000.0, t0=140.0, dt=1.0)
    d = _drift_block(cold + mid + warm)
    assert d["drift_flag"] is True
    assert d["drift_kind"] == "warming"
    assert d["ttft_p95_spread_ratio"] > 1.3
    assert d["ttft_p95_drift_ratio"] < 1.0      # end/end alone looks like a win
    assert "worst in the first window" in d["drift_headline"]
    assert "cannot attribute the cause" in d["drift_headline"]
    assert "cold start" not in d["drift_headline"]


def test_midrun_spike_is_not_reported_as_stable():
    """Ends match, middle is 10x worse. first/last ratio is ~1.0 here, so only
    a worst-to-best spread catches it."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    spike = _rows(25, base_ttft=1000.0, t0=70.0, dt=1.0)
    b = _rows(25, base_ttft=100.0, t0=140.0, dt=1.0)
    d = _drift_block(a + spike + b)
    assert len(d["windows"]) >= 3
    assert 0.9 < d["ttft_p95_drift_ratio"] < 1.1   # endpoints agree
    assert d["drift_flag"] is True                 # but the run is not stable
    assert d["drift_kind"] == "spike"


def test_genuinely_steady_run_stays_stable():
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=105.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=110.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_flag"] is False
    assert d["drift_kind"] == "stable"


def test_degrading_run_is_labeled_degrading():
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=200.0, t0=70.0, dt=1.0)
    late = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(early + mid + late)
    assert d["drift_kind"] == "degrading"
    assert "measured caller path slowed" in d["drift_headline"]
    assert "cannot attribute the cause" in d["drift_headline"]


def test_first_visible_stability_scores_visible_latency_not_reasoning_start():
    rows = []
    for window, visible_ms in enumerate((400.0, 1200.0, 4000.0)):
        for row in _rows(25, base_ttft=100.0, t0=window * 70.0, dt=1.0):
            row.update({
                "ttfv_ms": visible_ms,
                "caller_ttft_ms": 100.0,
                "caller_ttfv_ms": visible_ms,
                "visible_content_seen": True,
                "stream_complete": True,
                "parse_errors": 0,
            })
            rows.append(row)

    visible = summarize(rows, ttft_definition="first_visible")["drift"]
    content = summarize(rows, ttft_definition="first_content")["drift"]

    assert visible["latency_metric"] == "caller_ttfv_ms"
    assert visible["latency_event"] == "ttfv"
    assert visible["ttfv_p95_spread_ratio"] == 10.0
    assert visible["drift_kind"] == "degrading"
    assert "TTFV (first visible content)" in visible["drift_headline"]
    assert all("ttfv_p95" in window for window in visible["windows"])
    assert all("ttft_p95" not in window for window in visible["windows"])

    assert content["latency_metric"] == "caller_ttft_ms"
    assert content["ttft_p95_spread_ratio"] == 1.0
    assert content["drift_kind"] == "stable"


def test_first_visible_stability_rejects_event_survivor_percentiles():
    rows = []
    for window in range(3):
        for i, row in enumerate(
                _rows(40, base_ttft=100.0, t0=window * 70.0, dt=1.0)):
            visible = i < 20
            row.update({
                "ttfv_ms": 400.0 if visible else None,
                "caller_ttfv_ms": 400.0 if visible else None,
                "visible_content_seen": visible,
                "valid_tool_calls": 0 if visible else 1,
                "stream_complete": True,
                "parse_errors": 0,
            })
            rows.append(row)

    drift = summarize(rows, ttft_definition="first_visible")["drift"]

    assert "drift_kind" not in drift
    assert drift["counted_windows"] == 0
    assert all(window["latency_coverage"] == 0.5
               for window in drift["windows"])
    assert all(window["event_survivorship"] is True
               for window in drift["windows"])


def test_first_visible_stability_keeps_reasoning_only_second_half():
    rows = []
    for i in range(240):
        visible = i < 120
        rows.append({
            "ok": True,
            "scheduled_s": i * 0.5,
            "first_send_unix": 1_700_000_000.0 + i * 0.5,
            "t_send_unix": 1_700_000_000.0 + i * 0.5,
            "ttft_ms": 10.0,
            "ttfv_ms": 100.0 if visible else None,
            "e2e_ms": 200.0,
            "visible_content_seen": visible,
            "reasoning_seen": not visible,
            "valid_tool_calls": 0,
            "stream_complete": True,
            "parse_errors": 0,
        })

    drift = summarize(rows, ttft_definition="first_visible")["drift"]

    assert drift["window_clock_n"] == drift["window_clock_of"] == 240
    assert len(drift["windows"]) == 2
    assert drift["windows"][0]["latency_coverage"] == 1.0
    assert drift["windows"][1]["latency_coverage"] == 0.0
    assert drift["windows"][1]["event_survivorship"] is False
    assert drift["windows"][1]["counted"] is False


def test_unstable_run_says_so_in_html():
    cold = _rows(25, base_ttft=31000.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=3500.0, t0=70.0, dt=1.0)
    warm = _rows(25, base_ttft=2000.0, t0=140.0, dt=1.0)
    h = render_html(summarize(cold + mid + warm), "warmup")
    assert "unstable" in h
    assert "stable</span>" not in h.replace("unstable", "")


def test_noisy_run_is_variable_not_degrading():
    """Real warm-endpoint shape: p95 dips then rises, ending near where it
    started. The max lands in the last window, but the windows do not move one
    way, so calling it degradation overstates the data. It is noise, and the
    number still should not be quoted as steady state."""
    a = _rows(25, base_ttft=1900.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=1300.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=2200.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_flag"] is True          # not steady, so still flagged
    assert d["drift_kind"] == "variable"    # but no trend is claimed
    assert "noisy" in d["drift_headline"]


def test_degrading_requires_every_window_to_rise():
    """A run that rises overall but dips in the middle is not a clean trend."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=50.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_kind"] == "variable"


def test_prompts_mode_warns_when_prompts_are_recycled():
    """A small prompt set cycled over a long run means most requests are
    verbatim repeats, which the endpoint prompt cache serves. The achieved
    cache fraction then describes the replay, not production traffic, so the
    report has to say so."""
    meta = {"input_mode": "prompts", "endpoint_path": "/e",
            "prompts_file": "p.jsonl", "prompts_count": 10}
    rows = _rows(100)
    for index, row in enumerate(rows):
        row["prompt_index"] = index % 10
    s = summarize(rows, run_meta=meta)
    r = s["replay"]
    assert r["distinct_prompts"] == 10
    assert r["avg_sends_per_prompt"] == 10
    assert r["sent"]["repeat_requests"] == 90
    assert r["attempted"]["repeat_requests"] == 90
    assert r["successful"]["repeat_requests"] == 90
    assert "prompt cache" in r["warning"]
    assert "CAUTION (prompt replay)" in render_markdown(s, "replay")
    assert "banner warn" in render_html(s, "replay")


def test_prompts_mode_quiet_when_every_prompt_is_sent_once():
    meta = {"input_mode": "prompts", "endpoint_path": "/e",
            "prompts_file": "p.jsonl", "prompts_count": 120}
    rows = _rows(100)
    for index, row in enumerate(rows):
        row["prompt_index"] = index
    s = summarize(rows, run_meta=meta)
    assert s["replay"]["warning"] is None


def test_prompt_repeats_use_persisted_indexes_despite_failure_and_interleave():
    """A failed request still reached the wire and may prime a cache. The
    successful-row count must not erase a later reuse of prompt zero."""
    rows = _rows(3)
    for row, prompt_index in zip(rows, (0, 1, 0)):
        row["prompt_index"] = prompt_index
    rows[1].update(ok=False, status=503, error="upstream unavailable")
    summary = summarize(rows, run_meta={
        "input_mode": "prompts", "endpoint_path": "/e",
        "prompts_file": "p.jsonl", "prompts_count": 2,
    })

    replay = summary["replay"]
    assert replay["attempted"]["repeat_requests"] == 1
    assert replay["sent"]["repeat_requests"] == 1
    assert replay["successful"]["repeat_requests"] == 1
    assert replay["repeat_requests"] == 1
    assert "1 of 3 requests that reached the wire" in replay["warning"]


def test_prompt_repeat_count_is_unavailable_when_indexes_are_missing():
    rows = _rows(3)
    rows[0]["prompt_index"] = 0
    rows[2]["prompt_index"] = 0
    summary = summarize(rows, run_meta={
        "input_mode": "prompts", "endpoint_path": "/e",
        "prompts_file": "p.jsonl", "prompts_count": 2,
    })

    replay = summary["replay"]
    assert replay["sent"]["status"] == "unavailable"
    assert replay["sent"]["repeat_requests"] is None
    assert replay["repeat_requests"] is None
    assert "No repeat count was inferred" in replay["warning"]


def test_profile_mode_has_no_replay_block():
    s = summarize(_rows(100), run_meta={"input_mode": "profile",
                                        "endpoint_path": "/e"})
    assert "replay" not in s


def _calibration_evidence(*, overlap: bool, missing_hash: bool = False):
    replay = _rows(2)
    for index, row in enumerate(replay):
        row.update(phase="replay", request_body_sha256=(
            ("a" if index == 0 else "b") * 64))
    calibration = _rows(1, t0=-1.0)[0]
    calibration.update(
        phase="calibration",
        request_body_sha256=("a" if overlap else "c") * 64)
    if missing_hash:
        replay[1].pop("request_body_sha256")
    return replay, [calibration, *replay]


def test_calibration_reports_exact_payload_overlap_and_blocks_cold_claim():
    replay, all_rows = _calibration_evidence(overlap=True)
    summary = summarize(replay, rate_limit_results=all_rows)
    calibration = summary["calibration_warmth"]

    assert calibration["status"] == "caution"
    assert calibration["exact_overlap_status"] == "available"
    assert calibration["overlapping_request_body_sha256_count"] == 1
    assert calibration["replay_rows_with_calibrated_payload"] == 1
    assert calibration["replay_share_with_calibrated_payload"] == 0.5
    assert "must not be described as cold-cache" in calibration["warning"]
    markdown = render_markdown(summary, "calibrated")
    html = render_html(summary, "calibrated")
    assert "CAUTION (calibration warm state)" in markdown
    assert "CALIBRATION_WARM_STATE" in markdown
    assert "Calibration and warm state" in html
    assert "CALIBRATION_WARM_STATE" in html


def test_calibration_without_exact_overlap_still_warns_about_warm_state():
    replay, all_rows = _calibration_evidence(overlap=False)
    calibration = summarize(
        replay, rate_limit_results=all_rows)["calibration_warmth"]

    assert calibration["exact_overlap_status"] == "available"
    assert calibration["overlapping_request_body_sha256_count"] == 0
    assert calibration["replay_rows_with_calibrated_payload"] == 0
    assert "No exact request-body SHA-256 overlap" in calibration["warning"]
    assert "still warms endpoint" in calibration["warning"]


def test_calibration_overlap_is_unavailable_when_any_hash_is_missing():
    replay, all_rows = _calibration_evidence(
        overlap=True, missing_hash=True)
    calibration = summarize(
        replay, rate_limit_results=all_rows)["calibration_warmth"]

    assert calibration["exact_overlap_status"] == "unavailable"
    assert calibration["overlapping_request_body_sha256_count"] is None
    assert calibration["replay_rows_with_calibrated_payload"] is None
    assert calibration["replay_body_hashes_reported"] == 1
    assert "Exact payload overlap is unavailable" in calibration["warning"]


def test_no_calibration_has_no_warm_state_warning():
    replay = _rows(2)
    for index, row in enumerate(replay):
        row.update(phase="replay", request_body_sha256=("ab" * 32))
    calibration = summarize(
        replay, rate_limit_results=replay)["calibration_warmth"]

    assert calibration["status"] == "not_run"
    assert calibration["calibration_requests"] == 0
    assert calibration["warning"] is None


def test_tiny_trailing_window_cannot_manufacture_a_verdict():
    """A run whose duration is not a multiple of the window leaves a partial
    trailing window. One slow request in it must not become a trend: a p95
    over a handful of requests is one outlier away from inventing one."""
    steady = _rows(400, base_ttft=1000.0, t0=0.0, dt=0.3)     # windows 0 and 1
    tail = _rows(1, base_ttft=4000.0, t0=125.0)               # window 2, n=1
    d = _drift_block(steady + tail)
    assert d["windows"][-1]["n"] == 1
    assert d["windows"][-1]["counted"] is False
    assert d["skipped_windows"] == 1
    assert d["drift_kind"] == "stable"       # not "degrading"
    assert d["drift_flag"] is False


def test_two_windows_cannot_name_a_direction():
    """Two points separate nothing. The run is still flagged unstable, but no
    trend is claimed off it."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=400.0, t0=70.0, dt=1.0)
    d = _drift_block(a + b)
    assert d["drift_flag"] is True
    assert d["drift_kind"] == "variable"
    assert "not enough to call a direction" in d["drift_headline"]


def test_no_usable_window_says_so_instead_of_stable():
    """Every window too small to count. The report must not print a stable
    verdict it has no data for."""
    a = _rows(3, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(3, base_ttft=9000.0, t0=70.0, dt=1.0)
    d = _drift_block(a + b)
    assert "drift_kind" not in d
    assert "cannot be judged" in d["note"]
    h = render_html(summarize(a + b), "nodata")
    assert "not enough data" in h
    assert "pill ok'>stable" not in h


def test_windows_with_no_ttft_are_not_counted():
    """A window whose requests all failed to produce a TTFT has p95 None. It
    must not be compared by value against the real windows."""
    good = _rows(25, base_ttft=1000.0, t0=0.0, dt=1.0)
    blind = [dict(r, ttft_ms=None) for r in _rows(25, t0=70.0, dt=1.0)]
    later = _rows(25, base_ttft=5000.0, t0=140.0, dt=1.0)
    d = _drift_block(good + blind + later)
    assert d["windows"][1]["ttft_p95"] is None
    assert d["windows"][1]["counted"] is False
    assert d["drift_kind"] == "variable"     # 2 counted windows, no direction


def test_report_states_which_harness_version_and_latency_basis():
    """A 0.2.x TTFT included connection setup and a 0.3.x TTFT does not, so a
    report has to say which it is before anyone puts two in one column."""
    s = summarize(_rows(120))
    # pinned to the package, not a literal, so a version bump does not
    # need a test edit and cannot silently stop being stamped
    assert s["harness_version"] == __version__
    assert "NOT included" in s["latency_basis"]
    assert "immediately before conn.request" in s["latency_basis"]
    assert "include request upload" in s["latency_basis"]
    assert "first bounded response-body chunk" in s["latency_basis"]
    assert "HTTPResponse.read1" in s["latency_basis"]
    assert "not necessarily the first response byte" in s["latency_basis"]
    assert "visible, reasoning, or refusal delta" in s["latency_basis"]
    assert "excludes tool-call fragments" in s["latency_basis"]
    assert "first visible content and first tool-call fragment remain separate" \
        in s["latency_basis"]
    assert "latency basis" in render_markdown(s, "v")
    html = render_html(s, "v")
    assert "Latency basis" in html
    assert "TTFB (first bounded response-body chunk)" in html
    assert "TTFB (first byte)" not in html
    assert "TTFT (configured first content)" in html
    assert "TTFT (first token)" not in html


def test_first_visible_is_primary_in_both_customer_reports():
    rows = _rows(120)
    for row in rows:
        row["ttfv_ms"] = row["ttft_ms"] + 800.0
        row["caller_ttfv_ms"] = row["ttfv_ms"] + 20.0
        row["caller_ttft_ms"] = row["ttft_ms"] + 20.0
    summary = summarize(rows, ttft_definition="first_visible")

    markdown = render_markdown(summary, "visible")
    html = render_html(summary, "visible")

    assert "TTFV (configured first visible content)" in markdown
    assert "TTFT (first content; diagnostic)" in markdown
    assert markdown.index("TTFV (configured") < markdown.index(
        "TTFT (first content; diagnostic)")
    assert "Exact caller TTFV p50" in html
    assert "Exact caller TTFT p50" not in html
    assert html.index("TTFV (configured") < html.index(
        "TTFT (first content; diagnostic)")


def test_mixed_response_models_invalidate_single_model_benchmark():
    rows = _rows(600, dt=0.25)
    for i, row in enumerate(rows):
        row.update({
            "status": 200,
            "response_model": (
                "databricks-glm-5-2" if i < 300 else "wrong-model"),
            "response_object": "chat.completion.chunk",
            "system_fingerprint": "fp-a" if i < 300 else "fp-b",
        })
    summary = summarize(
        rows,
        run_meta={"endpoint_model": "databricks-glm-5-2"},
        acceptance={
            "ttft_ms": {"p95": 500.0},
            "success_rate": 0.99,
        },
    )

    identity = summary["response_identity"]
    assert identity["status"] == "invalid"
    assert identity["models"]["counts"] == {
        "databricks-glm-5-2": 300,
        "wrong-model": 300,
    }
    assert identity["unexpected_models"] == ["wrong-model"]
    html = render_html(summary, "mixed model")
    assert "Measurement invalid" in html
    assert "multiple response model values" in html


def test_consistent_response_model_is_bound_and_fingerprint_rotation_is_context():
    rows = _rows(120)
    for i, row in enumerate(rows):
        row.update({
            "status": 200,
            "response_model": "databricks-glm-5-2",
            "response_object": "chat.completion.chunk",
            "system_fingerprint": "fp-a" if i < 60 else "fp-b",
        })
    summary = summarize(
        rows,
        run_meta={"endpoint_model": "databricks-glm-5-2"},
    )

    identity = summary["response_identity"]
    assert identity["status"] == "bound"
    assert identity["invalid"] is None
    assert identity["request_model_match"] == "exact"
    assert identity["system_fingerprints"]["distinct_values_at_least"] == 2


def test_consistent_resolved_model_alias_is_route_bound_but_unverified():
    rows = _rows(20)
    for row in rows:
        row.update({
            "status": 200,
            "response_model": "vendor-model-2026-08-10",
            "served_model_name": "production-route-a",
            "response_object": "chat.completion.chunk",
        })
    summary = summarize(rows, run_meta={
        "endpoint_model": "vendor-model-latest",
        "endpoint_metadata": {
            "name": "customer-endpoint",
            "served_entities": [{"name": "production-route-a"}],
        },
    })

    identity = summary["response_identity"]
    assert identity["status"] == "caution"
    assert identity["invalid"] is None
    assert identity["request_model_match"] == \
        "consistent_difference_unverified"
    assert identity["unexpected_models"] == ["vendor-model-2026-08-10"]
    assert "binds the route but does not prove" in identity["warning"]
    assert "RESPONSE_MODEL_IDENTITY_UNVERIFIED" in \
        render_markdown(summary, "alias")


def test_consistent_different_model_without_binding_is_caution_not_bound():
    rows = _rows(20)
    for row in rows:
        row.update({
            "status": 200,
            "response_model": "wrong-or-resolved-model",
            "response_object": "chat.completion.chunk",
        })
    identity = summarize(
        rows, run_meta={"endpoint_model": "requested-model"})[
            "response_identity"]

    assert identity["status"] == "caution"
    assert identity["invalid"] is None
    assert identity["request_model_match"] == \
        "consistent_difference_unverified"
    assert "trusted alias map" in identity["warning"]


def test_custom_endpoint_name_is_not_mistaken_for_response_model_identity():
    rows = _rows(12)
    for row in rows:
        row.update({
            "status": 200,
            "response_model": "qwen-underlying-model",
            "served_model_name": "production-route-a",
            "response_object": "chat.completion.chunk",
        })
    summary = summarize(rows, run_meta={
        "endpoint_model": None,
        "endpoint_metadata": {
            "name": "customer-pt-endpoint",
            "served_entities": [{"name": "production-route-a"}],
        },
    })

    identity = summary["response_identity"]
    assert identity["status"] == "bound"
    assert identity["invalid"] is None
    assert identity["expected_models"] == []
    assert identity["expected_served_model_names"] == ["production-route-a"]


def test_served_model_header_must_match_an_active_control_plane_entity():
    rows = _rows(12)
    for row in rows:
        row.update({
            "status": 200,
            "response_model": "qwen-underlying-model",
            "served_model_name": "stale-route",
            "response_object": "chat.completion.chunk",
        })
    summary = summarize(rows, run_meta={
        "endpoint_metadata": {
            "name": "customer-pt-endpoint",
            "served_entities": [{"name": "production-route-a"}],
        },
    })

    identity = summary["response_identity"]
    assert identity["status"] == "invalid"
    assert identity["unexpected_served_model_names"] == ["stale-route"]


def _fail(n, t0=0.0, dt=1.0):
    return [{"ok": False, "t_send_unix": t0 + i * dt, "ttft_ms": None,
             "e2e_ms": None, "error": "upstream timeout", "status": 504}
            for i in range(n)]


def test_endpoint_collapsing_into_errors_is_not_stable():
    """The breaking-point run PRODUCTION_TESTING stage 2 tells you to do. The
    endpoint falls over in the last window, most requests fail, and the few
    survivors come back fast. Scoring successes alone reads that as steady,
    which is the worst possible answer for a test whose whole purpose is
    finding where the endpoint bends."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)   # fast survivors
    rows += _fail(140, t0=140.0, dt=0.3)                   # the collapse
    d = _drift_block([r for r in rows if r["ok"]],
                     [r for r in rows if not r["ok"]])
    assert d["drift_kind"] == "failing"
    assert d["drift_flag"] is True
    assert "84 percent" in d["drift_headline"]
    assert "surviving numbers" in d["drift_headline"]
    assert "latency comparison is not reported for a failing run" \
        in d["drift_headline"]
    # the named window is the biggest failure, so the clause reconciling it
    # against the highest RATE has to be there too, or the two disagree
    assert "highest loss rate was window 3" in d["drift_headline"]


def test_a_collapsing_window_is_judged_for_errors_not_for_latency():
    """The window where the endpoint broke has few SUCCESSES. It must still
    reach the error verdict, which is sized on ATTEMPTS, while staying out of
    the latency comparison, whose p95 would be survivors only."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)
    fails = _fail(140, t0=140.0, dt=0.3)
    d = _drift_block(rows, fails)
    collapsed = [w for w in d["windows"] if w["window"] == 2][0]
    assert collapsed["n"] == 25              # few successes
    assert collapsed["errors"] == 134
    assert collapsed["error_counted"] is True   # reaches the error verdict
    assert collapsed["counted"] is False        # excluded from latency


def test_per_window_errors_render_in_both_formats():
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.5)
    rows += _rows(60, base_ttft=205.0, t0=70.0, dt=0.5)
    fails = _fail(40, t0=70.0, dt=0.5)
    s = summarize(rows + fails)
    md = render_markdown(s, "errs")
    h = render_html(s, "errs")
    assert "errors" in md
    assert "<th scope='col'>errors</th>" in h
    assert "40 (" in md          # count and share shown together


def test_a_uniformly_lossy_run_is_not_called_failing():
    """Steady 8 percent errors across every window is a bad endpoint, but it
    is not a breaking point, and the error rate is already reported. Only a
    window that is materially worse than the rest earns the failing verdict."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(60, base_ttft=200.0 + w, t0=t0, dt=0.5)
        fails += _fail(5, t0=t0, dt=0.5)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] != "failing"


def test_a_total_outage_window_is_not_dropped_for_having_no_p95():
    """The window where every request failed has no p95 at all. Gating the
    error verdict on the latency gate would make a total outage invisible,
    which is worse than the partial-collapse bug."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=205.0, t0=140.0, dt=0.3)
    fails = _fail(150, t0=70.0, dt=0.3)
    d = _drift_block(rows, fails)
    dead = [w for w in d["windows"] if w["n"] == 0][0]
    assert dead["errors"] == 150
    assert dead["ttft_p95"] is None
    assert d["drift_kind"] == "failing"


def test_a_run_failing_in_every_window_is_still_failing():
    """Past the knee, every window sheds requests, so worst and best error
    rates are both high and a delta test alone cannot see it."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(70, base_ttft=200.0 + w, t0=t0, dt=0.3)
        fails += _fail(30, t0=t0, dt=0.3)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"


def test_a_shedding_window_cannot_anchor_the_latency_spread():
    """The collapsed window's survivors are fast, so letting it into the
    latency comparison makes the fastest number in the table the one the
    endpoint produced while falling over."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)   # fast survivors
    fails = _fail(140, t0=140.0, dt=0.3)
    d = _drift_block(rows, fails)
    collapsed = [w for w in d["windows"] if w["errors"] == 134][0]
    assert collapsed["p95_survivorship"] is True
    assert collapsed["counted"] is False
    # the failing branch returns before any latency comparison is computed,
    # so there is no "best" at all. this also fails loudly if the failing and
    # survivorship thresholds ever diverge enough for both to be reachable.
    assert "ttft_p95_best" not in d


def test_mild_uniform_loss_still_gets_a_latency_verdict():
    """Losing a few percent leaves a p95 worth comparing. Excluding those
    windows would silently drop the verdict on an otherwise healthy run."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(60, base_ttft=200.0 + w, t0=t0, dt=0.3)
        fails += _fail(5, t0=t0, dt=0.3)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "stable"
    assert all(w["counted"] for w in d["windows"])


def test_a_heavily_shedding_small_window_is_not_sized_out():
    """A breaking-point run ends in a trailing partial window. Sizing the
    error rule purely on median attempts would drop exactly the window the
    run exists to find."""
    rows = _rows(200, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(200, base_ttft=201.0, t0=70.0, dt=0.2)
    rows += _rows(200, base_ttft=202.0, t0=140.0, dt=0.2)
    rows += _rows(30, base_ttft=203.0, t0=210.0, dt=0.2)
    fails = _fail(15, t0=216.0, dt=0.2)          # 33 percent of a small window
    d = _drift_block(rows, fails)
    small = d["windows"][-1]
    assert small["attempts"] < 60                 # well under the median
    assert small["error_counted"] is True         # judged anyway
    assert d["drift_kind"] == "failing"


def test_a_run_where_everything_failed_says_so():
    """Zero successes must not fall through to 'stability was never
    established'. It is the most complete failure there is."""
    d = _drift_block([], _fail(50, t0=0.0) + _fail(50, t0=70.0))
    assert d["drift_kind"] == "failing"
    assert "no request belonged to the scored latency-outcome population" in \
        d["drift_headline"]


def test_the_named_window_is_the_largest_failure_not_the_highest_rate():
    """A tiny tail window at 100 percent should not outrank the window where
    a hundred requests actually died."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=70.0, dt=0.3)
    fails = _fail(120, t0=70.0, dt=0.3)      # big collapse, 83 percent
    fails += _fail(4, t0=140.0, dt=0.3)      # tiny tail, 100 percent
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"
    assert "window 1" in d["drift_headline"]      # the substantive one
    assert "100 percent" not in d["drift_headline"]


def test_retry_exhausted_failures_keep_their_original_send_time():
    """The client stamps the FIRST send, not the moment of final failure. A
    request retried past a read timeout would otherwise land whole windows
    later and invent a trailing window of errors."""
    import time
    from traffic_replay.client import EndpointClient, EndpointConfig

    class SlowFailingConn:
        """Connects, accepts the request, then dies. Each attempt burns time,
        the way a read timeout does."""
        sock = None

        def connect(self): pass

        def request(self, *a, **k):
            time.sleep(0.15)
            raise OSError("connection reset by peer")

        def close(self): pass

    cfg = EndpointConfig(base_url="http://127.0.0.1:1",
                         path="/serving-endpoints/x/invocations",
                         max_retries=2)
    c = EndpointClient(cfg, token=None)
    c._connect = lambda: SlowFailingConn()

    before = time.time()
    scheduled_monotonic = time.monotonic()
    r = c.send([{"role": "user", "content": "hi"}], 8, "req-1",
               scheduled_s=0.0, dispatch_lag_ms=0.0, intended=(0, 0, None, 0),
               chars_sent=2, scheduled_monotonic=scheduled_monotonic)
    after = time.time()

    assert r.ok is False
    # the whole call spanned at least two sleeps, so a final-failure stamp
    # would sit well after the first send
    assert after - before > 0.25
    assert r.first_send_unix < before + 0.15
    assert r.t_send_unix > r.first_send_unix
    assert r.connection_attempts == 3
    assert r.request_attempts == 3
    assert r.retry_reasons == ["transport_error_after_post",
                               "transport_error_after_post"]
    assert r.queue_wait_ms is not None
    assert r.caller_e2e_ms >= 400


def test_a_total_outage_actually_renders_its_verdict():
    """The zero-success block reaches summary.json, but both renderers used
    to gate on the window list, which is empty there, so the card printed no
    verdict at all while compare warned about the same run."""
    fails = [{"ok": False, "t_send_unix": float(i), "ttft_ms": None,
              "e2e_ms": None, "error": "upstream refused", "status": 503}
             for i in range(120)]
    s = summarize(fails)
    assert s["drift"]["drift_kind"] == "failing"
    md = render_markdown(s, "outage")
    h = render_html(s, "outage")
    assert "failing" in md.lower()
    assert "unstable: failing" in h
    assert "no request belonged to the scored latency-outcome population" in md


def test_one_stray_failure_does_not_flip_a_healthy_run():
    """A run whose duration is not a multiple of the window leaves a tiny
    tail. At low rates it holds a couple of requests, and one reset there
    must not read as a breaking point."""
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(60, base_ttft=201.0, t0=70.0, dt=0.2)
    d = _drift_block(rows, _fail(1, t0=125.0))
    assert d["drift_kind"] != "failing"


def test_the_headline_window_always_trips_the_bar_itself():
    """Naming by absolute errors alone names the huge low-rate window, whose
    3 percent is a rounding error next to a 30 percent collapse, and whose
    rate can round to 0 percent on a bigger denominator."""
    rows = _rows(2000, base_ttft=200.0, t0=0.0, dt=0.02)     # big, clean-ish
    rows += _rows(70, base_ttft=201.0, t0=70.0, dt=0.2)
    fails = _fail(60, t0=0.0, dt=0.02)                       # 3 percent
    fails += _fail(30, t0=84.0, dt=0.2)                      # 30 percent
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"
    # the eligibility filter is what this pins: without it the argmax by
    # absolute errors names the big low-rate window instead.
    assert d["drift_headline"].startswith("window 1 failed 30 percent")
    assert "failed 0 percent" not in d["drift_headline"]


def test_a_measured_zero_dispatch_lag_prints_as_zero_not_nan():
    """A measured 0.0 is a real value. Collapsing it with `or` would print
    nan on every clean run, which is what the first fix did."""
    md = render_markdown(summarize(_rows(60)), "lag")
    assert "dispatch lag p95 0 ms" in md
    assert "nan" not in md


def test_the_window_table_is_a_real_markdown_table():
    """A GFM table cannot interrupt a paragraph. Without a blank line the
    whole stability block renders as literal pipes, and report.md is the file
    that gets pasted into a ticket."""
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(60, base_ttft=205.0, t0=70.0, dt=0.2)
    rows += _rows(60, base_ttft=210.0, t0=140.0, dt=0.2)
    md = render_markdown(summarize(rows), "tbl")
    block = md[md.index("stability over time"):].splitlines()
    header = next(
        i for i, line in enumerate(block) if line.startswith("| window |"))
    assert block[header - 1].strip() == ""      # blank line before the table


def test_a_total_outage_card_does_not_claim_per_window_p95():
    fails = [{"ok": False, "t_send_unix": float(i), "ttft_ms": None,
              "e2e_ms": None, "error": "refused", "status": 503}
             for i in range(60)]
    s = summarize(fails)
    assert "window p95 in ms" not in render_html(s, "o")
    assert "| window |" not in render_markdown(s, "o")


def _paced(n, offered_qps, service_s, pool, ttft=100.0, jitter=0.0):
    """Rows shaped like a run where the pool can only serve `pool` at a time
    and each request occupies a worker for `service_s`. Requests are stamped
    when a worker frees up, which is what an open-loop client against a
    saturated pool actually produces."""
    rnd = random.Random(7)
    rows, free = [], [0.0] * pool
    for i in range(n):
        want = i / offered_qps
        svc = service_s * (1.0 + rnd.uniform(0, jitter)) if jitter else service_s
        w = min(range(pool), key=lambda k: free[k])
        actual = max(want, free[w])
        free[w] = actual + svc
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + actual,
                     "ttft_ms": ttft, "ttfb_ms": 1.0, "e2e_ms": ttft * 2,
                     "connect_ms": 8.0,
                     # the dispatcher is fine, it just queues: this is the
                     # number that stays small while the client is drowning
                     "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    return rows


def test_a_saturated_pool_shows_up_as_wire_lateness_not_dispatch_lag():
    """ThreadPoolExecutor.submit() queues instead of blocking, so the
    dispatcher never notices a full pool. Measured on a real run: dispatch
    lag p95 of 5 ms while requests reached the endpoint 92 seconds late."""
    rows = _paced(240, offered_qps=8.0, service_s=1.0, pool=2)
    s = summarize(rows)
    arr = s["arrivals"]
    assert arr["dispatch_lag_ms"]["p95"] < 10           # dispatcher looks fine
    assert arr["wire_lateness_ms"]["p95"] > 10_000      # reality
    assert s["client"]["warning"] is not None
    # states the observation, not a cause it cannot know
    assert "did not start HTTP requests on schedule" in s["client"]["warning"]
    assert "endpoint receipt were not observed" in s["client"]["warning"]
    assert "read the stability card to tell them apart" in s["client"]["warning"]


def test_the_caution_is_above_the_tables_in_both_formats():
    rows = _paced(240, offered_qps=8.0, service_s=1.0, pool=2)
    s = summarize(rows)
    md = render_markdown(s, "sat")
    assert md.index("CAUTION (client saturation)") < md.index(
        "| final-attempt request-path metric (ms; clock starts immediately "
        "before conn.request; connection setup excluded) |")
    assert "banner warn" in render_html(s, "sat")


def test_a_client_that_keeps_up_is_not_warned():
    """The negative control. Verified against a real 20 rps run that the
    endpoint itself confirmed receiving at 20.7 rps: no caution."""
    rows = _paced(1200, offered_qps=20.0, service_s=0.06, pool=64)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_wire_lateness_is_reported_even_when_nothing_is_wrong():
    rows = _paced(600, offered_qps=20.0, service_s=0.06, pool=64)
    s = summarize(rows)
    md = render_markdown(s, "ok")
    assert "HTTP request-start lateness p95" in md
    assert s["arrivals"]["wire_lateness_ms"]["n"] == 600


def test_a_rate_shortfall_alone_is_enough_to_warn():
    """Isolates the shortfall arm: sends stay close to schedule for most of
    the run, so p95 lateness stays under a second and the drifting arm cannot
    fire, but the run still takes far longer than it was asked to."""
    rows = []
    for i in range(400):
        want = i / 10.0
        # on time for 96 percent of the run, then a hard stall at the end
        actual = want if i < 384 else want + 40.0
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + actual, "ttft_ms": 100.0,
                     "ttfb_ms": 1.0, "e2e_ms": 200.0, "connect_ms": 8.0,
                     "dispatch_lag_ms": 4.0, "prompt_tokens": 100,
                     "completion_tokens": 10})
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000     # drifting silent
    assert s["client"]["achieved_qps"] < s["client"]["offered_qps"] * 0.8
    # states what the span statistic supports, not "never"
    assert "fewer requests per second than the" in s["client"]["warning"]


def test_a_late_but_complete_run_does_not_claim_a_shortfall():
    """The drifting arm alone. The run average held, so the total load did
    arrive, and saying it was never driven at the rate would contradict the
    achieved figure printed two keys away."""
    # a transient stall that recovers, which is the real shape this arm
    # exists for: total load arrives, but not when the schedule wanted it
    rows = []
    for i in range(600):
        want = i / 20.0
        late = 4.0 if 200 <= i < 320 else 0.0     # 20 percent of the run
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + want + late,
                     "ttft_ms": 100.0, "ttfb_ms": 1.0, "e2e_ms": 200.0,
                     "connect_ms": 8.0, "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    s = summarize(rows)
    c = s["client"]
    assert c["achieved_qps"] >= c["offered_qps"] * 0.8      # no shortfall
    assert "fewer requests per second" not in c["warning"]
    assert "arrived reshaped" in c["warning"]


def test_heavy_retries_are_not_reported_as_a_client_shortfall():
    """offered and achieved must come from one population. Mixing them makes
    the ratio the non-retry fraction, so an endpoint dropping connections
    would read as a slow client, which is backwards."""
    for frac in (0.2, 0.3, 0.5):
        rows = _paced(400, offered_qps=20.0, service_s=0.04, pool=64)
        for i, r in enumerate(rows):
            if i % int(1 / frac) == 0:
                r["retries"] = 1
        s = summarize(rows)
        assert "client" not in s, f"false shortfall at retry fraction {frac}"


def test_a_healthy_run_with_jittery_service_times_stays_silent():
    """The negative control with zero variance proves too little. Real service
    times are heavy tailed, and that is the shape most likely to produce a
    false positive against the 1s threshold."""
    rows = _paced(1200, offered_qps=20.0, service_s=0.06, pool=64, jitter=4.0)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_the_printed_rates_reconcile_with_the_arrival_bullet():
    """The caution's 'delivered' figure and the believability block's achieved
    arrival rate describe the same run, so they must not disagree because a
    chunk of rows retried in the middle."""
    rows = []
    for i in range(500):
        want = i / 20.0
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + want * 1.6,
                     "ttft_ms": 100.0, "ttfb_ms": 1.0, "e2e_ms": 200.0,
                     "connect_ms": 8.0, "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    for r in rows[200:400]:
        r["retries"] = 1                    # 40 percent, mid-run
    s = summarize(rows)
    c = s["client"]
    assert c["offered_qps"] > 19.0          # the true offered rate, not 12
    bullet = s["arrivals"]["achieved_qps_overall"]
    assert abs(c["achieved_qps"] - bullet) / bullet < 0.15


def test_a_retried_row_is_timed_from_its_first_attempt():
    """t_send_unix belongs to whichever attempt produced the result, so on a
    retry it carries the endpoint's delay. first_send_unix says when the load
    was actually offered, and that is what client lateness must be built on.
    No row needs excluding once the honest stamp exists."""
    rows = _paced(200, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r["first_send_unix"] = r["t_send_unix"]
    # a request that failed, retried, then came back 120s later
    rows[10]["retries"] = 1
    rows[10]["t_send_unix"] += 120.0          # contaminated
    # first_send_unix left alone: it still says when the load went out
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["n"] == len(rows)   # nothing dropped
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000       # not blamed on the client
    assert "client" not in s


def test_every_retry_shape_is_timed_honestly():
    """The three client return paths (non-200, empty stream, exhausted) all
    carry first_send_unix, so none of them can inject endpoint delay into
    client lateness."""
    rows = _paced(300, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r["first_send_unix"] = r["t_send_unix"]
    for i, (status, ok) in enumerate([(503, False), (200, False), (None, False)]):
        r = rows[50 + i * 50]
        r["retries"] = 1
        r["status"] = status
        r["ok"] = ok
        r["t_send_unix"] += 130.0             # every one carries endpoint delay
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_rows_without_the_field_fall_back_to_t_send_unix():
    """A requests.jsonl written by an older harness has no first_send_unix.
    It should still produce a wire-lateness series rather than an empty one."""
    rows = _paced(120, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r.pop("first_send_unix", None)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["n"] == len(rows)


def test_the_client_distinguishes_connection_attempts_from_http_sends():
    """Drives the real EndpointClient rather than hand-built dicts. A response
    proves an HTTP send occurred; a connection refusal proves one did not."""
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from traffic_replay.client import EndpointClient, EndpointConfig

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = b'{"error":"nope"}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _time.sleep(0.2)
    try:
        cfg = EndpointConfig(base_url=f"http://127.0.0.1:{port}",
                             path="/serving-endpoints/x/invocations")
        c = EndpointClient(cfg, token=None)
        r = c.send([{"role": "user", "content": "hi"}], 8, "r1",
                   scheduled_s=0.0, dispatch_lag_ms=0.0,
                   intended=(0, 0, None, 0), chars_sent=2)
        assert r.ok is False and r.status == 503          # the non-200 path
        assert r.first_send_unix is not None
        assert r.first_attempt_unix <= r.first_send_unix
        assert r.request_attempts == 1
    finally:
        srv.shutdown()
        srv.server_close()

    # exhausted-retry path: nothing listening at all
    cfg2 = EndpointConfig(base_url="http://127.0.0.1:1",
                          path="/serving-endpoints/x/invocations",
                          max_retries=1)
    c2 = EndpointClient(cfg2, token=None)
    r2 = c2.send([{"role": "user", "content": "hi"}], 8, "r2",
                 scheduled_s=0.0, dispatch_lag_ms=0.0,
                 intended=(0, 0, None, 0), chars_sent=2)
    assert r2.ok is False
    assert r2.first_attempt_unix is not None
    assert r2.first_send_unix is None
    assert r2.request_attempts == 0


# ---- concurrency actually reached -----------------------------------------

def _spans(n, start_rate, service_s, t0=1_000_000.0):
    """Rows whose send times and durations produce a known overlap."""
    return [{"ok": True, "scheduled_s": i / start_rate,
             "t_send_unix": t0 + i / start_rate,
             "first_send_unix": t0 + i / start_rate,
             "ttft_ms": 100.0, "ttfb_ms": 1.0, "e2e_ms": service_s * 1000.0,
             "connect_ms": 8.0, "dispatch_lag_ms": 4.0,
             "prompt_tokens": 100, "completion_tokens": 10}
            for i in range(n)]


def test_concurrency_measures_actual_overlap():
    """20 rps against a 1.5s service time is 30 in flight by construction."""
    from traffic_replay.metrics import _concurrency_block
    rows = _spans(600, start_rate=20.0, service_s=1.5)
    c = _concurrency_block(rows, asked=30)
    assert 28 <= c["in_flight_p50"] <= 32
    assert "warning" not in c
    assert c["sizing_concurrency_requested"] == 30


def test_concurrency_warns_when_the_load_never_arrived():
    """The real failure: the endpoint sheds, so the run holds a fraction of
    what was asked and every latency number describes the lighter load."""
    from traffic_replay.metrics import _concurrency_block
    rows = _spans(600, start_rate=20.0, service_s=0.15)   # only ~3 in flight
    c = _concurrency_block(rows, asked=30)
    assert c["in_flight_p50"] < 10
    assert "sized from an unloaded estimate of 30" in c["warning"]
    assert "not a held concurrency target" in c["warning"]


def test_concurrency_caution_renders_above_the_tables():
    rows = _spans(600, start_rate=20.0, service_s=0.15)
    s = summarize(rows, concurrency_target=30)
    md = render_markdown(s, "conc")
    assert md.index("CAUTION (concurrency not reached)") < md.index(
        "| final-attempt request-path metric (ms; clock starts immediately "
        "before conn.request; connection setup excluded) |")
    assert "banner warn" in render_html(s, "conc")


def test_concurrency_is_reported_even_when_it_was_reached():
    rows = _spans(600, start_rate=20.0, service_s=1.5)
    s = summarize(rows, concurrency_target=30)
    assert "concurrency" in s
    assert "concurrency actually in flight" in render_markdown(s, "c")
    assert "Concurrency in flight" in render_html(s, "c")


def test_no_concurrency_block_without_enough_rows():
    from traffic_replay.metrics import _concurrency_block
    assert _concurrency_block(_spans(1, 20.0, 1.0), asked=30) is None


# ---- whose SLA targets are these ------------------------------------------

def test_the_scorecard_names_where_its_targets_came_from():
    rows = _rows(120)
    s = summarize(rows, acceptance={"targets_are": "yours, passed on the "
                                                   "command line",
                                    "ttft_ms": {"p95": 900}})
    assert s["sla"]["targets_source"] == "yours, passed on the command line"
    assert "targets_warning" not in s["sla"]
    assert "targets from yours" in render_markdown(s, "sla")


def test_illustrative_targets_are_flagged_so_they_do_not_read_as_yours():
    """A bundled profile ships example targets. Scoring MET and MISS against
    them without saying so invites someone to act on placeholder numbers."""
    rows = _rows(120)
    s = summarize(rows, acceptance={"ttft_ms": {"p95": 900},
                                    "note": "illustrative targets. replace "
                                            "with the ones you agreed."})
    assert "illustrative" in s["sla"]["targets_warning"]
    md = render_markdown(s, "sla")
    assert "CAUTION (targets)" in md
    assert "banner warn" in render_html(s, "sla")


def test_naming_the_source_does_not_suppress_the_illustrative_warning():
    """The runner now stamps targets_are on every run. The warning used to be
    conditional on that field being absent, so stamping it would have silently
    retired the one thing stopping a reader from acting on example numbers."""
    rows = _rows(120)
    s = summarize(rows, acceptance={"targets_are": "this profile",
                                    "ttft_ms": {"p95": 900},
                                    "note": "illustrative targets. replace "
                                            "with the ones you agreed."})
    assert s["sla"]["targets_source"] == "this profile"
    assert "illustrative" in s["sla"]["targets_warning"]
    assert "CAUTION (targets)" in render_markdown(s, "sla")


# ---- reasoning truncation makes ttfv a survivor number --------------------

def _reasoning_rows(n_visible, n_truncated):
    """Successful rows. The truncated ones ran out of output tokens while
    still reasoning, so they carry a ttfr but never a ttfv."""
    rows = []
    for i in range(n_visible):
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 900.0,
                     "ttfr_ms": 900.0, "ttfv_ms": 8000.0 + i,
                     "e2e_ms": 13000.0, "finish_reason": "stop"})
    for i in range(n_truncated):
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 900.0,
                     "ttfr_ms": 900.0, "ttfv_ms": None,
                     "e2e_ms": 23000.0, "finish_reason": "length"})
    for i, r in enumerate(rows):
        r["t_send_unix"] = 1_700_000_000.0 + i * 0.25
        r["first_send_unix"] = r["t_send_unix"]
    return rows


def test_ttfv_percentiles_say_how_many_requests_they_leave_out():
    s = summarize(
        _reasoning_rows(55, 132),
        acceptance={"ttft_ms": {"p50": 500}},
    )
    assert s["ttfv_ms"]["missing"] == 132
    assert s["ttfv_ms"]["of"] == 187
    note = render_markdown(s, "note")
    assert "55 of 187" in note
    assert "visible-content subset" in note
    assert "artifact does not establish why" in note
    assert "first visible-or-reasoning content delta" in note
    assert "first visible content" in note
    assert "first token of" not in note
    html = render_html(s, "note")
    assert "first visible-or-reasoning content delta" in html
    assert "first visible content" in html
    assert "first token of" not in html


def test_scoring_first_visible_warns_when_most_requests_never_got_there():
    """The scorecard grades TTFT against ttfv when the SLA scores the first
    visible token. Marking MET or MISS off the 29% that finished thinking
    would read as a verdict on the whole run."""
    s = summarize(_reasoning_rows(55, 132),
                  acceptance={"ttft_ms": {"p50": 500}},
                  ttft_definition="first_visible")
    w = s["sla"]["coverage_warning"]
    assert "132 of 187" in w and "ttfv_ms" in w
    assert "CAUTION (coverage)" in render_markdown(s, "sla")
    assert "banner warn" in render_html(s, "sla")


def test_no_coverage_warning_when_every_request_produced_visible_text():
    s = summarize(_reasoning_rows(120, 0),
                  acceptance={"ttft_ms": {"p50": 500}},
                  ttft_definition="first_visible")
    assert "coverage_warning" not in s["sla"]
    assert s["ttfv_ms"]["missing"] == 0


# ---- transport success is not answer success ------------------------------

def _answer_rows(answered, silent, truncated_but_visible=0):
    """Rows as the client now writes them. `silent` returned HTTP 200 with a
    well formed stream and nothing readable, which is what a reasoning model
    does when it spends the whole budget thinking."""
    rows = []
    for _ in range(answered):
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 900.0,
                     "ttfv_ms": 950.0, "e2e_ms": 1200.0,
                     "stream_complete": True, "visible_content_seen": True,
                     "truncated": False, "parse_errors": 0,
                     "finish_reason": "stop"})
    for _ in range(truncated_but_visible):
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 900.0,
                     "ttfv_ms": 950.0, "e2e_ms": 1200.0,
                     "stream_complete": True, "visible_content_seen": True,
                     "truncated": True, "parse_errors": 0,
                     "finish_reason": "length"})
    for _ in range(silent):
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 900.0,
                     "ttfv_ms": None, "e2e_ms": 1200.0,
                     "stream_complete": True, "visible_content_seen": False,
                     "truncated": True, "parse_errors": 0,
                     "finish_reason": "length"})
    for i, r in enumerate(rows):
        r["t_send_unix"] = 1_700_000_000.0 + i * 0.25
        r["first_send_unix"] = r["t_send_unix"]
    return rows


def test_a_200_with_no_visible_content_is_not_a_successful_answer():
    s = summarize(_answer_rows(answered=55, silent=132))
    a = s["answers"]
    assert a["transport_ok"] == 187
    assert a["answered"] == 55
    assert a["no_visible_content"] == 132
    assert a["answer_rate"] == 55 / 187
    assert s["ttft_ms"]["n"] == 55
    assert s["latency_population"]["kind"] == "readable_answers"
    md = render_markdown(s, "answers")
    assert "produced at least one visible or reasoning content delta" in md
    assert "returned HTTP 200:" not in md  # status was not retained by rows


def test_silent_responses_count_against_the_success_rate():
    """The defect this guards: 187 requests, zero errors, zero readable
    answers, reported as a 100 percent success rate."""
    s = summarize(_answer_rows(answered=0, silent=100),
                  acceptance={"success_rate": 0.99})
    assert s["sla"]["success_rate"]["actual"] == 0.0
    assert s["sla"]["success_rate"]["met"] is False


def test_truncation_alone_is_not_a_failure():
    """The harness caps max_tokens at the sampled output size on purpose, so
    finishing on "length" is how a run hits its target output length."""
    s = summarize(_answer_rows(answered=0, silent=0, truncated_but_visible=50),
                  acceptance={"success_rate": 0.99})
    assert s["answers"]["truncated"] == 50
    assert s["answers"]["answered"] == 50
    assert s["sla"]["success_rate"]["met"] is True


def test_a_run_with_no_answers_at_all_renders_invalid_not_green():
    s = summarize(_answer_rows(answered=0, silent=80),
                  acceptance={"ttft_ms": {"p50": 500}},
                  ttft_definition="first_visible")
    assert "invalid" in s["answers"]
    html = render_html(s, "no answers")
    assert "INVALID" in html
    assert "Meets every acceptance target" not in html
    md = render_markdown(s, "no answers")
    assert "verdict: INVALID" in md


def test_an_unmeasured_target_is_not_scored_as_a_pass():
    """met is None used to count as a pass, so a target with nothing behind
    it rendered the green banner."""
    # p75 is not one of the quantiles the summary computes, so this target
    # has no measurement behind it while the run itself is healthy
    s = summarize(_answer_rows(answered=40, silent=0),
                  acceptance={"ttft_ms": {"p50": 5000, "p75": 5000}})
    rows = [r for k in ("ttft_vs_target", "ttfg_vs_target")
            for r in s["sla"][k]]
    assert any(r["met"] is None for r in rows), "need an unmeasured row"
    html = render_html(s, "partial")
    assert "Meets every acceptance target" not in html
    assert "not measured" in render_markdown(s, "partial")


# ---- the two renderers must not disagree about the verdict ---------------

def _mixed(silent, good):
    r = [{"ok": True, "phase": "replay", "ttft_ms": 100.0, "ttfr_ms": 100.0,
          "ttfv_ms": None, "e2e_ms": 200.0, "stream_complete": True,
          "visible_content_seen": False, "truncated": True,
          "parse_errors": 0, "finish_reason": "length"} for _ in range(silent)]
    r += [{"ok": True, "phase": "replay", "ttft_ms": 100.0, "ttfr_ms": 100.0,
           "ttfv_ms": 110.0, "e2e_ms": 200.0, "stream_complete": True,
           "visible_content_seen": True, "truncated": False,
           "parse_errors": 0, "finish_reason": "stop"} for _ in range(good)]
    for i, x in enumerate(r):
        x["t_send_unix"] = 1_700_000_000.0 + i * 0.25
        x["first_send_unix"] = x["t_send_unix"]
    return r


def _md_verdict(s):
    return [line for line in render_markdown(s, "x").splitlines()
            if line.startswith("verdict:")][0]


def test_an_answer_collapse_is_not_green_without_a_success_rate_target():
    """success_rate is optional, and configs/run_pt_full.json omits it. With
    no success-rate row there was nothing for a collapse in readable answers
    to miss, so 55 of 187 answered still rendered the green banner."""
    s = summarize(_mixed(132, 55),
                  acceptance={"ttft_ms": {"p50": 5000},
                              "ttfg_ms": {"p50": 5000}})
    assert s["answers"]["answer_rate"] < 0.30
    assert "Meets every acceptance target" not in render_html(s, "x")
    assert "132 of 187" in _md_verdict(s)


def test_markdown_and_html_agree_on_the_verdict():
    """They each used to compute their own. The html counted the success-rate
    row and the markdown did not, so report.md, the file people paste into
    email, called a failing run a pass."""
    for silent, good, acc in (
            (132, 55, {"ttft_ms": {"p50": 5000}, "success_rate": 0.99}),
            (132, 55, {"ttft_ms": {"p50": 5000}, "ttfg_ms": {"p50": 5000}}),
            (0, 187, {"ttft_ms": {"p50": 5000}, "success_rate": 0.99}),
            (187, 0, {"ttft_ms": {"p50": 5000}})):
        s = summarize(_mixed(silent, good), acceptance=acc)
        green_html = "Meets every acceptance target" in render_html(s, "x")
        green_md = _md_verdict(s) == "verdict: meets every acceptance target"
        assert green_html == green_md, (silent, good, acc, _md_verdict(s))


def test_a_success_rate_miss_reaches_the_markdown_verdict():
    s = summarize(_mixed(0, 100), acceptance={"success_rate": 0.99})
    s["sla"]["success_rate"] = {"target": 0.99, "actual": 0.5, "met": False}
    assert "missed" in _md_verdict(s) or "without a readable" in _md_verdict(s)


def test_the_invalid_sentence_names_the_counter_that_drove_it():
    """It used to assert every request produced no visible content, which is
    false when the real cause was a stream that never terminated, and it sat
    directly under a no_visible_content of 0."""
    rows = _mixed(0, 60)
    for r in rows:
        r["stream_complete"] = False
    s = summarize(rows, acceptance={"ttft_ms": {"p50": 5000}})
    inv = s["answers"]["invalid"]
    assert s["answers"]["no_visible_content"] == 0
    assert "never terminated their stream" in inv
    assert "60 of 60" in inv


def test_parse_errors_do_not_get_misreported_as_missing_visible_content():
    rows = _mixed(0, 12)
    for row in rows:
        row["parse_errors"] = 1
    summary = summarize(rows, acceptance={"ttft_ms": {"p50": 5000}})

    invalid = summary["answers"]["invalid"]
    assert summary["answers"]["no_visible_content"] == 0
    assert "produced a reportable completed answer" in invalid
    assert "hit unrecoverable parse errors" in invalid
    assert "12 of 12" in invalid
    assert "produced visible content or a valid tool call" not in invalid


def test_old_rows_are_not_retroactively_failed_by_the_answers_block():
    """Merging a 0.3.0 run dir with a 0.4.0 one used to report answer_rate
    0.5 next to a success rate of 1.0, because the guard was all-or-nothing
    while the SLA block guards per row."""
    new = _mixed(0, 50)
    old = [{"ok": True, "phase": "replay", "ttft_ms": 100.0, "e2e_ms": 200.0,
            "finish_reason": "stop",
            "t_send_unix": 1_700_000_100.0 + i * 0.25,
            "first_send_unix": 1_700_000_100.0 + i * 0.25} for i in range(50)]
    s = summarize(new + old, acceptance={"success_rate": 0.99})
    a = s["answers"]
    assert a["scored"] == 50, "only rows carrying the field are scored"
    assert a["transport_ok"] == 100
    assert a["answer_rate"] == 1.0
    assert s["sla"]["success_rate"]["met"] is True


# ---- concurrency is measured exactly, not sampled ------------------------

def test_a_brief_spike_reaches_the_reported_peak():
    """The old implementation took 41 samples across the run and called the
    highest one the peak. A spike shorter than the gap between samples was
    invisible. This builds a run that sits at 2 in flight and spikes to 12
    for 40 ms, which 41 samples over 100 seconds would miss."""
    base = 1_700_000_000.0
    rows = []
    # steady background: 2 in flight across 100 seconds
    for i in range(100):
        rows.append({"ok": True, "phase": "replay", "e2e_ms": 2000.0,
                     "t_send_unix": base + i, "first_send_unix": base + i})
    # a 40 ms spike of 10 extra requests, right in the middle of the run
    for i in range(10):
        rows.append({"ok": True, "phase": "replay", "e2e_ms": 40.0,
                     "t_send_unix": base + 50.0,
                     "first_send_unix": base + 50.0})
    c = _concurrency_block(rows, None)
    assert c["in_flight_max"] >= 12, c
    # and the spike is brief, so it must not drag the time-weighted median
    assert c["in_flight_p50"] <= 3, c


def test_concurrency_percentiles_are_time_weighted():
    """A level held briefly must not count the same as one held throughout."""
    base = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "e2e_ms": 100_000.0,
             "t_send_unix": base, "first_send_unix": base} for _ in range(4)]
    rows += [{"ok": True, "phase": "replay", "e2e_ms": 10.0,
              "t_send_unix": base + 50.0, "first_send_unix": base + 50.0}
             for _ in range(20)]
    c = _concurrency_block(rows, None)
    assert c["in_flight_p50"] == 4, c
    assert c["in_flight_max"] >= 24, c


def test_empty_middle_load_window_reports_zero_occupancy_and_sizing_warning():
    base = 1_700_000_000.0
    rows = [
        {"ok": True, "phase": "replay", "first_send_unix": base,
         "t_send_unix": base, "finished_unix": base + 1.0},
        {"ok": True, "phase": "replay", "first_send_unix": base + 10.0,
         "t_send_unix": base + 10.0, "finished_unix": base + 11.0},
    ]

    concurrency = _concurrency_block(rows, asked=30)

    assert concurrency["in_flight_p50"] == 0.0
    assert concurrency["in_flight_p95"] == 0.0
    assert concurrency["in_flight_max"] == 1.0
    assert "observed in-flight p50 was 0" in concurrency["warning"]


def test_equal_send_timestamp_burst_retains_peak_and_explicit_window_caution():
    base = 1_700_000_000.0
    rows = [
        {"ok": True, "phase": "replay", "first_send_unix": base,
         "t_send_unix": base, "finished_unix": base + 1.0}
        for _ in range(7)
    ]

    concurrency = _concurrency_block(rows, asked=None)

    assert concurrency["in_flight_max"] == 7.0
    assert concurrency["in_flight_p50"] == 7.0
    assert "same timestamp" in concurrency["warning"]
    assert "response-drain interval" in concurrency["method"]


# ---- rate conventions and observation windows ---------------------------

def test_the_arrival_rate_uses_the_send_span_not_the_drain():
    """Throughput is divided by the observation interval, which runs to the
    last completion. The arrival rate must not be: charging it for the drain
    understates the load that was actually offered."""
    base = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "ttft_ms": 50.0, "e2e_ms": 5000.0,
             "prompt_tokens": 100, "completion_tokens": 10,
             "scheduled_s": i * 0.1,
             "t_send_unix": base + i * 0.1,
             "first_send_unix": base + i * 0.1} for i in range(100)]
    s = summarize(rows)
    # sent at exactly 10 per second
    assert abs(s["arrivals"]["achieved_qps_overall"] - 10.0) < 1e-6
    # 1000 output tokens over a 14.9s observation interval, not 9.9s
    expected = 1000 / (14.9 / 60.0)
    assert abs(s["throughput"]["output_tokens_per_min"] - expected) < 1.0


def test_sparse_shard_arrivals_use_the_complete_logical_schedule_window():
    base = 1_700_000_000.0
    rows = [
        {"ok": True, "phase": "replay", "ttft_ms": 50.0,
         "e2e_ms": 100.0, "prompt_tokens": 100,
         "completion_tokens": 10, "scheduled_s": stamp,
         "caller_send_ms": 0.0, "t_send_unix": base + stamp,
         "first_send_unix": base + stamp, "request_attempts": 1}
        for stamp in (30.0, 30.1)
    ]

    summary = summarize(rows, schedule_meta={
        "seconds": 60, "requests": 2, "shard": "2/4",
        "rates_describe": "the whole run, not this shard",
    })
    arrivals = summary["arrivals"]

    assert abs(arrivals["on_wire_qps_active_span"] - 10.0) < 1e-4
    assert arrivals["scheduled_qps"] == 2 / 60
    assert arrivals["achieved_qps_overall"] == 2 / 60
    assert arrivals["logical_schedule_seconds"] == 60.0
    assert arrivals["scheduled_qps_basis"] == \
        "scheduled requests / logical schedule seconds"


def test_sparse_schedule_token_throughput_and_cost_share_full_window():
    base = 1_700_000_000.0
    rows = [
        {"ok": True, "phase": "replay", "ttft_ms": 50.0,
         "e2e_ms": 100.0, "prompt_tokens": 100,
         "completion_tokens": 10, "scheduled_s": stamp,
         "caller_send_ms": 0.0, "t_send_unix": base + stamp,
         "first_send_unix": base + stamp,
         "finished_unix": base + stamp + 0.1,
         "request_attempts": 1, "retries": 0, "retry_reasons": []}
        for stamp in (30.0, 30.1)
    ]

    summary = summarize(
        rows,
        schedule_meta={"seconds": 60, "requests": 2},
        pricing={"mode": "provisioned", "dbu_per_hour": 10.0})

    throughput = summary["throughput"]
    assert throughput["input_tokens_per_min"] == 200.0
    assert throughput["output_tokens_per_min"] == 20.0
    assert throughput["observation_seconds"] == 60.0
    assert throughput["duration_basis"] == \
        "max(logical_schedule_seconds,response_drain)"
    assert summary["cost"]["observation_seconds"] == \
        throughput["observation_seconds"]


def _partially_delivered_schedule(sent_indexes: set[int]) -> list[dict]:
    base = 1_700_000_000.0
    rows = []
    for i in range(100):
        common = {
            "scheduled_s": i * 0.1,
            "dispatch_lag_ms": 0.0,
            # Presence of this exact field distinguishes current rows from
            # legacy evidence; None is proof that no POST began.
            "caller_send_ms": 0.0 if i in sent_indexes else None,
        }
        if i in sent_indexes:
            rows.append({
                **common, "ok": True, "phase": "replay",
                "ttft_ms": 50.0, "e2e_ms": 100.0,
                "t_send_unix": base + i * 0.1,
                "first_send_unix": base + i * 0.1,
                "prompt_tokens": 100, "completion_tokens": 10,
                "request_attempts": 1,
            })
        else:
            rows.append({
                **common, "ok": False, "phase": "replay",
                "error": "pending limit reached before worker start",
                "request_attempts": 0,
            })
    return rows


def test_unsent_tail_cannot_report_full_scheduled_qps():
    summary = summarize(_partially_delivered_schedule(set(range(50))))
    arrivals = summary["arrivals"]

    assert arrivals["scheduled_qps"] == 10.0
    assert abs(arrivals["on_wire_qps_active_span"] - 10.0) < 1e-6
    assert abs(arrivals["achieved_qps_overall"] - 5.0) < 1e-6
    assert arrivals["schedule_delivery_fraction"] == 0.5
    assert arrivals["scheduled_requests_not_sent"] == 50
    assert summary["client"]["scheduled_requests_not_sent"] == 50
    assert "never reached an HTTP POST" in summary["client"]["warning"]


def test_interleaved_unsent_requests_reduce_delivery_rate_too():
    summary = summarize(
        _partially_delivered_schedule(set(range(0, 100, 2))))
    arrivals = summary["arrivals"]

    assert arrivals["schedule_delivery_fraction"] == 0.5
    assert abs(arrivals["achieved_qps_overall"] - 5.0) < 0.1
    assert arrivals["on_wire_qps_active_span"] < 6.0
    assert "client" in summary


def test_unsent_tail_and_stretched_sent_prefix_are_not_double_penalized():
    rows = _partially_delivered_schedule(set(range(50)))
    base = 1_700_000_000.0
    for i, row in enumerate(rows[:50]):
        # Sent prefix arrives at 5 rps across almost the entire 10-second
        # logical window. The other half is unsent. That is 5 delivered rps,
        # not 2.5 from multiplying the same shortfall twice.
        row["caller_send_ms"] = i * 100.0
        row["first_send_unix"] = base + i * 0.2
        row["t_send_unix"] = base + i * 0.2

    arrivals = summarize(rows)["arrivals"]

    assert abs(arrivals["on_wire_qps_active_span"] - 5.0) < 1e-6
    assert abs(arrivals["achieved_qps_overall"] - 5.0) < 0.1
    assert arrivals["schedule_delivery_fraction"] == 0.5


def test_failed_tail_extends_the_observation_window_instead_of_zero_duration():
    base = 1_700_000_000.0
    rows = [
        {"ok": True, "phase": "replay", "ttft_ms": 50.0,
         "e2e_ms": 100.0, "prompt_tokens": 100, "completion_tokens": 10,
         "t_send_unix": base + i * 0.1,
         "first_send_unix": base + i * 0.1,
         "finished_unix": base + i * 0.1 + 0.1}
        for i in range(10)
    ]
    rows.append(
        {"ok": False, "phase": "replay", "error": "read timeout",
         "e2e_ms": None, "caller_e2e_ms": 60_000.0,
         "t_send_unix": base + 0.95, "first_send_unix": base + 0.95,
         "finished_unix": base + 60.95})
    s = summarize(rows)
    expected = 1000 / (60.95 / 60.0)
    assert abs(s["throughput"]["input_tokens_per_min"] - expected) < 0.1
    assert s["throughput"]["completion_time_coverage"] == 1.0


def test_failed_requests_are_included_in_in_flight_occupancy():
    from traffic_replay.metrics import _concurrency_block

    base = 1_700_000_000.0
    successes = [
        {"ok": True, "e2e_ms": 100.0,
         "first_send_unix": base + i * 0.1,
         "t_send_unix": base + i * 0.1,
         "finished_unix": base + i * 0.1 + 0.1}
        for i in range(10)
    ]
    timeouts = [
        {"ok": False, "error": "read timeout", "e2e_ms": None,
         "first_send_unix": base + 0.95 + i * 0.001,
         "t_send_unix": base + 0.95 + i * 0.001,
         "finished_unix": base + 60.95 + i * 0.001}
        for i in range(20)
    ]
    c = _concurrency_block(successes + timeouts, asked=30)
    assert c["in_flight_max"] >= 21
    assert c["sent_requests"] == 30
    assert c["measured_requests"] == 30
    assert c["coverage"] == 1.0


def test_truncation_by_the_global_cap_is_counted_separately():
    """Ending on length at your own sampled target means the replay worked.
    Ending on it because the global cap bound first means the run never
    reproduced the profile's output distribution."""
    base = 1_700_000_000.0
    rows = []
    for i in range(40):          # hit their own target, healthy
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 50.0,
                     "e2e_ms": 100.0, "stream_complete": True,
                     "visible_content_seen": True, "truncated": True,
                     "parse_errors": 0, "finish_reason": "length",
                     "intended_output_tokens": 64, "max_tokens_requested": 64,
                     "t_send_unix": base + i, "first_send_unix": base + i})
    for i in range(10):          # cap bound first, distribution not reproduced
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 50.0,
                     "e2e_ms": 100.0, "stream_complete": True,
                     "visible_content_seen": True, "truncated": True,
                     "parse_errors": 0, "finish_reason": "length",
                     "intended_output_tokens": 200,
                     "max_tokens_requested": 64,
                     "t_send_unix": base + 40 + i,
                     "first_send_unix": base + 40 + i})
    a = summarize(rows)["answers"]
    assert a["truncated"] == 50
    assert a["truncated_by_global_cap"] == 10


# ---- coordinated omission and retry occupancy ---------------------------

def test_client_queue_wait_is_reported_as_experienced_latency():
    """The classic way a saturated load generator reports a healthy tail.
    The latency clock starts when a worker gets around to sending, so a
    request that sat in the client queue for ten seconds still reports
    whatever the endpoint took once it finally went out."""
    base = 1_700_000_000.0
    rows = []
    for i in range(50):
        sched = i * 0.1
        lag = 0.0 if i < 25 else 10.0      # client falls 10s behind halfway
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 50.0,
                     "e2e_ms": 200.0, "scheduled_s": sched,
                     "t_send_unix": base + sched + lag,
                     "first_send_unix": base + sched + lag})
    s = summarize(rows)
    # the endpoint really did take 200 ms every time
    assert s["e2e_ms"]["p95"] == 200.0
    # but a caller asking on schedule waited far longer
    assert s["e2e_corrected_ms"]["p95"] > 9000
    assert "e2e_corrected_ms" in s and "latency_correction_note" in s
    assert "caller experienced" in render_markdown(s, "x")


def test_no_correction_is_reported_when_the_client_kept_up():
    base = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "ttft_ms": 50.0, "e2e_ms": 200.0,
             "scheduled_s": i * 0.1,
             "t_send_unix": base + i * 0.1,
             "first_send_unix": base + i * 0.1} for i in range(50)]
    s = summarize(rows)
    assert s["e2e_corrected_ms"]["p95"] == s["e2e_ms"]["p95"]


def test_a_retried_request_occupies_a_worker_for_its_whole_life():
    """first_send_unix is the first attempt, e2e_ms belongs to the attempt
    that succeeded. Pairing them put the span before the request was on the
    wire and understated occupancy."""
    from traffic_replay.metrics import _concurrency_block
    T = 1_700_000_000.0
    retried = {"ok": True, "phase": "replay", "e2e_ms": 300.0, "retries": 1,
               "first_send_unix": T, "t_send_unix": T + 2.0}
    filler = [{"ok": True, "phase": "replay", "e2e_ms": 300.0,
               "first_send_unix": T + i * 0.05,
               "t_send_unix": T + i * 0.05} for i in range(1, 60)]
    c = _concurrency_block([retried] + filler, None)
    assert c is not None
    # the retried row must still be in flight at T+2.1, which it would not
    # be if its span ended at T+0.3
    solo = _concurrency_block([retried,
                               {"ok": True, "phase": "replay", "e2e_ms": 10.0,
                                "first_send_unix": T + 2.1,
                                "t_send_unix": T + 2.1},
                               {"ok": True, "phase": "replay", "e2e_ms": 10.0,
                                "first_send_unix": T + 2.2,
                                "t_send_unix": T + 2.2}], None)
    assert solo["in_flight_max"] >= 2


# ---- a PASS on service time is not a PASS for the caller ----------------

def test_caller_experienced_latency_is_the_sla_measurement_not_a_warning():
    """A queued request must fail in the scorecard itself, not show a service
    time PASS with a warning elsewhere on the page."""
    base = 1_700_000_000.0
    rows = []
    for i in range(300):
        sched = i * 0.1
        lag = 0.0 if i < 150 else 10.0
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 50.0,
                     "e2e_ms": 200.0, "scheduled_s": sched,
                     "t_send_unix": base + sched + lag,
                     "first_send_unix": base + sched + lag,
                     "prompt_tokens": 100, "completion_tokens": 10})
    s = summarize(rows, acceptance={"ttfg_ms": {"p95": 1500}})
    scored = s["sla"]["ttfg_vs_target"][0]
    assert scored["met"] is False
    assert scored["scored_metric"] == "e2e_corrected_ms"
    assert scored["actual_ms"] > 9000
    assert "Meets every acceptance target" not in render_html(s, "x")
    assert "latency basis: caller experienced" in render_markdown(s, "x")


def test_missing_token_usage_is_shown_and_downgrades_the_verdict():
    """Coverage was computed and then never rendered, so a run reporting
    usage on half its responses printed confident throughput and cost."""
    base = 1_700_000_000.0
    rows = []
    for i in range(200):
        r = {"ok": True, "phase": "replay", "ttft_ms": 50.0, "e2e_ms": 200.0,
             "t_send_unix": base + i * 0.1, "first_send_unix": base + i * 0.1}
        if i % 2 == 0:
            r["prompt_tokens"] = 100
            r["completion_tokens"] = 10
        rows.append(r)
    s = summarize(rows, acceptance={"ttfg_ms": {"p95": 1500}})
    assert s["throughput"]["usage_coverage"] == 0.5
    assert s["throughput"]["coverage_warning"]
    md = render_markdown(s, "x")
    assert "CAUTION (token usage)" in md
    assert "Meets every acceptance target" not in render_html(s, "x")


def test_idle_time_inside_the_window_counts_as_zero_in_flight():
    """The sweep used to start at the first event, so a sparse run reported
    a concurrency it held only a third of the time."""
    from traffic_replay.metrics import _concurrency_block
    T = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "e2e_ms": 1000.0,
             "t_send_unix": T + i * 3.0,
             "first_send_unix": T + i * 3.0} for i in range(6)]
    c = _concurrency_block(rows, None)
    assert c["in_flight_p50"] == 0.0, c
    assert c["in_flight_max"] == 1.0


# ---- adversarial: every way a bad run tried to read green ---------------

def _clean(n, **extra):
    base = 1_700_000_000.0
    out = []
    for i in range(n):
        r = {"ok": True, "phase": "replay", "ttft_ms": 50.0, "e2e_ms": 200.0,
             "prompt_tokens": 100, "completion_tokens": 10,
             "stream_complete": True, "visible_content_seen": True,
             "truncated": False, "parse_errors": 0,
             "t_send_unix": base + i * 0.1,
             "first_send_unix": base + i * 0.1}
        r.update(extra)
        out.append(r)
    return out


def _v(s):
    return [x for x in render_markdown(s, "x").splitlines()
            if x.startswith("verdict:")][0]


def test_sparse_concurrency_does_not_claim_a_load_it_never_held():
    """The edge-aware sweep was added and then used only for the peak, so
    the percentiles still began at the first event."""
    from traffic_replay.metrics import _concurrency_block
    T = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "e2e_ms": 1000.0,
             "t_send_unix": T + t, "first_send_unix": T + t}
            for t in (0.0, 4.5, 9.0)]
    c = _concurrency_block(rows, None)
    assert c["in_flight_p50"] == 0.0, c
    # and a genuinely steady run still reads steady
    steady = [{"ok": True, "phase": "replay", "e2e_ms": 5000.0,
               "t_send_unix": T + i * 0.1,
               "first_send_unix": T + i * 0.1} for i in range(100)]
    assert _concurrency_block(steady, None)["in_flight_p50"] == 50.0


def test_ttft_target_is_scored_on_caller_time():
    base = 1_700_000_000.0
    rows = []
    for i in range(300):
        sched = i * 0.1
        lag = 0.0 if i < 150 else 2.0
        rows.append({"ok": True, "phase": "replay", "ttft_ms": 50.0,
                     "e2e_ms": 30000.0, "scheduled_s": sched,
                     "t_send_unix": base + sched + lag,
                     "first_send_unix": base + sched + lag,
                     "prompt_tokens": 100, "completion_tokens": 10,
                     "stream_complete": True, "visible_content_seen": True,
                     "truncated": False, "parse_errors": 0})
    s = summarize(rows, acceptance={"ttft_ms": {"p95": 500}})
    scored = s["sla"]["ttft_vs_target"][0]
    assert scored["met"] is False
    assert scored["scored_metric"] == "ttft_corrected_ms"
    assert scored["actual_ms"] > 1900
    assert "Meets every acceptance target" not in render_html(s, "x")


def test_cache_shape_mismatch_cannot_render_green():
    rows = _clean(400, intended_cache_fraction=0.60, cached_tokens=0,
                  cached_tokens_source="prompt_tokens_details.cached_tokens")
    # Stretch the sends enough to establish stable windows, isolating cache.
    for i, row in enumerate(rows):
        row["t_send_unix"] = 1_700_000_000.0 + i * 0.5
        row["first_send_unix"] = row["t_send_unix"]
    s = summarize(rows, acceptance={"ttfg_ms": {"p95": 5000}})
    assert s["cache_fidelity"]["status"] == "unverified"
    assert "did not reproduce" in s["cache_fidelity"]["warning"]
    assert "Meets every acceptance target" not in render_html(s, "x")
    assert "CAUTION (cache fidelity)" in render_markdown(s, "x")


def test_usage_missing_only_on_the_output_side_is_still_partial():
    """Coverage keyed on prompt_tokens alone, so a response reporting input
    and not output counted as full coverage while halving throughput."""
    rows = _clean(200)
    for i, r in enumerate(rows):
        if i % 2:
            r.pop("completion_tokens")
    s = summarize(rows, acceptance={"ttfg_ms": {"p95": 5000}})
    assert s["throughput"]["usage_coverage"] == 0.5
    assert "Meets every acceptance target" not in render_html(s, "x")


def test_a_run_clipped_by_the_global_cap_is_not_green():
    """Truncation at a request's own target is the replay working. Truncation
    by the global cap means the output distribution was never reproduced."""
    rows = _clean(200, truncated=True, intended_output_tokens=200,
                  max_tokens_requested=64)
    s = summarize(rows, acceptance={"ttfg_ms": {"p95": 5000}})
    assert s["answers"]["truncated_by_global_cap"] == 200
    assert "Meets every acceptance target" not in render_html(s, "x")
    assert "cut short by max_output_tokens_cap" in _v(s)


def test_a_run_with_no_targets_still_gets_a_verdict():
    """Both renderers computed the verdict inside the SLA branch, so a run
    with no acceptance targets showed none at all."""
    s = summarize(_clean(300))
    assert "no acceptance targets" in _v(s)
    assert "banner" in render_html(s, "x")


def test_a_run_whose_stability_was_never_established_is_not_green():
    """Absence of a stability verdict was reading as a passing one. Three
    shapes reach it: a run too short to window, a run where no window carries
    a usable sample, and a merged run where drift is blanked by design."""
    s = summarize(_clean(400), acceptance={"ttfg_ms": {"p95": 5000}})
    assert (s.get("drift") or {}).get("drift_kind") is None
    assert "Meets every acceptance target" not in render_html(s, "x")
    assert "stability over the run was not established" in _v(s)


def test_a_success_rate_target_needs_enough_requests_to_miss_it():
    """Two requests cannot demonstrate a 99 percent success rate."""
    s = summarize(_clean(2), acceptance={"success_rate": 0.99})
    assert "cannot demonstrate" in _v(s)
    sr = s["sla"]["success_rate"]
    assert sr["actual"] == 1.0
    assert sr["met"] is True
    assert sr["statistically_demonstrated"] is False
    assert sr["one_sided_95pct_wilson_lower"] < 0.99


def test_success_rate_green_requires_confidence_bound_to_clear_target():
    thin = summarize(_clean(1_899), acceptance={"success_rate": 0.999})
    thin_sr = thin["sla"]["success_rate"]
    assert thin_sr["actual"] == 1.0
    assert thin_sr["one_sided_95pct_wilson_lower"] < 0.999
    assert thin_sr["statistically_demonstrated"] is False
    assert "cannot demonstrate" in _v(thin)

    sufficient = summarize(_clean(3_000),
                           acceptance={"success_rate": 0.999})
    sufficient_sr = sufficient["sla"]["success_rate"]
    assert sufficient_sr["one_sided_95pct_wilson_lower"] >= 0.999
    assert sufficient_sr["statistically_demonstrated"] is True


def test_the_arrival_rate_counts_only_rows_it_measured_the_span_over():
    """A half-stamped input would otherwise report double the true rate."""
    base = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "ttft_ms": 50.0, "e2e_ms": 200.0,
             "t_send_unix": base + i * 0.1,
             "first_send_unix": base + i * 0.1} for i in range(100)]
    rows += [{"ok": True, "phase": "replay", "ttft_ms": 50.0,
              "e2e_ms": 200.0} for _ in range(100)]      # no send stamp
    s = summarize(rows)
    assert abs(s["arrivals"]["achieved_qps_overall"] - 10.0) < 0.2
