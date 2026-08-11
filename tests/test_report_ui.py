"""Decision-first, responsive, safe report UI contract tests."""
from __future__ import annotations

from pathlib import Path

from traffic_replay.json_input import loads_strict
from traffic_replay.metrics import (
    _html_quota_gauges,
    render_html,
    render_markdown,
    summarize,
    write_outputs,
)


def _rows(n: int = 240) -> list[dict]:
    rows = []
    for index in range(n):
        sent = float(index)
        rows.append({
            "ok": True,
            "phase": "replay",
            "status": 200,
            "t_send_unix": sent,
            "first_send_unix": sent,
            "t_completed_unix": sent + 0.2,
            "scheduled_s": sent,
            "queue_wait_ms": 0.0,
            "ttfb_ms": 80.0,
            "ttft_ms": 100.0,
            "e2e_ms": 200.0,
            "interchunk_max_ms": 25.0,
            "visible_content_seen": True,
            "stream_complete": True,
            "parse_errors": 0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "finish_reason": "stop",
        })
    return rows


def _summary(*, quota_limited: bool = False) -> dict:
    replay = _rows()
    all_phases = list(replay)
    if quota_limited:
        all_phases.append({
            "ok": False,
            "phase": "preflight",
            "status": 429,
            "error": "redacted response body sha256=abc",
        })
    return summarize(
        replay,
        schedule_meta={
            "seconds": 240,
            "requests": 240,
            "rate_min": 1.0,
            "rate_p50": 1.0,
            "rate_p95": 1.0,
            "rate_max": 1.0,
            "spiky": False,
            "source": "test schedule",
        },
        run_meta={
            "input_mode": "profile",
            "endpoint_path": "/serving-endpoints/test/invocations",
            "endpoint_metadata": {
                "name": "test-endpoint",
                "served_entities": [{"name": "test-model"}],
            },
            "artifact_id": "artifact-ui-contract",
            "label": "customer load shape",
        },
        acceptance={
            "targets_are": "customer requirements",
            "ttft_ms": {"p95": 50},
            "success_rate": 0.99,
        },
        rate_limit_results=all_phases,
    )


def test_first_screen_model_keeps_quota_sla_and_capacity_independent():
    body = render_html(_summary(quota_limited=True), "quota-limited run")

    assert "Measurement invalid" in body
    assert "Acceptance checks missed" in body
    assert "HTTP 429 / rate-limit rejection observed" in body
    assert "Endpoint capacity inconclusive" in body
    assert "1/241" in body
    assert "preflight: 1 captured, 0 send-timestamped, 1 send " \
           "timing/outcome unknown" in body
    assert "Tested load held" not in body
    customer_summary = body[body.index("id='summary'"):
                            body.index("id='limitations'")]
    assert "Measurement invalid" in customer_summary
    assert "Acceptance checks missed" in customer_summary
    assert "HTTP 429 / rate-limit rejection observed" in customer_summary
    assert "Capacity not tested" in customer_summary
    assert body.index("Result at this tested load") < body.index(
        "What this run does not prove")
    assert body.index("What this run does not prove") < body.index(
        "id='latency'")
    assert body.index("id='latency'") < body.index(
        "Reliability and completion")
    assert body.index("Reliability and completion") < body.index(
        "What was tested")
    assert body.index("What was tested") < body.index(
        "Databricks engineering diagnostics")
    assert body.index("Decision summary") > body.index(
        "Databricks engineering diagnostics")
    assert body.index("Final-attempt request-path latency") > body.index(
        "Databricks engineering diagnostics")


def test_customer_takeaway_leads_with_caller_latency_and_plain_caveats():
    summary = _summary()
    body = render_html(summary, "customer report")
    markdown = render_markdown(summary, "customer report")

    takeaway = body.index("Result at this tested load")
    decision = body.index("Databricks engineering diagnostics")
    request_path = body.index("Final-attempt request-path latency")
    assert takeaway < decision < request_path
    assert "Latency for the final request attempt" in body
    assert "Time to first response" in body
    assert "Complete response" in body
    assert "Answer quality was not evaluated" in body
    assert "do not establish correctness" in body
    assert "Synthetic workload shape" in body
    assert "240 logical requests over 240 s" in body
    assert body.index("What was tested") < body.index(
        "Field definitions and raw evidence")
    assert "does not establish quota headroom" in body
    assert "calibration" in body.lower()
    assert "Calibration estimates characters per token" in body
    assert "integrity says whether the recorded artifact bytes verify" in body
    assert markdown.index("## Customer takeaway") < markdown.index(
        "## Decision states")
    assert "user-perceived latency from scheduled request time" in markdown
    assert "Answer quality:** not evaluated" in markdown
    assert "integrity says whether the recorded artifact bytes verify" \
        in markdown


def test_report_shell_is_self_contained_responsive_semantic_and_printable():
    body = render_html(_summary(), "responsive report")

    assert "<main class='wrap'>" in body
    assert "<nav class='report-nav' aria-label='Report sections'>" in body
    assert "@media(max-width:640px)" in body
    assert "@media print" in body
    assert "page-break-inside:avoid" in body
    assert "UNSEALED PRINT/PDF DERIVATIVE" in body
    assert "artifact artifact-ui-contract" in body
    assert "internal hashes are not a digital signature" in body
    assert ".print-footer{display:block;" in body
    assert "position:fixed" not in body
    assert "counter(page)" not in body
    assert "<script" not in body
    assert "<link" not in body
    assert "http://" not in body and "https://" not in body
    assert "href='#summary'>Summary</a>" in body
    assert "href='#latency'>Latency</a>" in body
    assert "href='#reliability'>Reliability</a>" in body
    assert "href='#test-scope'>Test scope</a>" in body
    assert "href='#engineering'>Engineering</a>" in body
    assert "class='disclosure engineering-disclosure' id='engineering'" in body
    assert "class='disclosure verification-disclosure' id='verification'" in body
    assert "class='disclosure glossary-disclosure' id='field-glossary'" in body
    assert "Scroll horizontally; the Metric column stays visible." in body
    assert "class='table-scroll' tabindex='0' role='region'" in body
    assert ".dense-table .sticky-col{position:sticky" in body

    print_css = body.split("@media print", 1)[1].split("</style>", 1)[0]
    assert ".engineering-disclosure{break-before:page" in print_css
    assert "details.disclosure>.detail-body{" in print_css
    assert "display:block!important" in print_css
    assert "Why these states" in body
    assert "every canonical gate code and message" in body
    assert "Measurement validity" in body
    assert "<section class='gate-detail' id='decision-reasons'" in body
    assert "class='gate-reason-list'" in body
    assert ".table-scroll:focus-visible{outline:3px solid var(--blue)" in body
    assert "table:not(.dense-table){display:table;width:100%" in body
    assert "table:not(.dense-table) th.lbl{width:44%}" in body
    mobile_css = body.split("@media(max-width:640px)", 1)[1].split(
        "@media print", 1)[0]
    assert ".customer-summary .summary-lead{font-size:16px}" in mobile_css
    assert ".kpi-grid{grid-template-columns:1fr 1fr" in mobile_css
    assert ".customer-latency-grid,.scope-grid{" in mobile_css
    for heading in (
            "Evidence integrity", "Measurement validity", "Acceptance checks",
            "Quota state", "Endpoint capacity"):
        assert heading in body


def test_additional_cautions_surface_identity_stability_and_runtime_admission():
    summary = _summary()
    summary["response_identity"].update({
        "status": "invalid",
        "invalid": "response model changed during the run",
    })
    summary["drift"] = {
        "drift_kind": "variable",
        "drift_headline": "tail latency varied across windows",
    }
    summary["runtime_quota_admission"] = {
        "status": "invalid_evidence",
        "denied_rows": 0,
        "denied_attempts_in_captured_rows": 0,
    }
    summary["run"]["transport"] = {
        "production_connection_policy_match": False,
        "production_comparability_warning": (
            "production uses a pooled HTTP/2 connection"),
    }

    body = render_html(summary, "caution coverage")

    assert "Additional measurement and workload cautions" in body
    assert "Response model identity" in body
    assert "response model changed during the run" in body
    assert "Stability" in body and "tail latency varied across windows" in body
    assert "Runtime quota admission" in body
    assert "failed its invariants" in body
    assert "Transport parity" in body
    assert "production uses a pooled HTTP/2 connection" in body


def test_all_canonical_caution_details_precede_metrics_in_html_and_markdown():
    summary = _summary()
    summary["cache_fidelity"] = {
        "warning": "cache reporting was unavailable for this run",
    }
    summary["network_path"] = {
        "warning": "generator placement differs from production",
    }
    summary["run"].update({
        "endpoint_metadata_warning": "post-drain metadata was unavailable",
        "transport": {
            "production_connection_policy_match": False,
            "production_comparability_warning": (
                "production connection reuse was not established"),
        },
    })

    body = render_html(summary, "all caution details")
    markdown = render_markdown(summary, "all caution details")

    expected = {
        "CACHE_FIDELITY_UNVERIFIED": (
            "cache reporting was unavailable for this run"),
        "NETWORK_PATH_CAUTION": "generator placement differs from production",
        "ENDPOINT_METADATA_STABILITY_UNVERIFIED": (
            "post-drain metadata was unavailable"),
        "PRODUCTION_TRANSPORT_UNVERIFIED": (
            "production connection reuse was not established"),
    }
    for code, message in expected.items():
        assert code in body and message in body
        assert body.index(code) > body.index(
            "Databricks engineering diagnostics")
        assert code in markdown and message in markdown
        assert markdown.index(code) < markdown.index(
            "final-attempt request-path metric")
    decision_block = body[body.index("Decision summary"):
                          body.index("Load delivery and workload facts", body.index(
                              "Databricks engineering diagnostics"))]
    assert "plus " not in decision_block


def test_unsealed_report_never_calls_captured_quota_rows_sealed():
    body = render_html(_summary(), "unsealed quota wording")
    markdown = render_markdown(_summary(), "unsealed quota wording")
    gauge = _html_quota_gauges({
        "comparisons": {
            "queries_per_hour": {
                "configured_limit": 100,
                "observed_max": 10,
                "observed_ratio_to_nominal_limit": 0.1,
                "warning_utilization": 0.8,
            },
        },
    })

    assert "This is captured run evidence." in gauge
    assert "captured traffic phases" in body
    assert "captured traffic phases" in markdown
    assert "sealed run evidence" not in body.lower()
    assert "sealed traffic phases" not in body.lower()
    assert "sealed traffic phases" not in markdown.lower()


def test_hard_timeout_row_exists_only_when_a_timeout_target_was_configured():
    no_timeout = _summary()
    assert no_timeout["sla"]["hard_timeout_breaches"] == 0
    assert "hard timeout breaches" not in render_html(
        no_timeout, "no timeout target")
    assert "hard timeout breaches" not in render_markdown(
        no_timeout, "no timeout target")

    with_timeout = summarize(
        _rows(),
        acceptance={
            "targets_are": "customer requirements",
            "hard_timeouts": {"ttft_s": 1.0},
        },
    )
    assert "hard timeout breaches" in render_html(
        with_timeout, "timeout target")
    assert "hard timeout breaches" in render_markdown(
        with_timeout, "timeout target")


def test_near_complete_usage_is_still_labeled_as_subset_throughput():
    rows = _rows()
    rows[-1]["parse_errors"] = 1
    summary = summarize(rows)
    html = render_html(summary, "subset throughput")

    assert summary["throughput"]["usage_coverage"] == 239 / 240
    assert summary["throughput"]["coverage_warning"]
    assert "Throughput: clean usage subset" in html
    assert "clean usage subset; 99.6% row coverage" in html


def test_run_provenance_is_near_decision_not_an_orphanable_final_block():
    summary = _summary()
    summary["run"].update({
        "profile_label": "Customer production traffic profile",
        "merge_note": "Merged from two manifest-bound shards.",
    })

    body = render_html(summary, "provenance placement")

    provenance = body.index(
        "<div class='run-context-notes' aria-label='Run context notes'>")
    assert body.index("Decision summary") < provenance
    engineering_workload = body.index(
        "Load delivery and workload facts", provenance)
    assert provenance < engineering_workload
    assert body.index("Label:</b> customer load shape") < engineering_workload
    assert body.index("Profile:</b> Customer production traffic profile") \
        < engineering_workload
    assert body.count("aria-label='Run context notes'") == 1
    assert ".run-context-notes{break-inside:avoid;page-break-inside:avoid" \
        in body


def test_missing_run_counts_never_render_as_a_green_zero():
    summary = _summary()
    for key in ("requests_total", "requests_ok", "requests_failed",
                "error_rate", "timeout_failures"):
        summary.pop(key, None)

    body = render_html(summary, "legacy incomplete summary")

    summary_block = body[body.index("id='summary'"):
                         body.index("id='limitations'")]
    assert "Harness successful" in summary_block
    assert "NOT REPORTED" in summary_block
    reliability_block = body[body.index("id='reliability'"):
                             body.index("id='test-scope'")]
    assert "unknown</div><div class='k'>Timeouts" in reliability_block
    error_card = body[body.index("Replay error rate"):]
    error_card = error_card[:error_card.index("</div></div>") + 12]
    assert "pill neutral" in error_card
    assert "NOT REPORTED" in error_card
    assert "0.00%" not in error_card


def test_exact_caller_latency_is_primary_and_zero_throughput_is_visible():
    rows = _rows()
    for row in rows:
        row["caller_ttft_ms"] = 500.0
        row["caller_e2e_ms"] = 700.0
    summary = summarize(rows)
    summary["throughput"]["completion_tokens_per_min"] = 0.0
    summary["throughput"]["all_completion_tokens_per_min"] = 0.0
    summary["throughput"]["output_tokens_per_min"] = 0.0

    body = render_html(summary, "caller-first report")

    assert "Exact caller TTFT p50" in body \
        and ">500 <span class='u'>ms" in body
    assert "Exact caller end to end p95" in body \
        and ">700 <span class='u'>ms" in body
    assert body.index("Exact caller TTFT p50") < body.index(
        "Final-attempt request-path latency")
    assert "all-completion throughput" in body
    assert ">0 <span class='u'>completion tok/min</span>" in body


def test_stability_chart_has_units_alt_text_and_preserves_missing_gaps():
    summary = _summary()
    summary["drift"] = {
        "window_seconds": 60,
        "drift_kind": "variable",
        "drift_flag": True,
        "drift_headline": "middle window has no answer latency",
        "windows": [
            {"window": 0, "n": 40, "attempts": 40, "errors": 0,
             "error_rate": 0.0, "ttft_p95": 100.0, "e2e_p95": 200.0,
             "counted": True},
            {"window": 1, "n": 0, "attempts": 40, "errors": 40,
             "error_rate": 1.0, "ttft_p95": None, "e2e_p95": None,
             "counted": False},
            {"window": 2, "n": 40, "attempts": 40, "errors": 0,
             "error_rate": 0.0, "ttft_p95": 300.0, "e2e_p95": 500.0,
             "counted": True},
        ],
    }
    body = render_html(summary, "gapped chart")

    assert "role='img'" in body
    assert "Tail latency over time" in body
    assert "Missing values are gaps, not zeros" in body
    assert "Exact per-window stability values in milliseconds" in body
    assert "nan" not in body.lower()
    assert "chart-dot chart-dot-secondary" in body
    assert "style='color:#6b55c5'" in body
    assert "E2E p95</span>" in body


def test_quota_gauge_never_labels_projection_as_observed():
    body = _html_quota_gauges({
        "comparisons": {
            "input_tokens_per_minute": {
                "configured_limit": 1_000,
                "observed_max": 200,
                "observed_ratio_to_nominal_limit": 0.20,
                "steady_state_projection": 1_200,
                "ratio_to_nominal_limit": 1.20,
                "warning_utilization": 0.60,
            },
        },
    })

    assert "observed captured window</span><span>20.0%" in body
    assert "200 / 1,000 configured; warning at 60.0%" in body
    assert "sustained-rate projection</span><span>120.0%" in body
    assert "1,200 / 1,000 configured; warning at 60.0%" in body
    assert "projection from a short observation" in body
    assert "observed captured window</span><span>120.0%" not in body
    # The configured 60% warning threshold, not a hard-coded 80%, controls
    # tone.  The observed 20% bar remains neutral and projection is red.
    assert body.count("gauge-fill bad") == 1
    assert "gauge-fill warn" not in body


def test_written_json_and_both_human_reports_share_decision_states(tmp_path):
    summary = _summary(quota_limited=True)
    sealed_rows = _rows() + [{
        "ok": False, "phase": "preflight", "status": 429,
        "error": "redacted response body sha256=abc",
    }]
    out = write_outputs(sealed_rows, summary, tmp_path, "parity")
    stored = loads_strict((Path(out) / "summary.json").read_bytes())
    html = (Path(out) / "report.html").read_text()
    markdown = (Path(out) / "report.md").read_text()

    expected = {
        "measurement_validity": "INVALID",
        "customer_sla": "MISS",
        "quota_state": "EXCEEDED",
        "endpoint_capacity": "INCONCLUSIVE",
    }
    for key, code in expected.items():
        assert stored["decision"][key]["code"] == code
        label = stored["decision"][key]["label"]
        assert label in html
        assert label in markdown


def test_single_run_markdown_neutralizes_customer_structure():
    summary = _summary()
    hostile = "title | split\n<script>x</script> ![fetch](https://evil.invalid/x)"
    summary["run"]["label"] = hostile
    summary["run"]["profile_label"] = hostile
    markdown = render_markdown(summary, hostile)

    assert "<script>" not in markdown
    assert "![fetch](https://evil.invalid/x)" not in markdown
    assert "title | split" not in markdown
    assert "title &#124; split" in markdown
    assert markdown.count("| decision | state | reason |") == 1


def test_tool_call_only_success_is_never_called_a_content_delta():
    row = {
        "ok": True,
        "phase": "replay",
        "status": 200,
        "visible_content_seen": False,
        "reasoning_seen": False,
        "valid_tool_calls": 1,
        "stream_complete": True,
        "parse_errors": 0,
        "ttf_tool_call_ms": 42.0,
        "e2e_ms": 60.0,
        "t_send_unix": 100.0,
        "first_send_unix": 100.0,
    }
    summary = summarize([row])
    answers = summary["answers"]
    html = render_html(summary, "tool-only")
    markdown = render_markdown(summary, "tool-only")

    assert answers["harness_successful"] == 1
    assert answers["content_delta_streams"] == 0
    assert answers["tool_call_only_outcomes"] == 1
    assert "1 produced a content delta" not in html
    assert "1 produced a content delta" not in markdown
    assert "visible or reasoning content delta</th><td>0" in html
    assert "visible or reasoning content delta: 0" in markdown
    assert "1 harness-successful" in html


def test_success_rate_confidence_gate_matches_json_html_and_markdown(tmp_path):
    rows = _rows(1_200)
    summary = summarize(
        rows,
        schedule_meta={
            "seconds": 1_200,
            "requests": 1_200,
            "rate_min": 1.0,
            "rate_p50": 1.0,
            "rate_p95": 1.0,
            "rate_max": 1.0,
            "spiky": False,
            "source": "test schedule",
        },
        acceptance={"success_rate": 0.9999},
    )

    assert summary["sla"]["success_rate"]["actual"] == 1.0
    assert summary["sla"]["success_rate"]["met"] is True
    assert summary["sla"]["success_rate"][
        "statistically_demonstrated"] is False

    out = write_outputs(rows, summary, tmp_path, "confidence")
    stored = loads_strict((Path(out) / "summary.json").read_bytes())
    html = (Path(out) / "report.html").read_text()
    markdown = (Path(out) / "report.md").read_text()

    sla = stored["decision"]["customer_sla"]
    assert sla["code"] == "INCONCLUSIVE"
    assert sla["reason_codes"] == [
        "SUCCESS_RATE_CONFIDENCE_NOT_DEMONSTRATED"]
    assert "Acceptance checks inconclusive" in html
    assert "Acceptance checks inconclusive" in markdown
    assert "Configured acceptance checks passed" not in html
    assert "Configured acceptance checks passed" not in markdown
    assert "NOT PROVEN" in html


def test_html_and_markdown_strip_bidi_controls_from_untrusted_metadata():
    summary = _summary(quota_limited=True)
    hostile = "Trusted\u202eLIAF\u2066"
    summary["run"].update({
        "label": hostile,
        "profile_label": hostile,
        "merge_note": hostile,
        "endpoint_path": f"/serving-endpoints/{hostile}/invocations",
    })
    summary["run"]["endpoint_metadata"]["served_entities"] = [
        {"name": hostile}]
    summary["http_429"]["scope"] = hostile

    html = render_html(summary, hostile)
    markdown = render_markdown(summary, hostile)

    for control in ("\u202e", "\u2066"):
        assert control not in html
        assert control not in markdown
    assert "TrustedLIAF" in html
    assert "TrustedLIAF" in markdown
