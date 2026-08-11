"""Customer-first HTML report presentation contract.

These tests intentionally exercise the rendered document rather than private
renderer helpers.  The customer summary is the default reading path; detailed
verification and backend diagnostics must remain available without crowding
that path.
"""
from __future__ import annotations

from html.parser import HTMLParser
import re

from traffic_replay.metrics import render_html, summarize
from traffic_replay.report_decision import (
    IntegrityContext,
    build_report_decision,
)


class _Node:
    def __init__(self, tag: str, attrs=(), parent: "_Node | None" = None):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.children: list[_Node | str] = []

    def text(self, *, excluding: "_Node | None" = None) -> str:
        if self is excluding:
            return ""
        return " ".join(
            child if isinstance(child, str) else child.text(excluding=excluding)
            for child in self.children
        )

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.walk()


class _Document(HTMLParser):
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, body: str):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self._stack = [self.root]
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs, self._stack[-1])
        self._stack[-1].children.append(node)
        if tag not in self._VOID:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data):
        self._stack[-1].children.append(data)

    def by_id(self, node_id: str) -> _Node:
        matches = [node for node in self.root.walk()
                   if node.attrs.get("id") == node_id]
        assert len(matches) == 1, f"expected one #{node_id}, got {len(matches)}"
        return matches[0]

    def details(self, summary_text: str) -> _Node:
        wanted = summary_text.casefold()
        for node in self.root.walk():
            if node.tag != "details":
                continue
            summaries = [child for child in node.children
                         if isinstance(child, _Node)
                         and child.tag == "summary"]
            if summaries and wanted in _plain(summaries[0].text()).casefold():
                return node
        raise AssertionError(f"no details disclosure named {summary_text!r}")


def _plain(value: str) -> str:
    return " ".join(value.split())


def _class_tokens(node: _Node) -> set[str]:
    return set(node.attrs.get("class", "").split())


def _reliability_value(document: _Document, label: str) -> str:
    reliability = document.by_id("reliability")
    for node in reliability.walk():
        if "reliability-item" not in _class_tokens(node):
            continue
        if label.casefold() not in _plain(node.text()).casefold():
            continue
        values = [child for child in node.children
                  if isinstance(child, _Node)
                  and "v" in _class_tokens(child)]
        assert len(values) == 1
        return _plain(values[0].text())
    raise AssertionError(f"no reliability item named {label!r}")


def _rows(n: int, *, first_response=None, complete_response=None) -> list[dict]:
    first_response = first_response or [700.0 + index for index in range(n)]
    complete_response = complete_response or [
        value + 250.0 for value in first_response]
    base = 1_800_000_000.0
    rows = []
    for index, (first_ms, complete_ms) in enumerate(zip(
            first_response, complete_response, strict=True)):
        sent = base + index
        rows.append({
            "ok": True,
            "phase": "replay",
            "status": 200,
            "scheduled_s": float(index),
            "t_send_unix": sent,
            "first_send_unix": sent,
            "t_completed_unix": sent + complete_ms / 1000.0,
            "queue_wait_ms": 0.0,
            "ttfb_ms": 350.0,
            "ttft_ms": 450.0,
            "e2e_ms": 700.0,
            "caller_ttft_ms": float(first_ms),
            "caller_e2e_ms": float(complete_ms),
            "interchunk_max_ms": 25.0,
            "visible_content_seen": True,
            "stream_complete": True,
            "parse_errors": 0,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "response_model": "test-model",
            "finish_reason": "length",
            "truncated": True,
            "intended_output_tokens": 20,
            "max_tokens_requested": 20,
        })
    return rows


def _summary(n: int = 3, *, acceptance: dict | None = None) -> dict:
    if n == 3:
        rows = _rows(
            n,
            first_response=[736.0, 789.0, 1086.0],
            complete_response=[1016.0, 1043.0, 1318.0],
        )
        seconds = 9
    else:
        rows = _rows(n)
        seconds = n
    return summarize(
        rows,
        schedule_meta={
            "seconds": seconds,
            "requests": n,
            "rate_min": n / seconds,
            "rate_p50": n / seconds,
            "rate_p95": n / seconds,
            "rate_max": n / seconds,
            "spiky": False,
            "source": "customer-report-v2 fixture",
        },
        run_meta={
            "input_mode": "profile",
            "endpoint_path": "/serving-endpoints/test/invocations",
            "endpoint_model": "test-model",
            "artifact_id": "customer-report-v2",
            "preflight_gate": {
                "skipped": False,
                "attempted": 2,
                "reachable": 2,
                "readable": 2,
                "reasoning_probe_requests": 0,
                "outcome": "preflight_passed",
                "force_requested": False,
                "gate_satisfied": True,
            },
            "transport": {
                "connection_policy_id": "fresh_http1_per_physical_attempt",
                "production_connection_policy_declared": "pooled_http2",
                "production_connection_policy_match": False,
                "production_comparability_warning": (
                    "test transport did not match the pooled production path"),
            },
            "request_params": {
                "endpoint_adapter": "openai.chat_completions.sse/v1",
                "response_mode": "streaming",
                "temperature": 0.0,
                "max_output_tokens_cap": 20,
                "extra_body": {},
            },
        },
        acceptance=acceptance,
    )


def _verification_context(summary: dict) -> dict:
    integrity = IntegrityContext(
        "verified",
        "The verifier established internal SHA-256 consistency; this is not "
        "a digital signature.",
    )
    return {
        "view_label": "EXTERNAL VERIFIED VIEW",
        "receipt_id": "receipt-customer-report-v2",
        "source_artifact_id": "customer-report-v2",
        "source_manifest_sha256": "a" * 64,
        "verifier_version": "9.9.9-test",
        "verified_at_utc": "2026-08-11T12:00:00Z",
        "assurance": (
            "Internal SHA-256 consistency was verified; this is not a "
            "digital signature."),
        "decision": build_report_decision(summary, integrity),
        "source_reproducibility": {
            "code": "PASS",
            "reason": "The exact source checkout was reconstructed.",
            "reason_codes": [],
        },
        "verifier_reproducibility": {
            "code": "PASS",
            "reason": "The exact verifier checkout was reconstructed.",
            "reason_codes": [],
        },
    }


def test_small_run_leads_with_descriptive_latency_and_honest_limitations():
    html = render_html(_summary(), "customer report v2")
    document = _Document(html)
    summary = _plain(document.by_id("summary").text())
    latency = _plain(document.by_id("latency").text())
    reliability = _plain(document.by_id("reliability").text())

    assert "Time to first response" in latency
    assert "Complete response" in latency
    assert re.search(r"\b789(?:\.0)?\s*ms\b", latency)
    assert re.search(r"\b736(?:\.0)?(?:\s*(?:to|[-–])\s*)1,?086(?:\.0)?\s*ms\b",
                     latency)
    assert re.search(r"\b1,?043(?:\.0)?\s*ms\b", latency)
    assert re.search(r"\b1,?016(?:\.0)?(?:\s*(?:to|[-–])\s*)1,?318(?:\.0)?\s*ms\b",
                     latency)
    assert "observed median" in latency.casefold()
    assert "observed range" in latency.casefold()
    assert "insufficient sample" in latency.casefold()
    assert "p95" not in latency.casefold()
    assert "p99" not in latency.casefold()

    assert re.search(r"(?:no sla|sla (?:was )?not (?:tested|evaluated)|no customer acceptance)",
                     summary, re.IGNORECASE)
    assert re.search(r"3/3.*(?:output|responses?).*(?:cap|truncat)",
                     summary, re.IGNORECASE)
    assert re.search(r"capacity.*(?:not tested|not established|inconclusive)",
                     summary, re.IGNORECASE)
    assert re.search(r"stability.*(?:not tested|not established|inconclusive)",
                     summary, re.IGNORECASE)
    assert "0 Model refusals" in reliability


def test_tail_latency_is_gated_by_sample_size_and_explicit_request():
    medium_html = render_html(_summary(150), "150 request report")
    medium_latency = _plain(_Document(medium_html).by_id("latency").text())
    assert re.search(r"p95.{0,80}insufficient sample",
                     medium_latency, re.IGNORECASE)
    # The computed p95 is 841.55 ms, but it must not be presented as a
    # decision-grade customer number below the 200-request support floor.
    assert not re.search(r"p95.{0,40}\b842\s*ms\b", medium_latency,
                         re.IGNORECASE)
    assert "p99" not in medium_latency.casefold()

    large = _summary(1000, acceptance={
        "targets_are": "customer requirements",
        "ttft_ms": {"p99": 2_000},
    })
    large_html = render_html(large, "1000 request report")
    large_latency = _plain(_Document(large_html).by_id("latency").text())
    assert "p99" in large_latency.casefold()
    assert not re.search(r"p99.{0,80}insufficient sample",
                         large_latency, re.IGNORECASE)
    assert re.search(r"p99.{0,80}\b1,?689\s*ms\b", large_latency,
                     re.IGNORECASE)


def test_verified_context_is_compact_and_full_provenance_is_disclosed():
    summary = _summary()
    digest = "a" * 64
    html = render_html(
        summary,
        "verified customer report",
        verification_context=_verification_context(summary),
    )
    document = _Document(html)
    verification = document.details("Verification and reproducibility")
    verification_text = _plain(verification.text())
    outside_text = _plain(document.root.text(excluding=verification))

    assert "open" not in verification.attrs
    assert verification.attrs.get("id") == "verification"
    assert "Evidence verified" in verification_text
    assert digest in verification_text
    assert "Source reproducibility" in verification_text
    assert "Verifier reproducibility" in verification_text
    assert "receipt-customer-report-v2" in verification_text
    assert digest not in outside_text
    assert "The exact source checkout was reconstructed" not in outside_text
    assert "The exact verifier checkout was reconstructed" not in outside_text
    assert html.count(digest) == 1


def test_unsupported_customer_target_is_not_rendered_as_pass():
    summary = _summary(10, acceptance={
        "targets_are": "customer requirements",
        "ttft_ms": {"p50": 2_000},
    })
    html = render_html(summary, "unsupported target")
    document = _Document(html)
    customer_summary = _plain(document.by_id("summary").text())
    latency = _plain(document.by_id("latency").text())

    assert "Customer target p50" in latency
    assert "Insufficient sample - need 20, have 10" in latency
    assert "NOT PROVEN" in customer_summary
    assert "NOT PROVEN" in latency
    assert not re.search(r"Acceptance.{0,100}\bPASS\b", customer_summary)
    assert not re.search(r"Customer target p50.{0,120}\bPASS\b", latency)


def test_measured_window_never_ends_before_a_later_incomplete_send():
    rows = _rows(2)
    rows[1].pop("t_completed_unix")
    rows[1]["e2e_ms"] = None
    rows[1]["caller_e2e_ms"] = None
    summary = summarize(rows)

    window = summary["measured_window"]
    assert window["ended_at_utc"] == "2027-01-15T08:00:01+00:00"
    assert window["completion_time_coverage"] == 0.5
    assert "does not prove that every response had completed" in window["warning"]

    scope = _plain(_Document(render_html(
        summary, "partial measured window")).by_id("test-scope").text())
    assert re.search(r"Completion(?:-time)? coverage", scope)
    assert "50.0%" in scope
    assert window["end_basis"] in scope
    assert window["warning"] in scope


def test_timeout_count_uses_untruncated_failure_evidence():
    rows = _rows(3)
    failure_counts = {
        "upstream reset": 7,
        "connection refused": 6,
        "bad gateway": 5,
        "service unavailable": 4,
        "protocol failure": 3,
        "read timeout": 1,
    }
    base = 1_800_001_000.0
    offset = 0
    for error, count in failure_counts.items():
        for _ in range(count):
            sent = base + offset
            rows.append({
                "ok": False,
                "phase": "replay",
                "status": 500,
                "error": error,
                "scheduled_s": float(3 + offset),
                "first_send_unix": sent,
                "t_send_unix": sent,
                "finished_unix": sent + 0.5,
            })
            offset += 1

    summary = summarize(rows)
    assert summary["timeout_failures"]["count"] == 1
    assert summary["timeout_failures"]["classification_complete"] is True
    assert all("timeout" not in key.casefold()
               for key in summary["failures_by_error"])

    document = _Document(render_html(summary, "untruncated timeout count"))
    assert _reliability_value(document, "Timeouts") == "1"


def test_incomplete_timeout_classification_never_renders_a_false_zero():
    rows = _rows(3)
    base = 1_800_002_000.0
    rows.extend([
        {
            "ok": False,
            "phase": "replay",
            "status": 500,
            "error": "connection reset",
            "scheduled_s": 3.0,
            "first_send_unix": base,
            "t_send_unix": base,
            "finished_unix": base + 0.5,
        },
        {
            "ok": False,
            "phase": "replay",
            "status": 500,
            "scheduled_s": 4.0,
            "first_send_unix": base + 1,
            "t_send_unix": base + 1,
            "finished_unix": base + 1.5,
        },
    ])

    summary = summarize(rows)
    timeout_evidence = summary["timeout_failures"]
    assert timeout_evidence["count"] == 0
    assert timeout_evidence["classification_complete"] is False

    document = _Document(render_html(
        summary, "incomplete timeout classification"))
    reliability = _plain(document.by_id("reliability").text())
    assert _reliability_value(document, "Timeouts") != "0"
    assert "lower bound" in reliability.casefold()


def test_large_run_scope_uses_aggregate_observed_token_ranges():
    rows = _rows(200)
    for index, row in enumerate(rows):
        row["prompt_tokens"] = 80 + index
        row["completion_tokens"] = 10 + index % 11
        row["finish_reason"] = "stop"
        row["truncated"] = False
    summary = summarize(rows)
    assert "latency_observations" not in summary
    prompt = summary["token_targeting"]["observed_prompt_tokens"]
    output = summary["token_targeting"]["observed_completion_tokens"]
    assert (prompt["min"], prompt["max"]) == (80.0, 279.0)
    assert (output["min"], output["max"]) == (10.0, 20.0)

    scope = _plain(_Document(render_html(
        summary, "aggregate token scope")).by_id("test-scope").text())
    assert re.search(r"Prompt size\s+80–279 tokens", scope)
    assert re.search(r"Output size\s+10–20 tokens", scope)
    assert "not retained in aggregate" not in scope


def test_missing_input_mode_is_unknown_not_synthetic():
    summary = summarize(_rows(3))
    summary.get("run", {}).pop("input_mode", None)

    html = render_html(summary, "input mode absent")
    scope = _plain(_Document(html).by_id("test-scope").text())

    assert "input: not recorded" in html
    assert "Input mode not recorded" in scope
    assert "Synthetic workload shape" not in scope


def test_requested_model_and_served_backend_are_source_qualified():
    summary = _summary()
    summary["run"]["endpoint_model"] = "request-body-alias"
    summary["run"]["endpoint_metadata"] = {
        "name": "serving-endpoint",
        "served_entities": [{
            "name": "active-served-entity",
            "foundation_model": {"name": "foundation-model-name"},
        }],
    }

    html = render_html(summary, "model identity labels")
    scope = _plain(_Document(html).by_id("test-scope").text())

    assert "requested model: request-body-alias" in html
    assert "Requested model request-body-alias" in scope
    assert "Active served entity active-served-entity" in scope
    assert "Foundation model foundation-model-name" in scope


def test_customer_scope_separates_burst_peak_from_window_average():
    summary = _summary(200)
    summary["schedule"].update({
        "seconds": 400,
        "requests": 200,
        "rate_min": 0.0,
        "rate_p50": 0.0,
        "rate_p95": 0.0,
        "rate_max": 50.0,
        "spiky": True,
        "source": "burst timestamps fixture",
    })
    summary["arrivals"].update({
        "achieved_qps_overall": 0.5,
        "scheduled_qps": 0.5,
        "logical_schedule_seconds": 400.0,
    })
    summary["observed_rate_windows"] = {
        "physical_queries_per_one_second_by_request_start": {
            "max": 50,
            "window_seconds": 1.0,
            "events_total": 200,
            "attempt_counts_exact": True,
            "all_attempt_timestamps_exact": True,
        },
    }

    scope = _plain(_Document(render_html(
        summary, "bursty customer scope")).by_id("test-scope").text())
    assert re.search(r"Average over logical window\s+0\.50\s+(?:req/s|RPS)",
                     scope)
    assert "Configured rate-curve peak 50 logical requests/s" in scope
    assert "Scheduled 1-second peak" not in scope
    assert "Observed 1-second peak 50 all-phase physical HTTP POST starts/s" \
        in scope
    assert "Bursty schedule" in scope
    assert "not a sustained rate" in scope.casefold()
    assert "200 logical requests over 400 s" in scope
    assert "200 all-phase physical HTTP POST attempts across replay plus " \
        "setup phases" in scope


def test_flat_logical_schedule_does_not_infer_burstiness_from_post_attempts():
    summary = _summary(200)
    summary["schedule"].update({
        "seconds": 20,
        "requests": 200,
        "rate_min": 10.0,
        "rate_p50": 10.0,
        "rate_p95": 10.0,
        "rate_max": 10.0,
        "spiky": False,
        "source": "flat logical schedule fixture",
    })
    summary["arrivals"].update({
        "achieved_qps_overall": 10.0,
        "scheduled_qps": 10.0,
        "logical_schedule_seconds": 20.0,
    })
    summary["physical_post_attempts"] = {
        "logical_rows_with_additional_attempts": 200,
        "additional_attempts": 400,
        "recorded_retry_triggers": {"transport": 400},
        "distinct_retry_triggers_at_least": 1,
        "retry_trigger_categories_truncated": False,
        "retry_trigger_coverage_rows": 200,
        "legacy_retry_marked_rows_without_attempt_count": 0,
    }
    summary["observed_rate_windows"] = {
        "traffic_scope": {
            "rows": 200,
            "sent_rows": 200,
            "physical_attempts_estimate": 600,
        },
        "physical_queries_per_one_second_by_request_start": {
            "max": 30,
            "window_seconds": 1.0,
            "events_total": 600,
            "logical_rows": 200,
            "attempt_counts_exact": True,
            "all_attempt_timestamps_exact": True,
        },
    }

    scope = _plain(_Document(render_html(
        summary, "flat logical schedule")).by_id("test-scope").text())
    assert "Bursty schedule" not in scope
    assert "Not classified as bursty" in scope
    assert "Flat schedule" not in scope
    assert "200 logical requests over 20 s" in scope
    assert "Configured rate-curve peak 10 logical requests/s" in scope
    assert "Scheduled 1-second peak" not in scope
    assert "Observed 1-second peak 30 all-phase physical HTTP POST starts/s" \
        in scope
    assert "600 all-phase physical HTTP POST attempts across replay plus " \
        "setup phases" in scope
    assert not re.search(r"Observed 1-second peak\s+30\s+requests/s",
                         scope)


def test_token_fidelity_mismatch_is_visible_before_engineering_details():
    rows = _rows(200)
    for index, row in enumerate(rows):
        row.update({
            "prompt_tokens": 40 + index % 12,
            "intended_input_tokens": 40,
            "completion_tokens": 5,
            "intended_output_tokens": 5,
            "max_tokens_requested": 5,
            "finish_reason": "length",
            "truncated": True,
        })
    summary = summarize(rows)
    targeting = summary["token_targeting"]
    assert targeting["status"] == "mismatch"
    assert targeting["warning"]

    document = _Document(render_html(summary, "token mismatch"))
    customer_text = " ".join((
        _plain(document.by_id("summary").text()),
        _plain(document.by_id("limitations").text()),
        _plain(document.by_id("test-scope").text()),
    ))
    assert "Token-shape mismatch" in customer_text
    assert re.search(r"intended(?: prompt)?\s+40(?:–40)? tokens",
                     customer_text, re.IGNORECASE)
    assert re.search(r"observed(?: prompt)?\s+40–51 tokens",
                     customer_text, re.IGNORECASE)
    assert targeting["warning"] in customer_text


def test_legacy_token_warning_is_customer_visible_without_status_or_tables():
    summary = _summary(200)
    warning = (
        "legacy token evidence says the measured prompt shape did not "
        "reproduce the declared workload")
    targeting = summary["token_targeting"]
    targeting.pop("status", None)
    targeting.pop("intended_prompt_tokens", None)
    targeting.pop("intended_completion_tokens", None)
    targeting["warning"] = warning

    document = _Document(render_html(summary, "legacy token warning"))
    before_engineering = " ".join((
        _plain(document.by_id("summary").text()),
        _plain(document.by_id("limitations").text()),
        _plain(document.by_id("test-scope").text()),
    ))
    assert "Token-shape mismatch" in before_engineering
    assert warning in before_engineering


def test_999_eligible_rows_support_p95_but_never_publish_a_p99_number():
    summary = _summary(999)
    assert summary["sample"]["supports"] == ["p50", "p90", "p95"]
    assert summary["sample"]["indicative_only"] == ["p99"]

    document = _Document(render_html(summary, "999 eligible observations"))
    customer_latency = " ".join((
        _plain(document.by_id("summary").text()),
        _plain(document.by_id("latency").text()),
    ))
    next_test = _plain(document.by_id("next-test").text())
    assert re.search(r"p95 supported(?:\s*\(n=999\))?", customer_latency,
                     re.IGNORECASE)
    assert "p99 requires 1,000 eligible successful observations; have 999" \
        in customer_latency
    assert not re.search(r"p99\s+[\d,]+(?:\.\d+)?\s*ms", customer_latency,
                         re.IGNORECASE)
    assert "1,000 eligible successful observations for p99" in next_test


def test_customer_spike_wording_reports_observation_without_backend_cause():
    summary = _summary(200)
    summary["drift"] = {
        "drift_kind": "spike",
        "drift_flag": True,
        "drift_headline": (
            "a middle window is much worse than the ends: something "
            "transient hit the endpoint mid-run"),
        "latency_p95_best": 1_244.0,
        "latency_p95_worst": 6_515.0,
        "latency_p95_spread_ratio": 5.24,
        "counted_windows": 20,
        "window_seconds": 60,
    }

    document = _Document(render_html(summary, "spike attribution boundary"))
    limitations = document.by_id("limitations")
    stability = [node for node in limitations.walk()
                 if "limitation" in _class_tokens(node)
                 and "Stability" in _plain(node.text())]
    assert stability
    stability_text = _plain(stability[0].text())
    assert "hit the endpoint" not in stability_text.casefold()
    assert re.search(r"latency spike observed in the measured (?:path|run)",
                     stability_text, re.IGNORECASE)
    assert "unable to attribute without backend correlation" \
        in stability_text.casefold()


def test_legacy_failing_drift_causal_text_is_sanitized_for_customers():
    summary = _summary(200)
    summary["drift"] = {
        "drift_kind": "failing",
        "drift_flag": True,
        "drift_headline": "the endpoint collapsed during the final window",
        "counted_windows": 3,
        "window_seconds": 60,
    }

    document = _Document(render_html(summary, "legacy failing drift"))
    limitations = document.by_id("limitations")
    stability = [node for node in limitations.walk()
                 if "limitation" in _class_tokens(node)
                 and "Stability" in _plain(node.text())]
    assert stability
    stability_text = _plain(stability[0].text())
    assert "endpoint collapsed" not in stability_text.casefold()
    assert "measured path/run" in stability_text.casefold()
    assert "unable to attribute" in stability_text.casefold()


def test_retry_observation_is_total_only_without_fake_attempt_breakdown():
    rows = _rows(
        1,
        first_response=[1_050.0],
        complete_response=[1_400.0],
    )
    rows[0].update({
        "caller_send_ms": 650.0,
        "connect_ms": 50.0,
        "request_attempts": 2,
    })
    summary = summarize(rows)
    observation = summary["latency_observations"]["rows"][0]
    components = (
        "client_before_connection_ms",
        "connection_setup_ms",
        "request_to_first_response_ms",
        "first_response_to_complete_ms",
        "retry_or_unattributed_ms",
    )
    if observation["decomposition_status"] == "complete":
        assert sum(observation[key] for key in components) == \
            observation["caller_complete_ms"]
    else:
        assert observation["decomposition_status"] == "total_only_after_retry"
        assert all(observation[key] is None for key in components)

    latency = _Document(render_html(summary, "retry observation")).by_id(
        "latency")
    request_rows = [node for node in latency.walk()
                    if "waterfall-row" in _class_tokens(node)]
    assert len(request_rows) == 1
    request_text = _plain(request_rows[0].text())
    assert re.search(r"1,?400\s*ms", request_text)
    assert re.search(r"retry(?: segment)? attribution unavailable",
                     request_text, re.IGNORECASE)
    assert "n/a ms" not in request_text.casefold()


def test_met_target_is_not_green_when_measurement_is_not_valid():
    acceptance = {
        "targets_are": "customer requirements",
        "ttft_ms": {"p50": 2_000},
    }
    caution = _summary(200, acceptance=acceptance)
    invalid = _summary(200, acceptance=acceptance)
    invalid["response_identity"] = {
        "status": "invalid",
        "invalid": "response model identity changed during the run",
    }

    for expected_state, summary in (("CAUTION", caution),
                                    ("INVALID", invalid)):
        target = summary["sla"]["ttft_vs_target"][0]
        assert target["met"] is True
        assert target["statistically_demonstrated"] is True
        decision = build_report_decision(summary)
        assert decision["measurement_validity"]["code"] == expected_state

        document = _Document(render_html(
            summary, f"{expected_state.lower()} measurement"))
        customer_summary = document.by_id("summary")
        latency = document.by_id("latency")
        assert "NOT PROVEN" in _plain(customer_summary.text())
        assert "NOT PROVEN" in _plain(latency.text())
        assert not any("tone-ok" in _class_tokens(node)
                       and "Acceptance" in _plain(node.text())
                       for node in customer_summary.walk())
        assert not any("target-pass" in _class_tokens(node)
                       for node in latency.walk())


def test_unverified_source_and_verified_receipt_use_one_acceptance_policy():
    summary = _summary(1000, acceptance={
        "targets_are": "customer requirements",
        "ttft_ms": {"p95": 2_000},
    })
    summary["run"]["transport"].update({
        "production_connection_policy_match": True,
        "production_comparability_warning": None,
    })
    summary["drift"] = {"drift_kind": "stable"}
    source_decision = build_report_decision(summary)
    assert source_decision["measurement_validity"]["code"] == "VALID"
    assert source_decision["customer_sla"]["code"] == "PASS"

    source = _Document(render_html(summary, "unverified source"))
    assert "NOT PROVEN" in _plain(source.by_id("summary").text())
    assert "NOT PROVEN" in _plain(source.by_id("latency").text())
    assert not any("target-pass" in _class_tokens(node)
                   for node in source.by_id("latency").walk())

    verified = _Document(render_html(
        summary,
        "verified receipt",
        verification_context=_verification_context(summary),
    ))
    assert any("target-pass" in _class_tokens(node)
               for node in verified.by_id("latency").walk())


def test_customer_navigation_and_progressive_disclosures_are_stable():
    html = render_html(_summary(), "customer navigation")
    document = _Document(html)
    navs = [node for node in document.root.walk()
            if node.tag == "nav"
            and node.attrs.get("aria-label") == "Report sections"]
    assert len(navs) == 1
    anchors = [node for node in navs[0].walk() if node.tag == "a"]
    assert [_plain(node.text()) for node in anchors] == [
        "Summary", "Latency", "Reliability", "Test scope", "Engineering",
    ]
    assert [node.attrs.get("href") for node in anchors] == [
        "#summary", "#latency", "#reliability", "#test-scope", "#engineering",
    ]

    positions = [html.index(f"id='{node_id}'") for node_id in (
        "summary", "latency", "reliability", "test-scope", "engineering")]
    assert positions == sorted(positions)

    engineering = document.details("Databricks engineering diagnostics")
    glossary = document.details("Field definitions and raw evidence")
    assert engineering.attrs.get("id") == "engineering"
    assert glossary.attrs.get("id") == "field-glossary"
    assert "open" not in engineering.attrs
    assert "open" not in glossary.attrs
    assert "Final-attempt request-path latency" in engineering.text()
    assert "Missing, unknown" in glossary.text()


def test_zero_http_429_is_neutral_and_report_is_self_contained():
    html = render_html(_summary(), "neutral quota report")
    reliability = _plain(_Document(html).by_id("reliability").text())

    assert re.search(r"(?:0\s+HTTP 429|No HTTP 429)", reliability,
                     re.IGNORECASE)
    assert re.search(
        r"class=['\"][^'\"]*neutral[^'\"]*['\"][^>]*>"
        r"[^<]*(?:0\s+HTTP 429|No HTTP 429)",
        html,
        re.IGNORECASE,
    )
    assert "quota headroom" in reliability.casefold()
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.casefold()
    assert "<link" not in html.casefold()
    assert "http://" not in html and "https://" not in html
