"""Focused contract tests for the canonical report decision model."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from traffic_replay.report_decision import (
    IntegrityContext,
    build_report_decision,
)


def _summary(*, with_sla: bool = True) -> dict:
    summary = {
        "requests_total": 1_000,
        "requests_ok": 1_000,
        "requests_failed": 0,
        "answers": {
            "judged": 1_000,
            "acceptable_outcomes": 1_000,
            "answered": 1_000,
        },
        "sample": {
            "n": 1_000,
            "supports": ["p50", "p90", "p95", "p99"],
            "indicative_only": [],
        },
        "drift": {"drift_kind": "stable"},
        "latency_population": {
            "kind": "readable_answers",
            "n": 1_000,
        },
        "token_targeting": {"status": "verified", "warning": None},
        "throughput": {"coverage_warning": None},
        "arrivals": {"achieved_qps_overall": 8.25},
        "schedule": {
            "requests": 1_000,
            "seconds": 120,
            "source": "customer trace",
        },
        "run": {"aggregation_valid": True},
        "http_429_count": 0,
        "http_429": {
            "count": 0,
            "request_rows_examined": 1_005,
            "http_status_observed_for": 1_005,
            "phases": {},
            "scope": "all supplied request phases",
        },
        "rate_limits": {
            "binding": {"binding_complete": True},
            "warning": None,
        },
    }
    if with_sla:
        summary["sla"] = {
            "targets_source": "customer requirements",
            "acceptance_config": {
                "ttft_ms": {"p95": 900},
                "ttfg_ms": {"p95": 2_500},
                "hard_timeouts": {"ttft_s": 15, "ttfg_s": 45},
                "interchunk_ms": 2_000,
                "success_rate": 0.99,
            },
            "ttft_vs_target": [{
                "quantile": "p95", "target_ms": 900,
                "actual_ms": 500, "met": True,
            }],
            "ttfg_vs_target": [{
                "quantile": "p95", "target_ms": 2_500,
                "actual_ms": 1_500, "met": True,
            }],
            "hard_timeout_breaches": 0,
            "hard_timeout_basis": {
                "ttft_cap_ms": 15_000,
                "ttfg_cap_ms": 45_000,
                "interchunk_cap_ms": 2_000,
            },
            "interchunk_breaches": 0,
            "success_rate": {
                "target": 0.99,
                "actual": 1.0,
                "met": True,
                "successes": 1_000,
                "attempts": 1_000,
                "statistically_demonstrated": True,
            },
        }
    return summary


VERIFIED = IntegrityContext(
    status="verified", reason="manifest and all bound artifacts match")


def test_clean_verified_run_keeps_all_five_decisions_separate():
    decision = build_report_decision(_summary(), VERIFIED)

    assert decision["decision_schema_version"] == 1
    assert decision["evidence_integrity"]["code"] == "VERIFIED"
    assert decision["measurement_validity"]["code"] == "VALID"
    assert decision["customer_sla"]["code"] == "PASS"
    assert decision["quota_state"]["code"] == "NOT_OBSERVED"
    assert decision["endpoint_capacity"]["code"] == "HELD_AT_TESTED_LOAD"
    assert decision["endpoint_capacity"]["endpoint_ceiling_established"] is False
    assert decision["quota_state"]["provider_headroom_established"] is False
    assert "does not establish provider quota headroom" in \
        decision["quota_state"]["reason"]
    assert "not an endpoint ceiling" in \
        decision["endpoint_capacity"]["reason"]


def test_sla_miss_is_retained_when_quota_makes_capacity_inconclusive():
    summary = _summary()
    summary["sla"]["ttft_vs_target"][0].update(
        {"actual_ms": 1_200, "met": False})
    summary["http_429_count"] = 3
    summary["http_429"].update({
        "count": 3,
        "request_rows_examined": 1_008,
        "http_status_observed_for": 1_008,
        "phases": {"preflight": 1, "replay": 2},
    })

    decision = build_report_decision(summary, VERIFIED)

    assert decision["measurement_validity"]["code"] == "INVALID"
    assert decision["customer_sla"]["code"] == "MISS"
    assert decision["quota_state"]["code"] == "EXCEEDED"
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"
    assert decision["quota_state"]["http_429"] == {
        "http_429_count": 3,
        "request_rows_examined": 1_008,
        "http_status_observed_for": 1_008,
        "phases": {"preflight": 1, "replay": 2},
        "scope": "all supplied request phases",
        "evidence_inconsistent": False,
    }
    assert "3/1008" in decision["quota_state"]["reason"]
    assert "preflight=1, replay=2" in decision["quota_state"]["reason"]


def test_passing_sla_checks_are_visibly_qualified_on_invalid_measurement():
    summary = _summary()
    summary["http_429_count"] = 1
    summary["http_429"].update({
        "count": 1,
        "request_rows_examined": 1_006,
        "http_status_observed_for": 1_006,
        "phases": {"replay": 1},
    })

    decision = build_report_decision(summary, VERIFIED)

    sla = decision["customer_sla"]
    assert sla["code"] == "PASS"
    assert sla["severity"] == "warning"
    assert sla["label"] == "Acceptance checks passed - invalid run"
    assert "not a clean acceptance pass" in sla["reason"]
    assert "MEASUREMENT_BLOCKS_CLEAN_SLA_PASS" in sla["reason_codes"]


@pytest.mark.parametrize(
    "phase", ["preflight", "probe", "sizing", "calibration", "replay"])
def test_one_429_in_any_captured_phase_exceeds_quota(phase):
    summary = _summary()
    summary["http_429_count"] = 1
    summary["http_429"].update({
        "count": 1,
        "request_rows_examined": 1_006,
        "http_status_observed_for": 1_006,
        "phases": {phase: 1},
    })

    decision = build_report_decision(summary, VERIFIED)

    assert decision["quota_state"]["code"] == "EXCEEDED"
    assert decision["quota_state"]["http_429"]["phases"] == {phase: 1}
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"
    assert "QUOTA_REJECTION_OBSERVED" in \
        decision["measurement_validity"]["reason_codes"]


def test_no_429_means_only_none_observed_not_headroom():
    decision = build_report_decision(_summary(), VERIFIED)
    quota = decision["quota_state"]

    assert quota["code"] == "NOT_OBSERVED"
    assert quota["http_429"]["http_429_count"] == 0
    assert quota["http_429"]["request_rows_examined"] == 1_005
    assert quota["provider_headroom_established"] is False
    assert "headroom" in quota["reason"]


def test_incomplete_http_status_coverage_is_unknown_not_clean():
    summary = _summary()
    summary["http_429"]["http_status_observed_for"] = 1_000

    decision = build_report_decision(summary, VERIFIED)

    assert decision["quota_state"]["code"] == "UNKNOWN"
    assert "1000/1005" in decision["quota_state"]["reason"]
    assert decision["measurement_validity"]["code"] == "CAUTION"
    assert "HTTP_STATUS_COVERAGE_INCOMPLETE" in \
        decision["measurement_validity"]["reason_codes"]
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"


def test_positive_429_alias_cannot_be_erased_by_conflicting_nested_count():
    summary = _summary()
    summary["http_429_count"] = 1
    summary["http_429"].update({
        "count": 0,
        "phases": {"replay": 1},
    })

    decision = build_report_decision(summary, VERIFIED)

    assert decision["quota_state"]["code"] == "EXCEEDED"
    assert decision["quota_state"]["http_429"][
        "evidence_inconsistent"] is True
    assert "HTTP_429_EVIDENCE_INCONSISTENT" in \
        decision["quota_state"]["reason_codes"]


def test_contradictory_empty_population_is_unknown_not_not_evaluated():
    summary = _summary()
    summary["http_429"].update({
        "request_rows_examined": 0,
        "http_status_observed_for": 1,
    })

    decision = build_report_decision(summary, VERIFIED)

    assert decision["quota_state"]["code"] == "UNKNOWN"
    assert decision["measurement_validity"]["code"] == "INVALID"
    assert "HTTP_429_EVIDENCE_INCONSISTENT" in \
        decision["quota_state"]["reason_codes"]


def test_no_targets_is_not_evaluated_not_a_pass():
    decision = build_report_decision(_summary(with_sla=False), VERIFIED)

    assert decision["customer_sla"]["code"] == "NOT_EVALUATED"
    assert decision["customer_sla"]["reason_codes"] == ["NO_SLA_TARGETS"]
    assert "no pass or miss is claimed" in \
        decision["customer_sla"]["reason"]


def test_unmeasured_configured_target_is_sla_inconclusive():
    summary = _summary()
    summary["sla"]["ttft_vs_target"][0].update(
        {"actual_ms": None, "met": None})

    decision = build_report_decision(summary, VERIFIED)

    assert decision["customer_sla"]["code"] == "INCONCLUSIVE"
    assert decision["customer_sla"]["evaluation"]["unmeasured"] == 1


def test_success_point_estimate_without_required_confidence_is_inconclusive():
    summary = _summary()
    summary["sla"]["success_rate"].update({
        "target": 0.9999,
        "actual": 1.0,
        "met": True,
        "successes": 1_200,
        "attempts": 1_200,
        "one_sided_95pct_wilson_lower": 0.99775,
        "statistically_demonstrated": False,
    })

    decision = build_report_decision(summary, VERIFIED)
    sla = decision["customer_sla"]

    assert sla["code"] == "INCONCLUSIVE"
    assert sla["label"] == "Acceptance checks inconclusive"
    assert "Wilson lower confidence bound did not" in sla["reason"]
    assert sla["reason_codes"] == [
        "SUCCESS_RATE_CONFIDENCE_NOT_DEMONSTRATED"]
    assert sla["evaluation"][
        "success_rate_confidence_not_demonstrated"] == 1


def test_legacy_success_rate_without_confidence_field_keeps_point_estimate():
    summary = _summary()
    summary["sla"]["success_rate"].pop("statistically_demonstrated")

    decision = build_report_decision(summary, VERIFIED)

    assert decision["customer_sla"]["code"] == "PASS"


@pytest.mark.parametrize("path,warning,expected_code", [
    (("token_targeting",), "input token shape missed by 40%",
     "TOKEN_FIDELITY_UNVERIFIED"),
    (("cache_fidelity",), "cache fraction was not reproduced",
     "CACHE_FIDELITY_UNVERIFIED"),
    (("sla",), "TTFT exists for only 80 of 100 answers",
     "SLA_COVERAGE_INCOMPLETE"),
])
def test_fidelity_and_coverage_warnings_are_measurement_cautions_without_erasure(
        path, warning, expected_code):
    summary = _summary()
    if path == ("sla",):
        summary["sla"]["coverage_warning"] = warning
    else:
        summary[path[0]] = {"warning": warning}

    decision = build_report_decision(summary, VERIFIED)

    assert decision["measurement_validity"]["code"] == "CAUTION"
    assert expected_code in decision["measurement_validity"]["reason_codes"]
    assert decision["customer_sla"]["code"] == "PASS"
    assert decision["quota_state"]["code"] == "NOT_OBSERVED"
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"


def test_answer_invalidity_does_not_erase_sla_or_quota_facts():
    summary = _summary()
    summary["answers"]["invalid"] = "no request produced a readable answer"

    decision = build_report_decision(summary, VERIFIED)

    assert decision["measurement_validity"]["code"] == "INVALID"
    assert decision["customer_sla"]["code"] == "PASS"
    assert decision["quota_state"]["code"] == "NOT_OBSERVED"


def test_incompatible_aggregate_is_measurement_invalid_only():
    summary = _summary()
    summary["run"] = {
        "aggregation_valid": False,
        "compatibility_issues": ["model differs", "profile differs"],
    }

    decision = build_report_decision(summary, VERIFIED)

    measurement = decision["measurement_validity"]
    assert measurement["code"] == "INVALID"
    assert "INCOMPATIBLE_AGGREGATE" in measurement["reason_codes"]
    assert "model differs" in measurement["reason"]
    assert decision["customer_sla"]["code"] == "PASS"


def test_unknown_endpoint_binding_blocks_capacity_claim_only():
    summary = _summary()
    summary["rate_limits"]["binding"] = {
        "binding_complete": False,
        "reasons": ["endpoint metadata was not captured"],
    }
    # Keep the rate-limit warning absent to prove the binding itself is a gate.
    summary["rate_limits"]["warning"] = None

    decision = build_report_decision(summary, VERIFIED)

    assert decision["measurement_validity"]["code"] == "CAUTION"
    assert "ENDPOINT_BINDING_UNVERIFIED" in \
        decision["measurement_validity"]["reason_codes"]
    assert decision["customer_sla"]["code"] == "PASS"
    assert decision["quota_state"]["code"] == "NOT_OBSERVED"
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"
    assert "ENDPOINT_BINDING_UNVERIFIED" in \
        decision["endpoint_capacity"]["reason_codes"]


def test_summary_only_integrity_is_verify_required_and_blocks_capacity_claim():
    decision = build_report_decision(_summary())

    assert decision["evidence_integrity"]["code"] == "VERIFY_REQUIRED"
    assert decision["measurement_validity"]["code"] == "VALID"
    assert decision["customer_sla"]["code"] == "PASS"
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"
    assert "EVIDENCE_NOT_VERIFIED" in \
        decision["endpoint_capacity"]["reason_codes"]


def test_explicit_tamper_context_is_not_rendered_as_unverified_ambiguity():
    decision = build_report_decision(
        _summary(), {"status": "tampered", "reason": "summary digest differs"})

    integrity = decision["evidence_integrity"]
    assert integrity["code"] == "TAMPERED"
    assert integrity["label"] == "Integrity failed"
    assert integrity["reason"] == "summary digest differs"
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"


def test_nonquota_failures_are_a_failed_test_point_not_a_ceiling():
    summary = _summary()
    summary.update({
        "requests_ok": 990,
        "requests_failed": 10,
    })
    summary["answers"].update({
        "judged": 1_000,
        "acceptable_outcomes": 990,
        "answered": 990,
    })

    decision = build_report_decision(summary, VERIFIED)

    capacity = decision["endpoint_capacity"]
    assert decision["measurement_validity"]["code"] == "VALID"
    assert capacity["code"] == "NOT_HELD_AT_TESTED_LOAD"
    assert capacity["endpoint_ceiling_established"] is False
    assert "not the endpoint ceiling" in capacity["reason"]


def test_tested_load_uses_only_direct_summary_facts():
    decision = build_report_decision(_summary(), VERIFIED)

    assert decision["tested_load"] == {
        "measured_replay_requests": 1_000,
        "measured_replay_ok": 1_000,
        "measured_replay_failed": 0,
        "acceptable_outcomes": 1_000,
        "answer_rows_judged": 1_000,
        "achieved_qps_overall": 8.25,
        "scheduled_requests": 1_000,
        "scheduled_seconds": 120,
        "schedule_source": "customer trace",
        "captured_quota_request_rows": 1_005,
        "claim_boundary": (
            "Observed tested-load facts only; they do not establish an "
            "endpoint ceiling or provider quota headroom."),
    }


def test_unverified_operator_pricing_qualifies_measurement_state():
    summary = _summary()
    summary["cost"] = {
        "mode": "per_token",
        "complete": True,
        "applicability_warning": (
            "rates were supplied but not bound to this product/tier"),
    }

    decision = build_report_decision(summary, VERIFIED)

    measurement = decision["measurement_validity"]
    assert measurement["code"] == "CAUTION"
    assert "PRICING_APPLICABILITY_UNVERIFIED" in measurement["reason_codes"]
    assert decision["endpoint_capacity"]["code"] == "INCONCLUSIVE"


def test_model_is_deterministic_json_serializable_and_does_not_mutate_input():
    summary = _summary()
    # Deliberately insert phase keys out of order; canonical output sorts them.
    summary["http_429_count"] = 2
    summary["http_429"].update({
        "count": 2,
        "request_rows_examined": 1_007,
        "http_status_observed_for": 1_007,
        "phases": {"replay": 1, "calibration": 1},
    })
    before = deepcopy(summary)

    first = build_report_decision(summary, VERIFIED)
    second = build_report_decision(summary, VERIFIED)
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)

    assert first == second
    assert summary == before
    assert json.loads(encoded) == first
    assert list(first["quota_state"]["http_429"]["phases"]) == [
        "calibration", "replay"]


@pytest.mark.parametrize("context", [
    {"status": "green"},
    {"status": "verified", "unexpected": True},
])
def test_integrity_context_is_closed_and_fails_fast(context):
    with pytest.raises(ValueError):
        build_report_decision(_summary(), context)
