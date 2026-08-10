"""Canonical, presentation-independent decision states for run reports.

The benchmark answers several different questions.  Combining them into one
red/amber/green banner makes it too easy for a clean latency percentile to
hide a quota rejection, or for an invalid measurement to erase an observed
acceptance-target miss. This module deliberately keeps five decisions independent and
returns only JSON-serializable values so Markdown, HTML, and automation can
render the same facts.

``build_report_decision`` consumes a canonical ``summary.json`` object.  Its
HTTP-429 block is produced from captured request calls by the runner and
can include setup phases as well as replay.  Artifact integrity is a separate
input: a summary cannot prove the seal that contains it, so the default is
``VERIFY_REQUIRED`` until a caller supplies explicit verification context.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

from .artifacts import sanitize_display_text


DECISION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IntegrityContext:
    """Result of verifying the sealed artifact that contains the summary.

    ``status`` is intentionally small and closed.  The verifier, rather than
    a report renderer, owns the transition to ``verified`` or ``tampered``.
    """

    status: str = "verify_required"
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"verified", "verify_required", "tampered"}:
            raise ValueError(
                "integrity status must be verified, verify_required, or "
                "tampered")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("integrity reason must be a string or None")


def _one_line(value: object, *, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", sanitize_display_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _finite_number(value: object, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _state(code: str, label: str, reason: str,
           reason_codes: list[str], *, severity: str,
           reason_details: list[tuple[str, str]] | None = None) -> dict:
    """Build one decision state without losing its individual gate reasons.

    ``reason`` remains the compact compatibility summary used by existing
    automation. ``reason_details`` is the presentation-independent, ordered
    code/message list.  Reports render that list above every metric so a third
    or later caution cannot disappear behind a ``plus N more`` summary.
    """
    unique_codes = list(dict.fromkeys(reason_codes))
    if reason_details is None:
        normalized_details = [
            {"code": item, "message": _one_line(reason, limit=1_000)}
            for item in unique_codes
        ]
    else:
        normalized_details = []
        seen: set[tuple[str, str]] = set()
        for detail_code, detail_message in reason_details:
            item = (
                _one_line(detail_code, limit=160),
                _one_line(detail_message, limit=1_000),
            )
            if item in seen:
                continue
            seen.add(item)
            normalized_details.append({
                "code": item[0],
                "message": item[1],
            })
        detail_codes = {item["code"] for item in normalized_details}
        for item in unique_codes:
            if item not in detail_codes:
                normalized_details.append({
                    "code": item,
                    "message": _one_line(reason, limit=1_000),
                })
        unique_codes = list(dict.fromkeys([
            *unique_codes,
            *(item["code"] for item in normalized_details),
        ]))
    return {
        "code": code,
        "label": label,
        "severity": severity,
        "reason": _one_line(reason),
        "reason_codes": unique_codes,
        "reason_details": normalized_details,
    }


def _integrity_context(value: IntegrityContext | Mapping | None) \
        -> IntegrityContext:
    if value is None:
        return IntegrityContext()
    if isinstance(value, IntegrityContext):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("integrity must be an IntegrityContext, mapping, or None")
    unknown = set(value) - {"status", "reason"}
    if unknown:
        raise ValueError(
            "unknown integrity context field(s): " + ", ".join(
                sorted(str(key) for key in unknown)))
    return IntegrityContext(
        status=value.get("status", "verify_required"),
        reason=value.get("reason"),
    )


def _evidence_integrity(context: IntegrityContext) -> dict:
    if context.status == "verified":
        reason = context.reason or (
            "The sealed artifact and its manifest were explicitly verified "
            "before this decision was rendered.")
        return _state(
            "VERIFIED", "Evidence verified", reason,
            ["ARTIFACT_VERIFIED"], severity="pass")
    if context.status == "tampered":
        reason = context.reason or (
            "Artifact verification failed; at least one sealed byte or "
            "manifest binding does not match.")
        return _state(
            "TAMPERED", "Integrity failed", reason,
            ["ARTIFACT_TAMPERED"], severity="fail")
    reason = context.reason or (
        "This decision was rendered from a summary without an explicit "
        "sealed-artifact verification result; verify the manifest before "
        "relying on it.")
    return _state(
        "VERIFY_REQUIRED", "Verification required", reason,
        ["ARTIFACT_NOT_VERIFIED"], severity="warning")


def _quota_facts(summary: Mapping) -> dict:
    block = summary.get("http_429")
    block = block if isinstance(block, Mapping) else {}
    nested_count = _nonnegative_int(block.get("count"))
    alias_count = _nonnegative_int(summary.get("http_429_count"))
    inconsistent = (
        nested_count is not None and alias_count is not None
        and nested_count != alias_count)
    if "count" in block and nested_count is None:
        inconsistent = True
    if "http_429_count" in summary and alias_count is None:
        inconsistent = True
    # Never let a missing or conflicting alias erase positive 429 evidence.
    count = max(value for value in (nested_count, alias_count, 0)
                if value is not None)
    rows = _nonnegative_int(block.get("request_rows_examined"))
    observed = _nonnegative_int(block.get("http_status_observed_for"))
    if "request_rows_examined" in block and rows is None:
        inconsistent = True
    if "http_status_observed_for" in block and observed is None:
        inconsistent = True
    if rows is not None and count > rows:
        inconsistent = True
    if rows is not None and observed is not None and observed > rows:
        inconsistent = True
    if observed is not None and observed < count:
        inconsistent = True

    phases_value = block.get("phases")
    phases: dict[str, int] = {}
    phase_evidence_valid = isinstance(phases_value, Mapping)
    if phase_evidence_valid:
        for raw_name, raw_count in phases_value.items():
            phase_count = _nonnegative_int(raw_count)
            if phase_count is None:
                phase_evidence_valid = False
                continue
            if phase_count:
                name = _one_line(raw_name, limit=80) or "unlabeled"
                phases[name] = phases.get(name, 0) + phase_count
    phases = {name: phases[name] for name in sorted(phases)}
    if count and (not phase_evidence_valid or sum(phases.values()) != count):
        inconsistent = True

    local = summary.get("runtime_quota_admission")
    local = local if isinstance(local, Mapping) else {}
    local_denied_rows = _nonnegative_int(local.get("denied_rows")) or 0
    local_denied_attempts = _nonnegative_int(
        local.get("denied_attempts_in_captured_rows"))
    if local_denied_attempts is None:
        # Compatibility with short-lived development artifacts from before
        # preflight/probe/sizing rows were included in this evidence block.
        local_denied_attempts = _nonnegative_int(
            local.get("denied_attempts_in_replay_rows"))
    local_denied_attempts = local_denied_attempts or 0
    local_status = (local.get("status")
                    if isinstance(local.get("status"), str) else None)
    local_invariant_errors = local.get("invariant_errors")
    if not isinstance(local_invariant_errors, list):
        local_invariant_errors = []
    return {
        "count": count,
        "request_rows_examined": rows,
        "http_status_observed_for": observed,
        "phases": phases,
        "evidence_inconsistent": inconsistent,
        "scope": (_one_line(block["scope"])
                  if isinstance(block.get("scope"), str) else None),
        "local_guard_status": local_status,
        "local_guard_denied_rows": local_denied_rows,
        "local_guard_denied_attempts": local_denied_attempts,
        "local_guard_invariant_errors": [
            _one_line(item) for item in local_invariant_errors],
    }


def _quota_state(facts: dict) -> dict:
    count = facts["count"]
    rows = facts["request_rows_examined"]
    observed = facts["http_status_observed_for"]
    phases = facts["phases"]
    evidence = {
        "http_429_count": count,
        "request_rows_examined": rows,
        "http_status_observed_for": observed,
        "phases": phases,
        "scope": facts["scope"],
        "evidence_inconsistent": facts["evidence_inconsistent"],
    }
    local_evidence = {
        "status": facts["local_guard_status"],
        "denied_rows": facts["local_guard_denied_rows"],
        "denied_attempts_in_captured_rows": facts[
            "local_guard_denied_attempts"],
        "invariant_errors": facts["local_guard_invariant_errors"],
    }

    if count:
        denominator = str(rows) if rows is not None and rows >= count else "unknown"
        phase_text = ", ".join(
            f"{name}={amount}" for name, amount in phases.items())
        reason = (
            f"HTTP 429 occurred in {count}/{denominator} captured request "
            f"rows" + (f" ({phase_text})" if phase_text else "")
            + ". This proves a quota or rate-limit rejection, but not which "
            "dimension or component enforced it.")
        out = _state(
            "EXCEEDED", "HTTP 429 / rate-limit rejection observed", reason,
            ["HTTP_429_OBSERVED"] + (
                ["HTTP_429_EVIDENCE_INCONSISTENT"]
                if facts["evidence_inconsistent"] else []),
            severity="fail")
    elif (facts["local_guard_denied_rows"]
          or facts["local_guard_denied_attempts"]
          or facts["local_guard_status"] == "denied"):
        out = _state(
            "LOCAL_GUARD_REFUSED", "Runtime quota admission refused",
            "The command-local guard refused at least one physical POST "
            "before send. The requested load was not delivered; this is a "
            "harness quota-safety stop, not endpoint-capacity evidence.",
            ["RUNTIME_QUOTA_ADMISSION_REFUSED"], severity="fail")
    elif (facts["local_guard_status"] == "invalid_evidence"
          or facts["local_guard_invariant_errors"]):
        out = _state(
            "UNKNOWN", "Quota state unknown",
            "Runtime quota-guard evidence failed its internal invariants.",
            ["RUNTIME_QUOTA_EVIDENCE_INVALID"], severity="warning")
    elif facts["evidence_inconsistent"]:
        out = _state(
            "UNKNOWN", "Quota state unknown",
            "The HTTP-429 aliases or denominators disagree, so the absence "
            "of a recorded 429 is not trustworthy.",
            ["HTTP_429_EVIDENCE_INCONSISTENT"], severity="warning")
    elif rows == 0:
        out = _state(
            "NOT_EVALUATED", "Quota not evaluated",
            "No captured request row was available to check for HTTP 429.",
            ["NO_CAPTURED_REQUEST_ROWS"], severity="neutral")
    elif rows is None:
        out = _state(
            "UNKNOWN", "Quota state unknown",
            "The summary does not contain a complete HTTP-status evidence "
            "population, so quota rejections cannot be assessed.",
            ["HTTP_STATUS_EVIDENCE_MISSING"], severity="warning")
    elif observed is None or observed < rows:
        shown = "unknown" if observed is None else str(observed)
        out = _state(
            "UNKNOWN", "Quota state unknown",
            f"No HTTP 429 was observed, but HTTP status was retained for "
            f"only {shown}/{rows} captured request rows.",
            ["HTTP_STATUS_COVERAGE_INCOMPLETE"], severity="warning")
    else:
        out = _state(
            "NOT_OBSERVED", "No quota rejection observed",
            f"No HTTP 429 was observed in {rows}/{rows} captured request "
            "rows. This does not establish provider quota headroom.",
            ["HTTP_429_NOT_OBSERVED"], severity="pass")
    out["http_429"] = evidence
    out["runtime_quota_admission"] = local_evidence
    out["provider_headroom_established"] = False
    return out


def _count_integrity_issues(summary: Mapping) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    reasons: list[str] = []
    total = _nonnegative_int(summary.get("requests_total"))
    ok = _nonnegative_int(summary.get("requests_ok"))
    failed = _nonnegative_int(summary.get("requests_failed"))
    if total is None or ok is None or failed is None:
        codes.append("SUMMARY_COUNTS_INCOMPLETE")
        reasons.append("request totals are missing or malformed")
    elif ok + failed != total:
        codes.append("SUMMARY_COUNTS_INCONSISTENT")
        reasons.append(
            f"request counts disagree ({ok} ok + {failed} failed != {total} total)")
    elif total == 0:
        codes.append("NO_MEASURED_REQUESTS")
        reasons.append("the measured replay contains no request")
    return codes, reasons


def _warning(summary: Mapping, path: tuple[str, ...]) -> str | None:
    value: object = summary
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _one_line(value) if isinstance(value, str) and value.strip() else None


def _measurement_state(summary: Mapping, quota_facts: dict) -> dict:
    invalid_codes, invalid_reasons = _count_integrity_issues(summary)
    answers = summary.get("answers")
    answers = answers if isinstance(answers, Mapping) else {}
    if isinstance(answers.get("invalid"), str) and answers["invalid"].strip():
        invalid_codes.append("NO_ACCEPTABLE_OUTCOME")
        invalid_reasons.append(_one_line(answers["invalid"]))

    response_identity = summary.get("response_identity")
    response_identity = (response_identity
                         if isinstance(response_identity, Mapping) else {})
    identity_invalid = response_identity.get("invalid")
    if (response_identity.get("status") == "invalid"
            or (isinstance(identity_invalid, str)
                and identity_invalid.strip())):
        invalid_codes.append("RESPONSE_MODEL_IDENTITY_INVALID")
        invalid_reasons.append(_one_line(
            identity_invalid or "response model identity was inconsistent"))

    run = summary.get("run")
    run = run if isinstance(run, Mapping) else {}
    preflight_gate = run.get("preflight_gate")
    if isinstance(preflight_gate, Mapping) \
            and preflight_gate.get("outcome") == \
            "preflight_forced_unreadable":
        invalid_codes.append("FORCED_UNREADABLE_PREFLIGHT")
        invalid_reasons.append(
            "measured load was explicitly forced after the representative "
            "preflight produced an incomplete answer")
    if run.get("endpoint_metadata_stability") == "changed":
        invalid_codes.append("ENDPOINT_METADATA_CHANGED_DURING_RUN")
        invalid_reasons.append(
            "serving endpoint metadata changed between the pre-run and "
            "post-drain snapshots")
    if run.get("aggregation_valid") is False:
        invalid_codes.append("INCOMPATIBLE_AGGREGATE")
        issues = run.get("compatibility_issues")
        if isinstance(issues, list) and issues:
            detail = "; ".join(_one_line(item, limit=120)
                               for item in issues[:2])
            invalid_reasons.append("aggregate inputs are incompatible: " + detail)
        else:
            invalid_reasons.append("aggregate inputs were not proven compatible")

    if quota_facts["count"]:
        invalid_codes.append("QUOTA_REJECTION_OBSERVED")
        invalid_reasons.append(
            f"{quota_facts['count']} captured request row(s) returned "
            "HTTP 429")
    elif quota_facts["evidence_inconsistent"]:
        invalid_codes.append("HTTP_429_EVIDENCE_INCONSISTENT")
        invalid_reasons.append("HTTP-429 evidence is internally inconsistent")
    if (quota_facts["local_guard_denied_rows"]
            or quota_facts["local_guard_denied_attempts"]
            or quota_facts["local_guard_status"] == "denied"):
        invalid_codes.append("RUNTIME_QUOTA_ADMISSION_REFUSED")
        invalid_reasons.append(
            "the runtime quota guard refused physical POST admission")
    if (quota_facts["local_guard_status"] == "invalid_evidence"
            or quota_facts["local_guard_invariant_errors"]):
        invalid_codes.append("RUNTIME_QUOTA_EVIDENCE_INVALID")
        invalid_reasons.append(
            "runtime quota guard evidence failed internal invariants")

    if invalid_codes:
        reason = "; ".join(invalid_reasons)
        return _state(
            "INVALID", "Measurement invalid", reason, invalid_codes,
            severity="fail",
            reason_details=list(zip(invalid_codes, invalid_reasons)))

    cautions: list[tuple[str, str]] = []
    quota_rows = quota_facts["request_rows_examined"]
    quota_observed = quota_facts["http_status_observed_for"]
    if quota_rows is None:
        cautions.append((
            "HTTP_STATUS_EVIDENCE_MISSING",
            "the captured HTTP-status population is not recorded"))
    elif quota_rows > 0 and (
            quota_observed is None or quota_observed < quota_rows):
        shown = "unknown" if quota_observed is None else str(quota_observed)
        cautions.append((
            "HTTP_STATUS_COVERAGE_INCOMPLETE",
            f"HTTP status was retained for only {shown}/{quota_rows} "
            "captured request rows"))
    warning_paths = (
        ("RESPONSE_MODEL_IDENTITY_UNVERIFIED",
         ("response_identity", "warning")),
        ("SLA_TARGET_PROVENANCE_WARNING", ("sla", "targets_warning")),
        ("SLA_COVERAGE_INCOMPLETE", ("sla", "coverage_warning")),
        ("CALLER_LATENCY_COVERAGE_INCOMPLETE",
         ("sla", "caller_latency_warning")),
        ("LATENCY_POPULATION_INCOMPLETE",
         ("latency_population", "warning")),
        ("TOKEN_USAGE_COVERAGE_INCOMPLETE",
         ("throughput", "coverage_warning")),
        ("COST_COVERAGE_INCOMPLETE", ("cost", "coverage_warning")),
        ("PRICING_APPLICABILITY_UNVERIFIED",
         ("cost", "applicability_warning")),
        ("CACHE_FIDELITY_UNVERIFIED", ("cache_fidelity", "warning")),
        ("TOKEN_FIDELITY_UNVERIFIED", ("token_targeting", "warning")),
        ("LOAD_DELIVERY_UNVERIFIED", ("client", "warning")),
        ("CONCURRENCY_FIDELITY_UNVERIFIED", ("concurrency", "warning")),
        ("RATE_LIMIT_EVIDENCE_INCOMPLETE", ("rate_limits", "warning")),
        ("NETWORK_PATH_CAUTION", ("network_path", "warning")),
        ("ENDPOINT_METADATA_STABILITY_UNVERIFIED",
         ("run", "endpoint_metadata_warning")),
    )
    for code, path in warning_paths:
        message = _warning(summary, path)
        if message:
            cautions.append((code, message))

    transport = run.get("transport")
    if not isinstance(transport, Mapping):
        cautions.append((
            "PRODUCTION_TRANSPORT_EVIDENCE_MISSING",
            "the benchmark transport contract was not recorded, so its "
            "connection behavior cannot be compared with production"))
    else:
        transport_warning = transport.get(
            "production_comparability_warning")
        if isinstance(transport_warning, str) and transport_warning.strip():
            cautions.append((
                "PRODUCTION_TRANSPORT_UNVERIFIED",
                _one_line(transport_warning)))
        elif transport.get("production_connection_policy_match") is not True:
            cautions.append((
                "PRODUCTION_TRANSPORT_EVIDENCE_INCONSISTENT",
                "the transport artifact did not contain an explicit exact "
                "production-policy match"))

    sample = summary.get("sample")
    sample = sample if isinstance(sample, Mapping) else {}
    indicative = sample.get("indicative_only")
    if isinstance(indicative, list) and indicative:
        cautions.append((
            "SAMPLE_SIZE_LIMITED",
            "sample size leaves " + ", ".join(
                sorted(_one_line(item, limit=20) for item in indicative))
            + " indicative only"))
    elif not sample:
        cautions.append((
            "SAMPLE_EVIDENCE_MISSING", "sample-size evidence is missing"))

    drift = summary.get("drift")
    drift = drift if isinstance(drift, Mapping) else {}
    drift_kind = drift.get("drift_kind")
    if not drift_kind:
        cautions.append((
            "STABILITY_NOT_ESTABLISHED",
            _one_line(drift.get("note") or "stability was not established")))
    elif drift_kind != "stable":
        cautions.append((
            "STABILITY_NOT_HELD", f"latency state was {drift_kind}"))

    binding = summary.get("rate_limits")
    binding = binding.get("binding") if isinstance(binding, Mapping) else None
    if isinstance(binding, Mapping) and binding.get("binding_complete") is False:
        cautions.append((
            "ENDPOINT_BINDING_UNVERIFIED",
            "configured rate limits were not bound to captured endpoint metadata"))

    if cautions:
        codes = [code for code, _message in cautions]
        reasons = [message for _code, message in cautions]
        reason = "; ".join(reasons)
        return _state(
            "CAUTION", "Use with caution", reason, codes,
            severity="warning", reason_details=cautions)

    return _state(
        "VALID", "Measurement valid",
        "No configured validity, compatibility, workload-fidelity, or "
        "coverage gate flagged this measurement.",
        ["MEASUREMENT_GATES_CLEAR"], severity="pass")


def _positive_target(value: object) -> bool:
    number = _finite_number(value)
    return number is not None and number > 0


def _sla_state(summary: Mapping) -> dict:
    sla = summary.get("sla")
    if not isinstance(sla, Mapping) or not sla:
        return _state(
            "NOT_EVALUATED", "Acceptance checks not evaluated",
            "No customer acceptance targets were supplied, so no pass or "
            "miss is claimed.",
            ["NO_SLA_TARGETS"], severity="neutral")

    checks = 0
    misses = 0
    unmeasured = 0
    confidence_not_demonstrated = 0
    for key in ("ttft_vs_target", "ttfg_vs_target"):
        rows = sla.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or row.get("target_ms") is None:
                continue
            checks += 1
            if row.get("met") is False:
                misses += 1
            elif row.get("met") is not True:
                unmeasured += 1

    acceptance = sla.get("acceptance_config")
    acceptance = acceptance if isinstance(acceptance, Mapping) else {}
    hard = acceptance.get("hard_timeouts")
    hard = hard if isinstance(hard, Mapping) else {}
    hard_configured = any(_positive_target(hard.get(key))
                          for key in ("ttft_s", "ttfg_s"))
    basis = sla.get("hard_timeout_basis")
    basis = basis if isinstance(basis, Mapping) else {}
    hard_configured = hard_configured or any(
        _positive_target(basis.get(key))
        for key in ("ttft_cap_ms", "ttfg_cap_ms"))
    if hard_configured:
        checks += 1
        breaches = _nonnegative_int(sla.get("hard_timeout_breaches"))
        hard_unmeasured = _nonnegative_int(
            sla.get("hard_timeout_unmeasured"))
        if breaches is None:
            unmeasured += 1
        elif breaches:
            misses += 1
        elif hard_unmeasured is None or hard_unmeasured:
            unmeasured += 1

    inter_configured = (
        acceptance.get("interchunk_ms") is not None
        or basis.get("interchunk_cap_ms") is not None)
    if inter_configured:
        checks += 1
        breaches = _nonnegative_int(sla.get("interchunk_breaches"))
        inter_unmeasured = _nonnegative_int(
            sla.get("interchunk_unmeasured"))
        if breaches is None:
            unmeasured += 1
        elif breaches:
            misses += 1
        elif inter_unmeasured is None or inter_unmeasured:
            unmeasured += 1

    success = sla.get("success_rate")
    success = success if isinstance(success, Mapping) else {}
    success_configured = (
        _positive_target(acceptance.get("success_rate"))
        or _positive_target(success.get("target")))
    if success_configured:
        checks += 1
        if success.get("met") is False:
            misses += 1
        elif success.get("met") is True:
            # A point estimate at or above target is not enough for a high
            # reliability claim. Current summaries explicitly say whether
            # the one-sided Wilson lower bound also clears the target. Only
            # an absent legacy field preserves the historical point-estimate
            # behavior.
            if success.get("statistically_demonstrated") is False:
                unmeasured += 1
                confidence_not_demonstrated += 1
        else:
            unmeasured += 1

    if not checks:
        return _state(
            "NOT_EVALUATED", "Acceptance checks not evaluated",
            "An acceptance block is present, but it contains no scored "
            "customer acceptance target.",
            ["NO_SLA_TARGETS"], severity="neutral")

    source_warning = _warning(summary, ("sla", "targets_warning"))
    details = {
        "checks": checks,
        "misses": misses,
        "unmeasured": unmeasured,
        "success_rate_confidence_not_demonstrated": (
            confidence_not_demonstrated),
        "targets_source": (_one_line(sla["targets_source"])
                           if isinstance(sla.get("targets_source"), str)
                           else None),
        "target_provenance_warning": source_warning,
    }
    if misses:
        miss_codes = ["SLA_TARGET_MISSED"]
        if unmeasured > confidence_not_demonstrated:
            miss_codes.append("SLA_TARGET_UNMEASURED")
        if confidence_not_demonstrated:
            miss_codes.append(
                "SUCCESS_RATE_CONFIDENCE_NOT_DEMONSTRATED")
        out = _state(
            "MISS", "Acceptance checks missed",
            f"{misses} of {checks} configured acceptance check(s) missed"
            + (f"; {unmeasured} were unmeasured" if unmeasured else "") + ".",
            miss_codes, severity="fail")
    elif unmeasured:
        ordinary_unmeasured = unmeasured - confidence_not_demonstrated
        if confidence_not_demonstrated and not ordinary_unmeasured:
            reason = (
                "The observed success rate met its target, but its one-sided "
                "95% Wilson lower confidence bound did not; no acceptance "
                "pass is claimed.")
        else:
            reason = (
                f"{unmeasured} of {checks} configured acceptance check(s) "
                "were "
                "inconclusive"
                + (f", including {confidence_not_demonstrated} success-rate "
                   "confidence check(s)" if confidence_not_demonstrated else "")
                + "; no acceptance pass is claimed.")
        inconclusive_codes = []
        if ordinary_unmeasured:
            inconclusive_codes.append("SLA_TARGET_UNMEASURED")
        if confidence_not_demonstrated:
            inconclusive_codes.append(
                "SUCCESS_RATE_CONFIDENCE_NOT_DEMONSTRATED")
        out = _state(
            "INCONCLUSIVE", "Acceptance checks inconclusive",
            reason, inconclusive_codes, severity="warning")
    else:
        qualifier = (
            " The target source is marked with a provenance warning; read it "
            "before treating these as customer-approved targets."
            if source_warning else "")
        out = _state(
            "PASS", "Configured acceptance checks passed",
            f"All {checks} configured acceptance check(s) passed.{qualifier}",
            ["SLA_TARGETS_MET"] + (
                ["SLA_TARGET_PROVENANCE_WARNING"] if source_warning else []),
            severity="pass" if not source_warning else "warning")
    out["evaluation"] = details
    return out


def _tested_load(summary: Mapping, quota_facts: dict) -> dict:
    schedule = summary.get("schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    arrivals = summary.get("arrivals")
    arrivals = arrivals if isinstance(arrivals, Mapping) else {}
    answers = summary.get("answers")
    answers = answers if isinstance(answers, Mapping) else {}
    return {
        "measured_replay_requests": _nonnegative_int(
            summary.get("requests_total")),
        "measured_replay_ok": _nonnegative_int(summary.get("requests_ok")),
        "measured_replay_failed": _nonnegative_int(
            summary.get("requests_failed")),
        "acceptable_outcomes": _nonnegative_int(
            answers.get("acceptable_outcomes")),
        "answer_rows_judged": _nonnegative_int(answers.get("judged")),
        "achieved_qps_overall": _finite_number(
            arrivals.get("achieved_qps_overall"), nonnegative=True),
        "scheduled_requests": _nonnegative_int(schedule.get("requests")),
        "scheduled_seconds": _nonnegative_int(schedule.get("seconds")),
        "schedule_source": (_one_line(schedule["source"], limit=160)
                            if isinstance(schedule.get("source"), str)
                            else None),
        "captured_quota_request_rows": quota_facts[
            "request_rows_examined"],
        "claim_boundary": (
            "Observed tested-load facts only; they do not establish an "
            "endpoint ceiling or provider quota headroom."),
    }


def _capacity_state(summary: Mapping, integrity: dict, measurement: dict,
                    sla: dict, quota: dict, tested_load: dict) -> dict:
    total = tested_load["measured_replay_requests"]
    if total == 0:
        out = _state(
            "NOT_EVALUATED", "Capacity not evaluated",
            "The measured replay contains no request, so there is no "
            "tested-load capacity observation.",
            ["NO_MEASURED_REQUESTS"], severity="neutral")
    elif quota["code"] in {"EXCEEDED", "LOCAL_GUARD_REFUSED"}:
        evidence = quota["http_429"]
        count = evidence["http_429_count"]
        rows = evidence["request_rows_examined"]
        denominator = rows if rows is not None and rows >= count else "unknown"
        if quota["code"] == "LOCAL_GUARD_REFUSED":
            reason = (
                "The runtime quota guard refused new physical POSTs before "
                "send, so the requested load was not delivered. That is a "
                "local quota-safety stop, not an endpoint-capacity ceiling.")
            reason_codes = ["QUOTA_LIMITED_CAPACITY_INCONCLUSIVE"]
        else:
            reason = (
                f"HTTP 429 occurred in {count}/{denominator} captured request "
                "rows. That is quota-limited evidence, not an endpoint-capacity "
                "ceiling.")
            reason_codes = ["QUOTA_LIMITED_CAPACITY_INCONCLUSIVE"]
        out = _state(
            "INCONCLUSIVE", "Endpoint capacity inconclusive",
            reason, reason_codes, severity="warning")
    else:
        run = summary.get("run")
        run = run if isinstance(run, Mapping) else {}
        binding = run.get("invocation_binding")
        if not isinstance(binding, Mapping):
            # Compatibility fallback for older quota-aware artifacts, whose
            # only route/control-plane binding lived under rate_limits.
            rate_limits = summary.get("rate_limits")
            binding = (rate_limits.get("binding")
                       if isinstance(rate_limits, Mapping) else None)
        binding_complete = bool(
            isinstance(binding, Mapping)
            and binding.get("binding_complete") is True)
        blockers: list[tuple[str, str]] = []
        if integrity["code"] != "VERIFIED":
            blockers.append((
                "EVIDENCE_NOT_VERIFIED",
                "the sealed artifact has not passed explicit integrity verification"))
        if measurement["code"] != "VALID":
            blockers.append((
                "MEASUREMENT_NOT_VALID",
                f"measurement state is {measurement['code'].lower()}"))
        if quota["code"] not in {"NOT_OBSERVED"}:
            blockers.append((
                "QUOTA_STATE_NOT_CLEAR",
                f"quota state is {quota['code'].lower()}"))
        if not binding_complete:
            blockers.append((
                "ENDPOINT_BINDING_UNVERIFIED",
                "endpoint identity and deployment-mode binding is not verified"))

        if blockers:
            reason = "; ".join(message for _code, message in blockers)
            reason += ". Tested-load facts remain observations, not a ceiling."
            out = _state(
                "INCONCLUSIVE", "Endpoint capacity inconclusive", reason,
                [code for code, _message in blockers], severity="warning",
                reason_details=blockers)
        else:
            failed = tested_load["measured_replay_failed"]
            judged = tested_load["answer_rows_judged"]
            answered = tested_load["acceptable_outcomes"]
            did_not_hold = bool(
                failed is not None and failed > 0
                or judged is not None and answered is not None
                and answered < judged)
            if did_not_hold:
                out = _state(
                    "NOT_HELD_AT_TESTED_LOAD", "Tested load not held",
                    "The verified, bound run recorded request failures or "
                    "unacceptable outcomes at the tested load. This locates "
                    "a failed test point, not the endpoint ceiling.",
                    ["TESTED_LOAD_FAILURE_OBSERVED"], severity="fail")
            else:
                sla_note = (
                    " The configured acceptance checks still missed and must "
                    "be read separately."
                    if sla["code"] == "MISS" else "")
                out = _state(
                    "HELD_AT_TESTED_LOAD", "Tested load held",
                    f"The verified, bound run completed its {total} measured "
                    "replay request(s) without a recorded failure or "
                    "unacceptable outcome. This is a tested point, not an "
                    f"endpoint ceiling or quota-headroom claim.{sla_note}",
                    ["TESTED_LOAD_HELD"], severity="pass")
    out["endpoint_ceiling_established"] = False
    out["provider_headroom_established"] = False
    return out


def build_report_decision(
        summary: Mapping,
        integrity: IntegrityContext | Mapping | None = None) -> dict:
    """Return the canonical five-state decision model for a run summary.

    The function is pure: it performs no I/O, does not mutate ``summary``,
    emits no timestamp, and returns only values accepted by ``json.dumps``.
    """
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    context = _integrity_context(integrity)
    evidence = _evidence_integrity(context)
    quota_facts = _quota_facts(summary)
    quota = _quota_state(quota_facts)
    measurement = _measurement_state(summary, quota_facts)
    sla = _sla_state(summary)
    # Preserve the independently observed check outcome, but never paint a
    # clean green SLA pass on an invalid or qualified measurement.  This is
    # especially important in a cropped/mobile screenshot where the reason
    # text may not be visible beside the state label.
    if sla["code"] == "PASS" and measurement["code"] != "VALID":
        sla = dict(sla)
        if measurement["code"] == "INVALID":
            sla["label"] = "Acceptance checks passed - invalid run"
        else:
            sla["label"] = "Acceptance checks passed - qualified"
        sla["severity"] = "warning"
        sla["reason"] = _one_line(
            f"{sla['reason']} The check outcome is retained, but the "
            f"measurement state is {measurement['code'].lower()}, so this "
            "is not a clean acceptance pass.")
        sla["reason_codes"] = list(dict.fromkeys([
            *sla["reason_codes"], "MEASUREMENT_BLOCKS_CLEAN_SLA_PASS",
        ]))
        sla["reason_details"] = [
            *list(sla.get("reason_details") or []),
            {
                "code": "MEASUREMENT_BLOCKS_CLEAN_SLA_PASS",
                "message": (
                    "The independently observed acceptance outcome is retained, "
                    f"but the measurement state is {measurement['code'].lower()}, "
                    "so this is not a clean acceptance pass."
                ),
            },
        ]
    tested_load = _tested_load(summary, quota_facts)
    capacity = _capacity_state(
        summary, evidence, measurement, sla, quota, tested_load)
    return {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "evidence_integrity": evidence,
        "measurement_validity": measurement,
        "customer_sla": sla,
        "quota_state": quota,
        "endpoint_capacity": capacity,
        "tested_load": tested_load,
    }


__all__ = [
    "DECISION_SCHEMA_VERSION",
    "IntegrityContext",
    "build_report_decision",
]
