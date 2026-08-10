"""Integrity chain for rate-sweep aggregate evidence.

A sweep is not a loose Markdown file next to several runs. It is a sealed
aggregate whose manifest binds the rendered conclusion, the exact
base configuration, and the already-sealed manifest and summary identity of
every completed rung used to reach that conclusion.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import hmac
from html import escape as html_escape
import math
import os
from pathlib import Path
import stat
import time
from urllib.parse import quote
import uuid

from . import __version__
from .aggregate import (
    _artifact_declarations,
    _fsync_directory,
    _fsync_fd,
    _has_path,
    _identity_digest,
    _read_regular_bytes,
    _require_regular,
    _require_run_dir,
    _scan_request_journal,
    _stable,
    _verify_artifacts,
    _write_compare_fd,
)
from .artifacts import (
    canonical_sha256,
    redact_secrets,
    sanitize_title,
    snapshot_source_state,
    strict_json_dumps,
)
from .schedule import MAX_EXACT_ANALYSIS_REQUEST_ROWS


_WRITING_MARKER = ".traffic-replay-writing"
_COMPLETE_MARKER = ".traffic-replay-complete"
_SWEEP_DECISION_SCHEMA_VERSION = 6
_SWEEP_RENDERER_SCHEMA_VERSION = 6


def _decision_percent_display(*values: object) -> tuple[str, ...]:
    """Render decision percentages without hiding boundary differences."""
    valid = all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value)) for value in values)
    if not valid:
        return tuple(
            "-" if value is None else str(value) for value in values)
    numbers = [float(value) * 100.0 for value in values]
    for decimals in range(1, 11):
        rendered = [f"{number:.{decimals}f}%" for number in numbers]
        collision = any(
            numbers[left] != numbers[right]
            and rendered[left] == rendered[right]
            for left in range(len(numbers))
            for right in range(left + 1, len(numbers)))
        if not collision:
            return tuple(rendered)
    return tuple(f"{number:.15g}%" for number in numbers)


def rate_label(value: int | float) -> str:
    """Injective, filesystem-safe rendering of one finite positive float."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid sweep rate: {value!r}")
    text = repr(number)
    return text[:-2] if text.endswith(".0") else text


def _strict_object(raw: bytes, label: str, path: Path) -> dict:
    from .json_input import loads_strict

    try:
        value = loads_strict(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def sweep_acceptance_policy(acceptance: object) -> tuple[bool, str]:
    """Require an explicit, customer-owned latency and reliability policy."""
    if not isinstance(acceptance, dict):
        return False, "no acceptance policy was configured"
    targets_are = acceptance.get("targets_are")
    if not isinstance(targets_are, str) or not targets_are.strip():
        return False, (
            "acceptance policy does not record who owns the targets")
    provenance = " ".join((
        targets_are,
        str(acceptance.get("note") or ""),
    )).lower()
    disallowed = {
        "illustrative", "example", "placeholder", "sample", "demo",
        "default", "test fixture", "replace before", "not customer",
    }
    if any(marker in provenance for marker in disallowed):
        return False, (
            "the available acceptance policy is illustrative, placeholder, "
            "or explicitly not customer-owned")
    ownership_markers = {
        "customer", "client", "yours", "agreed", "production slo",
        "production sla", "production requirement",
    }
    if not any(marker in targets_are.lower()
               for marker in ownership_markers):
        return False, (
            "targets_are does not positively identify customer ownership")
    success = acceptance.get("success_rate")
    has_reliability = (
        isinstance(success, (int, float)) and not isinstance(success, bool)
        and math.isfinite(float(success)) and 0 < float(success) < 1)
    hard = acceptance.get("hard_timeouts")
    hard = hard if isinstance(hard, dict) else {}
    has_latency = bool(
        acceptance.get("ttft_ms") or acceptance.get("ttfg_ms")
        or any(hard.get(key) is not None for key in ("ttft_s", "ttfg_s"))
        or acceptance.get("interchunk_ms") is not None)
    missing = []
    if not has_latency:
        missing.append("an explicit latency criterion")
    if not has_reliability:
        missing.append("an explicit success_rate criterion")
    if missing:
        return False, "missing " + " and ".join(missing)
    return True, "explicit customer latency and reliability policy"


def _summary_acceptance(summary: dict) -> dict | None:
    sla = summary.get("sla") or {}
    configured = sla.get("acceptance_config")
    if isinstance(configured, dict):
        return configured
    # Older summaries did not persist acceptance_config. Reconstruct only the
    # existence of policy dimensions, never invented threshold values.
    reconstructed: dict = {}
    if any((sla.get(name) or []) for name in (
            "ttft_vs_target", "ttfg_vs_target")):
        for name, key in (("ttft_vs_target", "ttft_ms"),
                          ("ttfg_vs_target", "ttfg_ms")):
            rows = sla.get(name) or []
            if rows:
                reconstructed[key] = {
                    str(row.get("quantile")): row.get("target_ms")
                    for row in rows if row.get("target_ms") is not None}
    success = sla.get("success_rate") or {}
    if success.get("target") is not None:
        reconstructed["success_rate"] = success["target"]
    if sla.get("targets_warning"):
        reconstructed["note"] = "illustrative"
    return reconstructed or None


def _quota_axis(summary: dict) -> str:
    local = summary.get("runtime_quota_admission")
    if isinstance(local, dict):
        if local.get("status") == "denied" \
                or (isinstance(local.get("denied_rows"), int)
                    and not isinstance(local.get("denied_rows"), bool)
                    and local["denied_rows"] > 0):
            return "LIMITED"
        if local.get("status") == "invalid_evidence":
            return "UNKNOWN"
    count = summary.get("http_429_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return "LIMITED"
    rate_limits = summary.get("rate_limits")
    if not isinstance(rate_limits, dict):
        return "UNKNOWN"
    comparisons = rate_limits.get("comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        return "UNKNOWN"
    statuses = {
        str(item.get("status")) for item in comparisons.values()
        if isinstance(item, dict)
    }
    if any(status in {
            "run_evidence_at_or_above_nominal_limit",
            "run_evidence_warning_threshold_reached",
            "short_observation_projection_at_or_above_warning",
            } for status in statuses):
        return "NEAR_LIMIT"
    if any(status.startswith("short_observation") for status in statuses):
        return "HEADROOM_UNESTABLISHED_SHORT_WINDOW"
    if any(status in {"unmeasured", "incomplete_run_evidence"}
           for status in statuses):
        return "UNKNOWN"
    return "NO_429_OBSERVED"


def _selected_latency_projection(summary: dict) -> dict:
    sla = summary.get("sla") or {}
    definition = (sla.get("ttft_definition")
                  or (summary.get("run") or {}).get("ttft_definition")
                  or "first_content")
    raw = "ttfv_ms" if definition == "first_visible" else "ttft_ms"
    corrected = ("ttfv_corrected_ms" if definition == "first_visible"
                 else "ttft_corrected_ms")
    configured = sla.get("ttft_metric")
    metric = configured if isinstance(configured, str) else (
        corrected if (summary.get(corrected) or {}).get("n") else raw)
    table = summary.get(metric) or {}
    e2e_metric = sla.get("ttfg_metric")
    if not isinstance(e2e_metric, str):
        e2e_metric = ("e2e_corrected_ms"
                      if (summary.get("e2e_corrected_ms") or {}).get("n")
                      else "e2e_ms")
    e2e = summary.get(e2e_metric) or {}
    success = sla.get("success_rate")
    success = success if isinstance(success, dict) else {}
    arrivals = summary.get("arrivals")
    arrivals = arrivals if isinstance(arrivals, dict) else {}
    request_start = arrivals.get("http_request_start_lateness_ms")
    if not isinstance(request_start, dict):
        request_start = arrivals.get("wire_lateness_ms")
    request_start = request_start if isinstance(request_start, dict) else {}
    dispatch_lag = arrivals.get("dispatch_lag_ms")
    dispatch_lag = dispatch_lag if isinstance(dispatch_lag, dict) else {}
    response_identity = summary.get("response_identity")
    response_identity = (response_identity
                         if isinstance(response_identity, dict) else {})
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    runtime_quota = summary.get("runtime_quota_admission")
    runtime_quota = runtime_quota if isinstance(runtime_quota, dict) else {}
    transport = run.get("transport")
    transport = transport if isinstance(transport, dict) else {}
    actual_policy = transport.get("connection_policy_id")
    actual_policy = (actual_policy.strip()
                     if isinstance(actual_policy, str)
                     and actual_policy.strip() else None)
    declared_policy = transport.get(
        "production_connection_policy_declared")
    declared_policy = (declared_policy.strip()
                       if isinstance(declared_policy, str)
                       and declared_policy.strip() else None)
    match_value = transport.get("production_connection_policy_match")
    policy_match = match_value if isinstance(match_value, bool) else None
    warning_value = transport.get("production_comparability_warning")
    transport_warning = (warning_value.strip()
                         if isinstance(warning_value, str)
                         and warning_value.strip() else None)
    assurance_value = transport.get(
        "production_connection_policy_assurance")
    transport_assurance = (assurance_value.strip()
                           if isinstance(assurance_value, str)
                           and assurance_value.strip() else None)
    transport_exact = bool(
        actual_policy is not None
        and declared_policy == actual_policy
        and policy_match is True
        and warning_value is None)
    if transport_exact:
        transport_status = "MATCH"
    elif policy_match is True or (
            warning_value is not None
            and not isinstance(warning_value, str)):
        transport_status = "INCONSISTENT"
    else:
        transport_status = "UNVERIFIED"
    return {
        "first_event_definition": definition,
        "latency_metric": metric,
        "latency_basis": ("caller_experienced" if metric.endswith(
            "_corrected_ms") else "final_attempt_request_path"),
        "latency_n": table.get("n"),
        "latency_p50": table.get("p50"),
        "latency_p95": table.get("p95"),
        "e2e_metric": e2e_metric,
        "e2e_basis": ("caller_experienced" if e2e_metric.endswith(
            "_corrected_ms") else "final_attempt_request_path"),
        "e2e_n": e2e.get("n"),
        "e2e_p50": e2e.get("p50"),
        "e2e_p95": e2e.get("p95"),
        "success_rate_target": success.get("target"),
        "success_rate_actual": success.get("actual"),
        "success_rate_wilson_lower_95": success.get(
            "one_sided_95pct_wilson_lower"),
        "success_rate_statistically_demonstrated": success.get(
            "statistically_demonstrated"),
        "request_start_lateness_p95": request_start.get("p95"),
        "dispatch_lag_p95": dispatch_lag.get("p95"),
        "response_identity_status": response_identity.get("status"),
        "endpoint_metadata_stability": run.get(
            "endpoint_metadata_stability"),
        "runtime_quota_admission_status": runtime_quota.get("status"),
        "runtime_quota_guard_id": runtime_quota.get("guard_id"),
        "transport_connection_policy_id": actual_policy,
        "production_connection_policy_declared": declared_policy,
        "production_connection_policy_match": policy_match,
        "production_comparability_warning": transport_warning,
        "production_connection_policy_assurance": transport_assurance,
        "transport_parity_status": transport_status,
    }


def _transport_parity_exact(rung: dict) -> bool:
    """Require internally consistent, explicit production transport parity."""
    actual = rung.get("transport_connection_policy_id")
    declared = rung.get("production_connection_policy_declared")
    warning = rung.get("production_comparability_warning")
    return bool(
        isinstance(actual, str) and actual.strip()
        and isinstance(declared, str) and declared == actual
        and rung.get("production_connection_policy_match") is True
        and warning is None
        and rung.get("transport_parity_status") == "MATCH")


def _transport_parity_reason(rung: dict) -> str:
    warning = rung.get("production_comparability_warning")
    if isinstance(warning, str) and warning.strip():
        return warning.strip()
    actual = rung.get("transport_connection_policy_id")
    declared = rung.get("production_connection_policy_declared")
    match = rung.get("production_connection_policy_match")
    if not isinstance(actual, str) or not actual.strip():
        return "the benchmark connection policy was not recorded"
    if not isinstance(declared, str) or not declared.strip():
        return (
            "production connection behavior was not declared, so it cannot "
            f"be compared with benchmark policy {actual}")
    if declared != actual or match is not True:
        return (
            f"declared production policy {declared} does not have an exact "
            f"recorded match to benchmark policy {actual}")
    return "the transport parity fields are internally inconsistent"


def classify_sweep_rung(summary: dict) -> dict:
    """Project one run onto independent SLA, quota, and metric axes."""
    from .metrics import _verdict

    kind, text = _verdict(summary)
    quota_status = _quota_axis(summary)
    policy_ok, policy_reason = sweep_acceptance_policy(
        _summary_acceptance(summary))
    projection = _selected_latency_projection(summary)
    if quota_status == "LIMITED":
        state = "QUOTA_LIMITED"
    elif not policy_ok:
        state = "NO_CRITERION"
        kind = "caution"
        text = (f"no publishable capacity criterion: {policy_reason}. run "
                "only as an explicitly diagnostic sweep")
    elif kind == "invalid":
        state = "INVALID"
    elif kind == "miss":
        state = "FAIL"
    elif kind == "ok":
        state = "PASS"
    else:
        sla = summary.get("sla") or {}
        answers = summary.get("answers") or {}
        definitive_invalid = bool(
            (summary.get("client") or {}).get("warning")
            or (summary.get("concurrency") or {}).get("warning")
            or (summary.get("throughput") or {}).get("coverage_warning")
            or (summary.get("cache_fidelity") or {}).get("warning")
            or (summary.get("token_targeting") or {}).get("warning")
            or (summary.get("latency_population") or {}).get("warning")
            or (summary.get("network_path") or {}).get("warning")
            or sla.get("coverage_warning")
            or sla.get("caller_latency_warning")
            or (sla.get("hard_timeout_unmeasured") or 0) > 0
            or (sla.get("interchunk_unmeasured") or 0) > 0
            or ((answers.get("scored") or 0) > 0
                and (answers.get("truncated_by_global_cap") or 0)
                / answers["scored"] > 0.05))
        drift_kind = (summary.get("drift") or {}).get("drift_kind")
        if definitive_invalid:
            state = "INVALID"
        elif drift_kind and drift_kind != "stable":
            state = "FAIL"
        else:
            # Sample size, Wilson confidence, and an unestablished stability
            # window can improve at a higher-rate rung or longer observation.
            # They must not be treated as a definitive lower-rung failure.
            state = "INSUFFICIENT_EVIDENCE"
    if state not in {"INVALID", "QUOTA_LIMITED", "FAIL", "NO_CRITERION"} \
            and not _transport_parity_exact(projection):
        transport_reason = _transport_parity_reason(projection)
        state = "INSUFFICIENT_EVIDENCE"
        kind = "caution"
        if transport_reason not in text:
            text = (
                f"{text.rstrip('.')} Production transport parity is not "
                f"established: {transport_reason}.")
    return {
        "state": state,
        "kind": kind,
        "text": text,
        "quota_status": quota_status,
        **projection,
    }


def _rung_state(rung: dict) -> str:
    state = rung.get("state")
    resolved = str(state) if state is not None else {
        "ok": "PASS", "caution": "INSUFFICIENT_EVIDENCE",
        "miss": "FAIL", "invalid": "INVALID",
    }[rung["kind"]]
    # Legacy rung records predate the canonical decision object. Fail closed:
    # the absence of an exact transport match is not evidence of parity and
    # can never support a green held-rate/capacity conclusion.
    if resolved == "PASS" and not _transport_parity_exact(rung):
        return "INSUFFICIENT_EVIDENCE"
    return resolved


def _sweep_quota_evidence(source_paths: list[Path],
                          source_summaries: list[dict],
                          base_config: dict) -> dict:
    """Pool every manifest-bound request phase across the full ladder."""
    from .aggregate import _request_rows
    from .metrics import _rate_limit_evidence

    rows = [
        row
        for path in source_paths
        for row in _request_rows(path, _require_run_dir(path, "requests.jsonl"))
    ]
    endpoint_metadata = None
    metadata_values = []
    for summary in source_summaries:
        metadata = (summary.get("run") or {}).get("endpoint_metadata")
        if metadata is not None:
            metadata_values.append(metadata)
    if metadata_values and len({_stable(value) for value in metadata_values}) == 1:
        endpoint_metadata = metadata_values[0]
    limits = base_config.get("rate_limits")
    observed, configured = _rate_limit_evidence(
        rows, limits, {"endpoint_metadata": endpoint_metadata})
    count_429 = sum(
        row.get("status") == 429 and _sent_at_for_sweep(row) is not None
        for row in rows)
    from .metrics import _runtime_quota_admission_block
    final_guard_snapshot = None
    for summary in source_summaries:
        candidate = (summary.get("run") or {}).get("runtime_quota_guard")
        if isinstance(candidate, dict):
            final_guard_snapshot = candidate
    # A sweep's individual rung snapshots are command-cumulative while each
    # rung journal is local.  Pool every manifest-bound row and reconcile the
    # union exactly against the last cumulative snapshot.  Omitting the
    # baseline intentionally selects full-history validation in the helper.
    pooled_local = _runtime_quota_admission_block(
        rows, {"runtime_quota_guard": final_guard_snapshot}
        if final_guard_snapshot is not None else {})
    guard_ids = list(pooled_local.get("observed_guard_ids") or [])
    local_denied_rows = int(pooled_local.get("denied_rows") or 0)
    local_denied_attempts = int(
        pooled_local.get("denied_attempts_in_captured_rows") or 0)
    local_invariants = list(pooled_local.get("invariant_errors") or [])
    if limits is not None and final_guard_snapshot is None:
        local_invariants.append(
            "a quota-aware sweep has no final command-level runtime guard "
            "snapshot")
    if limits is not None and len(guard_ids) != 1:
        local_invariants.append(
            "a quota-aware sweep must use one command-level runtime guard "
            "across preflight and every rung")
    local_status = (
        "invalid_evidence" if local_invariants else
        str(pooled_local.get("status") or "not_configured"))
    local = {
        "status": local_status,
        "guard_ids": guard_ids,
        "denied_rows": local_denied_rows,
        "denied_attempts_in_captured_rows": local_denied_attempts,
        "invariant_errors": local_invariants,
    }
    quota_summary = {
        "http_429_count": count_429,
        "runtime_quota_admission": local,
    }
    if configured is not None:
        quota_summary["rate_limits"] = configured
    return {
        "traffic_population": "all_manifest_bound_request_phases_once",
        "request_rows": len(rows),
        "http_429_count": count_429,
        "quota_status": _quota_axis(quota_summary),
        "runtime_quota_admission": local,
        "observed_rate_windows": observed,
        "configured_rate_limits": configured,
    }


def _sent_at_for_sweep(row: dict) -> float | None:
    value = (row.get("first_send_unix") if "first_send_unix" in row
             else row.get("t_send_unix"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def sweep_outcome(rungs: list[dict], preflight: dict | None = None) -> dict:
    """Derive the only valid aggregate result from ordered rung evidence."""
    if not isinstance(rungs, list) or not rungs:
        raise ValueError("a sweep requires at least one rung attempt")
    rates = []
    for position, rung in enumerate(rungs):
        if not isinstance(rung, dict):
            raise ValueError(f"invalid sweep rung record at position {position}")
        rate = rung.get("rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                or not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError(f"invalid sweep rung rate at position {position}")
        rates.append(float(rate))
        if rung.get("kind") not in {"ok", "caution", "miss", "invalid"}:
            raise ValueError(f"invalid sweep verdict at position {position}")
        if rung.get("state") is not None and rung.get("state") not in {
                "PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "NO_CRITERION",
                "QUOTA_LIMITED", "INVALID"}:
            raise ValueError(f"invalid sweep state at position {position}")
    if any(right <= left for left, right in zip(rates, rates[1:])):
        raise ValueError("sweep rung rates must be strictly increasing")

    unverified = [r for r in rungs if r.get("source_position") is None]
    seen_definitive_fail = False
    non_monotonic = False
    for rung in rungs:
        state = _rung_state(rung)
        if state == "FAIL":
            seen_definitive_fail = True
        elif state == "PASS" and seen_definitive_fail:
            non_monotonic = True
    good = [r for r in rungs if _rung_state(r) == "PASS"]
    insufficient = [r for r in rungs
                    if _rung_state(r) == "INSUFFICIENT_EVIDENCE"]
    no_criterion = [r for r in rungs if _rung_state(r) == "NO_CRITERION"]
    quota_limited = [r for r in rungs
                     if _rung_state(r) == "QUOTA_LIMITED"]
    invalid_reasons = []
    if preflight is not None:
        if not isinstance(preflight, dict):
            raise ValueError("invalid sweep preflight outcome evidence")
        preflight_outcome = preflight.get("outcome")
        if preflight_outcome not in {
                "skipped", "preflight_passed", "preflight_refused",
                "preflight_forced_unreadable", "preflight_forced_failed",
                "preflight_state_unknown"}:
            raise ValueError("invalid sweep preflight outcome")
        if preflight_outcome == "preflight_forced_unreadable":
            invalid_reasons.append(
                "measured load was explicitly forced after an unreadable "
                "representative preflight")
        elif preflight_outcome in {
                "preflight_forced_failed", "preflight_refused",
                "preflight_state_unknown"}:
            invalid_reasons.append(
                "the representative preflight did not establish a valid "
                f"load gate ({preflight_outcome})")
    if unverified:
        invalid_reasons.append("one or more rung attempts produced no verified report")
    invalid_reports = [
        r for r in rungs
        if r.get("source_position") is not None
        and _rung_state(r) == "INVALID"]
    if invalid_reports:
        invalid_reasons.append(
            "one or more manifest-bound rung reports are invalid measurements")
    # PASS after a lower definitive FAIL prevents a monotonic boundary claim,
    # but it does not make sealed source evidence corrupt. Keep this as an
    # independent experiment-shape outcome so the verifier can distinguish a
    # valid inconclusive sweep from invalid artifacts or measurements.
    calibration_rows = sum(
        int(r.get("calibration_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    sizing_rows = sum(
        int(r.get("sizing_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    other_rows = sum(
        int(r.get("other_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    unknown_attempt_rows = sum(
        int(r.get("unknown_attempt_rows") or 0)
        for r in rungs if r.get("source_position") is not None)
    if calibration_rows:
        invalid_reasons.append(
            f"{calibration_rows} per-rung calibration request"
            f"{'s were' if calibration_rows != 1 else ' was'} mixed into "
            "the ladder")
    if sizing_rows:
        invalid_reasons.append(
            f"{sizing_rows} concurrency-sizing request"
            f"{'s were' if sizing_rows != 1 else ' was'} mixed into the ladder")
    if other_rows:
        invalid_reasons.append(
            f"{other_rows} request row{'s have' if other_rows != 1 else ' has'} "
            "an unrecognized traffic phase")
    if unknown_attempt_rows:
        invalid_reasons.append(
            f"{unknown_attempt_rows} request row"
            f"{'s have' if unknown_attempt_rows != 1 else ' has'} unknown "
            "provider-attempt timing or count")
    invalid = bool(invalid_reasons)
    highest_sla_passing = (None if invalid or non_monotonic or not good
                           else good[-1]["rate"])
    def achieved_rate(rung):
        value = rung.get("achieved_rps")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            return None
        return float(value)

    achieved_at_highest_requested = (
        None if highest_sla_passing is None else achieved_rate(good[-1]))
    passing_achieved = [
        (achieved_rate(rung), float(rung["rate"]), rung)
        for rung in good if achieved_rate(rung) is not None
    ]
    highest_achieved, requested_at_highest_achieved, _ = (
        max(passing_achieved, key=lambda item: (item[0], item[1]))
        if highest_sla_passing is not None and passing_achieved
        else (None, None, None))
    if invalid:
        capacity_conclusion = "INVALID_EVIDENCE"
    elif quota_limited:
        capacity_conclusion = "QUOTA_LIMITED_NO_ENDPOINT_CEILING"
    elif no_criterion:
        capacity_conclusion = "NO_CRITERION_DIAGNOSTIC_ONLY"
    elif non_monotonic:
        capacity_conclusion = "NON_MONOTONIC_NO_BOUNDARY"
    elif not good:
        capacity_conclusion = (
            "INSUFFICIENT_EVIDENCE" if insufficient else
            "LOWEST_TESTED_RATE_FAILED")
    elif _rung_state(rungs[-1]) == "PASS":
        capacity_conclusion = "TOP_OF_LADDER_PASSED"
    elif any(_rung_state(rung) == "FAIL" for rung in rungs):
        capacity_conclusion = "SLA_BOUNDARY_OBSERVED"
    else:
        capacity_conclusion = "PASSING_RATE_WITH_INSUFFICIENT_HIGHER_EVIDENCE"
    return {
        "invalid": invalid,
        "invalid_reasons": invalid_reasons,
        "unverified": unverified,
        "invalid_reports": invalid_reports,
        "non_monotonic": non_monotonic,
        "boundary_status": ("NON_MONOTONIC" if non_monotonic else
                            "MONOTONIC"),
        "calibration_rows": calibration_rows,
        "sizing_rows": sizing_rows,
        "other_rows": other_rows,
        "unknown_attempt_rows": unknown_attempt_rows,
        "good": good,
        "insufficient": insufficient,
        "no_criterion": no_criterion,
        "quota_limited": quota_limited,
        "highest_sla_passing_tested_rate": highest_sla_passing,
        "achieved_rate_at_highest_requested_sla_passing_rung":
            achieved_at_highest_requested,
        "highest_achieved_rate_at_sla_passing_rung":
            highest_achieved,
        "requested_rate_at_highest_achieved_sla_passing_rung":
            requested_at_highest_achieved,
        # Backward-compatible field name with corrected schema-v5 semantics:
        # a held/delivered claim is the achieved average, never the requested
        # open-loop rung label.
        "highest_held_rate": highest_achieved,
        "capacity_conclusion": capacity_conclusion,
        "exit_code": 2 if invalid else 1 if quota_limited or non_monotonic \
            else 0 if good or no_criterion else 1,
    }


def _validated_report_context(context: object, d: Path | None = None,
                              *, expected_endpoint: str | None = None,
                              rung_count: int | None = None) -> dict:
    where = f" in {d}" if d is not None else ""
    if not isinstance(context, dict):
        raise ValueError(f"invalid sweep report context{where}")
    legacy_context_fields = {
        "endpoint", "sweep_wall_s", "cooldown_s", "cooldown_events",
        "preflight",
    }
    v2_context_fields = legacy_context_fields | {
        "cooldown_records", "planned_rates", "attempted_rates",
        "omitted_rates", "progression_policy", "termination_reason",
        "sweep_quota_evidence",
    }
    if set(context) not in (legacy_context_fields, v2_context_fields):
        raise ValueError(f"unknown or missing sweep report context field{where}")
    is_v2 = set(context) == v2_context_fields
    endpoint = context.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip() \
            or endpoint != sanitize_title(endpoint):
        raise ValueError(f"invalid sweep report endpoint{where}")
    if expected_endpoint is not None and endpoint != expected_endpoint:
        raise ValueError(f"sweep report endpoint disagrees with base config{where}")
    wall = context.get("sweep_wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) < 0:
        raise ValueError(f"invalid sweep report wall time{where}")
    cooldown = context.get("cooldown_s")
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) \
            or not math.isfinite(float(cooldown)) or float(cooldown) < 0:
        raise ValueError(f"invalid sweep cooldown{where}")
    events = context.get("cooldown_events")
    if isinstance(events, bool) or not isinstance(events, int) or events < 0:
        raise ValueError(f"invalid sweep cooldown event count{where}")
    preflight = context.get("preflight")
    if not isinstance(preflight, dict) \
            or not isinstance(preflight.get("skipped"), bool):
        raise ValueError(f"invalid sweep preflight evidence{where}")
    legacy_preflight_fields = {
        "skipped", "attempted", "reachable", "readable",
        "reasoning_probe_requests",
    }
    current_preflight_fields = legacy_preflight_fields | {
        "outcome", "force_requested", "gate_satisfied",
    }
    preflight_fields = frozenset(preflight)
    if preflight_fields not in {
            frozenset(legacy_preflight_fields),
            frozenset(current_preflight_fields)}:
        raise ValueError(f"unknown or missing sweep preflight field{where}")
    counts = {}
    for field in ("attempted", "reachable", "readable",
                  "reasoning_probe_requests"):
        value = preflight.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid sweep preflight {field}{where}")
        counts[field] = value
    if counts["reachable"] > counts["attempted"] \
            or counts["readable"] > counts["reachable"]:
        raise ValueError(f"sweep preflight counts disagree{where}")
    if preflight["skipped"] and any(counts.values()):
        raise ValueError(f"skipped sweep preflight claims traffic{where}")
    if preflight_fields == frozenset(current_preflight_fields):
        force_requested = preflight.get("force_requested")
        gate_satisfied = preflight.get("gate_satisfied")
        outcome = preflight.get("outcome")
        if not isinstance(force_requested, bool) \
                or not isinstance(gate_satisfied, bool):
            raise ValueError(f"invalid sweep preflight gate flags{where}")
    else:
        force_requested = False
        complete = bool(
            counts["attempted"] > 0
            and counts["reachable"] == counts["attempted"]
            and counts["readable"] == counts["attempted"])
        outcome = ("skipped" if preflight["skipped"] else
                   "preflight_passed" if complete else
                   "preflight_state_unknown")
        gate_satisfied = complete
    allowed_outcomes = {
        "skipped", "preflight_passed", "preflight_refused",
        "preflight_forced_unreadable", "preflight_forced_failed",
        "preflight_state_unknown",
    }
    if outcome not in allowed_outcomes:
        raise ValueError(f"invalid sweep preflight outcome{where}")
    complete = bool(
        counts["attempted"] > 0
        and counts["reachable"] == counts["attempted"]
        and counts["readable"] == counts["attempted"])
    outcome_valid = (
        (outcome == "skipped" and preflight["skipped"]
         and not gate_satisfied)
        or (outcome == "preflight_passed" and not preflight["skipped"]
            and complete and gate_satisfied)
        or (outcome == "preflight_forced_unreadable"
            and not preflight["skipped"] and force_requested
            and not gate_satisfied
            and counts["attempted"] > 0
            and counts["reachable"] == counts["attempted"]
            and counts["readable"] < counts["attempted"])
        or (outcome == "preflight_forced_failed"
            and not preflight["skipped"] and force_requested
            and not gate_satisfied
            and counts["reachable"] < counts["attempted"])
        or (outcome in {"preflight_refused", "preflight_state_unknown"}
            and not preflight["skipped"] and not gate_satisfied)
    )
    if not outcome_valid:
        raise ValueError(f"sweep preflight outcome disagrees with counts{where}")
    if rung_count is not None:
        expected_events = 0
        if float(cooldown) > 0:
            expected_events = max(0, rung_count - 1)
            if not preflight["skipped"]:
                expected_events += 1
        if events != expected_events:
            raise ValueError(
                f"sweep cooldown accounting disagrees with attempted rungs{where}")
    normalized = {
        "endpoint": endpoint,
        "sweep_wall_s": float(wall),
        "cooldown_s": float(cooldown),
        "cooldown_events": events,
        "preflight": {
            "skipped": preflight["skipped"],
            **counts,
            "outcome": outcome,
            "force_requested": force_requested,
            "gate_satisfied": gate_satisfied,
        },
    }
    if is_v2:
        def rates_field(name: str) -> list[float]:
            raw = context.get(name)
            if not isinstance(raw, list):
                raise ValueError(f"invalid sweep {name}{where}")
            values = []
            for value in raw:
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)) or not math.isfinite(
                            float(value)) or float(value) <= 0:
                    raise ValueError(f"invalid sweep {name}{where}")
                values.append(float(value))
            if any(right <= left for left, right in zip(
                    values, values[1:])):
                raise ValueError(f"sweep {name} must be strictly increasing{where}")
            return values

        planned = rates_field("planned_rates")
        attempted = rates_field("attempted_rates")
        omitted = rates_field("omitted_rates")
        if attempted != planned[:len(attempted)] \
                or omitted != planned[len(attempted):]:
            raise ValueError(
                f"attempted/omitted sweep rates do not partition the plan{where}")
        if rung_count is not None and len(attempted) != rung_count:
            raise ValueError(f"attempted rates disagree with rung count{where}")
        policy = context.get("progression_policy")
        expected_policy = {
            "early_stop_on_definitive_fail",
            "diagnostic_only",
            "invalid_or_quota_always_stops",
        }
        if not isinstance(policy, dict) or set(policy) != expected_policy \
                or any(not isinstance(policy[field], bool)
                       for field in expected_policy) \
                or policy["invalid_or_quota_always_stops"] is not True:
            raise ValueError(f"invalid sweep progression policy{where}")
        termination = context.get("termination_reason")
        allowed_termination = {
            "completed_planned_ladder", "invalid_measurement",
            "quota_limited", "definitive_sla_failure_early_stop",
            "missing_capacity_criterion",
            "stopped_before_planned_ladder_end",
        }
        if termination not in allowed_termination:
            raise ValueError(f"invalid sweep termination reason{where}")
        if (not omitted) != (termination == "completed_planned_ladder"):
            raise ValueError(
                f"sweep termination reason disagrees with omitted rates{where}")
        cooldown_records = context.get("cooldown_records")
        if not isinstance(cooldown_records, list) \
                or len(cooldown_records) != events:
            raise ValueError(f"cooldown records disagree with event count{where}")
        normalized_records = []
        for record in cooldown_records:
            if not isinstance(record, dict) or set(record) != {
                    "after", "requested_s", "started_at_unix",
                    "finished_at_unix", "elapsed_s"}:
                raise ValueError(f"invalid cooldown record{where}")
            after = record.get("after")
            if not isinstance(after, str) or not after.strip() \
                    or after != sanitize_title(after):
                raise ValueError(f"invalid cooldown record label{where}")
            numbers = {}
            for field in ("requested_s", "started_at_unix",
                          "finished_at_unix", "elapsed_s"):
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"invalid cooldown {field}{where}")
                numbers[field] = float(value)
            if numbers["requested_s"] != float(cooldown) \
                    or numbers["elapsed_s"] < 0 \
                    or numbers["finished_at_unix"] < numbers["started_at_unix"]:
                raise ValueError(f"inconsistent cooldown record{where}")
            normalized_records.append({"after": after, **numbers})
        normalized.update({
            "cooldown_records": normalized_records,
            "planned_rates": planned,
            "attempted_rates": attempted,
            "omitted_rates": omitted,
            "progression_policy": dict(policy),
            "termination_reason": termination,
        })
        quota = context.get("sweep_quota_evidence")
        expected_quota_fields = {
            "traffic_population", "request_rows", "http_429_count",
            "quota_status", "observed_rate_windows",
            "configured_rate_limits", "runtime_quota_admission",
        }
        if not isinstance(quota, dict) or set(quota) != expected_quota_fields:
            raise ValueError(f"invalid sweep quota evidence{where}")
        if not isinstance(quota.get("traffic_population"), str) \
                or not quota["traffic_population"].strip():
            raise ValueError(f"invalid sweep quota population{where}")
        for field in ("request_rows", "http_429_count"):
            value = quota.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid sweep quota {field}{where}")
        if quota["http_429_count"] > quota["request_rows"]:
            raise ValueError(f"sweep quota 429 count exceeds rows{where}")
        if quota.get("quota_status") not in {
                "LIMITED", "NEAR_LIMIT",
                "HEADROOM_UNESTABLISHED_SHORT_WINDOW",
                "NO_429_OBSERVED", "UNKNOWN"}:
            raise ValueError(f"invalid sweep quota status{where}")
        if not isinstance(quota.get("observed_rate_windows"), dict) \
                or (quota.get("configured_rate_limits") is not None
                    and not isinstance(quota["configured_rate_limits"], dict)):
            raise ValueError(f"invalid sweep quota evidence body{where}")
        local = quota.get("runtime_quota_admission")
        expected_local_fields = {
            "status", "guard_ids", "denied_rows",
            "denied_attempts_in_captured_rows", "invariant_errors"}
        if not isinstance(local, dict) or set(local) != expected_local_fields \
                or local.get("status") not in {
                    "not_configured", "enforced", "denied",
                    "invalid_evidence"} \
                or not isinstance(local.get("guard_ids"), list) \
                or any(not isinstance(item, str) or not item
                       for item in local["guard_ids"]) \
                or not isinstance(local.get("invariant_errors"), list) \
                or any(not isinstance(item, str) or not item
                       for item in local["invariant_errors"]):
            raise ValueError(f"invalid sweep runtime quota evidence{where}")
        for field in ("denied_rows", "denied_attempts_in_captured_rows"):
            value = local.get(field)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise ValueError(
                    f"invalid sweep runtime quota {field}{where}")
        normalized["sweep_quota_evidence"] = copy.deepcopy(quota)
    return normalized


def render_sweep_report(rungs: list[dict], report_context: dict) -> str:
    """Render the only report text a sealed sweep is allowed to contain."""
    context = _validated_report_context(report_context)
    outcome = sweep_outcome(rungs, context["preflight"])

    def number(value, digits=0):
        return "-" if value is None else f"{value:,.{digits}f}"

    def percent(value):
        return "-" if value is None else f"{value:.1%}"

    def past_verdict(kind: str) -> str:
        return {
            "ok": "held", "caution": "cautioned", "miss": "missed",
            "invalid": "was invalid",
        }[kind]

    definitions = {rung.get("first_event_definition") for rung in rungs
                   if rung.get("first_event_definition") is not None}
    first_label = ("TTFV" if definitions == {"first_visible"} else "TTFT")
    transport_unmatched = [
        rung for rung in rungs if not _transport_parity_exact(rung)]

    rows = [
        f"| rate asked | achieved | in-flight p50 | error | {first_label} p50 | "
        f"{first_label} p95 | E2E p50 | success | Wilson 95% lower | "
        "target | SLA state | quota state | transport parity |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rung in rungs:
        success_text, lower_text, target_text = _decision_percent_display(
            rung.get("success_rate_actual"),
            rung.get("success_rate_wilson_lower_95"),
            rung.get("success_rate_target"))
        rows.append(
            f"| {rate_label(rung['rate'])} rps | "
            f"{number(rung['achieved_rps'], 1)} | "
            f"{number(rung['held'])} | {percent(rung['err'])} | "
            f"{number(rung['ttft_p50'])} | {number(rung['ttft_p95'])} | "
            f"{number(rung['e2e_p50'])} | "
            f"{success_text} | {lower_text} | {target_text} | "
            f"{_rung_state(rung)} | "
            f"{rung.get('quota_status') or 'UNKNOWN'} | "
            f"{rung.get('transport_parity_status') or 'UNVERIFIED'} |")

    unverified = outcome["unverified"]
    good = outcome["good"]
    if outcome["invalid"]:
        rates = ", ".join(rate_label(r["rate"]) for r in unverified)
        detail = "; ".join(outcome["invalid_reasons"])
        head = (
            f"INVALID SWEEP: {detail}. "
            + (f"Unverified rate{'s' if len(unverified) != 1 else ''}: "
               f"{rates} rps. " if unverified else "")
            + "All successful rungs are diagnostic only; this sweep makes no "
              "capacity conclusion.")
    elif outcome["quota_limited"]:
        first = outcome["quota_limited"][0]
        quota_evidence = context.get("sweep_quota_evidence") or {}
        local_admission = quota_evidence.get(
            "runtime_quota_admission") or {}
        http_429_count = int(quota_evidence.get("http_429_count") or 0)
        if local_admission.get("status") == "denied" and http_429_count:
            head = (
                f"LOCAL SAFETY STOP AND PROVIDER RATE LIMITING at "
                f"{rate_label(first['rate'])} rps. The runtime guard refused "
                "a physical POST before the requested load was fully "
                f"delivered; separately, {http_429_count} captured HTTP 429 "
                f"response{'s' if http_429_count != 1 else ''} were "
                "observed. Neither establishes an endpoint-capacity ceiling.")
        elif local_admission.get("status") == "denied":
            head = (
                f"LOCAL SAFETY STOP at {rate_label(first['rate'])} rps. "
                "The no-wait runtime guard refused a physical POST before "
                "the requested load was fully delivered. This is not HTTP "
                "429 evidence and supports no endpoint-capacity ceiling.")
        elif http_429_count > 0:
            head = (
                f"QUOTA-LIMITED at {rate_label(first['rate'])} rps. The "
                "artifact is valid, but HTTP 429 evidence supports no "
                "endpoint-capacity ceiling. Stop and identify the enforcing "
                "quota dimension.")
        else:
            head = (
                f"QUOTA-LIMITED at {rate_label(first['rate'])} rps. The "
                "available quota evidence supports no endpoint-capacity "
                "ceiling; inspect the bound request rows before retrying.")
    elif outcome["no_criterion"]:
        head = (
            "DIAGNOSTIC-ONLY SWEEP: no explicit customer latency plus "
            "reliability policy was scored. No rung is a held-rate or "
            "endpoint-capacity claim.")
    elif outcome["non_monotonic"]:
        head = (
            "NON-MONOTONIC SLA OUTCOME: a higher rung passed after a lower "
            "rung failed. The artifact is valid, but no capacity boundary "
            "can be inferred; repeat the ladder.")
    elif good:
        best = max(
            (rung for rung in good
             if isinstance(rung.get("achieved_rps"), (int, float))
             and not isinstance(rung.get("achieved_rps"), bool)
             and math.isfinite(float(rung["achieved_rps"]))
             and float(rung["achieved_rps"]) >= 0),
            key=lambda rung: (float(rung["achieved_rps"]),
                              float(rung["rate"])),
            default=good[-1])
        boundary_best = good[-1]
        head = ("Highest rate that held: "
                f"{number(best['achieved_rps'], 2)} delivered "
                "requests/second average on the "
                f"{rate_label(best['rate'])} requested rps rung, "
                f"with observed in-flight p50 {number(best['held'])}. "
                "This is the highest achieved average among SLA-passing "
                "rungs, not by itself "
                "an endpoint ceiling. Capacity conclusion: "
                f"{outcome['capacity_conclusion']}.")

        def sentence(value: str) -> str:
            value = value.strip()
            return value if value.endswith(".") else value + "."

        nxt = next((r for r in rungs
                    if r["rate"] > boundary_best["rate"]), None)
        if nxt:
            head += (f" The next rung, {rate_label(nxt['rate'])} rps, "
                     f"{past_verdict(nxt['kind'])}: "
                     + sentence(nxt["text"]))
        else:
            head += (
                " That was the top of the authorized ladder; no ceiling was "
                "established. Extend the ladder only in a newly authorized "
                "window after reviewing quota, cost, generator, and endpoint "
                "telemetry.")
    else:
        first = rungs[0]
        detail = str(first["text"]).strip()
        if outcome["insufficient"]:
            if transport_unmatched:
                head = (
                    "No publishable SLA-passing rung was established. "
                    "Production transport parity was not proven by an exact "
                    "actual-versus-declared connection-policy match on "
                    f"{len(transport_unmatched)} rung"
                    f"{'s' if len(transport_unmatched) != 1 else ''}; no "
                    "held-rate or capacity conclusion is allowed.")
            else:
                head = (
                    "No SLA-passing rung was established. Every usable rung "
                    "had insufficient evidence; increase duration/sample size "
                    "before making a capacity claim.")
        else:
            head = ("No rung held. The lowest rate tested "
                    f"({rate_label(first['rate'])} rps) already "
                    f"{past_verdict(first['kind'])}: "
                    + (detail if detail.endswith(".") else detail + "."))

    report_links = [
        (f"- {rate_label(r['rate'])} rps: `{r['dir']}/report.html`"
         if r.get("source_position") is not None else
         f"- {rate_label(r['rate'])} rps: no verified report was produced")
        for r in rungs]
    preflight = context["preflight"]
    if preflight["skipped"]:
        preflight_text = "Preflight traffic: skipped; 0 requests sent."
    else:
        preflight_text = (
            f"Preflight traffic: {preflight['attempted']} representative "
            f"requests attempted ({preflight['reachable']} reached HTTP 200, "
            f"{preflight['readable']} produced readable answers), plus "
            f"{preflight['reasoning_probe_requests']} explicitly requested "
            "reasoning-control probe requests. Gate outcome: "
            f"{preflight['outcome']}.")
    verified = [r for r in rungs if r.get("source_position") is not None]
    request_rows = sum(int(r["request_rows"]) for r in verified)
    replay_rows = sum(int(r["replay_rows"]) for r in verified)
    calibration_rows = sum(int(r["calibration_rows"]) for r in verified)
    sizing_rows = sum(int(r["sizing_rows"]) for r in verified)
    preflight_rows = sum(int(r["preflight_rows"]) for r in verified)
    probe_rows = sum(int(r["probe_rows"]) for r in verified)
    other_rows = sum(int(r["other_rows"]) for r in verified)
    unknown_attempt_rows = sum(
        int(r["unknown_attempt_rows"]) for r in verified)
    traffic_text = (
        f"Authenticated rung traffic: {request_rows} request rows "
        f"({replay_rows} replay, {calibration_rows} calibration, "
        f"{sizing_rows} sizing, {preflight_rows} preflight, "
        f"{probe_rows} probe, {other_rows} other; "
        f"{unknown_attempt_rows} rows with unknown provider-attempt "
        "timing/count).")
    if outcome["unverified"]:
        traffic_text += (
            f" {len(outcome['unverified'])} unverified rung attempt"
            f"{'s have' if len(outcome['unverified']) != 1 else ' has'} "
            "traffic that cannot be fully accounted from a sealed run.")
    cooldown_text = (
        f"Cooldown spacing: {context['cooldown_s']:g}s after preflight and "
        f"between measured rungs; {context['cooldown_events']} spacing "
        f"event{'s' if context['cooldown_events'] != 1 else ''} recorded. "
        "This sweep is sequential and stateful. Cooldown is spacing only; "
        "it proves neither QPH recovery nor provider burst or cache reset.")
    if context.get("cooldown_records"):
        elapsed = [record["elapsed_s"]
                   for record in context["cooldown_records"]]
        shortest = min(elapsed)
        cooldown_text += (
            f" Measured elapsed spacing ranged from {shortest:.3f}s to "
            f"{max(elapsed):.3f}s.")
        if context["cooldown_s"] and shortest < 0.95 * context["cooldown_s"]:
            cooldown_text += (
                " CAUTION: at least one requested cooldown was not actually "
                "observed for its full duration; do not claim that spacing.")
    plan_text = ""
    if context.get("planned_rates") is not None:
        plan_text = (
            "Planned ladder: "
            + ", ".join(rate_label(rate)
                        for rate in context["planned_rates"])
            + " rps. Attempted: "
            + (", ".join(rate_label(rate)
                         for rate in context["attempted_rates"]) or "none")
            + ". Omitted: "
            + (", ".join(rate_label(rate)
                         for rate in context["omitted_rates"]) or "none")
            + f". Termination: {context['termination_reason']}."
        )
    quota_text = ""
    if context.get("sweep_quota_evidence") is not None:
        quota = context["sweep_quota_evidence"]
        quota_text = (
            "Sweep-level quota evidence (all manifest-bound phases pooled "
            f"once): {quota['request_rows']} request rows, "
            f"{quota['http_429_count']} HTTP 429, status "
            f"{quota['quota_status']}.")
    if transport_unmatched:
        transport_text = (
            "CAUTION: production transport parity was not exactly "
            f"established for {len(transport_unmatched)} of {len(rungs)} "
            "attempted rungs. The benchmark's connection policy can change "
            "DNS/TCP/TLS pressure versus the production client, so these "
            "rungs are diagnostic-only and cannot support a held-rate or "
            "capacity claim.")
    else:
        policies = sorted({
            str(rung["transport_connection_policy_id"])
            for rung in rungs})
        transport_text = (
            "Production transport parity: exact actual-versus-declared "
            f"connection-policy match on every rung ({', '.join(policies)}).")

    body = "\n".join([
        f"# Rate ladder: {context['endpoint']}", "", head, "",
        (f"Sweep command wall time: {context['sweep_wall_s']:.1f}s. Per-rung "
         "wall time includes setup and response drain; the configured "
         "duration is offered-load schedule time."), "", preflight_text,
        traffic_text, cooldown_text, transport_text, "", "\n".join(rows), "",
        *([plan_text, ""] if plan_text else []),
        *([quota_text, ""] if quota_text else []),
        "The axis is arrival rate because that is what an open-loop generator "
        "controls. Concurrency is reported as measured, not as asked for. "
        "Under steady-state assumptions, mean in-flight is approximately "
        "achieved throughput times mean residence time; the reported p50 is "
        "an observed outcome rather than an input.", "",
        (f"Configured first-event metric: {first_label}; each rung seals its "
         "exact metric key and caller/request-path basis."), "",
        "Per-rung reports:", "", *report_links,
    ])
    return body + "\n"


def render_sweep_html(rungs: list[dict], report_context: dict,
                      artifact_id: str) -> str:
    """Render a sealed, dependency-free rate-ladder decision surface."""
    context = _validated_report_context(report_context)
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("invalid sweep HTML artifact_id")
    outcome = sweep_outcome(rungs, context["preflight"])
    markdown_report = render_sweep_report(rungs, context)
    sections = markdown_report.split("\n\n")
    decision_text = sections[1] if len(sections) > 1 else \
        "No canonical sweep decision was rendered."

    def esc(value: object) -> str:
        return html_escape(str(value), quote=True)

    def number(value: object, digits: int = 1) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            return "Not available"
        return f"{float(value):,.{digits}f}"

    def percent(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            return "Not available"
        return f"{float(value):.2%}"

    definitions = {
        rung.get("first_event_definition") for rung in rungs
        if rung.get("first_event_definition") is not None}
    first_label = "TTFV" if definitions == {"first_visible"} else "TTFT"
    transport_unmatched = [
        rung for rung in rungs if not _transport_parity_exact(rung)]
    if outcome["invalid"]:
        status_label, status_class = "INVALID SWEEP", "invalid"
    elif outcome["quota_limited"]:
        status_label, status_class = "SAFETY / QUOTA STOP", "stop"
    elif outcome["no_criterion"]:
        status_label, status_class = "DIAGNOSTIC ONLY", "review"
    elif outcome["non_monotonic"]:
        status_label, status_class = "NON-MONOTONIC", "review"
    elif outcome["good"]:
        status_label, status_class = "TESTED RATE HELD", "pass"
    elif outcome["insufficient"] and transport_unmatched:
        status_label, status_class = "TRANSPORT PARITY UNVERIFIED", "review"
    else:
        status_label, status_class = "NO PASSING RUNG", "review"

    verified = [rung for rung in rungs
                if rung.get("source_position") is not None]
    total_rows = sum(int(rung.get("request_rows") or 0)
                     for rung in verified)
    quota = context.get("sweep_quota_evidence") or {}
    local = quota.get("runtime_quota_admission") \
        if isinstance(quota, dict) else {}
    local = local if isinstance(local, dict) else {}
    highest_requested = outcome["highest_sla_passing_tested_rate"]
    highest_delivered = outcome[
        "highest_achieved_rate_at_sla_passing_rung"]
    requested_at_highest_delivered = outcome[
        "requested_rate_at_highest_achieved_sla_passing_rung"]

    rows = []
    details = []
    bars = []
    axis_values = [float(rung["rate"]) for rung in rungs]
    axis_values.extend(
        float(rung["achieved_rps"]) for rung in rungs
        if isinstance(rung.get("achieved_rps"), (int, float))
        and not isinstance(rung.get("achieved_rps"), bool)
        and math.isfinite(float(rung["achieved_rps"]))
        and float(rung["achieved_rps"]) >= 0)
    max_axis = max(axis_values, default=1.0)
    for position, rung in enumerate(rungs):
        success_text, lower_text, target_text = _decision_percent_display(
            rung.get("success_rate_actual"),
            rung.get("success_rate_wilson_lower_95"),
            rung.get("success_rate_target"))
        state = _rung_state(rung)
        row_class = (
            "pass" if state == "PASS" else
            "invalid" if state in {"INVALID", "QUOTA_LIMITED"} else
            "review")
        source_position = rung.get("source_position")
        if source_position is None:
            report_link = "<span class='muted'>No sealed run</span>"
        else:
            relative = f"{rung['dir']}/report.html"
            href = quote(relative, safe="/._~-")
            report_link = f"<a href='{esc(href)}'>Open run report</a>"
        latency_basis = rung.get("latency_basis") or "not recorded"
        transport_exact = _transport_parity_exact(rung)
        transport_class = "pass" if transport_exact else "review"
        transport_status = rung.get("transport_parity_status") or "UNVERIFIED"
        transport_note = (
            rung.get("production_connection_policy_assurance")
            if transport_exact else _transport_parity_reason(rung))
        rows.append(
            f"<tr><th scope='row' class='sticky-col'><span class='rate'>"
            f"{esc(rate_label(rung['rate']))}</span> rps</th>"
            f"<td>{esc(number(rung.get('achieved_rps'), 2))}</td>"
            f"<td>{esc(number(rung.get('held'), 1))}</td>"
            f"<td>{esc(percent(rung.get('err')))}</td>"
            f"<td>{esc(number(rung.get('ttft_p95'), 1))} ms"
            f"<small>{esc(latency_basis)}</small></td>"
            f"<td>{esc(number(rung.get('e2e_p50'), 1))} ms</td>"
            f"<td>{esc(success_text)}"
            f"<small>95% lower "
            f"{esc(lower_text)} · target {esc(target_text)}</small>"
            f"</td>"
            f"<td>{esc(number(rung.get('request_start_lateness_p95'), 1))} "
            "ms</td>"
            f"<td><span class='state {row_class}'>{esc(state)}</span>"
            f"<small>quota {esc(rung.get('quota_status') or 'UNKNOWN')}"
            f" · transport {esc(transport_status)}"
            f"</small></td><td>{report_link}</td></tr>")
        details.append(
            f"<article class='rung-detail'><div><span class='state "
            f"{row_class}'>{esc(state)}</span><h3>"
            f"{esc(rate_label(rung['rate']))} requested rps</h3></div>"
            f"<p>{esc(rung.get('text') or 'No rung reason recorded.')}</p>"
            "<dl>"
            f"<dt>Response identity</dt><dd>"
            f"{esc(rung.get('response_identity_status') or 'not recorded')}"
            "</dd>"
            f"<dt>Endpoint stability</dt><dd>"
            f"{esc(rung.get('endpoint_metadata_stability') or 'not recorded')}"
            "</dd>"
            f"<dt>Transport parity</dt><dd><span class='state "
            f"{transport_class}'>{esc(transport_status)}</span>"
            f"<br>benchmark policy: <code>"
            f"{esc(rung.get('transport_connection_policy_id') or 'not recorded')}"
            f"</code><br>declared production policy: <code>"
            f"{esc(rung.get('production_connection_policy_declared') or 'not recorded')}"
            f"</code><br>explicit exact match: "
            f"{'yes' if transport_exact else 'no'}<br>"
            f"{esc(transport_note or 'no transport assurance recorded')}</dd>"
            f"<dt>Runtime admission</dt><dd>"
            f"{esc(rung.get('runtime_quota_admission_status') or 'not recorded')}"
            "</dd>"
            f"<dt>Guard ID</dt><dd><code>"
            f"{esc(rung.get('runtime_quota_guard_id') or 'not recorded')}"
            "</code></dd>"
            f"<dt>Dispatch lag p95</dt><dd>"
            f"{esc(number(rung.get('dispatch_lag_p95'), 1))} ms</dd>"
            f"<dt>Captured request rows</dt><dd>"
            f"{esc(rung.get('request_rows') if source_position is not None else 'unverified')}"
            "</dd></dl></article>")
        asked_width = max(0.0, min(100.0,
            float(rung["rate"]) / max_axis * 100.0))
        achieved_value = rung.get("achieved_rps")
        achieved_number = (
            float(achieved_value)
            if isinstance(achieved_value, (int, float))
            and not isinstance(achieved_value, bool)
            and math.isfinite(float(achieved_value))
            and float(achieved_value) >= 0 else None)
        achieved_width = (0.0 if achieved_number is None else
                          max(0.0, min(100.0,
                              achieved_number / max_axis * 100.0)))
        bars.append(
            "<div class='bar-row' role='img' aria-label='"
            f"Asked {esc(rate_label(rung['rate']))} requests per second; "
            f"achieved {esc(number(achieved_number, 2))} requests per second'>"
            f"<div class='bar-label'>{esc(rate_label(rung['rate']))} rps</div>"
            "<div class='tracks'><div class='track'><span class='asked' "
            f"style='width:{asked_width:.3f}%'></span></div>"
            "<div class='track'><span class='achieved' "
            f"style='width:{achieved_width:.3f}%'></span></div></div>"
            f"<div class='bar-value'>{esc(number(achieved_number, 2))}</div>"
            "</div>")

    planned = context.get("planned_rates")
    planned_text = (
        ", ".join(rate_label(rate) for rate in planned) + " rps"
        if isinstance(planned, list) and planned else "not recorded")
    attempted = context.get("attempted_rates")
    attempted_text = (
        ", ".join(rate_label(rate) for rate in attempted) + " rps"
        if isinstance(attempted, list) and attempted else "none")
    omitted = context.get("omitted_rates")
    omitted_text = (
        ", ".join(rate_label(rate) for rate in omitted) + " rps"
        if isinstance(omitted, list) and omitted else "none")
    preflight = context["preflight"]
    if transport_unmatched:
        reasons = []
        for rung in transport_unmatched:
            reason = _transport_parity_reason(rung)
            if reason not in reasons:
                reasons.append(reason)
        transport_caution = (
            "<aside class='top-caution' role='note' "
            "aria-labelledby='transport-caution-heading'>"
            "<h2 id='transport-caution-heading'>Production transport parity "
            "is not established</h2><p>"
            f"{len(transport_unmatched)} of {len(rungs)} attempted rung(s) "
            "lack an explicit exact actual-versus-declared connection-policy "
            "match. No green held-rate or capacity conclusion is allowed. "
            f"{esc(' '.join(reasons))}</p></aside>")
        transport_card_value = "UNVERIFIED"
        transport_card_note = (
            f"{len(transport_unmatched)} of {len(rungs)} rung(s) lack an "
            "exact match")
    else:
        transport_caution = ""
        transport_card_value = "EXACT MATCH"
        transport_card_note = (
            "every rung binds the benchmark policy to the declared "
            "production policy")

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Rate ladder · {esc(context['endpoint'])}</title>
<style>
:root{{--ink:#172033;--muted:#61708a;--line:#d9e0ec;--paper:#fff;--wash:#f4f7fb;--navy:#173b73;--cyan:#007b8c;--green:#087443;--green-bg:#e9f8f0;--amber:#895500;--amber-bg:#fff4d8;--red:#b42318;--red-bg:#fff0ee;--shadow:0 18px 45px rgba(23,32,51,.10)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:linear-gradient(145deg,#edf3fb 0,#f8fafc 45%,#eef7f6 100%);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}a{{color:#075e93;text-underline-offset:3px}}.skip{{position:absolute;left:-9999px}}.skip:focus{{left:16px;top:12px;background:#fff;padding:10px;z-index:5}}.shell{{width:min(1440px,calc(100% - 32px));margin:28px auto 64px}}.hero{{padding:34px;border-radius:24px;background:linear-gradient(125deg,#132c55,#075e72);color:#fff;box-shadow:var(--shadow)}}.eyebrow{{font-size:.76rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;opacity:.8}}h1{{font-size:clamp(2rem,4vw,3.5rem);line-height:1.04;margin:.35rem 0 .7rem;letter-spacing:-.04em;overflow-wrap:anywhere}}h2{{font-size:1.45rem;line-height:1.2;margin:0}}h3{{font-size:1rem;margin:.35rem 0 0}}.artifact{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;overflow-wrap:anywhere;opacity:.78}}.hero-state{{display:inline-block;margin-top:18px;padding:7px 11px;border-radius:999px;font-size:.76rem;font-weight:900;letter-spacing:.06em;background:#fff;color:#173b73}}.hero-state.invalid,.state.invalid{{background:var(--red-bg);color:var(--red)}}.hero-state.stop{{background:#ffe6dd;color:#9d2d15}}.hero-state.review,.state.review{{background:var(--amber-bg);color:var(--amber)}}.hero-state.pass,.state.pass{{background:var(--green-bg);color:var(--green)}}.decision{{max-width:1050px;margin:18px 0 0;font-size:1.05rem}}.top-caution{{margin:18px 0 0;padding:16px 20px;border:1px solid #e3bd58;border-left:6px solid var(--amber);border-radius:14px;background:var(--amber-bg)}}.top-caution h2{{font-size:1.05rem}}.top-caution p{{margin:.4rem 0 0}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}nav a{{border:1px solid #b9c6d8;border-radius:999px;background:rgba(255,255,255,.84);padding:7px 11px;text-decoration:none;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}}.card,.section,.rung-detail{{background:var(--paper);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 22px rgba(23,32,51,.055)}}.card{{padding:16px;min-height:112px}}.card .label{{font-size:.72rem;color:var(--muted);font-weight:800;letter-spacing:.06em;text-transform:uppercase}}.card strong{{display:block;font-size:1.28rem;line-height:1.12;margin-top:10px;overflow-wrap:normal}}.card small,td small{{display:block;color:var(--muted);margin-top:4px}}.section{{padding:24px;margin-top:18px}}.section-head{{display:flex;justify-content:space-between;gap:14px;align-items:end;margin-bottom:16px}}.section-head p{{color:var(--muted);margin:0;max-width:760px}}.delivery{{display:grid;gap:9px;margin:0 0 18px;padding:15px;border:1px solid var(--line);border-radius:12px;background:#f8fafc}}.legend{{display:flex;gap:18px;color:var(--muted);font-size:.78rem}}.legend span:before{{content:"";display:inline-block;width:14px;height:7px;border-radius:99px;margin-right:6px;background:#9bacbf}}.legend .achieved-key:before{{background:#09859a}}.bar-row{{display:grid;grid-template-columns:80px 1fr 70px;align-items:center;gap:10px}}.bar-label,.bar-value{{font-variant-numeric:tabular-nums;font-weight:750}}.bar-value{{text-align:right}}.tracks{{display:grid;gap:3px}}.track{{height:8px;border-radius:99px;background:#e5eaf1;overflow:hidden}}.track span{{display:block;height:100%;border-radius:inherit}}.track .asked{{background:#9bacbf}}.track .achieved{{background:#09859a}}.scroll-hint{{margin:0 0 8px;padding:7px 10px;border-radius:8px;background:#eef4ff;color:#174ea6;font-size:.78rem;font-weight:750}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;overscroll-behavior-inline:contain;scrollbar-gutter:stable}}.table-wrap:focus-visible{{outline:3px solid #155eef;outline-offset:3px}}table{{border-collapse:separate;border-spacing:0;width:100%;min-width:1120px;background:#fff}}caption{{text-align:left;padding:11px 12px;background:#f8fafc;color:var(--muted);font-weight:700}}th,td{{padding:12px 11px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}thead th{{position:sticky;top:0;background:#edf3f9;color:#34435c;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;z-index:2}}tbody th{{font-weight:750}}.sticky-col{{position:sticky;inset-inline-start:0;z-index:3;box-shadow:6px 0 8px -8px #172033}}thead .sticky-col{{z-index:4;background:#edf3f9}}tbody .sticky-col{{background:#fff}}tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}.rate{{font-size:1.05rem}}.state{{display:inline-block;border-radius:999px;background:#edf1f7;color:#44516a;padding:4px 8px;font-size:.69rem;font-weight:900;letter-spacing:.035em}}.details-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.rung-detail{{padding:17px}}.rung-detail>div{{display:flex;align-items:center;gap:10px}}.rung-detail p{{color:#3e4b61}}dl{{display:grid;grid-template-columns:minmax(120px,.7fr) 1fr;gap:7px 12px;margin:12px 0 0}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}code{{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2f7;border-radius:5px;padding:2px 5px}}.method-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.method-grid article{{border-left:3px solid #79a8c8;padding-left:13px}}.method-grid p{{color:#45536a;margin:.5rem 0 0}}.provenance{{background:#162238;color:#e8eef8}}.provenance h2,.provenance dt{{color:#fff}}.provenance dt{{opacity:.72}}.provenance code{{background:#253653;color:#fff}}.muted{{color:var(--muted)}}.print-stamp{{display:none}}footer{{color:var(--muted);font-size:.82rem;padding:20px 2px}}@media(max-width:1050px){{.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.method-grid{{grid-template-columns:1fr}}}}@media(max-width:700px){{.shell{{width:min(100% - 18px,1440px);margin-top:9px}}.hero,.section{{padding:19px;border-radius:15px}}.cards,.details-grid{{grid-template-columns:1fr}}.section-head{{display:block}}.section-head p{{margin-top:7px}}}}@media print{{@page{{size:landscape;margin:9mm}}body{{background:#fff;font:9pt/1.35 Arial,sans-serif}}.shell{{width:100%;margin:0}}.hero{{box-shadow:none;border-radius:0;background:#173b73!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}nav{{display:none}}.print-stamp{{display:block;border:1px solid #98a2b3;padding:2.5mm 3mm;margin:3mm 0;background:#fff;color:#344054;text-align:center;font-size:8pt;line-height:1.25;break-inside:avoid}}.card,.section,.rung-detail,.top-caution{{box-shadow:none;break-inside:avoid}}.cards{{grid-template-columns:repeat(3,1fr)}}.details-grid{{grid-template-columns:1fr}}.scroll-hint{{display:none}}.table-wrap{{overflow:visible}}table{{min-width:0;font-size:8pt;table-layout:fixed}}th,td{{padding:5px 4px;overflow-wrap:anywhere}}thead th,.sticky-col{{position:static;box-shadow:none}}a{{color:inherit}}}}
</style>
</head>
<body><a class='skip' href='#main'>Skip to benchmark evidence</a><main class='shell' id='main'>
<header class='hero'><div class='eyebrow'>Sealed rate-ladder evidence</div><h1>{esc(context['endpoint'])}</h1><div class='artifact'>Artifact {esc(artifact_id)}</div><span class='hero-state {status_class}'>{esc(status_label)}</span><p class='decision'>{esc(decision_text)}</p></header>
<div class='print-stamp' role='note'>UNSEALED PRINT/PDF DERIVATIVE: verify the sweep manifest · artifact {esc(artifact_id)} · internal hashes are not a digital signature</div>
{transport_caution}
<nav aria-label='Report sections'><a href='#decision'>Decision</a><a href='#rungs'>Rungs</a><a href='#evidence'>Evidence</a><a href='#method'>Method</a></nav>
<section class='cards' id='decision' aria-label='Sweep decision summary'>
<article class='card'><div class='label'>Highest delivered rate at an SLA-passing rung</div><strong>{esc(number(highest_delivered, 2))} rps</strong><small>{esc(number(requested_at_highest_delivered, 2))} requested rps at that rung; highest requested passing rung {esc(number(highest_requested, 2))} rps; not an endpoint ceiling by itself</small></article>
<article class='card'><div class='label'>Capacity conclusion</div><strong>{esc(outcome['capacity_conclusion'].replace('_', ' '))}</strong><small><code>{esc(outcome['capacity_conclusion'])}</code> · boundary {esc(outcome['boundary_status'])}</small></article>
<article class='card'><div class='label'>Evidence population</div><strong>{total_rows:,} rows</strong><small>{len(verified)} sealed run(s); {len(rungs)} attempted rung(s)</small></article>
<article class='card'><div class='label'>Quota evidence</div><strong>{esc(quota.get('quota_status') or 'UNKNOWN')}</strong><small>{esc(quota.get('http_429_count', 0))} HTTP 429; local guard {esc(local.get('status') or 'not recorded')}</small></article>
<article class='card'><div class='label'>Transport parity</div><strong>{esc(transport_card_value)}</strong><small>{esc(transport_card_note)}</small></article>
<article class='card'><div class='label'>Termination</div><strong>{esc(context.get('termination_reason') or 'not recorded')}</strong><small>{context['sweep_wall_s']:.1f}s command wall time</small></article>
</section>
<section class='section' id='rungs'><div class='section-head'><div><div class='eyebrow'>Offered versus observed</div><h2 id='rungs-heading'>Rung evidence</h2></div><p>Latency uses each rung's sealed metric and basis. Reliability shows the observed answer-success fraction and its one-sided 95% Wilson lower confidence bound.</p></div><div class='delivery' aria-label='Asked and achieved request rates'><div class='legend'><span>Asked rate</span><span class='achieved-key'>Achieved rate</span></div>{''.join(bars)}</div><div class='scroll-hint' id='rungs-scroll-hint' role='note'><span aria-hidden='true'>↔</span> Scroll horizontally; the Asked column stays visible.</div><div class='table-wrap' tabindex='0' role='region' aria-labelledby='rungs-heading' aria-describedby='rungs-scroll-hint'><table><caption>Sealed offered-load, latency, reliability, quota, transport-parity, and source evidence for every attempted rung.</caption><thead><tr><th scope='col' class='sticky-col'>Asked</th><th scope='col'>Achieved rps</th><th scope='col'>In-flight p50</th><th scope='col'>Error</th><th scope='col'>{esc(first_label)} p95</th><th scope='col'>E2E p50</th><th scope='col'>Reliability</th><th scope='col'>Request-start late p95</th><th scope='col'>State</th><th scope='col'>Source</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='section' id='evidence'><div class='section-head'><div><div class='eyebrow'>Independent gates</div><h2>Why each rung received its state</h2></div><p>Response identity, endpoint stability, production transport parity, and runtime admission remain separate from SLA outcome; a safe local refusal is not a provider HTTP 429 or an endpoint ceiling.</p></div><div class='details-grid'>{''.join(details)}</div></section>
<section class='section' id='method'><div class='section-head'><div><div class='eyebrow'>Experiment contract</div><h2>Plan, traffic, and interpretation</h2></div></div><div class='method-grid'><article><h3>Ladder execution</h3><p>Planned: {esc(planned_text)}. Attempted: {esc(attempted_text)}. Omitted: {esc(omitted_text)}. Rungs ran sequentially with {context['cooldown_s']:g}s requested spacing; cooldown is not proof of quota or cache reset.</p></article><article><h3>Setup traffic</h3><p>{esc(preflight['attempted'])} representative preflight request(s), {esc(preflight['reachable'])} HTTP 200, {esc(preflight['readable'])} readable answer(s), and {esc(preflight['reasoning_probe_requests'])} reasoning-control probe request(s). Gate outcome: <code>{esc(preflight['outcome'])}</code>. Setup traffic is included in sweep-level quota evidence, not performance percentiles.</p></article><article><h3>Load and latency</h3><p>The open-loop input is arrival rate. Under steady-state assumptions, mean in-flight is approximately achieved throughput × mean residence time. Raw latency begins immediately before the final-attempt <code>conn.request</code> call and excludes connection setup; caller metrics include local queue/retry effects when selected.</p></article></div></section>
<section class='section provenance'><div class='section-head'><div><div class='eyebrow'>Evidence boundary</div><h2>Publication and verification</h2></div></div><dl><dt>Artifact ID</dt><dd><code>{esc(artifact_id)}</code></dd><dt>Endpoint</dt><dd>{esc(context['endpoint'])}</dd><dt>Runtime guard IDs</dt><dd><code>{esc(', '.join(local.get('guard_ids') or []) or 'not recorded')}</code></dd><dt>Traffic population</dt><dd>{esc(quota.get('traffic_population') or 'not recorded')}</dd></dl><p>This HTML and <code>sweep.md</code> are authoritative only when their SHA-256 and byte counts match <code>manifest.json</code> and the completion marker verifies. A browser print or PDF is an unsealed derivative. Internal hashes detect changes; they are not a digital signature.</p></section>
<footer>No scripts, remote assets, remote fonts, or network requests. Reliability confidence assumes independent request outcomes. Runtime admission covers this harness command and does not observe unrelated workspace traffic.</footer>
</main></body></html>
"""


def _atomic_text(dir_fd: int, name: str, value: str) -> dict:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"unsafe sweep artifact name: {name!r}")
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600, dir_fd=dir_fd)
    raw = value.encode("utf-8")
    try:
        _write_compare_fd(fd, raw, name)
        os.fsync(fd)
    except Exception:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    try:
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except Exception:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    _fsync_fd(dir_fd)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _claim_dir(requested: Path, artifact_id: str,
               created_at: float) -> tuple[Path, int]:
    """Claim a fresh directory; an existing path is never entered or reused."""
    requested.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10_000):
        candidate = (requested if attempt == 0 else requested.with_name(
            f"{requested.name}-{uuid.uuid4().hex[:12]}"))
        try:
            candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError:
            continue
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        dir_fd = -1
        try:
            dir_fd = os.open(candidate, flags)
            marker_fd = os.open(
                _WRITING_MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600, dir_fd=dir_fd)
            try:
                marker = strict_json_dumps({
                    "artifact_id": artifact_id,
                    "artifact_type": "sweep",
                    "status": "writing",
                    "created_at_unix": created_at,
                }).encode("utf-8") + b"\n"
                _write_compare_fd(marker_fd, marker, _WRITING_MARKER)
                os.fsync(marker_fd)
            finally:
                os.close(marker_fd)
            _fsync_fd(dir_fd)
            _fsync_directory(candidate.parent)
            return candidate, dir_fd
        except Exception:
            if dir_fd >= 0:
                os.close(dir_fd)
            raise
    raise RuntimeError(f"could not claim a unique sweep directory: {requested}")


def _verified_run_snapshot(run_dir: str | Path, position: int,
                           rate: float) -> tuple[dict, dict]:
    """Authenticate one completed run and snapshot its exact summary identity."""
    d = Path(run_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"sweep rung directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"sweep rung is not a regular directory: {d}")
    manifest = _require_run_dir(d, "summary.json")
    manifest_raw = _read_regular_bytes(d / "manifest.json")
    current_manifest = _strict_object(
        manifest_raw, "manifest.json", d / "manifest.json")
    if current_manifest != manifest:
        raise ValueError(f"input manifest changed while reading sweep rung: {d}")
    _strict_object(
        _read_regular_bytes(d / _COMPLETE_MARKER), "completion marker",
        d / _COMPLETE_MARKER)

    expected = _artifact_declarations(manifest, d)["summary.json"]
    summary_raw = _read_regular_bytes(d / "summary.json")
    summary_sha = hashlib.sha256(summary_raw).hexdigest()
    if not hmac.compare_digest(summary_sha, expected["sha256"]):
        raise ValueError(f"artifact SHA-256 mismatch for {d / 'summary.json'}")
    if len(summary_raw) != expected["bytes"]:
        raise ValueError(f"artifact byte count mismatch for {d / 'summary.json'}")
    summary = _strict_object(summary_raw, "summary.json", d / "summary.json")
    request_metadata = _artifact_declarations(manifest, d)["requests.jsonl"]
    phase_counts: dict[str, int] = {}
    parsed_rows = 0
    unknown_attempt_rows = 0

    def account(row: dict, line_number: int) -> None:
        nonlocal parsed_rows, unknown_attempt_rows
        phase = row.get("phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError(
                f"requests.jsonl record {line_number} has no phase in {d}")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        attempts = row.get("request_attempts")
        known_attempts = (isinstance(attempts, int)
                          and not isinstance(attempts, bool)
                          and attempts >= 0)
        sent_at = row.get("first_send_unix")
        known_send_time = (isinstance(sent_at, (int, float))
                           and not isinstance(sent_at, bool)
                           and math.isfinite(float(sent_at))
                           and float(sent_at) >= 0)
        if not known_attempts or (attempts > 0 and not known_send_time):
            unknown_attempt_rows += 1
        parsed_rows += 1

    _scan_request_journal(d / "requests.jsonl", request_metadata, account)
    if parsed_rows != request_metadata["row_count"]:
        raise ValueError(f"strict request row count disagrees with manifest in {d}")
    replay_rows = phase_counts.get("replay", 0)
    calibration_rows = phase_counts.get("calibration", 0)
    sizing_rows = phase_counts.get("sizing", 0)
    preflight_rows = phase_counts.get("preflight", 0)
    probe_rows = phase_counts.get("probe", 0)
    other_rows = (parsed_rows - replay_rows - calibration_rows - sizing_rows
                  - preflight_rows - probe_rows)
    schedule_rows = manifest["schedule_identity"]["shard_count"]
    if replay_rows != schedule_rows:
        raise ValueError(f"replay row count disagrees with schedule identity in {d}")

    source = {
        "position": position,
        "rate_requests_per_second": float(rate),
        "artifact_id": manifest["artifact_id"],
        "logical_run_id": manifest["logical_run_id"],
        "execution_id": manifest["execution_id"],
        "workload_id": manifest["workload_id"],
        "manifest": {
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "bytes": len(manifest_raw),
        },
        "summary": {
            "sha256": summary_sha,
            "bytes": len(summary_raw),
        },
        "request_rows": parsed_rows,
        "replay_rows": replay_rows,
        "calibration_rows": calibration_rows,
        "sizing_rows": sizing_rows,
        "preflight_rows": preflight_rows,
        "probe_rows": probe_rows,
        "other_rows": other_rows,
        "unknown_attempt_rows": unknown_attempt_rows,
        "effective_config_sha256": manifest.get("effective_config_sha256"),
        "effective_config": manifest.get("effective_config"),
    }
    return summary, source


def _validate_source_shape(source: object, position: int, d: Path) -> None:
    if not isinstance(source, dict) or source.get("position") != position:
        raise ValueError(f"invalid sweep source position in {d}")
    rate = source.get("rate_requests_per_second")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
            or not math.isfinite(float(rate)) or float(rate) <= 0:
        raise ValueError(f"invalid sweep source rate in {d}")
    for field in ("artifact_id", "logical_run_id", "execution_id", "workload_id"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid sweep source {field} in {d}")
    relative = source.get("relative_path")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"invalid sweep source relative_path in {d}")
    rel_path = Path(relative)
    if rel_path.is_absolute() or ".." in rel_path.parts or rel_path == Path("."):
        raise ValueError(f"unsafe sweep source relative_path in {d}")
    for field in ("manifest", "summary"):
        metadata = source.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid sweep source {field} metadata in {d}")
        _identity_digest(metadata.get("sha256"),
                         f"sources[{position}].{field}.sha256", d)
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid sweep source {field} byte count in {d}")
    phase_fields = ("replay_rows", "calibration_rows", "sizing_rows",
                    "preflight_rows", "probe_rows", "other_rows")
    for field in ("request_rows", *phase_fields, "unknown_attempt_rows"):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid sweep source {field} in {d}")
    if source["request_rows"] > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
        raise ValueError(
            f"sweep source request_rows exceeds the exact-analysis limit of "
            f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,} in {d}")
    if source["request_rows"] != sum(source[field] for field in phase_fields):
        raise ValueError(f"sweep source request phase counts disagree in {d}")
    if source["unknown_attempt_rows"] > source["request_rows"]:
        raise ValueError(f"sweep source unknown attempt count disagrees in {d}")
    effective = source.get("effective_config")
    if not isinstance(effective, dict):
        raise ValueError(f"invalid sweep source effective_config in {d}")
    digest = _identity_digest(
        source.get("effective_config_sha256"),
        f"sources[{position}].effective_config_sha256", d)
    if canonical_sha256(effective) != digest:
        raise ValueError(f"sweep source effective config digest disagrees in {d}")


def _nested_regular_dir(root: Path, relative: str) -> Path:
    """Resolve a nested directory while refusing every symlink component."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError(f"unsafe nested sweep source path: {relative!r}")
    current = root
    root_info = current.lstat()
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"sweep root is not a regular directory: {root}")
    for part in rel.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"missing nested sweep source: {current}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(
                f"nested sweep source component is not a regular directory: "
                f"{current}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"nested sweep source escapes aggregate: {current}") from exc
    return current


def _capture_base_identity(base_config: dict) -> dict:
    """Pin immutable workload inputs before the first measured rung."""
    from .runner import _read_stable_bytes

    inputs = {}
    for field, key in (("profile_path", "profile"),
                       ("prompts_file", "prompts"),
                       ("timestamps_file", "timestamps")):
        path = base_config.get(field)
        if not path:
            continue
        raw, _info = _read_stable_bytes(path, input_kind=key)
        inputs[key] = {
            "name": Path(path).name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    if len(inputs) not in {1, 2} or not ({"profile", "prompts"} & set(inputs)):
        raise ValueError("sweep base config has no identifiable workload input")
    return {"inputs": inputs}


def _validate_base_identity(identity: object, d: Path) -> dict:
    if not isinstance(identity, dict) or not isinstance(identity.get("inputs"), dict):
        raise ValueError(f"invalid sweep base identity in {d}")
    inputs = identity["inputs"]
    if len(inputs) not in {1, 2} or not ({"profile", "prompts"} & set(inputs)):
        raise ValueError(f"invalid sweep workload inputs in {d}")
    if any(key not in {"profile", "prompts", "timestamps"} for key in inputs):
        raise ValueError(f"unknown sweep workload input in {d}")
    for key, metadata in inputs.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid sweep {key} identity in {d}")
        name = metadata.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"invalid sweep {key} input name in {d}")
        _identity_digest(metadata.get("sha256"),
                         f"base_identity.inputs.{key}.sha256", d)
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid sweep {key} input size in {d}")
    return inputs


def _expected_rung_identity(base_config: dict, base_identity: dict,
                            rate: float) -> tuple[dict, str]:
    from .runner import RunConfig, _effective_config, _resolved_workload_id

    cfg = copy.deepcopy(base_config)
    cfg.update(
        qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, out_dir=f"rate_{rate_label(rate)}",
        title=(f"{base_config['title']} @ {rate_label(rate)} "
               "requests/second"))
    rc = RunConfig(**cfg)
    effective = _effective_config(rc, rc)
    workload_id = _resolved_workload_id(rc, base_identity["inputs"])
    return effective, workload_id


def _validate_source_compatibility(manifest: dict, base_config: dict,
                                   base_identity: dict, rate: float,
                                   d: Path) -> None:
    expected_config, expected_workload = _expected_rung_identity(
        base_config, base_identity, rate)
    actual_config = manifest.get("effective_config")
    if not isinstance(actual_config, dict) \
            or manifest.get("effective_config_sha256") != canonical_sha256(
                actual_config):
        raise ValueError(f"sweep rung effective config digest is invalid: {d}")
    if actual_config != expected_config:
        raise ValueError(
            f"sweep rung effective config does not match the sealed base and "
            f"rate {rate:g}: {d}")
    if manifest.get("workload_id") != expected_workload:
        raise ValueError(
            f"sweep rung workload_id does not match its sealed config: {d}")
    expected_inputs = base_identity["inputs"]
    actual_inputs = manifest.get("inputs")
    if not isinstance(actual_inputs, dict) or set(actual_inputs) != set(expected_inputs):
        raise ValueError(f"sweep rung workload inputs do not match the base: {d}")
    for key, expected in expected_inputs.items():
        actual = actual_inputs.get(key)
        if not isinstance(actual, dict) or any(
                actual.get(field) != expected[field]
                for field in ("sha256", "bytes")):
            raise ValueError(f"sweep rung {key} bytes do not match the base: {d}")
    endpoint = expected_config["endpoint"]
    for field, expected in (
            ("endpoint_base_url", endpoint.get("base_url")),
            ("endpoint_path", endpoint.get("path")),
            ("endpoint_model", endpoint.get("model"))):
        if manifest.get(field) != expected:
            raise ValueError(f"sweep rung {field} does not match the base: {d}")
    expected_mode = "prompts" if base_config.get("prompts_file") else "profile"
    if manifest.get("input_mode") != expected_mode:
        raise ValueError(f"sweep rung input mode does not match the base: {d}")
    primary = expected_inputs[expected_mode]
    if manifest.get("profile_sha256") != primary["sha256"]:
        raise ValueError(f"sweep rung primary input digest does not match: {d}")


def _validate_rung_record(record: object, source: dict, summary: dict,
                          source_path: Path, aggregate: Path) -> None:
    """Prove that a headline row is the projection of its bound summary."""
    from .metrics import _verdict

    if not isinstance(record, dict):
        raise ValueError(f"invalid sweep rung record in {aggregate}")
    position = source["position"]
    if record.get("source_position") != position:
        raise ValueError(f"sweep rung/source position mismatch in {aggregate}")
    rate = record.get("rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
            or float(rate) != float(source["rate_requests_per_second"]):
        raise ValueError(f"sweep rung/source rate mismatch in {aggregate}")
    shown_value = record.get("dir")
    if not isinstance(shown_value, str) or not shown_value.strip():
        raise ValueError(f"invalid sweep rung report directory in {aggregate}")
    shown_dir = Path(shown_value)
    if shown_dir.is_absolute() or ".." in shown_dir.parts \
            or shown_dir == Path("."):
        raise ValueError(f"unsafe sweep rung report directory in {aggregate}")
    if shown_dir.as_posix() != source["relative_path"]:
        raise ValueError(f"sweep rung report directory mismatch in {aggregate}")
    if source_path.resolve(strict=True) != \
            (aggregate / shown_dir).resolve(strict=True):
        raise ValueError(f"sweep rung path escapes aggregate in {aggregate}")

    current_contract = "state" in record
    if current_contract:
        decision = classify_sweep_rung(summary)
        kind, text = decision["kind"], decision["text"]
        first_p50 = decision["latency_p50"]
        first_p95 = decision["latency_p95"]
        e2e_p50 = decision["e2e_p50"]
    else:
        decision = None
        kind, text = _verdict(summary)
        first_p50 = (summary.get("ttft_ms") or {}).get("p50")
        first_p95 = (summary.get("ttft_ms") or {}).get("p95")
        e2e_p50 = (summary.get("e2e_ms") or {}).get("p50")
    expected = {
        "kind": kind,
        "text": text,
        "held": (summary.get("concurrency") or {}).get("in_flight_p50"),
        "achieved_rps": (summary.get("arrivals") or {}).get(
            "achieved_qps_overall"),
        "err": summary.get("error_rate"),
        "ttft_p50": first_p50,
        "ttft_p95": first_p95,
        "e2e_p50": e2e_p50,
        "request_rows": source["request_rows"],
        "replay_rows": source["replay_rows"],
        "calibration_rows": source["calibration_rows"],
        "sizing_rows": source["sizing_rows"],
        "preflight_rows": source["preflight_rows"],
        "probe_rows": source["probe_rows"],
        "other_rows": source["other_rows"],
        "unknown_attempt_rows": source["unknown_attempt_rows"],
    }
    if decision is not None:
        expected.update(decision)
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"sweep rung {position} {field} disagrees with manifest-bound "
                f"summary.json in {aggregate}")
    wall = record.get("wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) < 0:
        raise ValueError(f"invalid sweep rung wall time in {aggregate}")


class SweepArtifacts:
    """Exclusive sweep directory and its pending source-evidence chain."""

    def __init__(self, path: Path, dir_fd: int, artifact_id: str,
                 created_at: float, base_text: str, base_metadata: dict,
                 source_state: dict, base_config: dict,
                 base_identity: dict):
        self.path = path
        self._dir_fd = dir_fd
        self.artifact_id = artifact_id
        self.created_at = created_at
        self._base_text = base_text
        self._base_metadata = base_metadata
        self._source_state = source_state
        self._base_config = base_config
        self._base_identity = base_identity
        self._sources: list[tuple[Path, dict, dict]] = []
        self._source_inodes: dict[tuple[int, int], Path] = {}
        self._artifact_ids: dict[str, Path] = {}
        self._rates: set[float] = set()
        self._complete = False

    @classmethod
    def claim(cls, requested: str | Path, base_config: dict, *,
              identity_config: dict | None = None) -> "SweepArtifacts":
        from .runner import RunConfig
        import dataclasses

        if not isinstance(base_config, dict):
            raise ValueError("sweep base config must be an object")
        # Validate a private copy because RunConfig normalizes legacy fields.
        # The bytes sealed below remain exactly what the caller supplied.
        public_rc = RunConfig(**copy.deepcopy(base_config))
        identity_source = (base_config if identity_config is None
                           else identity_config)
        identity_rc = RunConfig(**copy.deepcopy(identity_source))
        public_value = dataclasses.asdict(public_rc)
        identity_value = dataclasses.asdict(identity_rc)
        for field in ("profile_path", "prompts_file", "timestamps_file"):
            public_path = public_value.pop(field)
            identity_path = identity_value.pop(field)
            if bool(public_path) != bool(identity_path) or (
                    public_path is not None
                    and Path(public_path).name != Path(identity_path).name):
                raise ValueError(
                    "sweep identity input names do not match its public config")
        if public_value != identity_value:
            raise ValueError(
                "sweep identity config may differ only by frozen input paths")
        created_at = time.time()
        artifact_id = f"sweep-{uuid.uuid4().hex}"
        # Capture Git/source identity before the output path exists, otherwise
        # a default results/ path inside the checkout makes its own run dirty.
        source_state = snapshot_source_state(Path(__file__).parent)
        base_identity = _capture_base_identity(identity_source)
        path, dir_fd = _claim_dir(Path(requested), artifact_id, created_at)
        try:
            safe_config = redact_secrets(base_config)
            base_text = strict_json_dumps(safe_config, indent=2) + "\n"
            metadata = _atomic_text(
                dir_fd, "sweep-base-config.json", base_text)
            return cls(path, dir_fd, artifact_id, created_at,
                       base_text, metadata, source_state, safe_config,
                       base_identity)
        except Exception:
            os.close(dir_fd)
            raise

    def add_rung(self, rate: float, run_dir: str | Path,
                 expected_summary: dict | None = None) -> tuple[dict, int]:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f"invalid sweep rung rate: {rate!r}")
        value = float(rate)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid sweep rung rate: {rate!r}")
        if value in self._rates:
            raise ValueError(f"duplicate sweep rung rate: {value:g}")
        d = Path(run_dir)
        summary, source = _verified_run_snapshot(d, len(self._sources), value)
        combined_rows = source["request_rows"] + sum(
            existing[1]["request_rows"] for existing in self._sources)
        if combined_rows > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
            raise ValueError(
                f"sweep sources declare {combined_rows:,} request rows, "
                f"above the exact-analysis resource envelope of "
                f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}")
        run_manifest = _strict_object(
            _read_regular_bytes(d / "manifest.json"), "manifest.json",
            d / "manifest.json")
        _validate_source_compatibility(
            run_manifest, self._base_config, self._base_identity, value, d)
        if expected_summary is not None and summary != expected_summary:
            raise ValueError(
                f"runner summary disagrees with manifest-bound summary.json: {d}")
        identity = d.stat()
        inode = (identity.st_dev, identity.st_ino)
        if inode in self._source_inodes:
            raise ValueError(
                f"duplicate sweep rung directory: {d} is the same as "
                f"{self._source_inodes[inode]}")
        artifact_id = source["artifact_id"]
        if artifact_id in self._artifact_ids:
            raise ValueError(
                f"duplicate input artifact_id {artifact_id!r}: {d} and "
                f"{self._artifact_ids[artifact_id]}")
        try:
            relative = d.resolve(strict=True).relative_to(
                self.path.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(
                f"sweep rung must be inside its aggregate directory: {d}") from exc
        if relative == Path("."):
            raise ValueError("the sweep aggregate cannot be its own rung")
        source["relative_path"] = relative.as_posix()
        safe_d = _nested_regular_dir(self.path, source["relative_path"])
        safe_identity = safe_d.stat()
        if (safe_identity.st_dev, safe_identity.st_ino) != inode:
            raise ValueError(f"sweep rung path changed while being added: {d}")
        self._source_inodes[inode] = d
        self._artifact_ids[artifact_id] = d
        self._rates.add(value)
        self._sources.append((d, source, summary))
        return summary, source["position"]

    def rung_accounting(self, position: int) -> dict:
        """Return manifest-bound phase counts for one already-added rung."""
        if isinstance(position, bool) or not isinstance(position, int) \
                or not 0 <= position < len(self._sources):
            raise ValueError(f"invalid sweep source position: {position!r}")
        source = self._sources[position][1]
        return {
            field: source[field] for field in (
                "request_rows", "replay_rows", "calibration_rows",
                "sizing_rows", "preflight_rows", "probe_rows", "other_rows",
                "unknown_attempt_rows")
        }

    def pooled_quota_evidence(self) -> dict:
        """Return full-ladder observed quota evidence before publication."""
        return _sweep_quota_evidence(
            [source[0] for source in self._sources],
            [source[2] for source in self._sources],
            self._base_config,
        )

    def seal(self, sweep_text: str, rungs: list[dict], *,
             exit_code: int, highest_held_rate: float | None,
             report_context: dict) -> Path:
        if self._complete or self._dir_fd < 0:
            raise RuntimeError("sweep artifact is already closed")
        if _read_regular_bytes(self.path / "sweep-base-config.json") \
                != self._base_text.encode("utf-8"):
            raise ValueError("sweep base config changed before sealing")

        # Re-verify every source binding immediately before publication. This
        # detects a run that was replaced or edited after it joined the sweep.
        sources = []
        source_summaries = []
        source_paths = []
        for position, (d, expected, _summary) in enumerate(self._sources):
            d = _nested_regular_dir(self.path, expected["relative_path"])
            current_summary, current = _verified_run_snapshot(
                d, position, expected["rate_requests_per_second"])
            current_manifest = _strict_object(
                _read_regular_bytes(d / "manifest.json"), "manifest.json",
                d / "manifest.json")
            _validate_source_compatibility(
                current_manifest, self._base_config, self._base_identity,
                expected["rate_requests_per_second"], d)
            current["relative_path"] = expected["relative_path"]
            if current != expected:
                raise ValueError(
                    f"sweep rung changed before aggregate sealing: {d}")
            sources.append(current)
            source_summaries.append(current_summary)
            source_paths.append(d)

        source_positions = [r.get("source_position") for r in rungs
                            if r.get("source_position") is not None]
        if source_positions != list(range(len(sources))):
            raise ValueError(
                "sweep rung records must reference each verified source "
                "exactly once and in order")
        for r in rungs:
            position = r.get("source_position")
            if position is None:
                claimed = [r.get(key) for key in (
                    "held", "achieved_rps", "err", "ttft_p50", "ttft_p95",
                    "e2e_p50", "success_rate_target",
                    "success_rate_actual", "success_rate_wilson_lower_95",
                    "success_rate_statistically_demonstrated",
                    "request_start_lateness_p95", "dispatch_lag_p95",
                    "response_identity_status",
                    "endpoint_metadata_stability",
                    "runtime_quota_admission_status",
                    "runtime_quota_guard_id",
                    "transport_connection_policy_id",
                    "production_connection_policy_declared",
                    "production_connection_policy_match",
                    "production_comparability_warning",
                    "production_connection_policy_assurance",
                    "transport_parity_status",
                    "request_rows", "replay_rows",
                    "calibration_rows", "sizing_rows", "preflight_rows",
                    "probe_rows", "other_rows", "unknown_attempt_rows")]
                if r.get("kind") != "invalid" or any(v is not None for v in claimed):
                    raise ValueError(
                        "an unsealed sweep attempt cannot contribute a verdict "
                        "or measurement")
            else:
                if isinstance(position, bool) or not isinstance(position, int):
                    raise ValueError("sweep source positions must be integers")
                _validate_rung_record(
                    r, sources[position], source_summaries[position],
                    source_paths[position], self.path)

        expected_endpoint = self._base_config["endpoint"]["path"]
        context = _validated_report_context(
            report_context, self.path, expected_endpoint=expected_endpoint,
            rung_count=len(rungs))
        outcome = sweep_outcome(rungs, context["preflight"])
        if highest_held_rate != outcome["highest_held_rate"]:
            raise ValueError("highest held rate disagrees with manifest-bound rungs")
        if isinstance(exit_code, bool) or exit_code != outcome["exit_code"]:
            raise ValueError("sweep exit code disagrees with manifest-bound rungs")

        if context.get("sweep_quota_evidence") is not None:
            rederived_quota = _sweep_quota_evidence(
                source_paths, source_summaries, self._base_config)
            if context["sweep_quota_evidence"] != rederived_quota:
                raise ValueError(
                    "sweep-level quota evidence disagrees with all bound "
                    "source request rows")
            local = rederived_quota.get("runtime_quota_admission") or {}
            if local.get("status") == "invalid_evidence":
                raise ValueError(
                    "quota-aware sweep did not retain one command-level "
                    "runtime guard across every bound source")
        if not outcome["unverified"]:
            if sum(source["preflight_rows"] for source in sources) \
                    != context["preflight"]["attempted"]:
                raise ValueError(
                    "manifest-bound preflight rows disagree with report context")
            if sum(source["probe_rows"] for source in sources) \
                    != context["preflight"]["reasoning_probe_requests"]:
                raise ValueError(
                    "manifest-bound probe rows disagree with report context")
            if any(source["preflight_rows"] or source["probe_rows"]
                   for source in sources[1:]):
                raise ValueError(
                    "preflight/probe traffic may be attached only to the first rung")
        canonical_report = render_sweep_report(rungs, context)
        if sweep_text != canonical_report:
            raise ValueError(
                "sweep.md is not the canonical report derived from rung evidence")

        sweep_metadata = _atomic_text(self._dir_fd, "sweep.md", sweep_text)
        sweep_html = render_sweep_html(rungs, context, self.artifact_id)
        sweep_html_metadata = _atomic_text(
            self._dir_fd, "sweep.html", sweep_html)
        source_state = self._source_state
        source_commit = source_state.get("git_commit")
        source_tree = source_state.get("source_tree_sha256")
        reconstructible = bool(
            source_state.get("git_dirty") is False
            and isinstance(source_commit, str) and source_commit.strip()
            and isinstance(source_tree, str) and len(source_tree) == 64)
        manifest = {
            "manifest_schema_version": 3,
            "sweep_decision_schema_version": _SWEEP_DECISION_SCHEMA_VERSION,
            "sweep_renderer_schema_version": _SWEEP_RENDERER_SCHEMA_VERSION,
            "artifact_type": "sweep",
            "artifact_id": self.artifact_id,
            "artifact_created_at_utc": datetime.fromtimestamp(
                self.created_at, timezone.utc).isoformat(),
            "artifact_created_at_unix": self.created_at,
            "operation": "rate_sweep",
            "harness_version": __version__,
            "git_commit": source_state.get("git_commit"),
            "git_dirty": source_state.get("git_dirty"),
            "source": source_state,
            "source_tree_sha256": source_state.get("source_tree_sha256"),
            "generator_source_reconstructible": reconstructible,
            "base_identity": self._base_identity,
            "report_context": context,
            "input_count": len(sources),
            "rung_count": len(rungs),
            "sources": sources,
            "rungs": redact_secrets(rungs),
            "highest_held_rate_requests_per_second": highest_held_rate,
            "highest_sla_passing_tested_rate_requests_per_second": outcome[
                "highest_sla_passing_tested_rate"],
            "highest_achieved_rate_at_sla_passing_rung_requests_per_second":
                outcome["highest_achieved_rate_at_sla_passing_rung"],
            "requested_rate_at_highest_achieved_sla_passing_rung_requests_per_second":
                outcome[
                    "requested_rate_at_highest_achieved_sla_passing_rung"],
            "achieved_rate_at_highest_requested_sla_passing_rung_requests_per_second":
                outcome[
                    "achieved_rate_at_highest_requested_sla_passing_rung"],
            "capacity_conclusion": outcome["capacity_conclusion"],
            "boundary_status": outcome["boundary_status"],
            "exit_code": int(exit_code),
            "sweep_valid": not outcome["invalid"],
            "invalid_reasons": outcome["invalid_reasons"],
            "artifacts": {
                "sweep-base-config.json": self._base_metadata,
                "sweep.md": sweep_metadata,
                "sweep.html": sweep_html_metadata,
            },
        }
        manifest_text = strict_json_dumps(manifest, indent=2) + "\n"
        manifest_metadata = _atomic_text(
            self._dir_fd, "manifest.json", manifest_text)
        completion_text = strict_json_dumps({
            "artifact_id": self.artifact_id,
            "artifact_type": "sweep",
            "status": "complete",
            "completed_at_unix": time.time(),
            "manifest_sha256": manifest_metadata["sha256"],
            "manifest_bytes": manifest_metadata["bytes"],
        }) + "\n"
        _atomic_text(self._dir_fd, _WRITING_MARKER, completion_text)
        os.replace(_WRITING_MARKER, _COMPLETE_MARKER,
                   src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
        _fsync_fd(self._dir_fd)
        self._complete = True
        self.close()
        _fsync_directory(self.path.parent)
        verify_sweep_output(self.path)
        return self.path

    def close(self) -> None:
        if self._dir_fd >= 0:
            os.close(self._dir_fd)
            self._dir_fd = -1


def verify_sweep_output(out_dir: str | Path) -> dict:
    """Verify a sweep aggregate's complete marker, manifest and artifacts."""
    d = Path(out_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"sweep directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"sweep directory is not a regular directory: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(f"sweep is still being written: {d}")
    for name in (_COMPLETE_MARKER, "manifest.json", "sweep.md", "sweep.html",
                 "sweep-base-config.json"):
        _require_regular(d / name, name)
    completion_raw = _read_regular_bytes(d / _COMPLETE_MARKER)
    completion = _strict_object(
        completion_raw, "completion marker", d / _COMPLETE_MARKER)
    manifest_raw = _read_regular_bytes(d / "manifest.json")
    manifest = _strict_object(manifest_raw, "manifest.json", d / "manifest.json")
    if manifest.get("manifest_schema_version") != 3 \
            or manifest.get("artifact_type") != "sweep":
        raise ValueError(f"unsupported sweep manifest in {d}")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"invalid sweep artifact_id in {d}")
    if completion.get("status") != "complete" \
            or completion.get("artifact_type") != "sweep" \
            or completion.get("artifact_id") != artifact_id:
        raise ValueError(f"completion marker and sweep manifest disagree in {d}")
    actual_manifest = hashlib.sha256(manifest_raw).hexdigest()
    actual_bytes = len(manifest_raw)
    expected_manifest = _identity_digest(
        completion.get("manifest_sha256"),
        "completion marker manifest_sha256", d)
    if not hmac.compare_digest(actual_manifest, expected_manifest):
        raise ValueError(f"manifest SHA-256 mismatch for sweep {d}")
    declared_bytes = completion.get("manifest_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) \
            or declared_bytes != actual_bytes:
        raise ValueError(f"manifest byte count mismatch for sweep {d}")
    _verify_artifacts(
        d, manifest, ("sweep-base-config.json", "sweep.md", "sweep.html"))
    declarations = _artifact_declarations(manifest, d)
    base_raw = _read_regular_bytes(d / "sweep-base-config.json")
    report_raw = _read_regular_bytes(d / "sweep.md")
    html_raw = _read_regular_bytes(d / "sweep.html")
    for name, raw in (("sweep-base-config.json", base_raw),
                      ("sweep.md", report_raw),
                      ("sweep.html", html_raw)):
        expected = declarations[name]
        if not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(), expected["sha256"]):
            raise ValueError(f"artifact SHA-256 mismatch for {d / name}")
        if len(raw) != expected["bytes"]:
            raise ValueError(f"artifact byte count mismatch for {d / name}")
    base_config = _strict_object(
        base_raw, "sweep-base-config.json", d / "sweep-base-config.json")
    if redact_secrets(base_config) != base_config:
        raise ValueError(f"sweep base config contains unredacted secrets in {d}")
    from .runner import RunConfig
    try:
        RunConfig(**copy.deepcopy(base_config))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sweep base config in {d}: {exc}") from exc
    base_identity = manifest.get("base_identity")
    _validate_base_identity(base_identity, d)
    expected_endpoint = base_config["endpoint"]["path"]
    sources = manifest.get("sources")
    if not isinstance(sources, list) \
            or manifest.get("input_count") != len(sources):
        raise ValueError(f"invalid sources in sweep manifest for {d}")
    seen = set()
    rates = set()
    source_summaries = []
    source_paths = []
    declared_source_rows = 0
    for position, source in enumerate(sources):
        _validate_source_shape(source, position, d)
        declared_source_rows += source["request_rows"]
        if declared_source_rows > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
            raise ValueError(
                f"sweep manifest declares {declared_source_rows:,} request "
                f"rows, above the exact-analysis resource envelope of "
                f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,} in {d}")
        artifact = source["artifact_id"]
        rate = float(source["rate_requests_per_second"])
        if artifact in seen:
            raise ValueError(f"duplicate input artifact_id in sweep manifest for {d}")
        if rate in rates:
            raise ValueError(f"duplicate input rate in sweep manifest for {d}")
        seen.add(artifact)
        rates.add(rate)
        source_path = _nested_regular_dir(d, source["relative_path"])
        summary, current = _verified_run_snapshot(source_path, position, rate)
        source_manifest = _strict_object(
            _read_regular_bytes(source_path / "manifest.json"),
            "manifest.json", source_path / "manifest.json")
        _validate_source_compatibility(
            source_manifest, base_config, base_identity, rate, source_path)
        current["relative_path"] = source["relative_path"]
        if current != source:
            raise ValueError(
                f"manifest-bound sweep source changed or was replaced: {source_path}")
        source_summaries.append(summary)
        source_paths.append(source_path)
    rungs = manifest.get("rungs")
    if not isinstance(rungs, list) or manifest.get("rung_count") != len(rungs):
        raise ValueError(f"invalid rung records in sweep manifest for {d}")
    referenced = [r.get("source_position") for r in rungs
                  if isinstance(r, dict) and r.get("source_position") is not None]
    if referenced != list(range(len(sources))):
        raise ValueError(f"sweep rung/source references disagree in {d}")
    for record in rungs:
        if not isinstance(record, dict):
            raise ValueError(f"invalid sweep rung record in {d}")
        position = record.get("source_position")
        if position is None:
            claimed = [record.get(key) for key in (
                "held", "achieved_rps", "err", "ttft_p50", "ttft_p95",
                "e2e_p50", "success_rate_target",
                "success_rate_actual", "success_rate_wilson_lower_95",
                "success_rate_statistically_demonstrated",
                "request_start_lateness_p95", "dispatch_lag_p95",
                "response_identity_status", "endpoint_metadata_stability",
                "runtime_quota_admission_status", "runtime_quota_guard_id",
                "transport_connection_policy_id",
                "production_connection_policy_declared",
                "production_connection_policy_match",
                "production_comparability_warning",
                "production_connection_policy_assurance",
                "transport_parity_status",
                "request_rows", "replay_rows",
                "calibration_rows", "sizing_rows", "preflight_rows",
                "probe_rows", "other_rows", "unknown_attempt_rows")]
            if record.get("kind") != "invalid" or any(
                    value is not None for value in claimed):
                raise ValueError(
                    f"unsealed sweep attempt claims a measurement in {d}")
        else:
            if isinstance(position, bool) or not isinstance(position, int):
                raise ValueError(f"sweep source positions must be integers in {d}")
            _validate_rung_record(
                record, sources[position], source_summaries[position],
                source_paths[position], d)
    context = _validated_report_context(
        manifest.get("report_context"), d,
        expected_endpoint=expected_endpoint, rung_count=len(rungs))
    outcome = sweep_outcome(rungs, context["preflight"])
    if manifest.get("highest_held_rate_requests_per_second") \
            != outcome["highest_held_rate"]:
        raise ValueError(f"highest held rate disagrees with sweep rungs in {d}")
    if manifest.get("sweep_decision_schema_version") != \
            _SWEEP_DECISION_SCHEMA_VERSION:
        raise ValueError(f"unsupported sweep decision schema in {d}")
    if manifest.get("sweep_renderer_schema_version") != \
            _SWEEP_RENDERER_SCHEMA_VERSION:
        raise ValueError(f"unsupported sweep renderer schema in {d}")
    if manifest.get(
            "highest_sla_passing_tested_rate_requests_per_second") \
            != outcome["highest_sla_passing_tested_rate"]:
        raise ValueError(
            f"highest SLA-passing tested rate disagrees with sweep rungs in {d}")
    if manifest.get(
            "highest_achieved_rate_at_sla_passing_rung_requests_per_second") \
            != outcome["highest_achieved_rate_at_sla_passing_rung"]:
        raise ValueError(
            f"highest achieved passing rate disagrees with sweep rungs in {d}")
    if manifest.get(
            "requested_rate_at_highest_achieved_sla_passing_rung_requests_per_second") \
            != outcome[
                "requested_rate_at_highest_achieved_sla_passing_rung"]:
        raise ValueError(
            f"requested rung for highest achieved passing rate disagrees "
            f"with sweep rungs in {d}")
    if manifest.get(
            "achieved_rate_at_highest_requested_sla_passing_rung_requests_per_second") \
            != outcome[
                "achieved_rate_at_highest_requested_sla_passing_rung"]:
        raise ValueError(
            f"achieved rate at highest requested passing rung disagrees "
            f"with sweep rungs in {d}")
    if manifest.get("capacity_conclusion") != outcome["capacity_conclusion"]:
        raise ValueError(f"capacity conclusion disagrees with sweep rungs in {d}")
    if manifest.get("boundary_status") != outcome["boundary_status"]:
        raise ValueError(f"boundary status disagrees with sweep rungs in {d}")
    if manifest.get("exit_code") != outcome["exit_code"]:
        raise ValueError(f"exit code disagrees with sweep rungs in {d}")
    if manifest.get("sweep_valid") is not (not outcome["invalid"]):
        raise ValueError(f"sweep_valid disagrees with rung evidence in {d}")
    if manifest.get("invalid_reasons") != outcome["invalid_reasons"]:
        raise ValueError(f"invalid reasons disagree with rung evidence in {d}")
    if context.get("sweep_quota_evidence") is not None:
        rederived_quota = _sweep_quota_evidence(
            source_paths, source_summaries, base_config)
        if context["sweep_quota_evidence"] != rederived_quota:
            raise ValueError(
                f"sweep-level quota evidence disagrees with source rows in {d}")
        local = rederived_quota.get("runtime_quota_admission") or {}
        if local.get("status") == "invalid_evidence":
            raise ValueError(
                f"quota-aware sweep did not retain one command-level runtime "
                f"guard across every bound source in {d}")
    if not outcome["unverified"]:
        if sum(source["preflight_rows"] for source in sources) \
                != context["preflight"]["attempted"]:
            raise ValueError(
                f"manifest-bound preflight rows disagree with report context in {d}")
        if sum(source["probe_rows"] for source in sources) \
                != context["preflight"]["reasoning_probe_requests"]:
            raise ValueError(
                f"manifest-bound probe rows disagree with report context in {d}")
        if any(source["preflight_rows"] or source["probe_rows"]
               for source in sources[1:]):
            raise ValueError(
                f"preflight/probe traffic is attached after the first rung in {d}")
    expected_report = render_sweep_report(rungs, context).encode("utf-8")
    if report_raw != expected_report:
        raise ValueError(
            f"sweep.md is not the canonical report derived from evidence in {d}")
    expected_html = render_sweep_html(
        rungs, context, artifact_id).encode("utf-8")
    if html_raw != expected_html:
        raise ValueError(
            f"sweep.html is not the canonical report derived from evidence in {d}")
    return manifest
