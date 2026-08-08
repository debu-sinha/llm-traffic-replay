"""Summaries and the honesty block.

Every latency table is printed WITH the context that decides whether it can
be believed: cached prompt-token fraction (endpoint-reported), achieved
arrival rate vs scheduled, wire lateness, error rate, and token
targeting error. A good p50 at the wrong cached-token fraction is a fake
result; this
module makes the pairing unavoidable.
"""
from __future__ import annotations

from datetime import datetime
import html
import json
import math
import time
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import (
    FINAL_REQUESTS,
    RunArtifacts,
    canonical_sha256,
    redact_secrets as _redact_secrets,
    sanitize_display_text,
    sanitize_title,
    sha256_bytes,
    snapshot_source_state,
    strict_json_dumps,
)

PCTS = (50, 90, 95, 99)


def _external_report_context(summary: dict, value: dict | None) -> dict | None:
    """Validate the explicit trust context for a verified derivative view.

    Normal source reports never supply this value and therefore remain
    VERIFY_REQUIRED. Only the external receipt builder supplies it, after it
    has verified the manifest and canonical artifacts. Reject unknown fields
    and unrelated decisions so this presentation hook cannot become a general
    way to paint an arbitrary summary green.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("verification_context must be a dict or None")
    required = {
        "view_label", "receipt_id", "source_artifact_id",
        "source_manifest_sha256", "verifier_version", "verified_at_utc",
        "assurance", "decision", "source_reproducibility",
        "verifier_reproducibility",
    }
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError("invalid verification_context: " + "; ".join(detail))
    if value.get("view_label") != "EXTERNAL VERIFIED VIEW":
        raise ValueError(
            "verification_context.view_label must be EXTERNAL VERIFIED VIEW")
    normalized = {}
    for field in (
            "receipt_id", "source_artifact_id", "verifier_version",
            "verified_at_utc", "assurance"):
        raw = value.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"verification_context.{field} must be non-empty")
        normalized[field] = sanitize_display_text(raw)
    digest = value.get("source_manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64 \
            or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError(
            "verification_context.source_manifest_sha256 must be SHA-256")
    normalized["source_manifest_sha256"] = digest.lower()
    try:
        verified_at = datetime.fromisoformat(
            normalized["verified_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "verification_context.verified_at_utc must be ISO-8601") from exc
    if verified_at.tzinfo is None:
        raise ValueError(
            "verification_context.verified_at_utc must include a timezone")
    assurance_lower = normalized["assurance"].lower()
    if "sha-256" not in assurance_lower \
            or "not a digital signature" not in assurance_lower:
        raise ValueError(
            "verification_context.assurance must state internal SHA-256 "
            "consistency and that it is not a digital signature")

    def reproducibility_state(field: str) -> dict:
        state = value.get(field)
        if not isinstance(state, dict) or set(state) != {
                "code", "reason", "reason_codes"}:
            raise ValueError(
                f"verification_context.{field} must contain exactly code, "
                "reason, and reason_codes")
        code = state.get("code")
        reason = state.get("reason")
        reason_codes = state.get("reason_codes")
        if code not in {"PASS", "FAILED"}:
            raise ValueError(
                f"verification_context.{field}.code must be PASS or FAILED")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"verification_context.{field}.reason must be non-empty")
        if not isinstance(reason_codes, list) or any(
                not isinstance(item, str) or not item.strip()
                for item in reason_codes):
            raise ValueError(
                f"verification_context.{field}.reason_codes must be strings")
        if (code == "PASS") != (not reason_codes):
            raise ValueError(
                f"verification_context.{field} code and reason_codes "
                "disagree")
        return {
            "code": code,
            "reason": sanitize_display_text(reason),
            "reason_codes": [
                sanitize_display_text(item) for item in reason_codes],
        }

    source_reproducibility = reproducibility_state(
        "source_reproducibility")
    verifier_reproducibility = reproducibility_state(
        "verifier_reproducibility")

    decision = value.get("decision")
    if not isinstance(decision, dict) \
            or decision.get("decision_schema_version") != 1:
        raise ValueError("verification_context.decision is invalid")
    evidence = decision.get("evidence_integrity")
    if not isinstance(evidence, dict) or evidence.get("code") != "VERIFIED":
        raise ValueError(
            "verification_context.decision must carry VERIFIED integrity")
    from .report_decision import IntegrityContext, build_report_decision
    baseline = build_report_decision(
        summary,
        IntegrityContext(
            "verified",
            "The external verifier established internal hash consistency; "
            "this is not a digital signature.",
        ),
    )
    for key in (
            "measurement_validity", "customer_sla", "quota_state",
            "tested_load"):
        if decision.get(key) != baseline.get(key):
            raise ValueError(
                f"verification_context.decision.{key} does not match summary")
    capacity = decision.get("endpoint_capacity")
    baseline_capacity = baseline.get("endpoint_capacity")
    required_provenance_gate = None
    if source_reproducibility["code"] == "FAILED":
        required_provenance_gate = "SOURCE_NOT_RECONSTRUCTIBLE"
    elif verifier_reproducibility["code"] == "FAILED":
        required_provenance_gate = "VERIFIER_SOURCE_NOT_RECONSTRUCTIBLE"
    if required_provenance_gate is not None \
            and isinstance(baseline_capacity, dict) \
            and baseline_capacity.get("code") == "HELD_AT_TESTED_LOAD" \
            and capacity == baseline_capacity:
        raise ValueError(
            "verification_context.decision cannot claim held capacity when "
            "source or verifier reproducibility failed")
    if capacity != baseline_capacity:
        reason_codes = (capacity or {}).get("reason_codes")
        provenance_gate = (
            required_provenance_gate is not None
            and isinstance(reason_codes, list)
            and required_provenance_gate in reason_codes)
        if not (
                isinstance(capacity, dict)
                and capacity.get("code") == "INCONCLUSIVE"
                and isinstance(baseline_capacity, dict)
                and baseline_capacity.get("code") == "HELD_AT_TESTED_LOAD"
                and provenance_gate
                and capacity.get("endpoint_ceiling_established") is False
                and capacity.get("provider_headroom_established") is False):
            raise ValueError(
                "verification_context.decision.endpoint_capacity does not "
                "match summary or an allowed reconstructibility gate")
    normalized.update({
        "view_label": "EXTERNAL VERIFIED VIEW",
        "decision": decision,
        "source_reproducibility": source_reproducibility,
        "verifier_reproducibility": verifier_reproducibility,
    })
    return normalized


def _tcp_connect_floor(network_path: dict) -> float | None:
    """Read current network-path evidence, with legacy artifact support."""
    value = network_path.get("tcp_connect_min_ms")
    if value is None:
        value = network_path.get("rtt_ms")
    return value


def _wilson_lower_95(successes: int, total: int) -> float | None:
    """One-sided 95% Wilson lower confidence bound for a success fraction."""
    if total <= 0 or successes < 0 or successes > total:
        return None
    z = 1.6448536269514722
    observed = successes / total
    z2 = z * z
    center = observed + z2 / (2.0 * total)
    radius = z * math.sqrt(
        observed * (1.0 - observed) / total
        + z2 / (4.0 * total * total))
    return max(0.0, (center - radius) / (1.0 + z2 / total))


def _concurrency_block(results: list[dict], asked: int | None) -> dict | None:
    """How many requests were actually in flight, by exact interval overlap.

    Every request that reached the wire belongs in occupancy, including an
    HTTP error or a transport timeout. Current rows record finished_unix for
    that purpose; legacy successful rows can be reconstructed from their
    final-attempt service duration.

    Every start and end is swept, so the maximum is a true peak rather than
    the highest of a fixed number of samples. An earlier version sampled 41
    points and called the result a peak, which understated it whenever the
    peak fell between two samples. The percentiles are time weighted, which
    is the right statistic for occupancy: a level held for one second out of
    sixty should not count the same as one held for thirty.
    """
    # a retried row starts at its FIRST attempt but e2e_ms belongs to the
    # attempt that succeeded, so pairing them put the span up to
    # (connect_timeout + read_timeout) x retries before the request was
    # actually on the wire. the request occupied a worker for the whole
    # stretch, so the span runs from the first send to the end of the
    # attempt that finished.
    spans = []
    sent_n = sum(1 for r in results if _sent_at(r) is not None)
    for r in results:
        start = _sent_at(r)
        end = _completed_at(r)
        if start is None or end is None:
            continue
        spans.append((start, max(end, start)))
    spans = [(a, b) for a, b in spans if b > a]
    if len(spans) < 2:
        return None
    # the window is the middle of the LOAD interval, which is bounded by
    # send times. anchoring it on completions instead let a single straggler
    # stretch the span into its own drain: 100 one-second requests plus one
    # that took 1000 seconds put the whole real run inside the first 10
    # percent, and the reported concurrency collapsed to 1.
    first_send = min(a for a, _ in spans)
    last_send = max(a for a, _ in spans)
    if last_send <= first_send:
        return None
    lo = first_send + (last_send - first_send) * 0.2
    hi = first_send + (last_send - first_send) * 0.8
    if hi <= lo:
        lo, hi = first_send, last_send

    def _sweep(spans_in, w_lo, w_hi):
        ev: list[tuple[float, int]] = []
        for a, b in spans_in:
            a2, b2 = max(a, w_lo), min(b, w_hi)
            if b2 > a2:
                ev.append((a2, 1))
                ev.append((b2, -1))
        if not ev:
            return None, {}
        ev.sort()
        c = pk = 0
        # start at the window edge, not the first event, so idle time inside
        # the window counts as the zero it was. a six second window holding
        # one one-second request is p50 0, not p50 1.
        prev_t = w_lo if w_lo is not None else ev[0][0]
        acc: dict[int, float] = {}
        for t, d in ev:
            if t > prev_t:
                acc[c] = acc.get(c, 0.0) + (t - prev_t)
            c += d
            pk = max(pk, c)
            prev_t = t
        if w_hi is not None and w_hi > prev_t:
            acc[c] = acc.get(c, 0.0) + (w_hi - prev_t)
        return pk, acc

    # the peak is taken over the WHOLE run, since a burst during ramp up is
    # real load the endpoint carried. cropping it and still calling it a peak
    # understated it.
    true_peak, _ = _sweep(spans, min(a for a, _ in spans),
                          max(b for _, b in spans))

    # the SAME edge-aware sweep, over the measurement window. an earlier
    # version added the sweep and then used it only for the peak, leaving
    # the percentiles on a loop that began at the first event, so leading
    # and trailing idle time inside the window still went uncounted.
    peak, held = _sweep(spans, lo, hi)
    if not held:
        return None
    total = sum(held.values())
    if total <= 0:
        return None

    def _tw(q: float) -> float:
        run = 0.0
        for level in sorted(held):
            run += held[level]
            if run >= total * q:
                return float(level)
        return float(max(held))

    med = _tw(0.5)
    out = {
        "in_flight_p50": med,
        "in_flight_p95": _tw(0.95),
        "in_flight_max": float(true_peak or peak),
        "in_flight_max_in_window": float(peak),
        "measured_over": "sent request rows with a recorded completion time",
        "method": ("exact interval overlap. percentiles are time weighted "
                   "over the middle 60 percent of the LOAD interval, bounded "
                   "by send times so one straggler cannot stretch the "
                   "window. the maximum is a true peak over the whole run"),
        "sent_requests": sent_n,
        "measured_requests": len(spans),
        "coverage": (len(spans) / sent_n) if sent_n else None,
    }
    warnings = []
    if sent_n and len(spans) / sent_n < 0.99:
        warnings.append(
            f"completion time was available for only {len(spans)} of "
            f"{sent_n} requests that reached the wire, so occupancy is "
            "incomplete")
    if asked:
        # --concurrency is a sizing input used to derive an open-loop arrival
        # rate. It is not a closed-loop controller and therefore must never be
        # labeled as concurrency the run promised to hold.
        out["sizing_concurrency_requested"] = asked
        if med < asked * 0.8:
            warnings.append(
                f"the open-loop rate was sized from an unloaded estimate of "
                f"{asked} concurrent requests, while observed in-flight p50 "
                f"was {med:.0f}. {asked} was a sizing input, not a held "
                "concurrency target; describe this run by its achieved QPS "
                f"and observed occupancy {med:.0f}.")
        elif med > asked * 1.25:
            # the arrival rate is derived from UNLOADED service time. under
            # load the service time rises and in-flight rises with it, so
            # overshoot is the direction this design biases toward. warning
            # on only the other direction let a run labeled "30 concurrent"
            # that actually held 65 go out clean.
            warnings.append(
                f"the open-loop rate was sized from an unloaded estimate of "
                f"{asked} concurrent requests, while observed in-flight p50 "
                f"was {med:.0f}. service time rose under load, so occupancy "
                "exceeded the sizing estimate. describe this run by its "
                f"achieved QPS and observed occupancy {med:.0f}, not as "
                f"holding {asked} concurrent requests.")
    if warnings:
        out["warning"] = " ".join(warnings)
    return out


def _sent_at(r: dict) -> float | None:
    """When the client began sending this request.

    `t_send_unix` belongs to whichever attempt produced the result, so on a
    retried row it carries the endpoint's delay. `first_send_unix` is the
    first attempt, which is when the load was actually offered. Rows written
    by an older harness only have the former.
    """
    value = (r.get("first_send_unix") if "first_send_unix" in r
             else r.get("t_send_unix"))
    if _nonnegative_finite(value):
        return float(value)
    return None


def _completed_at(r: dict) -> float | None:
    """When a sent request stopped occupying a worker/connection.

    New artifacts carry an exact epoch for successes and failures. For old
    artifacts, reconstruct only from recorded clocks; never turn a missing
    failure duration into zero.
    """
    start = _sent_at(r)
    if start is None:
        return None
    if "finished_unix" in r:
        value = r.get("finished_unix")
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(float(value)):
            return max(float(value), start)
        return None
    first_attempt = r.get("first_attempt_unix")
    caller = r.get("caller_e2e_ms")
    queue = r.get("queue_wait_ms")
    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
           and math.isfinite(float(v)) for v in (first_attempt, caller)):
        worker_ms = max(float(caller) - float(queue or 0.0), 0.0)
        return max(float(first_attempt) + worker_ms / 1000.0, start)
    service = r.get("e2e_ms")
    last = r.get("t_send_unix")
    if isinstance(service, (int, float)) and not isinstance(service, bool) \
            and math.isfinite(float(service)):
        base = (float(last) if isinstance(last, (int, float))
                and not isinstance(last, bool) else start)
        return max(base + max(float(service), 0.0) / 1000.0, start)
    return None


def _nonnegative_finite(value) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)) and value >= 0)


def _protocol_clean_success(row: dict) -> bool:
    """Whether a response is safe to use as successful protocol evidence.

    Current artifacts always carry ``stream_complete``.  Legacy artifacts do
    not, so absence is tolerated for backwards-compatible descriptive views;
    callers that make a completeness claim must separately require the field
    to be present.  An explicitly incomplete or corrupt stream is never
    eligible, even when an older client left ``ok`` set after seeing content.
    """
    parse_errors = row.get("parse_errors", 0)
    if (not isinstance(parse_errors, int)
            or isinstance(parse_errors, bool)
            or parse_errors < 0):
        return False
    if not row.get("ok") or parse_errors != 0:
        return False
    return ("stream_complete" not in row
            or row.get("stream_complete") is True)


def _usage_is_trustworthy(row: dict) -> bool:
    """Validate the recognized token-accounting invariants on a clean row."""
    if not _protocol_clean_success(row):
        return False
    prompt = row.get("prompt_tokens")
    completion = row.get("completion_tokens")
    if not _nonnegative_finite(prompt) or not _nonnegative_finite(completion):
        return False
    cached = row.get("cached_tokens")
    if cached is not None and (
            not _nonnegative_finite(cached) or float(cached) > float(prompt)):
        return False
    reasoning = row.get("reasoning_tokens")
    if reasoning is not None and (
            not _nonnegative_finite(reasoning)
            or float(reasoning) > float(completion)):
        return False
    total = row.get("total_tokens")
    if total is not None and (
            not _nonnegative_finite(total)
            or float(total) != float(prompt) + float(completion)):
        return False
    return True


def _rolling_peak(entries: list[tuple[float, float]],
                  window_seconds: float) -> dict:
    """Exact maximum sum over trailing half-open time windows.

    An event exactly ``window_seconds`` before another is outside the same
    window.  This matches a continuously sliding interval rather than a pair
    of inclusive endpoints whose elapsed duration is longer than the label.
    """
    if not _nonnegative_finite(window_seconds) or window_seconds <= 0:
        raise ValueError("rolling window must be positive and finite")
    ordered = []
    for stamp, value in entries:
        if not _nonnegative_finite(stamp) \
                or not _nonnegative_finite(value):
            raise ValueError(
                "rolling entries need finite non-negative timestamps and "
                "values")
        ordered.append((float(stamp), float(value)))
    ordered.sort()
    if not ordered:
        return {
            "max": None, "window_start_unix": None,
            "window_end_unix": None, "events_in_peak": 0,
            "events_total": 0,
        }
    left = 0
    running = 0.0
    peak = -1.0
    peak_left = 0
    peak_right = 0
    for right, (stamp, value) in enumerate(ordered):
        running += value
        while left <= right and stamp - ordered[left][0] >= window_seconds:
            running -= ordered[left][1]
            left += 1
        if running > peak:
            peak = running
            peak_left = left
            peak_right = right

    def clean(number: float):
        return int(number) if number.is_integer() else number

    end = ordered[peak_right][0]
    return {
        "max": clean(peak),
        "window_start_unix": end - window_seconds,
        "window_end_unix": end,
        "events_in_peak": peak_right - peak_left + 1,
        "events_total": len(ordered),
    }


def _rate_limit_evidence(results: list[dict], limits: dict | None,
                         run_meta: dict | None = None) -> tuple[dict, dict | None]:
    """Reconstruct this run's token/query windows and compare an as-of limit.

    Databricks counts input at request admission and reserves ``max_tokens``
    before admission, then credits unused output reservation back.  Persisted
    rows do not expose provider token-bucket state, the small burst buffer,
    other workspace traffic, or the exact timestamps of retry attempts.  The
    block therefore keeps observations and limitations separate and refuses a
    headroom conclusion when required coverage is incomplete.
    """
    rows = [row for row in results if isinstance(row, dict)]
    sent_rows = [r for r in rows if _sent_at(r) is not None]
    request_phases = {"preflight", "probe", "sizing", "calibration", "replay"}
    request_params = ((run_meta or {}).get("request_params") or {})
    extra_body = request_params.get("extra_body") or {}
    configured_service_tier = extra_body.get("service_tier", "default")
    observed_service_tiers = sorted({
        str(r.get("service_tier")) for r in sent_rows
        if isinstance(r.get("service_tier"), str)
        and r.get("service_tier").strip()
    })
    unexpected_service_tiers = [
        tier for tier in observed_service_tiers if tier != "default"]
    service_tier_consistent = (
        configured_service_tier == "default"
        and not unexpected_service_tiers)

    def raw_sent_value(row: dict):
        return (row.get("first_send_unix") if "first_send_unix" in row
                else row.get("t_send_unix"))

    invalid_timestamp_rows = sum(
        raw_sent_value(row) is not None and _sent_at(row) is None
        for row in rows if row.get("phase") in request_phases)
    unknown_outcome_rows = 0
    for row in rows:
        if row.get("phase") not in request_phases:
            continue
        attempt_value = row.get("request_attempts")
        if _sent_at(row) is None and (
                attempt_value is None
                or (isinstance(attempt_value, int)
                    and not isinstance(attempt_value, bool)
                    and attempt_value > 0)):
            unknown_outcome_rows += 1

    def attempt_observation(row: dict) -> tuple[int, bool]:
        """Return a conservative attempt count and whether it is exact.

        Current rows carry ``request_attempts``.  Legacy ``retries`` did not
        distinguish a connection failure before POST from a request that may
        have reached the provider, so it can size offered demand but can never
        make a rolling comparison complete.
        """
        value = row.get("request_attempts")
        if isinstance(value, int) and not isinstance(value, bool) \
                and value > 0:
            return value, True
        legacy = row.get("retries")
        if isinstance(legacy, int) and not isinstance(legacy, bool) \
                and legacy >= 0:
            return legacy + 1, False
        return 1, False

    attempt_info = [(row, *attempt_observation(row)) for row in sent_rows]
    attempt_counts_exact = all(exact for _row, _count, exact in attempt_info)
    attempt_timestamps_exact = all(
        exact and count == 1 for _row, count, exact in attempt_info)
    single_attempt = bool(sent_rows) and attempt_timestamps_exact
    clean_usage_rows = {id(row) for row in sent_rows
                        if _usage_is_trustworthy(row)}
    protocol_evidence_complete = all(
        "stream_complete" in row for row in sent_rows)
    input_entries = [
        (_sent_at(r), float(r["prompt_tokens"]) * count)
        for r, count, _exact in attempt_info
        if id(r) in clean_usage_rows]
    output_entries = [
        (_completed_at(r),
         (float(r["completion_tokens"])
          if id(r) in clean_usage_rows else 0.0))
        for r, _count, _exact in attempt_info
        if _completed_at(r) is not None
        and (id(r) in clean_usage_rows or r.get("status") == 429)]
    reservation_entries = [
        (_sent_at(r), float(r["max_tokens_requested"]) * count)
        for r, count, _exact in attempt_info
        if _nonnegative_finite(r.get("max_tokens_requested"))]
    query_entries = [
        (_sent_at(r), float(count)) for r, count, _exact in attempt_info]

    sent_n = len(sent_rows)
    input_coverage = len(input_entries) / sent_n if sent_n else None
    output_coverage = len(output_entries) / sent_n if sent_n else None
    reservation_coverage = len(reservation_entries) / sent_n if sent_n else None

    def add_horizon(evidence: dict, entries: list[tuple[float, float]],
                    window_seconds: float) -> dict:
        stamps = [float(stamp) for stamp, _value in entries]
        horizon = max(stamps) - min(stamps) if len(stamps) >= 2 else None
        projection = None
        if horizon is not None and horizon > 0:
            projection = sum(float(value) for _stamp, value in entries) \
                / horizon * window_seconds
        evidence.update({
            "window_seconds": window_seconds,
            "observation_horizon_seconds": horizon,
            "observation_covers_full_window": (
                horizon is not None and horizon >= window_seconds),
            "steady_state_projection": projection,
            "projection_note": (
                "total observed demand divided by the first-to-last event "
                "span and projected over the configured window; this is a "
                "diagnostic sustained-rate projection, not provider state"),
        })
        return evidence

    phases: dict[str, dict] = {}
    for row in rows:
        phase = str(row.get("phase") or "unlabeled")
        item = phases.setdefault(
            phase, {"rows": 0, "sent_rows": 0,
                    "unknown_outcome_rows": 0,
                    "physical_attempts_estimate": 0,
                    "attempt_counts_exact": True})
        item["rows"] += 1
        if _sent_at(row) is not None:
            count, exact = attempt_observation(row)
            item["sent_rows"] += 1
            item["physical_attempts_estimate"] += count
            item["attempt_counts_exact"] = (
                item["attempt_counts_exact"] and exact)
        elif phase in request_phases:
            attempt_value = row.get("request_attempts")
            if attempt_value is None or (
                    isinstance(attempt_value, int)
                    and not isinstance(attempt_value, bool)
                    and attempt_value > 0):
                item["unknown_outcome_rows"] += 1
    observed = {
        "traffic_scope": {
            "rows": len(rows),
            "sent_rows": sent_n,
            "physical_attempts_estimate": sum(
                count for _row, count, _exact in attempt_info),
            "attempt_count_unknown_rows": sum(
                not exact for _row, _count, exact in attempt_info),
            "unknown_outcome_rows": unknown_outcome_rows,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "phases": phases,
            "note": (
                "quota windows include every sealed request phase supplied "
                "by the runner, not only measured replay"),
        },
        "input_tokens_by_first_send": add_horizon(
            _rolling_peak(input_entries, 60.0) | {
            "reported_rows": len(input_entries),
            "sent_rows": sent_n,
            "coverage": input_coverage,
            "is_lower_bound": bool(sent_n and input_coverage != 1.0),
            "attempts_grouped_at_first_send": True,
        }, input_entries, 60.0),
        "actual_output_tokens_by_completion": add_horizon(
            _rolling_peak(output_entries, 60.0) | {
                "reported_rows": len(output_entries),
                "sent_rows": sent_n,
                "coverage": output_coverage,
                "timing_is_approximate": True,
            }, output_entries, 60.0),
        "offered_output_token_reservation_demand_by_first_send": add_horizon(
            _rolling_peak(reservation_entries, 60.0) | {
                "reported_rows": len(reservation_entries),
                "sent_rows": sent_n,
                "coverage": reservation_coverage,
                "includes_credit_back": False,
                "is_observed_provider_consumption": False,
                "note": (
                    "gross max_tokens offered to pre-admission checks; a "
                    "rejected request is demand, not a consumed reservation"),
            }, reservation_entries, 60.0),
        "physical_queries_by_first_send": add_horizon(
            _rolling_peak(query_entries, 3600.0) | {
                "logical_rows": sent_n,
                "attempt_counts_exact": attempt_counts_exact,
                "all_attempt_timestamps_exact": attempt_timestamps_exact,
                "confirmed_http_200_rows": sum(
                    row.get("status") == 200 for row in sent_rows),
                "provider_processing_ambiguous_rows": sum(
                    row.get("status") != 200 for row in sent_rows),
                "is_observed_provider_processed_count": False,
            }, query_entries, 3600.0),
        "single_physical_attempt_per_row": single_attempt,
        "protocol_evidence_complete": protocol_evidence_complete,
        "service_tier": {
            "configured": configured_service_tier,
            "observed": observed_service_tiers,
            "consistent_with_standard_pay_per_token": service_tier_consistent,
            "note": (
                "the standard pay-per-token quota model in this report "
                "requires an absent/default request tier; an observed "
                "non-default response tier invalidates that accounting "
                "model"),
        },
        "note": (
            "input tokens are endpoint-reported request totals attributed to "
            "first send. actual output is attributed to request completion "
            "because per-token generation timestamps are unavailable. offered "
            "output demand groups requested max_tokens at first send and does "
            "not claim rejected demand was reserved or model provider "
            "credit-back. retry timestamps, provider burst-buffer state, and "
            "traffic from other callers are not observable in a run artifact"),
    }
    if limits is None:
        return observed, None

    warning_at = float(limits["warning_utilization"])
    comparisons = {}
    warnings = []
    from .endpoint_meta import rate_limit_endpoint_binding
    binding = rate_limit_endpoint_binding(
        limits,
        (run_meta or {}).get("endpoint_metadata"),
    )
    if not binding["binding_complete"]:
        warnings.append(
            "the configured rate-limit model/deployment could not be bound "
            "to captured endpoint metadata")
    if not service_tier_consistent:
        warnings.append(
            "the request or response service tier was not exact default, so "
            "the standard pay-per-token quota model does not apply")

    def compare(name: str, limit_key: str, evidence: dict, *,
                trustworthy: bool, qualifier: str) -> None:
        if limit_key not in limits:
            return
        configured = float(limits[limit_key])
        display_name = name.replace("_", " ")
        measured = evidence.get("max")
        projected = evidence.get("steady_state_projection")
        short_horizon = not evidence.get("observation_covers_full_window")
        comparison_value = measured
        if short_horizon and projected is not None:
            comparison_value = max(float(measured or 0.0), float(projected))
        utilization = (float(comparison_value) / configured
                       if comparison_value is not None else None)
        scope_complete = not unknown_outcome_rows and not invalid_timestamp_rows
        trustworthy = (trustworthy and scope_complete
                       and binding["binding_complete"]
                       and service_tier_consistent)
        if measured is None:
            status = "unmeasured"
        elif not trustworthy:
            status = "incomplete_run_evidence"
        elif float(measured) / configured >= 1.0:
            status = "run_evidence_at_or_above_nominal_limit"
        elif short_horizon:
            status = ("short_observation_projection_at_or_above_warning"
                      if projected is not None and utilization is not None
                      and utilization >= warning_at else
                      "short_observation_incomplete")
        elif utilization >= warning_at:
            status = "run_evidence_warning_threshold_reached"
        else:
            status = "run_evidence_below_warning_threshold"
        comparisons[name] = {
            "configured_limit": configured,
            "observed_max": measured,
            "observed_ratio_to_nominal_limit": (
                float(measured) / configured if measured is not None else None),
            "steady_state_projection": projected,
            "comparison_value": comparison_value,
            "ratio_to_nominal_limit": utilization,
            # Kept for one release as an explicitly non-provider alias.
            "utilization": utilization,
            "warning_utilization": warning_at,
            "status": status,
            "comparison_is_complete": trustworthy and not short_horizon,
            "provider_headroom_established": False,
            "observation_horizon_seconds": evidence.get(
                "observation_horizon_seconds"),
            "window_seconds": evidence.get("window_seconds"),
            "qualifier": qualifier,
        }
        if status == "unmeasured":
            warnings.append(f"{display_name} could not be measured")
        elif status == "incomplete_run_evidence":
            warnings.append(
                f"{display_name} cannot establish headroom because required "
                "request "
                "usage or physical-attempt timing is incomplete")
        elif status.startswith("short_observation"):
            projected_text = (
                " unavailable" if projected is None else
                f" {projected:,.1f} ({utilization:.1%} of the nominal limit)")
            warnings.append(
                f"{display_name} was observed for less than its "
                f"{evidence.get('window_seconds', 0):g}-second window; "
                f"the sustained-rate projection is{projected_text}. this "
                "short run cannot establish sustained quota headroom")
        elif status == "run_evidence_at_or_above_nominal_limit":
            warnings.append(
                f"this run's {qualifier} was {utilization:.1%} of the "
                "configured nominal limit")
        elif status == "run_evidence_warning_threshold_reached":
            warnings.append(
                f"this run's {qualifier} was {utilization:.1%} of the "
                "configured nominal limit, "
                f"above the {warning_at:.0%} warning threshold")

    compare(
        "input_tokens_per_minute", "input_tokens_per_minute",
        observed["input_tokens_by_first_send"],
        trustworthy=(sent_n > 0 and input_coverage == 1.0 and single_attempt
                     and protocol_evidence_complete),
        qualifier="endpoint-reported input-token contribution")
    compare(
        "output_tokens_per_minute", "output_tokens_per_minute",
        observed["offered_output_token_reservation_demand_by_first_send"],
        trustworthy=(sent_n > 0 and reservation_coverage == 1.0
                     and single_attempt),
        qualifier=("conservative gross max_tokens demand offered to "
                   "pre-admission checks before rejection or credit-back"))
    compare(
        "queries_per_hour", "queries_per_hour",
        observed["physical_queries_by_first_send"],
        trustworthy=(sent_n > 0 and single_attempt
                     and not observed["physical_queries_by_first_send"][
                         "provider_processing_ambiguous_rows"]),
        qualifier=("physical POST demand grouped at client first send; not "
                   "the provider's processed-query counter"))
    block = {
        "configured": _redact_secrets(limits),
        "binding": binding,
        "comparisons": comparisons,
        "warning": "; ".join(warnings) if warnings else None,
        "external_usage_warning": (
            "these comparisons cover only traffic recorded by this run. "
            "provider token-bucket state, burst allowance, and other callers "
            "are not observed, and offered reservation demand is not consumed "
            "quota. no comparison establishes provider headroom; confirm "
            "provider telemetry before a production capacity claim"),
    }
    return observed, block


def _pct_table(values: list[float | None]) -> dict:
    xs = np.array([v for v in values if v is not None], dtype=float)
    if xs.size == 0:
        return {f"p{p}": None for p in PCTS} | {"n": 0}
    out = {f"p{p}": float(np.percentile(xs, p)) for p in PCTS}
    out["n"] = int(xs.size)
    out["mean"] = float(xs.mean())
    return out


def _verdict(s: dict) -> tuple[str, str]:
    """The run's verdict, as (kind, sentence). kind is one of
    invalid / miss / caution / ok.

    Both renderers call this, so report.md and the html cannot disagree.

    Green requires positive evidence that the run is a valid measurement,
    not merely the absence of a missed latency target. Enumerating specific
    failure modes kept leaving doors open: a run with an 8 percent error
    rate, or one that never held the concurrency on its label, or one whose
    endpoint collapsed mid-run, could all satisfy a latency target and print
    "meets every acceptance target". Anything that undermines the
    measurement now downgrades the verdict and says which thing did.
    """
    sla = s.get("sla") or {}
    a = s.get("answers") or {}
    rows = [r for k in ("ttft_vs_target", "ttfg_vs_target")
            for r in (sla.get(k) or [])]
    misses = sum(1 for r in rows if r["met"] is False)
    if sla.get("hard_timeout_breaches"):
        misses += 1
    if sla.get("interchunk_breaches"):
        misses += 1
    if (sla.get("success_rate") or {}).get("met") is False:
        misses += 1
    unmeasured = sum(1 for r in rows
                     if r["met"] is None and r.get("target_ms") is not None)

    # A 429 is not evidence that the endpoint itself reached a serving
    # capacity limit. It says only that some rate-limit or quota policy
    # rejected a request; the limiting dimension and the component that
    # enforced it require provider telemetry. Keep this ahead of the ordinary
    # success/error-rate gates: a low 429 rate can still satisfy a customer's
    # success-rate target, but it can never support a clean capacity claim.
    http_429_count = s.get("http_429_count")
    if isinstance(http_429_count, int) \
            and not isinstance(http_429_count, bool) \
            and http_429_count > 0:
        evidence = s.get("http_429") or {}
        examined = evidence.get("request_rows_examined")
        denominator = (f" of {examined}" if isinstance(examined, int)
                       and examined >= http_429_count else "")
        return "invalid", (
            f"quota-limited: {http_429_count}{denominator} request "
            f"{'row returned' if http_429_count == 1 else 'rows returned'} "
            "HTTP 429. this run supports no endpoint-capacity conclusion; "
            "identify the enforcing limit and dimension in provider telemetry")

    if a.get("invalid"):
        return "invalid", a["invalid"]
    _run = s.get("run") or {}
    if _run.get("aggregation_valid") is False:
        issues = _run.get("compatibility_issues") or []
        detail = "; ".join(str(x) for x in issues[:3])
        return "invalid", (
            "this aggregate combined inputs that were not proven compatible"
            + (f": {detail}" if detail else "")
            + ". read the source runs separately")

    # answers gate the banner on their own. an SLA block with no success_rate
    # key has no row that a collapse in readable answers can miss, so without
    # this a run that answered 29 percent of the time rendered green.
    rate = a.get("answer_rate")
    floor = (sla.get("success_rate") or {}).get("target") or 0.99
    if rate is not None and rate < floor:
        n = a.get("judged") or a.get("attempted") or 0
        bad = n - (a.get("answered") or 0)
        return "miss", (
            f"{bad} of {n} requests did not produce a readable answer "
            f"({rate:.1%} answered). latency figures describe only the ones "
            "that answered")

    err = s.get("error_rate")
    if err and err > 0.0:
        got = s.get("requests_failed") or 0
        tot = s.get("requests_total") or 0
        if err > (1.0 - floor):
            return "miss", (
                f"{got} of {tot} requests failed ({err:.2%}). latency "
                "percentiles cover only the ones that came back, and on a "
                "shedding endpoint those are the fast ones")

    if misses:
        return "miss", (f"{misses} acceptance target"
                        f"{'s' if misses != 1 else ''} missed")

    # met the targets. now decide whether the run is good enough to say so.
    doubts = []
    if sla.get("targets_warning"):
        doubts.append(str(sla["targets_warning"]))
    if unmeasured:
        doubts.append(f"{unmeasured} target"
                      f"{'s' if unmeasured != 1 else ''} had no measurement "
                      "behind them")
    if sla.get("coverage_warning"):
        doubts.append("the scored metric is missing on many requests")
    if err:
        doubts.append(f"{s.get('requests_failed') or 0} requests failed")
    if (s.get("concurrency") or {}).get("warning"):
        doubts.append("observed concurrency diverged substantially from the "
                      "unloaded estimate used to size the open-loop rate")
    if (s.get("client") or {}).get("warning"):
        doubts.append("the load did not reach the endpoint on schedule")
    if sla.get("caller_latency_warning"):
        doubts.append(sla["caller_latency_warning"])
    if (s.get("throughput") or {}).get("coverage_warning"):
        doubts.append("token usage was missing on many responses, so "
                      "throughput and cost cover a subset")
    if (s.get("rate_limits") or {}).get("warning"):
        doubts.append(str((s.get("rate_limits") or {})["warning"]))
    if (s.get("cost") or {}).get("coverage_warning"):
        doubts.append("aggregate or effective cost could not be computed "
                      "because usage or physical-attempt evidence was "
                      "incomplete")
    if (s.get("cost") or {}).get("applicability_warning"):
        doubts.append("the supplied pricing rates were not provenance-bound "
                      "to this provider/model/product/tier run")
    if (s.get("cache_fidelity") or {}).get("warning"):
        doubts.append((s.get("cache_fidelity") or {})["warning"])
    if (s.get("token_targeting") or {}).get("warning"):
        doubts.append((s.get("token_targeting") or {})["warning"])
    if (s.get("latency_population") or {}).get("warning"):
        doubts.append((s.get("latency_population") or {})["warning"])
    _npw = (s.get("network_path") or {})
    if _npw.get("warning"):
        doubts.append(str(_npw["warning"]))
    _cap = a.get("truncated_by_global_cap") or 0
    _scored_n = a.get("scored") or 0
    if _scored_n and _cap / _scored_n > 0.05:
        doubts.append(
            f"{_cap} of {_scored_n} responses were cut short by "
            "max_output_tokens_cap rather than by their own target, so the "
            "run did not reproduce the profile's output sizes and "
            "end-to-end is correspondingly short")
    _drift = s.get("drift") or {}
    dk = _drift.get("drift_kind")
    if dk and dk != "stable":
        doubts.append(f"latency was {dk} across the run")
    elif not dk:
        # no verdict at all: too short to window, no window with a usable
        # sample, or a merged run where drift is blanked by construction.
        # not knowing whether latency held is not the same as it holding.
        doubts.append("stability over the run was not established"
                      + (f" ({_drift['note']})" if _drift.get("note") else ""))
    # a scored target on a quantile the sample cannot support is not a pass
    _samp = s.get("sample") or {}
    _weak = set(_samp.get("indicative_only") or [])
    # the sample gate counts successful requests, but the SCORED metric can
    # be missing on some of them. re-derive the floor from the number of
    # values actually behind the table this target reads.
    _need = {"p50": 20, "p90": 100, "p95": 200, "p99": 1000}
    _defn = sla.get("ttft_definition") or "first_content"
    _key = "ttft_ms" if _defn == "first_content" else "ttfv_ms"
    _n_scored = (s.get(_key) or {}).get("n") or 0
    if _n_scored:
        _weak |= {q for q, need in _need.items() if _n_scored < need}
    _scored_weak = sorted({r["quantile"] for r in rows
                           if r["quantile"] in _weak})
    _sr = sla.get("success_rate") or {}
    if _sr.get("met") is True \
            and _sr.get("statistically_demonstrated") is False:
        doubts.append(
            f"the observed success rate met {_sr['target']}, but its "
            f"one-sided 95% Wilson lower bound is "
            f"{_sr['one_sided_95pct_wilson_lower']:.4%}, so this sample "
            "cannot demonstrate the target")
    if _scored_weak:
        doubts.append(f"{', '.join(_scored_weak)} scored on "
                      f"{_samp.get('n')} requests, which cannot support "
                      f"{'that quantile' if len(_scored_weak) == 1 else 'those quantiles'}")
    _had_targets = bool(rows or sla.get("success_rate"))
    _lead = ("met every acceptance target, but " if _had_targets
             else "no acceptance targets were given, and ")
    if doubts:
        return "caution", (_lead + ", and ".join(doubts)
                           + ". read those before quoting this run")
    if not _had_targets:
        return "caution", ("no acceptance targets were given, so nothing was "
                           "scored. pass your own to get a verdict")
    return "ok", "meets every acceptance target"


def _answered(r: dict) -> bool:
    """Did this request produce a usable assistant outcome?

    Transport success is not answer success. A reasoning model that spends
    its whole token budget thinking returns HTTP 200, a well formed stream,
    a finish reason, and nothing a user could read.

    Truncation deliberately does NOT disqualify. This harness sets max_tokens
    to the sampled output size on purpose, so finish_reason "length" is the
    normal ending for a run hitting its target output length. Truncation is
    reported as its own rate instead, because the thing that separates a
    short answer from no answer is whether visible content or a structurally
    valid tool call appeared at all. A partial or malformed tool-call fragment
    is deliberately not enough.
    """
    return bool((r.get("visible_content_seen")
                 or (r.get("valid_tool_calls") or 0) > 0)
                and r.get("stream_complete")
                and not r.get("parse_errors"))


def _content_delta_seen(r: dict) -> bool:
    """Whether a current-format row emitted visible or reasoning content.

    A valid tool-call-only stream is a successful request, but it did not
    emit a content delta. Keeping this predicate separate prevents reports
    from turning tool-call success into a false content-count claim.
    """
    return bool(r.get("visible_content_seen") or r.get("reasoning_seen"))


def _answer_block(results: list[dict]) -> dict | None:
    """Answer completion, separately from HTTP and content-stream success.

    ``ok`` is a harness success field: current rows may satisfy it with a
    visible/reasoning content delta or a structurally valid tool call. It is
    not an HTTP-status or content-delta counter. Keep those populations
    separate so a tool-only response is not called content, a reasoning-only
    HTTP 200 is not presented as a readable answer, and a response-bearing
    stream is not called HTTP 200 when status was not retained by a legacy
    row.
    """
    ok = [r for r in results if _protocol_clean_success(r)]
    observed_fields = {
        "visible_content_seen", "reasoning_seen", "valid_tool_calls"}
    scored = [r for r in results if observed_fields.intersection(r)]
    legacy_failures = [r for r in results
                       if not r.get("ok")
                       and not observed_fields.intersection(r)]
    if not scored and not legacy_failures:
        return None          # rows written before this was recorded
    n_observed = len(scored)
    complete = sum(1 for r in scored if _answered(r))
    judged = n_observed + len(legacy_failures)
    statuses = [r.get("status") for r in results if r.get("status") is not None]
    out = {
        "attempted": len(results),
        # Clean harness-success count, retained under its historical key for
        # automation compatibility. It is not necessarily an HTTP 200 or a
        # content-bearing stream; corrupt/incomplete rows are excluded.
        "transport_ok": len(ok),
        "harness_successful": len(ok),
        "content_delta_streams": sum(
            1 for r in scored if _content_delta_seen(r)),
        # Backward-compatible alias, corrected to its literal meaning for
        # current rows. Legacy successes without observability are excluded.
        "content_streams": sum(
            1 for r in scored if _content_delta_seen(r)),
        "unclassified_legacy_successes": sum(
            1 for r in results
            if r.get("ok") and not observed_fields.intersection(r)),
        "http_status_observed_for": len(statuses),
        "http_200": sum(1 for status in statuses if status == 200),
        "scored": n_observed,
        "answered": complete,
        "acceptable_outcomes": complete,
        "valid_tool_call_outcomes": sum(
            1 for r in scored if (r.get("valid_tool_calls") or 0) > 0),
        "tool_call_only_outcomes": sum(
            1 for r in scored
            if (r.get("valid_tool_calls") or 0) > 0
            and not r.get("visible_content_seen")),
        "valid_tool_calls_total": sum(
            int(r.get("valid_tool_calls") or 0) for r in scored),
        "no_visible_content": sum(
            1 for r in scored if not r.get("visible_content_seen")),
        "no_acceptable_outcome": sum(
            1 for r in scored
            if not r.get("visible_content_seen")
            and not (r.get("valid_tool_calls") or 0) > 0),
        "stream_incomplete": sum(
            1 for r in scored if not r.get("stream_complete")),
        "parse_errors": sum(1 for r in scored if r.get("parse_errors")),
        "truncated": sum(1 for r in scored if r.get("truncated")),
        # the denominator is every request we can judge: the ones that came
        # back and carry the fields, plus the ones that failed outright. a
        # request that failed did not produce an answer and belongs here.
        # rows written before these fields existed are NOT counted, because
        # they are unmeasurable rather than unanswered, and counting them
        # would fail a merged 0.3.0 shard for having old-format rows.
        "judged": judged,
        # a row whose budget was cut by the global cap rather than by its own
        # sampled target is a different animal: "length" there means the run
        # did NOT reach the output size the profile asked for, which shortens
        # end-to-end and caps output throughput.
        "truncated_by_global_cap": sum(
            1 for r in scored
            if r.get("truncated") and r.get("max_tokens_requested")
            and r.get("intended_output_tokens")
            and r["max_tokens_requested"] < r["intended_output_tokens"]),
        "answer_rate": (round(complete / judged, 6) if judged else None),
        "answer_rate_of_transport_ok": (round(complete / len(ok), 6)
                                        if ok else None),
        "note": "an acceptable outcome means visible content or at least one "
                "structurally valid tool call arrived and the stream finished "
                "cleanly. it does NOT mean the answer or tool choice was "
                "correct. truncation alone is not counted as a failure. a "
                "partial or malformed tool-call fragment is not accepted.",
    }
    if complete == 0 and judged:
        # name the counter that actually drove it. asserting "produced no
        # visible content" when the real cause was a stream that never
        # terminated puts a false statement next to a zero counter.
        cause = max((("returned no visible content or valid tool call",
                      out["no_acceptable_outcome"]),
                     ("never terminated their stream", out["stream_incomplete"]),
                     ("hit unrecoverable parse errors", out["parse_errors"]),
                     ("failed before a content stream was established",
                      len(legacy_failures))),
                    key=lambda kv: kv[1])
        out["invalid"] = (
            f"not one of the {judged} requests with answer observability "
            "produced a reportable completed answer. a reportable answer "
            "requires visible content or a valid tool call, a complete "
            "stream, and no unrecoverable parse error. most requests "
            f"{cause[0]} "
            f"({cause[1]} of {judged}). there is no latency-to-answer in this "
            "run and nothing "
            "here is a performance result.")
    return out


def summarize(results: list[dict], schedule_meta: dict | None = None,
              run_meta: dict | None = None,
              acceptance: dict | None = None,
              ttft_definition: str = "first_content",
              pricing: dict | None = None,
              concurrency_target: int | None = None,
              rate_limits: dict | None = None,
              rate_limit_results: list[dict] | None = None) -> dict:
    ok = [r for r in results if _protocol_clean_success(r)]
    failed = [r for r in results if not _protocol_clean_success(r)]
    safe_run_meta = _redact_secrets(run_meta or {})

    # Current rows say whether visible content arrived and the stream ended
    # cleanly. When that observability exists, the primary latency tables are
    # answer latencies, not percentiles over reasoning-only or malformed HTTP
    # successes. Older rows are retained as an explicitly unclassified legacy
    # population rather than silently mixed into user-facing numbers.
    answer_observed = [
        r for r in ok
        if "visible_content_seen" in r or "valid_tool_calls" in r]
    answered = [r for r in answer_observed if _answered(r)]
    latency_ok = answered if answer_observed else ok
    unclassified_ok = len(ok) - len(answer_observed)
    latency_population = {
        "kind": (("acceptable_content_or_tool_outcomes"
                  if any((r.get("valid_tool_calls") or 0) > 0
                         for r in answered)
                  else "readable_answers") if answer_observed
                 else "legacy_content_streams_unverified"),
        "n": len(latency_ok),
        "content_streams": len(ok),
        "answer_observed_for": len(answer_observed),
        "excluded_unreadable": (len(answer_observed) - len(answered)
                                if answer_observed else 0),
        "unclassified_legacy_rows": unclassified_ok,
        "note": (
            "primary latency percentiles include only requests that produced "
            "visible content or a structurally valid tool call and finished "
            "with no parse errors"
            if answer_observed else
            "these legacy rows do not record answer observability, so latency "
            "percentiles describe content-bearing response streams and cannot "
            "be claimed as latency to a readable answer"),
    }
    if answer_observed and unclassified_ok:
        latency_population["warning"] = (
            f"{unclassified_ok} successful legacy rows do not record whether "
            "they produced a readable answer, so they are excluded from the "
            "primary answer-latency population")

    # achieved cache, endpoint-reported only
    usage_trustworthy = [r for r in ok if _usage_is_trustworthy(r)]
    ach = [(r["cached_tokens"] / r["prompt_tokens"])
           for r in usage_trustworthy
           if r.get("cached_tokens") is not None
           and r.get("prompt_tokens")]
    cache_sources = sorted({r.get("cached_tokens_source")
                            for r in usage_trustworthy
                            if r.get("cached_tokens_source")})
    intended_cache = [r.get("intended_cache_fraction") for r in results
                      if r.get("intended_cache_fraction") is not None]
    cache_intended_rows = [
        r for r in results
        if r.get("intended_cache_fraction") is not None]
    paired_cache_error = [
        abs((r["cached_tokens"] / r["prompt_tokens"])
            - r["intended_cache_fraction"])
        for r in usage_trustworthy
        if r.get("cached_tokens") is not None and r.get("prompt_tokens")
        and r.get("intended_cache_fraction") is not None]
    invalid_cache_rows = sum(
        1 for r in usage_trustworthy
        if r.get("cached_tokens") is not None and r.get("prompt_tokens")
        and not 0 <= r["cached_tokens"] / r["prompt_tokens"] <= 1)

    # Token targeting is a paired workload-fidelity check, not just a p50
    # decoration. Synthetic/profile runs claim an input and output shape; an
    # otherwise fast run at one tenth of that shape is not evidence for the
    # declared workload. max_tokens is only a cap, so output mismatch is
    # reported as mismatch rather than blamed on the endpoint.
    def positive_number(value) -> bool:
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)) and value > 0)

    def nonnegative_number(value) -> bool:
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)) and value >= 0)

    input_intended_rows = [
        r for r in results
        if positive_number(r.get("intended_input_tokens"))]
    input_eligible = [
        r for r in input_intended_rows
        if _usage_is_trustworthy(r)]
    input_pairs = [
        (float(r["prompt_tokens"]), float(r["intended_input_tokens"]))
        for r in input_eligible if positive_number(r.get("prompt_tokens"))]
    ratios = [actual / intended for actual, intended in input_pairs]
    input_errors_pct = [abs(ratio - 1.0) * 100.0 for ratio in ratios]

    output_intended_rows = [
        r for r in results
        if positive_number(r.get("intended_output_tokens"))]
    output_eligible = [
        r for r in output_intended_rows
        if _usage_is_trustworthy(r)]
    output_pairs = [
        (float(r["completion_tokens"]), float(r["intended_output_tokens"]))
        for r in output_eligible
        if nonnegative_number(r.get("completion_tokens"))]
    out_ratios = [actual / intended for actual, intended in output_pairs]
    output_errors_pct = [abs(ratio - 1.0) * 100.0 for ratio in out_ratios]
    targeting_warnings = []
    tolerance_pct = 10.0
    input_coverage = (len(input_pairs) / len(input_intended_rows)
                      if input_intended_rows else None)
    output_coverage = (len(output_pairs) / len(output_intended_rows)
                       if output_intended_rows else None)
    input_error_table = _pct_table(input_errors_pct)
    output_error_table = _pct_table(output_errors_pct)
    if input_intended_rows:
        if input_coverage is not None and input_coverage < 0.99:
            targeting_warnings.append(
                f"prompt-token usage was reported for only "
                f"{len(input_pairs)} of {len(input_intended_rows)} captured "
                "profile requests with declared input targets")
        elif ((input_error_table.get("p50") or 0.0) > tolerance_pct
              or (input_error_table.get("p95") or 0.0) > tolerance_pct):
            targeting_warnings.append(
                "endpoint-reported input tokens did not reproduce the "
                f"declared profile within ±{tolerance_pct:.0f}% "
                f"(absolute relative error p50 "
                f"{input_error_table['p50']:.1f}%, p95 "
                f"{input_error_table['p95']:.1f}%)")
    if output_intended_rows:
        if output_coverage is not None and output_coverage < 0.99:
            targeting_warnings.append(
                f"completion-token usage was reported for only "
                f"{len(output_pairs)} of {len(output_intended_rows)} captured "
                "profile requests with declared output targets")
        elif ((output_error_table.get("p50") or 0.0) > tolerance_pct
              or (output_error_table.get("p95") or 0.0) > tolerance_pct):
            targeting_warnings.append(
                "endpoint-reported output tokens did not reproduce the "
                f"declared profile within ±{tolerance_pct:.0f}% "
                f"(absolute relative error p50 "
                f"{output_error_table['p50']:.1f}%, p95 "
                f"{output_error_table['p95']:.1f}%). max_tokens is a cap, "
                "not a promise that a model will generate to that length")
    finish_reasons: dict[str, int] = {}
    for r in ok:
        fr = r.get("finish_reason")
        if fr:
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    # arrival honesty
    #
    # dispatch_lag_ms is stamped in the dispatcher thread just before the
    # request is handed to the pool. ThreadPoolExecutor.submit() never
    # blocks, it queues, so that number cannot see a saturated pool: it
    # reports single-digit ms while requests sit in the queue for minutes.
    # The number that matters is when the client began sending, which is
    # first_send_unix, against when the schedule wanted it.
    lags = [r.get("dispatch_lag_ms") for r in results
            if r.get("dispatch_lag_ms") is not None]
    wire = []
    # every row carries first_send_unix, the moment its FIRST attempt went
    # out. t_send_unix belongs to whichever attempt produced the result, so
    # on a retried row it carries the endpoint's delay rather than saying
    # when the load was offered. no row needs excluding once the honest
    # stamp is available. older rows without the field fall back.
    exact_wait = [float(r["queue_wait_ms"]) for r in results
                  if r.get("queue_wait_ms") is not None]
    wire.extend(exact_wait)
    # Rows from harnesses predating exact monotonic caller clocks can still be
    # reconstructed from epoch send stamps. Never overwrite an exact field:
    # an explicit None means the newer client did not put a request on wire.
    stamped = [r for r in results
               if "queue_wait_ms" not in r
               and r.get("scheduled_s") is not None
               and _sent_at(r) is not None]
    if stamped:
        # one offset, taken from the row that was earliest relative to its own
        # schedule. minimizing the two series independently would subtract a
        # constant no request experienced, and would let one slow first send
        # zero out real lateness everywhere.
        offset = min(_sent_at(r) - r["scheduled_s"] for r in stamped)
        for r in stamped:
            late = ((_sent_at(r) - r["scheduled_s"]) - offset) * 1000.0
            wire.append(max(late, 0.0))
            # coordinated omission. the latency clock starts when a worker
            # actually sends, so a request that sat in the client queue for
            # a minute still reports whatever the endpoint took once it
            # finally went out. that is the classic way a saturated load
            # generator reports a healthy tail. the corrected figure adds
            # the wait, which is what a caller who asked at the scheduled
            # moment actually experienced.
            r["queue_wait_ms"] = max(late, 0.0)
    wire_note = None
    if results and not wire:
        wire_note = ("wire lateness is not reported: no request carried an "
                     "exact queue-wait clock or legacy schedule/send stamps.")
    retried = sum(1 for r in results if r.get("retries"))

    # observation interval, not the send window. token totals include
    # generations that finish after the last request went out, so dividing
    # by (last_send - first_send) overstates throughput by the length of the
    # drain. with a 99 second send window and 60 second generations that is
    # about 61 percent high.
    dur = None
    send_span = None
    sent: list[float] = []
    done: list[float] = []
    if results:
        sent = [_sent_at(r) for r in results if _sent_at(r) is not None]
        done = [_completed_at(r) for r in results
                if _completed_at(r) is not None]
        if sent:
            if len(done) == len(sent):
                dur = max(max(done) - min(sent), 1e-9)
            # the ARRIVAL rate belongs on the send span. dividing it by the
            # observation interval above would charge it for the drain and
            # understate the load that was actually offered.
            send_span = max(max(sent) - min(sent), 1e-9)

    # throughput in the customer's own vocabulary (tokens per minute)
    usage_rows = [r for r in results if _usage_is_trustworthy(r)]
    in_tok = sum(float(r["prompt_tokens"]) for r in usage_rows)
    out_tok = sum(float(r["completion_tokens"]) for r in usage_rows)
    cached_tok = sum(float(r.get("cached_tokens") or 0) for r in usage_rows)
    dur_min = (dur / 60.0) if dur else None
    # how many successful responses actually reported usage. a run where
    # only a tenth of them do would otherwise understate token throughput
    # and per-token cost tenfold with nothing said about it.
    usage_n = len(usage_rows)
    usage_coverage = (usage_n / len(results)) if results else None
    complete_request_evidence = (
        results if rate_limit_results is None else rate_limit_results)
    http_429 = _http_429_evidence(
        complete_request_evidence,
        scope=("measured replay rows" if rate_limit_results is None else
               "all supplied request phases"))
    token_windows, rate_limit_block = _rate_limit_evidence(
        complete_request_evidence, rate_limits, safe_run_meta)

    summary = {
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "requests_retried": retried,
        "error_rate": len(failed) / len(results) if results else None,
        "failures_by_error": _top_errors(failed),
        "failures_by_http_status": _failures_by_http_status(failed),
        # Top-level aliases keep simple report/automation consumers from
        # having to understand the richer evidence block. The count/rate can
        # include preflight, probe, sizing, and calibration requests when the
        # runner supplied those captured, later manifest-bound request rows.
        "http_429_count": http_429["count"],
        "http_429_rate": http_429["rate"],
        "http_429": http_429,
        "quota_limited": http_429["quota_limited"],
        "ttft_ms": _pct_table([r.get("ttft_ms") for r in latency_ok]),
        "ttf_tool_call_ms": _pct_table(
            [r.get("ttf_tool_call_ms") for r in latency_ok]),
        "ttfb_ms": _pct_table([r.get("ttfb_ms") for r in latency_ok]),
        "connect_ms": _pct_table([r.get("connect_ms") for r in ok]),
        "e2e_ms": _pct_table([r.get("e2e_ms") for r in latency_ok]),
        "interchunk_max_ms": _pct_table(
            [r.get("interchunk_max_ms") for r in latency_ok]),
        "throughput": {
            "input_tokens_per_min": in_tok / dur_min if dur_min else None,
            "output_tokens_per_min": out_tok / dur_min if dur_min else None,
            "usage_coverage": usage_coverage,
            "completion_time_coverage": (
                len(done) / len(sent) if sent else None),
            "note": ("endpoint-reported token counts over the observation "
                     "interval, which runs from the first send to the last "
                     "completion so generations finishing during the drain "
                     "are inside the window they belong to"),
            "coverage_warning": (
                (f"completion time was available for only {len(done)} of "
                 f"{len(sent)} requests that reached the wire, so token "
                 "throughput is withheld rather than treating failed "
                 "requests as zero-duration")
                if sent and len(done) != len(sent) else
                (None if usage_coverage is None or usage_coverage == 1.0 else
                 f"only {usage_n} of {len(results)} attempted requests "
                 "returned a clean, complete stream with internally sane "
                 "token usage, so these totals cover that subset, not the "
                 "run")),
        },
        "observed_rate_windows": token_windows,
        "achieved_cache_fraction": _pct_table(ach) | {
            "reported_for_n": len(ach),
            "eligible_successes": len(usage_trustworthy),
            "eligible_requests": len(results),
            "coverage": (len(ach) / len(results)) if results else None,
            "source_fields": (cache_sources
                              or (["SOURCE FIELD NOT RECORDED"] if ach else
                                  ["NOT REPORTED BY ENDPOINT"])),
        },
        "intended_cache_fraction": _pct_table(intended_cache),
        "latency_population": latency_population,
        "token_targeting": {
            "input_intended_requests": len(input_intended_rows),
            "input_eligible_successes": len(input_eligible),
            "input_reported_n": len(input_pairs),
            "input_coverage": input_coverage,
            "input_reported_over_intended": _pct_table(ratios),
            "input_abs_relative_error_pct": input_error_table,
            "reported_over_intended_p50":
                float(np.percentile(ratios, 50)) if ratios else None,
            "abs_error_pct_p50":
                float(np.percentile([abs(x - 1.0) for x in ratios], 50) * 100)
                if ratios else None,
            "output_intended_requests": len(output_intended_rows),
            "output_eligible_successes": len(output_eligible),
            "output_reported_n": len(output_pairs),
            "output_coverage": output_coverage,
            "output_reported_over_intended": _pct_table(out_ratios),
            "output_abs_relative_error_pct": output_error_table,
            "output_reported_over_intended_p50":
                float(np.percentile(out_ratios, 50)) if out_ratios else None,
            "output_abs_error_pct_p50":
                float(np.percentile([abs(x - 1.0) for x in out_ratios], 50)
                      * 100) if out_ratios else None,
            "finish_reasons": finish_reasons,
            "tolerance_pct": tolerance_pct,
            "status": ("not_applicable" if not input_intended_rows
                       and not output_intended_rows else
                       "verified" if not targeting_warnings else "mismatch"),
            "warning": "; ".join(targeting_warnings)
            if targeting_warnings else None,
            "note": "endpoint-reported token counts are the source of truth. "
                    "input side is calibrated, output side is only reported "
                    "(models may stop before max_tokens: finish_reason stop "
                    "vs length)",
        },
        "arrivals": {
            # count the rows the span was measured over, not every row. a
            # half-stamped input would otherwise report double the rate.
            "achieved_qps_overall": ((len(sent) - 1) / send_span
                                     if send_span and len(sent) > 1
                                     else None),
            "dispatch_lag_ms": _pct_table(lags),
            "wire_lateness_ms": _pct_table(wire),
            **({"wire_lateness_note": wire_note} if wire_note else {}),
            "note": "dispatch lag is how late the dispatcher handed the "
                    "request to the pool. wire lateness is how late the "
                    "client began sending the request, which is the one "
                    "that grows when the client is the bottleneck, because a "
                    "saturated pool queues rather than blocking the "
                    "dispatcher.",
        },
        "schedule": schedule_meta or {},
        "run": safe_run_meta,
    }
    if rate_limit_block is not None:
        summary["rate_limits"] = rate_limit_block
    for field in ("ttft_ms", "ttf_tool_call_ms"):
        values = [r.get(field) for r in latency_ok]
        summary[field]["missing"] = sum(v is None for v in values)
        summary[field]["of"] = len(values)
    if intended_cache:
        tolerance = 0.10
        err = _pct_table(paired_cache_error)
        coverage = (len(paired_cache_error) / len(cache_intended_rows)
                    if cache_intended_rows else None)
        warnings = []
        if not paired_cache_error:
            warnings.append(
                "the workload specified a cached prompt-token fraction, but "
                "the endpoint did not report enough cache usage to verify it")
        elif invalid_cache_rows:
            warnings.append(
                f"{invalid_cache_rows} responses reported cached tokens outside "
                "the valid zero-to-prompt-token range")
        elif ((err.get("p50") or 0) > tolerance
              or (err.get("p95") or 0) > tolerance):
            warnings.append(
                "the achieved cached prompt-token fraction did not reproduce "
                f"the intended workload within ±{tolerance:.2f} "
                f"(absolute error p50 {err['p50']:.3f}, p95 {err['p95']:.3f})")
        if coverage is not None and coverage < 0.99:
            warnings.append(
                f"cache usage was reported for only "
                f"{len(paired_cache_error)} of {len(cache_intended_rows)} "
                "captured profile requests with declared cache targets")
        summary["cache_fidelity"] = {
            "status": "verified" if not warnings else "unverified",
            "tolerance_abs": tolerance,
            "paired_n": len(paired_cache_error),
            "intended_requests": len(cache_intended_rows),
            "coverage": coverage,
            "absolute_error": err,
            "warning": "; ".join(warnings) if warnings else None,
            "note": "cache fraction is cached prompt tokens divided by all "
                    "prompt tokens for each request; it is not request hit rate",
        }
    # A minimum TCP connect duration is useful location context but is not an
    # exact RTT and cannot be subtracted from TTFT to recover endpoint time.
    _np = safe_run_meta.get("network_path")
    if _np and _tcp_connect_floor(_np) is not None:
        floor = float(_tcp_connect_floor(_np))
        _t = (summary.get("ttft_ms") or {}).get("p50")
        _np = dict(_np)
        _np["tcp_connect_min_ms"] = floor
        # Old artifacts may already carry these invalid derived fields. Never
        # repeat or re-render them as current evidence.
        _np.pop("ttft_p50_less_rtt", None)
        _np.pop("share_of_ttft_p50", None)
        if _t:
            _np["tcp_connect_floor_to_ttft_p50_ratio"] = round(
                floor / _t, 4)
        _np["interpretation"] = (
            "TCP connect duration is a network-path floor and location "
            "diagnostic. It is not an exact RTT or endpoint processing-time "
            "measurement and must not be subtracted from TTFT.")
        summary["network_path"] = _np

    # time per output token, after the first. this is the metric the serving
    # docs use to reason about generation length: latency is roughly
    # TTFT + TPOT * output_tokens, so TPOT is what says whether a longer
    # answer still fits the budget. every other serving benchmark reports
    # it, under this name or as time-between-tokens.
    tpot = []
    for r in latency_ok:
        n_out = r.get("completion_tokens")
        t, e = r.get("ttft_ms"), r.get("e2e_ms")
        if n_out and n_out > 1 and t is not None and e is not None and e >= t:
            tpot.append((e - t) / (n_out - 1))
    if tpot:
        summary["tpot_ms"] = _pct_table(tpot)
        summary["tpot_note"] = (
            "time per output token after the first, (e2e - ttft) / "
            "(output_tokens - 1). latency for a longer answer is roughly "
            "ttft + tpot * output_tokens, so this is the number that says "
            "whether a longer generation still fits the budget. computed "
            f"over the {len(tpot)} requests that produced more than one token")

    answers = _answer_block(results)
    if answers:
        summary["answers"] = answers
    for fld in ("ttfr_ms", "ttfv_ms"):
        vals = [r.get(fld) for r in latency_ok]
        if any(v is not None for v in vals):
            summary[fld] = _pct_table(vals)
            # a reasoning model that runs out of max_tokens mid-thought
            # returns a successful response with no visible content at all.
            # those rows carry no ttfv, so the percentiles above describe
            # only the requests that finished thinking soonest. that is the
            # same survivorship the error path already guards against, and
            # it is worse here because nothing failed.
            summary[fld]["missing"] = sum(1 for v in vals if v is None)
            summary[fld]["of"] = len(vals)
    # Latency as the caller experienced it includes time the scheduled
    # request waited in the load generator. SLA evaluation below prefers these
    # tables; the service-time tables remain available for endpoint diagnosis.
    # TTFV must be corrected too when first_visible is the configured TTFT.
    caller_fields = (
        ("ttft_ms", "caller_ttft_ms", "ttft_corrected_ms"),
        ("ttfv_ms", "caller_ttfv_ms", "ttfv_corrected_ms"),
        ("ttf_tool_call_ms", "caller_ttf_tool_call_ms",
         "ttf_tool_call_corrected_ms"),
        ("e2e_ms", "caller_e2e_ms", "e2e_corrected_ms"),
    )
    exact_caller_n = 0
    reconstructed_caller_n = 0
    for base_f, caller_f, corr_f in caller_fields:
        vals = []
        for r in latency_ok:
            if caller_f in r:
                if r.get(caller_f) is not None:
                    vals.append(r[caller_f])
                    exact_caller_n += 1
            elif (r.get(base_f) is not None
                  and r.get("queue_wait_ms") is not None):
                vals.append(r[base_f] + r["queue_wait_ms"])
                reconstructed_caller_n += 1
        if vals:
            summary[corr_f] = _pct_table(vals)
    if any(k in summary for k in ("ttft_corrected_ms", "ttfv_corrected_ms",
                                  "ttf_tool_call_corrected_ms",
                                  "e2e_corrected_ms")):
        summary["latency_correction_note"] = (
            "caller-experienced figures measure from the exact monotonic "
            "scheduled target through the observed event, including worker "
            "queueing, connection setup, retries and fallbacks. Legacy rows "
            "without exact clocks are reconstructed as service time plus "
            "queue wait. Configured latency targets and hard caps prefer these "
            "figures whenever available.")
        summary["latency_correction_provenance"] = {
            "exact_values": exact_caller_n,
            "legacy_reconstructed_values": reconstructed_caller_n,
        }
    reason_vals = [r.get("reasoning_tokens") for r in usage_rows]
    if any(v is not None for v in reason_vals):
        total = sum(v for v in reason_vals if v)
        summary["reasoning_tokens"] = _pct_table(reason_vals)
        summary["reasoning_tokens_total"] = total
        summary["reasoning_tokens_source"] = next(
            (r.get("reasoning_tokens_source") for r in usage_rows
             if r.get("reasoning_tokens_source")), None)
        if dur_min:
            summary["throughput"]["reasoning_tokens_per_min"] = total / dur_min
    if summary.get("reasoning_tokens_total") is None:
        # endpoint did not report a reasoning-token count (some models do
        # not). fall back to counting reasoning_content deltas in the stream,
        # clearly labeled as an estimate.
        chunk_rows = [r for r in results if _protocol_clean_success(r)]
        chunk_vals = [r.get("reasoning_chunks") for r in chunk_rows]
        if any(chunk_vals):
            ctotal = sum(v for v in chunk_vals if v)
            summary["reasoning_stream_deltas"] = _pct_table(chunk_vals)
            summary["reasoning_stream_deltas_total"] = ctotal
            summary["reasoning_stream_deltas_source"] = \
                "counted reasoning_content SSE deltas (not token counts)"
            if dur_min:
                summary["throughput"]["reasoning_stream_deltas_per_min"] = \
                    ctotal / dur_min
    n_ok = len(latency_ok)
    # a quantile needs enough observations ABOVE it to be an estimate rather
    # than an anecdote. at n=100 there is a 37 percent chance of drawing no
    # sample at all beyond the true p99, so the old "100 is fine for p99"
    # threshold was not defensible. the rule here is roughly ten
    # observations past the quantile: n >= 10/(1-q).
    _need = {"p50": 20, "p90": 100, "p95": 200, "p99": 1000}
    _unsupported = [q for q, need in _need.items() if n_ok < need]
    if n_ok == 0:
        sample_warning = ("no successful requests, so there are no latency "
                          "numbers to read. check the failures block")
    elif _unsupported:
        sample_warning = (
            f"{n_ok} successful requests supports "
            + (", ".join(q for q in _need if q not in _unsupported)
               or "no quantile")
            + ". " + ", ".join(_unsupported) + " "
            + ("is" if len(_unsupported) == 1 else "are")
            + " indicative only, since a quantile needs roughly ten "
            "observations past it to be an estimate. "
            + f"reach {min(_need[q] for q in _unsupported)} for the next one")
    else:
        sample_warning = None
    summary["sample"] = {
        "n": n_ok,
        "supports": [q for q in _need if q not in _unsupported],
        "indicative_only": _unsupported,
        "warning": sample_warning,
    }
    # the client is part of the instrument. if it could not deliver the load
    # it was asked for, the endpoint was never tested at that rate, and every
    # latency number below describes a lighter load than the one on the label.
    # NOT schedule_meta["rate_p50"]. that is the median of the rate curve, so
    # on a bursty schedule it is the quiet rate rather than the offered one,
    # and shard() does not rescale it, so every sharded run would read as a
    # shortfall. the rows carry their own schedule, which is invariant to both.
    # BOTH sides come from `stamped`. mixing populations makes the ratio the
    # non-retry fraction, so a run with many endpoint-caused retries would
    # read as a client shortfall, which is the mirror of the bug the retry
    # exclusion exists to prevent.
    # the RATIO is computed over `stamped`, so one outlier send cannot skew
    # it. the PRINTED rates count every scheduled row, so "delivered" lines
    # up with the achieved arrival rate in the believability block rather
    # than being quietly scaled down by the retry fraction.
    offered = None
    all_sched = [r["scheduled_s"] for r in results
                 if r.get("scheduled_s") is not None]
    if len(all_sched) > 1:
        span_all = max(all_sched) - min(all_sched)
        if span_all > 0:
            # n-1 intervals across n arrivals
            offered = (len(all_sched) - 1) / span_all
    # measure the achieved rate over the same population as wire lateness.
    # a single retried request stamps its LAST attempt, which can stretch the
    # run's apparent span by a read timeout and halve the apparent rate.
    achieved = summary["arrivals"]["achieved_qps_overall"]
    stretch = None
    if len(stamped) > 1 and offered:
        sends = [_sent_at(r) for r in stamped]
        scheds = [r["scheduled_s"] for r in stamped]
        span_send = max(sends) - min(sends)
        span_sched = max(scheds) - min(scheds)
        if span_send > 0 and span_sched > 0:
            stretch = span_send / span_sched
            achieved = offered / stretch
    wire_p95 = (summary["arrivals"]["wire_lateness_ms"] or {}).get("p95")
    short = bool(offered and achieved and achieved < offered * 0.8)
    drifting = bool(wire_p95 and wire_p95 > 1000.0)
    if short or drifting:
        parts, conclusion = [], []
        if short:
            parts.append(
                f"the schedule asked for about {offered:.1f} requests/second "
                f"over the run and {achieved:.1f} was delivered")
            conclusion.append(
                "the run delivered fewer requests per second than the "
                "schedule asked for, so these latency numbers describe a "
                "lighter load than the one on the label")
        if drifting:
            lp = (f"{wire_p95 / 1000:.1f}s" if wire_p95 < 10_000
                  else f"{wire_p95 / 1000:.0f}s")
            parts.append(
                f"95 percent of requests reached the endpoint within {lp} of "
                f"their scheduled time, the rest later")
            if not short:
                conclusion.append(
                    "the run-average rate stayed within 20 percent of the "
                    "schedule, so the load did arrive, but it arrived "
                    "reshaped: the instantaneous rate the endpoint saw is not "
                    "the one the schedule describes")
        summary["client"] = {
            "offered_qps": offered, "achieved_qps": achieved,
            "wire_lateness_p95_ms": wire_p95,
            "warning": (
                f"{'. '.join(parts)}. {'. '.join(conclusion)}. the offered "
                "load did not reach the endpoint on schedule, either because "
                "the client could not keep up or because the endpoint slowed "
                "and back-pressured the pool. read the stability card to tell "
                "them apart, since a client-side limit leaves endpoint latency "
                "flat. if it is the client, raise max_concurrency, lower the "
                "rate, or shard the schedule across machines. dispatch lag "
                "stays small either way, because a full pool queues rather "
                "than blocking the dispatcher."
),
        }

    conc = _concurrency_block(results, concurrency_target
                              or safe_run_meta.get("sizing_concurrency_requested")
                              or safe_run_meta.get("concurrency_target"))
    if conc:
        summary["concurrency"] = conc

    summary["drift"] = _drift_block(latency_ok, failed)

    # every report states which harness produced it and what the latency
    # numbers include. 0.3.0 moved the TCP/TLS handshake out of the timed
    # region, so a 0.2.x TTFT and a 0.3.x TTFT are not the same measurement
    # and must not be put in one column.
    summary["harness_version"] = __version__
    summary["latency_basis"] = (
        "final-attempt clocks begin immediately before conn.request on an "
        "already-established connection, so they include request upload. "
        "ttfb ends at the first iterated response-body line (not necessarily "
        "the first response byte). ttft ends at the first nonempty visible or "
        "reasoning content delta and excludes tool-call fragments; first "
        "visible content and first tool-call fragment remain separate metrics. "
        "TCP and TLS setup is measured separately as connect_ms and is NOT "
        "included. changed in 0.3.0: 0.2.x and earlier included connection "
        "setup in these numbers.")

    # prompts mode cycles the supplied prompts (runner: prompt_msgs[i % m]).
    # once the set has been through once, every later request is a verbatim
    # repeat, which makes them eligible for endpoint prompt-cache reuse. the
    # fraction then describes the replay, not the caller's production mix.
    rm = safe_run_meta
    pc = rm.get("prompts_count")
    if rm.get("input_mode") == "prompts" and pc:
        repeats = (n_ok / pc) if pc else 0.0
        summary["replay"] = {
            "distinct_prompts": pc,
            "requests": n_ok,
            "avg_sends_per_prompt": repeats,
            "repeat_requests": max(0, n_ok - pc),
            "repeat_share": (max(0, n_ok - pc) / n_ok) if n_ok else 0.0,
            "warning": (
                f"{pc} distinct prompts covered {n_ok} requests, so "
                f"{max(0, n_ok - pc)} of them "
                f"({max(0, n_ok - pc) / n_ok * 100:.0f} percent) repeat a "
                f"prompt already sent and are eligible for endpoint prompt "
                f"cache reuse. treat the reported cached prompt-token fraction "
                f"and TTFT as replay "
                f"behavior, not your production prompt mix. supply at least "
                f"as many distinct prompts as requests, or read only the "
                f"first {pc} requests, to see cold behavior."
                if n_ok > pc else None),
        }
    if pricing:
        summary["cost"] = _cost_block(results, dur, in_tok, out_tok, cached_tok,
                                      pricing)
    if acceptance:
        summary["sla"] = _evaluate_sla(ok, len(results), summary, acceptance,
                                       ttft_definition)
    return summary


def _drift_block(ok: list[dict], failed: list[dict] | None = None,
                 window_s: int = 60, min_window_n: int = 20) -> dict:
    """Per-window errors and p95 over the run, and whether it held steady.

    Two questions, two gates. "Was the endpoint erroring" is answered from
    attempted requests, so a window that lost everything still reaches the
    verdict rather than vanishing for having no p95. "Did latency move" is
    answered from successful requests, and a window that shed more than a
    fifth of its requests is left out of that comparison, because a p95 over
    survivors is not a latency measurement.

    `failed` is optional so existing single-argument callers keep working.
    The latency verdict needs two counted windows to say anything and three
    before it names a direction, since two points cannot separate a trend
    from noise.
    """
    if not ok:
        n_failed = len([f for f in (failed or [])
                        if f.get("t_send_unix") is not None])
        if n_failed:
            return {
                "windows": [], "window_seconds": window_s,
                "drift_kind": "failing", "drift_flag": True,
                "drift_headline": (
                    f"every request failed ({n_failed} of them). there is no "
                    "latency to report, and nothing here is a performance "
                    "result. read the failures block"),
                "note": "no successful requests",
            }
        return {"windows": [], "note": "no successful requests"}
    failed = failed or []
    # a row with no send stamp cannot be placed in a window. failures were
    # already filtered for it; successes were not, and a pooled or
    # hand-built input without the field raised a KeyError here.
    ok = [r for r in ok if r.get("t_send_unix") is not None]
    everything = ok + [f for f in failed if f.get("t_send_unix") is not None]
    if not everything:
        return {"windows": [], "note": "no request carried a send time, so "
                                       "stability cannot be judged"}
    t0 = min(r["t_send_unix"] for r in everything)
    buckets: dict[int, list] = {}
    errs: dict[int, int] = {}
    for r in ok:
        w = int((r["t_send_unix"] - t0) // window_s)
        buckets.setdefault(w, []).append(r)
    # failures get their own count per window. an endpoint that collapses
    # serves fewer successes, and those survivors are often the fast ones, so
    # looking at successes alone reads a breakdown as "it got faster".
    for r in failed:
        if r.get("t_send_unix") is None:
            continue
        w = int((r["t_send_unix"] - t0) // window_s)
        buckets.setdefault(w, [])
        errs[w] = errs.get(w, 0) + 1
    short = {"windows": [], "window_seconds": window_s,
             "note": f"run shorter than two {window_s}s windows, cannot show "
                     "drift. run for minutes to test sustained acceptance "
                     "targets."}
    if len(buckets) < 2:
        return short
    rows = []
    for w in sorted(buckets):
        rs = buckets[w]
        tt = [x.get("ttft_ms") for x in rs if x.get("ttft_ms") is not None]
        ee = [x.get("e2e_ms") for x in rs if x.get("e2e_ms") is not None]
        e = errs.get(w, 0)
        attempts = len(rs) + e
        rows.append({
            "window": w, "n": len(rs), "errors": e, "attempts": attempts,
            "error_rate": (e / attempts) if attempts else 0.0,
            "ttft_p95": float(np.percentile(tt, 95)) if tt else None,
            "e2e_p95": float(np.percentile(ee, 95)) if ee else None,
        })
    # a window has to be big enough, both absolutely and relative to the rest
    # of the run, before its p95 is allowed to move the verdict.
    # true median, and cap the relative term so one very large window cannot
    # push the bar high enough to discard otherwise usable windows.
    # two different questions need two different gates.
    #
    # "was the endpoint erroring" is answered from ATTEMPTS, because a window
    # that lost every request has no p95 at all and would otherwise vanish.
    # "did latency move" is answered from SUCCESSES, because a p95 over a
    # handful of survivors is not a latency measurement.
    med_att = float(np.median([r["attempts"] for r in rows]))
    err_floor = max(min_window_n, min(0.25 * med_att, 50.0))
    med_ok = float(np.median([r["n"] for r in rows]))
    p95_floor = max(min_window_n, min(0.25 * med_ok, 50.0))
    for r in rows:
        # a window that shed heavily is evidence regardless of size. a
        # trailing partial window is exactly where a breaking-point run ends,
        # and sizing it out would hide the thing being looked for.
        r["error_counted"] = bool(
            r["attempts"] >= err_floor
            or (r["errors"] >= 5 and r["error_rate"] > 0.20))
        # a window that shed requests reports a p95 over survivors only, and
        # survivors skew fast. it must not anchor the latency comparison, or
        # the fastest number in the table is the one the endpoint produced
        # while falling over.
        # a higher bar than the failing verdict on purpose. losing a few
        # percent still leaves a p95 worth comparing, losing a fifth does not.
        r["p95_survivorship"] = bool(r["error_rate"] > 0.20)
        r["counted"] = bool(r["n"] >= p95_floor
                            and r["ttft_p95"] is not None
                            and not r["p95_survivorship"])
    err_counted = [r for r in rows if r["error_counted"]]
    counted = [r for r in rows if r["counted"]]
    skipped = len(rows) - len(counted)
    note = ("per-window counts, errors and p95. two rules decide the verdict. "
            "first, the run is failing when one window lost more than 5 "
            "percent of its requests while the others held, or when every "
            "window is losing more than 10 percent, because a p95 over "
            "survivors is not a latency result. otherwise the run is "
            "unstable when the worst "
            "counted window's TTFT p95 is more than 1.3x the best, in either "
            "direction, so warmup and mid-run spikes both show up. E2E p95 is "
            "printed alongside but not scored. a window is left out of the "
            f"latency comparison when it has fewer than {p95_floor:.0f} "
            "successful requests, when no request returned a content delta, "
            "or "
            "when it lost more than a fifth of its requests.")
    worst_err = max((r["error_rate"] for r in err_counted), default=0.0)
    base_err = min((r["error_rate"] for r in err_counted), default=0.0)
    # two ways to be failing: one window fell over while the rest held, or the
    # whole run sits past the knee and every window sheds requests. the second
    # needs an absolute test, since uniform loss has no delta.
    failing = bool(worst_err > 0.05
                   and (worst_err > base_err + 0.05 or base_err > 0.10))
    if failing:
        # name the window where the most requests actually died, not the
        # highest percentage: a 6-request tail at 100 percent is noise next
        # to a 165-request window at 84 percent. but only windows that
        # themselves trip the bar are eligible, or a huge window with a
        # rounding-error rate could be named and print "failed 0 percent".
        eligible = [r for r in err_counted if r["error_rate"] > 0.05]
        bad_w = max(eligible or err_counted,
                    key=lambda r: (r["errors"], r["error_rate"]))
        also = ""
        if bad_w["error_rate"] < worst_err:
            top = max(err_counted, key=lambda r: r["error_rate"])
            also = (f" the highest loss rate was window {top['window']} at "
                    f"{top['error_rate'] * 100:.0f} percent.")
        return {
            "windows": rows, "window_seconds": window_s,
            "counted_windows": len(counted), "skipped_windows": skipped,
            "worst_window_error_rate": worst_err,
            "drift_kind": "failing", "drift_flag": True,
            "drift_headline": (
                f"window {bad_w['window']} failed "
                f"{bad_w['error_rate'] * 100:.0f} percent of its requests. "
                "latency percentiles only cover requests that came back, so "
                "the surviving numbers in that window describe what the "
                "endpoint could still serve, not what it was asked for. read "
                "this as a breaking point, not a latency result." + also
                + " the window-to-window latency comparison is not reported "
                "for a failing run"),
            "note": note,
        }
    if len(counted) < 2:
        errs_dominate = any(r["error_rate"] > 0.05 for r in rows)
        return {"windows": rows, "window_seconds": window_s,
                "counted_windows": len(counted), "skipped_windows": skipped,
                "note": ("not enough windows carry a usable latency sample, "
                         "so stability cannot be judged. "
                         + ("requests were failing, so read the error rate "
                            "rather than running the same load for longer."
                            if errs_dominate else
                            "run longer, or raise the rate so each window "
                            "holds enough requests."))}

    vals = [r["ttft_p95"] for r in counted]
    first, last = vals[0], vals[-1]
    best, worst = min(vals), max(vals)
    ratio = (last / first) if first else None
    spread = (worst / best) if best else None
    unstable = bool(spread and spread > 1.3)
    rising = all(b >= a for a, b in zip(vals, vals[1:]))
    falling = all(b <= a for a, b in zip(vals, vals[1:]))
    if not unstable:
        kind = "stable"
        headline = "steady across the run"
    elif len(vals) < 3:
        kind = "variable"
        headline = ("two windows moved apart, which is not enough to call a "
                    "direction. run longer to tell a trend from noise")
    elif rising and worst == vals[-1]:
        kind = "degrading"
        headline = ("TTFT p95 rises across every counted window: the endpoint "
                    "got slower as the run went on")
    elif falling and worst == vals[0]:
        kind = "warming"
        headline = ("TTFT p95 is worst in the first window and falls from "
                    "there: early requests are cold start, not steady state. "
                    "quote the later windows or warm up before measuring")
    elif worst not in (vals[0], vals[-1]):
        kind = "spike"
        headline = ("a middle window is much worse than the ends: something "
                    "transient hit the endpoint mid-run")
    else:
        kind = "variable"
        headline = ("windows move up and down without a clear trend. the run "
                    "is noisy rather than drifting, so one p95 from it is not "
                    "a steady-state number")
    return {
        "windows": rows, "window_seconds": window_s,
        "counted_windows": len(counted), "skipped_windows": skipped,
        "ttft_p95_drift_ratio": ratio,
        "ttft_p95_spread_ratio": spread,
        "ttft_p95_best": best, "ttft_p95_worst": worst,
        "drift_kind": kind,
        "drift_headline": headline,
        "drift_flag": unstable,
        "note": note,
    }


def _cost_block(rows: list[dict], dur, in_tok: int, out_tok: int,
                cached_tok: int, pricing: dict) -> dict:
    """Diagnostic arithmetic over replay rows using unverified input rates.

    The harness does not fetch a price, bind the supplied rate to a commercial
    product, or observe provider billing for every physical POST.  Exact-looking
    aggregate fields are therefore emitted only when every logical replay row
    has a known zero-send outcome or one clean, single-attempt response with
    sane usage.  The applicability warning is unconditional until a future
    pricing schema seals provider/model/product/region/tier/date provenance.
    """
    mode = pricing.get("mode", "per_token")
    usd = pricing.get("usd_per_dbu")
    attempted = len(rows)
    successful = sum(_protocol_clean_success(r) for r in rows)
    indexed_usage_rows = [
        (i, r) for i, r in enumerate(rows) if _usage_is_trustworthy(r)]
    usage_rows = [r for _i, r in indexed_usage_rows]
    tok_total = sum(float(r["prompt_tokens"])
                    + float(r["completion_tokens"]) for r in usage_rows)
    usage_coverage = ((len(usage_rows) / attempted) if attempted
                      else (1.0 if tok_total else None))

    # A final response reports usage only for that response.  It cannot prove
    # whether an earlier POST reached the provider, generated tokens, or was
    # billed.  Keep these classes disjoint so contradictory rows such as
    # request_attempts=0 plus a retry marker can never be treated as unsent.
    def attempt_class(row: dict) -> str:
        value = row.get("request_attempts")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return "unknown"

        if "connection_attempts" in row:
            connections = row.get("connection_attempts")
            if (not isinstance(connections, int)
                    or isinstance(connections, bool)
                    or connections < value):
                return "unknown"

        retries_present = "retries" in row
        reasons_present = "retry_reasons" in row
        retries = row.get("retries")
        reasons = row.get("retry_reasons")
        if retries_present != reasons_present:
            return "unknown"
        if retries_present and (
                not isinstance(retries, int)
                or isinstance(retries, bool)
                or retries < 0):
            return "unknown"
        if reasons_present and (
                not isinstance(reasons, list)
                or any(not isinstance(reason, str) or not reason
                       for reason in reasons)):
            return "unknown"
        if retries_present and reasons_present and retries != len(reasons):
            return "unknown"
        if ((retries_present and retries > 0)
                or (reasons_present and reasons)):
            return "ambiguous"
        if value == 0:
            response_evidence = (
                row.get("ok") is True
                or _sent_at(row) is not None
                or row.get("status") is not None
                or any(row.get(name) is not None for name in (
                    "prompt_tokens", "completion_tokens", "total_tokens")))
            if response_evidence:
                return "unknown"
            return "unsent"
        if value == 1:
            return "single"
        return "ambiguous"

    attempt_classes = [attempt_class(r) for r in rows]
    known_unsent = attempt_classes.count("unsent")
    exact_single_attempts = attempt_classes.count("single")
    ambiguous_retry_rows = attempt_classes.count("ambiguous")
    unknown_attempt_rows = attempt_classes.count("unknown")
    exact_usage_rows = [
        r for i, r in indexed_usage_rows if attempt_classes[i] == "single"]

    def completeness(eligible_count: int) -> tuple[bool, float | None, list[str]]:
        complete = eligible_count + known_unsent == attempted
        coverage = ((eligible_count + known_unsent) / attempted
                    if attempted else None)
        gaps = []
        if ambiguous_retry_rows:
            gaps.append(
                f"{ambiguous_retry_rows} row(s) had multiple or retry-marked "
                "physical POSTs whose earlier billed usage is not observed")
        if unknown_attempt_rows:
            gaps.append(
                f"{unknown_attempt_rows} row(s) lacked exact physical-attempt "
                "accounting")
        missing_usage = exact_single_attempts - eligible_count
        if missing_usage:
            gaps.append(
                f"{missing_usage} single-POST row(s) did not have one clean, "
                "complete, internally sane usage response")
        return complete, coverage, gaps

    applicability_warning = (
        "rates were supplied by the operator and were not fetched or bound to "
        "a verified provider, model, commercial product, cloud, region, "
        "service tier, effective date, contract, or DBU-to-USD conversion. "
        "this is diagnostic rate arithmetic over measured replay rows, not a "
        "current Databricks price, invoice, or full-harness cost")

    if mode == "provisioned":
        dph = pricing.get("dbu_per_hour")
        if dph is None:
            return {"mode": mode, "error": "provisioned needs dbu_per_hour"}
        complete, coverage, gaps = completeness(len(exact_usage_rows))
        exact_tok_total = sum(
            float(r["prompt_tokens"]) + float(r["completion_tokens"])
            for r in exact_usage_rows)
        dur_hr = (dur / 3600.0) if dur else None
        tph = (exact_tok_total / dur_hr) if dur_hr and complete else None
        eff = (dph / (tph / 1e6)) if tph else None
        block = {"mode": "provisioned", "dbu_per_hour": dph,
                 "effective_dbu_per_1m_tokens": eff,
                 "tokens_measured": exact_tok_total if complete else None,
                 "tokens_measured_subset": tok_total,
                 "usage_coverage": usage_coverage,
                 "usage_rows": len(usage_rows),
                 "exact_single_usage_rows": len(exact_usage_rows),
                 "successful_rows": successful,
                 "attempted_rows": attempted,
                 "known_unsent_rows": known_unsent,
                 "ambiguous_retry_rows": ambiguous_retry_rows,
                 "unknown_attempt_rows": unknown_attempt_rows,
                 "coverage": coverage,
                 "complete": complete,
                 "scope": "measured_replay_interval_only",
                 "provenance_verified": False,
                 "applicability_warning": applicability_warning,
                 "coverage_warning": (
                     None if complete or not rows else
                     "; ".join(gaps) + ". effective provisioned cost per "
                     "token and its token-throughput denominator are "
                     "unavailable; final-response token usage is retained "
                     "only as a measured-subset diagnostic"),
                 "note": "provisioned throughput bills by capacity (DBU/hour), "
                         "not per token. effective cost per 1M tokens is the "
                         "hourly rate over tokens served per hour at the "
                         "measured replay throughput. the supplied hourly rate "
                         "and its applicability are unverified."}
        if usd is not None:
            block["usd_per_hour"] = dph * usd
            block["effective_usd_per_1m_tokens"] = (
                eff * usd if eff is not None else None)
            block["usd_per_dbu"] = usd
        return block

    inp = pricing.get("input_dbu_per_m")
    out = pricing.get("output_dbu_per_m")
    if inp is None or out is None:
        return {"mode": mode,
                "error": "per_token needs input_dbu_per_m and output_dbu_per_m"}
    cache = pricing.get("cache_read_dbu_per_m")
    cache = cache if cache is not None else inp
    # Missing cached_tokens is harmless only when cached and uncached input
    # have the same price. With a cache discount it is a required billing
    # field: treating missing as zero silently prices an unknown row at the
    # expensive rate and invents a total.
    indexed_priced_rows = [
        (i, r) for i, r in indexed_usage_rows
        if ((r.get("cached_tokens") is None and cache == inp)
            or (r.get("cached_tokens") is not None
                and 0 <= r["cached_tokens"] <= r["prompt_tokens"]))]
    priced_rows = [r for _i, r in indexed_priced_rows]
    per = []
    measured_cached = 0
    for r in priced_rows:
        pt = r["prompt_tokens"]
        ct = r.get("cached_tokens") or 0
        comp = r["completion_tokens"]
        uncached = max(pt - ct, 0)
        per.append(uncached / 1e6 * inp + ct / 1e6 * cache + comp / 1e6 * out)
        measured_cached += ct
    measured_total = sum(per)
    n = len(per)
    exact_single_rows = [
        r for i, r in indexed_priced_rows if attempt_classes[i] == "single"]
    complete, coverage, gaps = completeness(len(exact_single_rows))
    total = measured_total if complete else None
    block = {
        "mode": "per_token",
        "dbu_per_request": _pct_table(per),
        "priced_rows": n,
        "successful_rows": successful,
        "attempted_rows": attempted,
        "known_unsent_rows": known_unsent,
        "ambiguous_retry_rows": ambiguous_retry_rows,
        "unknown_attempt_rows": unknown_attempt_rows,
        "coverage": coverage,
        "complete": complete,
        "scope": "measured_replay_rows_only",
        "provenance_verified": False,
        "applicability_warning": applicability_warning,
        "dbu_total_measured_subset": measured_total,
        "dbu_total": total,
        "dbu_per_1k_requests": ((total / attempted * 1000)
                                 if complete and attempted else None),
        "dbu_per_min": ((total / (dur / 60.0))
                         if complete and dur else None),
        "cache_dbu_saved": (measured_cached / 1e6
                            * max(inp - cache, 0.0)) if complete else None,
        "rates_dbu_per_m": {"input": inp, "output": out, "cache_read": cache},
        "coverage_warning": (
            None if complete or not rows else
            "; ".join(gaps) + ". aggregate replay cost, "
            "cost per 1,000 requests, cost per minute and cache savings are "
            "unavailable; the measured subset is retained only for "
            "diagnosis"),
        "note": "arithmetic from clean endpoint-reported tokens times "
                "unverified user-supplied rates. cached input uses the "
                "supplied cache-read rate. setup, sizing, calibration and "
                "probe traffic are outside this replay-only block.",
    }
    if usd is not None:
        block["usd_per_dbu"] = usd
        block["usd_total"] = total * usd if total is not None else None
        block["usd_total_measured_subset"] = measured_total * usd
        block["usd_per_1k_requests"] = (block["dbu_per_1k_requests"] * usd
                                        if block["dbu_per_1k_requests"] is not None
                                        else None)
        block["usd_per_min"] = (block["dbu_per_min"] * usd
                                if block["dbu_per_min"] is not None else None)
        block["cache_usd_saved"] = (
            block["cache_dbu_saved"] * usd
            if block["cache_dbu_saved"] is not None else None)
    return block


def _evaluate_sla(ok: list[dict], total: int, summary: dict,
                  acceptance: dict,
                  ttft_definition: str = "first_content") -> dict:
    """Score the run against customer acceptance targets.

    Expected shape (all sections optional):
      ttft_ms:  {p50: 500, p90: 800, p95: 900, p99: 1600}
      ttfg_ms:  {p50: 700, ...}          evaluated against measured E2E
      hard_timeouts: {ttft_s: 15, ttfg_s: 45}   over-budget requests count
                                                as acceptance-target failures
      success_rate: 0.9999
    """
    stated = acceptance.get("targets_are")
    illustrative = bool(acceptance.get("note")
                        and "illustrative" in str(acceptance["note"]).lower())
    out: dict = {"targets_source": stated or "the run configuration",
                 "ttft_definition": ttft_definition,
                 "acceptance_config": _redact_secrets(acceptance)}
    if illustrative:
        out["targets_warning"] = (
            f"these targets came from {out['targets_source']} and are "
            "illustrative, so the pass and fail marks below score against "
            "example numbers rather than yours. pass your own with "
            "--ttft-p95 and --ttfg-p95, or put them in your profile.")

    def score(name, table_key, targets, service_key):
        rows = []
        for q, target in (targets or {}).items():
            actual = (summary.get(table_key) or {}).get(q)
            rows.append({
                "quantile": q, "target_ms": target,
                "actual_ms": round(actual, 1) if actual is not None else None,
                "met": (actual <= target) if actual is not None else None,
                "scored_metric": table_key,
                "service_metric": service_key,
            })
        out[name] = rows

    raw_ttft_key = ("ttft_ms" if ttft_definition == "first_content"
                    else "ttfv_ms")
    corrected_ttft_key = ("ttft_corrected_ms"
                          if ttft_definition == "first_content"
                          else "ttfv_corrected_ms")
    ttft_key = (corrected_ttft_key if (summary.get(corrected_ttft_key) or {}).get("n")
                else raw_ttft_key)
    ttfg_key = ("e2e_corrected_ms"
                if (summary.get("e2e_corrected_ms") or {}).get("n")
                else "e2e_ms")
    out["ttft_metric"] = ttft_key
    out["ttfg_metric"] = ttfg_key
    out["latency_basis"] = (
        "caller_experienced" if (ttft_key.endswith("_corrected_ms")
                                  or ttfg_key.endswith("_corrected_ms"))
        else "service_time_no_schedule_wait_available")
    score("ttft_vs_target", ttft_key, acceptance.get("ttft_ms"), raw_ttft_key)
    _miss = (summary.get(raw_ttft_key) or {}).get("missing") or 0
    _of = (summary.get(raw_ttft_key) or {}).get("of") or 0
    if _of and _miss / _of > 0.05:
        out["coverage_warning"] = (
            f"{_miss} of {_of} successful requests never produced the token "
            f"this scores ({raw_ttft_key}), so the marks below describe the "
            f"{_of - _miss} that did. those are the fastest ones. raise the "
            "output token budget until responses stop truncating, then "
            "re-run.")
    score("ttfg_vs_target", ttfg_key, acceptance.get("ttfg_ms"), "e2e_ms")

    # A partial corrected population is not safe to green-light: it can omit
    # precisely the requests that queued. Score what is available, but make
    # the missing caller timing an explicit validity warning.
    caller_gaps = []
    for raw_key, corrected_key, label in (
            (raw_ttft_key, corrected_ttft_key, "TTFT"),
            ("e2e_ms", "e2e_corrected_ms", "end-to-end")):
        raw_n = (summary.get(raw_key) or {}).get("n") or 0
        corrected_n = (summary.get(corrected_key) or {}).get("n") or 0
        if raw_n and corrected_n < raw_n:
            caller_gaps.append(f"{label} caller timing exists for "
                               f"{corrected_n} of {raw_n} measured answers")
    if caller_gaps:
        out["caller_latency_warning"] = (
            "; ".join(caller_gaps)
            + ". caller-experienced acceptance targets cannot be proven from "
              "that coverage")

    hard = acceptance.get("hard_timeouts") or {}
    ttft_cap = (hard.get("ttft_s") or 0) * 1000.0
    ttfg_cap = (hard.get("ttfg_s") or 0) * 1000.0
    inter_cap = acceptance.get("interchunk_ms")
    timeouts = inter_breaches = 0
    failing = set()
    for idx, r in enumerate(ok):
        first = r.get(raw_ttft_key)
        end = r.get("e2e_ms")
        caller_first_key = ("caller_ttft_ms"
                            if ttft_definition == "first_content"
                            else "caller_ttfv_ms")
        if caller_first_key in r:
            first_for_caller = r.get(caller_first_key)
        elif first is not None and r.get("queue_wait_ms") is not None:
            first_for_caller = first + r["queue_wait_ms"]
        else:
            first_for_caller = first
        if "caller_e2e_ms" in r:
            end_for_caller = r.get("caller_e2e_ms")
        elif end is not None and r.get("queue_wait_ms") is not None:
            end_for_caller = end + r["queue_wait_ms"]
        else:
            end_for_caller = end
        missing_visible_breach = bool(
            ttft_cap and ttft_definition == "first_visible"
            and "visible_content_seen" in r
            and not r.get("visible_content_seen"))
        over_time = bool(
            missing_visible_breach
            or (ttft_cap and first_for_caller is not None
                and first_for_caller > ttft_cap)
            or (ttfg_cap and end_for_caller is not None
                and end_for_caller > ttfg_cap))
        over_inter = bool(inter_cap) and r.get("interchunk_max_ms") is not None \
            and r["interchunk_max_ms"] > inter_cap
        if over_time:
            timeouts += 1
        if over_inter:
            inter_breaches += 1
        if over_time or over_inter:
            failing.add(idx)
        # a request that came back 200 with nothing readable is not a
        # success at any target. rows written before this was recorded
        # do not carry the field, and are left alone.
        if "visible_content_seen" in r and not _answered(r):
            failing.add(idx)
    out["hard_timeout_breaches"] = timeouts
    out["hard_timeout_basis"] = {
        "ttft_metric": raw_ttft_key,
        "ttft_cap_ms": ttft_cap or None,
        "ttfg_cap_ms": ttfg_cap or None,
        "interchunk_cap_ms": inter_cap,
        "includes_client_queue_wait": any(
            r.get("queue_wait_ms") is not None for r in ok),
        "prefers_exact_monotonic_caller_clocks": True,
        "missing_first_visible_counts_as_breach": (
            ttft_definition == "first_visible"),
    }
    if inter_cap is not None:
        out["interchunk_breaches"] = inter_breaches

    target_sr = acceptance.get("success_rate")
    if target_sr and total:
        successes = len(ok) - len(failing)
        actual_sr = successes / total
        lower_95 = _wilson_lower_95(successes, total)
        out["success_rate"] = {
            "target": target_sr,
            "actual": round(actual_sr, 6),
            "met": actual_sr >= target_sr,
            "successes": successes,
            "attempts": total,
            "one_sided_95pct_wilson_lower": round(lower_95, 6),
            "statistically_demonstrated": lower_95 >= target_sr,
            "note": "failures, hard-timeout breaches, interchunk breaches, "
                    "and responses that returned 200 with neither visible "
                    "content nor a structurally valid tool call count against "
                    "it. a clean benchmark verdict also requires the "
                    "one-sided 95% Wilson lower confidence bound to meet the "
                    "target; this assumes request outcomes are independent",
        }
    return out


def _top_errors(failed: list[dict], k: int = 5) -> dict:
    counts: dict[str, int] = {}
    for r in failed:
        # Error bodies are deliberately represented by digests in request
        # rows. Those digests vary across otherwise identical 429 responses,
        # so grouping quota failures only by the error string fragments the
        # most important operational signal. Preserve the detailed rows while
        # giving every 429 one stable aggregate key.
        key = ("http 429 (rate limited)" if _http_status(r) == 429
               else (r.get("error") or "unknown")[:80])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


def _http_status(row: dict) -> int | None:
    """Return a real HTTP status code, rejecting bools and loose coercion."""
    value = row.get("status")
    if isinstance(value, int) and not isinstance(value, bool) \
            and 100 <= value <= 599:
        return value
    return None


def _failures_by_http_status(failed: list[dict]) -> dict[str, int]:
    """Stable failure counts that survive varying redacted body digests."""
    counts: dict[int, int] = {}
    for row in failed:
        status = _http_status(row)
        if status is not None:
            counts[status] = counts.get(status, 0) + 1
    # String keys are stable before and after a JSON serialization round-trip.
    return {str(status): counts[status] for status in sorted(counts)}


def _http_429_evidence(rows: list[dict], *, scope: str) -> dict:
    """Count 429s over the complete request-row evidence supplied to metrics.

    ``rate_limit_results`` includes setup requests as well as measured replay.
    When it is available, use it so a throttled preflight or probe cannot be
    hidden by a later clean replay phase.
    """
    request_rows = [row for row in rows if isinstance(row, dict)]
    limited = [row for row in request_rows if _http_status(row) == 429]
    phases: dict[str, int] = {}
    for row in limited:
        phase = str(row.get("phase") or "unlabeled")
        phases[phase] = phases.get(phase, 0) + 1
    total = len(request_rows)
    observed = sum(_http_status(row) is not None for row in request_rows)
    count = len(limited)
    return {
        "count": count,
        "rate": count / total if total else None,
        "rate_denominator": "all supplied logical request rows",
        "request_rows_examined": total,
        "http_status_observed_for": observed,
        "phases": {name: phases[name] for name in sorted(phases)},
        "scope": scope,
        "quota_limited": bool(count),
        "endpoint_capacity_conclusion_allowed": not bool(count),
        "note": (
            "HTTP 429 establishes rate limiting, not which quota dimension "
            "or component enforced it. provider telemetry is required for "
            "that attribution and for any endpoint-capacity conclusion"
            if count else
            "no HTTP 429 was present in the supplied request rows; absence "
            "does not establish provider quota headroom"),
    }


def _err_cell(w: dict) -> str:
    """Per-window errors as count and share, shared by both renderers."""
    if not w.get("errors"):
        return "0"
    return f"{w['errors']} ({w['error_rate'] * 100:.0f}%)"


def _wire_p95(arr: dict) -> str:
    """How late the client began sending, versus the schedule. Unlike
    dispatch lag, this grows when the offered load is not being delivered."""
    v = (arr.get("wire_lateness_ms") or {}).get("p95")
    if v is None:
        return "n/a"
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def _lag_p95(arr: dict) -> str:
    """Dispatch lag p95, where a measured 0.0 is a real value and a missing
    one is not. `or` would collapse the two."""
    v = (arr.get("dispatch_lag_ms") or {}).get("p95")
    return "n/a" if v is None else f"{v:.0f}"


def _traffic_phase_summary(traffic_scope: dict) -> str:
    """Describe sealed request coverage without calling missing time 'unsent'."""
    parts = []
    for name, details in sorted(
            (traffic_scope.get("phases") or {}).items()):
        captured = details.get("rows", 0)
        timestamped = details.get("sent_rows", 0)
        unknown = details.get("unknown_outcome_rows", 0)
        text = (f"{name}: {captured} captured, {timestamped} send-"
                "timestamped")
        if unknown:
            text += f", {unknown} send timing/outcome unknown"
        attempts = details.get("physical_attempts_estimate", 0)
        if attempts:
            exact = details.get("attempt_counts_exact") is True
            text += (f", {attempts} physical POST attempt"
                     f"{'s' if attempts != 1 else ''} "
                     f"{'recorded' if exact else 'estimated'}")
        parts.append(text)
    return "; ".join(parts) or "none"


def render_markdown(summary: dict, title: str, *,
                    verification_context: dict | None = None) -> str:
    s = summary
    from .markdown import markdown_plain_text
    from .report_decision import build_report_decision
    verified_view = _external_report_context(s, verification_context)

    def inline(value) -> str:
        """One Markdown line; customer-controlled metadata cannot add blocks."""
        return markdown_plain_text(value)

    def row(name, t):
        if not t or t.get("n", 0) == 0:
            return f"| {name} | - | - | - | - | 0 |"
        return (f"| {name} | {t['p50']:.0f} | {t['p90']:.0f} | "
                f"{t['p95']:.0f} | {t['p99']:.0f} | {t['n']} |")

    ach = s["achieved_cache_fraction"]
    ach_line = ("NOT REPORTED BY ENDPOINT"
                if ach.get("n", 0) == 0 else
                f"p50 {ach['p50']:.3f} / p95 {ach['p95']:.3f} "
                f"(fields: {', '.join(ach['source_fields'])}, "
                f"n={ach['reported_for_n']})")
    intent = s["intended_cache_fraction"]
    tt = s["token_targeting"]
    arr = s["arrivals"]
    sched_src = (s.get("schedule") or {}).get("source", "synthetic")
    mode = (s.get("run") or {}).get("input_mode", "profile")

    # disqualifiers go ABOVE the tables. report.md is the file that gets pasted
    # into a ticket, and a caution printed below the numbers is one nobody
    # reads. same rule the comparison report follows.
    cautions: list[str] = []
    _nw = (s.get("network_path") or {}).get("warning")
    if _nw:
        cautions += [f"CAUTION (network distance): {inline(_nw)}", ""]
    _cw = (s.get("throughput") or {}).get("coverage_warning")
    if _cw:
        cautions += [f"CAUTION (token usage): {inline(_cw)}", ""]
    _costw = (s.get("cost") or {}).get("coverage_warning")
    if _costw:
        cautions += [f"CAUTION (cost coverage): {inline(_costw)}", ""]
    _costa = (s.get("cost") or {}).get("applicability_warning")
    if _costa:
        cautions += [f"CAUTION (pricing applicability): {inline(_costa)}", ""]
    _cachew = (s.get("cache_fidelity") or {}).get("warning")
    if _cachew:
        cautions += [f"CAUTION (cache fidelity): {inline(_cachew)}", ""]
    _tokenw = (s.get("token_targeting") or {}).get("warning")
    if _tokenw:
        cautions += [f"CAUTION (workload token fidelity): {inline(_tokenw)}", ""]
    _popw = (s.get("latency_population") or {}).get("warning")
    if _popw:
        cautions += [f"CAUTION (latency population): {inline(_popw)}", ""]
    _sw = (s.get("sample") or {}).get("warning")
    if _sw:
        cautions += [f"CAUTION (sample size): {inline(_sw)}", ""]
    _rw = (s.get("replay") or {}).get("warning")
    if _rw:
        cautions += [f"CAUTION (prompt replay): {inline(_rw)}", ""]
    _cw = (s.get("client") or {}).get("warning")
    if _cw:
        cautions += [f"CAUTION (client saturation): {inline(_cw)}", ""]
    _nw = (s.get("concurrency") or {}).get("warning")
    if _nw:
        cautions += [f"CAUTION (concurrency not reached): {inline(_nw)}", ""]
    _ratew = (s.get("rate_limits") or {}).get("warning")
    if _ratew:
        cautions += [f"CAUTION (rate-limit evidence): {inline(_ratew)}", ""]
    _http429 = s.get("http_429") or {}
    _http429_count = s.get("http_429_count")
    if isinstance(_http429_count, int) \
            and not isinstance(_http429_count, bool) \
            and _http429_count > 0:
        _http429_total = _http429.get("request_rows_examined")
        _http429_of = (f" of {_http429_total}"
                       if isinstance(_http429_total, int)
                       and _http429_total >= _http429_count else "")
        cautions += [
            f"INVALID (quota-limited): {_http429_count}{_http429_of} request "
            f"{'row returned' if _http429_count == 1 else 'rows returned'} "
            "HTTP 429. No endpoint-capacity conclusion can be drawn; use "
            "provider telemetry to identify the enforcing limit and dimension.",
            "",
        ]

    decision = (verified_view["decision"] if verified_view
                else build_report_decision(s))
    decision_rows = []
    for heading, key in (
            ("Evidence integrity", "evidence_integrity"),
            ("Measurement validity", "measurement_validity"),
            ("Acceptance checks", "customer_sla"),
            ("Quota state", "quota_state"),
            ("Endpoint capacity", "endpoint_capacity")):
        state = decision[key]
        decision_rows.append(
            f"| {heading} | {inline(state['label'])} | "
            f"{inline(state['reason'])} |")

    verified_intro = []
    if verified_view:
        source_repro = verified_view["source_reproducibility"]
        verifier_repro = verified_view["verifier_reproducibility"]
        verified_intro = [
            f"> **{verified_view['view_label']}**",
            ">",
            "> Integrity: **VERIFIED** (internal SHA-256 consistency)  ",
            f"> Source reproducibility: **{source_repro['code']}** - "
            f"{inline(source_repro['reason'])}"
            + (" (reason codes: "
               + ", ".join(inline(code) for code in source_repro[
                   "reason_codes"]) + ")"
               if source_repro["reason_codes"] else "") + "  ",
            f"> Verifier reproducibility: **{verifier_repro['code']}** - "
            f"{inline(verifier_repro['reason'])}"
            + (" (reason codes: "
               + ", ".join(inline(code) for code in verifier_repro[
                   "reason_codes"]) + ")"
               if verifier_repro["reason_codes"] else "") + "  ",
            f"> Source artifact: `{inline(verified_view['source_artifact_id'])}`  ",
            f"> Source manifest SHA-256: "
            f"`{verified_view['source_manifest_sha256']}`  ",
            f"> Verified by llm-traffic-replay "
            f"`{inline(verified_view['verifier_version'])}` at "
            f"`{inline(verified_view['verified_at_utc'])}`.  ",
            f"> Receipt: `{inline(verified_view['receipt_id'])}`  ",
            f"> {inline(verified_view['assurance'])}",
            "",
        ]
    lines = [
        f"# {inline(title)}",
        "",
        *verified_intro,
        "## Decision states",
        "",
        "These states are independent. A quota-limited run can still retain "
        "its separately observed acceptance outcome; no single traffic light erases "
        "another fact.",
        "",
        "| decision | state | reason |",
        "|---|---|---|",
        *decision_rows,
        "",
        "Claim boundary: observed tested-load facts do not establish an "
        "endpoint ceiling or provider quota headroom.",
        "",
        f"measured replay: {s['requests_total']} requests, "
        f"{s['requests_ok']} harness-successful, "
        f"{s['requests_failed']} failed "
        f"(replay error rate {100 * (s['error_rate'] or 0):.2f}%)",
        "",
        *cautions,
        f"latency population: "
        f"{(s.get('latency_population') or {}).get('note', 'not recorded')}",
        "",
        "| endpoint service metric (ms, from send) | p50 | p90 | p95 | p99 | n |",
        "|---|---|---|---|---|---|",
        row("TTFT", s["ttft_ms"]),
        row("TTF valid tool call", s.get("ttf_tool_call_ms")),
        row("TTFB", s["ttfb_ms"]),
        row("TTFG (E2E)", s["e2e_ms"]),
        row("interchunk max", s["interchunk_max_ms"]),
        "",
        "## Believability block (read before quoting any number above)",
        f"- achieved cached prompt-token fraction, endpoint-reported: "
        f"{ach_line}",
        ("- input: real prompts replayed verbatim, sizes and any cache "
         "reuse are the prompts' own"
         if mode == "prompts" else
         f"- constructed (intended) cache fraction: "
         f"p50 {intent['p50']:.3f} / p95 {intent['p95']:.3f}"
         if intent.get("n") else "- constructed cache fraction: n/a"),
        ("- token targeting: n/a for real prompts (no synthetic size to hit)"
         if mode == "prompts" else
         f"- token targeting: reported/intended p50 = "
         f"{tt['reported_over_intended_p50']:.3f} "
         f"(abs error {tt['abs_error_pct_p50']:.1f}%)"
         if tt.get("reported_over_intended_p50") else
         "- token targeting: endpoint did not report prompt_tokens"),
        (f"- output tokens: finish_reasons "
         f"{json.dumps(tt.get('finish_reasons') or {})} "
         "(real prompts: no intended output size, only reported)"
         if mode == "prompts" else
         f"- output tokens: reported/intended p50 = "
         f"{tt['output_reported_over_intended_p50']:.3f} "
         f"(finish_reasons {json.dumps(tt.get('finish_reasons') or {})})"
         if tt.get("output_reported_over_intended_p50") else
         "- output tokens: endpoint did not report completion_tokens"),
        f"- achieved arrival rate: {arr['achieved_qps_overall']:.2f} QPS "
        f"overall, dispatch lag p95 "
        f"{_lag_p95(arr)} ms, wire lateness p95 "
        f"{_wire_p95(arr)}"
        + (f" ({arr['wire_lateness_note']})" if arr.get("wire_lateness_note")
           else "")
        if arr.get("achieved_qps_overall") else "- arrivals: n/a",
        f"- arrival schedule: from trace {sched_src}"
        if sched_src != "synthetic" else "- arrival schedule: synthetic bursts",
        f"- failures: {json.dumps(s['failures_by_error'])}"
        if s["requests_failed"] else "- failures: none",
        f"- failed requests by HTTP status: "
        f"{json.dumps(s.get('failures_by_http_status') or {})}"
        if s["requests_failed"] else
        "- failed requests by HTTP status: none",
        (f"- HTTP 429 rate-limit responses: {_http429_count} of "
         f"{_http429.get('request_rows_examined')} request rows "
         f"({100 * _http429.get('rate'):.2f}%); scope: "
         f"{inline(_http429.get('scope'))}. This is quota-limited evidence, "
         "not an endpoint-capacity result."
         if isinstance(_http429_count, int) and _http429_count > 0
         and isinstance(_http429.get("rate"), (int, float)) else
         "- HTTP 429 rate-limit responses: none observed in supplied evidence"),
        f"- requests that needed a connection retry: {s['requests_retried']} "
        "(retried requests restart their latency clock. a nonzero count "
        "here means the tail has survivorship bias, read with care)"
        if s.get("requests_retried") else "- connection retries: none",
    ]
    npth = s.get("network_path") or {}
    floor = _tcp_connect_floor(npth)
    if floor is not None:
        ratio = npth.get("tcp_connect_floor_to_ttft_p50_ratio")
        lines.append(
            f"- network-path floor: {floor:.0f} ms minimum TCP connect to "
            f"{npth['endpoint_host']} ({', '.join(npth['endpoint_ips'][:3])})"
            + (f", a floor-to-TTFT-p50 ratio of {ratio:.1%}"
               if ratio is not None else "")
            + ". this is a location diagnostic, not exact RTT or endpoint "
              "processing time; do not subtract it from TTFT")
    conn = s.get("connect_ms") or {}
    if conn.get("n"):
        lines.append(
            f"- connection setup (DNS, TCP and TLS, ms): p50 "
            f"{conn['p50']:.0f} / p95 {conn['p95']:.0f}. this is EXCLUDED "
            f"from ttft/ttfb/ttfg, do not subtract it again. a handshake is "
            f"several round trips, so it is not the per-request network cost "
            f"of a pooled production client, it is an upper bound on it")
    cc = s.get("concurrency") or {}
    if cc.get("in_flight_p50") is not None:
        sized = (f", open-loop sizing input "
                 f"{cc['sizing_concurrency_requested']}"
                 if cc.get("sizing_concurrency_requested") else "")
        lines.append(
            f"- concurrency actually in flight: p50 {cc['in_flight_p50']:.0f}, "
            f"p95 {cc['in_flight_p95']:.0f}, peak "
            f"{cc['in_flight_max']:.0f}{sized} "
            f"({cc['measured_over']})")
    tp = s.get("tpot_ms") or {}
    if tp.get("n"):
        lines.append(
            f"- time per output token (TPOT): p50 {tp['p50']:.1f} / p95 "
            f"{tp['p95']:.1f} ms. latency for a longer answer is roughly "
            f"ttft + tpot x output_tokens, so a {tp['p50']:.1f} ms TPOT puts "
            f"a 500-token answer near "
            f"{(s.get('ttft_ms') or {}).get('p50', 0) + tp['p50'] * 500:.0f} "
            "ms")

    if s.get("e2e_corrected_ms"):
        c1 = s.get("ttft_corrected_ms") or {}
        cv = s.get("ttfv_corrected_ms") or {}
        ct = s.get("ttf_tool_call_corrected_ms") or {}
        c2 = s["e2e_corrected_ms"]
        lines += ["", "### latency as the caller experienced it", "",
                  "Includes time the request waited on the client, so these "
                  "are what someone asking at the scheduled moment actually "
                  "waited.", "",
                  "| metric | p50 | p95 | p99 |", "|---|---|---|---|"]
        if c1.get("p50") is not None:
            lines.append(f"| TTFT corrected | {c1['p50']:.0f} | "
                         f"{c1['p95']:.0f} | {c1['p99']:.0f} |")
        if cv.get("p50") is not None:
            lines.append(f"| TTFV corrected | {cv['p50']:.0f} | "
                         f"{cv['p95']:.0f} | {cv['p99']:.0f} |")
        if ct.get("p50") is not None:
            lines.append(f"| TTF valid tool call corrected | "
                         f"{ct['p50']:.0f} | {ct['p95']:.0f} | "
                         f"{ct['p99']:.0f} |")
        lines.append(f"| end-to-end corrected | {c2['p50']:.0f} | "
                     f"{c2['p95']:.0f} | {c2['p99']:.0f} |")
        lines += ["", s["latency_correction_note"]]

    lb = s.get("latency_basis")
    if lb:
        lines.append(f"- latency basis: {lb}")

    _reason_source = str(s.get("reasoning_tokens_source") or "")
    _legacy_reasoning_deltas = (
        s.get("reasoning_tokens_total")
        if "stream-counted" in _reason_source.lower() else None)
    rt = (None if _legacy_reasoning_deltas is not None
          else s.get("reasoning_tokens_total"))
    if rt is not None:
        rtab = s.get("reasoning_tokens") or {}
        rpm = (s.get("throughput") or {}).get("reasoning_tokens_per_min")
        permin = f", {rpm:,.0f}/min" if rpm else ""
        lines.append(
            f"- reasoning tokens: {rt:,} total{permin}, p50 "
            f"{rtab.get('p50', 0):.0f} per request "
            f"(field: {s.get('reasoning_tokens_source')})")
    rd = (s.get("reasoning_stream_deltas_total")
          if s.get("reasoning_stream_deltas_total") is not None
          else _legacy_reasoning_deltas)
    if rd is not None:
        rtab = s.get("reasoning_stream_deltas") or {}
        rpm = ((s.get("throughput") or {}).get(
            "reasoning_stream_deltas_per_min")
            or ((s.get("throughput") or {}).get("reasoning_tokens_per_min")
                if _legacy_reasoning_deltas is not None else None))
        permin = f", {rpm:,.0f} deltas/min" if rpm else ""
        lines.append(
            f"- reasoning stream deltas: {rd:,} total{permin}, p50 "
            f"{rtab.get('p50', 0):.0f} deltas per request "
            f"({s.get('reasoning_stream_deltas_source') or _reason_source}). "
            "these are SSE "
            "chunks, not tokens")

    tp = s.get("throughput") or {}
    if tp.get("input_tokens_per_min"):
        usage_coverage = tp.get("usage_coverage")
        coverage_text = (
            f"; clean usage coverage {usage_coverage:.1%}"
            if isinstance(usage_coverage, (int, float))
            and not isinstance(usage_coverage, bool) else "")
        lines += ["", f"throughput: {tp['input_tokens_per_min']:,.0f} input "
                      f"tokens/min, {tp['output_tokens_per_min']:,.0f} output "
                      "tokens/min (endpoint-reported counts over wall time"
                      f"{coverage_text})"]
    windows = s.get("observed_rate_windows") or {}
    win_input = windows.get("input_tokens_by_first_send") or {}
    win_reserved = (
        windows.get(
            "offered_output_token_reservation_demand_by_first_send") or {})
    win_actual = windows.get("actual_output_tokens_by_completion") or {}
    win_queries = windows.get("physical_queries_by_first_send") or {}
    if any(window.get("max") is not None
           for window in (win_input, win_reserved, win_actual, win_queries)):
        def rolling_value(window: dict) -> str:
            value = window.get("max")
            return "NOT REPORTED" if value is None else f"{value:,.0f}"

        traffic_scope = windows.get("traffic_scope") or {}
        phase_text = _traffic_phase_summary(traffic_scope)
        lines += ["", "rolling rate-window evidence:",
                  f"- captured traffic phases: {inline(phase_text)}",
                  f"- input tokens: {rolling_value(win_input)} maximum in a "
                  f"trailing 60-second request cohort "
                  + (f"(usage coverage {win_input['coverage']:.1%})"
                     if win_input.get("coverage") is not None else
                     "(usage coverage NOT REPORTED)"),
                  f"- offered output reservation demand: "
                  f"{rolling_value(win_reserved)} maximum requested "
                  "max_tokens in a trailing 60-second send cohort. this is "
                  "pre-admission demand, not observed provider consumption",
                  f"- actual output tokens: {rolling_value(win_actual)} "
                  "maximum when request totals are attributed to completion; "
                  "per-token generation timing was not available",
                  f"- offered physical POST demand: "
                  f"{rolling_value(win_queries)} maximum in a trailing "
                  "3,600-second cohort; this is not the provider's confirmed "
                  "processed-query counter"]
    limit_block = s.get("rate_limits") or {}
    if limit_block:
        configured = limit_block.get("configured") or {}
        binding = limit_block.get("binding") or {}
        if not binding.get("binding_complete"):
            binding_label = "NOT VERIFIED"
        elif binding.get("workspace_tier_verified"):
            binding_label = (
                "endpoint/model/deployment metadata and workspace tier "
                "verified")
        else:
            binding_label = (
                "endpoint/model/deployment metadata bound; workspace tier "
                "remains operator-asserted")
        lines += [
            "- configured rate-limit snapshot: "
            f"provider {inline(configured.get('provider'))}, model "
            f"{inline(configured.get('model'))}, deployment "
            f"{inline(configured.get('deployment_mode'))}, tier "
            f"{inline(configured.get('workspace_tier'))}; source "
            f"{inline(configured.get('source'))} as of "
            f"{inline(configured.get('as_of'))}; operator reverified "
            f"{inline(configured.get('verified_at') or 'NOT RECORDED')} with "
            f"max age {inline(configured.get('max_age_days') or 'NOT RECORDED')} "
            "days",
            f"- configured scope: {inline(configured.get('scope'))}",
            f"- endpoint binding: {binding_label}",
        ]
        for name, comparison in (limit_block.get("comparisons") or {}).items():
            observed_ratio = comparison.get(
                "observed_ratio_to_nominal_limit")
            observed_rendered = (
                "n/a" if observed_ratio is None else f"{observed_ratio:.1%}")
            ratio = comparison.get("ratio_to_nominal_limit")
            decision_rendered = "n/a" if ratio is None else f"{ratio:.1%}"
            projected = comparison.get("steady_state_projection")
            configured_limit = comparison.get("configured_limit")
            projected_ratio = (
                float(projected) / float(configured_limit)
                if isinstance(projected, (int, float))
                and not isinstance(projected, bool)
                and isinstance(configured_limit, (int, float))
                and not isinstance(configured_limit, bool)
                and configured_limit else None)
            lines.append(
                f"  - {inline(name.replace('_', ' '))}: observed "
                f"{comparison.get('observed_max')} / configured "
                f"{configured_limit} ({observed_rendered})"
                + (f", sustained projection {projected:.1f} "
                   f"({projected_ratio:.1%})"
                   if projected is not None else "")
                + f", conservative gate ratio {decision_rendered} "
                f"({inline(str(comparison.get('status')).replace('_', ' '))})")
        if limit_block.get("warning"):
            lines.append(
                f"- rate-limit warning: {inline(limit_block['warning'])}")
        lines.append(
            f"- scope warning: {inline(limit_block['external_usage_warning'])}")
    cost = s.get("cost")
    if cost and cost.get("error"):
        lines += ["", f"cost: config error, {inline(cost['error'])}"]
    elif cost and cost["mode"] == "per_token" and cost.get("coverage_warning"):
        lines += ["", "unverified user-supplied rate arithmetic: aggregate "
                  "replay total unavailable. "
                  + inline(cost["coverage_warning"]),
                  "pricing applicability warning: "
                  + inline(cost.get("applicability_warning") or "unverified")]
    elif cost and cost["mode"] == "per_token":
        dr = cost.get("dbu_per_request") or {}
        if dr.get("p50") is None:
            lines += ["", "cost: no successful requests to price"]
        else:
            usd = cost.get("usd_total")
            dollar = f" (${usd:,.4f} total)" if usd is not None else ""
            lines += ["", f"unverified user-supplied rate arithmetic "
                      f"(measured replay only): "
                      f"{dr['p50']:.4f} DBU/request p50, "
                      f"{cost['dbu_per_1k_requests']:,.2f} DBU/1k requests, "
                      f"{cost['dbu_per_min']:,.3f} DBU/min, cache saved "
                      f"{cost['cache_dbu_saved']:,.3f} DBU{dollar}",
                      "pricing applicability warning: "
                      + inline(cost.get("applicability_warning") or "unverified")]
    elif cost and cost["mode"] == "provisioned" \
            and cost.get("coverage_warning"):
        lines += ["", "unverified provisioned-rate arithmetic: effective "
                  "cost per 1M tokens unavailable. "
                  + inline(cost["coverage_warning"]),
                  "configured capacity rate: "
                  f"{cost['dbu_per_hour']} DBU/hour",
                  "pricing applicability warning: "
                  + inline(cost.get("applicability_warning") or "unverified")]
    elif cost:
        eff = cost.get("effective_dbu_per_1m_tokens")
        lines += ["", f"unverified provisioned-rate arithmetic "
                  f"({cost['dbu_per_hour']} DBU/hour): "
                  + (f"effective {eff:,.1f} DBU per 1M tokens at the measured "
                     f"throughput" if eff is not None
                     else "throughput too low to compute an effective rate"),
                  "pricing applicability warning: "
                  + inline(cost.get("applicability_warning") or "unverified")]
    rp = (s.get("run") or {}).get("request_params")
    if rp:
        eb = rp.get("extra_body") or {}
        line = (f"request params: temperature {rp.get('temperature')}, "
                "global max_tokens safety cap "
                f"{rp.get('max_output_tokens_cap')}")
        if eb:
            line += f", extra_body {json.dumps(eb)}"
        lines += ["", inline(line)]
    merge_note = (s.get("run") or {}).get("merge_note")
    if merge_note:
        lines += ["", inline(merge_note)]

    # report.md is the file that gets pasted into an email, so it shows the
    # same verdict the html does, from the same function, whether or not
    # acceptance targets were given.
    _kind, _text = _verdict(s)
    if _kind != "ok" or s.get("sla"):
        _pre = "INVALID: " if _kind == "invalid" else ""
        lines += ["", f"verdict: {_pre}{_text}"]

    a = s.get("answers")
    if a:
        answer_lines = ["", "## answers",
                  "", f"- attempted: {a['attempted']}",
                  f"- harness-successful: "
                  f"{a.get('harness_successful', a['transport_ok'])}",
                  f"- produced at least one visible or reasoning content "
                  f"delta: {a.get('content_delta_streams', 'NOT RECORDED')}"]
        if a.get("unclassified_legacy_successes"):
            answer_lines.append(
                f"- legacy successes without content/tool observability: "
                f"{a['unclassified_legacy_successes']}")
        if a.get("http_status_observed_for"):
            answer_lines.append(
                f"- returned HTTP 200: {a['http_200']} (status recorded for "
                f"{a['http_status_observed_for']} requests)")
        answer_lines += [f"- produced a readable answer or valid tool call: "
                         f"{a['answered']} "
                         f"({a['answer_rate']:.1%} of the "
                         f"{a.get('judged')} judged)"
                         if a.get("answer_rate") is not None else
                         "- produced a readable answer or valid tool call: "
                         f"{a['answered']}",
                  f"- valid tool-call outcomes: "
                  f"{a.get('valid_tool_call_outcomes', 0)} "
                  f"({a.get('tool_call_only_outcomes', 0)} tool-call-only; "
                  f"{a.get('valid_tool_calls_total', 0)} calls total)",
                  f"- judged requests with neither visible content nor a "
                  f"valid tool call: {a.get('no_acceptable_outcome', a['no_visible_content'])}",
                  f"- judged requests with no visible content: "
                  f"{a['no_visible_content']}",
                  f"- stream never terminated: {a['stream_incomplete']}",
                  f"- unrecoverable parse errors: {a['parse_errors']}",
                  f"- stopped at the requested output length: "
                  f"{a['truncated']}",
                  f"- cut short by the global token cap: "
                  f"{a['truncated_by_global_cap']}",
                  "", inline(a["note"])]
        lines += answer_lines
        if a.get("invalid"):
            lines += ["", f"INVALID: {inline(a['invalid'])}"]

    sla = s.get("sla")
    if sla:
        _tgt_src = sla.get("targets_source") or "the run configuration"
        _basis = (sla.get("latency_basis") or "unknown").replace("_", " ")
        lines += ["", f"## Acceptance scorecard (targets from {inline(_tgt_src)}; "
                  f"latency basis: {inline(_basis)})"]
        if sla.get("targets_warning"):
            lines += ["", f"CAUTION (targets): "
                      f"{inline(sla['targets_warning'])}"]
        if sla.get("coverage_warning"):
            lines += ["", f"CAUTION (coverage): "
                      f"{inline(sla['coverage_warning'])}"]
        if sla.get("caller_latency_warning"):
            lines += ["", f"CAUTION (caller timing): "
                      f"{inline(sla['caller_latency_warning'])}"]
        lines += ["", "| metric | quantile | target ms | actual ms | met |",
                  "|---|---|---|---|---|"]
        for name, key in (("TTFT", "ttft_vs_target"),
                          ("TTFG", "ttfg_vs_target")):
            for r in sla.get(key) or []:
                met = {True: "yes", False: "NO", None: "-"}[r["met"]]
                act = r["actual_ms"] if r["actual_ms"] is not None \
                    else "not measured"
                lines.append(f"| {name} | {r['quantile']} | {r['target_ms']} "
                             f"| {act} | {met} |")
        hard_basis = sla.get("hard_timeout_basis") or {}
        hard_timeout_configured = any(
            hard_basis.get(key) is not None
            for key in ("ttft_cap_ms", "ttfg_cap_ms"))
        if hard_timeout_configured:
            hard_breaches = sla.get("hard_timeout_breaches", 0)
            lines.append(f"| hard timeout breaches | - | - | "
                         f"{hard_breaches} | "
                         f"{'yes' if not hard_breaches else 'NO'} |")
        if "interchunk_breaches" in sla:
            ib = sla["interchunk_breaches"]
            lines.append(f"| interchunk breaches | - | - | {ib} | "
                         f"{'yes' if not ib else 'NO'} |")
        sr = sla.get("success_rate")
        if sr:
            lines.append(f"| success rate | - | {sr['target']} | "
                         f"{sr['actual']} | {'yes' if sr['met'] else 'NO'} |")
            demonstrated = sr.get("statistically_demonstrated")
            if demonstrated is not None:
                lines += ["", "success-rate evidence: "
                          f"{sr['successes']} successes in {sr['attempts']} "
                          "attempts; one-sided 95% Wilson lower bound "
                          f"{sr['one_sided_95pct_wilson_lower']:.6f}. "
                          + ("the confidence bound meets the target."
                             if demonstrated else
                             "the observed fraction meets the target, but "
                             "the confidence bound does not; this cannot be "
                             "a clean green-light result.")]


    if s.get("ttfr_ms"):
        tft = s["ttft_ms"].get("p50")
        _v = s.get("ttfv_ms") or {}
        tfv = _v.get("p50")
        _miss, _of = _v.get("missing") or 0, _v.get("of") or 0
        if tfv is None:
            vis = "no request emitted visible content within max_tokens"
        elif _miss:
            vis = (f"ttfv (first visible content) p50 {tfv:.0f} ms, but over "
                   f"only the {_of - _miss} of {_of} requests that produced "
                   "visible content. the rest ran out of output tokens still "
                   "reasoning, so that p50 is the fastest subset, not the run")
        else:
            vis = f"ttfv (first visible content) p50 {tfv:.0f} ms"
        lines += ["", "note: reasoning model detected. ttft (first visible-"
                  "or-reasoning content delta) "
                  f"p50 {tft:.0f} ms. {vis}. agree which "
                  "definition the configured acceptance target scores via "
                  "ttft_definition in the run "
                  "config."]

    drift = s.get("drift") or {}
    if drift.get("windows") or drift.get("drift_kind"):
        kind = drift.get("drift_kind")
        if not kind:
            flag = "NOT ENOUGH DATA"
        elif kind == "stable":
            flag = "stable"
        else:
            flag = f"UNSTABLE ({kind})"
        spread = drift.get("ttft_p95_spread_ratio")
        sp = (f" worst window is {spread:.1f}x the best."
              if spread else "")
        lines += ["", f"stability over time ({inline(flag)})."
                  f"{sp} {inline(drift.get('drift_headline') or drift.get('note', ''))}"]
        if drift.get("windows"):
            lines += ["", f"per-{drift.get('window_seconds', 60)}s windows, p95 in ms:",
                      "",
                      "| window | content-bearing streams | errors | TTFT p95 | E2E p95 |",
                      "|---|---|---|---|---|"]
        for w in (drift.get("windows") or []):
            tt = f"{w['ttft_p95']:.0f}" if w['ttft_p95'] is not None else "-"
            ee = f"{w['e2e_p95']:.0f}" if w['e2e_p95'] is not None else "-"
            mark = "" if w.get("counted", True) else " (not counted)"
            er = _err_cell(w)
            lines.append(
                f"| {w['window']}{mark} | {w['n']} | {er} | {tt} | {ee} |")
        # only when a verdict exists, otherwise the headline already IS the note
        if drift.get("drift_headline"):
            lines.append("")
            lines.append(f"note: {inline(drift.get('note', ''))}")
    elif drift.get("note"):
        lines += ["", f"stability over time: {inline(drift['note'])}"]

    em = (s.get("run") or {}).get("endpoint_metadata")
    if em:
        se = em.get("served_entities") or []
        detail = (", ".join(f"{inline(k)}={inline(v)}"
                            for k, v in se[0].items() if k != "name")
                  if se else "")
        _task = f"task {inline(em.get('task'))}, " if em.get("task") else ""
        lines += ["", f"endpoint under test: {inline(em.get('name'))}, "
                  f"{_task}route_optimized "
                  f"{inline(em.get('route_optimized'))}, "
                  f"ready {inline(em.get('ready'))}"
                  + (f", {detail}" if detail else "")]

    run_meta = s.get("run") or {}
    if run_meta.get("label"):
        lines += ["", f"**Label:** {inline(run_meta['label'])}"]
    if run_meta.get("profile_label"):
        lines += ["", f"**Profile:** {inline(run_meta['profile_label'])}"]
    if verified_view:
        source_repro = verified_view["source_reproducibility"]
        verifier_repro = verified_view["verifier_reproducibility"]
        lines += [
            "",
            "---",
            f"{verified_view['view_label']} derivative · source artifact "
            f"`{inline(verified_view['source_artifact_id'])}` · full manifest "
            f"SHA-256 `{verified_view['source_manifest_sha256']}` · "
            f"source reproducibility {source_repro['code']} · verifier "
            f"reproducibility {verifier_repro['code']} · "
            f"{inline(verified_view['assurance'])}",
        ]
    return "\n".join(lines) + "\n"


def _manifest(summary: dict, out: Path, *,
              start_provenance: dict | None = None,
              artifact_metadata: dict | None = None,
              artifact_id: str | None = None,
              ended_at_unix: float | None = None) -> dict:
    """Everything needed to trace a number back to what produced it.

    A latency figure with no record of which code, which traffic shape and
    which endpoint made it is an anecdote. This is deliberately mechanical:
    no judgment, no interpretation, just the state that would otherwise be
    reconstructed from memory months later.

    The endpoint identity is retained because the result is meaningless
    without it. Arbitrary request parameters are recursively redacted before
    this object is returned; provenance must not turn ``extra_body`` into a
    credential side channel.
    """
    import platform
    from datetime import datetime, timezone

    run = _redact_secrets(summary.get("run") or {})
    start = _redact_secrets(start_provenance or {})
    source = start.get("source") or snapshot_source_state(Path(__file__).parent)
    inputs = start.get("inputs") or {}
    prof_path = run.get("profile_path") or run.get("prompts_file")
    primary_key = ("profile" if run.get("input_mode") == "profile"
                   else "prompts" if run.get("input_mode") == "prompts"
                   else None)
    primary_input = inputs.get(primary_key) if primary_key else None
    prof_sha = ((primary_input or {}).get("sha256")
                if isinstance(primary_input, dict) else None)
    # Backward-compatible standalone write_outputs callers do not have a
    # start-of-run snapshot. They still receive a digest, but real runner runs
    # always carry the immutable pre-traffic value above.
    if prof_sha is None and prof_path and Path(prof_path).is_file():
        prof_sha = sha256_bytes(Path(prof_path).read_bytes())

    logical_run_id = (run.get("logical_run_id") or run.get("run_id")
                      or start.get("logical_run_id") or out.name)
    execution_id = (run.get("execution_id") or start.get("execution_id")
                    or artifact_id or out.name)
    artifact_id = (run.get("artifact_id") or start.get("artifact_id")
                   or artifact_id or out.name)
    workload_id = run.get("workload_id") or start.get("workload_id")
    effective_config = _redact_secrets(start.get("effective_config") or {})
    schedule_identity = (start.get("schedule_identity")
                         or run.get("schedule_identity"))
    index_identity = start.get("index_identity") or run.get("index_identity") or {}

    # Preserve a canonical, redacted identity snapshot in addition to its
    # digest. A digest alone can prove equality but cannot explain a mismatch.
    config_identity = _redact_secrets({
        "harness_version": summary.get("harness_version"),
        "latency_basis": summary.get("latency_basis"),
        "effective_config": effective_config,
        "workload_id": workload_id,
        "schedule_identity": schedule_identity,
        "index_identity": index_identity,
        "request_params": run.get("request_params"),
        "schedule": summary.get("schedule") or {},
        "sla_definition": {
            "ttft_definition": (run.get("ttft_definition")
                                or (summary.get("sla") or {}).get(
                                    "ttft_definition")),
            "targets_source": (summary.get("sla") or {}).get("targets_source"),
            "acceptance_config": (summary.get("sla") or {}).get(
                "acceptance_config"),
        },
        "pricing": {
            key: (summary.get("cost") or {}).get(key)
            for key in ("mode", "rates_dbu_per_m", "dbu_per_hour",
                        "usd_per_dbu")
            if (summary.get("cost") or {}).get(key) is not None
        },
    })
    config_sha = canonical_sha256(config_identity)
    effective_config_sha = (canonical_sha256(effective_config)
                            if effective_config else None)
    ended_at_unix = ended_at_unix if ended_at_unix is not None else time.time()
    ended = datetime.fromtimestamp(ended_at_unix, timezone.utc).isoformat()
    started_at_unix = start.get("run_started_at_unix")
    started = start.get("run_started_at_utc")
    if started is None and started_at_unix is not None:
        started = datetime.fromtimestamp(
            float(started_at_unix), timezone.utc).isoformat()
    manifest = {
        "manifest_schema_version": 3,
        "artifact_created_at_utc": ended,
        "run_started_at_utc": started,
        "run_started_at_unix": started_at_unix,
        "run_ended_at_utc": ended,
        "run_ended_at_unix": ended_at_unix,
        "run_id": logical_run_id,       # legacy alias
        "logical_run_id": logical_run_id,
        "workload_id": workload_id,
        "execution_id": execution_id,
        "artifact_id": artifact_id,
        "harness_version": summary.get("harness_version"),
        "git_commit": source.get("git_commit"),
        "git_dirty": source.get("git_dirty"),
        "source": source,
        "source_tree_sha256": source.get("source_tree_sha256"),
        "latency_basis": summary.get("latency_basis"),
        "profile": run.get("profile"),
        "profile_path": prof_path,
        "profile_sha256": prof_sha,
        "profile_sha256_16": prof_sha[:16] if prof_sha else None,
        "profile_provenance": run.get("profile_provenance"),
        "input_mode": run.get("input_mode"),
        "seed": run.get("seed"),
        "endpoint_path": run.get("endpoint_path"),
        "endpoint_base_url": run.get("endpoint_base_url"),
        "endpoint_model": run.get("endpoint_model"),
        "endpoint_metadata": run.get("endpoint_metadata"),
        "network_path": run.get("network_path"),
        "request_params": run.get("request_params"),
        "load_mode": run.get("load_mode"),
        "sizing_concurrency_requested": run.get(
            "sizing_concurrency_requested", run.get("concurrency_target")),
        "derived_qps": run.get("derived_qps"),
        "concurrency_target": run.get("concurrency_target"),
        "start_at_unix": run.get("start_at_unix"),
        "global_index_start": index_identity.get(
            "min", run.get("global_index_start")),
        "global_index_end": index_identity.get(
            "max", run.get("global_index_end")),
        "global_index_range": run.get("global_index_range"),
        "index_identity": index_identity or None,
        "schedule_identity": schedule_identity,
        "shard": run.get("shard"),
        "schedule": summary.get("schedule"),
        "config_sha256": config_sha,
        "config_identity": config_identity,
        "effective_config_sha256": effective_config_sha,
        "effective_config": effective_config,
        "inputs": inputs,
        "artifacts": artifact_metadata or {},
        "aggregation": run.get("aggregation"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "note": ("written by the harness, not by hand. a number quoted "
                 "without this cannot be reproduced or audited."),
    }
    return _redact_secrets(manifest)


def write_outputs(results, summary: dict, out_dir: str | Path,
                  title: str, *, artifact_run: RunArtifacts | None = None,
                  start_provenance: dict | None = None) -> Path:
    """Write a run without overwriting a same-second sibling.

    The runner historically named directories to one-second precision and
    used ``exist_ok=True``. Two launches in the same second then replaced one
    another's evidence file by file. Claim the directory with an exclusive
    marker, add a random suffix on collision, and replace each artifact from
    a same-directory temporary file so readers never observe a torn JSON or
    report file.
    """
    owned = artifact_run is None
    safe_title = sanitize_title(title)
    if artifact_run is None:
        now = time.time()
        from datetime import datetime, timezone
        artifact_run = RunArtifacts.claim(out_dir, start_provenance or {
            "run_started_at_unix": now,
            "run_started_at_utc": datetime.fromtimestamp(
                now, timezone.utc).isoformat(),
            "source": snapshot_source_state(Path(__file__).parent),
            "effective_config": {"title": safe_title},
        })
        try:
            for row in results or []:
                artifact_run.append(row)
        except BaseException as exc:
            artifact_run.abort(exc)
            raise
    out = artifact_run.path
    safe_summary = _redact_secrets(summary)
    # Persist the same five independent states that HTML and Markdown render.
    # A summary cannot authenticate the manifest that contains it, so this
    # embedded decision intentionally remains VERIFY_REQUIRED until an
    # external verifier supplies an explicit integrity context.
    from .report_decision import build_report_decision

    safe_summary["decision"] = build_report_decision(safe_summary)
    summary["decision"] = safe_summary["decision"]
    try:
        artifact_run.finalize_requests()
        artifact_run.atomic_text(
            "summary.json", strict_json_dumps(safe_summary, indent=2) + "\n")
        artifact_run.atomic_text(
            "report.md", render_markdown(safe_summary, safe_title))
        artifact_run.atomic_text(
            "report.html", render_html(safe_summary, safe_title))
        names = [FINAL_REQUESTS, "summary.json", "report.md", "report.html",
                 "start.json"]
        metadata = artifact_run.metadata(names)
        ended_at = time.time()
        manifest = _manifest(
            safe_summary, out,
            start_provenance=(start_provenance
                              or artifact_run.start_provenance),
            artifact_metadata=metadata,
            artifact_id=artifact_run.artifact_id,
            ended_at_unix=ended_at)
        # Manifest is deliberately last. Completion is a separate marker so a
        # crash between these two operations remains visibly incomplete.
        artifact_run.atomic_text(
            "manifest.json", strict_json_dumps(manifest, indent=2) + "\n")
        artifact_run.mark_complete()
        return out
    except BaseException as exc:
        artifact_run.abort(exc)
        raise
    finally:
        if owned and not artifact_run.complete:  # defensive close on errors
            artifact_run.close()


_HTML_STYLE = """<style>
:root{color-scheme:light;--canvas:#f5f7fb;--surface:#fff;--surface-2:#f8fafc;
 --ink:#172033;--muted:#556176;--quiet:#667085;--line:#d9e0e9;
 --blue:#075fce;--blue-soft:#eaf2ff;--green:#166534;--green-soft:#e9f7ef;
 --red:#b42318;--red-soft:#fff0ee;--amber:#8a4b08;--amber-soft:#fff6e8;
 --gray:#344054;--shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px rgba(16,24,40,.04)}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:76px}
body{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,
 sans-serif;color:var(--ink);background:var(--canvas);margin:0;line-height:1.48;
 -webkit-font-smoothing:antialiased}
a{color:var(--blue);text-underline-offset:3px}
a:focus-visible,summary:focus-visible{outline:3px solid #155eef;outline-offset:3px;
 border-radius:4px}
.wrap{max-width:1180px;margin:0 auto;padding:32px 28px 56px}
.external-verified{border:2px solid #6c87a8;border-radius:14px;padding:14px 16px;
 margin:0 0 12px;background:#f4f7fb;color:var(--ink);box-shadow:var(--shadow)}
.external-verified.repro-warning{border-color:#d19042;background:var(--amber-soft)}
.external-verified .verified-badge{display:inline-flex;border-radius:999px;padding:4px 9px;
 background:var(--gray);color:#fff;font-size:10px;font-weight:900;letter-spacing:.1em;
 text-transform:uppercase}.external-verified .verified-grid{display:grid;
 grid-template-columns:minmax(180px,.55fr) minmax(0,1.45fr);gap:5px 16px;margin-top:10px;
 font-size:12px}.external-verified dt{font-weight:800}.external-verified dd{margin:0;
 overflow-wrap:anywhere}.external-verified code{font-family:ui-monospace,SFMono-Regular,Menlo,
 Consolas,monospace;font-size:11px}.external-verified .assurance{grid-column:1/-1;margin:5px 0 0;
 color:var(--muted);font-size:11px}.verification-states{display:grid;
 grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:10px}
.verification-state{border:1px solid var(--line);border-radius:9px;background:#fff;
 padding:7px 9px;display:flex;justify-content:space-between;align-items:center;gap:8px;
 font-size:11px}.verification-state span{font-weight:750;color:var(--muted)}
.verification-state strong{font-size:10px;border-radius:999px;padding:2px 7px;
 letter-spacing:.04em}.verification-state .status-pass{color:var(--green);
 background:var(--green-soft)}.verification-state .status-failed{color:var(--red);
 background:var(--red-soft)}.repro-codes{color:var(--muted);font-size:10px}
.report-head{background:#0c1729;color:#fff;border-radius:18px;padding:28px 30px 24px;
 box-shadow:0 18px 44px rgba(12,23,41,.16)}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
 color:#b9d3ff;margin-bottom:8px}
h1{font-size:clamp(25px,3vw,38px);line-height:1.14;letter-spacing:-.025em;
 margin:0 0 10px;max-width:900px;overflow-wrap:anywhere}
.sub{color:#d3dceb;font-size:14px;margin:0;overflow-wrap:anywhere}
.meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.meta-chip{display:inline-flex;align-items:center;min-height:28px;padding:5px 10px;
 border:1px solid #31415a;border-radius:999px;color:#e5edf8;background:#15233a;
 font-size:12px;font-variant-numeric:tabular-nums;max-width:100%;min-width:0;
 white-space:normal;overflow-wrap:anywhere}
.report-nav{position:sticky;top:0;z-index:10;display:flex;gap:4px;overflow-x:auto;
 margin:14px 0 18px;padding:7px;background:rgba(255,255,255,.96);
 border:1px solid var(--line);border-radius:12px;box-shadow:0 4px 16px rgba(16,24,40,.06);
 scrollbar-width:thin;backdrop-filter:blur(10px)}
.report-nav a{flex:0 0 auto;padding:7px 10px;border-radius:7px;color:#344054;
 font-size:12px;font-weight:700;text-decoration:none}
.report-nav a:hover{background:var(--blue-soft);color:#064da8}
.decision-hero{border:1px solid var(--line);border-top:5px solid var(--gray);
 border-radius:16px;background:var(--surface);padding:22px 24px;margin:16px 0;
 box-shadow:var(--shadow)}
.decision-hero.state-ok{border-top-color:var(--green)}
.decision-hero.state-bad{border-top-color:var(--red)}
.decision-hero.state-warn{border-top-color:var(--amber)}
.decision-lead{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(260px,.8fr);
 gap:24px;align-items:start}
.status-kicker{font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;
 color:var(--quiet)}
.decision-hero h2{font-size:clamp(21px,2.4vw,30px);line-height:1.22;margin:7px 0 8px;
 letter-spacing:-.015em;text-transform:none;color:var(--ink)}
.decision-copy{color:var(--muted);font-size:14px;margin:0}
.claim-box{border-left:3px solid var(--line);padding-left:16px;font-size:13px}
.claim-box p{margin:0 0 9px}.claim-box p:last-child{margin-bottom:0}
.claim-box b{display:block;color:var(--ink);font-size:11px;letter-spacing:.06em;
 text-transform:uppercase;margin-bottom:2px}
.state-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;
 margin-top:20px}
.state-card{min-width:0;border:1px solid var(--line);border-radius:11px;padding:12px;
 background:var(--surface-2)}
.state-card .k{font-size:10px;color:var(--quiet);font-weight:800;letter-spacing:.07em;
 text-transform:uppercase}
.state-card .v{font-size:13px;font-weight:800;margin-top:5px;line-height:1.25;
 overflow-wrap:anywhere}
.state-card .why{font-size:11px;color:var(--muted);margin-top:5px;line-height:1.35}
.state-card .why{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
 overflow:hidden}
.gate-detail{margin-top:13px;border-top:1px solid var(--line);padding-top:10px}
.gate-detail summary{cursor:pointer;color:var(--muted);font-size:11px;font-weight:800;
 text-transform:uppercase;letter-spacing:.045em}.gate-detail .banner{margin-bottom:0}
.decision-reasons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px;
 margin:10px 0 0}.decision-reasons div{min-width:0}.decision-reasons dt{font-size:10px;
 color:var(--quiet);font-weight:800;text-transform:uppercase;letter-spacing:.04em}
.decision-reasons dd{margin:3px 0 0;color:var(--muted);font-size:12px}
.tone-ok .v{color:var(--green)}.tone-bad .v{color:var(--red)}
.tone-warn .v{color:var(--amber)}.tone-neutral .v{color:var(--gray)}
.section-head{display:flex;justify-content:space-between;align-items:end;gap:18px;
 margin:30px 2px 10px}
.section-head h2{font-size:18px;line-height:1.25;margin:0;letter-spacing:-.01em}
.section-head p{font-size:12px;color:var(--muted);margin:0;max-width:650px;text-align:right}
.fact-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:14px 0}
.fact{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:12px 13px;
 min-width:0}
.fact .k{font-size:10px;color:var(--quiet);font-weight:800;letter-spacing:.06em;
 text-transform:uppercase}.fact .v{font-size:18px;font-weight:800;margin-top:3px;
 font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.fact .u{font-size:11px;
 color:var(--muted);font-weight:500}.fact .note{font-size:10px;color:var(--muted);margin-top:4px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
 padding:18px 20px;margin:12px 0;box-shadow:var(--shadow);break-inside:avoid}
.card h2{font-size:12px;margin:0 0 5px;color:#0757b5;text-transform:uppercase;
 letter-spacing:.055em}
.cap{font-size:12px;color:var(--muted);margin:0 0 12px;max-width:880px}
.slanote{background:#eef6ff;border:1px solid #c8dcfa;border-radius:9px;
 padding:11px 14px;font-size:12px;color:#164d7d;margin-top:12px;line-height:1.5}
.slanote code{background:#daeafd;padding:1px 4px;border-radius:3px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.stat{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:14px 15px;box-shadow:0 1px 2px rgba(16,24,40,.03)}
.stat .k{font-size:10px;color:var(--quiet);font-weight:800;text-transform:uppercase;
 letter-spacing:.055em;line-height:1.35}
.stat .v{font-size:24px;font-weight:800;margin-top:6px;font-variant-numeric:tabular-nums;
 letter-spacing:-.02em;overflow-wrap:anywhere}
.stat .u{font-size:11px;color:var(--muted);font-weight:500;letter-spacing:0}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.table-scroll{max-width:100%;overflow-x:auto;overscroll-behavior-inline:contain;
 scrollbar-gutter:stable}.table-scroll:focus-visible{outline:3px solid var(--blue);
 outline-offset:3px}.scroll-hint{display:none}
caption{color:var(--muted);font-size:12px;text-align:left;padding:0 0 8px}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid #e9edf3;font-size:13px}
thead th{color:var(--quiet);font-weight:800;font-size:10px;text-transform:uppercase;
 letter-spacing:.045em;background:#fbfcfe}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
td.lbl,th.lbl{text-align:left;font-weight:650;color:#27364b}
td.n{color:var(--muted)}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;
 font-weight:800;line-height:1.35;white-space:nowrap}
.ok{background:var(--green-soft);color:var(--green)}
.bad{background:var(--red-soft);color:var(--red)}
.warn{background:var(--amber-soft);color:var(--amber)}
.neutral{background:#eef1f5;color:var(--gray)}
.banner{border-radius:10px;padding:12px 14px;margin:10px 0;font-weight:650;font-size:13px}
.banner.ok{background:var(--green-soft);color:var(--green);border:1px solid #a9dbbc}
.banner.bad{background:var(--red-soft);color:var(--red);border:1px solid #f1b5ae}
.banner.warn{background:var(--amber-soft);color:var(--amber);border:1px solid #ebca98}
.issue-card{border-left:5px solid var(--amber)}
.issue-card ul{margin:10px 0 0;padding-left:20px}.issue-card li{margin:8px 0;
 color:#364152;font-size:13px}.issue-card b{color:var(--ink)}
.believe{border-left:5px solid var(--amber)}
.believe ul{margin:0;padding-left:20px}
.believe li{margin:8px 0;font-size:13px;color:#364152}
.believe b{color:var(--ink)}
.label-note{background:#fff9e8;border:1px solid #e7c86f;border-radius:10px;
 padding:12px 15px;font-size:13px;color:#6b4e08;margin:12px 0}
.run-context-notes{margin:10px 0 14px}
.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.chart{border:1px solid var(--line);border-radius:11px;padding:14px;background:#fbfcfe}
.chart h3{font-size:13px;margin:0}.chart .chart-meta{font-size:11px;color:var(--muted);
 margin:2px 0 10px}.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart-axis{stroke:#7b8798;stroke-width:1}.chart-line{fill:none;stroke:var(--blue);
 stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.chart-area{fill:#dfeeff;opacity:.7}
.chart-dot{fill:var(--surface);stroke:var(--blue);stroke-width:2}.chart-label{fill:#475467;
 font-size:9px;font-family:inherit}.chart-bad{stroke:var(--red)}
.chart-secondary{stroke:#6b55c5}.chart-dot-secondary{stroke:#6b55c5}
.quota-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.gauge{border:1px solid var(--line);border-radius:10px;padding:13px;background:#fbfcfe}
.gauge-head{display:flex;justify-content:space-between;gap:8px;font-size:12px;font-weight:700}
.gauge-track{height:8px;background:#e5eaf0;border-radius:999px;overflow:hidden;margin:9px 0 6px}
.gauge-fill{height:100%;background:var(--blue);border-radius:999px}.gauge-fill.warn{background:#c66a08}
.gauge-fill.bad{background:var(--red)}.gauge-note{font-size:11px;color:var(--muted)}
details.evidence{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 margin:12px 0;break-inside:avoid}
details.evidence summary{cursor:pointer;padding:14px 16px;font-size:13px;font-weight:800;
 color:#27364b;list-style-position:inside}
details.evidence[open] summary{border-bottom:1px solid var(--line)}
details.evidence .detail-body{padding:4px 16px 16px}
.print-evidence{display:none}
.foot{color:var(--muted);font-size:11px;margin-top:24px;text-align:center}
.print-footer{display:none}
td.yes{color:var(--green);font-weight:800}
td.no{background:var(--red-soft);color:var(--red);font-weight:800}
td.na{color:var(--muted);font-weight:650}
.sr-only{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;
 margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;
 white-space:nowrap!important;border:0!important}
@media(max-width:900px){.state-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
 .fact-strip,.stats{grid-template-columns:repeat(3,minmax(0,1fr))}
 .decision-lead{grid-template-columns:1fr}.chart-grid{grid-template-columns:1fr}
 .quota-grid{grid-template-columns:1fr}.section-head{align-items:start;flex-direction:column;gap:4px}
 .section-head p{text-align:left}}
@media(max-width:640px){.wrap{padding:14px 12px 36px}.report-head{border-radius:14px;
 padding:16px}.external-verified .verified-grid{grid-template-columns:1fr;gap:2px}
 .verification-states{grid-template-columns:1fr;gap:4px}
 .external-verified .verified-grid dt{margin-top:5px}.external-verified .assurance{grid-column:auto}
 .report-head h1{font-size:22px;margin-bottom:7px}.report-head .sub{font-size:11px}
 .report-head .meta-artifact{display:inline-flex;max-width:100%;overflow-wrap:anywhere}
 .meta-row{margin-top:10px;gap:6px}.meta-chip{font-size:11px;min-height:26px;padding:4px 8px}
 .report-nav{margin:10px 0 14px;border-radius:9px}
 .report-nav a{padding:6px 9px;font-size:11px}.decision-hero{padding:14px 16px;
 margin-top:12px}.decision-hero h2{font-size:19px;margin:5px 0}.decision-hero .claim-box{display:none}
 .status-kicker{font-size:10px}.decision-copy{font-size:12px}
 .decision-copy{display:block;overflow:visible}.state-grid{grid-template-columns:1fr;gap:0;
 border:1px solid var(--line);border-radius:10px;overflow:hidden}
 .state-card{display:flex;align-items:center;justify-content:space-between;gap:10px;
 border:0;border-bottom:1px solid var(--line);border-radius:0;padding:6px 9px}
 .state-card:last-child{border-bottom:0}.state-card .v{margin:0;text-align:right}
 .state-card .k{font-size:9px}.state-card .v{font-size:11px}.state-card .why{display:none}
 .fact-strip{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}
 .section-head{margin:14px 2px 7px}.section-head h2{font-size:16px}.section-head p{display:none}
 .fact{padding:8px}.fact .k{font-size:8px}.fact .v{font-size:15px}.fact .u{font-size:9px}
 .fact .note{display:none}
 .decision-reasons{grid-template-columns:1fr}.decision-reasons dd{font-size:11px}
 .card{padding:15px 14px;border-radius:11px}.stat .v{font-size:21px}
 .scroll-hint{display:flex;align-items:center;gap:7px;margin:0 0 7px;padding:7px 9px;
 border-radius:8px;background:var(--blue-soft);color:#174ea6;font-size:12px;
 font-weight:750}.table-scroll{box-shadow:inset -12px 0 12px -14px var(--ink);
 -webkit-overflow-scrolling:touch}.dense-table{display:table;min-width:620px;
 white-space:nowrap}.dense-table .sticky-col{position:sticky;inset-inline-start:0;
 z-index:2;box-shadow:5px 0 7px -7px var(--ink);background:var(--surface)}
 .dense-table thead .sticky-col{z-index:4;background:#fbfcfe}
 table:not(.dense-table){display:table;width:100%;max-width:100%;table-layout:fixed;
 white-space:normal}table:not(.dense-table) th,table:not(.dense-table) td{
 overflow-wrap:anywhere;vertical-align:top}table:not(.dense-table) th.lbl{width:44%}
 th,td{padding:8px;font-size:12px}.believe li{font-size:12px}.sub{font-size:12px}}
@media print{@page{size:auto;margin:14mm 12mm 16mm}html{scroll-padding-top:0}
 body{background:#fff;font-size:10pt}.wrap{max-width:none;padding:0}.report-head{box-shadow:none;
 border:1px solid #9aa7b8;background:#fff;color:#111;padding:10px 12px}.external-verified{
 box-shadow:none;border:1.5px solid #6c87a8;background:#fff;color:#111;padding:7px 9px;
 break-inside:avoid;page-break-inside:avoid}.external-verified .verified-grid{font-size:8pt;
 margin-top:5px;gap:2px 8px}.external-verified .assurance{font-size:7.5pt}
 .external-verified .verified-badge{background:#fff;color:#111;
 border:1px solid #667085}
 .verification-states{gap:3px;margin-top:5px}.verification-state{padding:3px 5px;
 font-size:7.5pt}.verification-state strong{font-size:7pt}
 .external-verified.repro-warning{border-color:#d19042;background:#fff}
 .report-head h1{font-size:21px}
 .eyebrow,.sub{color:#344054}.meta-row{margin-top:8px}.meta-chip{background:#fff;color:#111;
 border-color:#aeb8c6;min-height:22px;padding:2px 7px;font-size:9px}.report-nav{display:none}
 .decision-hero,.card,.stat,.fact,.state-card,details.evidence{box-shadow:none;break-inside:avoid;
 page-break-inside:avoid}.decision-hero{margin-top:8px;padding:11px 13px}.decision-hero h2{font-size:18px;
 margin:4px 0}.decision-copy{font-size:10px}.decision-lead{gap:12px}.claim-box{font-size:9px;
 padding-left:10px}.claim-box b{font-size:8px}.state-grid{grid-template-columns:repeat(5,minmax(0,1fr));
 gap:4px;margin-top:9px}.state-card{padding:6px}.state-card .k{font-size:7px}.state-card .v{font-size:9px}
 .state-card .why,.gate-detail{display:none}.section-head{break-after:avoid;page-break-after:avoid;
 margin:12px 2px 6px}.section-head h2{font-size:15px}.section-head p{display:none}#workload{break-inside:avoid;
 page-break-inside:avoid}.fact-strip{grid-template-columns:repeat(6,minmax(0,1fr));gap:4px;margin:6px 0}
 .fact{padding:6px}.fact .k{font-size:7px}.fact .v{font-size:12px}.fact .u{font-size:8px}
 .fact .note{display:none}
 .stats{grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;margin:8px 0}
 .stat{padding:8px}.stat .k{font-size:8px}.stat .v{font-size:17px}
 table{break-inside:auto}th,td{padding:6px 7px}thead{display:table-header-group}
 tr{break-inside:avoid;page-break-inside:avoid}.label-note{margin:4px 0;padding:8px 10px}
 details.evidence{display:none}.print-evidence{display:block;break-inside:auto;
 page-break-inside:auto}.print-evidence li{break-inside:avoid;page-break-inside:avoid;
 margin:5px 0}
 .print-evidence h2{break-after:avoid;page-break-after:avoid}
 .chart svg{max-height:160px}.foot{display:none}.print-footer{display:block;
 border:1px solid #98a2b3;padding:2.5mm 3mm;margin:4mm 0 2mm;background:#fff;
 color:#344054;text-align:center;font-size:8pt;line-height:1.25;break-inside:avoid}
 .run-context-notes{break-inside:avoid;page-break-inside:avoid;margin:3mm 0}
 .scroll-hint{display:none}.table-scroll{overflow:visible;box-shadow:none}
 .dense-table{min-width:0}.dense-table .sticky-col{position:static;box-shadow:none}
 a{color:#111;text-decoration:none}}
</style>"""


def _html_stat(k, v, u=""):
    unit = f" <span class='u'>{html.escape(u)}</span>" if u else ""
    return (f"<div class='stat'><div class='k'>{html.escape(k)}</div>"
            f"<div class='v'>{v}{unit}</div></div>")


def _html_fact(label: str, value: str, unit: str = "", note: str = "") -> str:
    """One compact, escaped statement of what the run actually exercised."""
    unit_html = f" <span class='u'>{html.escape(unit)}</span>" if unit else ""
    note_html = f"<div class='note'>{html.escape(note)}</div>" if note else ""
    return (f"<div class='fact'><div class='k'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(value)}{unit_html}</div>"
            f"{note_html}</div>")


def _html_stability_chart(drift: dict) -> str:
    """Accessible inline p95 trend chart; the table remains the exact source.

    A missing window is a gap, never a zero.  This is intentionally SVG-only:
    completed artifacts remain self-contained and cannot fetch remote code.
    """
    windows = drift.get("windows") or []
    series = []
    for key, label, css in (
            ("ttft_p95", "TTFT p95", "chart-line"),
            ("e2e_p95", "E2E p95", "chart-line chart-secondary")):
        points = []
        for position, window in enumerate(windows):
            value = window.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(float(value)) and float(value) >= 0:
                points.append((position, float(value)))
        if points:
            series.append((label, css, points))
    if not series:
        return ""
    values = [value for _label, _css, points in series for _x, value in points]
    ceiling = max(values) or 1.0
    n_windows = max(len(windows), 2)
    left, top, width, height = 46.0, 12.0, 566.0, 136.0

    def xy(position: int, value: float) -> tuple[float, float]:
        x = left + (position / max(n_windows - 1, 1)) * width
        y = top + height - (value / ceiling) * height
        return x, y

    paths = []
    for label, css, points in series:
        # Split around missing windows so a gap is not joined by a line.
        segments: list[list[tuple[int, float]]] = []
        for point in points:
            if not segments or point[0] != segments[-1][-1][0] + 1:
                segments.append([point])
            else:
                segments[-1].append(point)
        for segment in segments:
            coords = " ".join(f"{x:.1f},{y:.1f}"
                              for x, y in (xy(*point) for point in segment))
            if len(segment) == 1:
                x, y = xy(*segment[0])
                dot_class = (
                    "chart-dot chart-bad" if "chart-bad" in css else
                    "chart-dot chart-dot-secondary"
                    if "chart-secondary" in css else "chart-dot")
                paths.append(
                    f"<circle class='{dot_class}' cx='{x:.1f}' cy='{y:.1f}' "
                    "r='3'/>")
            else:
                paths.append(
                    f"<polyline class='{css}' points='{coords}'/>")
    middle = ceiling / 2.0
    window_seconds = drift.get("window_seconds") or 60
    desc = (f"p95 latency by {window_seconds}-second window; "
            f"{len(windows)} windows. Missing values are gaps, not zeros.")
    def legend_color(css: str) -> str:
        if "chart-bad" in css:
            return "#b42318"
        if "chart-secondary" in css:
            return "#6b55c5"
        return "#075fce"

    legend = "".join(
        f"<span><span aria-hidden='true' style='color:{legend_color(css)}'"
        f">&#8212;</span> {html.escape(label)}</span>"
        for label, css, _points in series)
    return (
        "<div class='chart'><h3>Tail latency over time</h3>"
        f"<div class='chart-meta'>{html.escape(desc)} &nbsp; {legend}</div>"
        "<svg viewBox='0 0 640 170' role='img' "
        "aria-labelledby='stability-chart-title stability-chart-desc'>"
        "<title id='stability-chart-title'>Tail latency over time</title>"
        f"<desc id='stability-chart-desc'>{html.escape(desc)}</desc>"
        f"<line class='chart-axis' x1='{left}' y1='{top}' x2='{left}' "
        f"y2='{top + height}'/><line class='chart-axis' x1='{left}' "
        f"y1='{top + height}' x2='{left + width}' y2='{top + height}'/>"
        f"<line class='chart-axis' x1='{left}' y1='{top + height / 2}' "
        f"x2='{left + width}' y2='{top + height / 2}'/>"
        f"<text class='chart-label' x='2' y='{top + 4:.1f}'>"
        f"{ceiling:,.0f} ms</text>"
        f"<text class='chart-label' x='2' y='{top + height / 2 + 4:.1f}'>"
        f"{middle:,.0f}</text>"
        f"<text class='chart-label' x='29' y='{top + height + 4:.1f}'>0</text>"
        + "".join(paths)
        + f"<text class='chart-label' x='{left}' y='166'>0 min</text>"
        + f"<text class='chart-label' text-anchor='end' x='{left + width}' "
          f"y='166'>{len(windows) * float(window_seconds) / 60:g} min</text>"
        + "</svg></div>")


def _html_quota_gauges(rate: dict) -> str:
    """Show captured-window evidence separately from short-run projections.

    On a short run, ``ratio_to_nominal_limit`` is the larger of the observed
    rolling maximum and a sustained-rate projection.  It must never be
    rendered as an observed percentage beside ``observed_max``.
    """
    labels = {
        "input_tokens_per_minute": "Input tokens / trailing 60 s",
        "output_tokens_per_minute": "Offered max_tokens / trailing 60 s",
        "queries_per_hour": "Physical POSTs / trailing 3,600 s",
    }
    gauges = []
    for key, comparison in (rate.get("comparisons") or {}).items():
        configured = comparison.get("configured_limit")
        observed = comparison.get("observed_max")
        if (isinstance(configured, bool)
                or not isinstance(configured, (int, float))
                or not math.isfinite(float(configured))
                or float(configured) <= 0):
            continue
        configured = float(configured)
        observed_ratio = comparison.get("observed_ratio_to_nominal_limit")
        if (isinstance(observed_ratio, bool)
                or not isinstance(observed_ratio, (int, float))
                or not math.isfinite(float(observed_ratio))
                or float(observed_ratio) < 0):
            if (isinstance(observed, bool)
                    or not isinstance(observed, (int, float))
                    or not math.isfinite(float(observed))
                    or float(observed) < 0):
                continue
            observed_ratio = float(observed) / configured
        observed_ratio = float(observed_ratio)
        warning_at = comparison.get("warning_utilization")
        if (isinstance(warning_at, bool)
                or not isinstance(warning_at, (int, float))
                or not math.isfinite(float(warning_at))
                or not 0 < float(warning_at) <= 1):
            # Compatibility for older sealed summaries that predate the
            # per-comparison threshold field.
            warning_at = 0.8
        warning_at = float(warning_at)
        label = labels.get(key, str(key).replace("_", " ").capitalize())

        def gauge(kind: str, value: object, ratio: float,
                  qualifier: str) -> str:
            def amount(item: object) -> str:
                number = float(item)
                return (f"{number:,.0f}" if number.is_integer()
                        else f"{number:,.1f}")

            percent = ratio * 100.0
            width = min(percent, 100.0)
            tone = ("bad" if ratio >= 1.0 else
                    "warn" if ratio >= warning_at else "")
            return (
                "<div class='gauge'>"
                f"<div class='gauge-head'><span>{html.escape(label)} - "
                f"{html.escape(kind)}</span><span>{percent:.1f}%</span></div>"
                "<div class='gauge-track' role='img' "
                f"aria-label='{html.escape(label)}, {html.escape(kind)}: "
                f"{percent:.1f} percent of the configured nominal limit; "
                f"warning threshold {warning_at * 100:.1f} percent'>"
                f"<div class='gauge-fill {tone}' "
                f"style='width:{width:.3f}%'></div></div>"
                f"<div class='gauge-note'>{amount(value)} / "
                f"{amount(configured)} configured; warning at "
                f"{warning_at * 100:.1f}%. {html.escape(qualifier)} "
                "Harness-local; provider headroom is not established."
                "</div></div>")

        gauges.append(gauge(
            "observed captured window", observed, observed_ratio,
            "This is captured run evidence."))
        projected = comparison.get("steady_state_projection")
        if (not isinstance(projected, bool)
                and isinstance(projected, (int, float))
                and math.isfinite(float(projected))
                and float(projected) >= 0):
            projected = float(projected)
            gauges.append(gauge(
                "sustained-rate projection", projected,
                projected / configured,
                "This is a projection from a short observation, not an "
                "observed rolling-window maximum."))
    if not gauges:
        return ""
    return "<div class='quota-grid'>" + "".join(gauges) + "</div>"


def _html_decision_hero(decision: dict, combined_gate_html: str) -> str:
    """Render independent decision states before any performance number."""
    ordered = (
        ("Evidence integrity", "evidence_integrity"),
        ("Measurement validity", "measurement_validity"),
        ("Acceptance checks", "customer_sla"),
        ("Quota state", "quota_state"),
        ("Endpoint capacity", "endpoint_capacity"),
    )
    tone_by_severity = {
        "pass": "ok", "fail": "bad", "warning": "warn",
        "neutral": "neutral",
    }
    cards = []
    reason_rows = []
    severities = []
    for heading, key in ordered:
        state = decision[key]
        severity = str(state.get("severity") or "neutral")
        severities.append(severity)
        tone = tone_by_severity.get(severity, "neutral")
        state_label = str(state.get("label") or state.get("code") or "UNKNOWN")
        cards.append(
            f"<div class='state-card tone-{tone}'>"
            f"<div class='k'>{html.escape(heading)}</div>"
            f"<div class='v'>{html.escape(state_label)}</div>"
            f"<div class='why'>{html.escape(str(state.get('reason') or ''))}</div>"
            "</div>")
        reason_rows.append(
            f"<div><dt>{html.escape(heading)}</dt>"
            f"<dd>{html.escape(str(state.get('reason') or ''))}</dd></div>")
    quota = decision["quota_state"]
    measurement = decision["measurement_validity"]
    sla = decision["customer_sla"]
    capacity = decision["endpoint_capacity"]
    integrity = decision["evidence_integrity"]
    if integrity.get("code") == "TAMPERED":
        headline = "Do not use this report: artifact integrity failed."
        lead = integrity["reason"]
    elif quota.get("code") == "EXCEEDED":
        headline = "No endpoint-capacity conclusion: quota rejection observed."
        lead = quota["reason"]
    elif measurement.get("code") == "INVALID":
        headline = "No performance conclusion: the measurement is invalid."
        lead = measurement["reason"]
    elif sla.get("code") == "MISS":
        headline = "Configured acceptance checks missed at this tested load."
        lead = sla["reason"]
    elif measurement.get("code") == "CAUTION":
        headline = "Diagnostic result: validity gates require review."
        lead = measurement["reason"]
    elif sla.get("code") == "PASS":
        headline = "Configured acceptance checks passed at this tested load."
        lead = sla["reason"]
    else:
        headline = "Run observed; no acceptance-check pass is claimed."
        lead = sla["reason"]
    if "fail" in severities:
        hero_tone = "bad"
    elif "warning" in severities:
        hero_tone = "warn"
    else:
        hero_tone = "ok"
    established = (
        "The report records the tested workload, captured outcomes, and "
        "independent acceptance-check and rate-limit states for this run."
    )
    not_established = (
        "It does not establish an endpoint ceiling, behavior for a different "
        "workload, or provider quota headroom."
    )
    return (
        f"<section class='decision-hero state-{hero_tone}' id='overview' "
        "aria-labelledby='decision-heading'>"
        "<div class='decision-lead'><div>"
        "<div class='status-kicker'>Decision summary</div>"
        f"<h2 id='decision-heading'>{html.escape(headline)}</h2>"
        f"<p class='decision-copy'>{html.escape(str(lead))}</p>"
        "</div>"
        "<div class='claim-box'>"
        f"<p><b>What this establishes</b>{html.escape(established)}</p>"
        f"<p><b>What this does not establish</b>"
        f"{html.escape(not_established)}</p>"
        f"<p><b>Capacity state</b>{html.escape(str(capacity['label']))}</p>"
        "</div></div>"
        f"<div class='state-grid'>{''.join(cards)}</div>"
        "<details class='gate-detail' id='decision-reasons'>"
        "<summary>Why these states · combined CLI exit-code gate</summary>"
        f"<dl class='decision-reasons'>{''.join(reason_rows)}</dl>"
        + combined_gate_html + "</details></section>")


def render_html(summary: dict, title: str, *,
                verification_context: dict | None = None) -> str:
    """A self-contained, styled HTML report built from the same summary the
    markdown uses. Stdlib only, no external assets, safe to open in a browser
    or attach to a deck."""
    s = summary
    verified_view = _external_report_context(s, verification_context)
    def esc(value: object) -> str:
        return html.escape(sanitize_display_text(value), quote=True)
    run = s.get("run") or {}
    mode = run.get("input_mode", "profile")

    def num(v, nd=0):
        return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "n/a"

    def has(t):
        return bool(t) and t.get("n", 0) > 0

    def display_text(value, limit):
        clean = sanitize_title(value)
        return clean if len(clean) <= limit else clean[:limit - 1].rstrip() + "…"

    # ---- header ----
    display_title = display_text(title, 160)
    ep = esc(display_text(run.get("endpoint_path") or "", 180))
    src = ("real prompts" if mode == "prompts" else "synthetic shape")
    def count_or_none(key: str) -> int | None:
        value = s.get(key)
        return (value if isinstance(value, int) and not isinstance(value, bool)
                and value >= 0 else None)

    total = count_or_none("requests_total")
    okc = count_or_none("requests_ok")
    failed = count_or_none("requests_failed")
    error_rate = s.get("error_rate")
    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float))
            or not math.isfinite(float(error_rate))
            or float(error_rate) < 0):
        error_rate = None
    else:
        error_rate = float(error_rate)
    total_text = f"{total:,}" if total is not None else "NOT REPORTED"
    ok_text = f"{okc:,}" if okc is not None else "NOT REPORTED"
    failed_text = f"{failed:,}" if failed is not None else "NOT REPORTED"
    sub = (f"Measured replay &middot; {ep} &middot; {src} &middot; "
           f"{total_text} requests, {ok_text} harness-successful, "
           f"{failed_text} failed")

    endpoint_meta = run.get("endpoint_metadata") or {}
    entity_names = [str(entity.get("name"))
                    for entity in (endpoint_meta.get("served_entities") or [])
                    if isinstance(entity, dict) and entity.get("name")]
    tested_entity = display_text(
        run.get("endpoint_model") or
        (entity_names[0] if entity_names else None) or
        endpoint_meta.get("name") or "not recorded", 120)
    header_chips = []
    if tested_entity != "not recorded":
        header_chips.append(("entity", f"entity: {tested_entity}"))
    header_chips.extend([
        ("mode", f"input: {src}"),
        ("version", f"harness: {s.get('harness_version') or 'not recorded'}"),
    ])
    if run.get("artifact_id"):
        header_chips.append((
            "artifact", f"artifact: {display_text(run['artifact_id'], 100)}"))
    verified_banner_html = ""
    if verified_view:
        source_repro = verified_view["source_reproducibility"]
        verifier_repro = verified_view["verifier_reproducibility"]
        repro_warning = (
            source_repro["code"] == "FAILED"
            or verifier_repro["code"] == "FAILED")

        def verification_state(label: str, code: str) -> str:
            css = "status-pass" if code in {"PASS", "VERIFIED"} \
                else "status-failed"
            return (
                "<div class='verification-state'>"
                f"<span>{esc(label)}</span>"
                f"<strong class='{css}'>{esc(code)}</strong></div>")

        def reproducibility_detail(state: dict) -> str:
            reason_codes = state["reason_codes"]
            codes = (
                "<br><span class='repro-codes'>Reason codes: "
                + ", ".join(f"<code>{esc(code)}</code>"
                            for code in reason_codes)
                + "</span>" if reason_codes else "")
            return esc(state["reason"]) + codes

        verified_banner_html = (
            f"<aside class='external-verified"
            f"{' repro-warning' if repro_warning else ''}' role='status' "
            "aria-label='External verification context'>"
            f"<span class='verified-badge'>{esc(verified_view['view_label'])}"
            "</span><div class='verification-states' "
            "aria-label='Independent verification states'>"
            + verification_state("Integrity", "VERIFIED")
            + verification_state("Source reproducibility", source_repro["code"])
            + verification_state(
                "Verifier reproducibility", verifier_repro["code"])
            + "</div><dl class='verified-grid'>"
            f"<dt>Source reproducibility</dt><dd>"
            f"{reproducibility_detail(source_repro)}</dd>"
            f"<dt>Verifier reproducibility</dt><dd>"
            f"{reproducibility_detail(verifier_repro)}</dd>"
            f"<dt>Source artifact</dt><dd><code>"
            f"{esc(verified_view['source_artifact_id'])}</code></dd>"
            f"<dt>Full manifest SHA-256</dt><dd><code>"
            f"{esc(verified_view['source_manifest_sha256'])}</code></dd>"
            f"<dt>Verifier</dt><dd>llm-traffic-replay "
            f"<code>{esc(verified_view['verifier_version'])}</code> at "
            f"<code>{esc(verified_view['verified_at_utc'])}</code></dd>"
            f"<dt>Receipt</dt><dd><code>{esc(verified_view['receipt_id'])}"
            "</code></dd>"
            f"<dd class='assurance'>{esc(verified_view['assurance'])}</dd>"
            "</dl></aside>")
    eyebrow = ("Benchmark evidence · external verification receipt"
               if verified_view else "Benchmark evidence · verify the manifest")
    header_html = (
        "<header class='report-head'>"
        f"<div class='eyebrow'>{esc(eyebrow)}</div>"
        f"<h1 title='{esc(sanitize_title(title))}'>{esc(display_title)}</h1>"
        f"<p class='sub'>{sub}</p>"
        "<div class='meta-row'>"
        + "".join(f"<span class='meta-chip meta-{kind}'>"
                  f"{esc(str(chip))}</span>" for kind, chip in header_chips)
        + "</div></header>")
    nav_links = [
        ("overview", "Decision"), ("workload", "Workload"),
        ("validity", "Validity"),
    ]
    if s.get("sla"):
        nav_links.append(("sla", "Acceptance"))
    nav_links.append(("performance", "Performance"))
    if s.get("drift"):
        nav_links.append(("stability", "Stability"))
    if s.get("observed_rate_windows") or s.get("rate_limits"):
        nav_links.append(("quota", "Quota"))
    nav_links.append(("evidence", "Evidence"))
    nav_html = (
        "<nav class='report-nav' aria-label='Report sections'>"
        + "".join(f"<a href='#{target}'>{esc(label)}</a>"
                  for target, label in nav_links)
        + "</nav>")

    schedule = s.get("schedule") or {}
    scheduled_requests = schedule.get("requests")
    load_seconds = schedule.get("seconds")
    scheduled_avg = None
    if isinstance(scheduled_requests, int) and not isinstance(
            scheduled_requests, bool) \
            and isinstance(load_seconds, (int, float)) \
            and not isinstance(load_seconds, bool) \
            and math.isfinite(float(load_seconds)) and float(load_seconds) > 0:
        scheduled_avg = scheduled_requests / float(load_seconds)
    achieved_qps = (s.get("arrivals") or {}).get("achieved_qps_overall")
    conc = s.get("concurrency") or {}
    throughput = s.get("throughput") or {}
    throughput_coverage = throughput.get("usage_coverage")
    throughput_note = "endpoint-reported usage"
    if isinstance(throughput_coverage, (int, float)) \
            and not isinstance(throughput_coverage, bool) \
            and throughput_coverage < 1.0:
        throughput_note = (
            f"clean usage subset; {throughput_coverage:.1%} row coverage")
    fact_items = [
        _html_fact("Scheduled average",
                   f"{scheduled_avg:,.2f}" if scheduled_avg is not None
                   else "NOT RECORDED", "RPS",
                   "open-loop load window"),
        _html_fact("Achieved arrival rate",
                   f"{achieved_qps:,.2f}" if isinstance(
                       achieved_qps, (int, float)) and not isinstance(
                           achieved_qps, bool) else "NOT MEASURED", "RPS",
                   "first sends over observed span"),
        _html_fact("Load window",
                   f"{float(load_seconds):,.0f}" if isinstance(
                       load_seconds, (int, float)) and not isinstance(
                           load_seconds, bool) else "NOT RECORDED", "s",
                   f"{scheduled_requests} scheduled requests"
                   if scheduled_requests is not None else "schedule unavailable"),
        _html_fact("Replay requests", total_text, "requests",
                   f"{ok_text} harness-successful; {failed_text} failed"),
        _html_fact("In-flight p95",
                   f"{conc['in_flight_p95']:,.0f}" if isinstance(
                       conc.get("in_flight_p95"), (int, float))
                   and not isinstance(conc.get("in_flight_p95"), bool)
                   else "NOT MEASURED", "requests",
                   f"peak {conc.get('in_flight_max', 'unknown')}"),
        _html_fact("Input throughput",
                   f"{throughput['input_tokens_per_min']:,.0f}"
                   if isinstance(throughput.get("input_tokens_per_min"),
                                 (int, float))
                   and not isinstance(throughput.get("input_tokens_per_min"),
                                      bool) else "NOT REPORTED", "tok/min",
                   throughput_note),
    ]
    facts_html = (
        "<section id='workload' aria-labelledby='workload-heading'>"
        "<div class='section-head'><h2 id='workload-heading'>What was tested"
        "</h2><p>Load and workload facts come before latency so a light or "
        "malformed run cannot look impressive out of context.</p></div>"
        f"<div class='fact-strip'>{''.join(fact_items)}</div></section>")

    # ---- stat cards ----
    cards = []
    provenance = s.get("latency_correction_provenance") or {}

    def exact_caller_table(corrected_key: str, service_key: str) -> dict | None:
        corrected = s.get(corrected_key)
        service = s.get(service_key)
        corrected = corrected if isinstance(corrected, dict) else {}
        service = service if isinstance(service, dict) else {}
        corrected_n = corrected.get("n")
        service_n = service.get("n")
        legacy_n = provenance.get("legacy_reconstructed_values")
        if (isinstance(corrected_n, int) and corrected_n > 0
                and isinstance(service_n, int) and service_n > 0
                and corrected_n == service_n
                and legacy_n == 0):
            return corrected
        return None

    ttft_service = s.get("ttft_ms") or {}
    ttft_caller = exact_caller_table("ttft_corrected_ms", "ttft_ms")
    ttft = ttft_caller or ttft_service
    ttft_label = "Caller TTFT" if ttft_caller else "Service-path TTFT"
    if has(ttft):
        cards.append(_html_stat(f"{ttft_label} p50", num(ttft["p50"]), "ms"))
        cards.append(_html_stat(f"{ttft_label} p95", num(ttft["p95"]), "ms"))
    e2e_service = s.get("e2e_ms") or {}
    e2e_caller = exact_caller_table("e2e_corrected_ms", "e2e_ms")
    e2e = e2e_caller or e2e_service
    e2e_label = "Caller end to end" if e2e_caller else \
        "Service-path end to end"
    if has(e2e):
        cards.append(_html_stat(f"{e2e_label} p95", num(e2e["p95"]), "ms"))
    err_cls = (
        "neutral" if failed is None or error_rate is None else
        "ok" if failed == 0 and error_rate == 0 else "bad")
    err_text = "NOT REPORTED" if error_rate is None else \
        f"{error_rate * 100:.2f}%"
    cards.append(f"<div class='stat'><div class='k'>Replay error rate</div>"
                 f"<div class='v'><span class='pill {err_cls}'>"
                 f"{err_text}</span></div></div>")
    http_429_count = s.get("http_429_count")
    http_429 = s.get("http_429") or {}
    if isinstance(http_429_count, int) \
            and not isinstance(http_429_count, bool) \
            and http_429_count > 0:
        http_429_rate = http_429.get("rate")
        rendered_rate = (f"{100 * http_429_rate:.2f}%"
                         if isinstance(http_429_rate, (int, float))
                         and not isinstance(http_429_rate, bool) else "n/a")
        cards.append(
            "<div class='stat'><div class='k'>HTTP 429 rate</div>"
            "<div class='v'><span class='pill bad'>"
            f"{rendered_rate}</span></div></div>")
    ach = s.get("achieved_cache_fraction") or {}
    if has(ach):
        cards.append(_html_stat("cached prompt-token fraction p50",
                                num(ach["p50"], 2), "fraction (0-1)"))
    else:
        cards.append("<div class='stat'><div class='k'>cached prompt-token "
                     "fraction</div>"
                     "<div class='v'><span class='pill neutral' "
                     "style='font-size:12px'>not reported</span></div></div>")
    tp = s.get("throughput") or {}
    if isinstance(tp.get("output_tokens_per_min"), (int, float)) \
            and not isinstance(tp.get("output_tokens_per_min"), bool):
        cards.append(_html_stat("output throughput",
                                num(tp["output_tokens_per_min"]), "tok/min"))
    stats = f"<div class='stats'>{''.join(cards)}</div>"

    # ---- SLA banner + scorecard ----
    sla_html = ""
    banner = ""
    sla = s.get("sla")
    if sla:
        rows = []
        misses = 0
        unmeasured = 0
        for name, key in (("TTFT", "ttft_vs_target"), ("TTFG", "ttfg_vs_target")):
            for r in sla.get(key) or []:
                met = r["met"]
                if met is False:
                    misses += 1
                elif met is None and r.get("target_ms") is not None:
                    unmeasured += 1
                cls = "yes" if met else ("no" if met is False else "na")
                cell = {True: "PASS", False: "NO", None: "-"}[met]
                rows.append(
                    f"<tr><th scope='row' class='lbl sticky-col'>{name} "
                    f"{esc(r['quantile'])} (ms)</th>"
                    f"<td>{num(r['target_ms'])}</td>"
                    f"<td>{num(r['actual_ms']) if r['actual_ms'] is not None else '-'}</td>"
                    f"<td class='{cls}'>{cell}</td></tr>")
        hard_basis = sla.get("hard_timeout_basis") or {}
        hard_timeout_configured = any(
            hard_basis.get(key) is not None
            for key in ("ttft_cap_ms", "ttfg_cap_ms"))
        ht = sla.get("hard_timeout_breaches")
        if hard_timeout_configured and ht is not None:
            cls = "yes" if ht == 0 else "no"
            rows.append(f"<tr><th scope='row' class='lbl sticky-col'>"
                        f"hard timeout "
                        f"breaches (count)</th>"
                        f"<td>-</td><td>{ht}</td>"
                        f"<td class='{cls}'>{'PASS' if ht == 0 else ht}</td></tr>")
            if ht:
                misses += 1
        ib = sla.get("interchunk_breaches")
        if ib is not None:
            cls = "yes" if ib == 0 else "no"
            rows.append(f"<tr><th scope='row' class='lbl sticky-col'>"
                        f"interchunk "
                        f"breaches (count)</th>"
                        f"<td>-</td><td>{ib}</td>"
                        f"<td class='{cls}'>{'PASS' if ib == 0 else ib}</td></tr>")
            if ib:
                misses += 1
        sr = sla.get("success_rate")
        if sr:
            met = sr["met"]
            cls = "yes" if met else "no"
            if met is False:
                misses += 1
            rows.append(
                f"<tr><th scope='row' class='lbl sticky-col'>success rate "
                f"(fraction 0-1)</th>"
                f"<td>{num(sr['target'], 4)}</td><td>{num(sr['actual'], 4)}</td>"
                f"<td class='{cls}'>{'PASS' if met else 'NO'}</td></tr>")
            lower = sr.get("one_sided_95pct_wilson_lower")
            demonstrated = sr.get("statistically_demonstrated")
            if lower is not None:
                confidence_cls = "yes" if demonstrated else "no"
                rows.append(
                    "<tr><th scope='row' class='lbl sticky-col'>"
                    "success-rate one-sided "
                    "95% Wilson lower bound</th>"
                    f"<td>{num(sr['target'], 4)}</td><td>{num(lower, 4)}</td>"
                    f"<td class='{confidence_cls}'>"
                    f"{'PASS' if demonstrated else 'NOT PROVEN'}</td></tr>")
        defn = esc(sla.get("ttft_definition", "first_content"))
        note_bits = []
        ttft_rows = sla.get("ttft_vs_target") or []
        if ttft_rows and all(r["actual_ms"] is None for r in ttft_rows):
            # in profile mode the per-request budget is
            # min(sampled_output_tokens, max_output_tokens_cap), so telling
            # someone to raise the cap is advice that cannot work: the
            # sampled value is the smaller one and still wins. name the knob
            # that actually binds for the mode this run used.
            _mode = ((s.get("run") or {}).get("input_mode") or "profile")
            _knob = ("the profile's <code>output_tokens</code> quantiles "
                     "(raising <code>max_output_tokens_cap</code> alone will "
                     "not help, the per-request budget is the smaller of the "
                     "two)"
                     if _mode == "profile" else
                     "<code>max_output_tokens_cap</code>")
            fix = (f" Raise {_knob}, or set <code>ttft_definition</code> to "
                   "<code>first_content</code>, to get a number."
                   if defn != "first_content" else
                   f" Raise {_knob} so requests reach that content."
                   " On a reasoning-only model no budget may be enough, and"
                   " the mode is the decision rather than the budget.")
            note_bits.append(
                f"TTFT actual is <b>-</b> because it is scored on "
                f"<b>{defn}</b> and no request emitted that content within "
                f"max_tokens (a reasoning model can spend the whole token "
                f"budget thinking).{fix} The latency table below still shows "
                f"TTFT for the first visible-or-reasoning content delta.")
        if s.get("ttfr_ms"):
            tft = (s.get("ttft_ms") or {}).get("p50")
            note_bits.append(
                "Reasoning model detected: TTFT (first visible-or-reasoning "
                f"content delta) p50 {num(tft)} ms arrives before the first "
                "visible content.")
        slanote = (f"<div class='slanote'>{' '.join(note_bits)}</div>"
                   if note_bits else "")
        basis = esc((sla.get("latency_basis") or "unknown").replace("_", " "))
        sla_html = (
            f"<div class='card' id='sla'><h2 id='sla-heading'>"
            f"Acceptance scorecard "
            f"(TTFT definition: {defn}; latency basis: {basis})</h2>"
            f"<div class='cap'>targets from {esc(sla.get('targets_source') or 'the run configuration')}. "
            f"target and actual share each row's unit, shown in the metric "
            f"name</div>"
            + (f"<div class='banner warn'>{esc(sla['targets_warning'])}</div>"
               if sla.get("targets_warning") else "")
            + (f"<div class='banner warn'>{esc(sla['coverage_warning'])}</div>"
               if sla.get("coverage_warning") else "")
            + (f"<div class='banner warn'>"
               f"{esc(sla['caller_latency_warning'])}</div>"
               if sla.get("caller_latency_warning") else "")
            + "<div class='scroll-hint' id='sla-scroll-hint' role='note'>"
              "<span aria-hidden='true'>↔</span> Scroll horizontally; the "
              "Metric column stays visible.</div>"
              "<div class='table-scroll' tabindex='0' role='region' "
              "aria-labelledby='sla-heading' "
              "aria-describedby='sla-scroll-hint'>"
              "<table class='dense-table'><caption class='sr-only'>"
              "Configured acceptance target, actual "
              "measurement, and result</caption><thead>"
            f"<tr><th scope='col' class='lbl sticky-col'>metric</th>"
            f"<th scope='col'>target</th><th scope='col'>actual</th>"
            f"<th scope='col'>result</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>{slanote}</div>")

    # one shared verdict, so report.md and this page cannot disagree, and it
    # renders whether or not acceptance targets were given. a run with no
    # targets can still be INVALID or carry cautions worth seeing.
    vkind, vtext = _verdict(s)
    if vkind != "ok" or sla:
        vcls = {"invalid": "bad", "miss": "bad",
                "caution": "warn", "ok": "ok"}[vkind]
        vpre = "INVALID: " if vkind == "invalid" else ""
        _cap = vtext[:1].upper() + vtext[1:] if not vpre else vtext
        banner = f"<div class='banner {vcls}'>{vpre}{esc(_cap)}</div>"

    # ---- latency table ----
    lat = []
    for label, key in (("TTFT (first content delta)", "ttft_ms"),
                       ("TTF valid tool call", "ttf_tool_call_ms"),
                       ("TTFB (first response-body line)", "ttfb_ms"),
                       ("TTFG (end to end)", "e2e_ms"),
                       ("interchunk max", "interchunk_max_ms"),
                       ("TTFR (first reasoning)", "ttfr_ms"),
                       ("TTFV (first visible)", "ttfv_ms")):
        t = s.get(key)
        if has(t):
            lat.append(
                f"<tr><th scope='row' class='lbl sticky-col'>{label}</th>"
                f"<td>{num(t['p50'])}</td>"
                f"<td>{num(t['p90'])}</td><td>{num(t['p95'])}</td>"
                f"<td>{num(t['p99'])}</td><td class='n'>{t['n']}</td></tr>")
    pop_note = esc((s.get("latency_population") or {}).get("note")
                   or "latency population was not recorded")
    lat_html = (
        "<div class='card' id='performance'><h2 id='performance-heading'>"
        "Endpoint service latency "
        "(milliseconds, from send)</h2>"
        f"<div class='cap'>{pop_note}. p50 to p99 are percentiles across that "
        "population, lower is better. n is the measured count; all values are "
        "in ms.</div><div class='scroll-hint' id='performance-scroll-hint' "
        "role='note'><span aria-hidden='true'>↔</span> Scroll horizontally; "
        "the Metric column stays visible.</div>"
        "<div class='table-scroll' tabindex='0' role='region' "
        "aria-labelledby='performance-heading' "
        "aria-describedby='performance-scroll-hint'>"
        "<table class='dense-table'><caption class='sr-only'>"
        "Endpoint-service latency "
        "percentiles in milliseconds</caption><thead>"
        "<tr><th scope='col' class='lbl sticky-col'>metric</th>"
        "<th scope='col'>p50</th>"
        "<th scope='col'>p90</th><th scope='col'>p95</th>"
        f"<th scope='col'>p99</th><th scope='col'>n</th></tr></thead>"
        f"<tbody>{''.join(lat)}</tbody></table></div></div>")

    # ---- believability panel ----
    bel = []
    npth = s.get("network_path") or {}
    floor = _tcp_connect_floor(npth)
    if floor is not None:
        ratio = npth.get("tcp_connect_floor_to_ttft_p50_ratio")
        bel.append(
            f"<li><b>Network-path floor</b>: {num(floor)} ms minimum TCP "
            f"connect to {esc(npth['endpoint_host'])} "
            f"({esc(', '.join(npth['endpoint_ips'][:3]))})"
            + (f", a floor-to-TTFT-p50 ratio of {ratio:.1%}"
               if ratio is not None else "")
            + ". This is a location diagnostic, not exact RTT or endpoint "
              "processing time; do not subtract it from TTFT.</li>")
    if has(ach):
        bel.append(f"<li><b>Achieved cached prompt-token fraction</b> "
                   f"(endpoint-reported, "
                   f"0-1, share of prompt tokens served from cache): "
                   f"p50 {num(ach['p50'], 3)} / p95 {num(ach['p95'], 3)} "
                   f"(field: {esc(', '.join(ach.get('source_fields') or []))})"
                   f"</li>")
    else:
        bel.append("<li><b>Achieved cached prompt-token fraction</b>: not "
                   "reported by this "
                   "endpoint (shown as unknown, never guessed)</li>")
    if mode == "prompts":
        bel.append("<li><b>Input</b>: real prompts replayed verbatim, sizes "
                   "and any cache reuse are the prompts' own</li>")
    else:
        intent = s.get("intended_cache_fraction") or {}
        tt = s.get("token_targeting") or {}
        if intent.get("n"):
            bel.append(f"<li><b>Constructed cache fraction</b> (intended): "
                       f"p50 {num(intent['p50'], 3)} / p95 "
                       f"{num(intent['p95'], 3)}</li>")
        if tt.get("reported_over_intended_p50"):
            bel.append(f"<li><b>Token targeting</b>: reported/intended p50 "
                       f"{num(tt['reported_over_intended_p50'], 3)} "
                       f"(abs error {num(tt['abs_error_pct_p50'], 1)}%)</li>")
    _reason_source = str(s.get("reasoning_tokens_source") or "")
    _legacy_reasoning_deltas = (
        s.get("reasoning_tokens_total")
        if "stream-counted" in _reason_source.lower() else None)
    rt = (None if _legacy_reasoning_deltas is not None
          else s.get("reasoning_tokens_total"))
    if rt is not None:
        rpm = (s.get("throughput") or {}).get("reasoning_tokens_per_min")
        pm = f", {num(rpm)}/min" if rpm else ""
        bel.append(f"<li><b>Reasoning tokens</b> (thinking tokens): {num(rt)} "
                   f"tokens total{pm} "
                   f"(field: {esc(str(s.get('reasoning_tokens_source')))})</li>")
    rd = (s.get("reasoning_stream_deltas_total")
          if s.get("reasoning_stream_deltas_total") is not None
          else _legacy_reasoning_deltas)
    if rd is not None:
        rpm = ((s.get("throughput") or {}).get(
            "reasoning_stream_deltas_per_min")
            or ((s.get("throughput") or {}).get("reasoning_tokens_per_min")
                if _legacy_reasoning_deltas is not None else None))
        pm = f", {num(rpm)} deltas/min" if rpm else ""
        bel.append(
            f"<li><b>Reasoning stream deltas</b>: {num(rd)} deltas total{pm} "
            f"(source: {esc(str(s.get('reasoning_stream_deltas_source') or _reason_source))}). "
            "These are SSE chunks, not tokens.</li>")
    arr = s.get("arrivals") or {}
    if arr.get("achieved_qps_overall"):
        lag = (arr.get("dispatch_lag_ms") or {}).get("p95")
        bel.append(f"<li><b>Arrival honesty</b>: "
                   f"{num(arr['achieved_qps_overall'], 2)} requests/second "
                   f"(QPS) overall. Dispatch lag p95 {num(lag)} ms is how "
                   f"late the dispatcher handed the request to the pool. "
                   f"Wire lateness p95 {_wire_p95(arr)} is how late it "
                   f"actually reached the endpoint, which is the one that "
                   f"grows when the offered load is not being delivered: a "
                   f"full pool queues rather than blocking the dispatcher. "
                   f"Neither is endpoint latency."
                   + (f" {esc(arr['wire_lateness_note'])}"
                      if arr.get("wire_lateness_note") else "")
                   + "</li>")
    conn = s.get("connect_ms") or {}
    if conn.get("n"):
        bel.append(f"<li><b>Connection setup</b> (DNS, TCP and TLS "
                   f"setup, in ms): p50 {num(conn['p50'])} / "
                   f"p95 {num(conn['p95'])}. This is <b>excluded</b> from "
                   f"TTFT, TTFB and TTFG, so do not subtract it again. A "
                   f"handshake takes several round trips, so treat it as an "
                   f"upper bound on network distance rather than the "
                   f"per-request network cost a pooled production client "
                   f"pays. Run the client from where production traffic "
                   f"originates for it to mean anything.</li>")
    fr = (s.get("token_targeting") or {}).get("finish_reasons")
    if fr:
        bel.append(f"<li><b>Finish reasons</b>: {esc(json.dumps(fr))} "
                   f"(stop vs length)</li>")
    if failed:
        bel.append(f"<li><b>Failures</b>: "
                   f"{esc(json.dumps(s.get('failures_by_error')))}</li>")
        bel.append(f"<li><b>Failed requests by HTTP status</b>: "
                   f"{esc(json.dumps(s.get('failures_by_http_status') or {}))}"
                   "</li>")
    else:
        bel.append("<li><b>Failures</b>: none</li>")
        bel.append("<li><b>Failed requests by HTTP status</b>: none</li>")
    if isinstance(http_429_count, int) \
            and not isinstance(http_429_count, bool) \
            and http_429_count > 0:
        total_429 = http_429.get("request_rows_examined")
        rate_429 = http_429.get("rate")
        rendered_429_rate = (f"{100 * rate_429:.2f}%"
                             if isinstance(rate_429, (int, float))
                             and not isinstance(rate_429, bool) else "n/a")
        bel.append(
            f"<li><b>HTTP 429 rate-limit responses</b>: {http_429_count} of "
            f"{esc(str(total_429))} request rows ({rendered_429_rate}); "
            f"scope: {esc(str(http_429.get('scope') or 'not recorded'))}. "
            "This is quota-limited evidence, not an endpoint-capacity "
            "result.</li>")
    rp = run.get("request_params")
    if rp:
        eb = rp.get("extra_body") or {}
        extra = f", extra_body {esc(json.dumps(eb))}" if eb else ""
        bel.append(f"<li><b>Request params</b>: temperature "
                   f"{esc(str(rp.get('temperature')))}, global max_tokens "
                   "safety cap "
                   f"{esc(str(rp.get('max_output_tokens_cap')))}{extra}</li>")
    cc = s.get("concurrency") or {}
    if cc.get("in_flight_p50") is not None:
        sized = (f", open-loop sizing input "
                 f"{cc['sizing_concurrency_requested']}"
                 if cc.get("sizing_concurrency_requested") else "")
        bel.append(f"<li><b>Concurrency in flight</b>: p50 "
                   f"{cc['in_flight_p50']:.0f}, p95 {cc['in_flight_p95']:.0f}, peak "
                   f"{cc['in_flight_max']:.0f}{sized} "
                   f"({esc(cc['measured_over'])})</li>")
    lb = s.get("latency_basis")
    if lb:
        bel.append(f"<li><b>Latency basis</b>: {esc(lb)}</li>")

    believe_items = "".join(bel)
    believe = (
        "<details class='evidence believe' id='evidence'>"
        "<summary>Measurement evidence: read before quoting a number</summary>"
        "<div class='detail-body'><h2 class='sr-only'>Believability "
        "(read before quoting a number)</h2>"
        f"<ul>{believe_items}</ul></div></details>"
        "<section class='card believe print-evidence' "
        "aria-labelledby='print-evidence-heading'>"
        "<h2 id='print-evidence-heading'>Measurement evidence: read before "
        "quoting a number</h2>"
        f"<ul>{believe_items}</ul></section>")

    # ---- throughput + merge note ----
    extra_cards = ""
    if tp.get("input_tokens_per_min"):
        usage_coverage = tp.get("usage_coverage")
        incomplete_usage = (
            isinstance(usage_coverage, (int, float))
            and not isinstance(usage_coverage, bool)
            and usage_coverage < 1.0)
        throughput_heading = (
            "Throughput: clean usage subset"
            if incomplete_usage else "Throughput")
        throughput_warning = (
            f"<div class='banner warn'>{esc(tp['coverage_warning'])}</div>"
            if incomplete_usage and tp.get("coverage_warning") else "")
        extra_cards = (
            f"<div class='card'><h2>{throughput_heading}</h2>"
            f"{throughput_warning}<table>"
            "<caption class='sr-only'>Endpoint-reported token throughput"
            "</caption><tbody>"
            f"<tr><th scope='row' class='lbl'>input tokens per minute</th>"
            f"<td>{num(tp['input_tokens_per_min'])} tok/min</td></tr>"
            f"<tr><th scope='row' class='lbl'>output tokens per minute</th>"
            f"<td>{num(tp['output_tokens_per_min'])} tok/min</td></tr>"
            f"</tbody></table></div>")
    windows = s.get("observed_rate_windows") or {}
    win_input = windows.get("input_tokens_by_first_send") or {}
    win_reserved = (
        windows.get(
            "offered_output_token_reservation_demand_by_first_send") or {})
    win_actual = windows.get("actual_output_tokens_by_completion") or {}
    win_queries = windows.get("physical_queries_by_first_send") or {}
    if any(window.get("max") is not None
           for window in (win_input, win_reserved, win_actual, win_queries)) \
            or s.get("rate_limits"):
        traffic_scope = windows.get("traffic_scope") or {}
        phase_text = _traffic_phase_summary(traffic_scope)
        coverage = win_input.get("coverage")
        rows = [
            ("captured traffic phases", phase_text),
            ("input tokens / trailing 60 s",
             f"{num(win_input.get('max'))} tok "
             + (f"({num(coverage * 100, 1)}% coverage)"
                if coverage is not None else "(coverage n/a)")),
            ("offered max_tokens demand / trailing 60 s",
             f"{num(win_reserved.get('max'))} tok; pre-admission demand, "
             "not observed consumption"),
            ("actual output attributed to completion / trailing 60 s",
             f"{num(win_actual.get('max'))} tok (approximate timing)"),
            ("offered physical POST demand / trailing 3,600 s",
             f"{num(win_queries.get('max'))}; not confirmed processed QPH"),
        ]
        rate = s.get("rate_limits") or {}
        for name, comparison in (rate.get("comparisons") or {}).items():
            observed_ratio = comparison.get(
                "observed_ratio_to_nominal_limit")
            ratio = comparison.get("ratio_to_nominal_limit")
            projected = comparison.get("steady_state_projection")
            configured_limit = comparison.get("configured_limit")
            projected_ratio = (
                float(projected) / float(configured_limit)
                if isinstance(projected, (int, float))
                and not isinstance(projected, bool)
                and isinstance(configured_limit, (int, float))
                and not isinstance(configured_limit, bool)
                and configured_limit else None)
            rows.append((
                name.replace("_", " "),
                f"observed {comparison.get('observed_max')} / configured "
                f"{configured_limit}"
                + ("; observed ratio n/a" if observed_ratio is None else
                   f"; observed ratio {observed_ratio:.1%}")
                + (f"; sustained projection {projected:.1f} "
                   f"({projected_ratio:.1%})"
                   if projected is not None else "")
                + ("; conservative gate ratio n/a" if ratio is None else
                   f"; conservative gate ratio {ratio:.1%}")
                + f" ({str(comparison.get('status')).replace('_', ' ')})"))
        warning = ""
        if rate:
            cfg = rate.get("configured") or {}
            binding = rate.get("binding") or {}
            if not binding.get("binding_complete"):
                binding_label = "NOT VERIFIED"
            elif binding.get("workspace_tier_verified"):
                binding_label = (
                    "endpoint/model/deployment metadata and workspace tier "
                    "verified")
            else:
                binding_label = (
                    "endpoint/model/deployment metadata bound; workspace "
                    "tier remains operator-asserted")
            warning = (
                f"<p><b>Configured snapshot:</b> provider "
                f"{esc(str(cfg.get('provider')))}, model "
                f"{esc(str(cfg.get('model')))}, deployment "
                f"{esc(str(cfg.get('deployment_mode')))}, tier "
                f"{esc(str(cfg.get('workspace_tier')))}; "
                f"{esc(str(cfg.get('source')))} as of "
                f"{esc(str(cfg.get('as_of')))}; operator reverified "
                f"{esc(str(cfg.get('verified_at') or 'NOT RECORDED'))} with "
                f"max age "
                f"{esc(str(cfg.get('max_age_days') or 'NOT RECORDED'))} "
                "days.</p>"
                f"<p><b>Scope:</b> {esc(str(cfg.get('scope')))}. "
                f"<b>Endpoint binding:</b> {esc(binding_label)}.</p>"
                + (f"<p class='warn'>{esc(str(rate['warning']))}</p>"
                   if rate.get("warning") else "")
                + f"<p>{esc(str(rate['external_usage_warning']))}</p>")
        rate_metric_keys = " ".join(
            str(name) for name in (rate.get("comparisons") or {}))
        extra_cards += (
            "<div class='card' id='quota' data-rate-metrics='"
            + esc(rate_metric_keys)
            + "'><h2>Rolling rate windows</h2>"
            + _html_quota_gauges(rate)
            + "<table><caption class='sr-only'>Exact rolling rate-window "
              "evidence and configured-limit comparisons</caption><tbody>"
            + "".join(
                f"<tr><th scope='row' class='lbl'>{esc(label)}</th>"
                f"<td>{esc(value)}</td></tr>"
                for label, value in rows)
            + f"</tbody></table>{warning}</div>")
    merge_note = run.get("merge_note")
    note_html = (f"<div class='label-note'>{esc(merge_note)}</div>"
                 if merge_note else "")

    # ---- provenance label ----
    # both, never one or the other. the profile carries its own warning (a
    # validation profile says never to quote its latency), and setting a run
    # label must not be able to hide it.
    parts = []
    if run.get("label"):
        parts.append(f"<div class='label-note'><b>Label:</b> "
                     f"{esc(run['label'])}</div>")
    if run.get("profile_label"):
        parts.append(f"<div class='label-note'><b>Profile:</b> "
                     f"{esc(run['profile_label'])}</div>")
    label_html = "".join(parts)
    provenance_html = (
        "<div class='run-context-notes' aria-label='Run context notes'>"
        f"{note_html}{label_html}</div>"
        if note_html or label_html else "")

    cost = s.get("cost")
    cost_html = ""
    if cost and cost.get("error"):
        cost_html = (f"<div class='card'><h2>Cost</h2>"
                     f"<div class='cap'>config error: {esc(cost['error'])}</div>"
                     f"</div>")
    elif cost and cost["mode"] == "per_token" and cost.get("coverage_warning"):
        cost_html = (
            "<div class='card'><h2>Unverified user-supplied rate arithmetic</h2>"
            "<div class='banner warn'>Aggregate replay total is unavailable. "
            + esc(cost["coverage_warning"])
            + "</div><div class='cap'>"
            + esc(cost.get("applicability_warning") or "")
            + "</div></div>")
    elif cost and cost["mode"] == "per_token" \
            and (cost.get("dbu_per_request") or {}).get("p50") is None:
        cost_html = ("<div class='card'><h2>Unverified user-supplied rate arithmetic</h2>"
                     "<div class='cap'>no successful requests to price</div>"
                     "</div>")
    elif cost and cost["mode"] == "per_token":
        usd = cost.get("usd_per_dbu")
        r = cost.get("rates_dbu_per_m") or {}

        def _money(dbu, nd=4):
            base = f"{num(dbu, nd)} DBU"
            if usd is not None and dbu is not None:
                base += f" (${num(dbu * usd, nd)})"
            return base
        rows = [
            f"<tr><th scope='row' class='lbl'>DBU per request (p50)</th>"
            f"<td>{_money(cost['dbu_per_request']['p50'])}</td></tr>",
            f"<tr><th scope='row' class='lbl'>DBU per request (p95)</th>"
            f"<td>{_money(cost['dbu_per_request']['p95'])}</td></tr>",
            f"<tr><th scope='row' class='lbl'>DBU per 1,000 requests</th>"
            f"<td>{_money(cost['dbu_per_1k_requests'], 2)}</td></tr>",
            f"<tr><th scope='row' class='lbl'>DBU per minute</th>"
            f"<td>{_money(cost['dbu_per_min'], 3)}</td></tr>",
            f"<tr><th scope='row' class='lbl'>cache DBUs saved</th>"
            f"<td>{_money(cost['cache_dbu_saved'], 3)}</td></tr>",
        ]
        cap = (f"Measured replay rows only. Per-token rates you supplied "
               f"(DBU/M): input {num(r.get('input'), 3)}, "
               f"output {num(r.get('output'), 3)}, cache-read {num(r.get('cache_read'), 3)}"
               + (f", at ${usd}/DBU" if usd else "")
               + ". Cached input uses the supplied cache-read rate.")
        cost_html = ("<div class='card'><h2>Unverified user-supplied rate arithmetic</h2>"
                     f"<div class='banner warn'>{esc(cost.get('applicability_warning') or '')}</div>"
                     f"<div class='cap'>{cap}</div><table>"
                     "<caption class='sr-only'>Estimated per-token cost"
                     "</caption><tbody>"
                     f"{''.join(rows)}</tbody></table></div>")
    elif cost and cost["mode"] == "provisioned" \
            and cost.get("coverage_warning"):
        cost_html = (
            "<div class='card'><h2>Unverified provisioned-rate arithmetic</h2>"
            "<div class='banner warn'>Effective cost per 1M tokens is "
            "unavailable. " + esc(cost["coverage_warning"]) + "</div>"
            "<div class='cap'>Configured capacity rate: "
            f"{num(cost['dbu_per_hour'], 3)} DBU/hour. "
            + esc(cost.get("applicability_warning") or "")
            + "</div></div>")
    elif cost:
        usd = cost.get("usd_per_dbu")
        eff = cost.get("effective_dbu_per_1m_tokens")
        effv = (f"{num(eff, 1)} DBU"
                + (f" (${num(eff * usd, 2)})" if usd and eff is not None else "")
                if eff is not None else "throughput too low to compute")
        rows = [
            f"<tr><th scope='row' class='lbl'>capacity rate</th>"
            f"<td>{num(cost['dbu_per_hour'], 3)} DBU/hour"
            + (f" (${num(cost['dbu_per_hour'] * usd, 3)})" if usd else "")
            + "</td></tr>",
            f"<tr><th scope='row' class='lbl'>effective cost per 1M tokens</th>"
            f"<td>{effv}</td></tr>",
        ]
        cost_html = ("<div class='card'><h2>Unverified provisioned-rate arithmetic</h2>"
                     f"<div class='banner warn'>{esc(cost.get('applicability_warning') or '')}</div>"
                     "<div class='cap'>provisioned throughput "
                     f"bills by capacity, so effective cost per 1M tokens is the "
                     f"hourly rate over tokens served per hour at the measured "
                     f"throughput. it improves as you fill the endpoint.</div>"
                     "<table><caption class='sr-only'>Estimated provisioned "
                     "capacity cost</caption><tbody>"
                     f"{''.join(rows)}</tbody></table></div>")

    issues = []

    def add_issue(label, value):
        if value:
            issues.append((label, str(value)))

    add_issue("Sample size", (s.get("sample") or {}).get("warning"))
    add_issue("Prompt replay", (s.get("replay") or {}).get("warning"))
    add_issue("Load delivery", (s.get("client") or {}).get("warning"))
    add_issue("Concurrency", (s.get("concurrency") or {}).get("warning"))
    add_issue("Network path", (s.get("network_path") or {}).get("warning"))
    add_issue("Token-usage coverage",
              (s.get("throughput") or {}).get("coverage_warning"))
    add_issue("Cost coverage", (s.get("cost") or {}).get("coverage_warning"))
    add_issue("Pricing applicability",
              (s.get("cost") or {}).get("applicability_warning"))
    add_issue("Cache fidelity", (s.get("cache_fidelity") or {}).get("warning"))
    add_issue("Token-shape fidelity",
              (s.get("token_targeting") or {}).get("warning"))
    add_issue("Latency population",
              (s.get("latency_population") or {}).get("warning"))
    sample_banner = ""
    if issues:
        sample_banner = (
            "<section class='card issue-card' id='validity' "
            "aria-labelledby='validity-heading'>"
            "<h2 id='validity-heading'>Validity and workload-fidelity gates</h2>"
            f"<div class='banner warn'>{len(issues)} issue"
            f"{'s' if len(issues) != 1 else ''} must be read before using "
            "the latency or cost figures.</div><ul>"
            + "".join(
                f"<li><b>{esc(label)}:</b> {esc(value)}</li>"
                for label, value in issues)
            + "</ul></section>")
    else:
        sample_banner = (
            "<section class='card issue-card' id='validity' "
            "aria-labelledby='validity-heading'>"
            "<h2 id='validity-heading'>Validity and workload-fidelity gates"
            "</h2><span class='pill neutral'>No additional warning blocks</span>"
            "<p class='cap' style='margin-top:8px'>Use the independent "
            "Measurement validity, Quota, and Evidence integrity states above "
            "as the decision gates.</p></section>")

    drift = s.get("drift") or {}
    if drift.get("windows") or drift.get("drift_kind"):
        wr = "".join(
            f"<tr><th scope='row' class='lbl sticky-col'>window {w['window']} "
            f"({w['n']} content-bearing streams)"
            f"{'' if w.get('counted', True) else ', not counted'}</th>"
            f"<td>{_err_cell(w)}</td>"
            f"<td>{num(w['ttft_p95'])}</td><td>{num(w['e2e_p95'])}</td></tr>"
            for w in (drift.get("windows") or []))
        kind = drift.get("drift_kind")
        if not kind:
            flag = "<span class='pill neutral'>not enough data</span>"
        elif kind == "stable":
            flag = "<span class='pill ok'>stable</span>"
        else:
            flag = f"<span class='pill bad'>unstable: {esc(kind)}</span>"
        spread = drift.get("ttft_p95_spread_ratio")
        sp = (f"worst window is {spread:.1f}x the best. " if spread else "")
        drift_html = (
            f"<div class='card' id='stability'><h2 id='stability-heading'>"
            f"Stability over time "
            f"&nbsp;{flag}</h2>"
            f"<div class='cap'>"
            f"{'per-' + str(drift.get('window_seconds', 60)) + 's windows, counts and p95 in ms. ' if drift.get('windows') else ''}"
            f"{sp}"
            f"{esc(drift.get('drift_headline') or drift.get('note', ''))}"
            f"{('<br>' + esc(drift.get('note', ''))) if drift.get('drift_headline') else ''}"
            f"</div>"
            + _html_stability_chart(drift)
            + (f"<div class='scroll-hint' id='stability-scroll-hint' "
               f"role='note'><span aria-hidden='true'>↔</span> Scroll "
               f"horizontally; the Window column stays visible.</div>"
               f"<div class='table-scroll' tabindex='0' role='region' "
               f"aria-labelledby='stability-heading' "
               f"aria-describedby='stability-scroll-hint'>"
               f"<table class='dense-table'><caption class='sr-only'>"
               f"Exact per-window stability "
               f"values in milliseconds</caption><thead><tr>"
               f"<th scope='col' class='lbl sticky-col'>window</th>"
               f"<th scope='col'>errors</th><th scope='col'>TTFT p95</th>"
               f"<th scope='col'>E2E p95</th></tr></thead><tbody>{wr}</tbody>"
               f"</table></div>"
               if drift.get("windows") else "")
            + "</div>")
    else:
        drift_html = (f"<div class='card' id='stability'>"
                      f"<h2>Stability over time</h2>"
                      f"<div class='cap'>{esc(drift.get('note', ''))}</div></div>"
                      if drift.get("note") else "")

    em = run.get("endpoint_metadata")
    em_html = ""
    if em:
        se = (em.get("served_entities") or [])
        detail = ""
        if se:
            detail = ", ".join(f"{esc(str(k))}: {esc(str(v))}"
                               for k, v in se[0].items() if k != "name")
        em_html = (
            f"<div class='card'><h2>Endpoint under test</h2>"
            f"<div class='cap'>read from the serving-endpoints API at run time, "
            f"so the report states what was tested</div><table>"
            "<caption class='sr-only'>Endpoint metadata recorded at run "
            "time</caption><tbody>"
            f"<tr><th scope='row' class='lbl'>name</th><td>{esc(str(em.get('name')))}</td></tr>"
            + (f"<tr><th scope='row' class='lbl'>task</th>"
               f"<td>{esc(str(em.get('task')))}</td></tr>"
               if em.get("task") else "")
            + f"<tr><th scope='row' class='lbl'>route optimized</th>"
            f"<td>{esc(str(em.get('route_optimized')))}</td></tr>"
            f"<tr><th scope='row' class='lbl'>ready</th><td>{esc(str(em.get('ready')))}</td></tr>"
            + (f"<tr><th scope='row' class='lbl'>served entity</th><td>{detail}</td></tr>"
               if detail else "")
            + "</tbody></table></div>")

    # the html is the artifact the README sends people to, so it must carry
    # the same facts the markdown does. answer counts, caller-experienced
    # latency and cap-driven truncation were markdown-only, which is exactly
    # the set the preflight tells a customer to go and read.
    ans_html = ""
    a = s.get("answers")
    if a:
        rate = (f"{a['answer_rate']:.1%}" if a.get("answer_rate") is not None
                else "n/a")
        rows_a = [("attempted", a.get("attempted")),
                  ("harness-successful",
                   a.get("harness_successful", a.get("transport_ok"))),
                  ("produced at least one visible or reasoning content delta",
                   a.get("content_delta_streams", "NOT RECORDED")),
                  ("produced a readable answer or valid tool call",
                   f"{a.get('answered')} ({rate} of "
                   f"{a.get('judged')} judged)"),
                  ("valid tool-call outcomes",
                   f"{a.get('valid_tool_call_outcomes', 0)} "
                   f"({a.get('tool_call_only_outcomes', 0)} tool-call-only; "
                   f"{a.get('valid_tool_calls_total', 0)} calls total)"),
                  ("judged request with neither visible content nor a valid tool call",
                   a.get("no_acceptable_outcome", a.get("no_visible_content"))),
                  ("judged request with no visible content",
                   a.get("no_visible_content")),
                  ("stream never terminated", a.get("stream_incomplete")),
                  ("unrecoverable parse errors", a.get("parse_errors")),
                  ("stopped at the requested output length",
                   a.get("truncated")),
                  ("cut short by the global token cap",
                   a.get("truncated_by_global_cap"))]
        if a.get("unclassified_legacy_successes"):
            rows_a.insert(3, (
                "legacy successes without content/tool observability",
                a["unclassified_legacy_successes"]))
        if a.get("http_status_observed_for"):
            rows_a.insert(2, (
                "returned HTTP 200",
                f"{a.get('http_200')} (status recorded for "
                f"{a.get('http_status_observed_for')} requests)"))
        ans_html = (
            "<div class='card'><h2>Answers</h2><table>"
            "<caption class='sr-only'>Answer and stream outcome counts"
            "</caption><tbody>"
            + "".join(f"<tr><th scope='row' class='lbl'>{esc(k)}</th>"
                      f"<td>{esc(str(v))}</td></tr>" for k, v in rows_a)
            + f"</tbody></table><div class='cap'>{esc(a.get('note') or '')}</div>"
            + (f"<div class='banner bad'>{esc(a['invalid'])}</div>"
               if a.get("invalid") else "")
            + "</div>")

    corr_html = ""
    if s.get("e2e_corrected_ms"):
        c1 = s.get("ttft_corrected_ms") or {}
        cv = s.get("ttfv_corrected_ms") or {}
        ct = s.get("ttf_tool_call_corrected_ms") or {}
        c2 = s["e2e_corrected_ms"]
        r_ = []
        if c1.get("p50") is not None:
            r_.append(("TTFT corrected (ms)", c1))
        if cv.get("p50") is not None:
            r_.append(("TTFV corrected (ms)", cv))
        if ct.get("p50") is not None:
            r_.append(("TTF valid tool call corrected (ms)", ct))
        r_.append(("end-to-end corrected (ms)", c2))
        corr_html = (
            "<div class='card'><h2 id='caller-latency-heading'>"
            "Latency as the caller experienced it</h2>"
            "<div class='cap'>Includes time the request waited on the "
            "client.</div><div class='scroll-hint' "
            "id='caller-latency-scroll-hint' role='note'>"
            "<span aria-hidden='true'>↔</span> Scroll horizontally; the "
            "Metric column stays visible.</div>"
            "<div class='table-scroll' tabindex='0' role='region' "
            "aria-labelledby='caller-latency-heading' "
            "aria-describedby='caller-latency-scroll-hint'>"
            "<table class='dense-table'><caption class='sr-only'>"
            "Client-corrected "
            "latency percentiles in milliseconds</caption><thead><tr>"
            "<th scope='col' class='lbl sticky-col'>metric</th>"
            "<th scope='col'>p50</th>"
            "<th scope='col'>p95</th><th scope='col'>p99</th></tr></thead><tbody>"
            + "".join(
                f"<tr><th scope='row' class='lbl sticky-col'>{esc(n)}</th>"
                      f"<td>{num(t['p50'])}</td><td>{num(t['p95'])}</td>"
                      f"<td>{num(t['p99'])}</td></tr>" for n, t in r_)
            + "</tbody></table></div><div class='cap'>"
            + esc(s.get("latency_correction_note") or "") + "</div></div>")

    from .report_decision import build_report_decision

    decision = (verified_view["decision"] if verified_view
                else build_report_decision(s))
    decision_html = _html_decision_hero(decision, banner)
    artifact_label = display_text(
        (s.get("run") or {}).get("artifact_id") or "NOT RECORDED", 120)
    if verified_view:
        print_stamp = (
            "<div class='print-footer' role='note'>EXTERNAL VERIFIED VIEW · "
            "PRINT/PDF DERIVATIVE: source artifact "
            f"{esc(verified_view['source_artifact_id'])} · full manifest "
            f"SHA-256 {esc(verified_view['source_manifest_sha256'])} · "
            f"verified by llm-traffic-replay "
            f"{esc(verified_view['verifier_version'])} at "
            f"{esc(verified_view['verified_at_utc'])} · "
            f"source reproducibility "
            f"{esc(verified_view['source_reproducibility']['code'])} · "
            f"verifier reproducibility "
            f"{esc(verified_view['verifier_reproducibility']['code'])} · "
            f"{esc(verified_view['assurance'])}</div>")
        foot_html = (
            "<div class='foot'>EXTERNAL VERIFIED VIEW · source run remains "
            "immutable · internal hash consistency is not a digital "
            "signature</div>")
    else:
        print_stamp = (
            "<div class='print-footer' role='note'>UNSEALED PRINT/PDF "
            "DERIVATIVE: verify the source manifest · artifact "
            f"{esc(artifact_label)} · internal hashes are not a digital "
            "signature</div>")
        foot_html = (
            "<div class='foot'>llm-traffic-replay report · artifact integrity "
            "requires manifest verification</div>")
    body = (
        f"<main class='wrap'>{verified_banner_html}{header_html}{print_stamp}"
        f"{nav_html}{decision_html}{provenance_html}"
        f"{facts_html}{sample_banner}{stats}{believe}"
        f"{em_html}{ans_html}{sla_html}{corr_html}{lat_html}"
        f"{drift_html}{extra_cards}{cost_html}"
        f"{foot_html}</main>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{esc(title)}</title>{_HTML_STYLE}"
            f"</head><body>{body}</body></html>")
