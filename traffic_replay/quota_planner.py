"""Conservative, pre-traffic quota planning for pay-per-token runs.

The provider limits supported here are admission controls, not endpoint
capacity.  This module therefore answers one deliberately narrow question:
could traffic emitted by this harness alone cross the configured warning
budget?  It never claims that provider headroom exists, because unrelated
workspace traffic is not observable from a local load generator.

Input-token planning uses a tokenizer-independent upper bound: one token per
UTF-8 byte of the complete submitted JSON body plus a fixed chat-framing
allowance. This includes message roles and metadata, model, tool schemas,
provider controls, and wire-level JSON framing rather than counting only
message content. Synthetic replay is planned at the larger of its configured
chars-per-token value and the hard post-calibration ceiling, so calibration
cannot enlarge an already-authorized request beyond the pre-traffic plan.
Output planning uses the offered ``max_tokens`` reservation, which is the
conservative admission-time quantity for the Databricks FMAPI pay-per-token
accounting model.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from collections import deque
from datetime import date
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - imports are runtime-local by design
    from .runner import RunConfig


class QuotaPlanError(ValueError):
    """A configured workload is not safe to start under its quota snapshot."""

    def __init__(self, plan: dict):
        self.plan = plan
        reasons = plan.get("refusal_reasons") or ["quota plan is incomplete"]
        if plan.get("refusal_stage") == "endpoint_binding":
            prefix = "endpoint binding refused before paid inference traffic: "
        else:
            prefix = "quota plan refused before endpoint traffic: "
        super().__init__(prefix + "; ".join(
            str(reason) for reason in reasons))


# ``calibrate_cpt`` in textgen.py clamps every measured value to 12.0.  Keep
# this explicit in the quota artifact: changing that runtime ceiling requires
# changing this planner and its regression test in the same review.
_CALIBRATED_CPT_HARD_MAX = 12.0

# Chat-template framing is provider-owned and is not present in the submitted
# JSON. Reserve this amount for every message plus one request-level block; a
# single fixed request allowance would become unsafe for long conversations.
# The complete JSON body itself is charged at the stricter tokenizer-
# independent limit of one token per UTF-8 byte.
_CHAT_FRAMING_TOKEN_ALLOWANCE = 64

# TextMaterializer's ASCII prose can require JSON escaping only for paragraph
# newlines. Six sentences of at least eight one-character words occupy more
# than 100 characters before the next two-newline paragraph marker. Reserve
# two escaped-newline expansion bytes per 100 content characters plus four for
# the suffix separator and rounding across prefix/suffix prose components.
# Regression tests compare this analytical bound with fully serialized bodies
# across seeds, prefix shapes, and every supported CPT extreme.
_SYNTHETIC_JSON_ESCAPE_BLOCK_CHARS = 100


def _synthetic_json_escape_overhead(content_chars: int) -> int:
    if content_chars <= 0:
        return 4
    return (2 * math.ceil(
        content_chars / _SYNTHETIC_JSON_ESCAPE_BLOCK_CHARS) + 4)


def _snapshot_freshness(rate_limits: dict, *, today: date | None = None) \
        -> dict:
    """Return an explicit, fail-closed freshness assessment.

    ``as_of`` describes the provider fact. ``verified_at`` records when the
    operator actually rechecked that fact. Paid traffic is allowed only while
    the latter remains inside the configured review window.
    """
    checked_on = today or date.today()
    verified_at = rate_limits.get("verified_at")
    max_age_days = rate_limits.get("max_age_days")
    out = {
        "status": "missing",
        "fresh": False,
        "checked_on": checked_on.isoformat(),
        "verified_at": verified_at,
        "max_age_days": max_age_days,
        "age_days": None,
        "source": rate_limits.get("source"),
        "source_as_of": rate_limits.get("as_of"),
    }
    if not isinstance(verified_at, str) \
            or isinstance(max_age_days, bool) \
            or not isinstance(max_age_days, int) \
            or max_age_days <= 0:
        return out
    try:
        verified_date = date.fromisoformat(verified_at)
    except ValueError:
        out["status"] = "invalid"
        return out
    age_days = (checked_on - verified_date).days
    out["age_days"] = age_days
    if age_days < 0:
        out["status"] = "invalid"
    elif age_days > max_age_days:
        out["status"] = "stale"
    else:
        out["status"] = "fresh"
        out["fresh"] = True
    return out


def _attempt_multiplier(rc: "RunConfig") -> int:
    """Worst configured number of POSTs one logical request can produce.

    ``max_retries`` covers transport retries.  The client can additionally
    retry once when ``stream_options`` is rejected and once after a proven
    credential refresh.  Counting both for every logical request is
    intentionally conservative; concurrent first requests can race before
    shared client state learns either result.
    """
    retries = int(rc.endpoint.get("max_retries", 0) or 0)
    return retries + 3


def _schedule(rc: "RunConfig") -> list[float]:
    from .schedule import load_trace, make_schedule

    if rc.timestamps_file:
        generated = load_trace(
            rc.timestamps_file, duration_cap_s=rc.duration_s)
    else:
        generated = make_schedule(
            duration_s=rc.duration_s,
            qps_base=rc.qps_base,
            qps_burst=rc.qps_burst,
            qps_min=rc.qps_min,
            qps_max=rc.qps_max,
            rate_scale=rc.rate_scale,
            seed=rc.seed + 16,
        )
    return [float(value) for value in generated["timestamps"]]


def _messages_input_upper_bound(endpoint, messages,
                                max_output: int) -> int | None:
    """Bound input from exact submitted JSON plus provider-owned framing."""
    if not isinstance(messages, list) or not messages:
        return None
    for message in messages:
        if not isinstance(message, dict):
            return None
    from .client import serialize_request_body

    body = serialize_request_body(
        endpoint, messages, max_output, endpoint.include_usage)
    return len(body) + _CHAT_FRAMING_TOKEN_ALLOWANCE * (len(messages) + 1)


def _prior_prompt_usage(row: dict) -> int | None:
    """Return only explicit, protocol-clean provider input usage.

    Prior rows describe setup POSTs which already happened. Their intended
    profile size is never a safe retrospective substitute, and a partial or
    malformed stream cannot be promoted to trusted token evidence merely
    because it happened to contain a usage-shaped object.
    """
    if row.get("status") != 200 or row.get("ok") is not True \
            or row.get("stream_complete") is not True:
        return None
    parse_errors = row.get("parse_errors")
    if not isinstance(parse_errors, int) or isinstance(parse_errors, bool) \
            or parse_errors != 0:
        return None
    prompt = row.get("prompt_tokens")
    completion = row.get("completion_tokens")
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt <= 0:
        return None
    if not isinstance(completion, int) or isinstance(completion, bool) \
            or completion < 0:
        return None
    cached = row.get("cached_tokens")
    if cached is not None and (
            not isinstance(cached, int) or isinstance(cached, bool)
            or cached < 0 or cached > prompt):
        return None
    reasoning = row.get("reasoning_tokens")
    if reasoning is not None and (
            not isinstance(reasoning, int) or isinstance(reasoning, bool)
            or reasoning < 0 or reasoning > completion):
        return None
    return prompt


def _prior_attempt_count(row: dict) -> int | None:
    """Validate the minimum cross-field evidence needed for POST counting."""
    attempts = row.get("request_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) \
            or attempts < 0:
        return None
    connections = row.get("connection_attempts")
    if not isinstance(connections, int) or isinstance(connections, bool) \
            or connections < attempts:
        return None
    retries = row.get("retries")
    retry_reasons = row.get("retry_reasons")
    if not isinstance(retries, int) or isinstance(retries, bool) \
            or retries < 0 or not isinstance(retry_reasons, list) \
            or any(not isinstance(reason, str) or not reason
                   for reason in retry_reasons) \
            or retries != len(retry_reasons):
        return None
    if attempts == 0:
        sent_evidence = (
            row.get("first_send_unix"), row.get("t_send_unix"),
            row.get("status"), row.get("prompt_tokens"),
            row.get("completion_tokens"), row.get("cached_tokens"),
            row.get("reasoning_tokens"),
            True if row.get("ok") is True else None,
            True if row.get("stream_complete") is True else None,
        )
        return 0 if all(value is None for value in sent_evidence) else None
    first_send = row.get("first_send_unix")
    if isinstance(first_send, bool) or not isinstance(first_send, (int, float)) \
            or not math.isfinite(float(first_send)) or first_send < 0:
        return None
    last_send = row.get("t_send_unix")
    if isinstance(last_send, bool) or not isinstance(last_send, (int, float)) \
            or not math.isfinite(float(last_send)) \
            or last_send < float(first_send):
        return None
    return attempts


def _plan_values(endpoint, plan: dict) -> tuple[int | None, int]:
    output_tokens = plan.get("max_output")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) \
            or output_tokens <= 0:
        raise ValueError("planned request needs a positive max_output")
    plan_endpoint = endpoint
    quota_extra_body = plan.get("_quota_extra_body")
    if quota_extra_body is not None:
        plan_endpoint = dataclasses.replace(
            endpoint, extra_body=copy.deepcopy(quota_extra_body))
    input_tokens = _messages_input_upper_bound(
        plan_endpoint, plan.get("messages"), output_tokens)
    return input_tokens, output_tokens


def _workload_values(endpoint, workload, index: int, *, post_calibration: bool) \
        -> tuple[int | None, int]:
    """Return a request's conservative input bound and output reservation."""
    if workload.prompts_mode:
        plan = workload.plan(index, f"quota-plan-prompt-{index}")
        return _plan_values(endpoint, plan)
    input_tokens = int(workload.draw["input_tokens"][index])
    cpt_ceiling = float(workload.rc.cpt)
    if post_calibration:
        cpt_ceiling = max(cpt_ceiling, _CALIBRATED_CPT_HARD_MAX)
    # Synthetic materialization is ASCII and targets round(tokens * cpt)
    # content characters.  ``ceil`` is a monotonic upper bound for that round,
    # and ASCII characters are one UTF-8 byte each.
    # TextMaterializer emits at most one system and one user message.
    output_tokens = min(
        int(workload.draw["output_tokens"][index]),
        int(workload.rc.max_output_tokens_cap),
    )
    # TextMaterializer emits at most one system and one user message. Serialize
    # that worst message shape with empty content using the exact client body
    # builder, then add the conservative ASCII content-byte ceiling. This
    # includes model, extra_body (including tools/schema), request controls,
    # roles, keys, and JSON syntax without materializing every large prompt.
    empty_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    serialized_empty = _messages_input_upper_bound(
        endpoint, empty_messages, output_tokens)
    if serialized_empty is None:  # pragma: no cover - fixed local shape
        return None, output_tokens
    content_chars = int(math.ceil(input_tokens * cpt_ceiling))
    input_upper_bound = (
        content_chars
        + _synthetic_json_escape_overhead(content_chars)
        + serialized_empty)
    return input_upper_bound, output_tokens


def _logical_events(rc: "RunConfig", *, offset_s: float = 0.0,
                    setup_plans: Iterable[dict] = (),
                    prior_rows: Iterable[dict] = (),
                    prevalidated=None) \
        -> tuple[list[dict], list[str], int]:
    """Return weighted attempt events and any unplannable input evidence.

    CLI callers can supply the exact endpoint-free validation result used by
    their paid preflight.  Reusing its schedule and sampled workload prevents
    the quota gate from silently planning a second read or a second random
    realization of the inputs it is about to authorize.
    """
    from .client import EndpointConfig
    from .runner import _PreparedWorkload

    multiplier = _attempt_multiplier(rc)
    endpoint = EndpointConfig(**rc.endpoint)
    if prevalidated is not None:
        if prevalidated.rc != rc:
            raise ValueError(
                "prevalidated quota inputs do not match the run configuration")
        if prevalidated.full_schedule is None:
            raise ValueError(
                "prevalidated quota inputs have no fixed schedule")
        timestamps = [
            float(value)
            for value in prevalidated.full_schedule["timestamps"]
        ]
        workload = prevalidated.workload
        if workload is None or workload.total_n != len(timestamps):
            raise ValueError(
                "prevalidated quota workload does not match its schedule")
    else:
        timestamps = _schedule(rc)
        workload = _PreparedWorkload(rc, len(timestamps)) if timestamps else None
    if not timestamps:
        raise ValueError(
            "schedule produced zero arrivals; increase the fixed rate or "
            "duration")
    events: list[dict] = []
    unknowns: list[str] = []

    def add(timestamp: float, input_tokens: int | None,
            output_tokens: int, attempts: int, phase: str) -> None:
        if input_tokens is None:
            unknowns.append(
                f"{phase} input tokens are unknown without provider usage")
        events.append({
            "t": float(timestamp),
            "input_tokens": (None if input_tokens is None
                             else int(input_tokens) * attempts),
            "output_tokens": int(output_tokens) * attempts,
            "queries": attempts,
            "logical_requests": 1,
            "phase": phase,
        })

    # Setup requests have no reliable duration before they are sent. Packing
    # them at the first replay instant is conservative for every rolling
    # window and prevents cooldown language from manufacturing headroom.
    for plan in setup_plans:
        inp, out = _plan_values(endpoint, plan)
        add(offset_s, inp, out, multiplier, "preflight_or_probe")

    for position, row in enumerate(prior_rows):
        if not isinstance(row, dict):
            unknowns.append(f"prior request row {position} is not an object")
            continue
        attempts = _prior_attempt_count(row)
        if attempts is None:
            unknowns.append(
                f"prior request row {position} has unknown provider attempts "
                "because its send evidence is incomplete or inconsistent")
            continue
        if attempts == 0:
            continue
        # Only provider-reported input usage can retrospectively replace the
        # pre-traffic byte upper bound.  Falling back to an intended profile
        # target here would make an already-observed request less conservative.
        inp = _prior_prompt_usage(row)
        out = row.get("max_tokens_requested")
        if not isinstance(out, int) or isinstance(out, bool) or out <= 0:
            unknowns.append(
                f"prior request row {position} has unknown max_tokens")
            continue
        add(offset_s, inp, out, attempts, "prior_request")

    # Every shard performs its own calibration pass.  All shards can start
    # together, so place those requests at the same conservative instant.
    calibration_n = min(int(rc.calibrate_n), len(timestamps))
    for index in range(calibration_n):
        inp, out = _workload_values(
            endpoint, workload, index, post_calibration=False)
        for _shard in range(int(rc.shard_total)):
            add(offset_s, inp, out, multiplier, "calibration")

    # A workspace quota is shared by the shards.  Plan the complete unsharded
    # schedule on every shard rather than blessing each fragment in isolation.
    for index, timestamp in enumerate(timestamps):
        inp, out = _workload_values(
            endpoint, workload, index, post_calibration=True)
        add(offset_s + timestamp, inp, out, multiplier, "replay")
    return events, sorted(set(unknowns)), len(timestamps)


def _setup_events(rc: "RunConfig", plans: Iterable[dict], *,
                  offset_s: float = 0.0) -> tuple[list[dict], list[str]]:
    """Plan only CLI preflight/probe traffic without building a run twice."""
    from .client import EndpointConfig

    multiplier = _attempt_multiplier(rc)
    endpoint = EndpointConfig(**rc.endpoint)
    events = []
    unknowns = []
    for plan in plans:
        inp, out = _plan_values(endpoint, plan)
        if inp is None:
            unknowns.append(
                "preflight_or_probe input tokens are unknown without "
                "provider usage")
        events.append({
            "t": float(offset_s),
            "input_tokens": None if inp is None else inp * multiplier,
            "output_tokens": out * multiplier,
            "queries": multiplier,
            "logical_requests": 1,
            "phase": "preflight_or_probe",
        })
    return events, sorted(set(unknowns))


def _rolling_peak(events: list[dict], field: str, window_s: float) \
        -> int | None:
    if any(event[field] is None for event in events):
        return None
    ordered = sorted(events, key=lambda event: event["t"])
    active: deque[dict] = deque()
    total = 0
    peak = 0
    for event in ordered:
        timestamp = event["t"]
        # The provider does not publish whether an event exactly on the
        # rolling-window boundary is included.  Keep it in the local budget
        # rather than manufacturing headroom from a boundary convention that
        # this harness cannot prove.  It falls out only once it is strictly
        # older than the configured window.
        while active and timestamp - active[0]["t"] > window_s:
            total -= int(active.popleft()[field])
        active.append(event)
        total += int(event[field])
        peak = max(peak, total)
    return peak


def _evaluate(events: list[dict], rate_limits: dict, *,
              unknowns: Iterable[str], logical_replay_requests: int,
              attempt_multiplier: int, plan_kind: str,
              planned_rungs: list[float] | None = None) -> dict:
    warning = float(rate_limits["warning_utilization"])
    fields = (
        ("input_tokens_per_minute", "input_tokens", 60.0,
         "complete serialized JSON byte upper bound plus chat framing"),
        ("output_tokens_per_minute", "output_tokens", 60.0,
         "offered max_tokens reservations"),
        ("queries_per_hour", "queries", 3600.0,
         "worst-case physical POST attempts"),
    )
    windows = {}
    refusal_reasons: list[str] = []
    freshness = _snapshot_freshness(rate_limits)
    if not freshness["fresh"]:
        status = freshness["status"]
        if status == "stale":
            refusal_reasons.append(
                "rate-limit snapshot is stale: verified "
                f"{freshness['verified_at']} ({freshness['age_days']} days "
                f"old), exceeding max_age_days={freshness['max_age_days']}; "
                "recheck the cited provider source before paid traffic")
        elif status == "missing":
            refusal_reasons.append(
                "rate-limit snapshot has no verified_at/max_age_days "
                "freshness proof; recheck the cited provider source before "
                "paid traffic")
        else:
            refusal_reasons.append(
                "rate-limit snapshot freshness metadata is invalid; recheck "
                "the cited provider source before paid traffic")
    unknown_list = sorted(set(str(item) for item in unknowns))
    for limit_name, event_field, seconds, evidence in fields:
        if limit_name not in rate_limits:
            continue
        limit = float(rate_limits[limit_name])
        peak = _rolling_peak(events, event_field, seconds)
        ratio = None if peak is None else float(peak) / limit
        entry = {
            "window_seconds": seconds,
            "planned_peak": peak,
            "configured_limit": limit,
            "ratio_to_configured_limit": ratio,
            "warning_ratio": warning,
            "evidence": evidence,
        }
        windows[limit_name] = entry
        if peak is None:
            refusal_reasons.append(
                f"{limit_name} cannot be bounded from the configured input")
        elif ratio >= warning:
            refusal_reasons.append(
                f"planned {limit_name} peak {peak:,} is {ratio:.1%} of "
                f"the configured {limit:g} limit, at or above the "
                f"{warning:.1%} warning budget")

    for unknown in unknown_list:
        if "unknown provider attempts" in unknown \
                or "is not an object" in unknown:
            refusal_reasons.append(
                "provider-attempt count is unknown, so planned query and "
                "token demand cannot be bounded")
        elif "unknown max_tokens" in unknown \
                and "output_tokens_per_minute" in rate_limits:
            refusal_reasons.append(
                "offered output reservation is unknown, so planned output "
                "token demand cannot be bounded")
    refusal_reasons = list(dict.fromkeys(refusal_reasons))

    if any("input tokens are unknown" in item for item in unknown_list) \
            and "input_tokens_per_minute" not in rate_limits:
        # Keep the limitation visible, but it is not a blocker when no input
        # token policy was supplied.
        pass
    plan = {
        "schema_version": 1,
        "plan_kind": plan_kind,
        "status": ("refused" if refusal_reasons
                   else "within_configured_harness_warning_budget"),
        "may_start": not refusal_reasons,
        "provider_headroom_proven": False,
        "workspace_external_traffic_included": False,
        "rate_limit_snapshot_freshness": freshness,
        "logical_replay_requests": int(logical_replay_requests),
        "planned_physical_attempts_worst_case": (
            None if any("unknown provider attempts" in item
                        for item in unknown_list)
            else int(sum(event["queries"] for event in events))),
        "physical_attempts_per_logical_worst_case": attempt_multiplier,
        "planned_rungs_requests_per_second": planned_rungs,
        "windows": windows,
        "unknowns": unknown_list,
        "refusal_reasons": refusal_reasons,
        "assumptions": [
            "Harness traffic is evaluated in isolation; unrelated workspace "
            "traffic can consume the same limits.",
            "Input demand is bounded at one token per UTF-8 byte of the "
            "complete serialized request JSON plus "
            f"{_CHAT_FRAMING_TOKEN_ALLOWANCE} tokens of chat framing for each "
            "message and one additional request-level block; roles, message "
            "metadata, model, tools, provider controls, and JSON syntax are "
            "included. This is intentionally stricter than intended profile "
            "token targets.",
            "Synthetic replay content is planned using the larger of its "
            "configured chars-per-token value and the calibrated hard maximum "
            f"of {_CALIBRATED_CPT_HARD_MAX:g}.",
            "Output uses max_tokens offered at admission, not eventual output "
            "consumption.",
            "Physical-attempt planning includes configured transport retries, "
            "one stream-options fallback, and one credential-refresh retry "
            "per logical request.",
            "Setup traffic is packed against the first replay window; spacing "
            "is not treated as proof of quota reset.",
        ],
        "rate_limits": copy.deepcopy(rate_limits),
    }
    return plan


def plan_run_quota(rc: "RunConfig", *,
                   setup_plans: Iterable[dict] = (),
                   prior_rows: Iterable[dict] = (),
                   prevalidated=None) -> dict | None:
    """Plan one run without making any network call."""
    if rc.rate_limits is None:
        return None
    if rc.sizing_concurrency is not None:
        plan = _evaluate(
            [], rc.rate_limits,
            unknowns=[
                "sizing_concurrency derives the replay rate from paid endpoint "
                "traffic, so the schedule is unknowable before traffic starts"
            ],
            logical_replay_requests=0,
            attempt_multiplier=_attempt_multiplier(rc),
            plan_kind="run",
        )
        plan["refusal_reasons"].append(
            "quota-aware runs require a fixed rate or timestamp trace")
        plan["may_start"] = False
        plan["status"] = "refused"
        return plan
    events, unknowns, replay_n = _logical_events(
        rc, setup_plans=setup_plans, prior_rows=prior_rows,
        prevalidated=prevalidated)
    return _evaluate(
        events, rc.rate_limits, unknowns=unknowns,
        logical_replay_requests=replay_n,
        attempt_multiplier=_attempt_multiplier(rc), plan_kind="run")


def plan_sweep_quota(base_config: dict, rates: Iterable[float], *,
                     duration_s: int, cooldown_s: float,
                     setup_plans: Iterable[dict] = (),
                     prevalidated_rungs: Iterable | None = None) -> dict | None:
    """Plan the union of every requested sweep rung before preflight traffic."""
    from .runner import RunConfig

    base = copy.deepcopy(base_config)
    if base.get("rate_limits") is None:
        return None
    rate_values = [float(rate) for rate in rates]
    if not rate_values:
        raise ValueError("quota planner needs at least one sweep rung")
    validated = (None if prevalidated_rungs is None
                 else list(prevalidated_rungs))
    if validated is not None and len(validated) != len(rate_values):
        raise ValueError(
            "prevalidated sweep inputs must match every requested rung")
    setup = list(setup_plans)
    all_events: list[dict] = []
    unknowns: list[str] = []
    replay_total = 0
    offset = 0.0

    # Setup occurs once for the whole sweep. Use a valid first-rung config to
    # determine the endpoint retry contract and request budgets.
    if validated is not None:
        first_rc = validated[0].rc
    else:
        first_cfg = copy.deepcopy(base)
        first_cfg.update(
            qps_base=rate_values[0], qps_burst=rate_values[0],
            qps_min=rate_values[0], qps_max=rate_values[0], rate_scale=1.0,
            duration_s=duration_s,
        )
        first_rc = RunConfig(**first_cfg)
    if setup:
        # Cooldown is operational spacing, not proof that a token bucket or
        # provider accounting window reset. Pack setup against the first rung
        # even when the command sleeps between them.
        offset = float(cooldown_s)
        setup_events, setup_unknowns = _setup_events(
            first_rc, setup, offset_s=offset)
        all_events.extend(setup_events)
        unknowns.extend(setup_unknowns)

    for position, rate in enumerate(rate_values):
        if validated is not None:
            checked = validated[position]
            rc = checked.rc
            if rc.sizing_concurrency is not None \
                    or rc.timestamps_file is not None \
                    or rc.duration_s != duration_s \
                    or float(rc.rate_scale) != 1.0 \
                    or any(float(value) != rate for value in (
                        rc.qps_base, rc.qps_burst, rc.qps_min, rc.qps_max)):
                raise ValueError(
                    f"prevalidated sweep rung {position} does not match "
                    f"{rate:g} requests/second for {duration_s}s")
        else:
            checked = None
            cfg = copy.deepcopy(base)
            cfg.update(
                qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
                rate_scale=1.0, duration_s=duration_s,
                out_dir=f"quota-plan-rate-{position}",
                title=f"quota plan at {rate:g} requests/second",
            )
            rc = RunConfig(**cfg)
        events, rung_unknowns, replay_n = _logical_events(
            rc, offset_s=offset, prevalidated=checked)
        all_events.extend(events)
        unknowns.extend(rung_unknowns)
        replay_total += replay_n
        offset += float(duration_s)
        if position < len(rate_values) - 1:
            offset += float(cooldown_s)

    return _evaluate(
        all_events, first_rc.rate_limits, unknowns=unknowns,
        logical_replay_requests=replay_total,
        attempt_multiplier=_attempt_multiplier(first_rc), plan_kind="sweep",
        planned_rungs=rate_values)


def bind_quota_plan_to_endpoint(plan: dict | None, binding: dict) \
        -> dict | None:
    """Attach control-plane binding evidence and make the final gate decision.

    Offline schedule safety and endpoint identity are independent facts.  A
    plan may reach this function only after the harness-only windows pass; it
    may start paid inference only when both facts are true.  A fresh copy is
    returned so callers cannot accidentally mutate a plan already written to
    an audit record.
    """
    if plan is None:
        return None
    bound = copy.deepcopy(plan)
    schedule_may_start = bool(
        bound.get("schedule_may_start", bound.get("may_start")))
    schedule_reasons = list(bound.get("schedule_refusal_reasons",
                                      bound.get("refusal_reasons") or []))
    binding_copy = copy.deepcopy(binding)
    binding_complete = binding_copy.get("binding_complete") is True
    binding_reasons = [
        f"endpoint binding: {reason}"
        for reason in (binding_copy.get("reasons") or [])
    ]
    if not binding_complete and not binding_reasons:
        binding_reasons = ["endpoint binding: verification was incomplete"]

    bound.update(
        schedule_may_start=schedule_may_start,
        schedule_refusal_reasons=schedule_reasons,
        endpoint_binding_required=True,
        endpoint_binding=binding_copy,
        may_start=bool(schedule_may_start and binding_complete),
    )
    bound["refusal_reasons"] = list(dict.fromkeys(
        schedule_reasons + ([] if binding_complete else binding_reasons)))
    if bound["may_start"]:
        bound["status"] = "ready_for_paid_inference"
        bound["refusal_stage"] = None
    else:
        bound["status"] = "refused"
        bound["refusal_stage"] = (
            "endpoint_binding" if schedule_may_start else "schedule")
    return bound


def enforce_quota_plan(plan: dict | None) -> None:
    if plan is not None and not plan.get("may_start"):
        raise QuotaPlanError(plan)


def render_quota_plan(plan: dict | None) -> str:
    """Short terminal rendering; the complete plan remains machine-readable."""
    if plan is None:
        return ""
    binding = plan.get("endpoint_binding")
    binding_blocked = bool(
        isinstance(binding, dict)
        and not binding.get("binding_complete")
        and plan.get("schedule_may_start"))
    if binding_blocked:
        headline = (
            "REFUSED: harness schedule is within its configured warning "
            "budget, but endpoint binding blocked paid inference")
    else:
        headline = (
            ("PASS" if plan["may_start"] else "REFUSED")
            + ": harness-only worst-case schedule; provider headroom is not "
              "proven")
    lines = ["[quota-plan] " + headline]
    freshness = plan.get("rate_limit_snapshot_freshness") or {}
    lines.append(
        "[quota-plan] rate-limit snapshot: "
        f"{str(freshness.get('status') or 'unknown').upper()}; "
        f"source as-of={freshness.get('source_as_of')}; "
        f"verified={freshness.get('verified_at')}; "
        f"age={freshness.get('age_days')} days; "
        f"max-age={freshness.get('max_age_days')} days; "
        f"checked={freshness.get('checked_on')}")
    for name, evidence in plan["windows"].items():
        peak = evidence["planned_peak"]
        ratio = evidence["ratio_to_configured_limit"]
        shown_peak = "unknown" if peak is None else f"{peak:,}"
        shown_ratio = "unknown" if ratio is None else f"{ratio:.1%}"
        lines.append(
            f"[quota-plan] {name}: {shown_peak} / "
            f"{evidence['configured_limit']:g} ({shown_ratio}); "
            f"warning at {evidence['warning_ratio']:.1%}")
    if isinstance(binding, dict):
        lines.append(
            "[quota-plan] endpoint binding: "
            + ("VERIFIED" if binding.get("binding_complete") else "REFUSED")
            + f"; configured={binding.get('configured_model')}; "
            f"observed={binding.get('observed_endpoint_name')}")
        lines.append(
            "[quota-plan] workspace tier "
            f"{binding.get('configured_workspace_tier')!r} is a configured "
            "assertion, not verified by endpoint metadata")
    for reason in plan["refusal_reasons"]:
        lines.append(f"[quota-plan] STOP: {reason}")
    lines.append(
        "[quota-plan] unrelated workspace traffic is not visible; passing "
        "this gate is not a provider-capacity claim")
    return "\n".join(lines)
