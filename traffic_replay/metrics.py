"""Summaries and the honesty block.

Every latency table is printed WITH the context that decides whether it can
be believed: cached prompt-token fraction (endpoint-reported), achieved
arrival rate vs scheduled, wire lateness, error rate, and token
targeting error. A good p50 at the wrong cached-token fraction is a fake
result; this
module makes the pairing unavoidable.
"""
from __future__ import annotations

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
    sanitize_title,
    sha256_bytes,
    snapshot_source_state,
    strict_json_dumps,
)

PCTS = (50, 90, 95, 99)


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
    if "first_send_unix" in r:
        return r.get("first_send_unix")
    return r.get("t_send_unix")


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
    if (s.get("cost") or {}).get("coverage_warning"):
        doubts.append("cost could not be computed for every successful "
                      "response because required usage fields were missing")
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


def _answer_block(results: list[dict]) -> dict | None:
    """Answer completion, separately from HTTP and content-stream success.

    ``ok`` is a harness field meaning that at least one visible or reasoning
    content delta arrived. It is not an HTTP-status counter. Keep the three
    populations separate so a reasoning-only HTTP 200 cannot be presented as
    a readable answer, and a content-bearing stream cannot be mislabeled as
    "returned HTTP 200" when status was not retained by a legacy row.
    """
    ok = [r for r in results if r.get("ok")]
    scored = [r for r in results
              if "visible_content_seen" in r or "valid_tool_calls" in r]
    legacy_failures = [r for r in results
                       if not r.get("ok")
                       and "visible_content_seen" not in r]
    if not scored and not legacy_failures:
        return None          # rows written before this was recorded
    n_observed = len(scored)
    complete = sum(1 for r in scored if _answered(r))
    judged = n_observed + len(legacy_failures)
    statuses = [r.get("status") for r in results if r.get("status") is not None]
    out = {
        "attempted": len(results),
        # Backward-compatible field name. Its definition is now explicit and
        # renderers never call it an HTTP counter.
        "transport_ok": len(ok),
        "content_streams": len(ok),
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
            f"produced visible content or a valid tool call. most of them "
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
              concurrency_target: int | None = None) -> dict:
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
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
    ach = [(r["cached_tokens"] / r["prompt_tokens"])
           for r in ok
           if r.get("cached_tokens") is not None
           and r.get("prompt_tokens")]
    cache_sources = sorted({r.get("cached_tokens_source") for r in ok
                            if r.get("cached_tokens_source")})
    intended_cache = [r.get("intended_cache_fraction") for r in results
                      if r.get("intended_cache_fraction") is not None]
    paired_cache_error = [
        abs((r["cached_tokens"] / r["prompt_tokens"])
            - r["intended_cache_fraction"])
        for r in ok
        if r.get("cached_tokens") is not None and r.get("prompt_tokens")
        and r.get("intended_cache_fraction") is not None]
    invalid_cache_rows = sum(
        1 for r in ok
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

    input_eligible = [r for r in ok
                      if positive_number(r.get("intended_input_tokens"))]
    input_pairs = [
        (float(r["prompt_tokens"]), float(r["intended_input_tokens"]))
        for r in input_eligible if positive_number(r.get("prompt_tokens"))]
    ratios = [actual / intended for actual, intended in input_pairs]
    input_errors_pct = [abs(ratio - 1.0) * 100.0 for ratio in ratios]

    output_eligible = [r for r in ok
                       if positive_number(r.get("intended_output_tokens"))]
    output_pairs = [
        (float(r["completion_tokens"]), float(r["intended_output_tokens"]))
        for r in output_eligible
        if nonnegative_number(r.get("completion_tokens"))]
    out_ratios = [actual / intended for actual, intended in output_pairs]
    output_errors_pct = [abs(ratio - 1.0) * 100.0 for ratio in out_ratios]
    targeting_warnings = []
    tolerance_pct = 10.0
    input_coverage = (len(input_pairs) / len(input_eligible)
                      if input_eligible else None)
    output_coverage = (len(output_pairs) / len(output_eligible)
                       if output_eligible else None)
    input_error_table = _pct_table(input_errors_pct)
    output_error_table = _pct_table(output_errors_pct)
    if input_eligible:
        if input_coverage is not None and input_coverage < 0.99:
            targeting_warnings.append(
                f"prompt-token usage was reported for only "
                f"{len(input_pairs)} of {len(input_eligible)} successful "
                "profile requests")
        elif ((input_error_table.get("p50") or 0.0) > tolerance_pct
              or (input_error_table.get("p95") or 0.0) > tolerance_pct):
            targeting_warnings.append(
                "endpoint-reported input tokens did not reproduce the "
                f"declared profile within ±{tolerance_pct:.0f}% "
                f"(absolute relative error p50 "
                f"{input_error_table['p50']:.1f}%, p95 "
                f"{input_error_table['p95']:.1f}%)")
    if output_eligible:
        if output_coverage is not None and output_coverage < 0.99:
            targeting_warnings.append(
                f"completion-token usage was reported for only "
                f"{len(output_pairs)} of {len(output_eligible)} successful "
                "profile requests")
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
    in_tok = sum(r["prompt_tokens"] for r in ok if r.get("prompt_tokens"))
    out_tok = sum(r["completion_tokens"] for r in ok
                  if r.get("completion_tokens"))
    cached_tok = sum(r["cached_tokens"] for r in ok if r.get("cached_tokens"))
    dur_min = (dur / 60.0) if dur else None
    # how many successful responses actually reported usage. a run where
    # only a tenth of them do would otherwise understate token throughput
    # and per-token cost tenfold with nothing said about it.
    usage_n = sum(1 for r in ok
                  if r.get("prompt_tokens") is not None
                  and r.get("completion_tokens") is not None)
    usage_coverage = (usage_n / len(ok)) if ok else None

    summary = {
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "requests_retried": retried,
        "error_rate": len(failed) / len(results) if results else None,
        "failures_by_error": _top_errors(failed),
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
                (None if usage_coverage is None or usage_coverage > 0.99 else
                 f"only {usage_n} of {len(ok)} successful responses reported "
                 "token usage, so these totals and any per-token cost below "
                 "cover that subset, not the run")),
        },
        "achieved_cache_fraction": _pct_table(ach) | {
            "reported_for_n": len(ach),
            "eligible_successes": len(ok),
            "coverage": (len(ach) / len(ok)) if ok else None,
            "source_fields": (cache_sources
                              or (["SOURCE FIELD NOT RECORDED"] if ach else
                                  ["NOT REPORTED BY ENDPOINT"])),
        },
        "intended_cache_fraction": _pct_table(intended_cache),
        "latency_population": latency_population,
        "token_targeting": {
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
            "status": ("not_applicable" if not input_eligible
                       and not output_eligible else
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
    for field in ("ttft_ms", "ttf_tool_call_ms"):
        values = [r.get(field) for r in latency_ok]
        summary[field]["missing"] = sum(v is None for v in values)
        summary[field]["of"] = len(values)
    if intended_cache:
        tolerance = 0.10
        err = _pct_table(paired_cache_error)
        coverage = (len(ach) / len(ok)) if ok else None
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
                f"cache usage was reported for only {len(ach)} of {len(ok)} "
                "content-bearing successful responses")
        summary["cache_fidelity"] = {
            "status": "verified" if not warnings else "unverified",
            "tolerance_abs": tolerance,
            "paired_n": len(paired_cache_error),
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
            # returns a successful response with no visible token at all.
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
            "queue wait. SLA latency targets and hard caps prefer these "
            "figures whenever available.")
        summary["latency_correction_provenance"] = {
            "exact_values": exact_caller_n,
            "legacy_reconstructed_values": reconstructed_caller_n,
        }
    reason_vals = [r.get("reasoning_tokens") for r in ok]
    if any(v is not None for v in reason_vals):
        total = sum(v for v in reason_vals if v)
        summary["reasoning_tokens"] = _pct_table(reason_vals)
        summary["reasoning_tokens_total"] = total
        summary["reasoning_tokens_source"] = next(
            (r.get("reasoning_tokens_source") for r in ok
             if r.get("reasoning_tokens_source")), None)
        if dur_min:
            summary["throughput"]["reasoning_tokens_per_min"] = total / dur_min
    if summary.get("reasoning_tokens_total") is None:
        # endpoint did not report a reasoning-token count (some models do
        # not). fall back to counting reasoning_content deltas in the stream,
        # clearly labeled as an estimate.
        chunk_vals = [r.get("reasoning_chunks") for r in ok]
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
        "ttft/ttfb/ttfg are timed from the moment the request bytes are sent "
        "on an already-established connection. TCP and TLS setup is measured "
        "separately as connect_ms and is NOT included. changed in 0.3.0: "
        "0.2.x and earlier included connection setup in these numbers.")

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
        summary["cost"] = _cost_block(ok, dur, in_tok, out_tok, cached_tok,
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
                     "drift. run for minutes to test sustained SLA."}
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
            "successful requests, when no request returned a first token, or "
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


def _cost_block(ok: list[dict], dur, in_tok: int, out_tok: int,
                cached_tok: int, pricing: dict) -> dict:
    """Cost from endpoint-reported tokens times user-supplied DBU rates.

    Rates come from the Databricks pricing page and are supplied in the run
    config, never fetched, so the report states the arithmetic and the numbers
    you gave it. Pay-per-token bills input, output, and cache-read separately
    (three DBU/M rates). Provisioned throughput bills capacity by the hour, so
    the useful figure is effective DBU per 1M tokens at the measured load.
    """
    mode = pricing.get("mode", "per_token")
    usd = pricing.get("usd_per_dbu")
    tok_total = in_tok + out_tok
    usage_rows = [r for r in ok
                  if r.get("prompt_tokens") is not None
                  and r.get("completion_tokens") is not None]
    usage_coverage = ((len(usage_rows) / len(ok)) if ok
                      else (1.0 if tok_total else None))

    if mode == "provisioned":
        dph = pricing.get("dbu_per_hour")
        if dph is None:
            return {"mode": mode, "error": "provisioned needs dbu_per_hour"}
        dur_hr = (dur / 3600.0) if dur else None
        tph = (tok_total / dur_hr) if dur_hr and usage_coverage == 1.0 else None
        eff = (dph / (tph / 1e6)) if tph else None
        block = {"mode": "provisioned", "dbu_per_hour": dph,
                 "effective_dbu_per_1m_tokens": eff,
                 "tokens_measured": tok_total,
                 "usage_coverage": usage_coverage,
                 "usage_rows": len(usage_rows),
                 "successful_rows": len(ok),
                 "coverage_warning": (
                     None if usage_coverage in (None, 1.0) else
                     f"token usage was reported for {len(usage_rows)} of "
                     f"{len(ok)} successful responses, so effective cost per "
                     "token is unavailable"),
                 "note": "provisioned throughput bills by capacity (DBU/hour), "
                         "not per token. effective cost per 1M tokens is the "
                         "hourly rate over tokens served per hour at the "
                         "measured throughput, so it improves as you fill the "
                         "endpoint. rates are user-supplied from the pricing "
                         "page."}
        if usd is not None:
            block["usd_per_hour"] = dph * usd
            if eff is not None:
                block["effective_usd_per_1m_tokens"] = eff * usd
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
    priced_rows = [
        r for r in usage_rows
        if ((r.get("cached_tokens") is None and cache == inp)
            or (r.get("cached_tokens") is not None
                and 0 <= r["cached_tokens"] <= r["prompt_tokens"]))]
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
    complete = n == len(ok)
    coverage = (n / len(ok)) if ok else None
    total = measured_total if complete else None
    block = {
        "mode": "per_token",
        "dbu_per_request": _pct_table(per),
        "priced_rows": n,
        "successful_rows": len(ok),
        "coverage": coverage,
        "complete": complete,
        "dbu_total_measured_subset": measured_total,
        "dbu_total": total,
        "dbu_per_1k_requests": ((total / n * 1000)
                                 if complete and n else None),
        "dbu_per_min": ((total / (dur / 60.0))
                         if complete and dur else None),
        "cache_dbu_saved": (measured_cached / 1e6
                            * max(inp - cache, 0.0)) if complete else None,
        "rates_dbu_per_m": {"input": inp, "output": out, "cache_read": cache},
        "coverage_warning": (
            None if complete or not ok else
            f"cost-required usage was present for {n} of {len(ok)} successful "
            "responses and passed token-accounting checks. aggregate cost, "
            "cost per 1,000 requests, cost per minute and cache savings are "
            "unavailable; the measured subset is retained only for "
            "diagnosis"),
        "note": "cost from endpoint-reported tokens times user-supplied DBU "
                "rates (Databricks pricing page). cached input is billed at "
                "the cache-read rate.",
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
                                                as SLA failures
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
            + ". a caller-experienced SLA cannot be proven from that coverage")

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
        key = (r.get("error") or "unknown")[:80]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:k])


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


def render_markdown(summary: dict, title: str) -> str:
    s = summary

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
        cautions += [f"CAUTION (network distance): {_nw}", ""]
    _cw = (s.get("throughput") or {}).get("coverage_warning")
    if _cw:
        cautions += [f"CAUTION (token usage): {_cw}", ""]
    _costw = (s.get("cost") or {}).get("coverage_warning")
    if _costw:
        cautions += [f"CAUTION (cost coverage): {_costw}", ""]
    _cachew = (s.get("cache_fidelity") or {}).get("warning")
    if _cachew:
        cautions += [f"CAUTION (cache fidelity): {_cachew}", ""]
    _tokenw = (s.get("token_targeting") or {}).get("warning")
    if _tokenw:
        cautions += [f"CAUTION (workload token fidelity): {_tokenw}", ""]
    _popw = (s.get("latency_population") or {}).get("warning")
    if _popw:
        cautions += [f"CAUTION (latency population): {_popw}", ""]
    _sw = (s.get("sample") or {}).get("warning")
    if _sw:
        cautions += [f"CAUTION (sample size): {_sw}", ""]
    _rw = (s.get("replay") or {}).get("warning")
    if _rw:
        cautions += [f"CAUTION (prompt replay): {_rw}", ""]
    _cw = (s.get("client") or {}).get("warning")
    if _cw:
        cautions += [f"CAUTION (client saturation): {_cw}", ""]
    _nw = (s.get("concurrency") or {}).get("warning")
    if _nw:
        cautions += [f"CAUTION (concurrency not reached): {_nw}", ""]

    lines = [
        f"# {title}",
        "",
        f"requests: {s['requests_total']} total, {s['requests_ok']} produced "
        f"a content delta, {s['requests_failed']} did not "
        f"(error rate {100 * (s['error_rate'] or 0):.2f}%)",
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
        lines += ["", f"throughput: {tp['input_tokens_per_min']:,.0f} input "
                      f"tokens/min, {tp['output_tokens_per_min']:,.0f} output "
                      "tokens/min (endpoint-reported counts over wall time)"]
    cost = s.get("cost")
    if cost and cost.get("error"):
        lines += ["", f"cost: config error, {cost['error']}"]
    elif cost and cost["mode"] == "per_token" and cost.get("coverage_warning"):
        lines += ["", "cost (per-token): unavailable for the full run. "
                  + cost["coverage_warning"]]
    elif cost and cost["mode"] == "per_token":
        dr = cost.get("dbu_per_request") or {}
        if dr.get("p50") is None:
            lines += ["", "cost: no successful requests to price"]
        else:
            usd = cost.get("usd_total")
            dollar = f" (${usd:,.4f} total)" if usd is not None else ""
            lines += ["", f"cost (per-token, user-supplied DBU rates): "
                      f"{dr['p50']:.4f} DBU/request p50, "
                      f"{cost['dbu_per_1k_requests']:,.2f} DBU/1k requests, "
                      f"{cost['dbu_per_min']:,.3f} DBU/min, cache saved "
                      f"{cost['cache_dbu_saved']:,.3f} DBU{dollar}"]
    elif cost:
        eff = cost.get("effective_dbu_per_1m_tokens")
        lines += ["", f"cost (provisioned, {cost['dbu_per_hour']} DBU/hour): "
                  + (f"effective {eff:,.1f} DBU per 1M tokens at the measured "
                     f"throughput" if eff is not None
                     else "throughput too low to compute an effective rate")]
    rp = (s.get("run") or {}).get("request_params")
    if rp:
        eb = rp.get("extra_body") or {}
        line = (f"request params: temperature {rp.get('temperature')}, "
                f"max_tokens cap {rp.get('max_output_tokens_cap')}")
        if eb:
            line += f", extra_body {json.dumps(eb)}"
        lines += ["", line]
    merge_note = (s.get("run") or {}).get("merge_note")
    if merge_note:
        lines += ["", merge_note]

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
                  f"- produced at least one content delta: "
                  f"{a.get('content_streams', a['transport_ok'])}"]
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
                  "", a["note"]]
        lines += answer_lines
        if a.get("invalid"):
            lines += ["", f"INVALID: {a['invalid']}"]

    sla = s.get("sla")
    if sla:
        _tgt_src = sla.get("targets_source") or "the run configuration"
        _basis = (sla.get("latency_basis") or "unknown").replace("_", " ")
        lines += ["", f"## SLA scorecard (targets from {_tgt_src}; "
                  f"latency basis: {_basis})"]
        if sla.get("targets_warning"):
            lines += ["", f"CAUTION (targets): {sla['targets_warning']}"]
        if sla.get("coverage_warning"):
            lines += ["", f"CAUTION (coverage): {sla['coverage_warning']}"]
        if sla.get("caller_latency_warning"):
            lines += ["", f"CAUTION (caller timing): "
                      f"{sla['caller_latency_warning']}"]
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
        lines.append(f"| hard timeout breaches | - | - | "
                     f"{sla.get('hard_timeout_breaches', 0)} | "
                     f"{'yes' if not sla.get('hard_timeout_breaches') else 'NO'} |")
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
            vis = (f"ttfv (first visible token) p50 {tfv:.0f} ms, but over "
                   f"only the {_of - _miss} of {_of} requests that produced "
                   "visible content. the rest ran out of output tokens still "
                   "reasoning, so that p50 is the fastest subset, not the run")
        else:
            vis = f"ttfv (first visible token) p50 {tfv:.0f} ms"
        lines += ["", "note: reasoning model detected. ttft (first token of "
                  f"either kind) p50 {tft:.0f} ms. {vis}. agree which "
                  "definition the SLA scores via ttft_definition in the run "
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
        lines += ["", f"stability over time ({flag})."
                  f"{sp} {drift.get('drift_headline') or drift.get('note', '')}"]
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
            lines.append(f"note: {drift.get('note', '')}")
    elif drift.get("note"):
        lines += ["", f"stability over time: {drift['note']}"]

    em = (s.get("run") or {}).get("endpoint_metadata")
    if em:
        se = em.get("served_entities") or []
        detail = (", ".join(f"{k}={v}" for k, v in se[0].items() if k != "name")
                  if se else "")
        _task = f"task {em.get('task')}, " if em.get("task") else ""
        lines += ["", f"endpoint under test: {em.get('name')}, {_task}"
                  f"route_optimized {em.get('route_optimized')}, "
                  f"ready {em.get('ready')}" + (f", {detail}" if detail else "")]

    run_meta = s.get("run") or {}
    if run_meta.get("label"):
        lines += ["", f"**Label: {run_meta['label']}**"]
    if run_meta.get("profile_label"):
        lines += ["", f"**Profile: {run_meta['profile_label']}**"]
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
:root{--blue:#1971c2;--green:#2f9e44;--red:#e03131;--amber:#e8590c;--gray:#495057}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,
 sans-serif;color:#1e1e1e;background:#f4f6f8;margin:0;padding:24px;line-height:1.45}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:23px;margin:0 0 4px}
.sub{color:#6b7280;font-size:13px;margin-bottom:6px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;
 margin:14px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card h2{font-size:13px;margin:0 0 4px;color:var(--blue);text-transform:uppercase;
 letter-spacing:.04em}
.cap{font-size:12px;color:#6b7280;margin:0 0 12px}
.slanote{background:#eef6fc;border:1px solid #cfe2f5;border-radius:8px;
 padding:10px 14px;font-size:12px;color:#1c4f77;margin-top:12px;line-height:1.5}
.slanote code{background:#dcecf7;padding:1px 4px;border-radius:3px}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.stat{flex:1 1 150px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;
 padding:14px 16px}
.stat .k{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:25px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.stat .u{font-size:12px;color:#9aa0a6;font-weight:400}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #eef0f2;font-size:13px}
th{color:#6b7280;font-weight:600;font-size:11px;text-transform:uppercase}
td.lbl,th.lbl{text-align:left;font-weight:600}
td.n{color:#9aa0a6}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
 font-weight:700}
.ok{background:#ebfbee;color:var(--green)}
.bad{background:#fff5f5;color:var(--red)}
.neutral{background:#f1f3f5;color:var(--gray)}
.banner{border-radius:12px;padding:14px 18px;margin:14px 0;font-weight:600;font-size:15px}
.banner.ok{background:#ebfbee;color:#1b7a34;border:1px solid #b2f2bb}
.banner.bad{background:#fff5f5;color:#c92a2a;border:1px solid #ffc9c9}
.banner.warn{background:#fff4e6;color:#b34700;border:1px solid #ffd8a8}
.believe{border-left:4px solid var(--amber)}
.believe ul{margin:0;padding-left:18px}
.believe li{margin:7px 0;font-size:13px;color:#3b4148}
.believe b{color:#1e1e1e}
.label-note{background:#fff9db;border:1px solid #ffe066;border-radius:10px;
 padding:12px 16px;font-size:13px;color:#7a5c00;margin:14px 0}
.foot{color:#9aa0a6;font-size:12px;margin-top:18px;text-align:center}
td.yes{color:var(--green);font-weight:700}
td.no{background:#fff5f5;color:var(--red);font-weight:700}
td.na{color:#c0c4c9}
</style>"""


def _html_stat(k, v, u=""):
    unit = f" <span class='u'>{html.escape(u)}</span>" if u else ""
    return (f"<div class='stat'><div class='k'>{html.escape(k)}</div>"
            f"<div class='v'>{v}{unit}</div></div>")


def render_html(summary: dict, title: str) -> str:
    """A self-contained, styled HTML report built from the same summary the
    markdown uses. Stdlib only, no external assets, safe to open in a browser
    or attach to a deck."""
    s = summary
    esc = html.escape
    run = s.get("run") or {}
    mode = run.get("input_mode", "profile")

    def num(v, nd=0):
        return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "n/a"

    def has(t):
        return bool(t) and t.get("n", 0) > 0

    # ---- header ----
    ep = esc(run.get("endpoint_path") or "")
    src = ("real prompts" if mode == "prompts" else "synthetic shape")
    total = s.get("requests_total") or 0
    okc = s.get("requests_ok") or 0
    failed = s.get("requests_failed") or 0
    err = (s.get("error_rate") or 0) * 100
    sub = (f"{ep} &middot; {src} &middot; {total} requests, {okc} produced a "
           f"content delta, {failed} did not")

    # ---- stat cards ----
    cards = []
    ttft = s.get("ttft_ms") or {}
    if has(ttft):
        cards.append(_html_stat("TTFT p50", num(ttft["p50"]), "ms"))
        cards.append(_html_stat("TTFT p95", num(ttft["p95"]), "ms"))
    e2e = s.get("e2e_ms") or {}
    if has(e2e):
        cards.append(_html_stat("End to end p95", num(e2e["p95"]), "ms"))
    err_cls = "ok" if failed == 0 else "bad"
    cards.append(f"<div class='stat'><div class='k'>error rate</div>"
                 f"<div class='v'><span class='pill {err_cls}'>"
                 f"{err:.2f}%</span></div></div>")
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
    if tp.get("output_tokens_per_min"):
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
                    f"<tr><td class='lbl'>{name} {esc(r['quantile'])} (ms)</td>"
                    f"<td>{num(r['target_ms'])}</td>"
                    f"<td>{num(r['actual_ms']) if r['actual_ms'] is not None else '-'}</td>"
                    f"<td class='{cls}'>{cell}</td></tr>")
        ht = sla.get("hard_timeout_breaches")
        if ht is not None:
            cls = "yes" if ht == 0 else "no"
            rows.append(f"<tr><td class='lbl'>hard timeout breaches (count)</td>"
                        f"<td>-</td><td>{ht}</td>"
                        f"<td class='{cls}'>{'PASS' if ht == 0 else ht}</td></tr>")
            if ht:
                misses += 1
        ib = sla.get("interchunk_breaches")
        if ib is not None:
            cls = "yes" if ib == 0 else "no"
            rows.append(f"<tr><td class='lbl'>interchunk breaches (count)</td>"
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
                f"<tr><td class='lbl'>success rate (fraction 0-1)</td>"
                f"<td>{num(sr['target'], 4)}</td><td>{num(sr['actual'], 4)}</td>"
                f"<td class='{cls}'>{'PASS' if met else 'NO'}</td></tr>")
            lower = sr.get("one_sided_95pct_wilson_lower")
            demonstrated = sr.get("statistically_demonstrated")
            if lower is not None:
                confidence_cls = "yes" if demonstrated else "no"
                rows.append(
                    "<tr><td class='lbl'>success-rate one-sided 95% Wilson "
                    "lower bound</td>"
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
                   f" Raise {_knob} so requests reach that token."
                   " On a reasoning-only model no budget may be enough, and"
                   " the mode is the decision rather than the budget.")
            note_bits.append(
                f"TTFT actual is <b>-</b> because it is scored on "
                f"<b>{defn}</b> and no request emitted that token within "
                f"max_tokens (a reasoning model can spend the whole token "
                f"budget thinking).{fix} The latency table below still shows "
                f"TTFT for the first token of any kind.")
        if s.get("ttfr_ms"):
            tft = (s.get("ttft_ms") or {}).get("p50")
            note_bits.append(
                f"Reasoning model detected: TTFT (first token of any kind) "
                f"p50 {num(tft)} ms arrives before the first visible token.")
        slanote = (f"<div class='slanote'>{' '.join(note_bits)}</div>"
                   if note_bits else "")
        basis = esc((sla.get("latency_basis") or "unknown").replace("_", " "))
        sla_html = (
            f"<div class='card'><h2>SLA scorecard "
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
            + "<table>"
            f"<tr><th class='lbl'>metric</th><th>target</th><th>actual</th>"
            f"<th>result</th></tr>{''.join(rows)}</table>{slanote}</div>")

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
    for label, key in (("TTFT (first token)", "ttft_ms"),
                       ("TTF valid tool call", "ttf_tool_call_ms"),
                       ("TTFB (first byte)", "ttfb_ms"),
                       ("TTFG (end to end)", "e2e_ms"),
                       ("interchunk max", "interchunk_max_ms"),
                       ("TTFR (first reasoning)", "ttfr_ms"),
                       ("TTFV (first visible)", "ttfv_ms")):
        t = s.get(key)
        if has(t):
            lat.append(
                f"<tr><td class='lbl'>{label}</td><td>{num(t['p50'])}</td>"
                f"<td>{num(t['p90'])}</td><td>{num(t['p95'])}</td>"
                f"<td>{num(t['p99'])}</td><td class='n'>{t['n']}</td></tr>")
    pop_note = esc((s.get("latency_population") or {}).get("note")
                   or "latency population was not recorded")
    lat_html = (
        "<div class='card'><h2>Endpoint service latency (milliseconds, from send)</h2>"
        f"<div class='cap'>{pop_note}. p50 to p99 are percentiles across that "
        "population, lower is better. n is the measured count; all values are "
        "in ms.</div><table>"
        "<tr><th class='lbl'>metric</th><th>p50</th><th>p90</th><th>p95</th>"
        f"<th>p99</th><th>n</th></tr>{''.join(lat)}</table></div>")

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
    else:
        bel.append("<li><b>Failures</b>: none</li>")
    rp = run.get("request_params")
    if rp:
        eb = rp.get("extra_body") or {}
        extra = f", extra_body {esc(json.dumps(eb))}" if eb else ""
        bel.append(f"<li><b>Request params</b>: temperature "
                   f"{esc(str(rp.get('temperature')))}, max_tokens cap "
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

    believe = (
        "<div class='card believe'><h2>Believability "
        "(read before quoting a number)</h2>"
        f"<ul>{''.join(bel)}</ul></div>")

    # ---- throughput + merge note ----
    extra_cards = ""
    if tp.get("input_tokens_per_min"):
        extra_cards = (
            f"<div class='card'><h2>Throughput</h2><table>"
            f"<tr><td class='lbl'>input tokens per minute</td>"
            f"<td>{num(tp['input_tokens_per_min'])} tok/min</td></tr>"
            f"<tr><td class='lbl'>output tokens per minute</td>"
            f"<td>{num(tp['output_tokens_per_min'])} tok/min</td></tr>"
            f"</table></div>")
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

    cost = s.get("cost")
    cost_html = ""
    if cost and cost.get("error"):
        cost_html = (f"<div class='card'><h2>Cost</h2>"
                     f"<div class='cap'>config error: {esc(cost['error'])}</div>"
                     f"</div>")
    elif cost and cost["mode"] == "per_token" and cost.get("coverage_warning"):
        cost_html = (
            "<div class='card'><h2>Cost (Databricks DBUs)</h2>"
            "<div class='banner warn'>Full-run cost is unavailable. "
            + esc(cost["coverage_warning"])
            + "</div></div>")
    elif cost and cost["mode"] == "per_token" \
            and (cost.get("dbu_per_request") or {}).get("p50") is None:
        cost_html = ("<div class='card'><h2>Cost (Databricks DBUs)</h2>"
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
            f"<tr><td class='lbl'>DBU per request (p50)</td>"
            f"<td>{_money(cost['dbu_per_request']['p50'])}</td></tr>",
            f"<tr><td class='lbl'>DBU per request (p95)</td>"
            f"<td>{_money(cost['dbu_per_request']['p95'])}</td></tr>",
            f"<tr><td class='lbl'>DBU per 1,000 requests</td>"
            f"<td>{_money(cost['dbu_per_1k_requests'], 2)}</td></tr>",
            f"<tr><td class='lbl'>DBU per minute</td>"
            f"<td>{_money(cost['dbu_per_min'], 3)}</td></tr>",
            f"<tr><td class='lbl'>cache DBUs saved</td>"
            f"<td>{_money(cost['cache_dbu_saved'], 3)}</td></tr>",
        ]
        cap = (f"per-token rates you supplied (DBU/M): input {num(r.get('input'), 3)}, "
               f"output {num(r.get('output'), 3)}, cache-read {num(r.get('cache_read'), 3)}"
               + (f", at ${usd}/DBU" if usd else "")
               + ". cached input is billed at the cache-read rate.")
        cost_html = (f"<div class='card'><h2>Cost (Databricks DBUs)</h2>"
                     f"<div class='cap'>{cap}</div><table>{''.join(rows)}"
                     f"</table></div>")
    elif cost:
        usd = cost.get("usd_per_dbu")
        eff = cost.get("effective_dbu_per_1m_tokens")
        effv = (f"{num(eff, 1)} DBU"
                + (f" (${num(eff * usd, 2)})" if usd and eff is not None else "")
                if eff is not None else "throughput too low to compute")
        rows = [
            f"<tr><td class='lbl'>capacity rate</td>"
            f"<td>{num(cost['dbu_per_hour'], 3)} DBU/hour"
            + (f" (${num(cost['dbu_per_hour'] * usd, 3)})" if usd else "")
            + "</td></tr>",
            f"<tr><td class='lbl'>effective cost per 1M tokens</td>"
            f"<td>{effv}</td></tr>",
        ]
        cost_html = (f"<div class='card'><h2>Cost (Databricks DBUs, "
                     f"provisioned)</h2><div class='cap'>provisioned throughput "
                     f"bills by capacity, so effective cost per 1M tokens is the "
                     f"hourly rate over tokens served per hour at the measured "
                     f"throughput. it improves as you fill the endpoint.</div>"
                     f"<table>{''.join(rows)}</table></div>")

    sw = (s.get("sample") or {}).get("warning")
    sample_banner = (f"<div class='banner warn'>{esc(sw)}</div>" if sw else "")
    rw = (s.get("replay") or {}).get("warning")
    if rw:
        sample_banner += f"<div class='banner warn'>{esc(rw)}</div>"
    cw = (s.get("client") or {}).get("warning")
    if cw:
        sample_banner += f"<div class='banner warn'>{esc(cw)}</div>"
    nw = (s.get("concurrency") or {}).get("warning")
    if nw:
        sample_banner += f"<div class='banner warn'>{esc(nw)}</div>"

    _netw = (s.get("network_path") or {}).get("warning")
    if _netw:
        sample_banner += f"<div class='banner warn'>{esc(_netw)}</div>"
    for warning in (
            (s.get("throughput") or {}).get("coverage_warning"),
            (s.get("cost") or {}).get("coverage_warning"),
            (s.get("cache_fidelity") or {}).get("warning"),
            (s.get("token_targeting") or {}).get("warning"),
            (s.get("latency_population") or {}).get("warning")):
        if warning:
            sample_banner += f"<div class='banner warn'>{esc(warning)}</div>"

    drift = s.get("drift") or {}
    if drift.get("windows") or drift.get("drift_kind"):
        wr = "".join(
            f"<tr><td class='lbl'>window {w['window']} "
            f"({w['n']} content-bearing streams)"
            f"{'' if w.get('counted', True) else ', not counted'}</td>"
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
            f"<div class='card'><h2>Stability over time &nbsp;{flag}</h2>"
            f"<div class='cap'>"
            f"{'per-' + str(drift.get('window_seconds', 60)) + 's windows, counts and p95 in ms. ' if drift.get('windows') else ''}"
            f"{sp}"
            f"{esc(drift.get('drift_headline') or drift.get('note', ''))}"
            f"{('<br>' + esc(drift.get('note', ''))) if drift.get('drift_headline') else ''}"
            f"</div>"
            + (f"<table><tr><th class='lbl'>window</th><th>errors</th>"
               f"<th>TTFT p95</th><th>E2E p95</th></tr>{wr}</table>"
               if drift.get("windows") else "")
            + "</div>")
    else:
        drift_html = (f"<div class='card'><h2>Stability over time</h2>"
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
            f"<tr><td class='lbl'>name</td><td>{esc(str(em.get('name')))}</td></tr>"
            + (f"<tr><td class='lbl'>task</td>"
               f"<td>{esc(str(em.get('task')))}</td></tr>"
               if em.get("task") else "")
            + f"<tr><td class='lbl'>route optimized</td>"
            f"<td>{esc(str(em.get('route_optimized')))}</td></tr>"
            f"<tr><td class='lbl'>ready</td><td>{esc(str(em.get('ready')))}</td></tr>"
            + (f"<tr><td class='lbl'>served entity</td><td>{detail}</td></tr>"
               if detail else "")
            + "</table></div>")

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
                  ("produced at least one content delta",
                   a.get("content_streams", a.get("transport_ok"))),
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
        if a.get("http_status_observed_for"):
            rows_a.insert(2, (
                "returned HTTP 200",
                f"{a.get('http_200')} (status recorded for "
                f"{a.get('http_status_observed_for')} requests)"))
        ans_html = (
            "<div class='card'><h2>Answers</h2><table>"
            + "".join(f"<tr><td class='lbl'>{esc(k)}</td>"
                      f"<td>{esc(str(v))}</td></tr>" for k, v in rows_a)
            + f"</table><div class='cap'>{esc(a.get('note') or '')}</div>"
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
            "<div class='card'><h2>Latency as the caller experienced it</h2>"
            "<div class='cap'>Includes time the request waited on the "
            "client.</div><table><tr><th class='lbl'>metric</th><th>p50</th>"
            "<th>p95</th><th>p99</th></tr>"
            + "".join(f"<tr><td class='lbl'>{esc(n)}</td>"
                      f"<td>{num(t['p50'])}</td><td>{num(t['p95'])}</td>"
                      f"<td>{num(t['p99'])}</td></tr>" for n, t in r_)
            + "</table><div class='cap'>"
            + esc(s.get("latency_correction_note") or "") + "</div></div>")

    body = (
        f"<div class='wrap'><h1>{esc(title)}</h1>"
        f"<div class='sub'>{sub}</div>{sample_banner}{banner}{stats}"
        f"{em_html}{ans_html}{sla_html}{lat_html}{corr_html}"
        f"{drift_html}{believe}{cost_html}"
        f"{extra_cards}{note_html}{label_html}"
        f"<div class='foot'>llm-traffic-replay report</div></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{esc(title)}</title>{_HTML_STYLE}"
            f"</head><body>{body}</body></html>")
