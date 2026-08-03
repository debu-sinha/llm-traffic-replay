"""Summaries and the honesty block.

Every latency table is printed WITH the context that decides whether it can
be believed: achieved cache-hit distribution (endpoint-reported), achieved
arrival rate vs scheduled, client dispatch lag, error rate, and token
targeting error. A good p50 at the wrong cache rate is a fake result; this
module makes the pairing unavoidable.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from . import __version__

PCTS = (50, 90, 95, 99)


def _pct_table(values: list[float | None]) -> dict:
    xs = np.array([v for v in values if v is not None], dtype=float)
    if xs.size == 0:
        return {f"p{p}": None for p in PCTS} | {"n": 0}
    out = {f"p{p}": float(np.percentile(xs, p)) for p in PCTS}
    out["n"] = int(xs.size)
    out["mean"] = float(xs.mean())
    return out


def summarize(results: list[dict], schedule_meta: dict | None = None,
              run_meta: dict | None = None,
              acceptance: dict | None = None,
              ttft_definition: str = "first_content",
              pricing: dict | None = None) -> dict:
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    # achieved cache, endpoint-reported only
    ach = [(r["cached_tokens"] / r["prompt_tokens"])
           for r in ok
           if r.get("cached_tokens") is not None
           and r.get("prompt_tokens")]
    cache_sources = sorted({r.get("cached_tokens_source") for r in ok
                            if r.get("cached_tokens_source")})

    # token targeting: endpoint-reported prompt tokens vs intended
    ratios = [r["prompt_tokens"] / r["intended_input_tokens"]
              for r in ok
              if r.get("prompt_tokens") and r.get("intended_input_tokens")]
    out_ratios = [r["completion_tokens"] / r["intended_output_tokens"]
                  for r in ok
                  if r.get("completion_tokens")
                  and r.get("intended_output_tokens")]
    finish_reasons: dict[str, int] = {}
    for r in ok:
        fr = r.get("finish_reason")
        if fr:
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

    # arrival honesty
    lags = [r.get("dispatch_lag_ms") for r in results
            if r.get("dispatch_lag_ms") is not None]
    retried = sum(1 for r in results if r.get("retries"))

    dur = None
    if results:
        t0 = min(r["t_send_unix"] for r in results)
        t1 = max(r["t_send_unix"] for r in results)
        dur = max(t1 - t0, 1e-9)

    # throughput in the customer's own vocabulary (tokens per minute)
    in_tok = sum(r["prompt_tokens"] for r in ok if r.get("prompt_tokens"))
    out_tok = sum(r["completion_tokens"] for r in ok
                  if r.get("completion_tokens"))
    cached_tok = sum(r["cached_tokens"] for r in ok if r.get("cached_tokens"))
    dur_min = (dur / 60.0) if dur else None

    summary = {
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "requests_retried": retried,
        "error_rate": len(failed) / len(results) if results else None,
        "failures_by_error": _top_errors(failed),
        "ttft_ms": _pct_table([r.get("ttft_ms") for r in ok]),
        "ttfb_ms": _pct_table([r.get("ttfb_ms") for r in ok]),
        "connect_ms": _pct_table([r.get("connect_ms") for r in ok]),
        "e2e_ms": _pct_table([r.get("e2e_ms") for r in ok]),
        "interchunk_max_ms": _pct_table(
            [r.get("interchunk_max_ms") for r in ok]),
        "throughput": {
            "input_tokens_per_min": in_tok / dur_min if dur_min else None,
            "output_tokens_per_min": out_tok / dur_min if dur_min else None,
            "note": "endpoint-reported token counts over wall time",
        },
        "achieved_cache_fraction": _pct_table(ach) | {
            "reported_for_n": len(ach),
            "source_fields": cache_sources or ["NOT REPORTED BY ENDPOINT"],
        },
        "intended_cache_fraction": _pct_table(
            [r.get("intended_cache_fraction") for r in results]),
        "token_targeting": {
            "reported_over_intended_p50":
                float(np.percentile(ratios, 50)) if ratios else None,
            "abs_error_pct_p50":
                float(abs(np.percentile(ratios, 50) - 1.0) * 100)
                if ratios else None,
            "output_reported_over_intended_p50":
                float(np.percentile(out_ratios, 50)) if out_ratios else None,
            "output_abs_error_pct_p50":
                float(abs(np.percentile(out_ratios, 50) - 1.0) * 100)
                if out_ratios else None,
            "finish_reasons": finish_reasons,
            "note": "endpoint-reported token counts are the source of truth. "
                    "input side is calibrated, output side is only reported "
                    "(models may stop before max_tokens: finish_reason stop "
                    "vs length)",
        },
        "arrivals": {
            "achieved_qps_overall": len(results) / dur if dur else None,
            "dispatch_lag_ms": _pct_table(lags),
            "note": "dispatch lag is client lateness vs the schedule; "
                    "sustained growth means the client, not the endpoint, "
                    "is the bottleneck",
        },
        "schedule": schedule_meta or {},
        "run": run_meta or {},
    }
    for fld in ("ttfr_ms", "ttfv_ms"):
        vals = [r.get(fld) for r in ok]
        if any(v is not None for v in vals):
            summary[fld] = _pct_table(vals)
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
            summary["reasoning_tokens"] = _pct_table(chunk_vals)
            summary["reasoning_tokens_total"] = ctotal
            summary["reasoning_tokens_source"] = \
                "stream-counted reasoning deltas (estimate)"
            if dur_min:
                summary["throughput"]["reasoning_tokens_per_min"] = \
                    ctotal / dur_min
    n_ok = len(ok)
    if n_ok == 0:
        sample_warning = ("no successful requests, so there are no latency "
                          "numbers to read. check the failures block")
    elif n_ok < 30:
        sample_warning = ("very small sample: treat p95/p99 as indicative "
                          "only, run more requests for a stable tail")
    elif n_ok < 100:
        sample_warning = "small sample: p99 is unstable below ~100 requests"
    else:
        sample_warning = None
    summary["sample"] = {"n": n_ok, "warning": sample_warning}
    summary["drift"] = _drift_block(ok, failed)

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
    # repeat, which the endpoint prompt cache serves. the achieved cache
    # fraction then describes the replay, not the caller's production mix.
    rm = run_meta or {}
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
                f"prompt already sent and are served from the endpoint prompt "
                f"cache. treat the achieved cache fraction and TTFT as replay "
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
    everything = ok + [f for f in failed if f.get("t_send_unix") is not None]
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

    if mode == "provisioned":
        dph = pricing.get("dbu_per_hour")
        if dph is None:
            return {"mode": mode, "error": "provisioned needs dbu_per_hour"}
        dur_hr = (dur / 3600.0) if dur else None
        tph = (tok_total / dur_hr) if dur_hr else None
        eff = (dph / (tph / 1e6)) if tph else None
        block = {"mode": "provisioned", "dbu_per_hour": dph,
                 "effective_dbu_per_1m_tokens": eff,
                 "tokens_measured": tok_total,
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
    per = []
    for r in ok:
        pt = r.get("prompt_tokens") or 0
        ct = r.get("cached_tokens") or 0
        comp = r.get("completion_tokens") or 0
        uncached = max(pt - ct, 0)
        per.append(uncached / 1e6 * inp + ct / 1e6 * cache + comp / 1e6 * out)
    total = sum(per)
    n = len(per)
    block = {
        "mode": "per_token",
        "dbu_per_request": _pct_table(per),
        "dbu_total": total,
        "dbu_per_1k_requests": (total / n * 1000) if n else None,
        "dbu_per_min": (total / (dur / 60.0)) if dur else None,
        "cache_dbu_saved": cached_tok / 1e6 * max(inp - cache, 0.0),
        "rates_dbu_per_m": {"input": inp, "output": out, "cache_read": cache},
        "note": "cost from endpoint-reported tokens times user-supplied DBU "
                "rates (Databricks pricing page). cached input is billed at "
                "the cache-read rate.",
    }
    if usd is not None:
        block["usd_per_dbu"] = usd
        block["usd_total"] = total * usd
        block["usd_per_1k_requests"] = (block["dbu_per_1k_requests"] * usd
                                        if block["dbu_per_1k_requests"] is not None
                                        else None)
        block["usd_per_min"] = (block["dbu_per_min"] * usd
                                if block["dbu_per_min"] is not None else None)
        block["cache_usd_saved"] = block["cache_dbu_saved"] * usd
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
    out: dict = {"targets_source": "profile acceptance_targets",
                "ttft_definition": ttft_definition}

    def score(name, table_key, targets):
        rows = []
        for q, target in (targets or {}).items():
            actual = (summary.get(table_key) or {}).get(q)
            rows.append({
                "quantile": q, "target_ms": target,
                "actual_ms": round(actual, 1) if actual is not None else None,
                "met": (actual <= target) if actual is not None else None,
            })
        out[name] = rows

    ttft_key = "ttft_ms" if ttft_definition == "first_content" else "ttfv_ms"
    score("ttft_vs_target", ttft_key, acceptance.get("ttft_ms"))
    score("ttfg_vs_target", "e2e_ms", acceptance.get("ttfg_ms"))

    hard = acceptance.get("hard_timeouts") or {}
    ttft_cap = (hard.get("ttft_s") or 0) * 1000.0
    ttfg_cap = (hard.get("ttfg_s") or 0) * 1000.0
    inter_cap = acceptance.get("interchunk_ms")
    timeouts = inter_breaches = 0
    failing = set()
    for idx, r in enumerate(ok):
        over_time = bool(
            (ttft_cap and (r.get("ttft_ms") or 0) > ttft_cap)
            or (ttfg_cap and (r.get("e2e_ms") or 0) > ttfg_cap))
        over_inter = bool(inter_cap) and r.get("interchunk_max_ms") is not None \
            and r["interchunk_max_ms"] > inter_cap
        if over_time:
            timeouts += 1
        if over_inter:
            inter_breaches += 1
        if over_time or over_inter:
            failing.add(idx)
    out["hard_timeout_breaches"] = timeouts
    if inter_cap is not None:
        out["interchunk_breaches"] = inter_breaches

    target_sr = acceptance.get("success_rate")
    if target_sr and total:
        actual_sr = (len(ok) - len(failing)) / total
        out["success_rate"] = {
            "target": target_sr,
            "actual": round(actual_sr, 6),
            "met": actual_sr >= target_sr,
            "note": "failures, hard-timeout breaches, and interchunk breaches "
                    "count against it",
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
    _sw = (s.get("sample") or {}).get("warning")
    if _sw:
        cautions += [f"CAUTION (sample size): {_sw}", ""]
    _rw = (s.get("replay") or {}).get("warning")
    if _rw:
        cautions += [f"CAUTION (prompt replay): {_rw}", ""]

    lines = [
        f"# {title}",
        "",
        f"requests: {s['requests_total']} total, {s['requests_ok']} ok, "
        f"{s['requests_failed']} failed "
        f"(error rate {100 * (s['error_rate'] or 0):.2f}%)",
        "",
        *cautions,
        "| metric (ms) | p50 | p90 | p95 | p99 | n |",
        "|---|---|---|---|---|---|",
        row("TTFT", s["ttft_ms"]),
        row("TTFB", s["ttfb_ms"]),
        row("TTFG (E2E)", s["e2e_ms"]),
        row("interchunk max", s["interchunk_max_ms"]),
        "",
        "## Believability block (read before quoting any number above)",
        f"- achieved cache fraction, endpoint-reported: {ach_line}",
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
        f"{_lag_p95(arr)} ms"
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
    conn = s.get("connect_ms") or {}
    if conn.get("n"):
        lines.append(
            f"- connection setup (DNS, TCP and TLS, ms): p50 "
            f"{conn['p50']:.0f} / p95 {conn['p95']:.0f}. this is EXCLUDED "
            f"from ttft/ttfb/ttfg, do not subtract it again. a handshake is "
            f"several round trips, so it is not the per-request network cost "
            f"of a pooled production client, it is an upper bound on it")
    lb = s.get("latency_basis")
    if lb:
        lines.append(f"- latency basis: {lb}")

    rt = s.get("reasoning_tokens_total")
    if rt is not None:
        rtab = s.get("reasoning_tokens") or {}
        rpm = (s.get("throughput") or {}).get("reasoning_tokens_per_min")
        permin = f", {rpm:,.0f}/min" if rpm else ""
        lines.append(
            f"- reasoning tokens: {rt:,} total{permin}, p50 "
            f"{rtab.get('p50', 0):.0f} per request "
            f"(field: {s.get('reasoning_tokens_source')})")

    tp = s.get("throughput") or {}
    if tp.get("input_tokens_per_min"):
        lines += ["", f"throughput: {tp['input_tokens_per_min']:,.0f} input "
                      f"tokens/min, {tp['output_tokens_per_min']:,.0f} output "
                      "tokens/min (endpoint-reported counts over wall time)"]
    cost = s.get("cost")
    if cost and cost.get("error"):
        lines += ["", f"cost: config error, {cost['error']}"]
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

    sla = s.get("sla")
    if sla:
        _tgt_src = "run config" if mode == "prompts" else "profile config"
        lines += ["", f"## SLA scorecard (targets from the {_tgt_src})",
                  "", "| metric | quantile | target ms | actual ms | met |",
                  "|---|---|---|---|---|"]
        for name, key in (("TTFT", "ttft_vs_target"),
                          ("TTFG", "ttfg_vs_target")):
            for r in sla.get(key) or []:
                met = {True: "yes", False: "NO", None: "-"}[r["met"]]
                lines.append(f"| {name} | {r['quantile']} | {r['target_ms']} "
                             f"| {r['actual_ms']} | {met} |")
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

    if s.get("ttfr_ms"):
        tft = s["ttft_ms"].get("p50")
        tfv = (s.get("ttfv_ms") or {}).get("p50")
        vis = (f"ttfv (first visible token) p50 {tfv:.0f} ms"
               if tfv is not None else
               "some requests emitted no visible content within max_tokens")
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
                      "| window | n (ok) | errors | TTFT p95 | E2E p95 |",
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


def write_outputs(results: list[dict], summary: dict, out_dir: str | Path,
                  title: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "requests.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(render_markdown(summary, title))
    (out / "report.html").write_text(render_html(summary, title))
    return out


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
    sub = (f"{ep} &middot; {src} &middot; {total} requests, {okc} ok, "
           f"{failed} failed")

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
        cards.append(_html_stat("achieved cache p50", num(ach["p50"], 2),
                                "hit fraction (0-1)"))
    else:
        cards.append("<div class='stat'><div class='k'>achieved cache</div>"
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
        for name, key in (("TTFT", "ttft_vs_target"), ("TTFG", "ttfg_vs_target")):
            for r in sla.get(key) or []:
                met = r["met"]
                if met is False:
                    misses += 1
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
        defn = esc(sla.get("ttft_definition", "first_content"))
        note_bits = []
        ttft_rows = sla.get("ttft_vs_target") or []
        if ttft_rows and all(r["actual_ms"] is None for r in ttft_rows):
            fix = (" Raise <code>max_output_tokens_cap</code>, or set "
                   "<code>ttft_definition</code> to <code>first_content</code>,"
                   " to get a number."
                   if defn != "first_content" else
                   " Raise <code>max_output_tokens_cap</code> so requests reach "
                   "that token.")
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
        sla_html = (
            f"<div class='card'><h2>SLA scorecard "
            f"(TTFT scored on {defn})</h2>"
            f"<div class='cap'>target and actual share each row's unit, shown "
            f"in the metric name</div><table>"
            f"<tr><th class='lbl'>metric</th><th>target</th><th>actual</th>"
            f"<th>result</th></tr>{''.join(rows)}</table>{slanote}</div>")
        if misses == 0:
            banner = ("<div class='banner ok'>Meets every acceptance target"
                      "</div>")
        else:
            banner = (f"<div class='banner bad'>{misses} acceptance "
                      f"target{'s' if misses != 1 else ''} missed</div>")

    # ---- latency table ----
    lat = []
    for label, key in (("TTFT (first token)", "ttft_ms"),
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
    lat_html = (
        "<div class='card'><h2>Latency (milliseconds)</h2>"
        "<div class='cap'>p50 to p99 are percentiles across requests, lower is "
        "better. n is the request count. all values in ms.</div><table>"
        "<tr><th class='lbl'>metric</th><th>p50</th><th>p90</th><th>p95</th>"
        f"<th>p99</th><th>n</th></tr>{''.join(lat)}</table></div>")

    # ---- believability panel ----
    bel = []
    if has(ach):
        bel.append(f"<li><b>Achieved cache fraction</b> (endpoint-reported, "
                   f"0-1, share of prompt tokens served from cache): "
                   f"p50 {num(ach['p50'], 3)} / p95 {num(ach['p95'], 3)} "
                   f"(field: {esc(', '.join(ach.get('source_fields') or []))})"
                   f"</li>")
    else:
        bel.append("<li><b>Achieved cache fraction</b>: not reported by this "
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
    rt = s.get("reasoning_tokens_total")
    if rt is not None:
        rpm = (s.get("throughput") or {}).get("reasoning_tokens_per_min")
        pm = f", {num(rpm)}/min" if rpm else ""
        bel.append(f"<li><b>Reasoning tokens</b> (thinking tokens): {num(rt)} "
                   f"tokens total{pm} "
                   f"(field: {esc(str(s.get('reasoning_tokens_source')))})</li>")
    arr = s.get("arrivals") or {}
    if arr.get("achieved_qps_overall"):
        lag = (arr.get("dispatch_lag_ms") or {}).get("p95")
        bel.append(f"<li><b>Arrival honesty</b>: "
                   f"{num(arr['achieved_qps_overall'], 2)} requests/second "
                   f"(QPS) overall, dispatch lag p95 {num(lag)} ms (client "
                   f"lateness, not endpoint latency)</li>")
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

    drift = s.get("drift") or {}
    if drift.get("windows") or drift.get("drift_kind"):
        wr = "".join(
            f"<tr><td class='lbl'>window {w['window']} ({w['n']} ok)"
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
            f"{f'per-' + str(drift.get('window_seconds', 60)) + 's windows, counts and p95 in ms. ' if drift.get('windows') else ''}"
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

    body = (
        f"<div class='wrap'><h1>{esc(title)}</h1>"
        f"<div class='sub'>{sub}</div>{sample_banner}{banner}{stats}"
        f"{em_html}{sla_html}{lat_html}{drift_html}{believe}{cost_html}"
        f"{extra_cards}{note_html}{label_html}"
        f"<div class='foot'>llm-traffic-replay report</div></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{esc(title)}</title>{_HTML_STYLE}"
            f"</head><body>{body}</body></html>")
