"""Summaries and the honesty block.

Every latency table is printed WITH the context that decides whether it can
be believed: achieved cache-hit distribution (endpoint-reported), achieved
arrival rate vs scheduled, client dispatch lag, error rate, and token
targeting error. A good p50 at the wrong cache rate is a fake result; this
module makes the pairing unavoidable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
              ttft_definition: str = "first_content") -> dict:
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
    if acceptance:
        summary["sla"] = _evaluate_sla(ok, len(results), summary, acceptance,
                                       ttft_definition)
    return summary


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

    lines = [
        f"# {title}",
        "",
        f"requests: {s['requests_total']} total, {s['requests_ok']} ok, "
        f"{s['requests_failed']} failed "
        f"(error rate {100 * (s['error_rate'] or 0):.2f}%)",
        "",
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
        f"overall; dispatch lag p95 "
        f"{arr['dispatch_lag_ms'].get('p95', float('nan')):.0f} ms"
        if arr.get("achieved_qps_overall") else "- arrivals: n/a",
        f"- arrival schedule: from trace {sched_src}"
        if sched_src != "synthetic" else "- arrival schedule: synthetic bursts",
        f"- failures: {json.dumps(s['failures_by_error'])}"
        if s["requests_failed"] else "- failures: none",
        f"- requests that needed a connection retry: {s['requests_retried']} "
        "(retried requests restart their latency clock; a nonzero count "
        "here means the tail has survivorship bias, read with care)"
        if s.get("requests_retried") else "- connection retries: none",
    ]
    tp = s.get("throughput") or {}
    if tp.get("input_tokens_per_min"):
        lines += ["", f"throughput: {tp['input_tokens_per_min']:,.0f} input "
                      f"tokens/min, {tp['output_tokens_per_min']:,.0f} output "
                      "tokens/min (endpoint-reported counts over wall time)"]
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
                  f"either kind) p50 {tft:.0f} ms; {vis}. agree which "
                  "definition the SLA scores via ttft_definition in the run "
                  "config."]

    label = (s.get("run") or {}).get("label")
    if label:
        lines += ["", f"**Label: {label}**"]
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
    return out
