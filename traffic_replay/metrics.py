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
    label = run.get("label") or run.get("profile_label")
    label_html = (f"<div class='label-note'><b>Label:</b> {esc(label)}</div>"
                  if label else "")

    body = (
        f"<div class='wrap'><h1>{esc(title)}</h1>"
        f"<div class='sub'>{sub}</div>{banner}{stats}"
        f"{sla_html}{lat_html}{believe}{extra_cards}{note_html}{label_html}"
        f"<div class='foot'>llm-traffic-replay report</div></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{esc(title)}</title>{_HTML_STYLE}"
            f"</head><body>{body}</body></html>")
