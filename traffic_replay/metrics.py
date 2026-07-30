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
              run_meta: dict | None = None) -> dict:
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

    # arrival honesty
    lags = [r.get("dispatch_lag_ms") for r in results
            if r.get("dispatch_lag_ms") is not None]
    retried = sum(1 for r in results if r.get("retries"))

    dur = None
    if results:
        t0 = min(r["t_send_unix"] for r in results)
        t1 = max(r["t_send_unix"] for r in results)
        dur = max(t1 - t0, 1e-9)

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
            "note": "endpoint-reported prompt_tokens are the source of truth",
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
    return summary


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
        row("E2E", s["e2e_ms"]),
        "",
        "## Believability block (read before quoting any number above)",
        f"- achieved cache fraction, endpoint-reported: {ach_line}",
        f"- constructed (intended) cache fraction: "
        f"p50 {intent['p50']:.3f} / p95 {intent['p95']:.3f}"
        if intent.get("n") else "- constructed cache fraction: n/a",
        f"- token targeting: reported/intended p50 = "
        f"{tt['reported_over_intended_p50']:.3f} "
        f"(abs error {tt['abs_error_pct_p50']:.1f}%)"
        if tt.get("reported_over_intended_p50") else
        "- token targeting: endpoint did not report prompt_tokens",
        f"- achieved arrival rate: {arr['achieved_qps_overall']:.2f} QPS "
        f"overall; dispatch lag p95 "
        f"{arr['dispatch_lag_ms'].get('p95', float('nan')):.0f} ms"
        if arr.get("achieved_qps_overall") else "- arrivals: n/a",
        f"- failures: {json.dumps(s['failures_by_error'])}"
        if s["requests_failed"] else "- failures: none",
        f"- requests that needed a connection retry: {s['requests_retried']} "
        "(retried requests restart their latency clock; a nonzero count "
        "here means the tail has survivorship bias, read with care)"
        if s.get("requests_retried") else "- connection retries: none",
    ]
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
