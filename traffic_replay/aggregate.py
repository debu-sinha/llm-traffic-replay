"""Pool sharded runs (merge) and compare runs side by side (compare).

Both read the standard outputs write_outputs produced (summary.json,
requests.jsonl). Nothing here re-measures: merge re-summarizes the pooled
replay rows, compare tabulates existing summaries. Keeping them out of the
run path means a laptop can aggregate results a fleet of machines produced.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import summarize, write_outputs


def _load_summary(d: Path) -> dict:
    p = d / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _run_title(d: Path, summ: dict) -> str:
    return (summ.get("run") or {}).get("title") or d.name


def _require_run_dir(d: Path, need: str) -> None:
    if not d.is_dir():
        raise ValueError(f"input run dir not found: {d}")
    if not (d / need).exists():
        raise ValueError(f"{d} is not a run dir (missing {need})")


def _replay_rows(d: Path) -> list[dict]:
    rows = []
    for line in (d / "requests.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") == "replay":
            rows.append(r)
    return rows


def merge_runs(out_dir, input_dirs, title=None, acceptance=None,
               force=False) -> Path:
    """Concatenate replay rows from each run dir and re-summarize the union."""
    dirs = [Path(d) for d in input_dirs]
    for d in dirs:
        _require_run_dir(d, "requests.jsonl")
    endpoints, rows = set(), []
    for d in dirs:
        ep = (_load_summary(d).get("run") or {}).get("endpoint_path")
        if ep:
            endpoints.add(ep)
        rows += _replay_rows(d)
    if len(endpoints) > 1 and not force:
        raise ValueError(
            "refusing to merge runs with different endpoint paths: "
            f"{sorted(endpoints)}. pass force=True to override.")
    meta = {
        "merged_from": [str(d) for d in dirs],
        "endpoint_path": sorted(endpoints)[0] if len(endpoints) == 1
        else "MIXED",
        "label": f"merged from {len(dirs)} runs",
        "merge_note": (f"pooled from {len(dirs)} run dirs. throughput is over "
                       "the union wall-clock window, so it is the aggregate "
                       "rate only when the shards ran concurrently."),
    }
    summary = summarize(rows, run_meta=meta, acceptance=acceptance)
    return write_outputs(rows, summary, out_dir,
                         title or f"merged: {len(dirs)} runs")


def _cell(v, fmt="{:.0f}") -> str:
    return fmt.format(v) if v is not None else "-"


def compare_runs(out_dir, input_dirs) -> Path:
    """Tabulate several runs one column each, on identical measurement, and
    warn when their achieved cache rates diverge enough to make the latency
    comparison meaningless."""
    dirs = [Path(d) for d in input_dirs]
    for d in dirs:
        _require_run_dir(d, "summary.json")
    summ = [_load_summary(d) for d in dirs]
    titles = [_run_title(d, s) for d, s in zip(dirs, summ)]
    n = len(titles)
    hdr = "| metric / quantile | " + " | ".join(titles) + " |"
    sep = "|---" * (n + 1) + "|"
    L = ["# endpoint comparison", "",
         "Runs measured on the same instrument. Read the believability "
         "section before trusting the latency tables.", ""]

    def pct(name, key):
        L.extend([f"## {name}", hdr, sep])
        for q in ("p50", "p90", "p95", "p99"):
            cells = [_cell((s.get(key) or {}).get(q)) for s in summ]
            L.append(f"| {q} | " + " | ".join(cells) + " |")
        L.append("")

    pct("TTFT (ms)", "ttft_ms")
    pct("TTFG / E2E (ms)", "e2e_ms")
    pct("interchunk max (ms)", "interchunk_max_ms")

    def scalar(label, fn, fmt="{:.0f}"):
        return f"| {label} | " + " | ".join(_cell(fn(s), fmt)
                                             for s in summ) + " |"

    L.extend(["## rates and throughput", hdr, sep,
              scalar("error rate", lambda s: s.get("error_rate"), "{:.4f}"),
              scalar("achieved cache p50",
                     lambda s: (s.get("achieved_cache_fraction") or {}).get("p50"),
                     "{:.3f}"),
              scalar("input tokens/min",
                     lambda s: (s.get("throughput") or {}).get("input_tokens_per_min"),
                     "{:,.0f}"),
              scalar("output tokens/min",
                     lambda s: (s.get("throughput") or {}).get("output_tokens_per_min"),
                     "{:,.0f}"), ""])

    L.extend(["## believability (read before trusting the latency tables)",
              hdr, sep,
              scalar("achieved cache p50",
                     lambda s: (s.get("achieved_cache_fraction") or {}).get("p50"),
                     "{:.3f}"),
              scalar("achieved cache p95",
                     lambda s: (s.get("achieved_cache_fraction") or {}).get("p95"),
                     "{:.3f}"),
              scalar("dispatch lag p95 (ms)",
                     lambda s: ((s.get("arrivals") or {}).get("dispatch_lag_ms")
                                or {}).get("p95")), ""])

    caches = [(s.get("achieved_cache_fraction") or {}).get("p50") for s in summ]
    have = [c for c in caches if c is not None]
    if len(have) >= 2 and (max(have) - min(have)) > 0.10:
        L.extend([
            f"**WARNING: achieved cache p50 spans {min(have):.3f} to "
            f"{max(have):.3f}, a gap over 0.10. Comparing latency at "
            "different cache rates is not a fair comparison. Match the "
            "cache rates before quoting these numbers.**", ""])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    return out
