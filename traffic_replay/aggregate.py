"""Pool sharded runs (merge) and compare runs side by side (compare).

Both read the standard outputs write_outputs produced (summary.json,
requests.jsonl). Nothing here re-measures: merge re-summarizes the pooled
replay rows, compare tabulates existing summaries. Keeping them out of the
run path means a laptop can aggregate results a fleet of machines produced.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import _pct_table, summarize, write_outputs


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
        run = _load_summary(d).get("run") or {}
        # identity is host plus model plus route. comparing the route alone
        # pooled two different providers whenever both served
        # /v1/chat/completions, which is most of them.
        ident = (run.get("endpoint_base_url"), run.get("endpoint_model"),
                 run.get("endpoint_path"))
        if any(x is not None for x in ident):
            endpoints.add(ident)
        rows += _replay_rows(d)
    if len(endpoints) > 1 and not force:
        _shown = sorted(
            " ".join(str(x) for x in ident if x) for ident in endpoints)
        raise ValueError(
            "refusing to merge runs from different endpoints. identity is "
            f"host, model and route: {_shown}. pass force=True to override.")
    # prompts-mode shards each cycled the same prompt file, so the pooled
    # cache fraction is still replay behavior. carry the fields summarize()
    # needs, otherwise the merged report shows the cache number with no note.
    modes = {(_load_summary(d).get("run") or {}).get("input_mode") for d in dirs}
    counts = {(_load_summary(d).get("run") or {}).get("prompts_count")
              for d in dirs}
    meta = {
        "merged_from": [str(d) for d in dirs],
        "endpoint_path": (next(iter(endpoints))[2] if len(endpoints) == 1
                          else "MIXED"),
        "label": f"merged from {len(dirs)} runs",
        **({"input_mode": "prompts", "prompts_count": counts.pop()}
           if modes == {"prompts"} and len(counts) == 1
           and None not in counts else {}),
        "merge_note": (f"pooled from {len(dirs)} run dirs. throughput is over "
                       "the union wall-clock window, so it is the aggregate "
                       "rate only when the shards ran concurrently."),
    }
    # cost is a per-run figure (rates can differ across pooled runs), so
    # it is not recomputed here; read each run report for its own cost.
    summary = summarize(rows, run_meta=meta, acceptance=acceptance)
    # drift buckets on absolute send time from the pooled minimum. shards that
    # ran at different times produce windows spanning the gap between them, so
    # a trend across pooled rows would describe the schedule, not the endpoint.
    # same hazard as drift below: shards start at different wall-clock times,
    # so a single schedule-vs-send offset across pooled rows reads the gap
    # between shards as lateness.
    summary["arrivals"]["wire_lateness_ms"] = _pct_table([])
    summary["arrivals"]["wire_lateness_note"] = (
        "wire lateness is not computed for a merged run, because pooled rows "
        "come from separate runs and the offset between them would read as "
        "lateness. read each run's own report. dispatch lag below is pooled "
        "and still meaningful, since it is measured within each run.")
    summary.pop("client", None)
    # corrected latency is computed against one schedule offset. pooling rows
    # from runs that started at different wall-clock times makes that offset
    # meaningless: two 200 ms runs an hour apart would report a corrected p95
    # of an hour. same reason wire lateness is blanked.
    for k in ("ttft_corrected_ms", "e2e_corrected_ms",
              "latency_correction_note"):
        summary.pop(k, None)
    summary["latency_correction_note"] = (
        "caller-experienced latency is not computed for a merged run, "
        "because it measures against each run's own schedule and pooled "
        "rows come from different ones. read each run's own report.")
    # concurrency is interval overlap across pooled rows. shards that never
    # ran at the same time have no overlap, so a merged run would report a
    # p50 of 0 in flight. same reason wire lateness and drift are blanked.
    if summary.pop("concurrency", None) is not None:
        summary["concurrency_note"] = (
            "concurrency in flight is not computed for a merged run, because "
            "it is measured by interval overlap and shards that ran at "
            "different times do not overlap. read each run's own report.")
    summary["drift"] = {
        "windows": [], "window_seconds": 60,
        "note": "stability over time is not computed for a merged run. the "
                "pooled rows come from separate runs, so time windows would "
                "span the gaps between them. that also means a merged run "
                "cannot report a breaking point, so if any shard was shedding "
                "requests, read its own report. the pooled error rate below "
                "still counts every failure.",
    }
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
         "Runs measured on the same instrument. Read the warnings and the "
         "believability section before trusting the latency tables.", ""]

    # Everything that can make a side-by-side dishonest goes ABOVE the tables.
    # A reader who stops after the first screen still sees the disqualifiers.
    warns: list[str] = []

    # 0.3.0 moved TCP/TLS setup out of the timed region. putting a 0.2.x
    # column next to a 0.3.x column compares two different measurements.
    vers = {(s.get("harness_version") or "unknown") for s in summ}
    if len(vers) > 1:
        warns.append(
            "these runs came from different harness versions "
            f"({', '.join(sorted(vers))}). 0.3.0 stopped counting TCP/TLS "
            "setup inside TTFT, TTFB and TTFG, so latency columns across "
            "that boundary are not the same measurement. re-run the older "
            "one before comparing.")

    # cache parity. one endpoint reporting no cache at all is the common case
    # when putting Databricks next to a provider that does not report cached
    # tokens, and it is the most misleading comparison the tool can produce,
    # so it has to be louder than a missing cell in a table.
    def _cache_cell(s, q):
        """A missing cache value means the endpoint never reported the field.
        A dash reads like a formatting gap, so say what it actually is."""
        acf = s.get("achieved_cache_fraction") or {}
        v = acf.get(q)
        return "NOT REPORTED" if v is None else f"{v:.3f}"

    caches = [(s.get("achieved_cache_fraction") or {}).get("p50") for s in summ]
    missing = [t for t, c in zip(titles, caches) if c is None]
    have = [c for c in caches if c is not None]
    # a missing value means the endpoint did not report the field, NOT that it
    # served nothing from cache. a reported zero comes through as 0.0.
    if missing and have:
        warns.append(
            f"{', '.join(missing)} did not report cached tokens, so its cache "
            f"usage is unknown, while another run measured a cache p50 of "
            f"{max(have):.3f}. Serving a cached prompt is far cheaper than "
            "serving a cold one, so unless you can establish the unknown side "
            "independently these latency columns may not be measuring the "
            "same work. Do not present this as a like-for-like result.")
    elif missing and not have:
        warns.append(
            "no run reported cached tokens, so cache usage is unknown for "
            "every column. Prompt-cache hit rate is usually the single "
            "biggest driver of the latency you are about to compare. Confirm "
            "how each endpoint handles caching before quoting these numbers.")
    if len(have) >= 2 and (max(have) - min(have)) > 0.10:
        warns.append(
            f"achieved cache p50 spans {min(have):.3f} to {max(have):.3f}, a "
            "gap over 0.10. Comparing latency at different cache rates is not "
            "a fair comparison. Match the cache rates before quoting these "
            "numbers.")

    # error rates. percentiles over a run that dropped requests carry
    # survivorship bias, and the failures are often the slow ones.
    bad = [(t, s.get("error_rate") or 0.0) for t, s in zip(titles, summ)
           if (s.get("error_rate") or 0.0) > 0.01]
    if bad:
        detail = ", ".join(f"{t} at {r * 100:.1f} percent" for t, r in bad)
        warns.append(
            f"these runs failed requests: {detail}. Latency percentiles only "
            "cover requests that succeeded, so a run that dropped its slowest "
            "requests can look faster than one that served them. Read the "
            "error rate next to every latency number below.")

    # sample size. a tail number needs requests behind it.
    thin = [(t, (s.get("sample") or {}).get("n"))
            for t, s in zip(titles, summ)
            if (s.get("sample") or {}).get("warning")]
    if thin:
        detail = ", ".join(f"{t} ({n} requests)" for t, n in thin)
        warns.append(
            f"small samples: {detail}. p99 is unstable below about 100 "
            "requests. Run longer before quoting a tail.")

    # stability. a run still warming up is not a steady-state number.
    moving = [(t, (s.get("drift") or {}).get("drift_kind"))
              for t, s in zip(titles, summ)
              if (s.get("drift") or {}).get("drift_flag")]
    if moving:
        detail = ", ".join(f"{t} ({k})" for t, k in moving)
        broke = [t for t, k in moving if k == "failing"]
        one = len(broke) == 1
        extra = (f" {', '.join(broke)} {'was' if one else 'were'} shedding "
                 f"requests, which {'is a breaking point' if one else 'are breaking points'} "
                 f"rather than {'a latency result' if one else 'latency results'}, "
                 f"so {'its' if one else 'their'} "
                 "surviving percentiles are not comparable to anything."
                 if broke else "")
        warns.append(
            f"these runs were not in steady state: {detail}. Read each run's "
            "stability card. A warming endpoint compared against a warm one "
            "is a measurement artifact, not a difference between "
            f"providers.{extra}")
    # no verdict at all is not the same as passing. a run too short to bucket,
    # or whose windows were too thin to count, was never checked.
    unjudged = [t for t, s in zip(titles, summ)
                if (s.get("drift") or {}).get("drift_kind") is None]
    if unjudged:
        why = {t: ((s.get("drift") or {}).get("note") or "no stability data")
               for t, s in zip(titles, summ)
               if (s.get("drift") or {}).get("drift_kind") is None}
        detail = " ".join(f"{t}: {w}" for t, w in why.items())
        warns.append(
            f"stability was never established for {', '.join(unjudged)}, so "
            "these columns were not checked for warmup or degradation. "
            f"Reported reason per run. {detail}")

    if warns:
        L.append("## Read this before the tables")
        L.append("")
        for w in warns:
            L.append(f"> WARNING: {w}")
            L.append("")
    else:
        L += ["Comparability checks (harness version, cache reporting and "
              "parity, error rate, sample size, steady state) all passed on "
              "these runs.", ""]

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
              "| achieved cache p50 | " + " | ".join(
                  _cache_cell(s, "p50") for s in summ) + " |",
              scalar("input tokens/min",
                     lambda s: (s.get("throughput") or {}).get("input_tokens_per_min"),
                     "{:,.0f}"),
              scalar("output tokens/min",
                     lambda s: (s.get("throughput") or {}).get("output_tokens_per_min"),
                     "{:,.0f}"),
              scalar("reasoning tokens (total)",
                     lambda s: s.get("reasoning_tokens_total"),
                     "{:,.0f}"),
              scalar("DBU per 1k requests",
                     lambda s: (s.get("cost") or {}).get("dbu_per_1k_requests"),
                     "{:,.2f}"), ""])

    L.extend(["## believability (read before trusting the latency tables)",
              hdr, sep,
              "| achieved cache p50 | " + " | ".join(
                  _cache_cell(s, "p50") for s in summ) + " |",
              "| achieved cache p95 | " + " | ".join(
                  _cache_cell(s, "p95") for s in summ) + " |",
              scalar("dispatch lag p95 (ms)",
                     lambda s: ((s.get("arrivals") or {}).get("dispatch_lag_ms")
                                or {}).get("p95")),
              scalar("wire lateness p95 (ms)",
                     lambda s: ((s.get("arrivals") or {}).get("wire_lateness_ms")
                                or {}).get("p95")), ""])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    return out
