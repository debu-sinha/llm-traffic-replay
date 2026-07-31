"""Run orchestration: schedule -> paced dispatch -> results.

Pacing: each request has an absolute scheduled time. A dispatcher thread
sleeps until each timestamp and submits the request to a bounded thread
pool. If the pool is saturated, the submit itself is late; that lateness is
recorded per request as dispatch_lag_ms and summarized, so client
saturation is visible in the report instead of silently polluting latency.

Warmup/calibration: the first `calibrate_n` requests run at low rate before
the schedule proper; their endpoint-reported prompt_tokens recalibrate the
chars-per-token ratio used to build subsequent request text.
"""
from __future__ import annotations

import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import profile as prof
from .client import EndpointClient, EndpointConfig, new_request_id
from .metrics import summarize, write_outputs
from .prefix_pool import PrefixPool
from .schedule import load_trace, make_schedule, schedule_report, shard
from .textgen import TextMaterializer, calibrate_cpt


@dataclasses.dataclass
class RunConfig:
    profile_path: str
    endpoint: dict                    # EndpointConfig fields
    duration_s: int = 300
    qps_base: float = 25.0
    qps_burst: float = 350.0
    qps_min: float = 10.0
    qps_max: float = 500.0
    rate_scale: float = 1.0
    max_concurrency: int = 256
    seed: int = 7
    cpt: float = 4.0
    calibrate_n: int = 12
    shard_index: int = 0
    shard_total: int = 1
    timestamps_file: str | None = None  # real arrival trace replaces synthetic
    pool_docs_per_bucket: int = 40      # cache-pool shape knobs
    pool_zipf_s: float = 1.1
    out_dir: str = "results"
    title: str = "traffic replay"
    label: str = ""
    max_output_tokens_cap: int = 512  # safety cap for smoke runs; full runs raise it


def _token(cfg: EndpointConfig) -> str | None:
    return os.environ.get(cfg.auth_token_env) or None


def run(rc: RunConfig, token_override: str | None = None,
        quiet: bool = False) -> dict:
    p = prof.Profile.from_json(rc.profile_path)
    ecfg = EndpointConfig(**rc.endpoint)
    client = EndpointClient(ecfg, token_override or _token(ecfg))
    mat = TextMaterializer(cpt=rc.cpt)
    pool = PrefixPool(seed=rc.seed + 4,
                      docs_per_bucket=rc.pool_docs_per_bucket,
                      zipf_s=rc.pool_zipf_s)

    if rc.timestamps_file:
        sched = load_trace(rc.timestamps_file, duration_cap_s=rc.duration_s)
    else:
        sched = make_schedule(
            duration_s=rc.duration_s, qps_base=rc.qps_base,
            qps_burst=rc.qps_burst, qps_min=rc.qps_min, qps_max=rc.qps_max,
            rate_scale=rc.rate_scale, seed=rc.seed + 16)
    if rc.shard_total > 1:
        sched = shard(sched, rc.shard_index, rc.shard_total)
    ts = sched["timestamps"]
    n = len(ts)
    if n == 0:
        raise RuntimeError("schedule produced zero arrivals; "
                           "raise rate_scale or duration")

    draw = prof.sample(p, n, seed=rc.seed)
    assign = pool.assign(draw["prefix_tokens"])

    if not quiet:
        print(f"[runner] {n} scheduled arrivals over {rc.duration_s}s "
              f"(rate_scale {rc.rate_scale}), profile '{p.name}'")
        if p.label:
            print(f"[runner] profile label: {p.label}")

    results: list[dict] = []

    # ---- calibration pass (sequential, low rate) -----------------------
    calib_n = min(rc.calibrate_n, n)
    chars_total = 0
    ptok_total = 0
    for i in range(calib_n):
        rid = new_request_id()
        msgs = mat.messages(rid, int(assign.doc_id[i]),
                            int(assign.prefix_tokens[i]),
                            pool.doc_len.get(int(assign.doc_id[i]), 0),
                            int(draw["suffix_tokens"][i]))
        chars = sum(len(m["content"]) for m in msgs)
        res = client.send(
            msgs, min(int(draw["output_tokens"][i]), rc.max_output_tokens_cap),
            rid, scheduled_s=0.0, dispatch_lag_ms=0.0,
            intended=(int(draw["input_tokens"][i]),
                      int(draw["output_tokens"][i]),
                      float(draw["cache_target_fraction"][i]),
                      int(assign.doc_id[i])),
            chars_sent=chars)
        d = dataclasses.asdict(res)
        d["phase"] = "calibration"
        results.append(d)
        if res.ok and res.prompt_tokens:
            chars_total += chars
            ptok_total += res.prompt_tokens

    if ptok_total:
        new_cpt = calibrate_cpt(mat.cpt, chars_total, ptok_total)
        if not quiet:
            print(f"[runner] cpt calibrated {mat.cpt:.2f} -> {new_cpt:.2f} "
                  f"(from {ptok_total} reported prompt tokens)")
        mat = TextMaterializer(cpt=new_cpt)

    # ---- paced replay ----------------------------------------------------
    idx0 = calib_n
    t0 = time.monotonic() + 0.25
    inflight: list = []
    with ThreadPoolExecutor(max_workers=rc.max_concurrency) as ex:
        for i in range(idx0, n):
            target = t0 + (ts[i] - ts[idx0])
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            lag_ms = max((time.monotonic() - target) * 1000.0, 0.0)

            rid = new_request_id()
            msgs = mat.messages(rid, int(assign.doc_id[i]),
                                int(assign.prefix_tokens[i]),
                                pool.doc_len.get(int(assign.doc_id[i]), 0),
                                int(draw["suffix_tokens"][i]))
            chars = sum(len(m["content"]) for m in msgs)
            fut = ex.submit(
                client.send, msgs,
                min(int(draw["output_tokens"][i]), rc.max_output_tokens_cap),
                rid, float(ts[i]), lag_ms,
                (int(draw["input_tokens"][i]), int(draw["output_tokens"][i]),
                 float(draw["cache_target_fraction"][i]),
                 int(assign.doc_id[i])),
                chars)
            inflight.append(fut)

        for fut in as_completed(inflight):
            d = dataclasses.asdict(fut.result())
            d["phase"] = "replay"
            results.append(d)

    meta = {
        "profile": p.name, "profile_provenance": p.provenance,
        "profile_label": p.label, "cpt_final": mat.cpt,
        "endpoint_path": ecfg.path, "label": rc.label,
        "title": rc.title,
        "shard": f"{rc.shard_index + 1}/{rc.shard_total}",
    }
    summary = summarize([r for r in results if r.get("phase") == "replay"],
                        schedule_meta=schedule_report(sched), run_meta=meta,
                        acceptance=(p.extra or {}).get("acceptance_targets"))
    out = write_outputs(results, summary,
                        Path(rc.out_dir) / time.strftime("%Y%m%d-%H%M%S"),
                        rc.title)
    if not quiet:
        print(f"[runner] wrote {out}/report.md")
    return {"summary": summary, "out_dir": str(out), "results_n": len(results)}
