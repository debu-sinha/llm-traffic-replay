"""Run orchestration: schedule -> paced dispatch -> results.

Two input modes share the same dispatch and measurement path:
  profile mode  (profile_path): synthetic text generated to a statistical
                shape (sizes, cache structure).
  prompts mode  (prompts_file): the user's real prompts, replayed verbatim.

Pacing: open loop. Each request has an absolute scheduled time, and the
dispatcher thread sleeps until that timestamp and submits into a bounded
thread pool. It never waits for a response before firing the next request,
so a slow endpoint does not throttle the offered rate. That is the point: a
closed-loop generator quietly reduces load as the endpoint slows, and you
never find the knee.

Two different lateness numbers come out of this, and they answer different
questions. dispatch_lag_ms is stamped in the dispatcher just before the
submit, so it sees the dispatcher falling behind but NOT a saturated pool,
because ThreadPoolExecutor.submit() queues rather than blocking. Wire
lateness, computed in metrics from first_send_unix against the schedule, is
when the client began sending, and it grows under either. Read wire lateness
to decide whether the client kept up.

Warmup/calibration: the first `calibrate_n` requests run at low rate before
the schedule proper. In profile mode their endpoint-reported prompt_tokens
recalibrate the chars-per-token ratio used to build later request text; in
prompts mode the text is fixed, so the warmup only primes the endpoint.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
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
    endpoint: dict                    # EndpointConfig fields
    profile_path: str | None = None   # profile mode: synthetic text to a shape
    prompts_file: str | None = None   # prompts mode: replay real prompt text
    duration_s: int = 300
    qps_base: float = 25.0
    qps_burst: float = 350.0
    qps_min: float = 10.0
    qps_max: float = 500.0
    rate_scale: float = 1.0
    max_concurrency: int = 256
    concurrency: int | None = None    # "hold N requests in flight". when set,
                                      # a short sizing pass measures service
                                      # time and the arrival rate and pool are
                                      # derived from it, overriding qps_* and
                                      # max_concurrency. load tests are
                                      # specified this way; the harness does
                                      # the arithmetic.
    seed: int = 7
    cpt: float = 4.0
    calibrate_n: int = 12
    shard_index: int = 0
    shard_total: int = 1
    timestamps_file: str | None = None  # real arrival trace replaces synthetic
    pool_docs_per_bucket: int = 40      # cache-pool shape knobs (profile mode)
    pool_zipf_s: float = 1.1
    out_dir: str = "results"
    title: str = "traffic replay"
    label: str = ""
    max_output_tokens_cap: int = 512  # safety cap; full runs raise it
    acceptance_targets: dict | None = None  # SLA targets (either mode)
    pricing: dict | None = None              # DBU cost rates (see metrics)
    capture_endpoint_metadata: bool = True   # read serving-endpoint config
    measure_network_path: bool = True        # time the round trip to it
    ttft_definition: str = "first_content"   # or "first_visible"; sla scores it


def _shard_concurrency(rc) -> int | None:
    """Concurrency this shard is responsible for.

    Sizing derives one rate for the whole target concurrency, then `shard()`
    hands each worker every Nth arrival. A shard therefore offers rate/N and
    holds about concurrency/N, so comparing its measured in-flight against
    the unsharded number reports every shard as falling short.
    """
    if not rc.concurrency:
        return None
    return max(1, int(round(rc.concurrency / max(1, rc.shard_total))))


def _size_for_concurrency(rc: "RunConfig", ecfg, token, out_rows: list,
                          quiet: bool) -> "RunConfig":
    """Turn "hold N in flight" into an arrival rate and a pool size.

    Load tests are specified in concurrency, the generator is specified in
    arrival rate, and converting between them needs the endpoint's service
    time, which nobody knows before measuring. So measure it: send a few
    requests sequentially, take the median and p95 end-to-end, then set

        rate = concurrency / e2e_p50
        pool = rate * e2e_p95 * headroom

    Sizing the pool off p95 rather than p50 matters. At p50 the pool is right
    half the time and queues the other half, and a queued request is one the
    endpoint never saw on schedule.
    """
    import numpy as _np

    from .client import EndpointClient
    from .textgen import TextMaterializer as _TM
    from . import profile as _prof
    from .prefix_pool import PrefixPool as _PP

    probe_n = max(4, min(rc.calibrate_n, 8))
    client = EndpointClient(ecfg, token,
                            refresh=lambda: _token(ecfg))
    if rc.prompts_file:
        from .prompts import load_prompts
        msgs_list = load_prompts(rc.prompts_file)
        def _mk(i):
            m = msgs_list[i % len(msgs_list)]
            return m, rc.max_output_tokens_cap, (0, 0, None, i % len(msgs_list)), \
                sum(len(x["content"]) for x in m)
    else:
        p = _prof.Profile.from_json(rc.profile_path)
        mat = _TM(cpt=rc.cpt)
        pool = _PP(seed=rc.seed + 4, docs_per_bucket=rc.pool_docs_per_bucket,
                   zipf_s=rc.pool_zipf_s)
        draw = _prof.sample(p, probe_n, seed=rc.seed)
        assign = pool.assign(draw["prefix_tokens"])
        def _mk(i):
            m = mat.messages(f"size-{i}", int(assign.doc_id[i]),
                             int(assign.prefix_tokens[i]),
                             pool.doc_len.get(int(assign.doc_id[i]), 0),
                             int(draw["suffix_tokens"][i]))
            return (m, min(int(draw["output_tokens"][i]),
                           rc.max_output_tokens_cap),
                    (int(draw["input_tokens"][i]),
                     int(draw["output_tokens"][i]),
                     float(draw["cache_target_fraction"][i]),
                     int(assign.doc_id[i])),
                    sum(len(x["content"]) for x in m))

    e2e = []
    for i in range(probe_n):
        msgs, max_out, intended, chars = _mk(i)
        res = client.send(msgs, max_out, new_request_id(), scheduled_s=0.0,
                          dispatch_lag_ms=0.0, intended=intended,
                          chars_sent=chars)
        d = dataclasses.asdict(res)
        d["phase"] = "sizing"
        out_rows.append(d)
        if res.ok and res.e2e_ms:
            e2e.append(res.e2e_ms)

    if not e2e:
        raise RuntimeError(
            "sizing pass got no successful response, so the arrival rate for "
            f"concurrency {rc.concurrency} cannot be derived. check auth and "
            "the endpoint path, or set qps_base and max_concurrency directly.")

    p50 = float(_np.percentile(e2e, 50)) / 1000.0
    p95 = float(_np.percentile(e2e, 95)) / 1000.0
    rate = rc.concurrency / max(p50, 1e-3)
    pool_size = max(rc.concurrency * 2,
                    int(math.ceil(rate * p95 * 1.5)))
    if not quiet:
        print(f"[runner] sizing from {len(e2e)} probe requests: e2e p50 "
              f"{p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms")
        print(f"[runner] to hold {rc.concurrency} in flight: offering "
              f"{rate:.2f} rps, pool {pool_size}")
    return dataclasses.replace(
        rc, qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, max_concurrency=pool_size)


def _token_from_profile(name: str) -> str | None:
    """Resolve a ~/.databrickscfg profile to a bearer token.

    A PAT profile stores the token directly. An OAuth profile stores no
    usable bearer token, so the Databricks CLI is asked to mint one, which
    also refreshes it if it has expired. Returns None if neither works, and
    the caller falls back to the environment variable.
    """
    import configparser
    import json as _json
    import subprocess
    from pathlib import Path

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE",
                                   Path.home() / ".databrickscfg"))
    parser = configparser.ConfigParser()
    if cfg_path.exists():
        parser.read(cfg_path)
        if parser.has_section(name) or name == "DEFAULT":
            sect = parser[name]
            tok = sect.get("token")
            # a PAT is usable as-is. an OAuth profile has auth_type set and
            # either no token or a stale one, so prefer the CLI there.
            if tok and not sect.get("auth_type"):
                return tok
    try:
        out = subprocess.run(["databricks", "auth", "token", "-p", name],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            return _json.loads(out.stdout).get("access_token") or None
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _token(cfg: EndpointConfig) -> str | None:
    if cfg.auth_profile:
        tok = _token_from_profile(cfg.auth_profile)
        if tok:
            return tok
        # falling through silently means a typo runs unauthenticated and
        # surfaces later as a wall of 401s or "sizing got no response"
        print(f"auth profile {cfg.auth_profile!r} did not resolve to a token, "
              f"falling back to ${cfg.auth_token_env}", file=sys.stderr)
    return os.environ.get(cfg.auth_token_env) or None


def run(rc: RunConfig, token_override: str | None = None,
        quiet: bool = False) -> dict:
    prompts_mode = bool(rc.prompts_file)
    if prompts_mode and rc.profile_path:
        raise ValueError("set profile_path or prompts_file, not both")
    if not prompts_mode and not rc.profile_path:
        raise ValueError("set profile_path (synthetic shape) or "
                         "prompts_file (real prompt text)")

    ecfg = EndpointConfig(**rc.endpoint)
    token = token_override or _token(ecfg)
    client = EndpointClient(ecfg, token,
                            refresh=lambda: _token(ecfg))
    req_params = {"temperature": ecfg.temperature,
                  "max_output_tokens_cap": rc.max_output_tokens_cap,
                  "extra_body": ecfg.extra_body or {}}
    # where the client sits relative to the endpoint. this is cheap, and
    # without it a run generated from the wrong region silently folds a
    # round trip into every latency number it prints.
    net_path = None
    if rc.measure_network_path:
        from .netpath import measure_network_path
        net_path = measure_network_path(ecfg.base_url)
        if net_path and not quiet:
            print(f"[runner] network: {net_path['rtt_ms']:.0f} ms round trip "
                  f"to {net_path['endpoint_host']} "
                  f"({', '.join(net_path['endpoint_ips'][:2])})")

    endpoint_meta = None
    if rc.capture_endpoint_metadata:
        from .endpoint_meta import fetch_endpoint_metadata
        endpoint_meta = fetch_endpoint_metadata(ecfg.base_url, ecfg.path,
                                                token, timeout=5.0)

    # ---- sizing pass, only when the caller asked for a concurrency --------
    sizing_rows: list[dict] = []
    if rc.concurrency:
        rc = _size_for_concurrency(rc, ecfg, token, sizing_rows, quiet)

    # arrival schedule is shared by both modes
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

    if prompts_mode:
        from .prompts import load_prompts
        prompt_msgs = load_prompts(rc.prompts_file)
        m = len(prompt_msgs)

        def make_request(i, rid):
            msgs = prompt_msgs[i % m]
            chars = sum(len(x["content"]) for x in msgs)
            # no synthetic target: intended input/output 0, cache unset
            return msgs, rc.max_output_tokens_cap, (0, 0, None, i % m), chars
    else:
        p = prof.Profile.from_json(rc.profile_path)
        mat = TextMaterializer(cpt=rc.cpt)
        pool = PrefixPool(seed=rc.seed + 4,
                          docs_per_bucket=rc.pool_docs_per_bucket,
                          zipf_s=rc.pool_zipf_s)
        draw = prof.sample(p, n, seed=rc.seed)
        assign = pool.assign(draw["prefix_tokens"])

        def make_request(i, rid):
            msgs = mat.messages(rid, int(assign.doc_id[i]),
                                int(assign.prefix_tokens[i]),
                                pool.doc_len.get(int(assign.doc_id[i]), 0),
                                int(draw["suffix_tokens"][i]))
            chars = sum(len(x["content"]) for x in msgs)
            max_out = min(int(draw["output_tokens"][i]),
                          rc.max_output_tokens_cap)
            intended = (int(draw["input_tokens"][i]),
                        int(draw["output_tokens"][i]),
                        float(draw["cache_target_fraction"][i]),
                        int(assign.doc_id[i]))
            return msgs, max_out, intended, chars

    if not quiet:
        if prompts_mode:
            print(f"[runner] {n} scheduled arrivals over {rc.duration_s}s, "
                  f"replaying {m} real prompts from {rc.prompts_file}")
        else:
            print(f"[runner] {n} scheduled arrivals over {rc.duration_s}s "
                  f"(rate_scale {rc.rate_scale}), profile '{p.name}'")
            if p.label:
                print(f"[runner] profile label: {p.label}")

    results: list[dict] = list(sizing_rows)

    # ---- calibration / warmup pass (sequential, low rate) --------------
    # calibration consumes the first calibrate_n scheduled arrivals, so a
    # schedule shorter than that leaves nothing to replay and the report
    # says "0 total" on a run that really did send requests. sharding makes
    # this easier to hit, since n is per shard while calibrate_n is per
    # process.
    if rc.calibrate_n >= n:
        raise ValueError(
            f"calibrate_n is {rc.calibrate_n} but the schedule only has {n} "
            f"arrivals, so calibration would consume all of them and the "
            f"replay would measure nothing. lower calibrate_n below {n}, or "
            f"raise duration_s or the arrival rate."
            + (f" note this is shard {rc.shard_index + 1} of "
               f"{rc.shard_total}, which gets every {rc.shard_total}th "
               "arrival, so its schedule is that much shorter."
               if rc.shard_total > 1 else ""))
    calib_n = min(rc.calibrate_n, n)
    chars_total = 0
    ptok_total = 0
    for i in range(calib_n):
        rid = new_request_id()
        msgs, max_out, intended, chars = make_request(i, rid)
        res = client.send(msgs, max_out, rid, scheduled_s=0.0,
                          dispatch_lag_ms=0.0, intended=intended,
                          chars_sent=chars)
        d = dataclasses.asdict(res)
        d["phase"] = "calibration"
        results.append(d)
        if res.ok and res.prompt_tokens:
            chars_total += chars
            ptok_total += res.prompt_tokens

    # recalibrate chars/token only in profile mode (real prompts are fixed)
    if not prompts_mode and ptok_total:
        new_cpt = calibrate_cpt(mat.cpt, chars_total, ptok_total)
        if not quiet:
            print(f"[runner] cpt calibrated {mat.cpt:.2f} -> {new_cpt:.2f} "
                  f"(from {ptok_total} reported prompt tokens)")
        mat = TextMaterializer(cpt=new_cpt)

    # ---- paced replay ----------------------------------------------------
    idx0 = calib_n
    t0 = time.monotonic() + 0.25
    inflight: list = []
    # the dispatcher submits every request and only then collects, so
    # completions have to report themselves through a callback or the line
    # would sit at zero until the last arrival went out.
    from .progress import Progress
    prog = Progress(n - idx0, float(rc.duration_s), enabled=not quiet)
    with ThreadPoolExecutor(max_workers=rc.max_concurrency) as ex:
        for i in range(idx0, n):
            target = t0 + (ts[i] - ts[idx0])
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            lag_ms = max((time.monotonic() - target) * 1000.0, 0.0)

            rid = new_request_id()
            msgs, max_out, intended, chars = make_request(i, rid)
            fut = ex.submit(client.send, msgs, max_out, rid,
                            float(ts[i]), lag_ms, intended, chars)
            # the callback runs on the worker thread the moment the request
            # finishes, which is what lets the in-flight gauge be live
            # rather than a count of what has been handed to the pool.
            fut.add_done_callback(
                lambda f: prog.done(f.result()) if not f.cancelled() else None)
            prog.sent()
            inflight.append(fut)
            prog.paint()

        for fut in as_completed(inflight):
            d = dataclasses.asdict(fut.result())
            d["phase"] = "replay"
            results.append(d)
            prog.paint()
    prog.finish()

    if prompts_mode:
        meta = {
            "input_mode": "prompts",
            "prompts_file": rc.prompts_file, "prompts_count": m,
            "endpoint_path": ecfg.path, "label": rc.label, "title": rc.title,
            "request_params": req_params, "endpoint_metadata": endpoint_meta,
            "network_path": net_path,
            "shard": f"{rc.shard_index + 1}/{rc.shard_total}",
            "concurrency_target": _shard_concurrency(rc),
            # identity of the thing under test. without these, compare and
            # merge cannot tell two different providers apart when both sit
            # behind the same route.
            "endpoint_base_url": ecfg.base_url,
            "endpoint_model": ecfg.model,
            "profile_path": rc.profile_path,
            "seed": rc.seed,
        }
        acceptance = rc.acceptance_targets
    else:
        meta = {
            "input_mode": "profile",
            "profile": p.name, "profile_provenance": p.provenance,
            "profile_label": p.label, "cpt_final": mat.cpt,
            "endpoint_path": ecfg.path, "label": rc.label, "title": rc.title,
            "request_params": req_params, "endpoint_metadata": endpoint_meta,
            "network_path": net_path,
            "shard": f"{rc.shard_index + 1}/{rc.shard_total}",
            "concurrency_target": _shard_concurrency(rc),
            # identity of the thing under test. without these, compare and
            # merge cannot tell two different providers apart when both sit
            # behind the same route.
            "endpoint_base_url": ecfg.base_url,
            "endpoint_model": ecfg.model,
            "profile_path": rc.profile_path,
            "prompts_file": rc.prompts_file,
            "seed": rc.seed,
        }
        acceptance = (rc.acceptance_targets
                      or (p.extra or {}).get("acceptance_targets"))

    # name the origin, so the scorecard cannot credit the profile for numbers
    # the run config supplied. the CLI stamps its own before we get here.
    if acceptance and "targets_are" not in acceptance:
        acceptance = {**acceptance,
                      "targets_are": ("the run config" if rc.acceptance_targets
                                      else "this profile")}

    summary = summarize([r for r in results if r.get("phase") == "replay"],
                        schedule_meta=schedule_report(sched), run_meta=meta,
                        acceptance=acceptance,
                        ttft_definition=rc.ttft_definition,
                        pricing=rc.pricing,
                        concurrency_target=_shard_concurrency(rc))
    out = write_outputs(results, summary,
                        Path(rc.out_dir) / time.strftime("%Y%m%d-%H%M%S"),
                        rc.title)
    if not quiet:
        print(f"[runner] wrote {out}/report.html (open in a browser) "
              f"and {out}/report.md")
    return {"summary": summary, "out_dir": str(out), "results_n": len(results)}
