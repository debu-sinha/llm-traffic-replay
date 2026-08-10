"""The report must be a faithful summary of the raw per-request log.

This re-derives the headline numbers straight from requests.jsonl with
independent code and asserts the summary matches. It is the guard that a
customer can trust a shared benchmark: the report says what the data says.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run


def test_report_matches_independent_recomputation():
    d = tempfile.mkdtemp()
    truth = Path(d) / "truth.jsonl"
    srv = serve(0, truth, reasoning_tokens=5)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.3)
    try:
        rc = RunConfig(
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "NONE"},
            profile_path="configs/profile_agent_blended.json",
            duration_s=8, qps_base=3.0, qps_burst=6.0, qps_min=1.0,
            qps_max=8.0, max_concurrency=6, calibrate_n=3,
            out_dir=os.path.join(d, "r"), title="accuracy",
            max_output_tokens_cap=40,
            pricing={"mode": "per_token", "input_dbu_per_m": 20.0,
                     "output_dbu_per_m": 62.857, "cache_read_dbu_per_m": 2.0,
                     "usd_per_dbu": 0.07})
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()

    od = Path(out["out_dir"])
    summ = json.load(open(od / "summary.json"))
    rows = [json.loads(x) for x in
            (od / "requests.jsonl").read_text().splitlines()]
    rep = [r for r in rows if r.get("phase") == "replay"]
    ok = [r for r in rep if r.get("ok")]
    assert ok, "no replay requests"

    def pct(vals, q):
        vals = [v for v in vals if v is not None]
        return float(np.percentile(vals, q)) if vals else None

    def approx(a, b):
        if a is None and b is None:
            return True
        return (a is not None and b is not None
                and abs(a - b) <= 1e-6 * max(1.0, abs(b)))

    # counts
    assert summ["requests_total"] == len(rep)
    assert summ["requests_ok"] == len(ok)
    assert summ["requests_failed"] == len(rep) - len(ok)

    # latency percentiles
    for key in ("ttft_ms", "ttfb_ms", "e2e_ms"):
        for q in ("p50", "p95"):
            assert approx(summ[key][q],
                          pct([r.get(key) for r in ok], int(q[1:]))), key

    # Throughput covers the complete logical load window, including idle time
    # before the first sampled arrival, plus any response drain beyond that
    # window. Dividing only from the first sampled send inflates a sparse or
    # Poisson run whenever its first arrival is later than schedule time zero.
    # Use the first physical send for each row, not the final retry attempt.
    def sent(r):
        v = r.get("first_send_unix")
        return r["t_send_unix"] if v is None else v
    # finished_unix closes every sent interval, including retries and failed
    # requests. A service e2e duration belongs only to the final attempt and
    # cannot reconstruct the whole worker lifetime. Current rows carry exact
    # monotonic target-to-send time; retain an epoch fallback so this oracle
    # also describes the legacy contract without subtracting unrelated minima.
    assert all(r.get("finished_unix") is not None for r in rep)
    legacy = [r for r in rep
              if not isinstance(r.get("caller_send_ms"), (int, float))]
    legacy_origin = min(
        (sent(r) - float(r["scheduled_s"]) for r in legacy),
        default=None)
    completion_positions = []
    for r in rep:
        caller_send_ms = r.get("caller_send_ms")
        if isinstance(caller_send_ms, (int, float)):
            logical_send = (float(r["scheduled_s"])
                            + float(caller_send_ms) / 1000.0)
        elif legacy_origin is not None:
            logical_send = sent(r) - legacy_origin
        else:
            logical_send = float(r["scheduled_s"])
        completion_positions.append(
            logical_send + max(r["finished_unix"] - sent(r), 0.0))
    observation_s = max(float(rc.duration_s), *completion_positions)
    dmin = observation_s / 60.0
    intok = sum(r["prompt_tokens"] for r in ok if r.get("prompt_tokens"))
    outtok = sum(r["completion_tokens"] for r in ok
                 if r.get("completion_tokens"))
    assert summ["throughput"]["duration_basis"] == \
        "max(logical_schedule_seconds,response_drain)"
    assert approx(summ["throughput"]["observation_seconds"], observation_s)
    assert approx(summ["throughput"]["input_tokens_per_min"], intok / dmin)
    assert approx(summ["throughput"]["output_tokens_per_min"], outtok / dmin)

    # cost recomputed from rows and the same rates
    inp, out_r, cr = 20.0, 62.857, 2.0
    dbu = sum(
        max((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0), 0)
        / 1e6 * inp
        + (r.get("cached_tokens") or 0) / 1e6 * cr
        + (r.get("completion_tokens") or 0) / 1e6 * out_r
        for r in ok)
    assert approx(summ["cost"]["dbu_total"], dbu)
    assert approx(summ["cost"]["usd_total"], dbu * 0.07)

    # instrument accuracy: client first-visible vs mock true first-content
    tb = {json.loads(x)["request_id"]: json.loads(x)
          for x in truth.read_text().splitlines()}
    errs = [r["ttfv_ms"] - tb[r["request_id"]]["ttft_true_ms"]
            for r in ok
            if r.get("ttfv_ms") is not None and r["request_id"] in tb]
    if errs:
        assert abs(float(np.percentile(errs, 95))) < 60.0  # localhost overhead
