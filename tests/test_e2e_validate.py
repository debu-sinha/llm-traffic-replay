"""End-to-end instrument check: full pipeline against the bundled mock.

Asserts the three claims the README makes:
  1. Client-measured TTFT tracks server-true TTFT (small positive overhead).
  2. The constructed cache structure produces an endpoint-reported hit
     distribution near the profile target.
  3. Token targeting error against endpoint-reported prompt_tokens is small
     once cpt matches the endpoint (mock truth is exactly 4.0).
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run

@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("val")
    truth = workdir / "truth.jsonl"
    srv = serve(0, truth, per_token_ms=2.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    yield {"truth": truth, "workdir": workdir,
           "port": srv.server_address[1]}
    srv.shutdown()


@pytest.fixture(scope="module")
def run_out(mock):
    rc = RunConfig(
        profile_path=str(Path(__file__).parent.parent
                         / "configs" / "profile_validation_small.json"),
        endpoint={"base_url": f"http://127.0.0.1:{mock['port']}",
                  "path": "/serving-endpoints/mock/invocations",
                  "auth_token_env": "TRAFFIC_REPLAY_NO_TOKEN"},
        duration_s=20, qps_base=6.0, qps_burst=18.0, qps_min=2.0,
        qps_max=30.0, max_concurrency=64, cpt=4.0, calibrate_n=6,
        out_dir=str(mock["workdir"] / "results"),
        title="e2e test", label="test", max_output_tokens_cap=16,
    )
    out = run(rc, quiet=True)
    rows = [json.loads(l) for l in
            (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    truth = {json.loads(l)["request_id"]: json.loads(l)
             for l in mock["truth"].read_text().splitlines()}
    return {"out": out, "rows": rows, "truth": truth}


def test_no_failures(run_out):
    replay = [r for r in run_out["rows"] if r["phase"] == "replay"]
    assert len(replay) > 60
    failed = [r for r in replay if not r["ok"]]
    assert len(failed) == 0, f"failures: {[r['error'] for r in failed[:3]]}"


def test_instrument_error_bounded(run_out):
    deltas = []
    for r in run_out["rows"]:
        if r["phase"] != "replay" or not r["ok"]:
            continue
        tr = run_out["truth"].get(r["request_id"])
        if tr:
            deltas.append(r["ttft_ms"] - tr["ttft_true_ms"])
    assert len(deltas) > 60
    d = np.array(deltas)
    # client overhead must be small and positive-biased (localhost)
    assert np.percentile(d, 50) < 25.0, f"median error {np.percentile(d, 50)}"
    assert np.percentile(d, 95) < 80.0, f"p95 error {np.percentile(d, 95)}"
    assert np.percentile(d, 5) > -5.0  # client can never beat the server


def test_achieved_cache_near_target(run_out):
    summary = run_out["out"]["summary"]
    ach = summary["achieved_cache_fraction"]
    assert ach["n"] > 60, "endpoint-reported cache missing"
    # Overall includes cold first-uses (a large share at this small n) and
    # block quantization; the band is wide but real.
    assert 0.35 <= ach["p50"] <= 0.72, f"achieved p50 {ach['p50']}"
    assert ach["source_fields"] == ["prompt_tokens_details.cached_tokens"]

    # Warm-only view: drop each document's first use (the structural cold
    # miss), then the achieved fraction must sit near the 0.60 target.
    import numpy as np
    replay = sorted((r for r in run_out["rows"]
                     if r["phase"] == "replay" and r["ok"]
                     and r.get("cached_tokens") is not None
                     and r.get("prompt_tokens")),
                    key=lambda r: r["t_send_unix"])
    seen: set[int] = set()
    warm = []
    for r in replay:
        d = r.get("doc_id", -1)
        if d >= 0 and d in seen:
            warm.append(r["cached_tokens"] / r["prompt_tokens"])
        seen.add(d)
    assert len(warm) > 40, f"too few warm requests ({len(warm)})"
    warm_p50 = float(np.percentile(warm, 50))
    assert 0.45 <= warm_p50 <= 0.75, f"warm-only p50 {warm_p50}"


def test_token_targeting_tight_when_cpt_matches(run_out):
    tt = run_out["out"]["summary"]["token_targeting"]
    assert tt["abs_error_pct_p50"] is not None
    assert tt["abs_error_pct_p50"] < 12.0, f"targeting error {tt}"


def test_report_carries_believability_block(run_out):
    report = (Path(run_out["out"]["out_dir"]) / "report.md").read_text()
    assert "Believability block" in report
    assert "achieved cache fraction" in report
    assert "dispatch lag" in report


def test_interchunk_gap_measured_against_real_stream(run_out):
    inter = run_out["out"]["summary"]["interchunk_max_ms"]
    # mock streams completion chunks at per_token_ms=2.0; the widest gap per
    # request should be a few ms on localhost, never zero, never huge
    assert inter["n"] > 60
    assert 0.5 <= inter["p50"] <= 60.0, f"interchunk p50 {inter['p50']}"
