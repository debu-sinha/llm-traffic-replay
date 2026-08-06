"""Setting `concurrency` makes the harness derive the arrival rate and the
pool size from measured service time, instead of the user computing both."""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="conc-"))


def _cfg(port, **kw):
    base = dict(
        profile_path="configs/profile_validation_small.json",
        endpoint={"base_url": f"http://127.0.0.1:{port}",
                  "path": "/serving-endpoints/mock/invocations",
                  "auth_token_env": "UNUSED"},
        duration_s=12, calibrate_n=4, max_output_tokens_cap=16,
        capture_endpoint_metadata=False, out_dir=str(_tmp()),
        title="sizing", label="test")
    base.update(kw)
    return RunConfig(**base)


def _with_mock(make_cfg):
    """Bind an ephemeral port and hand it to the config builder.

    Fixed ports meant the two test runners could not run at the same time,
    and a socket left in TIME_WAIT failed the run outright.
    """
    srv = serve(0, str(_tmp() / "truth.jsonl"))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        return run(make_cfg(port), quiet=True)
    finally:
        srv.shutdown(); srv.server_close()


def test_sizing_concurrency_derives_a_fixed_rate_and_pool():
    """The hint sizes an open-loop rate; it is never claimed as held."""
    out = _with_mock(lambda p: _cfg(p, sizing_concurrency=8))
    s = out["summary"]
    sched = s["schedule"]
    # a rate was chosen, and it is not the RunConfig default of 25
    assert sched["rate_p50"] > 0
    assert abs(sched["rate_p50"] - 25.0) > 1e-6
    # and the run reports what concurrency actually happened, without
    # pretending the open-loop generator held the sizing hint
    assert "concurrency" in s
    assert "asked_for" not in s["concurrency"]
    assert s["concurrency"]["sizing_concurrency_requested"] == 8
    assert s["run"]["load_mode"] == "sizing_concurrency"
    assert s["run"]["sizing_concurrency_requested"] == 8
    assert s["run"]["derived_qps"] > 0


def test_the_sizing_rows_never_reach_the_summary():
    """The probe requests are real traffic, so they are written to
    requests.jsonl, but they must not be scored as part of the replay."""
    import json
    out = _with_mock(lambda p: _cfg(p, sizing_concurrency=6))
    rows = [json.loads(x) for x in
            (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    phases = {r.get("phase") for r in rows}
    assert "sizing" in phases
    replay = [r for r in rows if r.get("phase") == "replay"]
    assert out["summary"]["requests_total"] == len(replay)


def test_without_concurrency_the_configured_rate_is_used():
    out = _with_mock(lambda p: _cfg(p, qps_base=4.0, qps_burst=4.0,
                                    qps_min=4.0, qps_max=4.0,
                                    max_concurrency=8))
    assert abs(out["summary"]["schedule"]["rate_p50"] - 4.0) < 1e-6


def test_a_dead_endpoint_says_why_sizing_failed():
    """Deriving a rate needs at least one response. Failing with a clear
    reason beats dividing by a service time nobody measured."""
    rc = _cfg(1, sizing_concurrency=10)
    rc.endpoint["base_url"] = "http://127.0.0.1:1"
    try:
        run(rc, quiet=True)
        assert False, "expected the sizing pass to refuse"
    except RuntimeError as e:
        assert "sizing pass" in str(e)
        assert "qps_base" in str(e)      # tells them the manual way out
