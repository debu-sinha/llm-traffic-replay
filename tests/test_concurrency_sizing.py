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
        srv.shutdown()
        srv.server_close()


def test_worker_defaults_remain_bounded():
    fixed = _cfg(1)
    sized = _cfg(1, sizing_concurrency=8)
    assert fixed.max_concurrency == 256
    # None here preserves whether the caller omitted the sizing cap. The
    # sizing pass derives a pool and applies its separate 256-thread limit.
    assert sized.max_concurrency is None


def test_sizing_honors_explicit_and_default_worker_caps(monkeypatch):
    from traffic_replay import runner

    class Workload:
        def __init__(self, _rc, _n):
            pass

        def plan(self, i, request_id):
            return {
                "messages": [], "max_output": 1,
                "intended": (1, 1, 0.0, i), "chars": 1,
                "global_index": i, "sample_index": i,
                "prompt_index": None, "construction": None,
                "body_request_id": request_id,
            }

    monkeypatch.setattr(runner, "_PreparedWorkload", Workload)
    monkeypatch.setattr(runner, "_payload_hash", lambda *_args: "0" * 64)
    monkeypatch.setattr(
        runner, "_send_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner, "_annotate_result",
        lambda *_args: {
            "ok": True, "e2e_ms": 1000.0,
            "stream_complete": True, "parse_errors": 0,
        })

    def size(max_concurrency):
        rc = _cfg(1, sizing_concurrency=129, calibrate_n=4,
                  max_concurrency=max_concurrency)
        return runner._size_for_concurrency(
            rc, object(), object(), lambda _row: None, True,
            "workload-test", "execution-test")

    # The derived pool is at least 2 * 129 = 258. Omission is still bounded
    # to the safe default, while a caller-supplied lower ceiling wins exactly.
    assert size(None).max_concurrency == 256
    assert size(17).max_concurrency == 17


def test_sizing_rate_uses_mean_service_time_for_skewed_latency(monkeypatch):
    from traffic_replay import runner

    class Workload:
        def __init__(self, _rc, _n):
            pass

        def plan(self, i, request_id):
            return {
                "messages": [], "max_output": 1,
                "intended": (1, 1, 0.0, i), "chars": 1,
                "global_index": i, "sample_index": i,
                "prompt_index": None, "construction": None,
                "body_request_id": request_id,
            }

    latencies_ms = iter([100.0, 100.0, 100.0, 1000.0])
    monkeypatch.setattr(runner, "_PreparedWorkload", Workload)
    monkeypatch.setattr(runner, "_payload_hash", lambda *_args: "0" * 64)
    monkeypatch.setattr(
        runner, "_send_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner, "_annotate_result",
        lambda *_args: {
            "ok": True, "e2e_ms": next(latencies_ms),
            "stream_complete": True, "parse_errors": 0,
        })
    rc = _cfg(1, sizing_concurrency=4, calibrate_n=4,
              max_concurrency=None)

    sized = runner._size_for_concurrency(
        rc, object(), object(), lambda _row: None, True,
        "workload-test", "execution-test")

    # Little's Law: lambda = L / mean(W) = 4 / 0.325 seconds. The former
    # median-based calculation offered 40 rps and implied 13 mean in flight.
    assert abs(sized.qps_base - (4 / 0.325)) < 1e-12
    assert sized.qps_base == sized.qps_burst == sized.qps_min == sized.qps_max
    # p95 remains the conservative worker-headroom input.
    assert sized.max_concurrency == 16


def test_sizing_refuses_survivor_biased_partial_probe(monkeypatch):
    from traffic_replay import runner

    class Workload:
        def __init__(self, _rc, _n):
            pass

        def plan(self, i, request_id):
            return {
                "messages": [], "max_output": 1,
                "intended": (1, 1, 0.0, i), "chars": 1,
                "global_index": i, "sample_index": i,
                "prompt_index": None, "construction": None,
                "body_request_id": request_id,
            }

    calls = {"n": 0}

    def annotate(*_args):
        calls["n"] += 1
        clean = calls["n"] == 1
        return {
            "ok": clean, "e2e_ms": 1.0 if clean else None,
            "stream_complete": clean, "parse_errors": 0,
        }

    monkeypatch.setattr(runner, "_PreparedWorkload", Workload)
    monkeypatch.setattr(runner, "_payload_hash", lambda *_args: "0" * 64)
    monkeypatch.setattr(
        runner, "_send_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "_annotate_result", annotate)
    rc = _cfg(1, sizing_concurrency=8, calibrate_n=8)

    try:
        runner._size_for_concurrency(
            rc, object(), object(), lambda _row: None, True,
            "workload-test", "execution-test")
        assert False, "partial sizing evidence must be refused"
    except RuntimeError as exc:
        assert "1 clean, complete responses from 8 probes" in str(exc)


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
