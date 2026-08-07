"""The rate ladder.

The axis is arrival rate, not concurrency, and that is a correctness choice
rather than a convenience. An open-loop generator cannot hold a concurrency:
in-flight is arrival rate times service time, and service time rises under
load, so fixing the rate moves the concurrency. Offering concurrency as an
input would mean either lying about it or going closed loop, and closed loop
is what bakes coordinated omission into every other sweep in the category.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from traffic_replay.cli import _rungs


def test_a_range_becomes_a_geometric_ladder():
    """Geometric because the interesting region is multiplicative: 1 to 2
    matters as much as 16 to 32, and a linear ladder spends most of its
    rungs past the knee."""
    assert _rungs("1:32") == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert _rungs("1:16:5") == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_an_explicit_list_is_taken_as_given_and_sorted():
    assert _rungs("10,2,5") == [2.0, 5.0, 10.0]


def test_nonsense_is_refused_rather_than_producing_a_silent_ladder():
    # a loop rather than parametrize, because the stdlib runner has no marks
    for bad in ("32:1", "0:10", "-5:10", "abc", "", "1:2:3:4", "0", "-3",
                "1,,2", ",1", "1,"):
        try:
            _rungs(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} should have been refused")


def _rung(rate, kind, held=None, err=0.0):
    return {"rate": rate, "kind": kind, "text": f"{kind} at {rate}",
            "dir": f"/tmp/r{rate}", "held": held, "achieved_rps": rate,
            "err": err, "ttft_p50": 100.0, "ttft_p95": 200.0,
            "e2e_p50": 300.0}


class _Args:
    endpoint = "my-endpoint"


def test_the_ceiling_is_the_highest_rung_that_HELD():
    """Every sweep in this category anchors its ceiling on the highest rung
    it managed to submit, then reports a top rung its own error rate
    disqualifies. The ceiling here is the last one that stayed valid."""
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rungs = [_rung(1, "ok", held=2), _rung(2, "ok", held=5),
             _rung(4, "miss", held=9, err=0.4)]
    code = _sweep_report(rungs, tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "Highest rate that held: 2 requests/second" in body
    assert "carried about 5 concurrent" in body
    assert "The next rung, 4 rps, missed" in body
    assert code == 0


def test_a_caution_is_not_claimed_as_a_proven_held_rung():
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rungs = [_rung(1, "ok", held=2), _rung(2, "caution", held=5)]
    _sweep_report(rungs, tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "Highest rate that held: 1 requests/second" in body
    assert "next rung, 2 rps, cautioned" in body


def test_topping_out_says_the_ceiling_may_be_higher():
    """Reporting the top rung as the ceiling when nothing failed would
    understate the endpoint."""
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    _sweep_report([_rung(1, "ok", held=2), _rung(2, "ok", held=4)],
                  tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "top of the ladder" in body
    assert "Raise --rate" in body


def test_no_rung_holding_is_reported_and_exits_nonzero():
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    code = _sweep_report([_rung(1, "miss", err=0.5)], tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "No rung held" in body
    assert "lowest rate tested (1 rps)" in body
    assert code == 1


def test_missing_error_rate_is_not_printed_as_zero():
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rung = _rung(1, "invalid")
    rung["err"] = None
    _sweep_report([rung], tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "| 1 rps | 1.0 | - | - |" in body


def test_concurrency_is_reported_as_measured_not_as_asked():
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    _sweep_report([_rung(1, "ok", held=3)], tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "as measured, not as asked for" in body
    assert "| held |" in body


def test_the_config_the_sweep_builds_is_actually_a_valid_run_config():
    """The preflight adds a key RunConfig does not accept, and the single-run
    path pops it. The ladder did not, so every sweep died on rung 1 with a
    TypeError after the first run had already been paid for."""
    import copy
    import tempfile
    from pathlib import Path
    from traffic_replay.cli import _benchmark_config
    from traffic_replay.runner import RunConfig

    class A:
        host = "https://example.invalid"
        endpoint = "ep"
        auth_profile = None
        token_env = "T"
        model = None
        extra_body = None
        sizing_concurrency = None
        legacy_concurrency = None
        duration = 10
        out_dir = tempfile.mkdtemp()
        title = label = None
        input_tokens = "1000"
        output_tokens = "50"
        cache_hit_rate = "0.2,0.6"
        prompts = profile = None
        ttft_p50 = ttft_p90 = ttft_p95 = ttft_p99 = None
        ttfg_p50 = ttfg_p90 = ttfg_p95 = ttfg_p99 = None
        success_rate = 0.99

    base = _benchmark_config(A())
    base.pop("sizing_concurrency", None)
    cfg = copy.deepcopy(base)
    cfg.update(qps_base=4.0, qps_burst=4.0, qps_min=4.0, qps_max=4.0,
               rate_scale=1.0, duration_s=10,
               out_dir=str(Path(A.out_dir) / "rate_4"),
               max_concurrency=120)
    rc = RunConfig(**cfg)              # must not raise
    assert rc.qps_base == 4.0
    assert rc.sizing_concurrency is None, "the ladder sets a fixed rate"


def test_sweep_reuses_the_exact_workload_and_runs_one_preflight(monkeypatch):
    import json
    from traffic_replay.cli import main

    root = Path(tempfile.mkdtemp(prefix="sweep-exact-"))
    prompts = root / "prompts.jsonl"
    prompts.write_text('{"prompt":"real one"}\n{"prompt":"real two"}\n')
    preflight = []
    runs = []
    sleeps = []

    def fake_preflight(cfg, args):
        preflight.append(json.loads(json.dumps(cfg)))
        return None

    def fake_run(rc, quiet=False):
        runs.append(rc)
        d = Path(rc.out_dir) / "fake"
        d.mkdir(parents=True, exist_ok=True)
        return {"out_dir": str(d), "summary": {
            "arrivals": {"achieved_qps_overall": rc.qps_base},
            "error_rate": 0.0,
            "ttft_ms": {"p50": 10.0, "p95": 20.0},
            "e2e_ms": {"p50": 30.0},
            "concurrency": {"in_flight_p50": 2.0}}}

    monkeypatch.setattr("traffic_replay.cli._check_preflight", fake_preflight)
    monkeypatch.setattr("traffic_replay.runner.run", fake_run)
    monkeypatch.setattr("traffic_replay.metrics._verdict",
                        lambda summary: ("ok", "held"))
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    code = main([
        "sweep", "--host", "https://ws.example", "--endpoint", "ep",
        "--rate", "1,2", "--duration", "7", "--cooldown", "3",
        "--prompts", str(prompts), "--output-tokens", "40,90",
        "--auth-profile", "workspace-test",
        "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--max-concurrency", "17", "--max-pending-requests", "23",
        "--out-dir", str(root / "out")])

    assert code == 0
    assert len(preflight) == 1
    assert len(runs) == 2
    assert sleeps == [3]
    assert [r.qps_base for r in runs] == [1.0, 2.0]
    for rc in runs:
        assert rc.prompts_file == str(prompts)
        assert rc.max_output_tokens_cap == 135
        assert rc.endpoint["auth_profile"] == "workspace-test"
        assert rc.endpoint["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}}
        assert rc.max_concurrency == 17
        assert rc.max_pending_requests == 23
        assert rc.sizing_concurrency is None
    for rate in (1, 2):
        saved = json.loads((root / "out" / f"rate_{rate}" /
                            "run-config.json").read_text())
        assert saved["prompts_file"] == str(prompts)
        assert saved["endpoint"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}}
