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
    for bad in ("32:1", "0:10", "-5:10", "abc", "", "1:2:3:4", "0", "-3"):
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


def test_a_caution_still_counts_as_held_but_says_so():
    from traffic_replay.cli import _sweep_report
    tmp_path = Path(tempfile.mkdtemp(prefix='sweep-'))
    rungs = [_rung(1, "ok", held=2), _rung(2, "caution", held=5)]
    _sweep_report(rungs, tmp_path, _Args())
    body = (tmp_path / "sweep.md").read_text()
    assert "Highest rate that held: 2 requests/second" in body
    assert "Read it with care" in body


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
        concurrency = None
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
    base.pop("concurrency", None)
    base.pop("_input_tokens", None)
    cfg = copy.deepcopy(base)
    cfg.update(qps_base=4.0, qps_burst=4.0, qps_min=4.0, qps_max=4.0,
               rate_scale=1.0, duration_s=10,
               out_dir=str(Path(A.out_dir) / "rate_4"),
               max_concurrency=120)
    rc = RunConfig(**cfg)              # must not raise
    assert rc.qps_base == 4.0
    assert rc.concurrency is None, "the ladder sets a rate, not a concurrency"
