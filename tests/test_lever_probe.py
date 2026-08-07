"""Explicit, provider-qualified reasoning-control probes and refusal UX."""
from __future__ import annotations

import argparse
import contextlib
import io

import pytest

from traffic_replay.cli import _json_object_arg, _print_lever_report


def _cap(levers, budget=512):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_lever_report(levers, budget)
    return buf.getvalue()


def test_the_working_flag_is_printed_ready_to_paste():
    """The whole point: the user should be able to copy one line."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"},
         "verdict": "works", "detail": "answered, finish stop, 109 tokens"},
        {"name": "enable_thinking=false", "extra": {"enable_thinking": False},
         "verdict": "ignored", "detail": "accepted, still no visible answer"},
    ])
    assert """--extra-body '{"reasoning_effort": "none"}'""" in out
    assert "WORKS" in out


def test_a_rejection_keeps_the_reason_the_endpoint_gave():
    """The refusal is often the most useful line, because it names why."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"}, "verdict": "rejected",
         "detail": 'http 400: reasoning_effort="none" is not supported'},
    ])
    assert "rejected" in out
    assert "is not supported" in out


def test_when_nothing_works_it_says_so_and_names_the_next_move():
    """Silence here would leave the user with an unusable run and no idea
    what to change."""
    out = _cap([
        {"name": "reasoning_effort=minimal",
         "extra": {"reasoning_effort": "minimal"},
         "verdict": "ignored", "detail": "accepted, still no visible answer"},
    ])
    assert "none of the supplied candidates produced an answer" in out
    assert "--output-tokens" in out
    assert "wrong model for a budget this size" in out
    assert "--extra-body" not in out.split("none of the supplied")[1]


def test_the_first_working_lever_wins_when_several_do():
    """Candidate order is user-controlled, so the first working one wins."""
    out = _cap([
        {"name": "reasoning_effort=none",
         "extra": {"reasoning_effort": "none"}, "verdict": "works",
         "detail": "answered, finish stop, 109 tokens"},
        {"name": "reasoning_effort=low",
         "extra": {"reasoning_effort": "low"}, "verdict": "works",
         "detail": "answered, finish length, 512 tokens"},
    ])
    assert '{"reasoning_effort": "none"}' in out
    assert '{"reasoning_effort": "low"}' not in out.split("use this:")[1]


def test_an_errored_probe_does_not_break_the_report():
    out = _cap([{"name": "thinking.type=disabled",
                 "extra": {"thinking": {"type": "disabled"}},
                 "verdict": "error", "detail": "connection reset"}])
    assert "error" in out
    assert "none of the supplied candidates produced an answer" in out


def test_probe_argument_requires_a_finite_json_object():
    assert _json_object_arg('{"reasoning_effort":"none"}') == {
        "reasoning_effort": "none"}
    for value in ("[]", "null", '{"temperature": NaN}', "not-json",
                  '{"api_key":"sensitive-value"}'):
        with pytest.raises(argparse.ArgumentTypeError):
            _json_object_arg(value)


def test_probe_report_redacts_secret_values():
    secret = "sensitive-value-that-must-not-leak"
    out = _cap([{"name": "candidate 1 (api_key)",
                 "extra": {"api_key": secret},
                 "verdict": "works", "detail": "answered"}])
    assert secret not in out
    assert "<redacted>" in out


# ---- refusing a run we already know is void -----------------------------

class _Args:
    force = False


def test_it_refuses_and_hands_back_the_working_command():
    """Found by following our own guide as a new user. The preflight said the
    model could not answer, printed the exact flag that fixes it, then ran the
    full five minute test anyway and came back INVALID with 1,872 requests and
    zero readable answers."""
    import contextlib
    import io
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([{"name": "reasoning_effort=none",
                         "extra": {"reasoning_effort": "none"},
                         "verdict": "works", "detail": "answered"}], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "STOPPING before the load starts" in out
    assert """--extra-body '{"reasoning_effort": "none"}'""" in out
    assert "--force" in out


def test_when_nothing_works_it_refuses_and_says_what_to_change():
    import contextlib
    import io
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([{"name": "reasoning_effort=minimal",
                         "extra": {"reasoning_effort": "minimal"},
                         "verdict": "ignored", "detail": "no answer"}], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "no supplied reasoning-control candidate helped" in out
    assert "--output-tokens" in out
    assert "fits this output budget" in out


def test_when_nothing_was_probed_refusal_does_not_claim_a_probe_failed():
    from traffic_replay.cli import _refuse
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _refuse([], _Args())
    out = buf.getvalue()
    assert code == 3
    assert "no reasoning controls were probed" in out
    assert "--probe-extra-body" in out
    assert "no supplied reasoning-control candidate helped" not in out


def _no_answer_preflight():
    return {
        "attempted": 2,
        "reachable": 2,
        "readable": 0,
        "usage_reported": True,
        "cache_reported": True,
        "reasoning": True,
        "budgets": [40, 90],
        "budget": 90,
        "failed_probe_index": 1,
    }


def test_preflight_never_guesses_provider_controls(monkeypatch):
    from traffic_replay.cli import _check_preflight
    args = argparse.Namespace(force=False, probe_extra_body=[])

    monkeypatch.setattr("traffic_replay.cli._preflight",
                        lambda _cfg: _no_answer_preflight())

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("no control candidate was authorized")

    monkeypatch.setattr("traffic_replay.cli._probe_reasoning_levers",
                        unexpected_probe)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _check_preflight({}, args)
    assert code == 3
    assert "no provider controls were guessed" in buf.getvalue()


def test_preflight_probes_only_the_explicit_candidates(monkeypatch):
    from traffic_replay.cli import _check_preflight

    candidate = {"reasoning_effort": "none"}
    args = argparse.Namespace(force=False, probe_extra_body=[candidate])

    seen = {}
    monkeypatch.setattr("traffic_replay.cli._preflight",
                        lambda _cfg: _no_answer_preflight())

    def probe(_cfg, budget, candidates, probe_index):
        seen.update(budget=budget, candidates=candidates,
                    probe_index=probe_index)
        return [{"name": "candidate 1", "extra": candidate,
                 "verdict": "works", "detail": "answered"}]

    monkeypatch.setattr("traffic_replay.cli._probe_reasoning_levers", probe)
    with contextlib.redirect_stdout(io.StringIO()):
        code = _check_preflight({}, args)
    assert code == 3
    assert seen == {"budget": 90, "candidates": [candidate],
                    "probe_index": 1}
