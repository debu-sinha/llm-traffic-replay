"""Finding the control that turns reasoning down on THIS endpoint.

Every vendor spells it differently and several accept a flag and then
ignore it, so generic advice is not actionable. Measured on two Databricks
endpoints on the same day: GLM-5.2 accepts reasoning_effort=none, and Kimi
K2.7 rejects that exact value with "it is a thinking-only model" and needs
minimal, which on a 10k-token prompt is still not enough.
"""
from __future__ import annotations

import contextlib
import io

from traffic_replay.cli import _print_lever_report


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
         "detail": 'http 400: reasoning_effort="none" is not supported by '
                   'kimi-k2-7: it is a thinking-only model'},
    ])
    assert "rejected" in out
    assert "thinking-only model" in out


def test_when_nothing_works_it_says_so_and_names_the_next_move():
    """Silence here would leave the user with an unusable run and no idea
    what to change."""
    out = _cap([
        {"name": "reasoning_effort=minimal",
         "extra": {"reasoning_effort": "minimal"},
         "verdict": "ignored", "detail": "accepted, still no visible answer"},
    ])
    assert "none of them produced an answer" in out
    assert "--output-tokens" in out
    assert "wrong model for a budget this size" in out
    assert "--extra-body" not in out.split("none of them")[1]


def test_the_first_working_lever_wins_when_several_do():
    """They are ordered least-reasoning-first, so the first hit is the one
    that leaves the most budget for the answer."""
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
    assert "none of them produced an answer" in out


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
    assert "no reasoning control helped" in out
    assert "--output-tokens" in out
    assert "wrong model for an output budget this size" in out
