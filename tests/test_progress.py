"""The live status line.

A five minute run printed its setup lines and then went silent until the
report was written, so a run where every request came back 401 looked
exactly like a healthy one until it finished.
"""
from __future__ import annotations

import io

from traffic_replay.progress import Progress


class _Res:
    def __init__(self, ok=True, ttft_ms=100.0):
        self.ok = ok
        self.ttft_ms = ttft_ms


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_in_flight_is_dispatched_minus_completed():
    """The gauge that says whether the endpoint is keeping up. If it climbs
    and keeps climbing, the run has already given its answer."""
    p = Progress(total=10, duration_s=60, stream=io.StringIO())
    for _ in range(5):
        p.sent()
    assert p.in_flight == 5
    p.done(_Res())
    p.done(_Res())
    assert p.in_flight == 3
    assert p.completed == 2


def test_errors_are_counted_separately_from_completions():
    p = Progress(total=10, duration_s=60, stream=io.StringIO())
    for _ in range(4):
        p.sent()
    p.done(_Res(ok=True))
    p.done(_Res(ok=False))
    p.done(_Res(ok=False))
    assert p.completed == 3
    assert p.errors == 2
    assert p.in_flight == 1


def test_the_rolling_window_forgets_old_samples():
    """The percentile has to move when the endpoint moves. Over the whole
    run it would be anchored by history and would barely respond."""
    p = Progress(total=10, duration_s=60, stream=io.StringIO())
    p.done(_Res(ttft_ms=100.0))
    # a sample older than the window is dropped rather than averaged in
    p._recent[0] = (p._recent[0][0] - 3600.0, 100.0)
    p.done(_Res(ttft_ms=900.0))
    p50, _ = p._rolling()
    assert p50 == 900.0


def test_a_non_tty_gets_plain_lines_not_carriage_returns():
    """A carriage-return animation in a CI log is unreadable."""
    buf = io.StringIO()
    p = Progress(total=10, duration_s=60, stream=buf)
    p.sent()
    p.paint(force=True)
    out = buf.getvalue()
    assert "\r" not in out
    assert "\033[K" not in out
    assert out.endswith("\n")
    assert "in flight 1" in out


def test_a_tty_rewrites_one_line_in_place():
    buf = _Tty()
    p = Progress(total=10, duration_s=60, stream=buf)
    p.sent()
    p.paint(force=True)
    p.paint(force=True)
    out = buf.getvalue()
    assert out.count("\r") == 2, "each paint rewrites rather than appending"
    p.finish()
    assert buf.getvalue().endswith("\n"), "must not leave the cursor mid-line"


def test_quiet_writes_nothing_at_all():
    buf = io.StringIO()
    p = Progress(total=10, duration_s=60, stream=buf, enabled=False)
    p.sent()
    p.done(_Res())
    p.paint(force=True)
    p.finish()
    assert buf.getvalue() == ""
    # counters still work, they are just not shown
    assert p.completed == 1


def test_painting_is_rate_limited_so_it_cannot_flood_a_log():
    buf = io.StringIO()
    p = Progress(total=1000, duration_s=60, stream=buf)
    for _ in range(500):
        p.sent()
        p.paint()
    assert buf.getvalue().count("\n") <= 2, "unforced paints must be throttled"


def test_the_line_survives_a_result_with_no_ttft():
    """A failed request has no TTFT and must not break the counter."""
    p = Progress(total=10, duration_s=60, stream=io.StringIO())
    p.sent()
    p.done(_Res(ok=False, ttft_ms=None))
    assert p.errors == 1
    assert p._rolling() == (None, None)
