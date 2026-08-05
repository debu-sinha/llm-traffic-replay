"""Live progress while a run is in flight.

A five minute run used to print its setup lines and then go silent until the
report was written. You could not tell a healthy run from one where every
request was coming back 401, which is a bad way to spend five minutes and a
worse way to spend the forty that a rate ladder takes.

Three numbers earn their place on the line:

  in flight   the most legible saturation signal there is. if it climbs and
              keeps climbing, the endpoint is not keeping up and the run has
              already told you its answer.
  errors      turns the line into a reason to stop at ten seconds instead of
              at five minutes.
  TTFT p50    over a short trailing window, not the whole run, so it moves
              when the endpoint moves rather than being anchored by history.

On a terminal the line is rewritten in place. Everywhere else, which means
CI, it prints one plain line at a slower cadence, because a carriage-return
animation in a log file is unreadable. Progress goes to stderr so a caller
can redirect the report on stdout without catching any of this.
"""

from __future__ import annotations

import collections
import sys
import threading
import time

_WINDOW_S = 30.0  # trailing window for the rolling percentiles
_TTY_EVERY = 0.25
_PLAIN_EVERY = 15.0


class Progress:
    """Counters a dispatcher and its worker threads can both touch."""

    def __init__(
        self, total: int, duration_s: float, stream=None, enabled: bool = True
    ):
        self.total = total
        self.duration_s = duration_s
        self.dispatched = 0
        self.completed = 0
        self.errors = 0
        self._recent: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._stream = stream if stream is not None else sys.stderr
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._enabled = enabled
        self._last_paint = 0.0
        self._painted = False
        self._t0 = time.monotonic()

    # ---- called from the dispatcher thread ----
    def sent(self) -> None:
        with self._lock:
            self.dispatched += 1

    # ---- called from worker threads, so keep it short ----
    def done(self, res) -> None:
        now = time.monotonic()
        ok = bool(getattr(res, "ok", False))
        ttft = getattr(res, "ttft_ms", None)
        with self._lock:
            self.completed += 1
            if not ok:
                self.errors += 1
            if ttft is not None:
                self._recent.append((now, ttft))
                cutoff = now - _WINDOW_S
                while self._recent and self._recent[0][0] < cutoff:
                    self._recent.popleft()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return max(0, self.dispatched - self.completed)

    def _rolling(self) -> tuple[float | None, float | None]:
        with self._lock:
            vals = sorted(v for _, v in self._recent)
        if not vals:
            return None, None
        hi = min(len(vals) - 1, int(len(vals) * 0.95))
        return vals[len(vals) // 2], vals[hi]

    def paint(self, force: bool = False) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        every = _TTY_EVERY if self._tty else _PLAIN_EVERY
        if not force and (now - self._last_paint) < every:
            return
        self._last_paint = now

        el = now - self._t0
        p50, p95 = self._rolling()
        lat = f"ttft {p50:.0f}/{p95:.0f}ms" if p50 is not None else "ttft --"
        err = f"{self.errors} err" if self.errors else "0 err"
        line = (
            f"  {el:5.0f}s/{self.duration_s:.0f}s  "
            f"sent {self.dispatched}/{self.total}  "
            f"done {self.completed}  "
            f"in flight {self.in_flight}  "
            f"{lat}  {err}"
        )
        if self._tty:
            self._stream.write("\r\033[K" + line)
            self._stream.flush()
            self._painted = True
        else:
            self._stream.write(line.strip() + "\n")
            self._stream.flush()

    def finish(self) -> None:
        if not self._enabled:
            return
        self.paint(force=True)
        if self._tty and self._painted:
            self._stream.write("\n")
            self._stream.flush()
