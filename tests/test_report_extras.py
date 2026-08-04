"""Small-N gate, drift-over-time, network floor (connect), and endpoint
metadata in the report. These are the confidence features: they make a short
or misleading run say so, and they record what was actually tested."""
from __future__ import annotations

import random

from traffic_replay.metrics import (_drift_block, render_html,
                                    render_markdown, summarize)


def _rows(n, base_ttft=100.0, t0=0.0, dt=1.0):
    return [{"ok": True, "t_send_unix": t0 + i * dt, "ttft_ms": base_ttft,
             "ttfb_ms": 1.0, "e2e_ms": base_ttft * 2, "connect_ms": 8.0,
             "dispatch_lag_ms": 0.0, "prompt_tokens": 100,
             "completion_tokens": 10} for i in range(n)]


def test_small_n_warning_thresholds():
    assert "very small" in summarize(_rows(10))["sample"]["warning"]
    assert "small sample" in summarize(_rows(50))["sample"]["warning"]
    assert summarize(_rows(150))["sample"]["warning"] is None


def test_drift_flag_rises_with_a_rising_tail():
    # window 0 (0-60s) fast, window 2 (120-180s) slow -> drift
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    late = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(early + late)
    assert len(d["windows"]) >= 2
    assert d["drift_flag"] is True
    assert d["ttft_p95_drift_ratio"] > 1.3


def test_drift_needs_two_windows():
    d = _drift_block(_rows(30, t0=0.0, dt=1.0))  # all within 60s
    assert d["windows"] == []
    assert "two" in d["note"]


def test_connect_and_endpoint_render_in_html():
    s = summarize(_rows(120), run_meta={
        "input_mode": "profile", "endpoint_path": "/e",
        "endpoint_metadata": {"name": "acme-glm-prod-42", "task": "llm/v1/chat",
                              "route_optimized": True, "ready": "READY",
                              "served_entities": [{"name": "e",
                                                   "workload_type": "GPU_LARGE"}]}})
    h = render_html(s, "extras")
    assert "Connection setup" in h              # connect line
    assert "excluded" in h                      # states it is not in TTFT
    assert "8" in h                             # connect ms value
    assert "Endpoint under test" in h           # endpoint metadata card
    assert "acme-glm-prod-42" in h            # custom name shown
    assert "GPU_LARGE" in h                     # served entity workload


def test_stability_card_present_for_long_run():
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    late = _rows(25, base_ttft=110.0, t0=140.0, dt=1.0)
    h = render_html(summarize(early + late), "stability")
    assert "Stability over time" in h


def test_warmup_is_not_reported_as_stable():
    """A cold endpoint: window 0 is 15x slower than the last window
    because the endpoint was cold. Comparing only first to last calls that
    an improvement and passes it as stable, which would let a caller quote a
    blended p95 from a run that never reached steady state."""
    cold = _rows(25, base_ttft=31000.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=3500.0, t0=70.0, dt=1.0)
    warm = _rows(25, base_ttft=2000.0, t0=140.0, dt=1.0)
    d = _drift_block(cold + mid + warm)
    assert d["drift_flag"] is True
    assert d["drift_kind"] == "warming"
    assert d["ttft_p95_spread_ratio"] > 1.3
    assert d["ttft_p95_drift_ratio"] < 1.0      # end/end alone looks like a win
    assert "cold start" in d["drift_headline"]


def test_midrun_spike_is_not_reported_as_stable():
    """Ends match, middle is 10x worse. first/last ratio is ~1.0 here, so only
    a worst-to-best spread catches it."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    spike = _rows(25, base_ttft=1000.0, t0=70.0, dt=1.0)
    b = _rows(25, base_ttft=100.0, t0=140.0, dt=1.0)
    d = _drift_block(a + spike + b)
    assert len(d["windows"]) >= 3
    assert 0.9 < d["ttft_p95_drift_ratio"] < 1.1   # endpoints agree
    assert d["drift_flag"] is True                 # but the run is not stable
    assert d["drift_kind"] == "spike"


def test_genuinely_steady_run_stays_stable():
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=105.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=110.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_flag"] is False
    assert d["drift_kind"] == "stable"


def test_degrading_run_is_labeled_degrading():
    early = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=200.0, t0=70.0, dt=1.0)
    late = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(early + mid + late)
    assert d["drift_kind"] == "degrading"
    assert "slower" in d["drift_headline"]


def test_unstable_run_says_so_in_html():
    cold = _rows(25, base_ttft=31000.0, t0=0.0, dt=1.0)
    mid = _rows(25, base_ttft=3500.0, t0=70.0, dt=1.0)
    warm = _rows(25, base_ttft=2000.0, t0=140.0, dt=1.0)
    h = render_html(summarize(cold + mid + warm), "warmup")
    assert "unstable" in h
    assert "stable</span>" not in h.replace("unstable", "")


def test_noisy_run_is_variable_not_degrading():
    """Real warm-endpoint shape: p95 dips then rises, ending near where it
    started. The max lands in the last window, but the windows do not move one
    way, so calling it degradation overstates the data. It is noise, and the
    number still should not be quoted as steady state."""
    a = _rows(25, base_ttft=1900.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=1300.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=2200.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_flag"] is True          # not steady, so still flagged
    assert d["drift_kind"] == "variable"    # but no trend is claimed
    assert "noisy" in d["drift_headline"]


def test_degrading_requires_every_window_to_rise():
    """A run that rises overall but dips in the middle is not a clean trend."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=50.0, t0=70.0, dt=1.0)
    c = _rows(25, base_ttft=400.0, t0=140.0, dt=1.0)
    d = _drift_block(a + b + c)
    assert d["drift_kind"] == "variable"


def test_prompts_mode_warns_when_prompts_are_recycled():
    """A small prompt set cycled over a long run means most requests are
    verbatim repeats, which the endpoint prompt cache serves. The achieved
    cache fraction then describes the replay, not production traffic, so the
    report has to say so."""
    meta = {"input_mode": "prompts", "endpoint_path": "/e",
            "prompts_file": "p.jsonl", "prompts_count": 10}
    s = summarize(_rows(100), run_meta=meta)
    r = s["replay"]
    assert r["distinct_prompts"] == 10
    assert r["avg_sends_per_prompt"] == 10
    assert "prompt cache" in r["warning"]
    assert "CAUTION (prompt replay)" in render_markdown(s, "replay")
    assert "banner warn" in render_html(s, "replay")


def test_prompts_mode_quiet_when_every_prompt_is_sent_once():
    meta = {"input_mode": "prompts", "endpoint_path": "/e",
            "prompts_file": "p.jsonl", "prompts_count": 120}
    s = summarize(_rows(100), run_meta=meta)
    assert s["replay"]["warning"] is None


def test_profile_mode_has_no_replay_block():
    s = summarize(_rows(100), run_meta={"input_mode": "profile",
                                        "endpoint_path": "/e"})
    assert "replay" not in s


def test_tiny_trailing_window_cannot_manufacture_a_verdict():
    """A run whose duration is not a multiple of the window leaves a partial
    trailing window. One slow request in it must not become a trend: a p95
    over a handful of requests is one outlier away from inventing one."""
    steady = _rows(400, base_ttft=1000.0, t0=0.0, dt=0.3)     # windows 0 and 1
    tail = _rows(1, base_ttft=4000.0, t0=125.0)               # window 2, n=1
    d = _drift_block(steady + tail)
    assert d["windows"][-1]["n"] == 1
    assert d["windows"][-1]["counted"] is False
    assert d["skipped_windows"] == 1
    assert d["drift_kind"] == "stable"       # not "degrading"
    assert d["drift_flag"] is False


def test_two_windows_cannot_name_a_direction():
    """Two points separate nothing. The run is still flagged unstable, but no
    trend is claimed off it."""
    a = _rows(25, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(25, base_ttft=400.0, t0=70.0, dt=1.0)
    d = _drift_block(a + b)
    assert d["drift_flag"] is True
    assert d["drift_kind"] == "variable"
    assert "not enough to call a direction" in d["drift_headline"]


def test_no_usable_window_says_so_instead_of_stable():
    """Every window too small to count. The report must not print a stable
    verdict it has no data for."""
    a = _rows(3, base_ttft=100.0, t0=0.0, dt=1.0)
    b = _rows(3, base_ttft=9000.0, t0=70.0, dt=1.0)
    d = _drift_block(a + b)
    assert "drift_kind" not in d
    assert "cannot be judged" in d["note"]
    h = render_html(summarize(a + b), "nodata")
    assert "not enough data" in h
    assert "pill ok'>stable" not in h


def test_windows_with_no_ttft_are_not_counted():
    """A window whose requests all failed to produce a TTFT has p95 None. It
    must not be compared by value against the real windows."""
    good = _rows(25, base_ttft=1000.0, t0=0.0, dt=1.0)
    blind = [dict(r, ttft_ms=None) for r in _rows(25, t0=70.0, dt=1.0)]
    later = _rows(25, base_ttft=5000.0, t0=140.0, dt=1.0)
    d = _drift_block(good + blind + later)
    assert d["windows"][1]["ttft_p95"] is None
    assert d["windows"][1]["counted"] is False
    assert d["drift_kind"] == "variable"     # 2 counted windows, no direction


def test_report_states_which_harness_version_and_latency_basis():
    """A 0.2.x TTFT included connection setup and a 0.3.x TTFT does not, so a
    report has to say which it is before anyone puts two in one column."""
    s = summarize(_rows(120))
    assert s["harness_version"].startswith("0.3")
    assert "NOT included" in s["latency_basis"]
    assert "latency basis" in render_markdown(s, "v")
    assert "Latency basis" in render_html(s, "v")


def _fail(n, t0=0.0, dt=1.0):
    return [{"ok": False, "t_send_unix": t0 + i * dt, "ttft_ms": None,
             "e2e_ms": None, "error": "upstream timeout", "status": 504}
            for i in range(n)]


def test_endpoint_collapsing_into_errors_is_not_stable():
    """The breaking-point run PRODUCTION_TESTING stage 2 tells you to do. The
    endpoint falls over in the last window, most requests fail, and the few
    survivors come back fast. Scoring successes alone reads that as steady,
    which is the worst possible answer for a test whose whole purpose is
    finding where the endpoint bends."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)   # fast survivors
    rows += _fail(140, t0=140.0, dt=0.3)                   # the collapse
    d = _drift_block([r for r in rows if r["ok"]],
                     [r for r in rows if not r["ok"]])
    assert d["drift_kind"] == "failing"
    assert d["drift_flag"] is True
    assert "84 percent" in d["drift_headline"]
    assert "not what it was asked" in d["drift_headline"]
    # the named window is the biggest failure, so the clause reconciling it
    # against the highest RATE has to be there too, or the two disagree
    assert "highest loss rate was window 3" in d["drift_headline"]


def test_a_collapsing_window_is_judged_for_errors_not_for_latency():
    """The window where the endpoint broke has few SUCCESSES. It must still
    reach the error verdict, which is sized on ATTEMPTS, while staying out of
    the latency comparison, whose p95 would be survivors only."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)
    fails = _fail(140, t0=140.0, dt=0.3)
    d = _drift_block(rows, fails)
    collapsed = [w for w in d["windows"] if w["window"] == 2][0]
    assert collapsed["n"] == 25              # few successes
    assert collapsed["errors"] == 134
    assert collapsed["error_counted"] is True   # reaches the error verdict
    assert collapsed["counted"] is False        # excluded from latency


def test_per_window_errors_render_in_both_formats():
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.5)
    rows += _rows(60, base_ttft=205.0, t0=70.0, dt=0.5)
    fails = _fail(40, t0=70.0, dt=0.5)
    s = summarize(rows + fails)
    md = render_markdown(s, "errs")
    h = render_html(s, "errs")
    assert "errors" in md
    assert "<th>errors</th>" in h
    assert "40 (" in md          # count and share shown together


def test_a_uniformly_lossy_run_is_not_called_failing():
    """Steady 8 percent errors across every window is a bad endpoint, but it
    is not a breaking point, and the error rate is already reported. Only a
    window that is materially worse than the rest earns the failing verdict."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(60, base_ttft=200.0 + w, t0=t0, dt=0.5)
        fails += _fail(5, t0=t0, dt=0.5)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] != "failing"


def test_a_total_outage_window_is_not_dropped_for_having_no_p95():
    """The window where every request failed has no p95 at all. Gating the
    error verdict on the latency gate would make a total outage invisible,
    which is worse than the partial-collapse bug."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=205.0, t0=140.0, dt=0.3)
    fails = _fail(150, t0=70.0, dt=0.3)
    d = _drift_block(rows, fails)
    dead = [w for w in d["windows"] if w["n"] == 0][0]
    assert dead["errors"] == 150
    assert dead["ttft_p95"] is None
    assert d["drift_kind"] == "failing"


def test_a_run_failing_in_every_window_is_still_failing():
    """Past the knee, every window sheds requests, so worst and best error
    rates are both high and a delta test alone cannot see it."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(70, base_ttft=200.0 + w, t0=t0, dt=0.3)
        fails += _fail(30, t0=t0, dt=0.3)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"


def test_a_shedding_window_cannot_anchor_the_latency_spread():
    """The collapsed window's survivors are fast, so letting it into the
    latency comparison makes the fastest number in the table the one the
    endpoint produced while falling over."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(150, base_ttft=210.0, t0=70.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=140.0, dt=0.3)   # fast survivors
    fails = _fail(140, t0=140.0, dt=0.3)
    d = _drift_block(rows, fails)
    collapsed = [w for w in d["windows"] if w["errors"] == 134][0]
    assert collapsed["p95_survivorship"] is True
    assert collapsed["counted"] is False
    # the failing branch returns before any latency comparison is computed,
    # so there is no "best" at all. this also fails loudly if the failing and
    # survivorship thresholds ever diverge enough for both to be reachable.
    assert "ttft_p95_best" not in d


def test_mild_uniform_loss_still_gets_a_latency_verdict():
    """Losing a few percent leaves a p95 worth comparing. Excluding those
    windows would silently drop the verdict on an otherwise healthy run."""
    rows, fails = [], []
    for w, t0 in enumerate((0.0, 70.0, 140.0)):
        rows += _rows(60, base_ttft=200.0 + w, t0=t0, dt=0.3)
        fails += _fail(5, t0=t0, dt=0.3)
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "stable"
    assert all(w["counted"] for w in d["windows"])


def test_a_heavily_shedding_small_window_is_not_sized_out():
    """A breaking-point run ends in a trailing partial window. Sizing the
    error rule purely on median attempts would drop exactly the window the
    run exists to find."""
    rows = _rows(200, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(200, base_ttft=201.0, t0=70.0, dt=0.2)
    rows += _rows(200, base_ttft=202.0, t0=140.0, dt=0.2)
    rows += _rows(30, base_ttft=203.0, t0=210.0, dt=0.2)
    fails = _fail(15, t0=216.0, dt=0.2)          # 33 percent of a small window
    d = _drift_block(rows, fails)
    small = d["windows"][-1]
    assert small["attempts"] < 60                 # well under the median
    assert small["error_counted"] is True         # judged anyway
    assert d["drift_kind"] == "failing"


def test_a_run_where_everything_failed_says_so():
    """Zero successes must not fall through to 'stability was never
    established'. It is the most complete failure there is."""
    d = _drift_block([], _fail(50, t0=0.0) + _fail(50, t0=70.0))
    assert d["drift_kind"] == "failing"
    assert "every request failed" in d["drift_headline"]


def test_the_named_window_is_the_largest_failure_not_the_highest_rate():
    """A tiny tail window at 100 percent should not outrank the window where
    a hundred requests actually died."""
    rows = _rows(150, base_ttft=200.0, t0=0.0, dt=0.3)
    rows += _rows(25, base_ttft=190.0, t0=70.0, dt=0.3)
    fails = _fail(120, t0=70.0, dt=0.3)      # big collapse, 83 percent
    fails += _fail(4, t0=140.0, dt=0.3)      # tiny tail, 100 percent
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"
    assert "window 1" in d["drift_headline"]      # the substantive one
    assert "100 percent" not in d["drift_headline"]


def test_retry_exhausted_failures_keep_their_original_send_time():
    """The client stamps the FIRST send, not the moment of final failure. A
    request retried past a read timeout would otherwise land whole windows
    later and invent a trailing window of errors."""
    import time
    from traffic_replay.client import EndpointClient, EndpointConfig

    class SlowFailingConn:
        """Connects, accepts the request, then dies. Each attempt burns time,
        the way a read timeout does."""
        sock = None

        def connect(self): pass

        def request(self, *a, **k):
            time.sleep(0.15)
            raise OSError("connection reset by peer")

        def close(self): pass

    cfg = EndpointConfig(base_url="http://127.0.0.1:1",
                         path="/serving-endpoints/x/invocations",
                         max_retries=2)
    c = EndpointClient(cfg, token=None)
    c._connect = lambda: SlowFailingConn()

    before = time.time()
    r = c.send([{"role": "user", "content": "hi"}], 8, "req-1",
               scheduled_s=0.0, dispatch_lag_ms=0.0, intended=(0, 0, None, 0),
               chars_sent=2)
    after = time.time()

    assert r.ok is False
    # the whole call spanned at least two sleeps, so a final-failure stamp
    # would sit well after the first send
    assert after - before > 0.25
    assert r.t_send_unix < before + 0.15


def test_a_total_outage_actually_renders_its_verdict():
    """The zero-success block reaches summary.json, but both renderers used
    to gate on the window list, which is empty there, so the card printed no
    verdict at all while compare warned about the same run."""
    fails = [{"ok": False, "t_send_unix": float(i), "ttft_ms": None,
              "e2e_ms": None, "error": "upstream refused", "status": 503}
             for i in range(120)]
    s = summarize(fails)
    assert s["drift"]["drift_kind"] == "failing"
    md = render_markdown(s, "outage")
    h = render_html(s, "outage")
    assert "failing" in md.lower()
    assert "unstable: failing" in h
    assert "every request failed" in md


def test_one_stray_failure_does_not_flip_a_healthy_run():
    """A run whose duration is not a multiple of the window leaves a tiny
    tail. At low rates it holds a couple of requests, and one reset there
    must not read as a breaking point."""
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(60, base_ttft=201.0, t0=70.0, dt=0.2)
    d = _drift_block(rows, _fail(1, t0=125.0))
    assert d["drift_kind"] != "failing"


def test_the_headline_window_always_trips_the_bar_itself():
    """Naming by absolute errors alone names the huge low-rate window, whose
    3 percent is a rounding error next to a 30 percent collapse, and whose
    rate can round to 0 percent on a bigger denominator."""
    rows = _rows(2000, base_ttft=200.0, t0=0.0, dt=0.02)     # big, clean-ish
    rows += _rows(70, base_ttft=201.0, t0=70.0, dt=0.2)
    fails = _fail(60, t0=0.0, dt=0.02)                       # 3 percent
    fails += _fail(30, t0=84.0, dt=0.2)                      # 30 percent
    d = _drift_block(rows, fails)
    assert d["drift_kind"] == "failing"
    # the eligibility filter is what this pins: without it the argmax by
    # absolute errors names the big low-rate window instead.
    assert d["drift_headline"].startswith("window 1 failed 30 percent")
    assert "failed 0 percent" not in d["drift_headline"]


def test_a_measured_zero_dispatch_lag_prints_as_zero_not_nan():
    """A measured 0.0 is a real value. Collapsing it with `or` would print
    nan on every clean run, which is what the first fix did."""
    md = render_markdown(summarize(_rows(60)), "lag")
    assert "dispatch lag p95 0 ms" in md
    assert "nan" not in md


def test_the_window_table_is_a_real_markdown_table():
    """A GFM table cannot interrupt a paragraph. Without a blank line the
    whole stability block renders as literal pipes, and report.md is the file
    that gets pasted into a ticket."""
    rows = _rows(60, base_ttft=200.0, t0=0.0, dt=0.2)
    rows += _rows(60, base_ttft=205.0, t0=70.0, dt=0.2)
    rows += _rows(60, base_ttft=210.0, t0=140.0, dt=0.2)
    md = render_markdown(summarize(rows), "tbl")
    block = md[md.index("stability over time"):].splitlines()
    header = next(i for i, l in enumerate(block) if l.startswith("| window |"))
    assert block[header - 1].strip() == ""      # blank line before the table


def test_a_total_outage_card_does_not_claim_per_window_p95():
    fails = [{"ok": False, "t_send_unix": float(i), "ttft_ms": None,
              "e2e_ms": None, "error": "refused", "status": 503}
             for i in range(60)]
    s = summarize(fails)
    assert "window p95 in ms" not in render_html(s, "o")
    assert "| window |" not in render_markdown(s, "o")


def _paced(n, offered_qps, service_s, pool, ttft=100.0, jitter=0.0):
    """Rows shaped like a run where the pool can only serve `pool` at a time
    and each request occupies a worker for `service_s`. Requests are stamped
    when a worker frees up, which is what an open-loop client against a
    saturated pool actually produces."""
    rnd = random.Random(7)
    rows, free = [], [0.0] * pool
    for i in range(n):
        want = i / offered_qps
        svc = service_s * (1.0 + rnd.uniform(0, jitter)) if jitter else service_s
        w = min(range(pool), key=lambda k: free[k])
        actual = max(want, free[w])
        free[w] = actual + svc
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + actual,
                     "ttft_ms": ttft, "ttfb_ms": 1.0, "e2e_ms": ttft * 2,
                     "connect_ms": 8.0,
                     # the dispatcher is fine, it just queues: this is the
                     # number that stays small while the client is drowning
                     "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    return rows


def test_a_saturated_pool_shows_up_as_wire_lateness_not_dispatch_lag():
    """ThreadPoolExecutor.submit() queues instead of blocking, so the
    dispatcher never notices a full pool. Measured on a real run: dispatch
    lag p95 of 5 ms while requests reached the endpoint 92 seconds late."""
    rows = _paced(240, offered_qps=8.0, service_s=1.0, pool=2)
    s = summarize(rows)
    arr = s["arrivals"]
    assert arr["dispatch_lag_ms"]["p95"] < 10           # dispatcher looks fine
    assert arr["wire_lateness_ms"]["p95"] > 10_000      # reality
    assert s["client"]["warning"] is not None
    # states the observation, not a cause it cannot know
    assert "did not reach the endpoint on schedule" in s["client"]["warning"]
    assert "read the stability card to tell them apart" in s["client"]["warning"]


def test_the_caution_is_above_the_tables_in_both_formats():
    rows = _paced(240, offered_qps=8.0, service_s=1.0, pool=2)
    s = summarize(rows)
    md = render_markdown(s, "sat")
    assert md.index("CAUTION (client saturation)") < md.index("| metric (ms) |")
    assert "banner warn" in render_html(s, "sat")


def test_a_client_that_keeps_up_is_not_warned():
    """The negative control. Verified against a real 20 rps run that the
    endpoint itself confirmed receiving at 20.7 rps: no caution."""
    rows = _paced(1200, offered_qps=20.0, service_s=0.06, pool=64)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_wire_lateness_is_reported_even_when_nothing_is_wrong():
    rows = _paced(600, offered_qps=20.0, service_s=0.06, pool=64)
    s = summarize(rows)
    md = render_markdown(s, "ok")
    assert "wire lateness p95" in md
    assert s["arrivals"]["wire_lateness_ms"]["n"] == 600


def test_a_rate_shortfall_alone_is_enough_to_warn():
    """Isolates the shortfall arm: sends stay close to schedule for most of
    the run, so p95 lateness stays under a second and the drifting arm cannot
    fire, but the run still takes far longer than it was asked to."""
    rows = []
    for i in range(400):
        want = i / 10.0
        # on time for 96 percent of the run, then a hard stall at the end
        actual = want if i < 384 else want + 40.0
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + actual, "ttft_ms": 100.0,
                     "ttfb_ms": 1.0, "e2e_ms": 200.0, "connect_ms": 8.0,
                     "dispatch_lag_ms": 4.0, "prompt_tokens": 100,
                     "completion_tokens": 10})
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000     # drifting silent
    assert s["client"]["achieved_qps"] < s["client"]["offered_qps"] * 0.8
    # states what the span statistic supports, not "never"
    assert "fewer requests per second than the" in s["client"]["warning"]


def test_a_late_but_complete_run_does_not_claim_a_shortfall():
    """The drifting arm alone. The run average held, so the total load did
    arrive, and saying it was never driven at the rate would contradict the
    achieved figure printed two keys away."""
    # a transient stall that recovers, which is the real shape this arm
    # exists for: total load arrives, but not when the schedule wanted it
    rows = []
    for i in range(600):
        want = i / 20.0
        late = 4.0 if 200 <= i < 320 else 0.0     # 20 percent of the run
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + want + late,
                     "ttft_ms": 100.0, "ttfb_ms": 1.0, "e2e_ms": 200.0,
                     "connect_ms": 8.0, "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    s = summarize(rows)
    c = s["client"]
    assert c["achieved_qps"] >= c["offered_qps"] * 0.8      # no shortfall
    assert "fewer requests per second" not in c["warning"]
    assert "arrived reshaped" in c["warning"]


def test_heavy_retries_are_not_reported_as_a_client_shortfall():
    """offered and achieved must come from one population. Mixing them makes
    the ratio the non-retry fraction, so an endpoint dropping connections
    would read as a slow client, which is backwards."""
    for frac in (0.2, 0.3, 0.5):
        rows = _paced(400, offered_qps=20.0, service_s=0.04, pool=64)
        for i, r in enumerate(rows):
            if i % int(1 / frac) == 0:
                r["retries"] = 1
        s = summarize(rows)
        assert "client" not in s, f"false shortfall at retry fraction {frac}"


def test_a_healthy_run_with_jittery_service_times_stays_silent():
    """The negative control with zero variance proves too little. Real service
    times are heavy tailed, and that is the shape most likely to produce a
    false positive against the 1s threshold."""
    rows = _paced(1200, offered_qps=20.0, service_s=0.06, pool=64, jitter=4.0)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_the_printed_rates_reconcile_with_the_arrival_bullet():
    """The caution's 'delivered' figure and the believability block's achieved
    arrival rate describe the same run, so they must not disagree because a
    chunk of rows retried in the middle."""
    rows = []
    for i in range(500):
        want = i / 20.0
        rows.append({"ok": True, "scheduled_s": want,
                     "t_send_unix": 1_000_000.0 + want * 1.6,
                     "ttft_ms": 100.0, "ttfb_ms": 1.0, "e2e_ms": 200.0,
                     "connect_ms": 8.0, "dispatch_lag_ms": 4.0,
                     "prompt_tokens": 100, "completion_tokens": 10})
    for r in rows[200:400]:
        r["retries"] = 1                    # 40 percent, mid-run
    s = summarize(rows)
    c = s["client"]
    assert c["offered_qps"] > 19.0          # the true offered rate, not 12
    bullet = s["arrivals"]["achieved_qps_overall"]
    assert abs(c["achieved_qps"] - bullet) / bullet < 0.15


def test_a_retried_row_is_timed_from_its_first_attempt():
    """t_send_unix belongs to whichever attempt produced the result, so on a
    retry it carries the endpoint's delay. first_send_unix says when the load
    was actually offered, and that is what client lateness must be built on.
    No row needs excluding once the honest stamp exists."""
    rows = _paced(200, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r["first_send_unix"] = r["t_send_unix"]
    # a request that failed, retried, then came back 120s later
    rows[10]["retries"] = 1
    rows[10]["t_send_unix"] += 120.0          # contaminated
    # first_send_unix left alone: it still says when the load went out
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["n"] == len(rows)   # nothing dropped
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000       # not blamed on the client
    assert "client" not in s


def test_every_retry_shape_is_timed_honestly():
    """The three client return paths (non-200, empty stream, exhausted) all
    carry first_send_unix, so none of them can inject endpoint delay into
    client lateness."""
    rows = _paced(300, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r["first_send_unix"] = r["t_send_unix"]
    for i, (status, ok) in enumerate([(503, False), (200, False), (None, False)]):
        r = rows[50 + i * 50]
        r["retries"] = 1
        r["status"] = status
        r["ok"] = ok
        r["t_send_unix"] += 130.0             # every one carries endpoint delay
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["p95"] < 1000
    assert "client" not in s


def test_rows_without_the_field_fall_back_to_t_send_unix():
    """A requests.jsonl written by an older harness has no first_send_unix.
    It should still produce a wire-lateness series rather than an empty one."""
    rows = _paced(120, offered_qps=20.0, service_s=0.04, pool=64)
    for r in rows:
        r.pop("first_send_unix", None)
    s = summarize(rows)
    assert s["arrivals"]["wire_lateness_ms"]["n"] == len(rows)


def test_the_client_stamps_first_send_on_every_return_path():
    """Drives the real EndpointClient rather than hand-built dicts, so
    deleting first_send_unix from any _finish call fails here. Covers the
    non-200 path and the exhausted-retry path."""
    import json as _json
    import threading
    import time as _time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from traffic_replay.client import EndpointClient, EndpointConfig

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = b'{"error":"nope"}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _time.sleep(0.2)
    try:
        cfg = EndpointConfig(base_url=f"http://127.0.0.1:{port}",
                             path="/serving-endpoints/x/invocations")
        c = EndpointClient(cfg, token=None)
        r = c.send([{"role": "user", "content": "hi"}], 8, "r1",
                   scheduled_s=0.0, dispatch_lag_ms=0.0,
                   intended=(0, 0, None, 0), chars_sent=2)
        assert r.ok is False and r.status == 503          # the non-200 path
        assert r.first_send_unix is not None
        # strictly earlier: the stamp is taken before the handshake, while
        # t_send_unix is taken after. equality means the call site dropped it
        # and _finish fell back to t_send_unix.
        assert r.first_send_unix < r.t_send_unix
    finally:
        srv.shutdown(); srv.server_close()

    # exhausted-retry path: nothing listening at all
    cfg2 = EndpointConfig(base_url="http://127.0.0.1:1",
                          path="/serving-endpoints/x/invocations",
                          max_retries=1)
    c2 = EndpointClient(cfg2, token=None)
    r2 = c2.send([{"role": "user", "content": "hi"}], 8, "r2",
                 scheduled_s=0.0, dispatch_lag_ms=0.0,
                 intended=(0, 0, None, 0), chars_sent=2)
    assert r2.ok is False
    assert r2.first_send_unix is not None
