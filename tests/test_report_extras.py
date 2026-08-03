"""Small-N gate, drift-over-time, network floor (connect), and endpoint
metadata in the report. These are the confidence features: they make a short
or misleading run say so, and they record what was actually tested."""
from __future__ import annotations

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
