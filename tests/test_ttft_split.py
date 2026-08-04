"""TTFT split: reasoning-channel deltas (ttfr) are distinguished from the
first visible content delta (ttfv); ttft keeps first-of-either meaning; the
SLA scorecard scores whichever ttft_definition the run configures."""
import json
import tempfile
import threading
import time
from pathlib import Path

from traffic_replay.sse import StreamState, parse_sse_line, update_state
from traffic_replay.metrics import summarize
from traffic_replay.mock_server import serve
from traffic_replay.runner import RunConfig, run


# ---------- sse: reasoning vs visible ordering ----------
def _ev(js):
    return parse_sse_line("data: " + js)


def test_reasoning_delta_sets_reasoning_not_visible():
    st = StreamState()
    fired = update_state(st, _ev('{"choices":[{"delta":'
                                 '{"role":"assistant","reasoning_content":"hm"}}]}'))
    assert fired is True                      # first content of either kind
    assert st.saw_first_reasoning is True
    assert st.saw_first_visible is False
    assert st.content_chunks == 1


def test_reasoning_then_visible_ordering():
    st = StreamState()
    update_state(st, _ev('{"choices":[{"delta":{"reasoning_content":"a"}}]}'))
    update_state(st, _ev('{"choices":[{"delta":{"reasoning_content":"b"}}]}'))
    assert st.saw_first_reasoning and not st.saw_first_visible
    fired = update_state(st, _ev('{"choices":[{"delta":{"content":"X"}}]}'))
    assert fired is False                     # first-of-either already happened
    assert st.saw_first_visible is True
    assert st.content_chunks == 3


def test_visible_only_never_marks_reasoning():
    st = StreamState()
    update_state(st, _ev('{"choices":[{"delta":{"content":"X"}}]}'))
    assert st.saw_first_visible and not st.saw_first_reasoning


# ---------- metrics: scorecard follows ttft_definition ----------
def _row(i, ttft, ttfv, ttfr):
    return {"request_id": f"r{i}", "phase": "replay", "ok": True,
            "ttft_ms": ttft, "ttfr_ms": ttfr, "ttfv_ms": ttfv,
            "ttfb_ms": ttft - 2, "e2e_ms": ttfv + 500,
            "interchunk_max_ms": 4.0, "dispatch_lag_ms": 1.0,
            "t_send_unix": 1000.0 + i, "prompt_tokens": 1000,
            "completion_tokens": 40, "cached_tokens": None,
            "cached_tokens_source": None, "intended_input_tokens": 1000,
            "intended_output_tokens": 40, "intended_cache_fraction": 0.5,
            "content_chunks": 40, "finish_reason": "stop", "status": 200,
            "error": None, "doc_id": 1, "chars_sent": 4000, "retries": 0}


def test_scorecard_scores_configured_definition():
    # ttft (any) 100ms passes a 300ms target; ttfv (visible) 400ms fails it
    rows = [_row(i, ttft=100.0, ttfv=400.0, ttfr=100.0) for i in range(50)]
    accept = {"ttft_ms": {"p50": 300}}
    sc = summarize(rows, acceptance=accept, ttft_definition="first_content")
    sv = summarize(rows, acceptance=accept, ttft_definition="first_visible")
    rc = sc["sla"]["ttft_vs_target"][0]
    rv = sv["sla"]["ttft_vs_target"][0]
    assert rc["actual_ms"] == 100.0 and rc["met"] is True
    assert rv["actual_ms"] == 400.0 and rv["met"] is False
    assert sc["sla"]["ttft_definition"] == "first_content"
    assert sv["sla"]["ttft_definition"] == "first_visible"
    assert "ttfr_ms" in sc and "ttfv_ms" in sc


# ---------- e2e: reasoning stream through the real client + mock ----------
def test_reasoning_split_end_to_end():
    wd = Path(tempfile.mkdtemp(prefix="ttft-"))
    port = 8893
    srv = serve(port, wd / "truth.jsonl", reasoning_tokens=5,
                per_token_ms=3.0, ttft_base_ms=25.0, ms_per_1k_uncached=5.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    prof = wd / "prof.json"
    prof.write_text(json.dumps({
        "name": "reasoning_test",
        "input_tokens": {"p50": 800, "p95": 2000},
        "output_tokens": {"p50": 16, "p95": 24},
        "cache_fraction": {"p50": 0.30, "p95": 0.60},
        "acceptance_targets": {"ttft_ms": {"p50": 100000, "p95": 100000}},
    }))
    try:
        rc = RunConfig(
            profile_path=str(prof),
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "NO_TOKEN"},
            duration_s=8, qps_base=4.0, qps_burst=8.0, qps_min=1.0,
            qps_max=12.0, max_concurrency=16, cpt=4.0, calibrate_n=6,
            out_dir=str(wd / "out"), title="reasoning e2e", label="MOCK",
            max_output_tokens_cap=12, ttft_definition="first_visible")
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()
    s = out["summary"]
    assert "ttfr_ms" in s and "ttfv_ms" in s
    assert s["ttfr_ms"]["p50"] < s["ttfv_ms"]["p50"], \
        f"ttfr {s['ttfr_ms']['p50']} not < ttfv {s['ttfv_ms']['p50']}"
    scored = {r["quantile"]: r["actual_ms"] for r in s["sla"]["ttft_vs_target"]}
    assert abs(scored["p50"] - s["ttfv_ms"]["p50"]) < 0.6   # scored the ttfv table
    report = (Path(out["out_dir"]) / "report.md").read_text()
    assert "reasoning model detected" in report


# ---- the real client path, on a stream that never produces an answer -----
def test_a_reasoning_only_stream_is_not_counted_as_a_successful_answer():
    """End to end through the real client, not hand-written rows.

    The mock emits the reasoning channel and then stops on "length" with no
    visible delta, which is exactly what a reasoning model does when the
    token budget runs out mid-thought. Every request returns HTTP 200 with a
    well formed stream and a finish reason.

    This exists because every other test of these fields builds the row dict
    by hand. If the saw_first_visible derivation in sse.py or the
    stream_complete derivation in client.py drifts, those tests all still
    pass and this one does not.
    """
    wd = Path(tempfile.mkdtemp(prefix="reasononly-"))
    port = 8894
    srv = serve(port, wd / "truth.jsonl", reasoning_tokens=6, reasoning_only=1,
                per_token_ms=3.0, ttft_base_ms=25.0, ms_per_1k_uncached=5.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    prof = wd / "prof.json"
    prof.write_text(json.dumps({
        "name": "reasoning_only_test",
        "input_tokens": {"p50": 800, "p95": 2000},
        "output_tokens": {"p50": 16, "p95": 24},
        "cache_fraction": {"p50": 0.30, "p95": 0.60},
    }))
    try:
        rc = RunConfig(
            profile_path=str(prof),
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "NO_TOKEN"},
            duration_s=6, qps_base=4.0, qps_burst=8.0, qps_min=1.0,
            qps_max=12.0, max_concurrency=16, cpt=4.0, calibrate_n=4,
            out_dir=str(wd / "out"), title="reasoning only", label="MOCK",
            max_output_tokens_cap=12,
            acceptance_targets={"ttft_ms": {"p50": 100000},
                                "success_rate": 0.99})
        out = run(rc, quiet=True)
    finally:
        srv.shutdown()

    rows = [json.loads(x) for x in
            (Path(out["out_dir"]) / "requests.jsonl").read_text().splitlines()]
    replay = [r for r in rows if r.get("phase") == "replay"]
    assert replay, "no replay rows"

    # the transport was fine on every one of them
    assert all(r["ok"] for r in replay)
    assert all(r["status"] == 200 for r in replay)
    # and the client derived the answer facts correctly from the real stream
    assert all(r["stream_complete"] for r in replay)
    assert all(r["reasoning_seen"] for r in replay)
    assert not any(r["visible_content_seen"] for r in replay)
    assert all(r["truncated"] for r in replay)
    assert all(r["parse_errors"] == 0 for r in replay)

    s = out["summary"]
    a = s["answers"]
    assert a["complete_answers"] == 0
    assert a["no_visible_content"] == len(replay)
    assert a["stream_incomplete"] == 0, "the streams DID terminate cleanly"
    assert "invalid" in a
    assert s["sla"]["success_rate"]["met"] is False

    md = (Path(out["out_dir"]) / "report.md").read_text()
    assert "verdict: INVALID" in md
    html = (Path(out["out_dir"]) / "report.html").read_text()
    assert "Meets every acceptance target" not in html
