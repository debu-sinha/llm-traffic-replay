"""The one-command path an external user actually walks.

The value of `benchmark` is that someone with an endpoint URL and a rough
idea of their token sizes gets a correct report without authoring a profile
JSON, and gets stopped before spending five minutes producing a number that
would have been wrong.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from traffic_replay.cli import _pair, main


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="bench-"))


def test_a_single_number_becomes_a_p50_and_a_p95():
    p = _pair("10000", "input-tokens")
    assert p["p50"] == 10000
    assert p["p95"] > p["p50"]


def test_two_numbers_are_taken_as_given():
    assert _pair("10000,24000", "input-tokens") == {"p50": 10000, "p95": 24000}


def test_a_backwards_pair_is_refused():
    """p95 below p50 would fit a lognormal with negative sigma and silently
    produce nonsense sizes."""
    try:
        _pair("24000,10000", "input-tokens")
    except SystemExit as e:
        assert "p95 above p50" in str(e)
    else:
        raise AssertionError("should have refused")


def test_it_writes_a_profile_so_the_user_does_not_have_to():
    """The step this removes: hand-authoring a profile JSON before you can
    measure anything."""
    d = _tmp()
    os.environ["TR_BENCH_TOKEN"] = "not-a-real-token"
    try:
        main(["benchmark", "--host", "https://example.invalid",
              "--endpoint", "my-ep", "--token-env", "TR_BENCH_TOKEN",
              "--input-tokens", "8000,20000", "--output-tokens", "50,120",
              "--cache-hit-rate", "0.4,0.8",
              "--duration", "1", "--concurrency", "1",
              "--out-dir", str(d), "--skip-preflight"])
    except SystemExit:
        pass
    except Exception:
        pass          # the endpoint is unreachable on purpose
    finally:
        os.environ.pop("TR_BENCH_TOKEN", None)
    prof = json.loads((d / "profile.json").read_text())
    assert prof["input_tokens"] == {"p50": 8000, "p95": 20000}
    assert prof["output_tokens"] == {"p50": 50, "p95": 120}
    assert prof["cache_fraction"] == {"p50": 0.4, "p95": 0.8}
    # and it says where the numbers came from, so nobody quotes them as
    # measured traffic
    assert "not measured" in prof["provenance"]


def test_the_saved_config_reruns_the_same_experiment():
    """Reproducibility: the exact config is written next to the results."""
    d = _tmp()
    os.environ["TR_BENCH_TOKEN"] = "not-a-real-token"
    try:
        main(["benchmark", "--host", "https://example.invalid",
              "--endpoint", "my-ep", "--token-env", "TR_BENCH_TOKEN",
              "--duration", "1", "--concurrency", "1",
              "--ttft-p95", "900", "--success-rate", "0.99",
              "--out-dir", str(d), "--skip-preflight"])
    except Exception:
        pass
    finally:
        os.environ.pop("TR_BENCH_TOKEN", None)
    cfg = json.loads((d / "run-config.json").read_text())
    assert cfg["endpoint"]["path"] == "/serving-endpoints/my-ep/invocations"
    assert cfg["sizing_concurrency"] == 1
    assert "concurrency" not in cfg
    assert cfg["acceptance_targets"]["ttft_ms"]["p95"] == 900
    assert cfg["acceptance_targets"]["success_rate"] == 0.99
    assert cfg["acceptance_targets"]["targets_are"].startswith("yours")
    # the internal preflight key must not leak into the saved config
    assert "_input_tokens" not in cfg


def test_prompts_mode_honors_output_tokens_without_a_512_floor():
    d = _tmp()
    prompts = d / "prompts.jsonl"
    prompts.write_text('{"prompt":"hello"}\n')
    try:
        main(["benchmark", "--host", "https://example.invalid",
              "--endpoint", "my-ep", "--prompts", str(prompts),
              "--output-tokens", "40,90", "--duration", "1",
              "--sizing-concurrency", "1", "--out-dir", str(d),
              "--skip-preflight"])
    except Exception:
        pass
    cfg = json.loads((d / "run-config.json").read_text())
    assert cfg["prompts_file"] == str(prompts)
    assert cfg["max_output_tokens_cap"] == 135
    assert cfg["max_output_tokens_cap"] != 512


def test_constant_cli_pairs_are_allowed_and_zero_cache_stays_zero():
    assert _pair("32,32", "output-tokens") == {"p50": 32, "p95": 32}
    assert _pair("0", "cache-hit-rate") == {"p50": 0, "p95": 0}


def test_extra_body_reaches_the_endpoint_config():
    """This is how a user turns reasoning down, so it has to survive."""
    d = _tmp()
    os.environ["TR_BENCH_TOKEN"] = "not-a-real-token"
    try:
        main(["benchmark", "--host", "https://example.invalid",
              "--endpoint", "my-ep", "--token-env", "TR_BENCH_TOKEN",
              "--extra-body", '{"reasoning_effort": "none"}',
              "--duration", "1", "--concurrency", "1",
              "--out-dir", str(d), "--skip-preflight"])
    except Exception:
        pass
    finally:
        os.environ.pop("TR_BENCH_TOKEN", None)
    cfg = json.loads((d / "run-config.json").read_text())
    assert cfg["endpoint"]["extra_body"] == {"reasoning_effort": "none"}


def test_bad_extra_body_json_is_refused_before_the_run():
    d = _tmp()
    try:
        main(["benchmark", "--host", "https://example.invalid",
              "--endpoint", "my-ep", "--extra-body", "{not json",
              "--out-dir", str(d), "--skip-preflight"])
    except SystemExit as e:
        assert "not valid JSON" in str(e)
    else:
        raise AssertionError("should have refused")


def test_extra_body_must_be_a_finite_json_object():
    from traffic_replay.cli import _benchmark_config
    import argparse

    base = dict(
        host="https://example.invalid", endpoint="ep", auth_profile=None,
        token_env="T", model=None, sizing_concurrency=1,
        legacy_concurrency=None, duration=1, out_dir=str(_tmp()),
        title=None, label=None, input_tokens="10", output_tokens="2",
        cache_hit_rate="0", prompts=None,
        profile="configs/profile_validation_small.json", ttft_p50=None,
        ttft_p90=None, ttft_p95=None, ttft_p99=None, ttfg_p50=None,
        ttfg_p90=None, ttfg_p95=None, ttfg_p99=None, success_rate=None,
        max_concurrency=None, max_pending_requests=None, cmd="benchmark")
    for raw in ('[1, 2]', '{"x": NaN}'):
        with pytest.raises(SystemExit):
            _benchmark_config(argparse.Namespace(**base, extra_body=raw))


def test_malformed_quantile_pairs_are_not_silently_repaired():
    for raw in ("1,,2", ",1", "1,"):
        with pytest.raises(SystemExit):
            _pair(raw, "input-tokens")


# ---- provenance ---------------------------------------------------------

def test_every_run_writes_a_manifest_that_can_trace_the_number():
    """A latency figure with no record of which code, which traffic shape and
    which endpoint produced it is an anecdote."""
    import threading
    import time
    from traffic_replay.mock_server import serve
    from traffic_replay.runner import RunConfig, run

    d = _tmp()
    srv = serve(0, d / "t.jsonl")
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        out = run(RunConfig(
            profile_path="configs/profile_validation_small.json",
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "UNUSED"},
            duration_s=6, qps_base=5.0, qps_burst=5.0, qps_min=5.0,
            qps_max=5.0, calibrate_n=4, max_output_tokens_cap=16,
            capture_endpoint_metadata=False, out_dir=str(d / "r")),
            quiet=True)
    finally:
        srv.shutdown()

    m = json.loads((Path(out["out_dir"]) / "manifest.json").read_text())
    assert m["harness_version"]
    assert m["latency_basis"]
    assert m["profile"] == "validation_small"
    assert m["profile_sha256_16"], "the traffic shape must be pinned by hash"
    assert m["seed"] == 7
    assert m["endpoint_base_url"].startswith("http://127.0.0.1:")
    assert m["python"] and m["numpy"]
    assert m["input_mode"] == "profile"
    # git state, so a number can be tied to the code that made it
    assert "git_commit" in m and "git_dirty" in m


def test_the_manifest_carries_no_token():
    import threading
    import time
    from traffic_replay.mock_server import serve
    from traffic_replay.runner import RunConfig, run

    d = _tmp()
    os.environ["TR_MANIFEST_TOKEN"] = "dapi-secret-value-here"
    srv = serve(0, d / "t.jsonl")
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    try:
        out = run(RunConfig(
            profile_path="configs/profile_validation_small.json",
            endpoint={"base_url": f"http://127.0.0.1:{port}",
                      "path": "/serving-endpoints/mock/invocations",
                      "auth_token_env": "TR_MANIFEST_TOKEN"},
            duration_s=4, qps_base=5.0, qps_burst=5.0, qps_min=5.0,
            qps_max=5.0, calibrate_n=3, max_output_tokens_cap=16,
            capture_endpoint_metadata=False, out_dir=str(d / "r")),
            quiet=True)
    finally:
        srv.shutdown()
        os.environ.pop("TR_MANIFEST_TOKEN", None)
    raw = (Path(out["out_dir"]) / "manifest.json").read_text()
    assert "dapi-secret-value-here" not in raw
    assert "TR_MANIFEST_TOKEN" not in raw or "dapi" not in raw


# ---- an expired token must not read as an endpoint failure --------------

def test_an_expired_token_is_refreshed_rather_than_failing_the_run():
    """Measured for real: a 90 second run lost 171 of 281 requests to
    'http 403: Invalid Token' when the OAuth token expired mid-run. Every
    one of those read as an endpoint failure."""
    import http.server
    import threading

    state = {"calls": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            state["calls"] += 1
            auth = self.headers.get("Authorization", "")
            if "fresh" not in auth:          # the first token is expired
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'{"error":"Invalid Token"}')
                return
            body = (b'data: {"choices":[{"delta":{"content":"hi"},'
                    b'"finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    b'data: [DONE]\n\n')
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from traffic_replay.client import EndpointClient, EndpointConfig
    try:
        cfg = EndpointConfig(base_url=f"http://127.0.0.1:{port}",
                             path="/invocations", auth_token_env="UNUSED")
        client = EndpointClient(cfg, "expired-token",
                                refresh=lambda: "fresh-token")
        res = client.send([{"role": "user", "content": "x"}], 16, "r1",
                          scheduled_s=0.0, dispatch_lag_ms=0.0,
                          intended=(0, 0, None, -1), chars_sent=1)
    finally:
        srv.shutdown()

    assert res.ok, f"should have recovered, got {res.status}: {res.error}"
    assert res.status == 200
    assert client.token == "fresh-token"


def test_a_genuinely_bad_credential_still_fails_the_run():
    """Refreshing must be bounded, or a bad credential spins forever."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"nope"}')

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from traffic_replay.client import EndpointClient, EndpointConfig
    try:
        cfg = EndpointConfig(base_url=f"http://127.0.0.1:{port}",
                             path="/invocations", auth_token_env="UNUSED",
                             max_retries=0)
        n = {"i": 0}

        def _always_new():
            n["i"] += 1
            return f"token-{n['i']}"

        client = EndpointClient(cfg, "bad", refresh=_always_new)
        res = client.send([{"role": "user", "content": "x"}], 16, "r1",
                          scheduled_s=0.0, dispatch_lag_ms=0.0,
                          intended=(0, 0, None, -1), chars_sent=1)
    finally:
        srv.shutdown()

    assert not res.ok
    assert n["i"] <= 6, "refresh must be bounded"
    # and the reason the user sees names auth, not "exhausted retries"
    assert "401" in (res.error or ""), res.error


# ---- the verdict has to move the exit code, or it gates nothing ----------

def _summary_dir(kind):
    """A finished run directory whose verdict is the requested kind."""
    import tempfile
    from traffic_replay.metrics import summarize, write_outputs
    base = 1_700_000_000.0
    rows = [{"ok": True, "phase": "replay", "ttft_ms": 100.0, "e2e_ms": 200.0,
             "prompt_tokens": 100, "completion_tokens": 10,
             "stream_complete": True, "visible_content_seen": True,
             "truncated": False, "parse_errors": 0,
             "t_send_unix": base + i * 0.3,
             "first_send_unix": base + i * 0.3} for i in range(300)]
    if kind == "invalid":
        for r in rows:
            r["visible_content_seen"] = False
    target = 1 if kind == "miss" else 100000
    s = summarize(rows, acceptance={"ttft_ms": {"p50": target}},
                  run_meta={"label": "t"})
    d = Path(tempfile.mkdtemp(prefix="exit-"))
    write_outputs(rows, s, d, "t")
    return {"out_dir": str(d), "summary": s}


def test_a_missed_target_exits_nonzero():
    """It exited 0 no matter what, so the harness could not gate a build."""
    from traffic_replay.cli import _finish
    assert _finish(_summary_dir("miss")) == 1


def test_a_run_with_no_readable_answers_exits_two():
    from traffic_replay.cli import _finish
    assert _finish(_summary_dir("invalid")) == 2


def test_write_outputs_never_overwrites_a_same_second_run_directory():
    from traffic_replay.metrics import summarize, write_outputs
    base = _tmp()
    requested = base / "20260806-010203"
    rows = [{"ok": True, "ttft_ms": 10.0, "e2e_ms": 20.0,
             "t_send_unix": 1.0, "prompt_tokens": 10,
             "completion_tokens": 2}]
    first_summary = summarize(rows, run_meta={"title": "first"})
    second_summary = summarize(rows, run_meta={"title": "second"})
    first = write_outputs(rows, first_summary, requested, "first")
    second = write_outputs(rows, second_summary, requested, "second")
    assert first == requested
    assert second != first
    assert second.name.startswith(requested.name + "-")
    assert json.loads((first / "summary.json").read_text())["run"]["title"] \
        == "first"
    assert json.loads((second / "summary.json").read_text())["run"]["title"] \
        == "second"
    assert json.loads((first / "manifest.json").read_text())["run_id"] != \
        json.loads((second / "manifest.json").read_text())["run_id"]
    for out in (first, second):
        assert (out / ".traffic-replay-complete").exists()
        assert not list(out.glob("*.tmp"))


def test_persisted_provenance_is_full_length_and_secret_redacted():
    from traffic_replay.metrics import summarize, write_outputs
    base = _tmp()
    profile = base / "profile.json"
    profile.write_text('{"name":"shape"}\n')
    secret = "dapi0123456789supersecret"
    rows = [{"ok": True, "ttft_ms": 10.0, "e2e_ms": 20.0,
             "t_send_unix": 1.0, "prompt_tokens": 10,
             "completion_tokens": 2}]
    summary = summarize(rows, run_meta={
        "profile_path": str(profile), "profile": "shape", "seed": 7,
        "endpoint_base_url": "https://user:password@example.test",
        "request_params": {"temperature": 0.0, "extra_body": {
            "reasoning_effort": "low", "api_key": secret,
            "nested": {"authorization": f"Bearer {secret}",
                       "vendorAccessToken": secret}}}})
    assert summary["run"]["request_params"]["extra_body"]["api_key"] \
        == "<redacted>"
    out = write_outputs(rows, summary, base / "run", "redacted")
    persisted = "\n".join(
        (out / name).read_text()
        for name in ("summary.json", "report.md", "report.html", "manifest.json"))
    assert secret not in persisted
    assert "user:password@" not in persisted
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["profile_sha256"]) == 64
    assert len(manifest["config_sha256"]) == 64
    assert manifest["artifact_created_at_utc"].endswith("+00:00")
    assert manifest["run_id"] == out.name
    assert manifest["request_params"]["extra_body"]["reasoning_effort"] == "low"


def test_fail_on_none_always_exits_zero():
    from traffic_replay.cli import _finish
    assert _finish(_summary_dir("miss"), fail_on="none") == 0
    assert _finish(_summary_dir("invalid"), fail_on="none") == 0


def test_the_terminal_prints_the_report_not_sliced_json():
    """The old default was json.dumps(summary)[:4000], a JSON document cut
    mid-structure, so the first thing a user saw was invalid JSON."""
    import contextlib
    import io
    from traffic_replay.cli import _finish
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _finish(_summary_dir("miss"))
    out = buf.getvalue()
    assert "requests:" in out          # the report, not a JSON blob
    assert "MISS:" in out
    assert not out.lstrip().startswith("{")


def test_json_format_emits_exactly_one_parseable_document():
    import contextlib
    import io
    from traffic_replay.cli import _finish
    buf = io.StringIO()
    result = _summary_dir("miss")
    with contextlib.redirect_stdout(buf):
        assert _finish(result, fmt="json") == 1
    assert json.loads(buf.getvalue()) == result["summary"]
    assert "open in a browser" not in buf.getvalue()


def test_unknown_verdict_fails_closed(monkeypatch):
    import contextlib
    import io
    from traffic_replay.cli import _finish
    monkeypatch.setattr("traffic_replay.metrics._verdict",
                        lambda summary: ("unexpected", "bad verdict"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _finish({"out_dir": str(_tmp()), "summary": {}}, fmt="json")
    assert code == 2
    assert json.loads(buf.getvalue()) == {}
