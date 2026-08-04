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
    assert cfg["concurrency"] == 1
    assert cfg["acceptance_targets"]["ttft_ms"]["p95"] == 900
    assert cfg["acceptance_targets"]["success_rate"] == 0.99
    assert cfg["acceptance_targets"]["targets_are"].startswith("yours")
    # the internal preflight key must not leak into the saved config
    assert "_input_tokens" not in cfg


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
