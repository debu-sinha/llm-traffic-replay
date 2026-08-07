"""quickstart writes a runnable config from the few things a load test needs,
and auth resolves from a ~/.databrickscfg profile so nobody has to mint a
bearer token by hand."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from traffic_replay.cli import main
from traffic_replay.runner import RunConfig, _token, _token_from_profile
from traffic_replay.client import EndpointConfig


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="qs-"))


def _run_quickstart(out: Path, *extra):
    argv = ["quickstart",
            "--host", "https://ws.cloud.databricks.com",
            "--endpoint", "my-endpoint",
            "--profile", "configs/profile_validation_small.json",
            "--concurrency", "30",
            "--out", str(out), *extra]
    assert main(argv) == 0
    return json.loads(out.read_text())


def test_quickstart_writes_a_config_the_runner_accepts():
    cfg = _run_quickstart(_tmp() / "q.json")
    # This is an open-loop sizing hint, not a held closed-loop concurrency.
    assert cfg["sizing_concurrency"] == 30
    assert "concurrency" not in cfg
    assert cfg["endpoint"]["path"] == "/serving-endpoints/my-endpoint/invocations"
    RunConfig(**cfg)                      # constructs without extra fields


def test_a_full_endpoint_path_is_passed_through():
    cfg = _run_quickstart(_tmp() / "q.json",
                          "--endpoint", "/serving-endpoints/x/invocations")
    assert cfg["endpoint"]["path"] == "/serving-endpoints/x/invocations"


def test_sla_targets_are_expressible_on_the_command_line():
    """The reason to run this at all is "do we meet ours". If that needs a
    hand-edited JSON block, quickstart has not done its job."""
    cfg = _run_quickstart(_tmp() / "q.json",
                          "--ttft-p50", "500", "--ttft-p95", "900",
                          "--ttfg-p95", "1500", "--success-rate", "0.9999")
    at = cfg["acceptance_targets"]
    assert at["ttft_ms"] == {"p50": 500.0, "p95": 900.0}
    assert at["ttfg_ms"] == {"p95": 1500.0}
    assert at["success_rate"] == 0.9999
    assert "command line" in at["targets_are"]


def test_no_targets_means_no_acceptance_block_rather_than_a_guess():
    cfg = _run_quickstart(_tmp() / "q.json")
    assert "acceptance_targets" not in cfg


@pytest.mark.parametrize("flag,value", [
    ("--ttft-p95", "0"),
    ("--success-rate", "0"),
    ("--success-rate", "1.1"),
])
def test_quickstart_rejects_invalid_sla_instead_of_silently_dropping_it(
        flag, value):
    out = _tmp() / "invalid.json"
    with pytest.raises(SystemExit, match="invalid quickstart"):
        _run_quickstart(out, flag, value)
    assert not out.exists()


def test_quickstart_rejects_invalid_workload_before_writing(tmp_path):
    out = tmp_path / "invalid.json"
    with pytest.raises(SystemExit, match="invalid quickstart"):
        _run_quickstart(out, "--duration", "0")
    assert not out.exists()


def test_auth_profile_replaces_the_token_env_var():
    cfg = _run_quickstart(_tmp() / "q.json", "--auth-profile", "my-ws")
    assert cfg["endpoint"]["auth_profile"] == "my-ws"
    assert "auth_token_env" not in cfg["endpoint"]


def test_without_a_profile_it_still_names_the_env_var():
    cfg = _run_quickstart(_tmp() / "q.json")
    assert cfg["endpoint"]["auth_token_env"] == "DATABRICKS_TOKEN"


def test_a_pat_profile_resolves_without_shelling_out():
    """A PAT profile stores a usable token, so no CLI call is needed."""
    import os
    d = _tmp()
    (d / "cfg").write_text("[work]\nhost = https://x\ntoken = dapi-not-real\n")
    old = os.environ.get("DATABRICKS_CONFIG_FILE")
    os.environ["DATABRICKS_CONFIG_FILE"] = str(d / "cfg")
    try:
        assert _token_from_profile("work", "https://x") == "dapi-not-real"
    finally:
        if old is None:
            os.environ.pop("DATABRICKS_CONFIG_FILE", None)
        else:
            os.environ["DATABRICKS_CONFIG_FILE"] = old


def test_the_env_var_still_works_when_no_profile_is_set():
    import os
    os.environ["TR_TEST_TOKEN"] = "from-env"
    try:
        cfg = EndpointConfig(base_url="https://x", path="/p",
                             auth_token_env="TR_TEST_TOKEN")
        assert _token(cfg) == "from-env"
    finally:
        os.environ.pop("TR_TEST_TOKEN", None)


def test_an_unresolvable_profile_fails_closed_without_env_fallback():
    """A typo must not repurpose an unrelated environment credential."""
    import os
    import pytest
    from traffic_replay.runner import AuthProfileError
    os.environ["TR_TEST_TOKEN"] = "fallback"
    try:
        cfg = EndpointConfig(base_url="https://x", path="/p",
                             auth_profile="no-such-profile-here",
                             auth_token_env="TR_TEST_TOKEN")
        with pytest.raises(AuthProfileError, match="does not exist"):
            _token(cfg)
    finally:
        os.environ.pop("TR_TEST_TOKEN", None)
