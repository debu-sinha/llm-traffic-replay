"""Security and accounting invariants at the credential/transport boundary."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from traffic_replay.client import (EndpointClient, EndpointConfig,
                                   UnsafeBearerTransport, normalized_origin)
from traffic_replay.quota_planner import RuntimeQuotaGuard
from traffic_replay.runner import AuthProfileError, _token, _token_from_profile


def _profile_file(tmp_path, text: str) -> str:
    path = tmp_path / "databrickscfg"
    path.write_text(text)
    return str(path)


class _OAuthResponse:
    def __init__(self, status=200, body=None, *, content_type="application/json",
                 content_length="auto"):
        if body is None:
            body = (b'{"access_token":"m2m-token","token_type":"Bearer",'
                    b'"expires_in":3600,"scope":"all-apis"}')
        self.status = status
        self.body = body
        self.read_calls = []
        self.headers = {}
        if content_type is not None:
            self.headers["content-type"] = content_type
        if content_length == "auto":
            self.headers["content-length"] = str(len(body))
        elif content_length is not None:
            self.headers["content-length"] = content_length

    def getheader(self, name):
        return self.headers.get(name.casefold())

    def read(self, limit=-1):
        self.read_calls.append(limit)
        return self.body if limit < 0 else self.body[:limit]


def _install_m2m_transport(monkeypatch, response=None, *, failure=None):
    """Install a recording HTTPSConnection without touching the network."""
    seen = {"instances": []}
    response = response or _OAuthResponse()

    class Connection:
        def __init__(self, host, port, timeout, context):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.context = context
            self.closed = False
            self.request_args = None
            seen["instances"].append(self)

        def request(self, method, path, body, headers):
            self.request_args = (method, path, body, headers)
            if failure is not None:
                raise failure

        def getresponse(self):
            return response

        def close(self):
            self.closed = True

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    return seen


def test_profile_token_is_bound_to_its_normalized_origin(tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[work]\nhost = HTTPS://EXAMPLE.COM./\ntoken = dapi-not-real\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    # Case, a terminal DNS dot, a trailing slash, and an explicit default port
    # do not turn one origin into four different security identities.
    assert _token_from_profile("work", "https://example.com:443") == \
        "dapi-not-real"
    assert normalized_origin("HTTPS://EXAMPLE.COM./") == \
        ("https", "example.com", 443)


@pytest.mark.parametrize("url", [
    "https://example.com/serving",
    "https://example.com?redirect=elsewhere",
    "https://example.com#fragment",
])
def test_base_url_is_an_origin_and_request_path_is_configured_separately(url):
    with pytest.raises(ValueError, match="must be an origin"):
        EndpointConfig(base_url=url, path="/invocations")


@pytest.mark.parametrize("path", ["relative", "//other-host/path", "/bad\npath"])
def test_request_path_rejects_ambiguous_or_unsafe_forms(path):
    with pytest.raises(ValueError, match="path"):
        EndpointConfig(base_url="https://example.com", path=path)


def test_profile_host_mismatch_fails_before_credential_can_escape(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[work]\nhost = https://trusted.example\ntoken = dapi-secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setenv("SHOULD_NOT_FALL_BACK", "environment-secret")
    endpoint = EndpointConfig(
        base_url="https://attacker.example", path="/invocations",
        auth_profile="work", auth_token_env="SHOULD_NOT_FALL_BACK",
    )

    with pytest.raises(AuthProfileError, match="is bound to") as err:
        _token(endpoint)
    assert "dapi-secret" not in str(err.value)
    assert "environment-secret" not in str(err.value)


def test_missing_profile_host_fails_closed_without_invoking_cli(
        tmp_path, monkeypatch):
    cfg = _profile_file(tmp_path, "[oauth]\nauth_type = databricks-cli\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("CLI must not mint an unbound token")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(AuthProfileError, match="no configured host"):
        _token_from_profile("oauth", "https://workspace.example")
    assert called is False


def test_oauth_profile_accepts_one_strict_cli_token_envelope(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[oauth]\nhost = https://workspace.example\n"
        "auth_type = databricks-cli\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setattr(
        "traffic_replay.runner._run_cli_bounded",
        lambda *args, **kwargs: (0, b'{"access_token":"minted"}'),
    )

    assert _token_from_profile(
        "oauth", "https://workspace.example") == "minted"


@pytest.mark.parametrize("payload", [
    b'[{"access_token":"minted"}]',
    b'{"access_token":7}',
    b'{"access_token":"first","access_token":"second"}',
    b'{"access_token":"minted","expires_on":NaN}',
])
def test_oauth_profile_rejects_ambiguous_cli_token_envelope(
        tmp_path, monkeypatch, payload):
    cfg = _profile_file(
        tmp_path,
        "[oauth]\nhost = https://workspace.example\n"
        "auth_type = databricks-cli\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setattr(
        "traffic_replay.runner._run_cli_bounded",
        lambda *args, **kwargs: (0, payload),
    )

    with pytest.raises(AuthProfileError, match="token JSON|access token"):
        _token_from_profile("oauth", "https://workspace.example")


def test_u2m_cli_is_explicit_and_environment_auth_is_scrubbed(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[u2m]\nhost = https://workspace.example\n"
        "auth_type = databricks-cli\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setenv("DATABRICKS_TOKEN", "must-not-be-inherited")
    monkeypatch.setenv("DATABRICKS_HOST", "https://wrong.example")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return 0, b'{"access_token":"minted"}'

    monkeypatch.setattr(
        "traffic_replay.runner._run_cli_bounded", fake_run)
    assert _token_from_profile("u2m", "https://workspace.example") == \
        "minted"
    assert captured["command"] == [
        "databricks", "auth", "token", "-p", "u2m"]
    assert captured["timeout_s"] == 30.0
    assert captured["max_stdout_bytes"] == 64 * 1024
    assert captured["env"]["DATABRICKS_CONFIG_FILE"] == cfg
    assert "DATABRICKS_TOKEN" not in captured["env"]
    assert "DATABRICKS_HOST" not in captured["env"]


def test_host_only_profile_never_falls_back_to_cli_or_environment(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path, "[u2m]\nhost = https://workspace.example\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setenv("DATABRICKS_TOKEN", "environment-secret")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("host-only profile must not invoke the CLI")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(AuthProfileError, match="auth_type=databricks-cli") \
            as err:
        _token_from_profile("u2m", "https://workspace.example")
    assert called is False
    assert "environment-secret" not in str(err.value)


@pytest.mark.parametrize("auth_type", [None, "pat"])
def test_pat_profile_is_direct_with_or_without_explicit_type(
        tmp_path, monkeypatch, auth_type):
    type_line = "" if auth_type is None else f"auth_type = {auth_type}\n"
    cfg = _profile_file(
        tmp_path,
        "[pat]\nhost = https://workspace.example\n"
        f"{type_line}token = dapi-not-real\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("PAT must not invoke CLI"))
    monkeypatch.setattr(
        "http.client.HTTPSConnection",
        lambda *args, **kwargs: pytest.fail("PAT must not mint OAuth"))
    assert _token_from_profile("pat", "https://workspace.example") == \
        "dapi-not-real"


def test_default_is_a_real_profile_and_never_inherits_into_other_profiles(
        tmp_path, monkeypatch):
    default_secret = "dapi-default-secret"
    cfg = _profile_file(
        tmp_path,
        "[DEFAULT]\nhost = https://default.example\n"
        f"token = {default_secret}\n"
        "[work]\nhost = https://work.example\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    assert _token_from_profile("DEFAULT", "https://default.example") == \
        default_secret
    with pytest.raises(AuthProfileError, match="no supported credentials") \
            as err:
        _token_from_profile("work", "https://work.example")
    assert default_secret not in str(err.value)


@pytest.mark.parametrize("profile,match", [
    ("auth_type = pat\n", "requires a token"),
    ("token = pat\nclient_id = id\nclient_secret = secret\n", "mixes"),
    ("client_id = id\n", "add client_secret"),
    ("client_secret = secret\n", "add client_id"),
    ("auth_type = databricks-cli\ntoken = pat\n", "must not contain"),
    ("auth_type = oauth-m2m\ntoken = pat\n", "requires client_id"),
    ("auth_type = browser\n", "unsupported auth_type"),
    ("auth_type = pat\naccount_id = account\ntoken = pat\n",
     "unsupported workspace"),
])
def test_ambiguous_incomplete_and_unsupported_profiles_fail_closed(
        tmp_path, monkeypatch, profile, match):
    cfg = _profile_file(
        tmp_path,
        "[bad]\nhost = https://workspace.example\n" + profile,
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: pytest.fail("invalid profile invoked CLI"))
    monkeypatch.setattr(
        "http.client.HTTPSConnection",
        lambda *args, **kwargs: pytest.fail("invalid profile used network"))
    with pytest.raises(AuthProfileError, match=match):
        _token_from_profile("bad", "https://workspace.example")


@pytest.mark.parametrize("auth_type", [None, "oauth-m2m"])
def test_workspace_m2m_uses_bound_https_basic_client_credentials(
        tmp_path, monkeypatch, auth_type):
    type_line = "" if auth_type is None else f"auth_type = {auth_type}\n"
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://WORKSPACE.example.:443/\n"
        f"{type_line}client_id = client-id\n"
        "client_secret = client:secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    seen = _install_m2m_transport(monkeypatch)

    assert _token_from_profile(
        "m2m", "https://workspace.example") == "m2m-token"
    assert len(seen["instances"]) == 1
    conn = seen["instances"][0]
    assert (conn.host, conn.port, conn.timeout) == (
        "workspace.example", 443, 15.0)
    method, path, body, headers = conn.request_args
    assert method == "POST"
    assert path == "/oidc/v1/token"
    assert body == b"grant_type=client_credentials&scope=all-apis"
    assert headers["Accept"] == "application/json"
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert headers["Connection"] == "close"
    assert base64.b64decode(
        headers["Authorization"].removeprefix("Basic ")) == \
        b"client-id:client:secret"
    assert conn.closed is True


def test_m2m_origin_mismatch_precedes_credential_use_and_network(
        tmp_path, monkeypatch):
    client_secret = "client-secret-must-not-leak"
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://trusted.example\nclient_id = id\n"
        f"client_secret = {client_secret}\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    monkeypatch.setattr(
        "http.client.HTTPSConnection",
        lambda *args, **kwargs: pytest.fail("mismatched origin used network"))
    with pytest.raises(AuthProfileError, match="is bound to") as err:
        _token_from_profile("m2m", "https://attacker.example")
    assert client_secret not in str(err.value)


def test_m2m_requires_https_even_for_a_loopback_test_origin(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = http://127.0.0.1:8080\nclient_id = id\n"
        "client_secret = secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    with pytest.raises(AuthProfileError, match="requires an HTTPS workspace"):
        _token_from_profile("m2m", "http://127.0.0.1:8080")


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_m2m_http_failures_are_fingerprinted_without_body_or_credentials(
        tmp_path, monkeypatch, status):
    client_secret = "profile-secret-never-print"
    response_secret = "server-secret-never-print"
    body = response_secret.encode()
    response = _OAuthResponse(status=status, body=body,
                              content_type="text/plain")
    seen = _install_m2m_transport(monkeypatch, response)
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://workspace.example\nclient_id = client\n"
        f"client_secret = {client_secret}\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    with pytest.raises(AuthProfileError, match=f"HTTP {status}") as err:
        _token_from_profile("m2m", "https://workspace.example")
    message = str(err.value)
    assert f"bytes={len(body)}" in message
    assert hashlib.sha256(body).hexdigest() in message
    assert client_secret not in message
    assert response_secret not in message
    assert seen["instances"][0].closed is True


@pytest.mark.parametrize("body,content_type,match", [
    (b"not-json", "application/json", "invalid JSON"),
    (b'[{"access_token":"secret"}]', "application/json", "non-object"),
    (b'{"access_token":"first","access_token":"secret",'
     b'"token_type":"Bearer"}', "application/json", "invalid JSON"),
    (b'{"access_token":"secret","token_type":"Bearer",'
     b'"expires_in":NaN}', "application/json", "invalid JSON"),
    (b'{"access_token":"secret","token_type":"mac"}',
     "application/json", "Bearer token_type"),
    (b'{"access_token":"secret","token_type":"Bearer",'
     b'"scope":"wrong"}', "application/json", "unexpected scope"),
    (b'{"access_token":"secret","token_type":"Bearer",'
     b'"expires_in":false}', "application/json", "invalid expires_in"),
    (b'{"access_token":"secret","token_type":"Bearer"}',
     "text/html", "non-JSON Content-Type"),
    (b'{"access_token":"secret","token_type":"Bearer"}',
     None, "non-JSON Content-Type"),
])
def test_m2m_rejects_malformed_or_semantically_invalid_responses_without_leak(
        tmp_path, monkeypatch, body, content_type, match):
    response = _OAuthResponse(body=body, content_type=content_type)
    _install_m2m_transport(monkeypatch, response)
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://workspace.example\nclient_id = client\n"
        "client_secret = profile-secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    with pytest.raises(AuthProfileError, match=match) as err:
        _token_from_profile("m2m", "https://workspace.example")
    message = str(err.value)
    assert "profile-secret" not in message
    assert "secret" not in message
    assert err.value.__cause__ is None


@pytest.mark.parametrize("content_length", ["65537", "-1", "NaN", "1, 2"])
def test_m2m_rejects_oversized_or_malformed_content_length_before_read(
        tmp_path, monkeypatch, content_length):
    response = _OAuthResponse(content_length=content_length)
    _install_m2m_transport(monkeypatch, response)
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://workspace.example\nclient_id = client\n"
        "client_secret = secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    with pytest.raises(AuthProfileError, match="safety limit|malformed"):
        _token_from_profile("m2m", "https://workspace.example")
    assert response.read_calls == []


def test_m2m_rejects_chunked_response_beyond_the_bound(tmp_path, monkeypatch):
    response = _OAuthResponse(
        body=b"x" * (64 * 1024 + 1), content_length=None)
    _install_m2m_transport(monkeypatch, response)
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://workspace.example\nclient_id = client\n"
        "client_secret = secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    with pytest.raises(AuthProfileError, match="safety limit"):
        _token_from_profile("m2m", "https://workspace.example")
    assert response.read_calls == [64 * 1024 + 1]


@pytest.mark.parametrize("failure,match", [
    (TimeoutError("network-secret"), "timed out"),
    (OSError("network-secret"), "request failed"),
])
def test_m2m_transport_failures_are_actionable_and_never_echo_exception_text(
        tmp_path, monkeypatch, failure, match):
    seen = _install_m2m_transport(monkeypatch, failure=failure)
    cfg = _profile_file(
        tmp_path,
        "[m2m]\nhost = https://workspace.example\nclient_id = client\n"
        "client_secret = profile-secret\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)

    with pytest.raises(AuthProfileError, match=match) as err:
        _token_from_profile("m2m", "https://workspace.example")
    assert "network-secret" not in str(err.value)
    assert "profile-secret" not in str(err.value)
    assert err.value.__cause__ is None
    assert seen["instances"][0].closed is True


def test_u2m_cli_failures_never_echo_stdout_stderr_or_exception_payload(
        tmp_path, monkeypatch):
    cfg = _profile_file(
        tmp_path,
        "[u2m]\nhost = https://workspace.example\n"
        "auth_type = databricks-cli\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    secret = "cli-secret-never-print"
    monkeypatch.setattr(
        "traffic_replay.runner._run_cli_bounded",
        lambda *args, **kwargs: (17, secret.encode()),
    )

    with pytest.raises(AuthProfileError, match="status 17") as err:
        _token_from_profile("u2m", "https://workspace.example")
    assert secret not in str(err.value)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], 30, output=secret.encode(), stderr=secret.encode())

    monkeypatch.setattr(
        "traffic_replay.runner._run_cli_bounded", timeout)
    with pytest.raises(AuthProfileError, match="timed out") as err:
        _token_from_profile("u2m", "https://workspace.example")
    assert secret not in str(err.value)
    assert err.value.__cause__ is None


def test_malformed_config_error_does_not_echo_the_offending_line(
        tmp_path, monkeypatch):
    secret = "config-secret-never-print"
    cfg = _profile_file(
        tmp_path,
        "[bad]\nhost = https://workspace.example\n" + secret + "\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    with pytest.raises(AuthProfileError, match="syntax and permissions") as err:
        _token_from_profile("bad", "https://workspace.example")
    assert secret not in str(err.value)
    assert err.value.__cause__ is None


@pytest.mark.parametrize("url", [
    "http://example.com",
    "http://localhost.example.com",
    "http://10.0.0.1",
])
def test_bearer_token_is_rejected_on_remote_cleartext(url):
    with pytest.raises(UnsafeBearerTransport, match="cleartext HTTP"):
        EndpointClient(EndpointConfig(base_url=url, path="/p"), "secret")


@pytest.mark.parametrize("url", [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://127.255.255.254:8080",
    "http://[::1]:8080",
])
def test_bearer_token_is_allowed_only_on_explicit_loopback_test_hosts(url):
    client = EndpointClient(EndpointConfig(base_url=url, path="/p"), "test")
    assert client.scheme == "http"


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_total_timeout_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="total_timeout_s"):
        EndpointConfig(
            base_url="http://127.0.0.1:1", path="/p",
            total_timeout_s=value,
        )


@pytest.mark.parametrize("value", ["", "pooled", True, 7, {}])
def test_production_connection_policy_is_a_closed_exact_contract(value):
    with pytest.raises(ValueError, match="production_connection_policy"):
        EndpointConfig(
            base_url="https://workspace.example", path="/p",
            production_connection_policy=value,
        )


def test_transport_contract_qualifies_unknown_production_behavior():
    client = EndpointClient(EndpointConfig(
        base_url="http://127.0.0.1:1", path="/p"), None)

    contract = client.transport_contract()

    assert contract["connection_policy_id"] == \
        "fresh_http1_per_physical_attempt"
    assert contract["production_connection_policy_declared"] is None
    assert contract["production_connection_policy_match"] is False
    assert contract["production_comparability_warning"]
    assert contract["production_connection_policy_assurance"] is None


def test_transport_contract_records_an_exact_operator_assertion():
    client = EndpointClient(EndpointConfig(
        base_url="http://127.0.0.1:1", path="/p",
        production_connection_policy="fresh_http1_per_physical_attempt",
    ), None)

    contract = client.transport_contract()

    assert contract["production_connection_policy_match"] is True
    assert contract["production_comparability_warning"] is None
    assert "operator asserted" in \
        contract["production_connection_policy_assurance"]


@pytest.mark.parametrize("value", [-1, True, 3, 10 ** 400])
def test_physical_inference_retries_are_strictly_bounded(value):
    with pytest.raises(ValueError, match="integer from 0 to 2"):
        EndpointConfig(
            base_url="https://workspace.example", path="/p",
            max_retries=value,
        )


def test_huge_retry_count_is_rejected_when_run_config_is_validated():
    from traffic_replay.runner import RunConfig

    with pytest.raises(ValueError, match="integer from 0 to 2"):
        RunConfig(
            endpoint={
                "base_url": "https://workspace.example",
                "path": "/serving-endpoints/model/invocations",
                "max_retries": 10 ** 400,
            },
            profile_path="not-read-during-config-validation.json",
        )


def _mixed_profile_with_secrets(tmp_path, monkeypatch):
    secret = "dapi-cli-secret-never-print"
    cfg = _profile_file(
        tmp_path,
        "[bad]\nhost = https://workspace.example\n"
        f"token = {secret}\nclient_id = client-id\n"
        "client_secret = oauth-secret-never-print\n",
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", cfg)
    return secret


def test_benchmark_json_auth_failure_is_one_document_without_traceback(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import main

    secret = _mixed_profile_with_secrets(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "traffic_replay.runner.make_schedule",
        lambda **_kwargs: {
            "rates": [1.0], "counts": [1], "timestamps": [0.0]})
    rc = main([
        "benchmark", "--host", "https://workspace.example",
        "--endpoint", "model", "--auth-profile", "bad",
        "--fixed-rate", "1", "--duration", "1",
        "--out-dir", str(tmp_path / "benchmark"), "--format", "json",
    ])
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert rc == 2
    assert doc == {
        "passed": False,
        "stage": "authentication",
        "exit_code": 2,
        "error": (
            "Databricks auth profile 'bad' mixes a PAT token with OAuth "
            "client credentials; use one authentication method per profile"),
    }
    assert secret not in captured.out + captured.err
    assert "oauth-secret-never-print" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_run_json_auth_failure_is_one_document_without_traceback(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import main

    secret = _mixed_profile_with_secrets(tmp_path, monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    config = {
        "endpoint": {
            "base_url": "https://workspace.example",
            "path": "/serving-endpoints/model/invocations",
            "auth_profile": "bad",
        },
        "profile_path": str(repo / "configs/profile_validation_small.json"),
        "duration_s": 1,
        "sizing_concurrency": 1,
        "calibrate_n": 0,
        "capture_endpoint_metadata": False,
        "measure_network_path": False,
        "out_dir": str(tmp_path / "run"),
    }
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config))
    rc = main(["run", "--config", str(config_path), "--format", "json"])
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert rc == 2
    assert doc["stage"] == "authentication"
    assert doc["exit_code"] == 2
    assert secret not in captured.out + captured.err
    assert "oauth-secret-never-print" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_sweep_auth_failure_is_concise_stderr_without_traceback(
        tmp_path, monkeypatch, capsys):
    from traffic_replay.cli import main

    secret = _mixed_profile_with_secrets(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "traffic_replay.runner.make_schedule",
        lambda **_kwargs: {
            "rates": [1.0], "counts": [1], "timestamps": [0.0]})
    rc = main([
        "sweep", "--host", "https://workspace.example",
        "--endpoint", "model", "--auth-profile", "bad",
        "--rate", "1,2", "--duration", "1", "--cooldown", "0",
        "--diagnostic-only",
        "--out-dir", str(tmp_path / "sweep"),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err.startswith("authentication failed: ")
    combined = captured.out + captured.err
    assert secret not in combined
    assert "oauth-secret-never-print" not in combined
    assert "Traceback" not in combined


class _Sock:
    def settimeout(self, value):
        self.timeout = value


class _Response:
    def __init__(self, status: int, body: bytes = b"", events=()):
        self.status = status
        self._body = body
        self._events = events

    def read(self, n=-1):
        return self._body if n < 0 else self._body[:n]

    def __iter__(self):
        return iter(self._events)


def test_client_persists_databricks_served_model_response_header():
    events = (
        b'data: {"model":"underlying-model","choices":[{"delta":'
        b'{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Response(_Response):
        def getheader(self, name):
            return {
                "content-type": "text/event-stream; charset=utf-8",
                "served-model-name": "active-served-entity",
            }.get(name.casefold())

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "served-header",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert result.ok is True
    assert result.response_model == "underlying-model"
    assert result.served_model_name == "active-served-entity"


class _TimedFailure:
    sock = _Sock()

    def __init__(self):
        self.request_called_at = None

    def connect(self):
        time.sleep(0.04)

    def request(self, *args, **kwargs):
        self.request_called_at = time.time()
        raise OSError("reset after write began")

    def close(self):
        pass


def test_absolute_deadline_stops_a_continuous_heartbeat_stream():
    class RecordingSock:
        def __init__(self):
            self.timeouts = []

        def settimeout(self, value):
            self.timeouts.append(value)

    class Heartbeats:
        status = 200

        def __iter__(self):
            while True:
                time.sleep(0.008)
                yield b": keepalive\n\n"

    class Connection:
        def __init__(self):
            self.sock = RecordingSock()
            self.closed = False

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Heartbeats()

        def close(self):
            self.closed = True

    conn = Connection()
    client = EndpointClient(
        EndpointConfig(
            base_url="http://127.0.0.1:1", path="/p",
            read_timeout_s=1.0, total_timeout_s=0.035,
        ),
        None,
    )
    client._connect = lambda: conn
    started_unix = time.time()
    started = time.monotonic()
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "deadline-stream",
        0.0, 0.0, (0, 0, None, 0), 2,
        scheduled_monotonic=started,
    )
    elapsed = time.monotonic() - started

    assert 0.03 <= elapsed < 0.20
    assert result.ok is False
    assert result.status == 200
    assert result.error == (
        "request exceeded total timeout (total_timeout_s=0.035)")
    assert result.stream_complete is False
    assert result.ttfb_ms is not None
    assert result.e2e_ms >= 30
    assert result.caller_e2e_ms >= 30
    assert result.finished_unix >= started_unix
    assert result.request_attempts == 1
    assert conn.closed is True
    assert len(conn.sock.timeouts) >= 4
    assert all(0 < timeout <= 0.035 for timeout in conn.sock.timeouts)
    assert conn.sock.timeouts[-1] < conn.sock.timeouts[0]


def test_absolute_deadline_also_covers_connection_setup():
    class Connection:
        sock = _Sock()

        def __init__(self):
            self.timeout = None
            self.closed = False

        def connect(self):
            # A fake transport can ignore the requested socket timeout; the
            # client still checks the absolute clock immediately afterwards.
            time.sleep(0.025)

        def close(self):
            self.closed = True

    conn = Connection()
    client = EndpointClient(
        EndpointConfig(
            base_url="http://127.0.0.1:1", path="/p",
            connect_timeout_s=1.0, total_timeout_s=0.01,
        ),
        None,
    )
    client._connect = lambda: conn
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "deadline-connect",
        0.0, 0.0, (0, 0, None, 0), 2,
    )

    assert result.ok is False
    assert result.status is None
    assert result.error == (
        "request exceeded total timeout (total_timeout_s=0.01)")
    assert result.connection_attempts == 1
    assert result.request_attempts == 0
    assert result.first_attempt_unix is not None
    assert result.first_send_unix is None
    assert result.finished_unix is not None
    assert conn.timeout <= 0.01
    assert conn.closed is True


def test_absolute_deadline_expiring_during_quota_mark_never_posts():
    request_calls = []

    class SlowMarkGuard(RuntimeQuotaGuard):
        def mark_post_may_have_started(self, handle):
            evidence = super().mark_post_may_have_started(handle)
            time.sleep(0.03)
            return evidence

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            request_calls.append(True)

        def close(self):
            pass

    guard = SlowMarkGuard({
        "queries_per_hour": 100,
        "warning_utilization": 1.0,
    })
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False, total_timeout_s=0.01),
        None, runtime_quota_guard=guard)
    client._connect = Connection

    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "deadline-quota-mark",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert request_calls == []
    assert result.request_attempts == 0
    assert result.error == (
        "request exceeded total timeout (total_timeout_s=0.01)")
    assert result.quota_guard_events[0]["post_may_have_started"] is True
    assert result.quota_guard_events[0]["state"] == "committed"
    assert result.first_attempt_unix is not None
    assert result.first_send_unix is None
    assert result.finished_unix is not None


def test_send_timestamp_excludes_connection_setup_and_attempts_are_explicit():
    conn = _TimedFailure()
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token=None,
    )
    client._connect = lambda: conn
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r1", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.first_attempt_unix is not None
    assert result.first_send_unix is not None
    assert result.first_send_unix - result.first_attempt_unix >= 0.035
    assert abs(result.first_send_unix - conn.request_called_at) < 0.02
    assert result.connection_attempts == 1
    assert result.request_attempts == 1
    assert result.retries == 0
    assert result.retry_reasons == []


def test_exact_caller_clocks_preserve_uniform_schedule_delay():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    scheduled = time.monotonic() - 2.0
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "late", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=scheduled,
    )
    assert 1900 <= result.queue_wait_ms <= 2300
    assert 1900 <= result.caller_send_ms <= 2300
    assert 1900 <= result.caller_ttfb_ms <= 2300
    assert 1900 <= result.caller_ttft_ms <= 2300
    assert 1900 <= result.caller_ttfv_ms <= 2300
    assert result.caller_e2e_ms >= result.caller_ttft_ms
    assert result.ttft_ms < 300


def test_queue_wait_excludes_connection_setup_but_caller_latency_includes_it():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class SlowConnection:
        sock = _Sock()

        def connect(self):
            time.sleep(0.05)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = SlowConnection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "slow-connect", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=time.monotonic(),
    )
    assert result.connect_ms >= 40
    assert result.queue_wait_ms < result.connect_ms
    assert result.caller_send_ms >= result.connect_ms
    assert result.caller_ttft_ms >= result.connect_ms


class _ConnectFailure:
    sock = _Sock()

    def connect(self):
        raise OSError("connect refused")

    def close(self):
        pass


def test_failure_before_http_send_is_not_claimed_as_a_wire_send():
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=1),
        token=None,
    )
    client._connect = lambda: _ConnectFailure()
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r2", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.first_attempt_unix is not None
    assert result.first_send_unix is None
    assert result.t_send_unix is None
    assert result.connection_attempts == 2
    assert result.request_attempts == 0
    assert result.retries == 1
    assert result.retry_reasons == ["connection_error_before_post"]


def test_stream_options_fallback_is_counted_as_a_physical_request_retry():
    seen = []
    seen_bytes = []

    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, method, path, body, headers):
            seen_bytes.append(body)
            seen.append(json.loads(body))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(_Response(
            400, b'{"error":"stream_options include_usage unsupported"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token="local-test-token",
    )
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r3", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.ok is True
    assert len(seen) == 2
    assert "stream_options" in seen[0]
    assert "stream_options" not in seen[1]
    assert result.connection_attempts == 2
    assert result.request_attempts == 2
    assert result.retries == 1
    assert result.retry_reasons == ["stream_options_rejected"]
    assert result.physical_request_body_sha256s == [
        hashlib.sha256(body).hexdigest() for body in seen_bytes]
    assert result.physical_request_body_sha256s[0] != \
        result.physical_request_body_sha256s[1]


def test_runtime_quota_denial_occurs_before_the_physical_post():
    requests = []

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))

        def close(self):
            pass

    # Strict threshold contact is refused: limit=1 at 100% has no safe
    # positive integer below the threshold.
    guard = RuntimeQuotaGuard({
        "warning_utilization": 1.0,
        "queries_per_hour": 1,
    })
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None,
        runtime_quota_guard=guard)
    client._connect = Connection

    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "quota-denied",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert requests == []
    assert result.request_attempts == 0
    assert result.connection_attempts == 1
    assert result.quota_guard_denied is True
    assert len(result.quota_guard_events) == 1
    assert result.quota_guard_events[0]["decision"] == "denied"
    assert guard.tripped is True


def test_runtime_quota_can_refuse_a_fallback_retry_after_one_real_post():
    requests = []

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(_Response(
            400, b'{"error":"stream_options include_usage unsupported"}')),
        # The guard must refuse before this response can be reached.
        Connection(_Response(200)),
    ])
    guard = RuntimeQuotaGuard({
        "warning_utilization": 1.0,
        "queries_per_hour": 2,
    })
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None,
        runtime_quota_guard=guard)
    client._connect = lambda: next(connections)

    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "quota-retry-denied",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert len(requests) == 1
    assert result.request_attempts == 1
    assert result.connection_attempts == 2
    assert result.retries == 1
    assert result.retry_reasons == ["stream_options_rejected"]
    assert result.quota_guard_denied is True
    assert [event["decision"] for event in result.quota_guard_events] == [
        "admitted", "denied"]
    assert result.quota_guard_events[0]["state"] == "committed"
    assert guard.snapshot()["counts"]["committed"] == 1


def test_transport_failure_row_captures_terminal_quota_event():
    """The result is evaluated before an attempt's finally block runs.

    An ambiguous failure after conn.request must therefore settle the guard
    before RequestResult deep-copies event evidence; otherwise the sealed row
    says ``provisional`` while the command snapshot says ``committed``.
    """
    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            raise OSError("ambiguous failure after POST start")

        def close(self):
            pass

    guard = RuntimeQuotaGuard({
        "warning_utilization": 1.0,
        "queries_per_hour": 100,
    })
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False, max_retries=0), None,
        runtime_quota_guard=guard)
    client._connect = Connection

    result = client.send(
        [{"role": "user", "content": "hi"}], 8,
        "quota-ambiguous-transport", 0.0, 0.0,
        (0, 0, None, 0), 2)

    assert result.request_attempts == 1
    assert result.quota_guard_denied is False
    assert len(result.quota_guard_events) == 1
    assert result.quota_guard_events[0]["state"] == "committed"
    assert result.quota_guard_events[0]["post_may_have_started"] is True
    snapshot = guard.snapshot()
    assert snapshot["counts"]["committed"] == 1
    assert snapshot["provisional_reservations"] == 0


def test_guard_mark_failure_returns_exact_zero_post_evidence():
    ticks = iter((10, 11, 9))
    last = [9]

    def clock_ns():
        try:
            last[0] = next(ticks)
        except StopIteration:
            pass
        return last[0]

    requests = []

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))

        def close(self):
            pass

    guard = RuntimeQuotaGuard({
        "warning_utilization": 1.0, "queries_per_hour": 100,
    }, clock_ns=clock_ns, wall_clock=lambda: 1_700_000_000.0)
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False, max_retries=0), None,
        runtime_quota_guard=guard)
    client._connect = Connection

    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "mark-failure",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert requests == []
    assert result.request_attempts == 0
    assert result.quota_guard_denied is True
    assert len(result.quota_guard_events) == 1
    assert result.quota_guard_events[0]["state"] == "provisional"
    assert result.quota_guard_events[0]["post_may_have_started"] is False
    assert guard.snapshot()["tripped"] is True


def test_guard_commit_failure_returns_exact_ambiguous_post_evidence():
    ticks = iter((10, 11, 12, 9))
    last = [9]

    def clock_ns():
        try:
            last[0] = next(ticks)
        except StopIteration:
            pass
        return last[0]

    requests = []

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            requests.append((args, kwargs))

        def getresponse(self):
            return _Response(503, b'{"error":"busy"}')

        def close(self):
            pass

    guard = RuntimeQuotaGuard({
        "warning_utilization": 1.0, "queries_per_hour": 100,
    }, clock_ns=clock_ns, wall_clock=lambda: 1_700_000_000.0)
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False, max_retries=0), None,
        runtime_quota_guard=guard)
    client._connect = Connection

    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "commit-failure",
        0.0, 0.0, (0, 0, None, 0), 2)

    assert len(requests) == 1
    assert result.request_attempts == 1
    assert result.status == 503
    assert result.quota_guard_denied is True
    assert len(result.quota_guard_events) == 1
    assert result.quota_guard_events[0]["state"] == "provisional"
    assert result.quota_guard_events[0]["post_may_have_started"] is True
    snapshot = guard.snapshot()
    assert snapshot["tripped"] is True
    assert snapshot["provisional_reservations"] == 1


def test_concurrent_stream_options_rejections_each_fallback_once():
    delayed_started = threading.Event()
    release_delayed = threading.Event()
    connection_count = 0
    count_lock = threading.Lock()
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, role):
            self.role = role

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            if self.role == "delayed-400":
                delayed_started.set()
                assert release_delayed.wait(timeout=2.0)
                return _Response(
                    400,
                    b'{"error":"stream_options include_usage unsupported"}',
                )
            if self.role == "learning-400":
                return _Response(
                    400,
                    b'{"error":"stream_options include_usage unsupported"}',
                )
            return _Response(200, events=events)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)

    def connect():
        nonlocal connection_count
        with count_lock:
            connection_count += 1
            number = connection_count
        return Connection(
            {1: "delayed-400", 2: "learning-400"}.get(number, "success"))

    client._connect = connect
    with ThreadPoolExecutor(max_workers=2) as pool:
        delayed = pool.submit(
            client.send, [{"role": "user", "content": "hi"}], 8,
            "delayed", 0.0, 0.0, (0, 0, None, 0), 2)
        assert delayed_started.wait(timeout=1.0)
        learner = pool.submit(
            client.send, [{"role": "user", "content": "hi"}], 8,
            "learner", 0.0, 0.0, (0, 0, None, 0), 2)
        learned = learner.result(timeout=2.0)
        release_delayed.set()
        raced = delayed.result(timeout=2.0)

    assert learned.ok is True and raced.ok is True
    assert learned.request_attempts == raced.request_attempts == 2
    assert learned.retry_reasons == ["stream_options_rejected"]
    assert raced.retry_reasons == ["stream_options_rejected"]
    assert client._include_usage_supported is False


def test_exhausted_midstream_reset_preserves_observed_evidence():
    class BrokenResponse:
        status = 200

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise OSError("reset after content")

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return BrokenResponse()

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "partial-reset",
        0.0, 0.0, (0, 0, None, 0), 2,
        scheduled_monotonic=time.monotonic())

    assert result.ok is False
    assert result.status == 200
    assert result.ttfb_ms is not None
    assert result.ttft_ms is not None
    assert result.caller_ttft_ms is not None
    assert result.e2e_ms is not None
    assert result.content_chunks == 1
    assert result.visible_content_seen is True
    assert result.stream_complete is False
    assert result.request_attempts == 1
    assert result.error == "transport failed: OSError"


def test_failed_refresh_is_single_flight_and_respects_each_deadline():
    workers = 3
    response_barrier = threading.Barrier(workers)
    refresh_calls = 0
    refresh_lock = threading.Lock()

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            response_barrier.wait(timeout=1.0)
            return _Response(401, b'expired token')

        def close(self):
            pass

    def refresh():
        nonlocal refresh_calls
        with refresh_lock:
            refresh_calls += 1
        time.sleep(0.05)
        return None

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       include_usage=False, total_timeout_s=0.02),
        "stale-token", refresh=refresh)
    client._connect = Connection

    def invoke(index):
        started = time.monotonic()
        result = client.send(
            [{"role": "user", "content": "hi"}], 8, f"auth-{index}",
            0.0, 0.0, (0, 0, None, 0), 2)
        return time.monotonic() - started, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(invoke, range(workers)))

    assert refresh_calls == 1
    assert max(elapsed for elapsed, _result in outcomes) < 0.15
    assert all(result.error ==
               "request exceeded total timeout (total_timeout_s=0.02)"
               for _elapsed, result in outcomes)


@pytest.mark.parametrize("token", ["bad\r\ntoken", "bad token", "tökén"])
def test_endpoint_client_rejects_header_unsafe_tokens_without_echo(token):
    with pytest.raises(ValueError) as raised:
        EndpointClient(
            EndpointConfig(base_url="http://127.0.0.1:1", path="/p"),
            token)
    assert token not in str(raised.value)


def test_exact_caller_clock_includes_automatic_fallback_elapsed_time():
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class DelayedResponse(_Response):
        def read(self, n=-1):
            time.sleep(0.04)
            return super().read(n)

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(DelayedResponse(
            400, b'{"error":"stream_options unsupported"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "fallback", 0.0, 0.0,
        (0, 0, None, 0), 2, scheduled_monotonic=time.monotonic(),
    )
    assert result.ok is True
    assert result.request_attempts == 2
    assert result.caller_ttft_ms >= 35
    assert result.caller_e2e_ms >= 35
    assert result.ttft_ms < result.caller_ttft_ms


def test_retry_cannot_reuse_caller_milestones_from_a_failed_stream():
    class BrokenContentResponse:
        status = 200

        def __iter__(self):
            yield (b'data: {"choices":[{"delta":{"content":"stale"}}]}'
                   b'\n\n')
            raise OSError("stream reset after content")

    tool_events = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"name":"lookup","arguments":"{}"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{},'
        b'"finish_reason":"tool_calls"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(BrokenContentResponse()),
        Connection(_Response(200, events=tool_events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=1),
        None,
    )
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "look it up"}], 20, "retry-tool",
        0.0, 0.0, (3, 20, None, -1), 10,
        scheduled_monotonic=time.monotonic(),
    )

    assert result.ok is True
    assert result.request_attempts == 2
    assert result.retry_reasons == ["transport_error_after_post"]
    assert result.valid_tool_calls == 1
    assert result.ttft_ms is None
    assert result.ttfv_ms is None
    assert result.caller_ttft_ms is None
    assert result.caller_ttfv_ms is None
    assert result.ttf_tool_call_ms is not None
    assert result.caller_ttf_tool_call_ms is not None

    from traffic_replay.metrics import summarize
    summary = summarize(
        [json.loads(result.to_json())],
        acceptance={"ttft_ms": {"p50": 1}},
    )
    assert "ttft_corrected_ms" not in summary
    assert summary["sla"]["ttft_vs_target"][0]["met"] is False
    from traffic_replay.report_decision import (
        IntegrityContext,
        build_report_decision,
    )
    decision = build_report_decision(
        summary, IntegrityContext(status="verified", reason="test evidence"))
    assert decision["customer_sla"]["code"] == "MISS"


def test_generic_400_is_not_retried_or_persisted_verbatim():
    secret_body = b'{"error":"customer prompt: private-value"}'

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(400, secret_body)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "bad400", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.request_attempts == 1
    assert result.retry_reasons == []
    assert "private-value" not in result.error
    assert "sha256=" in result.error


def test_error_echoing_optional_field_without_rejecting_it_is_not_retried():
    body = (b'{"error":"invalid messages; received request with '
            b'stream_options.include_usage=true"}')

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(400, body)

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "bad-messages", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.request_attempts == 1
    assert result.retry_reasons == []


def test_request_serialization_failure_never_opens_a_connection():
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"), None)
    client._connect = lambda: (_ for _ in ()).throw(
        AssertionError("must not connect"))
    result = client.send(
        [{"role": "user", "content": object()}], 8, "bad-json", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.first_attempt_unix is None
    assert result.first_send_unix is None
    assert result.connection_attempts == 0
    assert result.request_attempts == 0
    assert result.error == "request serialization failed: TypeError"


def test_permission_403_does_not_trigger_token_refresh():
    refreshed = False

    def refresh():
        nonlocal refreshed
        refreshed = True
        return "new-token"

    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(403, b'{"error":"permission denied"}')

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"),
        "old-token", refresh=refresh)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "forbidden", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.request_attempts == 1
    assert refreshed is False


def test_auth_refresh_is_counted_and_only_the_fresh_token_is_retried():
    seen_auth = []
    events = (
        b'data: {"choices":[{"delta":{"content":"ok"},'
        b'"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    )

    class Connection:
        sock = _Sock()

        def __init__(self, response):
            self.response = response

        def connect(self):
            pass

        def request(self, method, path, body, headers):
            seen_auth.append(headers.get("Authorization"))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        Connection(_Response(401, b'{"error":"expired"}')),
        Connection(_Response(200, events=events)),
    ])
    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p",
                       max_retries=0),
        token="old-local-token", refresh=lambda: "new-local-token",
    )
    client._connect = lambda: next(connections)
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "r4", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )

    assert result.ok is True
    assert seen_auth == ["Bearer old-local-token", "Bearer new-local-token"]
    assert result.request_attempts == 2
    assert result.retries == 1
    assert result.retry_reasons == ["auth_token_refreshed"]


def test_refresh_callback_failure_is_a_result_not_a_worker_exception():
    class Connection:
        sock = _Sock()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response(401, b'{"error":"expired"}')

        def close(self):
            pass

    def fail_refresh():
        raise RuntimeError("secret provider detail")

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/p"),
        "old-token", refresh=fail_refresh)
    client._connect = Connection
    result = client.send(
        [{"role": "user", "content": "hi"}], 8, "refresh-fail", 0.0, 0.0,
        (0, 0, None, 0), 2,
    )
    assert result.ok is False
    assert result.error == "credential refresh failed: RuntimeError"
    assert "secret provider detail" not in result.error


def test_refresh_capable_bearer_flow_is_rejected_before_remote_cleartext_io():
    refreshed = False

    def refresh():
        nonlocal refreshed
        refreshed = True
        return "must-not-leak"

    with pytest.raises(UnsafeBearerTransport, match="cleartext HTTP"):
        EndpointClient(
            EndpointConfig(base_url="http://example.com", path="/p"),
            token=None, refresh=refresh,
        )
    assert refreshed is False
