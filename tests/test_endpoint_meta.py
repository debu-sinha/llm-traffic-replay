"""Endpoint metadata capture: works with any endpoint name and never breaks
a run. The name handling matters because a customer's endpoint may not use
the databricks- prefix (customer endpoints often do not)."""
from __future__ import annotations

import http.client

import pytest

from traffic_replay.endpoint_meta import (
    endpoint_name_from_path, fetch_endpoint_metadata, _summarize)


def test_name_extraction_handles_custom_names():
    assert endpoint_name_from_path(
        "/serving-endpoints/databricks-glm-5-2/invocations") \
        == "databricks-glm-5-2"
    # custom, non-standard name (no databricks- prefix)
    assert endpoint_name_from_path(
        "/serving-endpoints/acme-glm-prod-42/invocations") \
        == "acme-glm-prod-42"
    assert endpoint_name_from_path("/foo/bar") is None
    assert endpoint_name_from_path("") is None


def test_name_extraction_requires_the_real_route_prefix_and_is_canonical():
    assert endpoint_name_from_path(
        "/other/serving-endpoints/not-an-endpoint/invocations") is None
    assert endpoint_name_from_path(
        "/serving-endpoints/my%20endpoint/invocations") == "my endpoint"
    assert endpoint_name_from_path(
        "/serving-endpoints/%2e%2e/invocations") is None
    assert endpoint_name_from_path(
        "/serving-endpoints/a%2Fb/invocations") is None


@pytest.mark.parametrize("path", [
    "/serving-endpoints/model",
    "/serving-endpoints/model/",
    "/serving-endpoints/model/chat/completions",
    "/serving-endpoints/model/invocations/extra",
    "/serving-endpoints/model/invocations/",
    "/serving-endpoints//invocations",
    "//serving-endpoints/model/invocations",
    "serving-endpoints/model/invocations",
    "/serving-endpoints/model/invocations?",
    "/serving-endpoints/model/invocations?x=1",
    "/serving-endpoints/model/invocations#fragment",
])
def test_name_extraction_rejects_every_noncanonical_direct_route(path):
    assert endpoint_name_from_path(path) is None


def test_fetch_returns_none_without_crashing():
    # no token -> None, no name -> None, unreachable host -> None
    assert fetch_endpoint_metadata("https://x.example.com",
                                   "/serving-endpoints/a/invocations", None) is None
    assert fetch_endpoint_metadata("https://x.example.com",
                                   "/no/name/here", "tok") is None
    # unroutable host, short timeout, must return None not raise
    assert fetch_endpoint_metadata("https://127.0.0.1:9",
                                   "/serving-endpoints/a/invocations", "tok",
                                   timeout=0.2) is None


def test_metadata_never_sends_a_bearer_token_over_remote_cleartext(
        monkeypatch):
    def must_not_connect(*args, **kwargs):
        raise AssertionError("HTTP connection should not be attempted")

    monkeypatch.setattr(http.client, "HTTPConnection", must_not_connect)
    assert fetch_endpoint_metadata(
        "http://metadata.example", "/serving-endpoints/a/invocations",
        "secret") is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), True])
def test_invalid_metadata_timeout_is_rejected_without_network(timeout,
                                                               monkeypatch):
    monkeypatch.setattr(
        http.client, "HTTPSConnection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("connection should not be attempted")))
    assert fetch_endpoint_metadata(
        "https://x.example", "/serving-endpoints/a/invocations", "secret",
        timeout=timeout) is None


def test_metadata_duplicate_keys_fail_closed_without_echoing_body(
        monkeypatch, capsys):
    first = "private-first-endpoint-name"
    second = "private-second-endpoint-name"
    body = (f'{{"name":"{first}","name":"{second}",'
            '"config":{"served_entities":[]}}').encode()

    class Response:
        status = 200

        @staticmethod
        def getheader(_name):
            return str(len(body))

        @staticmethod
        def read(_limit):
            return body

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    assert fetch_endpoint_metadata(
        "https://metadata.example",
        "/serving-endpoints/a/invocations", "secret") is None

    diagnostic = capsys.readouterr().err
    assert "StrictJSONError" in diagnostic
    assert first not in diagnostic
    assert second not in diagnostic


def test_summarize_keeps_customer_relevant_fields():
    doc = {"name": "ep", "task": "llm/v1/chat", "route_optimized": True,
           "state": {"ready": "READY"},
           "config": {"served_entities": [
               {"name": "e", "workload_type": "GPU_LARGE",
                "workload_size": "Small", "provisioned_model_units": 4,
                "scale_to_zero_enabled": False,
                "foundation_model": {
                    "name": "system.ai.databricks-glm-5-2",
                    "version": "2026-08-01",
                    "irrelevant": "drop me too",
                },
                "irrelevant": "drop me"}]}}
    s = _summarize(doc)
    assert s["name"] == "ep" and s["ready"] == "READY"
    assert s["route_optimized"] is True
    e = s["served_entities"][0]
    assert e["workload_type"] == "GPU_LARGE" and e["provisioned_model_units"] == 4
    assert e["foundation_model"] == {
        "name": "system.ai.databricks-glm-5-2",
        "version": "2026-08-01",
    }
    assert "irrelevant" not in e


@pytest.mark.parametrize("doc", [
    [],
    {"config": []},
    {"config": {"served_entities": {}}},
    {"config": {"served_entities": ["bad"]}},
    {"config": {"served_entities": [{
        "name": "bad", "foundation_model": "not-an-object"}]}},
])
def test_malformed_metadata_shapes_are_rejected(doc):
    with pytest.raises(ValueError):
        _summarize(doc)


# Captured from a real Databricks serving-endpoints GET on 2026-08-02, against
# a custom-named endpoint with a provisioned served entity. Workspace host and
# customer identifiers scrubbed, JSON SHAPE untouched. The point of keeping the
# real shape is that a hand-written fixture is what let the "workload type and
# size" claim ship unobserved: the pay-per-token endpoint used for the live
# runs returns served_entities entries carrying only a name.
REAL_PROVISIONED_RESPONSE = {
    "name": "example-custom-endpoint",
    "route_optimized": True,
    "state": {"ready": "NOT_READY", "config_update": "NOT_UPDATING"},
    "config": {
        "served_entities": [
            {
                "name": "example_model-1",
                "entity_name": "example_catalog.example_schema.example_model",
                "entity_version": "1",
                "workload_type": "GPU_SMALL",
                "workload_size": "Large",
                "scale_to_zero_enabled": True,
            }
        ]
    },
}

# Same API, pay-per-token foundation model endpoint. The nested foundation-model
# identity is positive deployment evidence; workload fields remain optional.
REAL_PAY_PER_TOKEN_RESPONSE = {
    "name": "databricks-glm-5-2",
    "task": "llm/v1/chat",
    "route_optimized": False,
    "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
    "config": {"served_entities": [{
        "name": "databricks-glm-5-2",
        "foundation_model": {"name": "system.ai.databricks-glm-5-2"},
    }]},
}


def test_summarize_real_provisioned_response_shape():
    out = _summarize(REAL_PROVISIONED_RESPONSE)
    assert out["name"] == "example-custom-endpoint"
    assert out["route_optimized"] is True
    assert out["ready"] == "NOT_READY"
    se = out["served_entities"][0]
    assert se["workload_type"] == "GPU_SMALL"
    assert se["workload_size"] == "Large"


def test_summarize_real_pay_per_token_response_keeps_positive_identity():
    """Keep provider identity without inventing provisioned workload fields."""
    out = _summarize(REAL_PAY_PER_TOKEN_RESPONSE)
    assert out["ready"] == "READY"
    se = out["served_entities"][0]
    assert se["name"] == "databricks-glm-5-2"
    assert se["foundation_model"] == {
        "name": "system.ai.databricks-glm-5-2"}
    assert "workload_type" not in se
    assert "workload_size" not in se


def test_real_pay_per_token_shape_renders_without_a_served_entity_row():
    """Regression for the claim that shipped documented but unobserved: with
    only foundation identity, the card shows no provisioned workload detail."""
    from traffic_replay.metrics import render_html, summarize
    rows = [{"ok": True, "t_send_unix": float(i), "ttft_ms": 100.0,
             "ttfb_ms": 1.0, "e2e_ms": 200.0, "connect_ms": 8.0,
             "dispatch_lag_ms": 0.0, "prompt_tokens": 10,
             "completion_tokens": 2} for i in range(40)]
    meta = {"input_mode": "profile", "endpoint_path": "/e",
            "endpoint_metadata": _summarize(REAL_PAY_PER_TOKEN_RESPONSE)}
    h = render_html(summarize(rows, run_meta=meta), "ppt")
    assert "Endpoint under test" in h
    assert "databricks-glm-5-2" in h
    assert "GPU_" not in h
