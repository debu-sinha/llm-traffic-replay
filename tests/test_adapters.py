"""Conformance tests for versioned endpoint wire adapters.

These tests deliberately use an in-process transport.  They prove that the
load engine delegates wire semantics to an adapter without opening a socket,
while retaining the exact Chat Completions SSE envelope shipped before the
adapter boundary was introduced.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator

import pytest

import traffic_replay.adapters as adapter_registry
from traffic_replay.adapters import (
    DEFAULT_ENDPOINT_ADAPTER,
    EndpointAdapter,
    EventMilestones,
    ResponseOutcome,
    get_endpoint_adapter,
    list_endpoint_adapters,
    register_endpoint_adapter,
    resolve_endpoint_adapter,
)
from traffic_replay.cli import main
from traffic_replay.client import (
    EndpointClient,
    EndpointConfig,
    serialize_request_body,
)
from traffic_replay.sse import StreamState
from traffic_replay.runner import (
    RunConfig,
    _assert_row_adapter_contract,
    _payload_hash,
    _resolved_run_id,
    _resolved_workload_id,
)


@pytest.fixture(autouse=True)
def _isolated_adapter_registry(monkeypatch):
    """Keep process-wide test registrations from leaking into other tests."""
    monkeypatch.setattr(
        adapter_registry,
        "_ADAPTERS",
        dict(adapter_registry._ADAPTERS),
    )
    monkeypatch.setattr(
        adapter_registry,
        "_ADAPTER_FINGERPRINTS",
        dict(adapter_registry._ADAPTER_FINGERPRINTS),
    )
    monkeypatch.setattr(
        adapter_registry,
        "_ADAPTER_CONTRACT_DIGESTS",
        dict(adapter_registry._ADAPTER_CONTRACT_DIGESTS),
    )
    monkeypatch.setattr(
        adapter_registry,
        "_ADAPTER_GUARDS",
        dict(adapter_registry._ADAPTER_GUARDS),
    )


class _CustomEventsAdapter(EndpointAdapter):
    """Small non-SSE wire dialect used to prove every delegation point."""

    adapter_id = "qa.custom_events/v17"
    response_mode = "streaming"
    request_media_type = "application/vnd.qa-request+json"
    accept_media_type = "application/vnd.qa-events"
    accepted_response_media_types = ("application/vnd.qa-events",)
    usage_request_mode = "intrinsic"

    def validate_endpoint(self, endpoint) -> None:
        pass

    def serialize_request(
        self,
        endpoint,
        messages: list[dict],
        max_tokens: int,
        include_usage: bool,
    ) -> bytes:
        return json.dumps(
            {
                "dialect": "qa",
                "deployment": endpoint.model,
                "turns": messages,
                "limit": max_tokens,
                "metering": include_usage,
                "native": endpoint.extra_body,
            },
            separators=(",", ":"),
        ).encode()

    def iter_events(
        self, chunks: Iterable[bytes | str]
    ) -> Iterator[dict]:
        encoded = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode()
            for chunk in chunks
        )
        yield json.loads(encoded)

    def fold_event(
        self, state: StreamState, event: object
    ) -> EventMilestones:
        assert isinstance(event, dict)
        state.saw_first_content = True
        state.saw_first_visible = True
        state.content_chunks += 1
        state.finish_reason = str(event["terminal"])
        state.usage = dict(event["meter"])
        state.done = True
        return EventMilestones(
            first_content=True,
            first_visible=True,
            content_event=True,
        )

    def finalize(self, state: StreamState) -> None:
        pass

    def normalize_usage(self, usage: dict | None) -> dict:
        if usage is None:
            return {
                "prompt_tokens": None,
                "completion_tokens": None,
                "cached_tokens": None,
                "cached_tokens_source": None,
                "reasoning_tokens": None,
                "reasoning_tokens_source": None,
            }
        return {
            "prompt_tokens": usage["input"],
            "completion_tokens": usage["output"],
            "cached_tokens": usage["cached"],
            "cached_tokens_source": "qa.meter.cached",
            "reasoning_tokens": usage["reasoning"],
            "reasoning_tokens_source": "qa.meter.reasoning",
        }


def _global_wire_helper(messages, max_tokens) -> bytes:
    return json.dumps(
        {"marker": "original", "messages": messages, "limit": max_tokens},
        separators=(",", ":")).encode()


class _GlobalHelperAdapter(_CustomEventsAdapter):
    adapter_id = "qa.global_helper/v1"

    def serialize_request(
        self, endpoint, messages, max_tokens, include_usage
    ) -> bytes:
        return _global_wire_helper(messages, max_tokens)


def test_default_adapter_preserves_exact_chat_sse_request_bytes():
    messages = [{"role": "user", "content": "hello"}]
    cfg = EndpointConfig(base_url="https://example.invalid", path="/invoke")

    expected_without_usage = (
        b'{"messages":[{"role":"user","content":"hello"}],'
        b'"max_tokens":64,"temperature":0.0,"stream":true}'
    )
    expected_with_usage = (
        b'{"messages":[{"role":"user","content":"hello"}],'
        b'"max_tokens":64,"temperature":0.0,"stream":true,'
        b'"stream_options":{"include_usage":true}}'
    )

    assert cfg.adapter == DEFAULT_ENDPOINT_ADAPTER
    assert serialize_request_body(cfg, messages, 64, False) == \
        expected_without_usage
    assert serialize_request_body(cfg, messages, 64, True) == \
        expected_with_usage
    assert EndpointClient(cfg, None)._body(messages, 64, True) == \
        expected_with_usage


def test_default_adapter_can_omit_temperature_without_model_code_changes():
    cfg = EndpointConfig(
        base_url="https://example.invalid",
        path="/invoke",
        model="future-model-that-rejects-sampling-controls",
        temperature=None,
    )
    body = json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "hello"}], 64, False))
    assert "temperature" not in body
    assert body["model"] == "future-model-that-rejects-sampling-controls"


def test_default_adapter_accepts_future_model_ids_and_native_extra_body():
    cfg = EndpointConfig(
        base_url="https://example.invalid",
        path="/future/invoke",
        model="frontier-lab/model-X.17:2028-04+fp4",
        temperature=0.125,
        extra_body={
            "future_reasoning_control": {
                "mode": "adaptive-ultra",
                "budget": {"policy": "task-dependent"},
            },
            "provider_extension_2030": ["alpha", {"enabled": True}],
        },
    )
    messages = [{"role": "user", "content": "未来 model"}]

    raw = serialize_request_body(cfg, messages, 321, True)
    assert raw == (
        b'{"future_reasoning_control":{"mode":"adaptive-ultra",'
        b'"budget":{"policy":"task-dependent"}},'
        b'"provider_extension_2030":["alpha",{"enabled":true}],'
        b'"messages":[{"role":"user","content":"\xe6\x9c\xaa\xe6\x9d\xa5 model"}],'
        b'"max_tokens":321,"temperature":0.125,"stream":true,'
        b'"model":"frontier-lab/model-X.17:2028-04+fp4",'
        b'"stream_options":{"include_usage":true}}'
    )


@pytest.mark.parametrize(
    "owned_field",
    [
        "messages", "max_tokens", "temperature", "stream", "model",
        "stream_options",
    ],
)
def test_default_adapter_rejects_silently_ignored_field_collisions(
        owned_field):
    with pytest.raises(ValueError, match="fields owned by endpoint adapter"):
        EndpointConfig(
            base_url="https://example.invalid",
            path="/invoke",
            extra_body={owned_field: False},
        )


def test_serialization_revalidates_mutated_provider_controls():
    cfg = EndpointConfig(
        base_url="https://example.invalid",
        path="/invoke",
        extra_body={"reasoning_effort": "none"},
    )
    assert cfg.extra_body is not None
    cfg.extra_body["stream"] = False

    with pytest.raises(ValueError, match="fields owned by endpoint adapter"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "hello"}], 8, True)


def test_client_snapshots_nested_config_and_rejects_internal_adapter_mutation():
    cfg = EndpointConfig(
        base_url="https://example.invalid",
        path="/invoke",
        extra_body={"reasoning_effort": "none"},
    )
    client = EndpointClient(cfg, None)
    assert cfg.extra_body is not None
    cfg.extra_body["reasoning_effort"] = "high"

    body = json.loads(client._body(
        [{"role": "user", "content": "hello"}], 8, True))
    assert body["reasoning_effort"] == "none"

    client.cfg.adapter = "qa.changed/v1"
    with pytest.raises(ValueError, match="adapter changed"):
        client._body([{"role": "user", "content": "hello"}], 8, True)


def test_execution_adapter_attests_once_not_twice_per_request(monkeypatch):
    """Hashing and wire serialization must not walk the registry hot path."""
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke")
    calls = []
    original = adapter_registry._catalog_contract

    def counted(adapter, registered_id):
        calls.append(registered_id)
        return original(adapter, registered_id)

    monkeypatch.setattr(adapter_registry, "_catalog_contract", counted)
    client = EndpointClient(cfg, None)
    messages = [{"role": "user", "content": "hello"}]
    assert calls == [DEFAULT_ENDPOINT_ADAPTER]

    for _ in range(25):
        raw = client._body(messages, 64, False)
        assert _payload_hash(
            client.cfg, messages, 64,
            adapter_execution=client.adapter_execution,
        ) == hashlib.sha256(raw).hexdigest()

    # No full implementation/contract walk occurred in either per-request
    # serialization path. The evidence boundary performs exactly one recheck.
    assert calls == [DEFAULT_ENDPOINT_ADAPTER]
    contract = client.transport_contract()
    assert contract["endpoint_adapter"]["adapter_id"] == \
        DEFAULT_ENDPOINT_ADAPTER
    assert calls == [DEFAULT_ENDPOINT_ADAPTER, DEFAULT_ENDPOINT_ADAPTER]


def test_execution_adapter_mutation_fails_closed_at_evidence_boundary(
        monkeypatch):
    client = EndpointClient(
        EndpointConfig(base_url="https://example.invalid", path="/invoke"),
        None)
    original = adapter_registry._chat_payload

    def changed_helper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter_registry, "_chat_payload", changed_helper)

    # The execution intentionally avoids the expensive registry walk on the
    # request path, but a report cannot be sealed under mutated behavior.
    client._body([{"role": "user", "content": "hello"}], 8, False)
    with pytest.raises(RuntimeError, match="changed implementation"):
        client.transport_contract()


def test_execution_contract_snapshot_is_defensive():
    execution = resolve_endpoint_adapter(DEFAULT_ENDPOINT_ADAPTER)
    modified = execution.contract
    modified["accepted_response_media_types"].append("text/plain")
    assert execution.contract["accepted_response_media_types"] == [
        "text/event-stream"]


def test_unknown_adapter_is_rejected_before_any_network_operation(monkeypatch):
    network_calls: list[object] = []

    class _MustNotConnect:
        def __init__(self, *args, **kwargs):
            network_calls.append((args, kwargs))
            raise AssertionError("network must not be touched")

    monkeypatch.setattr("http.client.HTTPConnection", _MustNotConnect)
    monkeypatch.setattr("http.client.HTTPSConnection", _MustNotConnect)

    with pytest.raises(ValueError, match="unknown endpoint adapter"):
        EndpointConfig(
            base_url="https://example.invalid",
            path="/invoke",
            adapter="future.unknown_protocol/v1",
        )
    assert network_calls == []


@pytest.mark.parametrize("adapter_id", [
    "",
    " ",
    "custom",
    "custom/v",
    "custom/v0",
    "custom/v01",
    "custom/vbad",
    "custom/v1/trailing",
    " custom/v1",
    "custom/v1 ",
    "custom route/v1",
    "x" * 126 + "/v1",
])
def test_registry_rejects_invalid_adapter_ids(adapter_id):
    adapter = _CustomEventsAdapter()
    adapter.adapter_id = adapter_id
    with pytest.raises(ValueError, match="adapter_id"):
        register_endpoint_adapter(adapter)


def test_registry_rejects_non_adapter_and_duplicate_identity():
    with pytest.raises(TypeError, match="must extend EndpointAdapter"):
        register_endpoint_adapter(object())

    first = _CustomEventsAdapter()
    second = _CustomEventsAdapter()
    register_endpoint_adapter(first)
    assert get_endpoint_adapter(first.adapter_id) is first
    with pytest.raises(ValueError, match="already registered"):
        register_endpoint_adapter(second)
    assert get_endpoint_adapter(first.adapter_id) is first


def test_registry_rejects_per_instance_adapter_state():
    adapter = _CustomEventsAdapter()
    adapter.runtime_state = []
    with pytest.raises(ValueError, match="must be stateless"):
        register_endpoint_adapter(adapter)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("response_mode", "duplex", "response_mode"),
        ("request_media_type", "Application/JSON", "request_media_type"),
        ("accept_media_type", "text/event-stream; charset=utf-8",
         "accept_media_type"),
        ("accepted_response_media_types", (),
         "accepted_response_media_types"),
        ("accepted_response_media_types", ("text/plain", "text/plain"),
         "must not contain duplicates"),
        ("usage_request_mode", "best_effort", "usage_request_mode"),
    ],
)
def test_registry_rejects_invalid_wire_contracts(attribute, value, message):
    adapter = _CustomEventsAdapter()
    setattr(adapter, attribute, value)
    with pytest.raises(ValueError, match=message):
        register_endpoint_adapter(adapter)


def test_catalog_is_versioned_complete_sorted_and_defensively_copied():
    catalog = list_endpoint_adapters()
    assert [row["adapter_id"] for row in catalog] == sorted(
        row["adapter_id"] for row in catalog)

    builtin = next(
        row for row in catalog
        if row["adapter_id"] == DEFAULT_ENDPOINT_ADAPTER
    )
    assert set(builtin) == {
        "adapter_id",
        "response_mode",
        "request_media_type",
        "accept_media_type",
        "accepted_response_media_types",
        "usage_request_mode",
        "canonical_state",
        "outcome_policy",
        "probe_rejection_policy",
        "implementation",
        "implementation_sha256",
    }
    assert builtin == {
        "adapter_id": "openai.chat_completions.sse/v1",
        "response_mode": "streaming",
        "request_media_type": "application/json",
        "accept_media_type": "text/event-stream",
        "accepted_response_media_types": ["text/event-stream"],
        "usage_request_mode": "stream_options.include_usage",
        "canonical_state": "traffic_replay.sse.StreamState/v1",
        "outcome_policy": (
            "traffic_replay.adapters.OpenAIChatCompletionsSSEAdapter."
            "evaluate_outcome"),
        "probe_rejection_policy": (
            "traffic_replay.adapters.OpenAIChatCompletionsSSEAdapter."
            "probe_control_rejected"),
        "implementation": (
            "traffic_replay.adapters.OpenAIChatCompletionsSSEAdapter"),
        "implementation_sha256": builtin["implementation_sha256"],
    }
    assert re.fullmatch(r"[0-9a-f]{64}", builtin["implementation_sha256"])

    # A caller can modify its catalog document without mutating registry state.
    builtin["accepted_response_media_types"].append("text/plain")
    fresh = list_endpoint_adapters()
    fresh_builtin = next(
        row for row in fresh
        if row["adapter_id"] == DEFAULT_ENDPOINT_ADAPTER
    )
    assert fresh_builtin["accepted_response_media_types"] == [
        "text/event-stream"]
    assert fresh_builtin["implementation_sha256"] == \
        builtin["implementation_sha256"]


def test_catalog_detects_implementation_mutation_after_registration(
        monkeypatch):
    adapter = _CustomEventsAdapter()
    register_endpoint_adapter(adapter)

    def changed_normalize_usage(self, usage):
        return usage

    monkeypatch.setattr(
        _CustomEventsAdapter, "normalize_usage", changed_normalize_usage)
    with pytest.raises(RuntimeError, match="changed implementation"):
        get_endpoint_adapter(adapter.adapter_id)
    with pytest.raises(RuntimeError, match="changed implementation"):
        list_endpoint_adapters()


@pytest.mark.parametrize("helper_name", ["_chat_payload", "update_state"])
def test_catalog_fingerprint_covers_transitive_protocol_helpers(
        monkeypatch, helper_name):
    original = getattr(adapter_registry, helper_name)

    def changed_helper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter_registry, helper_name, changed_helper)
    with pytest.raises(RuntimeError, match="changed implementation"):
        get_endpoint_adapter(DEFAULT_ENDPOINT_ADAPTER)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda state: state.__setitem__("mode", "changed"),
            id="class-dict"),
        pytest.param(
            lambda state: state["prefixes"].append("changed"),
            id="nested-class-list"),
        pytest.param(
            lambda state: state["limits"].__setitem__("max", 999),
            id="nested-class-dict"),
        pytest.param(
            lambda state: state.__setitem__("mode", state.pop("mode")),
            id="class-dict-order"),
    ],
)
def test_mutable_class_state_cannot_change_wire_behavior_after_registration(
        mutate):
    class _MutableClassStateAdapter(_CustomEventsAdapter):
        adapter_id = "qa.mutable_class_state/v1"
        wire_policy = {
            "mode": "original",
            "prefixes": ["v1"],
            "limits": {"max": 8},
        }

        def serialize_request(
            self, endpoint, messages, max_tokens, include_usage
        ) -> bytes:
            return json.dumps(
                {"policy": self.wire_policy, "messages": messages},
                separators=(",", ":")).encode()

    adapter = _MutableClassStateAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    baseline = json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 8, False))
    assert baseline["policy"] == {
        "mode": "original", "prefixes": ["v1"], "limits": {"max": 8}}

    mutate(_MutableClassStateAdapter.wire_policy)

    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 8, False)
    with pytest.raises(RuntimeError, match="changed implementation"):
        list_endpoint_adapters()


def test_added_instance_state_after_registration_is_detected_before_use():
    adapter = _CustomEventsAdapter()
    register_endpoint_adapter(adapter)
    adapter.injected_wire_state = {"mode": "changed"}

    with pytest.raises(RuntimeError, match="changed implementation"):
        get_endpoint_adapter(adapter.adapter_id)


def test_mutated_positional_default_is_detected_before_serialization():
    class _DefaultStateAdapter(_CustomEventsAdapter):
        adapter_id = "qa.mutable_default/v1"

        def serialize_request(
            self, endpoint, messages, max_tokens, include_usage,
            policy={"marker": ["original"]},
        ) -> bytes:
            return json.dumps(policy, separators=(",", ":")).encode()

    adapter = _DefaultStateAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    assert json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 1, False)) == {
            "marker": ["original"]}

    defaults = _DefaultStateAdapter.serialize_request.__defaults__
    assert defaults is not None
    defaults[0]["marker"].append("changed")

    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 1, False)


def test_mutated_keyword_default_is_detected_before_serialization():
    class _KeywordDefaultAdapter(_CustomEventsAdapter):
        adapter_id = "qa.mutable_kwdefault/v1"

        def serialize_request(
            self, endpoint, messages, max_tokens, include_usage, *,
            policy={"marker": {"value": "original"}},
        ) -> bytes:
            return json.dumps(policy, separators=(",", ":")).encode()

    adapter = _KeywordDefaultAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    assert json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 1, False)) == {
            "marker": {"value": "original"}}

    defaults = _KeywordDefaultAdapter.serialize_request.__kwdefaults__
    assert defaults is not None
    defaults["policy"]["marker"]["value"] = "changed"

    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 1, False)


def test_mutated_closure_state_is_detected_before_serialization():
    policy = {"marker": ["original"]}

    class _ClosureStateAdapter(_CustomEventsAdapter):
        adapter_id = "qa.mutable_closure/v1"

        def serialize_request(
            self, endpoint, messages, max_tokens, include_usage
        ) -> bytes:
            return json.dumps(policy, separators=(",", ":")).encode()

    adapter = _ClosureStateAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    assert json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 1, False)) == {
            "marker": ["original"]}

    policy["marker"].append("changed")

    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 1, False)


def test_rebound_global_helper_is_detected_before_serialization(monkeypatch):
    adapter = _GlobalHelperAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    assert json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 3, False))[
            "marker"] == "original"

    def changed_helper(messages, max_tokens):
        raise AssertionError("mutated helper must never serialize a request")

    monkeypatch.setitem(
        _GlobalHelperAdapter.serialize_request.__globals__,
        "_global_wire_helper", changed_helper)

    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 3, False)


def test_mutated_class_helper_is_detected_before_serialization(monkeypatch):
    class _ClassHelperAdapter(_CustomEventsAdapter):
        adapter_id = "qa.class_helper/v1"

        @staticmethod
        def _envelope(messages, max_tokens):
            return {"marker": "original", "limit": max_tokens}

        def serialize_request(
            self, endpoint, messages, max_tokens, include_usage
        ) -> bytes:
            return json.dumps(
                self._envelope(messages, max_tokens),
                separators=(",", ":")).encode()

    adapter = _ClassHelperAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        adapter=adapter.adapter_id)
    assert json.loads(serialize_request_body(
        cfg, [{"role": "user", "content": "x"}], 3, False))[
            "marker"] == "original"

    def changed_helper(messages, max_tokens):
        raise AssertionError("mutated helper must never serialize a request")

    monkeypatch.setattr(
        _ClassHelperAdapter, "_envelope", staticmethod(changed_helper))
    with pytest.raises(RuntimeError, match="changed implementation"):
        serialize_request_body(
            cfg, [{"role": "user", "content": "x"}], 3, False)


@pytest.mark.parametrize("state_kind", ["opaque", "cyclic"])
def test_unsupported_class_state_is_rejected_without_partial_registration(
        state_kind):
    class _UnsupportedStateAdapter(_CustomEventsAdapter):
        adapter_id = f"qa.unsupported_{state_kind}/v1"

    if state_kind == "opaque":
        _UnsupportedStateAdapter.wire_state = object()
        expected = "unsupported opaque state"
    else:
        cycle: list[object] = []
        cycle.append(cycle)
        _UnsupportedStateAdapter.wire_state = cycle
        expected = "cyclic state"

    adapter = _UnsupportedStateAdapter()
    with pytest.raises(ValueError, match=expected):
        register_endpoint_adapter(adapter)
    with pytest.raises(ValueError, match="unknown endpoint adapter"):
        get_endpoint_adapter(adapter.adapter_id)


def test_contract_side_effect_is_rejected_without_partial_registration():
    class _MutatingContractAdapter(_CustomEventsAdapter):
        adapter_id = "qa.mutating_contract/v1"
        contract_calls = 0

        def contract(self):
            type(self).contract_calls += 1
            return super().contract()

    adapter = _MutatingContractAdapter()
    with pytest.raises(ValueError, match="changed while it was being registered"):
        register_endpoint_adapter(adapter)
    with pytest.raises(ValueError, match="unknown endpoint adapter"):
        get_endpoint_adapter(adapter.adapter_id)


class _Socket:
    def settimeout(self, value) -> None:
        self.timeout = value


def test_custom_adapter_controls_wire_events_usage_media_and_result_identity():
    adapter = _CustomEventsAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/custom-events",
        adapter=adapter.adapter_id,
        model="any-vendor/future-model-2040",
        extra_body={"native_knob": {"level": 17}},
    )
    wire: dict[str, object] = {}
    event = {
        "visible": "answer",
        "terminal": "custom_complete",
        "meter": {"input": 41, "output": 7, "cached": 13, "reasoning": 2},
    }

    class _Response:
        status = 200

        def getheader(self, name):
            if name.casefold() == "content-type":
                return "application/vnd.qa-events; charset=utf-8"
            return None

        def __iter__(self):
            return iter([json.dumps(event, separators=(",", ":")).encode()])

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

        def connect(self):
            wire["connected"] = True

        def request(self, method, path, body, headers):
            wire.update(method=method, path=path, body=body, headers=headers)

        def getresponse(self):
            return _Response()

        def close(self):
            wire["closed"] = True

    client = EndpointClient(cfg, token=None)
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "measure this"}],
        19,
        "custom-request",
        0.0,
        0.0,
        (41, 19, 0.25, -1),
        12,
    )

    assert wire["method"] == "POST"
    assert wire["path"] == "/custom-events"
    assert json.loads(wire["body"]) == {
        "dialect": "qa",
        "deployment": "any-vendor/future-model-2040",
        "turns": [{"role": "user", "content": "measure this"}],
        "limit": 19,
        "metering": True,
        "native": {"native_knob": {"level": 17}},
    }
    assert wire["headers"] == {
        "Content-Type": "application/vnd.qa-request+json",
        "Accept": "application/vnd.qa-events",
        "X-Request-Id": "custom-request",
    }
    assert wire["connected"] is True
    assert wire["closed"] is True

    assert result.ok is True, result.to_json()
    assert result.status == 200
    assert result.finish_reason == "custom_complete"
    assert result.visible_content_seen is True
    assert result.prompt_tokens == 41
    assert result.completion_tokens == 7
    assert result.cached_tokens == 13
    assert result.cached_tokens_source == "qa.meter.cached"
    assert result.reasoning_tokens == 2
    assert result.reasoning_tokens_source == "qa.meter.reasoning"
    assert result.response_content_type == "application/vnd.qa-events"
    assert result.endpoint_adapter == adapter.adapter_id
    assert result.response_mode == adapter.response_mode
    assert json.loads(result.to_json())["endpoint_adapter"] == \
        adapter.adapter_id

def test_adapter_owns_usage_control_rejection_status_and_fallback():
    class _StatusAwareUsageAdapter(_CustomEventsAdapter):
        adapter_id = "qa.status_usage/v1"
        usage_request_mode = "stream_options.include_usage"

        def include_usage_option_rejected(self, status, body):
            return status == 422 and body == b"metering control unsupported"

    adapter = _StatusAwareUsageAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/status-aware",
        adapter=adapter.adapter_id,
    )
    sent: list[dict] = []
    event = {
        "visible": "answer",
        "terminal": "complete",
        "meter": {"input": 3, "output": 2, "cached": 0, "reasoning": 0},
    }

    class _Response:
        def __init__(self, status, body=b""):
            self.status = status
            self.body = body

        def getheader(self, name):
            if self.status == 200 and name.casefold() == "content-type":
                return "application/vnd.qa-events"
            return None

        def read(self, _limit):
            return self.body

        def __iter__(self):
            return iter([json.dumps(event, separators=(",", ":")).encode()])

    class _Connection:
        def __init__(self, response):
            self.response = response
            self.sock = _Socket()

        def connect(self):
            pass

        def request(self, _method, _path, body, headers):
            sent.append(json.loads(body))

        def getresponse(self):
            return self.response

        def close(self):
            pass

    connections = iter([
        _Connection(_Response(422, b"metering control unsupported")),
        _Connection(_Response(200)),
    ])
    client = EndpointClient(cfg, None)
    client._connect = lambda: next(connections)

    result = client.send(
        [{"role": "user", "content": "x"}], 2, "status-usage",
        0.0, 0.0, (3, 2, None, -1), 1)

    assert result.ok is True, result.to_json()
    assert [body["metering"] for body in sent] == [True, False]
    assert result.connection_attempts == 2
    assert result.request_attempts == 2
    assert result.retry_reasons == ["stream_options_rejected"]


@pytest.mark.parametrize(
    "status,candidate,body,expected",
    [
        (400, {"reasoning_effort": "none"},
         b'{"error":{"message":"invalid parameter: reasoning_effort"}}',
         True),
        (400, {"chat_template_kwargs": {"enable_thinking": False}},
         b'{"detail":[{"loc":["body","chat_template_kwargs",'
         b'"enable_thinking"],"msg":"Extra inputs are not permitted"}]}',
         True),
        (400, {"reasoning_effort": "none"},
         b'{"error":{"message":"context length exceeded"}}', False),
        (400, {"reasoning_effort": "none"},
         b'{"error":{"message":"invalid request"}}', False),
        (400, {"reasoning_effort": "none"},
         b'{"error":{"message":"invalid parameter other_field; request '
         b'included reasoning_effort"}}', False),
        (400, {"reasoning_effort": "none"},
         b'{"error":{"message":"context is invalid",'
         b'"request":{"reasoning_effort":"none"}}}', False),
        (422, {"reasoning_effort": "none"},
         b'{"detail":"unsupported parameter reasoning_effort"}', False),
    ],
    ids=(
        "named-field", "nested-path", "unrelated-context", "vague-error",
        "different-field-with-candidate-echo", "structured-request-echo",
        "builtin-does-not-assume-422",
    ),
)
def test_builtin_probe_rejection_requires_named_candidate_and_proven_status(
        status, candidate, body, expected):
    adapter = get_endpoint_adapter(DEFAULT_ENDPOINT_ADAPTER)
    assert adapter.probe_control_rejected(status, body, candidate) is expected


def test_custom_adapter_can_prove_a_provider_specific_422_rejection():
    class _Provider422Adapter(_CustomEventsAdapter):
        adapter_id = "qa.provider_422_probe/v1"

        def probe_control_rejected(self, status, body, candidate):
            return status == 422 \
                and body == b"qa_control is not supported" \
                and "qa_control" in candidate

    adapter = _Provider422Adapter()
    register_endpoint_adapter(adapter)
    assert adapter.probe_control_rejected(
        422, b"qa_control is not supported", {"qa_control": "off"}) is True
    assert adapter.probe_control_rejected(
        400, b"qa_control is not supported", {"qa_control": "off"}) is False


def test_client_probe_classifier_retains_only_error_sample_digest_and_length():
    body = b'{"error":{"message":"invalid parameter reasoning_effort"}}'
    candidate = {"reasoning_effort": "none"}
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/probe",
        extra_body=candidate,
    )

    class _Response:
        status = 400

        def read(self, _limit):
            return body

    class _Connection:
        sock = _Socket()

        def connect(self):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(cfg, None)
    client._connect = lambda: _Connection()
    result = client.send(
        [{"role": "user", "content": "x"}], 2, "probe-safe-error",
        0.0, 0.0, (3, 2, None, -1), 1,
        probe_candidate=candidate)

    assert result.ok is False
    assert result.status == 400
    assert result.probe_candidate_rejected is True
    assert result.http_error_body_sample_bytes == len(body)
    assert result.http_error_body_sample_sha256 == \
        __import__("hashlib").sha256(body).hexdigest()
    persisted = result.to_json()
    assert body.decode() not in persisted
    assert "invalid parameter" not in persisted
    assert "body sample bytes=" in (result.error or "")


def test_custom_adapter_media_type_is_enforced_before_event_parsing():
    adapter = _CustomEventsAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/custom-events",
        adapter=adapter.adapter_id,
    )

    class _Response:
        status = 200

        def getheader(self, name):
            return "text/event-stream" if name.casefold() == \
                "content-type" else None

        def __iter__(self):
            raise AssertionError("wrong-media response must not be parsed")

    class _Connection:
        sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(cfg, token=None)
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "measure this"}],
        9,
        "wrong-media",
        0.0,
        0.0,
        (3, 9, None, -1),
        6,
    )

    assert result.ok is False
    assert result.error == "stream protocol validation failed"
    assert result.endpoint_adapter == adapter.adapter_id
    assert result.response_content_type is None
    assert result.parse_errors == 1


def test_cli_adapters_emits_machine_and_human_catalogs(capsys):
    expected_catalog = list_endpoint_adapters()

    assert main(["adapters", "--format", "json"]) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine == {
        "schema_version": "endpoint-adapter-catalog/v1",
        "adapters": expected_catalog,
    }

    assert main(["adapters"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"{row['adapter_id']}: {row['response_mode']}; response "
        f"{', '.join(row['accepted_response_media_types'])}; usage "
        f"{row['usage_request_mode']}"
        for row in expected_catalog
    ]


def test_catalog_fingerprint_is_stable_for_the_builtin_adapter():
    first = next(
        row for row in list_endpoint_adapters()
        if row["adapter_id"] == DEFAULT_ENDPOINT_ADAPTER
    )
    second = next(
        row for row in list_endpoint_adapters()
        if row["adapter_id"] == DEFAULT_ENDPOINT_ADAPTER
    )
    assert first["implementation_sha256"] == second["implementation_sha256"]
    assert len(bytes.fromhex(first["implementation_sha256"])) == 32


def test_adapter_identity_changes_run_and_workload_identity(tmp_path):
    adapter = _CustomEventsAdapter()
    register_endpoint_adapter(adapter)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    common = {
        "base_url": "https://example.invalid",
        "path": "/invoke",
        "model": "same-model",
    }
    default = RunConfig(
        endpoint=dict(common), profile_path=str(profile), duration_s=1,
        qps_base=1, qps_burst=1, qps_min=1, qps_max=1,
    )
    custom = RunConfig(
        endpoint={**common, "adapter": adapter.adapter_id},
        profile_path=str(profile), duration_s=1,
        qps_base=1, qps_burst=1, qps_min=1, qps_max=1,
    )
    assert _resolved_run_id(default) != _resolved_run_id(custom)
    inputs = {"profile": {"sha256": "a" * 64, "bytes": 2}}
    assert _resolved_workload_id(default, inputs) != \
        _resolved_workload_id(custom, inputs)


def test_adapter_owns_terminal_and_success_semantics():
    class _ProviderTerminalAdapter(_CustomEventsAdapter):
        adapter_id = "qa.provider_terminal/v1"

        def fold_event(self, state, event):
            state.saw_first_content = True
            state.saw_first_visible = True
            state.content_chunks += 1
            state.usage = dict(event["meter"])
            # Deliberately do not map the provider terminal to Chat's ``done``
            # or ``finish_reason`` fields. The adapter's outcome policy owns it.
            return EventMilestones(
                first_content=True,
                first_visible=True,
                content_event=True,
            )

        def evaluate_outcome(self, state):
            return ResponseOutcome(
                response_complete=True,
                output_observed=state.saw_first_visible,
                truncated=False,
                ok=True,
            )

    adapter = _ProviderTerminalAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/provider-terminal",
        adapter=adapter.adapter_id,
    )

    class _Response:
        status = 200

        def getheader(self, name):
            return "application/vnd.qa-events" if name.casefold() == \
                "content-type" else None

        def __iter__(self):
            event = {
                "meter": {
                    "input": 3, "output": 2, "cached": 0,
                    "reasoning": 0,
                },
            }
            return iter([json.dumps(event).encode()])

    class _Connection:
        sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(cfg, None)
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "x"}], 2, "terminal", 0, 0,
        (3, 2, None, -1), 1,
    )
    assert result.ok is True
    assert result.stream_complete is True
    assert result.finish_reason is None


def test_adapter_runtime_contract_failures_become_safe_failed_rows():
    class _BrokenUsageAdapter(_CustomEventsAdapter):
        adapter_id = "qa.broken_usage/v1"

        def normalize_usage(self, usage):
            return {"provider_tokens": 999}

    adapter = _BrokenUsageAdapter()
    register_endpoint_adapter(adapter)
    cfg = EndpointConfig(
        base_url="http://127.0.0.1:1",
        path="/broken-usage",
        adapter=adapter.adapter_id,
    )

    class _Response:
        status = 200

        def getheader(self, name):
            return "application/vnd.qa-events" if name.casefold() == \
                "content-type" else None

        def __iter__(self):
            event = {
                "terminal": "complete",
                "meter": {
                    "input": 3, "output": 2, "cached": 0,
                    "reasoning": 0,
                },
            }
            return iter([json.dumps(event).encode()])

    class _Connection:
        sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(cfg, None)
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "x"}], 2, "broken", 0, 0,
        (3, 2, None, -1), 1,
    )
    assert result.ok is False
    assert result.parse_errors == 1
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.error == "response protocol validation failed"
    assert result.parse_error_details == [
        "endpoint adapter usage normalization contract failed (TypeError)"
    ]


@pytest.mark.parametrize(
    "row,match",
    [
        ({"phase": "preflight"}, "endpoint_adapter"),
        (
            {
                "phase": "preflight",
                "endpoint_adapter": DEFAULT_ENDPOINT_ADAPTER,
            },
            "response_mode",
        ),
        (
            {
                "phase": "probe",
                "endpoint_adapter": "qa.other/v1",
                "response_mode": "streaming",
            },
            "endpoint_adapter",
        ),
        (
            {
                "phase": "probe",
                "endpoint_adapter": DEFAULT_ENDPOINT_ADAPTER,
                "response_mode": "non_streaming",
            },
            "response_mode",
        ),
    ],
)
def test_carried_setup_rows_require_the_exact_adapter_contract(row, match):
    cfg = EndpointConfig(
        base_url="https://workspace.example",
        path="/serving-endpoints/test/invocations",
    )

    with pytest.raises(ValueError, match=match):
        _assert_row_adapter_contract([row], cfg)


def test_carried_setup_rows_accept_the_exact_adapter_contract():
    cfg = EndpointConfig(
        base_url="https://workspace.example",
        path="/serving-endpoints/test/invocations",
    )

    _assert_row_adapter_contract(
        [
            {
                "phase": "preflight",
                "endpoint_adapter": DEFAULT_ENDPOINT_ADAPTER,
                "response_mode": "streaming",
            }
        ],
        cfg,
    )
