from __future__ import annotations

import json

import pytest

from traffic_replay.artifacts import redact_secrets
from traffic_replay.client import (
    EndpointClient,
    EndpointConfig,
    validate_extra_body_safety,
)


@pytest.mark.parametrize("key", (
    "secret",
    "auth",
    "api_secret",
    "provider-secret",
    "nested_client_secret",
))
def test_generic_secret_and_auth_keys_are_redacted_and_rejected(key):
    private = "PRIVATE-CUSTOMER-VALUE"
    value = {"safe": {key: private}}

    redacted = redact_secrets(value)
    assert redacted == {"safe": {key: "<redacted>"}}
    assert private not in repr(redacted)
    with pytest.raises(ValueError, match="must not contain credentials"):
        validate_extra_body_safety(value)


def test_behavioral_auth_and_token_controls_remain_visible():
    value = {
        "auth_type": "oauth-m2m",
        "max_tokens": 128,
        "reasoning_tokens": 64,
    }
    assert redact_secrets(value) == value


def _schema_bearing_extra_body():
    return {
        "tools": [{
            "type": "function",
            "function": {
                "name": "validate_login_form",
                "description": "Validate the shape without receiving values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "password": {
                            "type": "string",
                            "format": "password",
                            "description": "A password-shaped customer field.",
                        },
                        "credentials": {
                            "type": "object",
                            "properties": {
                                "api_key": {"type": "string"},
                                "authorization": {"type": "string"},
                            },
                            "required": ["api_key"],
                        },
                    },
                    "required": ["password", "credentials"],
                },
            },
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "credential_audit_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "$defs": {
                        "token": {
                            "type": "object",
                            "properties": {
                                "secret": {"type": "boolean"},
                            },
                        },
                    },
                    "properties": {
                        "password": {"type": "boolean"},
                        "headers": {"type": "array", "items": {
                            "type": "string"}},
                        "result": {"$ref": "#/$defs/token"},
                    },
                    "required": ["password", "headers", "result"],
                    "additionalProperties": False,
                },
            },
        },
    }


def test_schema_property_names_are_not_credential_controls():
    extra = _schema_bearing_extra_body()
    validate_extra_body_safety(extra)
    cfg = EndpointConfig(
        base_url="https://example.invalid", path="/invoke",
        extra_body=extra)
    assert cfg.extra_body == extra
    body = json.loads(EndpointClient(cfg, None)._body(
        [{"role": "user", "content": "validate this form"}], 32, False))
    assert body["tools"] == extra["tools"]
    assert body["response_format"] == extra["response_format"]


def test_schema_names_remain_exact_in_extra_body_evidence_context():
    extra = _schema_bearing_extra_body()
    wrapped = {"endpoint": {"extra_body": extra}}
    assert redact_secrets(wrapped) == wrapped

    # Carried probe evidence independently revalidates candidates.  It must
    # preserve the same schema names rather than reintroducing the old generic
    # redaction-equality false positive.
    from traffic_replay.runner import _validated_reasoning_probe_candidate
    bounded_probe = {"response_format": {"type": "json_schema",
        "json_schema": {"schema": {"properties": {
            "password": {"type": "string"}}}}}}
    assert _validated_reasoning_probe_candidate(
        bounded_probe, label="test_probe") == bounded_probe

    # The exemption is path-scoped.  An arbitrary evidence object named
    # ``properties`` does not become a generic redaction bypass.
    untrusted = {"properties": {"password": {"value": "opaque-secret"}}}
    assert redact_secrets(untrusted) == {
        "properties": {"password": "<redacted>"}}


@pytest.mark.parametrize("extra", (
    {"password": "<redacted>"},
    {"routing": {"api_key": "opaque-provider-value"}},
    {"headers": {"Content-Type": "application/json"}},
    {"properties": {"password": "opaque-invalid-schema-value"}},
    {"response_format": {"type": "json_schema", "json_schema": {
        "name": "unsafe", "schema": {"type": "object", "properties": {
            "password": {"type": "string", "default": "hunter2"},
        }}}}},
    {"tools": [{"type": "function", "function": {
        "name": "unsafe", "parameters": {"type": "object",
            "properties": {"safe_field": {"type": "string",
                "description": "Bearer dapi0123456789never-send"}}},
    }}]},
))
def test_schema_exemption_does_not_allow_credential_material(extra):
    with pytest.raises(ValueError, match="must not contain credentials"):
        validate_extra_body_safety(extra)
