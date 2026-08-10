from __future__ import annotations

import pytest

from traffic_replay.artifacts import redact_secrets
from traffic_replay.client import validate_extra_body_safety


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
