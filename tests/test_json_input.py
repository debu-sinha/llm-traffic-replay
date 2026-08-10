"""Strict JSON rejects ambiguity without echoing customer-controlled data."""
import math

import pytest

from traffic_replay.json_input import loads_strict


@pytest.mark.parametrize("raw", [
    '{"value":NaN}',
    '{"value":Infinity}',
    '{"value":-Infinity}',
    '{"value":1e999}',
    '{"value":-1e999}',
])
def test_strict_json_rejects_every_nonfinite_spelling(raw):
    with pytest.raises(ValueError, match="non-finite"):
        loads_strict(raw)


def test_strict_json_keeps_finite_numbers_and_nested_objects():
    value = loads_strict(
        '{"outer":{"count":3,"ratio":1.25},"items":[0,-2.5e-3]}')
    assert value == {
        "outer": {"count": 3, "ratio": 1.25},
        "items": [0, -0.0025],
    }
    assert all(math.isfinite(number) for number in (
        value["outer"]["ratio"], value["items"][1]))


def test_duplicate_key_diagnostic_redacts_payload_like_key_material():
    private_key = "Bearer " + "dapi" + ("x" * 40)
    raw = '{' + repr(private_key).replace("'", '"') + ':1,' \
        + repr(private_key).replace("'", '"') + ':2}'

    with pytest.raises(ValueError) as caught:
        loads_strict(raw)

    diagnostic = str(caught.value)
    assert private_key not in diagnostic
    assert "duplicate key <redacted; bytes=" in diagnostic
    assert "sha256=" in diagnostic


def test_duplicate_schema_key_remains_actionable_and_bounded():
    with pytest.raises(ValueError, match=r"duplicate key 'p50'"):
        loads_strict('{"p50":1,"p50":2}')


def test_excessive_nesting_is_a_safe_value_error_not_recursion_error():
    raw = "[" * 10_000 + "0" + "]" * 10_000
    with pytest.raises(ValueError, match="safe nesting depth"):
        loads_strict(raw)


def test_bytes_must_be_utf8():
    with pytest.raises(ValueError, match="not UTF-8 at byte offset"):
        loads_strict(b'{"value":"\xff"}')
