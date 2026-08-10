"""SSE parsing: TTFT keys on first CONTENT delta (role-only chunks must not
trigger it), usage extraction is defensive across provider field names."""
import json

from traffic_replay.sse import (StreamState, extract_usage,
                                finalize_tool_calls, iter_sse_events,
                                parse_sse_line, update_state)
from traffic_replay.client import EndpointClient, EndpointConfig


def test_role_only_chunk_is_not_content():
    st = StreamState()
    ev = parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}')
    assert update_state(st, ev) is False
    assert st.saw_first_content is False


def test_first_content_flags_once():
    st = StreamState()
    e1 = parse_sse_line('data: {"choices":[{"delta":{"content":"He"},"finish_reason":null}]}')
    e2 = parse_sse_line('data: {"choices":[{"delta":{"content":"llo"},"finish_reason":null}]}')
    assert update_state(st, e1) is True
    assert update_state(st, e2) is False
    assert st.content_chunks == 2


def test_done_and_finish_reason():
    st = StreamState()
    update_state(st, parse_sse_line(
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'))
    assert st.finish_reason == "stop"
    update_state(st, parse_sse_line("data: [DONE]"))
    assert st.done is True


def test_blank_and_comment_lines_ignored():
    assert parse_sse_line("") is None
    assert parse_sse_line(": keepalive") is None
    assert parse_sse_line("event: ping") is None


def test_parse_error_recorded_not_raised():
    st = StreamState()
    ev = parse_sse_line("data: {not-json-private-value")
    update_state(st, ev)
    assert st.errors and "invalid SSE JSON" in st.errors[0]
    assert "private-value" not in st.errors[0]
    assert "sha256=" in st.errors[0]


def test_duplicate_sse_keys_are_parse_errors_for_line_and_event_parsers():
    payload = ('{"choices":[{"delta":{"content":"first",'
               '"content":"second"}}]}')
    parsed = [
        parse_sse_line("data: " + payload),
        *iter_sse_events(["data: " + payload + "\n\n"]),
    ]

    assert len(parsed) == 2
    for event in parsed:
        st = StreamState()
        assert update_state(st, event) is False
        assert st.saw_first_content is False
        assert len(st.errors) == 1
        assert "invalid SSE JSON" in st.errors[0]
        assert "first" not in st.errors[0]
        assert "second" not in st.errors[0]
        assert "sha256=" in st.errors[0]


def test_nonfinite_sse_numbers_are_parse_errors_not_json_values():
    for payload in ('{"usage":{"prompt_tokens":NaN}}',
                    '{"usage":{"prompt_tokens":1e999}}'):
        parsed = [
            parse_sse_line("data: " + payload),
            *iter_sse_events(["data: " + payload + "\n\n"]),
        ]
        assert len(parsed) == 2
        for event in parsed:
            st = StreamState()
            update_state(st, event)
            assert st.usage is None
            assert st.errors and "invalid SSE JSON" in st.errors[0]


def test_invalid_utf8_is_hash_only_parse_error_never_visible_content():
    wire = b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n\n'
    parsed = [
        parse_sse_line(wire.splitlines()[0]),
        *iter_sse_events([wire]),
    ]

    assert len(parsed) == 2
    for event in parsed:
        st = StreamState()
        assert update_state(st, event) is False
        assert st.saw_first_content is False
        assert st.saw_first_visible is False
        assert len(st.errors) == 1
        assert "invalid SSE UTF-8" in st.errors[0]
        assert "sha256=" in st.errors[0]
        assert "\ufffd" not in st.errors[0]


def test_split_invalid_utf8_sequence_fails_the_event_safely():
    chunks = [b'data: {"choices":[{"delta":{"content":"\xc3',
              b'("}}]}\n\n']
    events = list(iter_sse_events(chunks))
    assert len(events) == 1
    st = StreamState()
    update_state(st, events[0])
    assert st.saw_first_visible is False
    assert st.errors and "invalid SSE UTF-8" in st.errors[0]


def test_excessive_sse_nesting_degrades_to_parse_error():
    payload = "[" * 10_000 + "0" + "]" * 10_000
    parsed = [
        parse_sse_line("data: " + payload),
        *iter_sse_events(["data: " + payload + "\n\n"]),
    ]
    assert len(parsed) == 2
    for event in parsed:
        st = StreamState()
        update_state(st, event)
        assert st.errors and "invalid SSE JSON" in st.errors[0]


def test_non_object_json_is_a_parse_error_not_a_crash():
    for payload in ("[]", "null", '"text"', "3"):
        st = StreamState()
        ev = parse_sse_line("data: " + payload)
        assert update_state(st, ev) is False
        assert st.errors


def test_unexpected_choice_shapes_are_recorded_not_raised():
    malformed = [
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{"delta": "not-an-object"}]},
        {"choices": [], "usage": []},
    ]
    for event in malformed:
        st = StreamState()
        assert update_state(st, event) is False
        assert st.errors


def test_whitespace_is_not_a_visible_answer():
    st = StreamState()
    event = parse_sse_line(
        'data: {"choices":[{"delta":{"content":"  \\n"}}]}')
    assert update_state(st, event) is True
    assert st.saw_first_content is True
    assert st.saw_first_visible is False


def test_structured_content_text_is_visible():
    st = StreamState()
    event = {"choices": [{"delta": {
        "content": [{"type": "text", "text": "hello"}]
    }}]}
    assert update_state(st, event) is True
    assert st.saw_first_visible is True


def test_tool_call_only_response_is_classified_separately():
    st = StreamState()
    event = {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "function": {"name": "lookup", "arguments": "{}"}
    }]}}]}
    assert update_state(st, event) is False
    assert st.saw_first_content is False
    assert st.saw_first_visible is False
    assert st.saw_first_tool_call is True
    assert st.tool_call_chunks == 1
    finalize_tool_calls(st)
    assert st.valid_tool_calls == 1


def test_refusal_delta_is_content_onset_but_not_visible_answer_content():
    st = StreamState()
    event = {"model": "glm", "object": "chat.completion.chunk",
             "choices": [{"delta": {"refusal": "blocked"},
                           "finish_reason": "stop"}]}

    assert update_state(st, event) is True
    assert st.saw_refusal is True
    assert st.refusal_chunks == 1
    assert st.saw_first_content is True
    assert st.saw_first_visible is False
    assert st.response_model == "glm"
    assert st.response_object == "chat.completion.chunk"
    assert st.errors == []


def test_conflicting_response_identity_is_a_protocol_error():
    st = StreamState()
    update_state(st, {"model": "glm-a", "id": "one", "choices": []})
    update_state(st, {"model": "glm-b", "id": "two", "choices": []})

    assert "stream reported conflicting model values" in st.errors
    assert "stream reported conflicting id values" in st.errors


def test_lone_surrogate_response_identity_is_a_protocol_error_not_state():
    st = StreamState()

    update_state(st, {"id": "\ud800", "model": "glm", "choices": []})

    assert st.response_id is None
    assert st.response_model == "glm"
    assert st.errors == [
        "stream event id contained a lone Unicode surrogate"]


def test_tool_fragment_state_has_a_cumulative_memory_bound():
    st = StreamState()
    fragment = "x" * 200_000
    for _ in range(10):
        update_state(st, {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": fragment, "arguments": fragment},
        }]}}]})

    retained = sum(len(value) for pieces in st._tool_names.values()
                   for value in pieces)
    retained += sum(len(value) for pieces in st._tool_arguments.values()
                    for value in pieces)
    assert retained <= 1024 * 1024
    assert any("cumulative tool fragment safety limit" in error
               for error in st.errors)


def test_fragmented_tool_call_is_validated_only_after_complete_json():
    st = StreamState()
    update_state(st, {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "call-1", "type": "function",
        "function": {"name": "look", "arguments": '{"city":'},
    }]}}]})
    update_state(st, {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "function": {"name": "up", "arguments": '"Paris"}'},
    }]}}]})
    assert st.saw_first_tool_call is True
    assert st.valid_tool_calls == 0
    finalize_tool_calls(st)
    assert st.valid_tool_calls == 1
    assert st.errors == []


def test_tool_fragments_from_distinct_choices_cannot_form_a_valid_call():
    st = StreamState()
    update_state(st, {"choices": [
        {"index": 0, "delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": "look", "arguments": '{"city":'},
        }]}},
    ]})
    update_state(st, {"choices": [
        {"index": 1, "delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": "up", "arguments": '"Paris"}'},
        }]}},
    ]})

    assert set(st._tool_names) == {(0, 0), (1, 0)}
    assert set(st._tool_arguments) == {(0, 0), (1, 0)}
    assert sum("multiple distinct choices" in error
               for error in st.errors) == 1

    finalize_tool_calls(st)

    assert st.valid_tool_calls == 0
    assert st._tool_names == {}
    assert st._tool_arguments == {}


def test_choice_index_is_validated_before_processing_delta():
    for invalid in (True, -1, "0", 1.5):
        st = StreamState()
        event = {"choices": [{
            "index": invalid,
            "delta": {"content": "must not be accepted"},
        }]}

        assert update_state(st, event) is False
        assert st.saw_first_content is False
        assert st.saw_first_visible is False
        assert st.content_chunks == 0
        assert len(st.errors) == 1
        assert "index must be a non-negative integer" in st.errors[0]


def test_single_nonzero_choice_index_preserves_fragmented_tool_call():
    st = StreamState()
    update_state(st, {"choices": [{
        "index": 7,
        "delta": {"tool_calls": [{
            "index": 2,
            "function": {"name": "look", "arguments": '{"city":'},
        }]},
    }]})
    update_state(st, {"choices": [{
        "index": 7,
        "delta": {"tool_calls": [{
            "index": 2,
            "function": {"name": "up", "arguments": '"Paris"}'},
        }]},
    }]})

    finalize_tool_calls(st)

    assert st.valid_tool_calls == 1
    assert st.errors == []


def test_invalid_tool_arguments_are_redacted_and_not_valid():
    st = StreamState()
    secret = "private-customer-value"
    update_state(st, {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "function": {"name": "lookup", "arguments": "{" + secret},
    }]}}]})
    finalize_tool_calls(st)
    assert st.valid_tool_calls == 0
    assert "invalid JSON" in st.errors[-1]
    assert "sha256=" in st.errors[-1]
    assert secret not in st.errors[-1]


def test_duplicate_tool_argument_keys_are_redacted_and_not_valid():
    st = StreamState()
    first = "private-first-value"
    second = "private-second-value"
    update_state(st, {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "function": {
            "name": "lookup",
            "arguments": f'{{"account":"{first}","account":"{second}"}}',
        },
    }]}}]})

    finalize_tool_calls(st)

    assert st.valid_tool_calls == 0
    assert "invalid JSON" in st.errors[-1]
    assert "sha256=" in st.errors[-1]
    assert first not in st.errors[-1]
    assert second not in st.errors[-1]


def test_nonfinite_tool_arguments_are_not_structurally_valid_json():
    for arguments in ('{"account":NaN}', '{"account":1e999}'):
        st = StreamState()
        update_state(st, {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"name": "lookup", "arguments": arguments},
        }]}}]})
        finalize_tool_calls(st)
        assert st.valid_tool_calls == 0
        assert st.errors and "invalid JSON" in st.errors[-1]
        assert "sha256=" in st.errors[-1]


def test_excessively_nested_tool_arguments_fail_without_recursion_error():
    arguments = "[" * 10_000 + "0" + "]" * 10_000
    st = StreamState()
    update_state(st, {"choices": [{"delta": {"tool_calls": [{
        "index": 0,
        "function": {"name": "lookup", "arguments": arguments},
    }]}}]})
    finalize_tool_calls(st)
    assert st.valid_tool_calls == 0
    assert st.errors and "invalid JSON" in st.errors[-1]
    assert "sha256=" in st.errors[-1]


def test_identical_singletons_may_repeat_but_conflicts_fail_closed():
    st = StreamState()
    finish = {"choices": [{"index": 0, "delta": {},
                            "finish_reason": "length"}]}
    usage = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    update_state(st, finish)
    update_state(st, finish)
    update_state(st, usage)
    update_state(st, usage)
    assert st.finish_reason == "length"
    assert st.usage == usage["usage"]
    assert st.errors == []

    conflicting_finish = {"choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}]}
    conflicting_usage = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    update_state(st, conflicting_finish)
    update_state(st, conflicting_finish)
    update_state(st, conflicting_usage)
    update_state(st, conflicting_usage)

    assert st.finish_reason == "length"
    assert st.usage == usage["usage"]
    assert st.errors == [
        "stream reported conflicting finish_reason values",
        "stream reported conflicting usage blocks",
    ]


def test_usage_repeat_comparison_is_json_type_sensitive():
    st = StreamState()
    update_state(st, {"usage": {"prompt_tokens": True}})
    update_state(st, {"usage": {"prompt_tokens": 1}})
    assert st.usage == {"prompt_tokens": 1}
    assert st.errors == [
        "stream usage prompt_tokens must be a non-negative integer"]


def test_progressive_databricks_usage_keeps_latest_cumulative_snapshot():
    """GLM reports one complete, cumulative usage object on every chunk."""
    st = StreamState()
    for completion_tokens in (1, 7, 13, 18, 24, 29, 33, 39, 45, 48, 64):
        update_state(st, {"service_tier": "default", "usage": {
            "prompt_tokens": 16,
            "completion_tokens": completion_tokens,
            "total_tokens": 16 + completion_tokens,
            "cache_read_input_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }})

    assert st.errors == []
    assert st.service_tier == "default"
    assert st.usage == {
        "prompt_tokens": 16,
        "completion_tokens": 64,
        "total_tokens": 80,
        "cache_read_input_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert extract_usage(st.usage) == {
        "prompt_tokens": 16,
        "completion_tokens": 64,
        "cached_tokens": 0,
        "cached_tokens_source": "prompt_tokens_details.cached_tokens",
        "reasoning_tokens": None,
        "reasoning_tokens_source": None,
    }


def test_progressive_usage_allows_later_output_details_but_not_input_changes():
    st = StreamState()
    update_state(st, {"usage": {
        "prompt_tokens": 20,
        "completion_tokens": 1,
        "total_tokens": 21,
        "completion_tokens_details": {"reasoning_tokens": None},
        "prompt_tokens_details": {"cached_tokens": 4},
    }})
    update_state(st, {"usage": {
        "prompt_tokens": 20,
        "completion_tokens": 9,
        "total_tokens": 29,
        "completion_tokens_details": {
            "reasoning_tokens": 7,
            "accepted_prediction_tokens": 2,
        },
        "prompt_tokens_details": {"cached_tokens": 4},
        "reasoning_tokens": 7,
    }})
    assert st.errors == []
    assert st.usage["completion_tokens"] == 9
    assert st.usage["completion_tokens_details"]["reasoning_tokens"] == 7

    changed_input = dict(st.usage)
    changed_input["prompt_tokens"] = 21
    update_state(st, {"usage": changed_input})
    assert st.usage["prompt_tokens"] == 20
    assert st.errors == [
        "stream usage total_tokens does not equal prompt_tokens plus "
        "completion_tokens"]


def test_progressive_usage_rejects_counter_regressions_and_missing_fields():
    base = {
        "prompt_tokens": 20,
        "completion_tokens": 9,
        "total_tokens": 29,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    for conflicting, expected in (
        ({**base, "completion_tokens": 8},
         "stream usage total_tokens does not equal prompt_tokens plus "
         "completion_tokens"),
        ({key: value for key, value in base.items() if key != "total_tokens"},
         "stream reported conflicting usage blocks"),
        ({**base, "prompt_tokens_details": {"cached_tokens": 5}},
         "stream reported conflicting usage blocks"),
    ):
        st = StreamState()
        update_state(st, {"usage": base})
        update_state(st, {"usage": conflicting})
        assert st.usage == base
        assert st.errors == [expected]


def test_service_tier_is_preserved_only_when_nonempty_and_stable():
    st = StreamState()
    update_state(st, {"service_tier": "default", "choices": []})
    update_state(st, {"service_tier": "default", "choices": []})
    assert st.service_tier == "default"
    assert st.errors == []

    update_state(st, {"service_tier": "priority", "choices": []})
    update_state(st, {"service_tier": "priority", "choices": []})
    assert st.service_tier == "default"
    assert st.errors == ["stream reported conflicting service_tier values"]


def test_invalid_service_tier_values_are_protocol_errors():
    for value in (None, "", "  ", 7, True, []):
        st = StreamState()
        update_state(st, {"service_tier": value, "choices": []})
        assert st.service_tier is None
        assert st.errors == [
            "stream event service_tier must be a non-empty string"]


def test_service_tier_is_bounded_and_rejects_lone_surrogates():
    cases = (
        ("x" * 513,
         "stream event service_tier exceeded the 512-character safety limit"),
        ("default\ud800",
         "stream event service_tier contained a lone Unicode surrogate"),
    )
    for value, expected in cases:
        st = StreamState()
        update_state(st, {"service_tier": value, "choices": []})
        assert st.service_tier is None
        assert st.errors == [expected]


def test_finish_reason_is_bounded_and_rejects_lone_surrogates():
    cases = (
        ("x" * 513,
         "stream choice 0 finish_reason exceeded the 512-character "
         "safety limit"),
        ("stop\ud800",
         "stream choice 0 finish_reason contained a lone Unicode surrogate"),
    )
    for value, expected in cases:
        st = StreamState()
        update_state(st, {"choices": [{
            "delta": {}, "finish_reason": value,
        }]})
        assert st.finish_reason is None
        assert st.errors == [expected]


def test_usage_arithmetic_invariants_fail_closed():
    cases = [
        ({"prompt_tokens": 10, "completion_tokens": 2,
          "total_tokens": 12,
          "prompt_tokens_details": {"cached_tokens": 11}},
         "cached tokens exceed prompt_tokens"),
        ({"prompt_tokens": 10, "completion_tokens": 2,
          "total_tokens": 12, "prompt_cache_hit_tokens": 11},
         "cached tokens exceed prompt_tokens"),
        ({"prompt_tokens": 10, "completion_tokens": 2,
          "total_tokens": 12,
          "completion_tokens_details": {"reasoning_tokens": 3}},
         "reasoning tokens exceed completion_tokens"),
        ({"prompt_tokens": 10, "completion_tokens": 2,
          "total_tokens": 12, "reasoning_tokens": 3},
         "reasoning tokens exceed completion_tokens"),
        ({"prompt_tokens": 10, "completion_tokens": 2,
          "total_tokens": 13},
         "total_tokens does not equal"),
        ({"prompt_tokens": -1, "completion_tokens": 2},
         "prompt_tokens must be a non-negative integer"),
        ({"prompt_tokens": 10, "completion_tokens_details": []},
         "completion_tokens_details must be an object or null"),
    ]
    for usage, expected in cases:
        st = StreamState()
        update_state(st, {"usage": usage})
        assert st.usage is None
        assert any(expected in error for error in st.errors)


def test_invalid_later_cumulative_usage_preserves_last_valid_snapshot():
    st = StreamState()
    valid = {
        "prompt_tokens": 16,
        "completion_tokens": 7,
        "total_tokens": 23,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    update_state(st, {"usage": valid})
    update_state(st, {"usage": {
        **valid,
        "completion_tokens": 9,
        "total_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }})
    assert st.usage == valid
    assert st.errors == [
        "stream usage reasoning tokens exceed completion_tokens at "
        "completion_tokens_details.reasoning_tokens"]


def test_multiline_sse_data_is_joined_and_eof_is_dispatched():
    lines = [
        ": comment\n",
        "event: message\n",
        'data: {"choices":\n',
        'data: [{"delta":{"content":"hello"}}]}\n',
        "\n",
        "data: [DONE]",
    ]
    events = list(iter_sse_events(lines))
    assert events == [
        {"choices": [{"delta": {"content": "hello"}}]},
        {"__done__": True},
    ]


def test_sse_incrementally_decodes_split_utf8_and_accepts_cr_line_endings():
    wire = ('data: {"choices":[{"delta":{"content":"café"}}]}'
            '\r\rdata: [DONE]\r').encode("utf-8")
    split = wire.index("é".encode("utf-8")) + 1
    events = list(iter_sse_events([wire[:split], wire[split:]]))
    assert events == [
        {"choices": [{"delta": {"content": "café"}}]},
        {"__done__": True},
    ]


def test_oversized_multiline_event_is_bounded_and_next_event_recovers():
    events = list(iter_sse_events([
        "data: 12345\n",
        "data: 67890\n",
        "\n",
        "data: {}\n\n",
    ], max_event_chars=8))
    assert len(events) == 2
    assert "exceeded 8" in events[0]["__parse_error__"]
    assert events[1] == {}


def test_sse_event_limit_must_be_a_positive_integer():
    import pytest

    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            list(iter_sse_events([], max_event_chars=value))


def test_malformed_tool_call_and_finish_reason_are_parse_errors():
    st = StreamState()
    update_state(st, {"choices": [{
        "delta": {"tool_calls": "not-structured"},
        "finish_reason": 7,
    }]})
    assert st.saw_first_tool_call is False
    assert st.tool_call_chunks == 0
    assert len(st.errors) == 2


def test_client_consumes_multiline_tool_call_stream_without_network():
    class _Socket:
        def settimeout(self, value):
            self.timeout = value

    class _Response:
        status = 200

        def __iter__(self):
            return iter([
                b'data: {"choices": [{"delta":\n',
                b'data: {"tool_calls": [{"index": 0, "function": '
                b'{"name": "lookup", "arguments": "{}"}}]}}]}\n',
                b'\n',
                b'data: {"choices":[{"delta":{},'
                b'"finish_reason":"tool_calls"}]}\n',
                b'\n',
                b'data: [DONE]\n',
                b'\n',
            ])

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/chat",
                       max_retries=0),
        token=None,
        refresh=None,
    )
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "look it up"}],
        20,
        "r1",
        0.0,
        0.0,
        (3, 20, None, -1),
        10,
    )
    assert result.ok is True
    assert result.status == 200
    assert result.tool_call_seen is True
    assert result.tool_call_chunks == 1
    assert result.valid_tool_calls == 1
    assert result.ttf_tool_call_ms is not None
    assert result.visible_content_seen is False
    assert result.ttft_ms is None
    assert result.finish_reason == "tool_calls"
    assert result.stream_complete is True


def test_client_rejects_tool_call_spliced_across_stream_choices():
    first = {"choices": [{"index": 0, "delta": {"tool_calls": [{
        "index": 0,
        "function": {"name": "look", "arguments": '{"city":'},
    }]}}]}
    second = {"choices": [{"index": 1, "delta": {"tool_calls": [{
        "index": 0,
        "function": {"name": "up", "arguments": '"Paris"}'},
    }]}}]}

    class _Socket:
        def settimeout(self, value):
            self.timeout = value

    class _Response:
        status = 200

        def __iter__(self):
            return iter([
                ("data: " + json.dumps(first) + "\n\n").encode(),
                ("data: " + json.dumps(second) + "\n\n").encode(),
                b"data: [DONE]\n\n",
            ])

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/chat",
                       max_retries=0),
        token=None,
        refresh=None,
    )
    client._connect = _Connection

    result = client.send(
        [{"role": "user", "content": "look it up"}],
        20,
        "r-multiple-choices",
        0.0,
        0.0,
        (3, 20, None, -1),
        10,
    )

    assert result.status == 200
    assert result.ok is False
    assert result.parse_errors == 1
    assert result.parse_error_details == [
        "stream returned multiple distinct choices; the benchmark requires "
        "exactly one response per request"
    ]
    assert result.tool_call_seen is True
    assert result.valid_tool_calls == 0
    assert result.stream_complete is True


def test_client_uses_final_progressive_usage_without_a_parse_error():
    usage_blocks = [
        {"prompt_tokens": 16, "completion_tokens": 1, "total_tokens": 17},
        {"prompt_tokens": 16, "completion_tokens": 7, "total_tokens": 23},
        {"prompt_tokens": 16, "completion_tokens": 7, "total_tokens": 23},
    ]
    events = [
        {"choices": [{"index": 0,
                      "delta": {"reasoning_content": "fragment"},
                      "finish_reason": None}],
         "service_tier": "default",
         "usage": usage_blocks[0]},
        {"choices": [{"index": 0,
                      "delta": {"reasoning_content": "fragment"},
                      "finish_reason": None}],
         "service_tier": "default",
         "usage": usage_blocks[1]},
        {"choices": [{"index": 0, "delta": {},
                      "finish_reason": "length"}],
         "service_tier": "default",
         "usage": usage_blocks[2]},
    ]

    class _Socket:
        def settimeout(self, value):
            self.timeout = value

    class _Response:
        status = 200

        def __iter__(self):
            lines = [
                ("data: " + json.dumps(event) + "\n\n").encode()
                for event in events
            ]
            return iter([*lines, b"data: [DONE]\n\n"])

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/chat",
                       max_retries=0),
        token=None,
        refresh=None,
    )
    client._connect = _Connection
    result = client.send(
        [{"role": "user", "content": "answer"}],
        7,
        "progressive-usage",
        0.0,
        0.0,
        (16, 7, None, -1),
        6,
    )

    assert result.status == 200
    assert result.ok is True
    assert result.prompt_tokens == 16
    assert result.completion_tokens == 7
    assert result.service_tier == "default"
    assert result.reasoning_chunks == 2
    assert result.truncated is True
    assert result.parse_errors == 0
    assert result.parse_error_details == []


def _send_protocol_events(events, *, done: bool):
    class _Socket:
        def settimeout(self, value):
            self.timeout = value

    class _Response:
        status = 200

        def __iter__(self):
            lines = [
                ("data: " + json.dumps(event) + "\n\n").encode()
                for event in events
            ]
            if done:
                lines.append(b"data: [DONE]\n\n")
            return iter(lines)

    class _Connection:
        def __init__(self):
            self.sock = _Socket()

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return _Response()

        def close(self):
            pass

    client = EndpointClient(
        EndpointConfig(base_url="http://127.0.0.1:1", path="/chat"), None)
    client._connect = _Connection
    return client.send(
        [{"role": "user", "content": "answer"}],
        16, "protocol", 0.0, 0.0, (8, 16, None, -1), 6)


def test_client_rejects_content_from_an_incomplete_stream():
    result = _send_protocol_events([{
        "choices": [{"index": 0, "delta": {"content": "answer"},
                     "finish_reason": None}],
        "service_tier": "default",
    }], done=False)

    assert result.status == 200
    assert result.ok is False
    assert result.error == "stream ended without [DONE] or a finish_reason"
    assert result.stream_complete is False
    assert result.visible_content_seen is True
    assert result.service_tier == "default"
    assert result.parse_errors == 0


def test_client_rejects_usage_corruption_even_with_content_and_terminal_event():
    result = _send_protocol_events([{
        "choices": [{"index": 0, "delta": {"content": "answer"},
                     "finish_reason": "stop"}],
        "service_tier": "priority",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 11,
        },
    }], done=True)

    assert result.status == 200
    assert result.ok is False
    assert result.error == "stream protocol validation failed"
    assert result.stream_complete is True
    assert result.visible_content_seen is True
    assert result.service_tier == "priority"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.parse_errors == 1
    assert result.parse_error_details == [
        "stream usage total_tokens does not equal prompt_tokens plus "
        "completion_tokens"]


def test_client_rejects_conflicting_service_tier_and_preserves_first_value():
    result = _send_protocol_events([
        {
            "choices": [{"index": 0, "delta": {"content": "answer"},
                         "finish_reason": None}],
            "service_tier": "default",
        },
        {
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "stop"}],
            "service_tier": "priority",
        },
    ], done=True)

    assert result.ok is False
    assert result.error == "stream protocol validation failed"
    assert result.stream_complete is True
    assert result.service_tier == "default"
    assert result.parse_error_details == [
        "stream reported conflicting service_tier values"]


def test_client_preserves_a_clean_stable_service_tier():
    result = _send_protocol_events([
        {
            "choices": [{"index": 0, "delta": {"content": "answer"},
                         "finish_reason": None}],
            "service_tier": "priority",
        },
        {
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "stop"}],
            "service_tier": "priority",
        },
    ], done=True)

    assert result.ok is True
    assert result.error is None
    assert result.stream_complete is True
    assert result.service_tier == "priority"
    assert result.parse_errors == 0


def test_usage_openai_style():
    u = extract_usage({"prompt_tokens": 100, "completion_tokens": 10,
                       "prompt_tokens_details": {"cached_tokens": 60}})
    assert u["cached_tokens"] == 60
    assert u["cached_tokens_source"] == "prompt_tokens_details.cached_tokens"


def test_usage_deepseek_style_and_flat():
    u = extract_usage({"prompt_tokens": 100, "prompt_cache_hit_tokens": 42})
    assert u["cached_tokens"] == 42
    u2 = extract_usage({"prompt_tokens": 100, "cached_tokens": 7})
    assert u2["cached_tokens"] == 7


def test_usage_absent_is_none_never_guessed():
    u = extract_usage(None)
    assert u["prompt_tokens"] is None and u["cached_tokens"] is None
    u2 = extract_usage({"prompt_tokens": 50})
    assert u2["cached_tokens"] is None and u2["cached_tokens_source"] is None


def test_usage_rejects_invalid_token_counts_without_crashing():
    for usage in ([], "bad", {"prompt_tokens": -1},
                  {"prompt_tokens": True}, {"prompt_tokens": float("nan")},
                  {"prompt_tokens": 10.9}):
        u = extract_usage(usage)
        assert u["prompt_tokens"] is None
    u = extract_usage({
        "prompt_tokens": 10.0,
        "completion_tokens": 2.0,
        "prompt_tokens_details": {"cached_tokens": -5},
    })
    assert u["prompt_tokens"] == 10
    assert u["completion_tokens"] == 2
    assert u["cached_tokens"] is None
