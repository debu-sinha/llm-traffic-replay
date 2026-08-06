"""SSE parsing: TTFT keys on first CONTENT delta (role-only chunks must not
trigger it), usage extraction is defensive across provider field names."""
from traffic_replay.sse import (StreamState, extract_usage, iter_sse_events,
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
        "index": 0, "function": {"name": "lookup"}
    }]}}]}
    assert update_state(st, event) is False
    assert st.saw_first_content is False
    assert st.saw_first_visible is False
    assert st.saw_first_tool_call is True
    assert st.tool_call_chunks == 1


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
                b'{"name": "lookup"}}]}}]}\n',
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
    assert result.ttf_tool_call_ms is not None
    assert result.visible_content_seen is False
    assert result.ttft_ms is None
    assert result.finish_reason == "tool_calls"
    assert result.stream_complete is True


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
