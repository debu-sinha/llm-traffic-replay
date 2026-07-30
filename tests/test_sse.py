"""SSE parsing: TTFT keys on first CONTENT delta (role-only chunks must not
trigger it), usage extraction is defensive across provider field names."""
from traffic_replay.sse import (StreamState, extract_usage, parse_sse_line,
                                update_state)


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
    ev = parse_sse_line("data: {not json")
    update_state(st, ev)
    assert st.errors and "not json" in st.errors[0]


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
