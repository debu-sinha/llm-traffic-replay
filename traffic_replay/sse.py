"""Minimal, dependency-free Server-Sent Events parsing for OpenAI-style
streaming chat completions.

The client feeds raw lines; this module yields parsed events and extracts
the fields the harness measures: first content token, usage block, finish.
Kept separate from the HTTP layer so it is unit-testable against fixtures.
"""
from __future__ import annotations

import codecs
import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .json_input import loads_strict


@dataclass
class StreamState:
    saw_first_content: bool = False
    saw_first_visible: bool = False       # first visible content delta
    saw_first_reasoning: bool = False     # first reasoning-channel delta
    saw_first_tool_call: bool = False     # first tool/function-call delta
    content_chunks: int = 0
    reasoning_chunks: int = 0             # count of reasoning-channel deltas
    tool_call_chunks: int = 0
    valid_tool_calls: int = 0
    finish_reason: str | None = None
    usage: dict | None = None
    service_tier: str | None = None
    done: bool = False
    errors: list[str] = field(default_factory=list)
    _tool_names: dict[tuple[int, int], list[str]] = field(
        default_factory=dict, repr=False)
    _tool_arguments: dict[tuple[int, int], list[str]] = field(
        default_factory=dict, repr=False)
    _choice_indexes_seen: set[int] = field(default_factory=set, repr=False)
    _multiple_choices_reported: bool = field(default=False, repr=False)
    _conflicting_finish_reported: bool = field(default=False, repr=False)
    _conflicting_usage_reported: bool = field(default=False, repr=False)
    _conflicting_service_tier_reported: bool = field(
        default=False, repr=False)


def _safe_parse_error(kind: str, payload: str | bytes) -> dict:
    """Return diagnostic metadata without persisting streamed content."""
    encoded = (payload if isinstance(payload, bytes)
               else payload.encode("utf-8", "surrogatepass"))
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return {"__parse_error__":
            f"{kind} (payload bytes={len(encoded)}, sha256={digest})"}


def parse_sse_line(line: bytes | str) -> dict | None:
    """Return the JSON payload of a `data:` line, {'__done__': True} for
    [DONE], or None for blanks/comments/other fields."""
    if isinstance(line, bytes):
        raw_line = line
        try:
            line = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return _safe_parse_error("invalid SSE UTF-8", raw_line)
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return {"__done__": True}
    try:
        event = loads_strict(payload)
    except ValueError:
        return _safe_parse_error("invalid SSE JSON", payload)
    if not isinstance(event, dict):
        return _safe_parse_error(
            f"SSE data must be a JSON object, got {type(event).__name__}",
            payload)
    return event


def iter_sse_events(lines: Iterable[bytes | str],
                    max_event_chars: int = 4 * 1024 * 1024
                    ) -> Iterator[dict]:
    """Yield complete SSE ``data`` events from an iterable of raw lines.

    SSE permits an event to contain multiple ``data:`` fields. Their values
    are joined with newlines and dispatched by a blank line. OpenAI-compatible
    servers normally use one data field per event, but treating each physical
    line as a complete event corrupts otherwise valid multiline streams.

    Non-data fields and comments are ignored. A final unterminated event is
    dispatched at EOF, which is useful for defensive interoperability with
    servers that omit the last blank line.
    """
    if not isinstance(max_event_chars, int) or isinstance(max_event_chars, bool) \
            or max_event_chars <= 0:
        raise ValueError("max_event_chars must be a positive integer")

    data: list[str] = []
    data_chars = 0
    discard_event = False
    buffered = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    at_stream_start = True

    def dispatch() -> dict | None:
        nonlocal data_chars, discard_event
        if discard_event:
            data.clear()
            data_chars = 0
            discard_event = False
            return None
        if not data:
            return None
        payload = "\n".join(data)
        data.clear()
        data_chars = 0
        if len(payload) > max_event_chars:
            return {"__parse_error__":
                    f"SSE event exceeded {max_event_chars} characters"}
        if payload.strip() == "[DONE]":
            return {"__done__": True}
        try:
            event = loads_strict(payload)
        except ValueError:
            return _safe_parse_error("invalid SSE JSON", payload)
        if not isinstance(event, dict):
            return _safe_parse_error(
                f"SSE data must be a JSON object, got {type(event).__name__}",
                payload)
        return event

    def consume_line(line: str) -> dict | None:
        nonlocal data_chars, discard_event
        if not line:
            return dispatch()
        if discard_event:
            return None
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if field != "data":
            return None
        if separator and value.startswith(" "):
            value = value[1:]
        added = len(value) + (1 if data else 0)
        if data_chars + added > max_event_chars:
            data.clear()
            data_chars = 0
            discard_event = True
            return {"__parse_error__":
                    f"SSE event exceeded {max_event_chars} characters"}
        data.append(value)
        data_chars += added
        return None

    def decoded_chunks() -> Iterator[str | dict]:
        """Decode bytes incrementally so UTF-8 code points may cross chunks."""
        nonlocal decoder
        for raw in lines:
            if isinstance(raw, bytes):
                try:
                    yield decoder.decode(raw, final=False)
                except UnicodeDecodeError as exc:
                    yield _safe_parse_error(
                        "invalid SSE UTF-8", bytes(exc.object))
                    return
            elif isinstance(raw, str):
                # Mixed byte/string streams are unusual, but flushing pending
                # byte state avoids joining half a code point to native text.
                try:
                    pending = decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    yield _safe_parse_error(
                        "invalid SSE UTF-8", bytes(exc.object))
                    return
                decoder = codecs.getincrementaldecoder("utf-8")(
                    errors="strict")
                yield pending + raw
            else:
                raise TypeError("SSE chunks must be bytes or strings")
        try:
            yield decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            yield _safe_parse_error("invalid SSE UTF-8", bytes(exc.object))

    for text in decoded_chunks():
        if isinstance(text, dict):
            yield text
            return
        if at_stream_start and text:
            text = text.removeprefix("\ufeff")
            at_stream_start = False
        buffered += text
        while True:
            lf = buffered.find("\n")
            cr = buffered.find("\r")
            indexes = [x for x in (lf, cr) if x >= 0]
            if not indexes:
                break
            end = min(indexes)
            # A terminal CR might be the first half of CRLF in the next
            # network chunk. Waiting preserves one logical blank separator.
            if buffered[end] == "\r" and end + 1 == len(buffered):
                break
            separator_len = (2 if buffered[end:end + 2] == "\r\n" else 1)
            line = buffered[:end]
            buffered = buffered[end + separator_len:]
            event = consume_line(line)
            if event is not None:
                yield event
        if len(buffered) > max_event_chars:
            yield {"__parse_error__":
                   f"SSE line exceeded {max_event_chars} characters"}
            buffered = ""
            data.clear()
            data_chars = 0
            discard_event = True

    if buffered:
        if buffered.endswith("\r"):
            buffered = buffered[:-1]
        event = consume_line(buffered)
        if event is not None:
            yield event
    event = dispatch()
    if event is not None:
        yield event


def _meaningful_text(value: object) -> bool:
    """Whether a provider content value contains user-visible text."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        for part in value:
            if isinstance(part, str) and part.strip():
                return True
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    if isinstance(value, dict):
        text = value.get("text")
        return isinstance(text, str) and bool(text.strip())
    return False


def _nonempty_delta(value: object) -> bool:
    """Whether a delta represents at least one emitted stream fragment."""
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, dict)):
        return bool(value)
    return False


def _usage_counter_may_increase(path: tuple[str | int, ...]) -> bool:
    """Whether ``path`` is an output counter in a cumulative usage block.

    Databricks-hosted GLM emits a complete usage object on every streamed
    chunk. Prompt/cache counts stay fixed while generated-token counters grow.
    OpenAI-compatible providers can put the output breakdown in a nested
    ``*_tokens_details`` object, so those numeric leaves are cumulative too.
    All other existing values must remain exactly equal.
    """
    if not path or not isinstance(path[0], str):
        return False
    return path[0] in {
        "completion_tokens",
        "completion_tokens_details",
        "output_tokens",
        "output_tokens_details",
        "reasoning_tokens",
        "total_tokens",
    }


def _usage_is_monotonic_extension(previous: dict, current: dict) -> bool:
    """Accept a later complete/cumulative usage snapshot, fail closed otherwise.

    Existing fields may not disappear. New fields and values replacing a
    prior ``null`` are evidence becoming more complete. Known output counters
    may increase, but input/cache metadata and unknown fields are immutable.
    The iterative walk avoids recursion failures on adversarial nesting.
    """
    stack: list[tuple[tuple[str | int, ...], object, object, bool]] = [
        ((), previous, current, True)
    ]
    while stack:
        path, left, right, allow_counter_progress = stack.pop()
        if left is None and right is not None:
            continue
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if not left.keys() <= right.keys():
                return False
            stack.extend(
                (path + (key,), left[key], right[key],
                 allow_counter_progress)
                for key in left
            )
            continue
        if isinstance(left, list):
            if len(left) != len(right):
                return False
            stack.extend(
                (path + (position,), old, new, False)
                for position, (old, new) in enumerate(zip(left, right))
            )
            continue
        if left == right:
            continue
        if allow_counter_progress and _usage_counter_may_increase(path):
            old_count = _token_count(left)
            new_count = _token_count(right)
            if old_count is not None and new_count is not None \
                    and new_count >= old_count:
                continue
        return False
    return True


def _usage_path_value(usage: dict,
                      path: tuple[str, ...]) -> tuple[bool, object]:
    """Return whether one exact usage path exists, including a null leaf."""
    node: object = usage
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _usage_invariant_errors(usage: dict) -> list[str]:
    """Validate recognized counters and their provider-independent algebra."""
    errors: list[str] = []

    for container in ("prompt_tokens_details", "completion_tokens_details"):
        if container in usage and usage[container] is not None \
                and not isinstance(usage[container], dict):
            errors.append(
                f"stream usage {container} must be an object or null")

    def count(path: tuple[str, ...]) -> int | None:
        present, raw = _usage_path_value(usage, path)
        if not present or raw is None:
            return None
        parsed = _token_count(raw)
        if parsed is None:
            errors.append(
                "stream usage " + ".".join(path)
                + " must be a non-negative integer")
        return parsed

    prompt = count(("prompt_tokens",))
    completion = count(("completion_tokens",))
    total = count(("total_tokens",))

    for path in CACHED_TOKEN_PATHS:
        cached = count(path)
        if cached is not None and prompt is not None and cached > prompt:
            errors.append(
                "stream usage cached tokens exceed prompt_tokens at "
                + ".".join(path))
    for path in REASONING_TOKEN_PATHS:
        reasoning = count(path)
        if reasoning is not None and completion is not None \
                and reasoning > completion:
            errors.append(
                "stream usage reasoning tokens exceed completion_tokens at "
                + ".".join(path))
    if prompt is not None and completion is not None and total is not None \
            and total != prompt + completion:
        errors.append(
            "stream usage total_tokens does not equal prompt_tokens plus "
            "completion_tokens")
    return errors


def _update_tool_calls(state: StreamState, value: object,
                       choice_index: int) -> bool:
    """Validate and retain only the structure needed to judge tool calls.

    Argument text is held only until the stream finishes so its assembled JSON
    can be validated; it is never copied into request artifacts.
    """
    legacy = isinstance(value, dict)
    items = [value] if legacy else value
    if not isinstance(items, list):
        state.errors.append(
            f"stream choice {choice_index} tool call must be an object or list")
        return False
    meaningful = False
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            state.errors.append(
                f"stream choice {choice_index} tool call {position} must be "
                "an object")
            continue
        index = item.get("index", position)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            state.errors.append(
                f"stream choice {choice_index} tool call index must be a "
                "non-negative integer")
            continue
        call_key = (choice_index, index)
        function = item if legacy else item.get("function")
        fragment = False
        for metadata in ("id", "type"):
            if metadata in item:
                field_value = item[metadata]
                if not isinstance(field_value, str):
                    state.errors.append(
                        f"stream choice {choice_index} tool call {metadata} "
                        "must be a string")
                elif field_value:
                    fragment = True
        if function is not None:
            if not isinstance(function, dict):
                state.errors.append(
                    f"stream choice {choice_index} tool call function must "
                    "be an object")
            else:
                if "name" in function:
                    name = function["name"]
                    if not isinstance(name, str):
                        state.errors.append(
                            f"stream choice {choice_index} tool call name "
                            "must be a string")
                    elif name:
                        state._tool_names.setdefault(call_key, []).append(name)
                        fragment = True
                if "arguments" in function:
                    arguments = function["arguments"]
                    if not isinstance(arguments, str):
                        state.errors.append(
                            f"stream choice {choice_index} tool call arguments "
                            "must be a string")
                    else:
                        state._tool_arguments.setdefault(call_key, []).append(
                            arguments)
                        fragment = True
        if not fragment:
            state.errors.append(
                f"stream choice {choice_index} tool call {position} was empty")
        meaningful = meaningful or fragment
    return meaningful


def finalize_tool_calls(state: StreamState) -> None:
    """Validate complete tool names and JSON arguments after all deltas."""
    if not state.saw_first_tool_call:
        return
    indexes = set(state._tool_names) | set(state._tool_arguments)
    valid = 0
    if len(state._choice_indexes_seen) > 1:
        state.valid_tool_calls = 0
        state._tool_names.clear()
        state._tool_arguments.clear()
        return
    for choice_index, tool_index in sorted(indexes):
        call_key = (choice_index, tool_index)
        label = f"stream choice {choice_index} tool call {tool_index}"
        name = "".join(state._tool_names.get(call_key, [])).strip()
        arguments = "".join(state._tool_arguments.get(call_key, []))
        if not name:
            state.errors.append(f"{label} did not identify a function name")
            continue
        if not arguments:
            state.errors.append(f"{label} did not provide JSON arguments")
            continue
        try:
            parsed = loads_strict(arguments)
        except ValueError:
            # Never persist argument content; it may contain customer data.
            encoded = arguments.encode("utf-8", "replace")
            digest = hashlib.sha256(encoded).hexdigest()[:16]
            state.errors.append(
                f"{label} arguments were invalid JSON "
                f"(bytes={len(encoded)}, sha256={digest})")
            continue
        if not isinstance(parsed, dict):
            state.errors.append(
                f"{label} arguments must decode to an object")
            continue
        valid += 1
    state.valid_tool_calls = valid
    state._tool_names.clear()
    state._tool_arguments.clear()


def update_state(state: StreamState, event: object) -> bool:
    """Fold one event into state. Returns True if this event carries the
    FIRST content delta (the TTFT moment)."""
    if not isinstance(event, dict):
        state.errors.append(
            f"stream event must be an object, got {type(event).__name__}")
        return False
    if event.get("__done__"):
        state.done = True
        return False
    if "__parse_error__" in event:
        state.errors.append(event["__parse_error__"])
        return False

    if "service_tier" in event:
        service_tier = event["service_tier"]
        if not isinstance(service_tier, str) or not service_tier.strip():
            if "stream event service_tier must be a non-empty string" \
                    not in state.errors:
                state.errors.append(
                    "stream event service_tier must be a non-empty string")
        elif state.service_tier is None:
            state.service_tier = service_tier
        elif service_tier != state.service_tier \
                and not state._conflicting_service_tier_reported:
            state.errors.append(
                "stream reported conflicting service_tier values")
            state._conflicting_service_tier_reported = True

    choices = event.get("choices")
    if choices is None:
        choices = []
    elif not isinstance(choices, list):
        state.errors.append("stream event choices must be a list")
        choices = []

    first_content = False
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            state.errors.append(
                f"stream choice {position} must be an object, got "
                f"{type(choice).__name__}")
            continue
        choice_index = choice.get("index", position)
        if not isinstance(choice_index, int) \
                or isinstance(choice_index, bool) or choice_index < 0:
            state.errors.append(
                f"stream choice {position} index must be a non-negative "
                "integer")
            continue
        state._choice_indexes_seen.add(choice_index)
        if len(state._choice_indexes_seen) > 1 \
                and not state._multiple_choices_reported:
            state.errors.append(
                "stream returned multiple distinct choices; the benchmark "
                "requires exactly one response per request")
            state._multiple_choices_reported = True
        delta = choice.get("delta")
        if delta is None:
            delta = {}
        elif not isinstance(delta, dict):
            state.errors.append(
                f"stream choice {choice_index} delta must be an object")
            delta = {}
        visible = delta.get("content")
        reasoning = delta.get("reasoning_content")
        tool_call = delta.get("tool_calls") or delta.get("function_call")
        has_visible_delta = _nonempty_delta(visible)
        has_reasoning_delta = _nonempty_delta(reasoning)
        has_tool_call_delta = (
            _update_tool_calls(state, tool_call, choice_index)
            if tool_call is not None else False)
        if has_visible_delta or has_reasoning_delta:
            state.content_chunks += 1
            if not state.saw_first_content:
                state.saw_first_content = True
                first_content = True
        if has_reasoning_delta:
            state.reasoning_chunks += 1
        if has_reasoning_delta and not state.saw_first_reasoning:
            state.saw_first_reasoning = True
        if _meaningful_text(visible) and not state.saw_first_visible:
            state.saw_first_visible = True
        if has_tool_call_delta:
            state.tool_call_chunks += 1
            state.saw_first_tool_call = True
        fr = choice.get("finish_reason")
        if isinstance(fr, str) and fr:
            if state.finish_reason is None:
                state.finish_reason = fr
            elif fr != state.finish_reason \
                    and not state._conflicting_finish_reported:
                state.errors.append(
                    "stream reported conflicting finish_reason values")
                state._conflicting_finish_reported = True
        elif fr is not None:
            state.errors.append(
                f"stream choice {choice_index} finish_reason must be a string")

    usage = event.get("usage")
    if usage is not None:
        if isinstance(usage, dict):
            invariant_errors = _usage_invariant_errors(usage)
            for detail in invariant_errors:
                if detail not in state.errors:
                    state.errors.append(detail)
            if not invariant_errors:
                if state.usage is None:
                    state.usage = usage
                elif _usage_is_monotonic_extension(state.usage, usage):
                    # Retain the newest cumulative snapshot. Keeping the first
                    # block undercounts Databricks GLM streams because the first
                    # streamed delta reports only the tokens generated so far.
                    state.usage = usage
                elif not state._conflicting_usage_reported:
                    state.errors.append(
                        "stream reported conflicting usage blocks")
                    state._conflicting_usage_reported = True
        else:
            state.errors.append("stream event usage must be an object")
    return first_content


# Known field paths for cached prompt tokens across providers. Checked in
# order; the first present wins. The report records WHICH path was found.
CACHED_TOKEN_PATHS = (
    ("prompt_tokens_details", "cached_tokens"),   # OpenAI-style
    ("prompt_cache_hit_tokens",),                 # DeepSeek-style
    ("cached_tokens",),                           # flat variants
    ("cache_read_input_tokens",),                 # Anthropic-style naming
)

# Reasoning (thinking) token counts, same convention.
REASONING_TOKEN_PATHS = (
    ("completion_tokens_details", "reasoning_tokens"),   # OpenAI o-series
    ("reasoning_tokens",),                               # flat variants
)


def _walk(usage: dict, paths) -> tuple[int | None, str | None]:
    """First present integer at any of `paths`, with its dotted source."""
    for path in paths:
        node = usage
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node and node[key] is not None:
                node = node[key]
            else:
                ok = False
                break
        parsed = _token_count(node) if ok else None
        if parsed is not None:
            return parsed, ".".join(path)
    return None, None


def _token_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) \
            and value >= 0 and value.is_integer():
        return int(value)
    return None


def extract_usage(usage: dict | None) -> dict:
    """Normalize a usage block. Absent fields come back None, never guessed."""
    if not isinstance(usage, dict) or not usage:
        return {"prompt_tokens": None, "completion_tokens": None,
                "cached_tokens": None, "cached_tokens_source": None,
                "reasoning_tokens": None, "reasoning_tokens_source": None}
    cached, cached_src = _walk(usage, CACHED_TOKEN_PATHS)
    reasoning, reasoning_src = _walk(usage, REASONING_TOKEN_PATHS)
    return {
        "prompt_tokens": _token_count(usage.get("prompt_tokens")),
        "completion_tokens": _token_count(usage.get("completion_tokens")),
        "cached_tokens": cached,
        "cached_tokens_source": cached_src,
        "reasoning_tokens": reasoning,
        "reasoning_tokens_source": reasoning_src,
    }
