"""Versioned endpoint-protocol adapters for the measurement client.

The scheduler, quota guard, metrics, and report code operate on one canonical
response state.  An adapter owns only wire-dialect behavior: request envelope,
response framing, event folding, usage normalization, and protocol-specific
fallback detection.  A new model that implements an existing dialect needs no
code change; a new dialect is added here (or registered by an embedding
application) and exercised by the adapter conformance suite.

Adapter identifiers are persisted in run evidence.  Their behavior must never
change incompatibly in place: introduce a new ``/vN`` identifier instead.
"""
from __future__ import annotations

import builtins
import json
import hashlib
import re
import threading
import types
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .sse import (
    StreamState,
    extract_usage,
    finalize_tool_calls,
    iter_sse_events,
    update_state,
)


DEFAULT_ENDPOINT_ADAPTER = "openai.chat_completions.sse/v1"
_CHAT_OWNED_FIELDS = frozenset({
    "messages",
    "max_tokens",
    "temperature",
    "stream",
    "model",
    "stream_options",
})


@dataclass(frozen=True)
class EventMilestones:
    """Canonical observations produced while folding one wire event."""

    first_content: bool = False
    first_reasoning: bool = False
    first_visible: bool = False
    first_tool_call: bool = False
    content_event: bool = False


@dataclass(frozen=True)
class ResponseOutcome:
    """Protocol-owned interpretation of one fully folded response.

    ``response_complete`` means the adapter observed the dialect's terminal
    boundary.  It does not imply that the model finished the requested task;
    ``truncated`` and downstream semantic/quality scoring remain separate.
    ``ok`` is limited to the response protocol and minimum output contract.
    """

    response_complete: bool
    output_observed: bool
    truncated: bool
    ok: bool
    error: str | None = None


class EndpointAdapter(ABC):
    """Stable extension contract between a wire protocol and the load engine.

    Implementations must be stateless and thread-safe.  Per-request state
    belongs in :class:`StreamState`, which deliberately contains no prompt or
    response text so benchmark artifacts cannot leak customer content.
    """

    adapter_id: str
    response_mode: str
    request_media_type = "application/json"
    accept_media_type: str
    accepted_response_media_types: tuple[str, ...]
    usage_request_mode: str

    @abstractmethod
    def validate_endpoint(self, endpoint: Any) -> None:
        """Reject an endpoint configuration incompatible with this dialect."""

    @abstractmethod
    def serialize_request(
        self,
        endpoint: Any,
        messages: list[dict],
        max_tokens: int,
        include_usage: bool,
    ) -> bytes:
        """Return the exact request bytes that will be sent on the wire."""

    @abstractmethod
    def iter_events(self, chunks: Iterable[bytes | str]) -> Iterator[dict]:
        """Parse bounded response chunks into provider events."""

    @abstractmethod
    def fold_event(
        self, state: StreamState, event: object
    ) -> EventMilestones:
        """Fold one provider event into canonical, content-free state."""

    @abstractmethod
    def finalize(self, state: StreamState) -> None:
        """Perform end-of-response validation and canonicalization."""

    @abstractmethod
    def normalize_usage(self, usage: dict | None) -> dict:
        """Map provider usage fields to canonical token counters."""

    def evaluate_outcome(self, state: StreamState) -> ResponseOutcome:
        """Classify terminal and output semantics for this wire dialect.

        The default is intentionally protocol-neutral.  Adapters with a
        provider-specific incomplete/failed state or truncation vocabulary
        must override it rather than teaching the HTTP client those terms.
        """
        response_complete = bool(state.done or state.finish_reason)
        output_observed = bool(
            state.saw_first_content or state.valid_tool_calls > 0)
        if state.errors:
            error = "response protocol validation failed"
        elif not response_complete:
            error = "response ended without a terminal event"
        elif not output_observed:
            error = "response ended with no content or valid tool call"
        else:
            error = None
        return ResponseOutcome(
            response_complete=response_complete,
            output_observed=output_observed,
            truncated=False,
            ok=error is None,
            error=error,
        )

    def include_usage_option_rejected(self, status: int, body: bytes) -> bool:
        """Whether this response rejects the adapter's optional usage control."""
        return False

    def probe_control_rejected(
        self, status: int, body: bytes, candidate: Mapping[str, object]
    ) -> bool:
        """Whether an HTTP error explicitly rejects this probe candidate.

        The body is a bounded, ephemeral response sample.  Implementations
        must return only a boolean classification; callers never retain its
        text.  The default is fail-closed because an HTTP 400/422 by itself
        does not establish which part of a request was rejected.
        """
        return False

    def request_headers(self, request_id: str) -> dict[str, str]:
        return {
            "Content-Type": self.request_media_type,
            "Accept": self.accept_media_type,
            "X-Request-Id": request_id,
        }

    def contract(self) -> dict:
        """Artifact-safe description of semantics that affect measurements."""
        return {
            "adapter_id": self.adapter_id,
            "response_mode": self.response_mode,
            "request_media_type": self.request_media_type,
            "accept_media_type": self.accept_media_type,
            "accepted_response_media_types": list(
                self.accepted_response_media_types),
            "usage_request_mode": self.usage_request_mode,
            "canonical_state": "traffic_replay.sse.StreamState/v1",
            "outcome_policy": (
                f"{type(self).__module__}.{type(self).__qualname__}."
                "evaluate_outcome"),
            "probe_rejection_policy": (
                f"{type(self).__module__}.{type(self).__qualname__}."
                "probe_control_rejected"),
        }


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _stream_options_rejected(body: bytes) -> bool:
    """Only retry a 400 that explicitly identifies our optional field."""
    text = body.decode("utf-8", "replace").casefold()
    names_field = "stream_options" in text or "include_usage" in text
    rejects_field = any(term in text for term in (
        "unsupported", "not supported", "unknown", "unrecognized",
        "unexpected", "not allowed", "not permitted", "cannot",
        "additional propert", "extra field", "invalid field",
        "invalid parameter",
    ))
    return names_field and rejects_field


_PROBE_ERROR_KEYS = frozenset({
    "argument",
    "cause",
    "detail",
    "error",
    "errors",
    "field",
    "loc",
    "location",
    "message",
    "msg",
    "param",
    "parameter",
    "path",
    "property",
    "reason",
    "type",
})
_PROBE_REJECTION_SUFFIXES = (
    "cannot be used",
    "extra field is not permitted",
    "extra fields are not permitted",
    "extra input is not permitted",
    "extra inputs are not permitted",
    "has an invalid value",
    "has invalid value",
    "is invalid",
    "is not allowed",
    "is not permitted",
    "is not supported",
    "may not be used",
    "not a valid argument",
    "not a valid field",
    "not a valid option",
    "not a valid parameter",
    "not allowed",
    "not permitted",
    "not supported",
    "unsupported",
    "value is not allowed",
    "value is not permitted",
    "value is not supported",
)


def _probe_candidate_identifiers(
    candidate: Mapping[str, object],
) -> tuple[str, ...]:
    """Return exact field and nested-path spellings for one candidate."""
    identifiers: set[str] = set()

    def walk(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                if not isinstance(raw_key, str) or not raw_key:
                    continue
                child_path = (*path, raw_key)
                identifiers.add(raw_key.casefold())
                identifiers.add(".".join(child_path).casefold())
                identifiers.add(("/" + "/".join(child_path)).casefold())
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child, path)

    walk(candidate, ())
    return tuple(sorted(identifiers, key=lambda item: (-len(item), item)))


def _probe_error_diagnostics(body: bytes) -> tuple[str, ...]:
    """Extract only bounded provider diagnostic strings, never request echoes."""
    decoded = body.decode("utf-8", "replace")
    try:
        parsed = json.loads(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        compact = " ".join(decoded.casefold().split())
        return (compact,) if compact else ()

    diagnostics: list[str] = []

    def primitive_strings(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out: list[str] = []
            for child in value:
                if isinstance(child, (str, int, float)) \
                        and not isinstance(child, bool):
                    out.append(str(child))
            return out
        return []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            direct: list[str] = []
            for raw_key, child in value.items():
                key = raw_key.casefold() if isinstance(raw_key, str) else ""
                if key in _PROBE_ERROR_KEYS:
                    direct.extend(primitive_strings(child))
                if isinstance(child, (dict, list)):
                    walk(child)
            compact = " ".join(" ".join(direct).casefold().split())
            if compact:
                diagnostics.append(compact)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            compact = " ".join(value.casefold().split())
            if compact:
                diagnostics.append(compact)

    walk(parsed)
    return tuple(dict.fromkeys(diagnostics))


def _chat_probe_control_rejected(
    body: bytes, candidate: Mapping[str, object]
) -> bool:
    """Conservatively bind rejection wording to a candidate field/path."""
    identifiers = _probe_candidate_identifiers(candidate)
    if not identifiers:
        return False
    descriptors = "(?:invalid|unexpected|unknown|unrecognized|unsupported)"
    nouns = "(?:argument|field|option|parameter|property)"
    suffixes = tuple(re.escape(item) for item in _PROBE_REJECTION_SUFFIXES)
    suffix = "(?:" + "|".join(suffixes) + ")"
    for diagnostic in _probe_error_diagnostics(body):
        for identifier in identifiers:
            named = (
                r"(?<![\w])" + re.escape(identifier) + r"(?![\w])")
            # Bind the wording grammatically, not merely by proximity.  This
            # avoids treating an echoed candidate as rejected when the same
            # response says that some other field or the context was invalid.
            prefix_named = (
                rf"{descriptors}\s+{nouns}\s*[:=]?\s*['\"`]?"
                rf"{named}['\"`]?"
            )
            named_suffix = (
                rf"['\"`]?{named}['\"`]?\s*(?:{nouns}\s*)?{suffix}"
            )
            invalid_value_for = (
                rf"invalid\s+value(?:\s+[^,;:.]{{0,48}})?\s+for\s+"
                rf"['\"`]?{named}['\"`]?"
            )
            if re.search(
                rf"(?:{prefix_named}|{named_suffix}|{invalid_value_for})",
                diagnostic):
                return True
    return False


def _chat_payload(
    endpoint: Any,
    messages: list[dict],
    max_tokens: int,
    *,
    stream: bool,
    include_usage: bool,
) -> dict:
    extra_body = endpoint.extra_body or {}
    collisions = sorted(_CHAT_OWNED_FIELDS.intersection(extra_body))
    if collisions:
        raise ValueError(
            "extra_body must not set fields owned by endpoint adapter "
            f"{DEFAULT_ENDPOINT_ADAPTER}: " + ", ".join(collisions))
    payload: dict = dict(extra_body)
    payload["messages"] = messages
    payload["max_tokens"] = int(max_tokens)
    if endpoint.temperature is not None:
        payload["temperature"] = endpoint.temperature
    payload["stream"] = stream
    if endpoint.model:
        payload["model"] = endpoint.model
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


class OpenAIChatCompletionsSSEAdapter(EndpointAdapter):
    """Tested streamed Chat Completions subset used by existing releases."""

    adapter_id = DEFAULT_ENDPOINT_ADAPTER
    response_mode = "streaming"
    accept_media_type = "text/event-stream"
    accepted_response_media_types = ("text/event-stream",)
    usage_request_mode = "stream_options.include_usage"

    def validate_endpoint(self, endpoint: Any) -> None:
        # ``model`` is optional for a dedicated serving-endpoint invocation
        # route and required only by shared/model-service routes.  Route-level
        # preflight, not a model-name heuristic, resolves that distinction.
        extra_body = getattr(endpoint, "extra_body", None)
        if isinstance(extra_body, dict):
            collisions = sorted(_CHAT_OWNED_FIELDS.intersection(extra_body))
            if collisions:
                raise ValueError(
                    "extra_body must not set fields owned by endpoint adapter "
                    f"{self.adapter_id}: " + ", ".join(collisions))

    def serialize_request(
        self,
        endpoint: Any,
        messages: list[dict],
        max_tokens: int,
        include_usage: bool,
    ) -> bytes:
        return _json_bytes(
            _chat_payload(
                endpoint,
                messages,
                max_tokens,
                stream=True,
                include_usage=include_usage,
            )
        )

    def iter_events(self, chunks: Iterable[bytes | str]) -> Iterator[dict]:
        return iter_sse_events(chunks)

    def fold_event(
        self, state: StreamState, event: object
    ) -> EventMilestones:
        before_chunks = state.content_chunks
        before_reasoning = state.saw_first_reasoning
        before_visible = state.saw_first_visible
        before_tool = state.saw_first_tool_call
        first_content = update_state(state, event)
        return EventMilestones(
            first_content=first_content,
            first_reasoning=(
                state.saw_first_reasoning and not before_reasoning),
            first_visible=state.saw_first_visible and not before_visible,
            first_tool_call=state.saw_first_tool_call and not before_tool,
            content_event=state.content_chunks > before_chunks,
        )

    def finalize(self, state: StreamState) -> None:
        finalize_tool_calls(state)

    def normalize_usage(self, usage: dict | None) -> dict:
        return extract_usage(usage)

    def evaluate_outcome(self, state: StreamState) -> ResponseOutcome:
        """Preserve the released Chat Completions completion semantics."""
        response_complete = bool(state.done or state.finish_reason)
        output_observed = bool(
            state.saw_first_content or state.valid_tool_calls > 0)
        if state.errors:
            error = "stream protocol validation failed"
        elif not response_complete:
            error = "stream ended without [DONE] or a finish_reason"
        elif not output_observed:
            error = "stream ended with no content or valid tool call"
        else:
            error = None
        return ResponseOutcome(
            response_complete=response_complete,
            output_observed=output_observed,
            truncated=state.finish_reason == "length",
            ok=error is None,
            error=error,
        )

    def include_usage_option_rejected(self, status: int, body: bytes) -> bool:
        return status == 400 and _stream_options_rejected(body)

    def probe_control_rejected(
        self, status: int, body: bytes, candidate: Mapping[str, object]
    ) -> bool:
        # This adapter's established validation contract is HTTP 400.  A 422
        # may mean request validation for another provider, but that requires
        # a provider-specific, versioned adapter override.
        return status == 400 and _chat_probe_control_rejected(body, candidate)


_REGISTRY_LOCK = threading.RLock()
_ADAPTERS: dict[str, EndpointAdapter] = {}
_ADAPTER_FINGERPRINTS: dict[str, str] = {}
_ADAPTER_CONTRACT_DIGESTS: dict[str, str] = {}
_ADAPTER_GUARDS: dict[str, object] = {}
_ADAPTER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/v[1-9][0-9]*\Z")
_MEDIA_TYPE_RE = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z")
def _identity(value: object) -> str:
    """Stable code identity which never falls back to address-bearing repr."""
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        cls = type(value)
        module, qualname = cls.__module__, cls.__qualname__
    return f"{module}.{qualname}"


def _material_sort_key(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"))


class _ImplementationMaterializer:
    """Create deterministic material for every reachable behavior input.

    Adapter extensions are ordinary Python classes, so bytecode alone is not
    an implementation identity.  A method can read a mutable class mapping,
    a default argument, a closure cell, or a module global without changing
    one opcode.  This walker either records those values recursively or
    rejects an opaque value.  It never uses arbitrary ``repr`` output because
    object addresses would make evidence non-reproducible while still saying
    nothing useful about behavior.
    """

    _CLASS_METADATA = frozenset({
        "__module__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__abstractmethods__",
        "__dataclass_fields__",
        "__dataclass_params__",
        "_abc_impl",
    })

    def __init__(self, root: type) -> None:
        self.root_module = root.__module__

    def _is_local_module(self, module: object) -> bool:
        return isinstance(module, str) and (
            module == self.root_module
            or module == __name__
            or module == __package__
            or module.startswith(f"{__package__}.")
        )

    @staticmethod
    def _unsupported(path: str, value: object) -> ValueError:
        cls = type(value)
        return ValueError(
            "endpoint adapter implementation contains unsupported opaque "
            f"state at {path}: {cls.__module__}.{cls.__qualname__}; use "
            "finite primitive values, bytes, functions, classes, or nested "
            "built-in tuple/list/set/frozenset/dict containers")

    def code(self, code: types.CodeType, path: str,
             function_stack: frozenset[int],
             class_stack: frozenset[int]) -> dict:
        constants = []
        for index, value in enumerate(code.co_consts):
            constant_path = f"{path}.constant[{index}]"
            if isinstance(value, types.CodeType):
                constants.append({
                    "code": self.code(
                        value, constant_path, function_stack, class_stack)
                })
            else:
                constants.append(self.value(
                    value, constant_path, frozenset(), function_stack,
                    class_stack))
        return {
            "argcount": code.co_argcount,
            "posonlyargcount": code.co_posonlyargcount,
            "kwonlyargcount": code.co_kwonlyargcount,
            "flags": code.co_flags,
            "bytecode": code.co_code.hex(),
            "exception_table": getattr(code, "co_exceptiontable", b"").hex(),
            "constants": constants,
            "names": list(code.co_names),
            "variables": list(code.co_varnames),
            "freevars": list(code.co_freevars),
            "cellvars": list(code.co_cellvars),
        }

    def function(self, function: object, path: str,
                 function_stack: frozenset[int],
                 class_stack: frozenset[int], *,
                 include_globals: bool = True) -> dict:
        if isinstance(function, types.MethodType):
            bound = function.__self__
            bound_material = (
                {"class": _identity(bound)} if isinstance(bound, type)
                else self.value(
                    bound, f"{path}.bound", frozenset(), function_stack,
                    class_stack)
            )
            return {
                "bound_method": self.function(
                    function.__func__, f"{path}.function", function_stack,
                    class_stack, include_globals=include_globals),
                "bound_to": bound_material,
            }
        if isinstance(function, (
                types.BuiltinFunctionType, types.BuiltinMethodType)):
            return {"builtin": _identity(function)}
        if not isinstance(function, types.FunctionType):
            raise self._unsupported(path, function)

        identity = _identity(function)
        marker = id(function)
        if marker in function_stack:
            return {"function_reference": identity}
        descendants = function_stack | {marker}

        closure = []
        for index, cell in enumerate(function.__closure__ or ()):
            try:
                cell_value = cell.cell_contents
            except ValueError:
                closure.append({"empty_cell": True})
            else:
                closure.append(self.dependency(
                    cell_value, f"{path}.closure[{index}]", descendants,
                    class_stack, names=set(function.__code__.co_names)))

        dependencies: dict[str, object] = {}
        if include_globals:
            namespace = function.__globals__
            names = set(function.__code__.co_names)
            for name in sorted(names):
                if name in namespace:
                    dependencies[name] = self.dependency(
                        namespace[name], f"{path}.global[{name}]",
                        descendants, class_stack, names=names)
                elif name in vars(builtins):
                    dependencies[name] = self.dependency(
                        vars(builtins)[name], f"{path}.builtin[{name}]",
                        descendants, class_stack, names=names)

        return {
            "identity": identity,
            "code": self.code(
                function.__code__, f"{path}.code", descendants, class_stack),
            "defaults": self.value(
                function.__defaults__, f"{path}.defaults", frozenset(),
                descendants, class_stack),
            "keyword_defaults": self.value(
                function.__kwdefaults__, f"{path}.keyword_defaults",
                frozenset(), descendants, class_stack),
            "closure": closure,
            "function_state": self.value(
                dict(vars(function)), f"{path}.state", frozenset(),
                descendants, class_stack),
            "dependencies": dependencies,
        }

    def module(self, module: types.ModuleType, path: str,
               function_stack: frozenset[int],
               class_stack: frozenset[int], names: set[str]) -> dict:
        members: dict[str, object] = {}
        namespace = vars(module)
        for name in sorted(names):
            if name.startswith("__") or name not in namespace:
                continue
            value = namespace[name]
            if value is module:
                continue
            members[name] = self.dependency(
                value, f"{path}.{name}", function_stack, class_stack,
                names=names,
                # External library functions are identified by their code and
                # defaults, but their entire module-global graph is outside
                # the adapter-owned implementation boundary.
                include_function_globals=self._is_local_module(
                    getattr(value, "__module__", None)))
        return {"module": module.__name__, "referenced_members": members}

    def class_value(self, cls: type, path: str,
                    function_stack: frozenset[int],
                    class_stack: frozenset[int], *,
                    include_external_bases: bool = False) -> dict:
        identity = _identity(cls)
        if not self._is_local_module(cls.__module__):
            return {"external_class": identity}
        marker = id(cls)
        if marker in class_stack:
            return {"class_reference": identity}
        descendants = class_stack | {marker}
        owners = []
        for owner in reversed(cls.__mro__):
            if owner is object or (
                    not include_external_bases
                    and not self._is_local_module(owner.__module__)):
                continue
            members: dict[str, object] = {}
            for name, raw in sorted(vars(owner).items()):
                if name in self._CLASS_METADATA:
                    continue
                member_path = f"{path}.{owner.__qualname__}.{name}"
                if isinstance(raw, (staticmethod, classmethod)):
                    members[name] = self.function(
                        raw.__func__, member_path, function_stack, descendants,
                        include_globals=True)
                elif isinstance(raw, property):
                    members[name] = {
                        accessor: (
                            None if function is None else self.function(
                                function, f"{member_path}.{accessor}",
                                function_stack, descendants,
                                include_globals=True))
                        for accessor, function in (
                            ("get", raw.fget), ("set", raw.fset),
                            ("delete", raw.fdel))
                    }
                elif isinstance(raw, types.FunctionType):
                    members[name] = self.function(
                        raw, member_path, function_stack, descendants,
                        include_globals=True)
                else:
                    members[name] = self.value(
                        raw, member_path, frozenset(), function_stack,
                        descendants)
            owners.append({"owner": _identity(owner), "members": members})
        return {"class": identity, "owners": owners}

    def dependency(self, value: object, path: str,
                   function_stack: frozenset[int],
                   class_stack: frozenset[int], *, names: set[str],
                   include_function_globals: bool = True) -> object:
        if isinstance(value, types.ModuleType):
            return self.module(
                value, path, function_stack, class_stack, names)
        if isinstance(value, type):
            return self.class_value(
                value, path, function_stack, class_stack)
        if isinstance(value, (
                types.FunctionType, types.MethodType,
                types.BuiltinFunctionType, types.BuiltinMethodType)):
            return self.function(
                value, path, function_stack, class_stack,
                include_globals=(include_function_globals and
                                 self._is_local_module(
                                     getattr(value, "__module__", None))))
        return self.value(
            value, path, frozenset(), function_stack, class_stack)

    def value(self, value: object, path: str,
              container_stack: frozenset[int],
              function_stack: frozenset[int],
              class_stack: frozenset[int]) -> object:
        value_type = type(value)
        if value is None:
            return {"none": True}
        if value_type is bool:
            return {"bool": value}
        if value_type is int:
            return {"int": str(value)}
        if value_type is float:
            # float.hex is exact, distinguishes -0.0, and has stable spellings
            # for infinities and NaN without emitting invalid JSON numbers.
            return {"float": value.hex()}
        if value_type is complex:
            return {"complex": [value.real.hex(), value.imag.hex()]}
        if value_type is str:
            return {"string": value}
        if value_type is bytes:
            return {"bytes": value.hex()}
        if value is Ellipsis:
            return {"ellipsis": True}
        if value is NotImplemented:
            return {"not_implemented": True}
        if isinstance(value, re.Pattern):
            return {
                "regular_expression": self.value(
                    value.pattern, f"{path}.pattern", frozenset(),
                    function_stack, class_stack),
                "flags": int(value.flags),
            }
        if isinstance(value, types.CodeType):
            return {"code": self.code(
                value, path, function_stack, class_stack)}
        if isinstance(value, type):
            return self.class_value(
                value, path, function_stack, class_stack)
        if isinstance(value, (
                types.FunctionType, types.MethodType,
                types.BuiltinFunctionType, types.BuiltinMethodType)):
            return self.function(
                value, path, function_stack, class_stack,
                include_globals=self._is_local_module(
                    getattr(value, "__module__", None)))

        if value_type not in {tuple, list, set, frozenset, dict}:
            raise self._unsupported(path, value)
        marker = id(value)
        if marker in container_stack:
            raise ValueError(
                "endpoint adapter implementation contains cyclic state at "
                f"{path}; cyclic behavior inputs are not supported")
        descendants = container_stack | {marker}
        if value_type in {tuple, list}:
            return {
                "tuple" if value_type is tuple else "list": [
                    self.value(
                        item, f"{path}[{index}]", descendants,
                        function_stack, class_stack)
                    for index, item in enumerate(value)
                ]
            }
        if value_type in {set, frozenset}:
            items = [
                self.value(
                    item, f"{path}[item]", descendants, function_stack,
                    class_stack)
                for item in value
            ]
            items.sort(key=_material_sort_key)
            return {"set" if value_type is set else "frozenset": items}

        entries = []
        for key, item in value.items():
            key_material = self.value(
                key, f"{path}[key]", descendants, function_stack,
                class_stack)
            item_material = self.value(
                item, f"{path}[value]", descendants, function_stack,
                class_stack)
            entries.append({"key": key_material, "value": item_material})
        # Built-in dict iteration order can affect exact serialized request
        # bytes, so preserve it. Set/frozenset remain canonically sorted above
        # because their iteration order is not a stable semantic contract.
        return {"dict": entries}


class _ObjectIdentity:
    """Internal identity token; never serialized into artifact evidence."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ObjectIdentity) \
            and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)


def _guard_snapshot(value: object,
                    stack: frozenset[int] = frozenset()) -> object:
    """Cheap exact runtime snapshot for state already admitted by the walker."""
    value_type = type(value)
    if value is None or value is Ellipsis or value is NotImplemented:
        return (value_type, value)
    if value_type in {bool, int, str, bytes}:
        return (value_type, value)
    if value_type is float:
        return (float, value.hex())
    if value_type is complex:
        return (complex, value.real.hex(), value.imag.hex())
    if isinstance(value, re.Pattern):
        return (re.Pattern, _guard_snapshot(value.pattern), int(value.flags))
    if isinstance(value, (
            types.CodeType, types.ModuleType, type, types.FunctionType,
            types.MethodType, types.BuiltinFunctionType,
            types.BuiltinMethodType)):
        return ("identity", _ObjectIdentity(value))
    if value_type not in {tuple, list, set, frozenset, dict}:
        raise ValueError("unsupported adapter integrity-guard state")
    marker = id(value)
    if marker in stack:
        raise ValueError("cyclic adapter integrity-guard state")
    descendants = stack | {marker}
    if value_type in {tuple, list}:
        identity = _ObjectIdentity(value) if value_type is list else None
        return (
            value_type,
            identity,
            tuple(_guard_snapshot(item, descendants) for item in value),
        )
    if value_type in {set, frozenset}:
        identity = _ObjectIdentity(value) if value_type is set else None
        return (
            value_type,
            identity,
            frozenset(_guard_snapshot(item, descendants) for item in value),
        )
    return (
        dict,
        _ObjectIdentity(value),
        tuple(
            (_guard_snapshot(key, descendants),
             _guard_snapshot(item, descendants))
            for key, item in value.items()
        ),
    )


def _guard_cell_snapshot(cell: object) -> object:
    try:
        value = cell.cell_contents
    except ValueError:
        return ("empty_cell",)
    return _guard_snapshot(value)


class _ImplementationGuard:
    """Precompiled mutation checks kept off the load generator's hot path."""

    def __init__(self, adapter: EndpointAdapter) -> None:
        self.adapter = adapter
        self.root_class = type(adapter)
        self.root_module = type(adapter).__module__
        self.instance_state = _guard_snapshot(vars(adapter))
        self.classes: list[dict] = []
        self.functions: list[dict] = []
        self.modules: list[dict] = []
        self._class_seen: set[tuple[int, bool]] = set()
        self._function_seen: set[tuple[int, bool]] = set()
        self._module_seen: set[tuple[int, tuple[str, ...]]] = set()
        self._capture_class(type(adapter), include_external_bases=True)

    def _is_local(self, module: object) -> bool:
        return isinstance(module, str) and (
            module == self.root_module
            or module == __name__
            or module == __package__
            or module.startswith(f"{__package__}.")
        )

    @staticmethod
    def _member_snapshot(raw: object) -> object:
        if isinstance(raw, staticmethod):
            return (staticmethod, _ObjectIdentity(raw.__func__))
        if isinstance(raw, classmethod):
            return (classmethod, _ObjectIdentity(raw.__func__))
        if isinstance(raw, property):
            return (
                property,
                tuple(None if function is None else _ObjectIdentity(function)
                      for function in (raw.fget, raw.fset, raw.fdel)),
            )
        return _guard_snapshot(raw)

    def _capture_value(self, value: object, names: set[str]) -> None:
        if isinstance(value, types.ModuleType):
            self._capture_module(value, names)
            return
        if isinstance(value, type):
            if self._is_local(value.__module__):
                self._capture_class(value, include_external_bases=False)
            return
        if isinstance(value, types.MethodType):
            self._capture_function(value.__func__, include_globals=True)
            return
        if isinstance(value, types.FunctionType):
            self._capture_function(
                value, include_globals=self._is_local(value.__module__))
            return
        value_type = type(value)
        if value_type in {tuple, list, set, frozenset}:
            for item in value:
                self._capture_value(item, names)
        elif value_type is dict:
            for key, item in value.items():
                self._capture_value(key, names)
                self._capture_value(item, names)

    def _capture_class(self, cls: type, *,
                       include_external_bases: bool) -> None:
        marker = (id(cls), include_external_bases)
        if marker in self._class_seen:
            return
        self._class_seen.add(marker)
        record = {
            "class": cls,
            "module": cls.__module__,
            "name": cls.__name__,
            "qualname": cls.__qualname__,
            "mro": tuple(cls.__mro__),
            "owners": [],
        }
        self.classes.append(record)
        for owner in reversed(cls.__mro__):
            if owner is object or (
                    not include_external_bases
                    and not self._is_local(owner.__module__)):
                continue
            raw_members = {
                name: raw for name, raw in vars(owner).items()
                if name not in _ImplementationMaterializer._CLASS_METADATA
            }
            record["owners"].append({
                "owner": owner,
                "names": tuple(sorted(raw_members)),
                "members": {
                    name: self._member_snapshot(raw)
                    for name, raw in raw_members.items()
                },
            })
            for raw in raw_members.values():
                if isinstance(raw, (staticmethod, classmethod)):
                    self._capture_function(
                        raw.__func__, include_globals=True)
                elif isinstance(raw, property):
                    for function in (raw.fget, raw.fset, raw.fdel):
                        if function is not None:
                            self._capture_function(
                                function, include_globals=True)
                elif isinstance(raw, types.FunctionType):
                    self._capture_function(raw, include_globals=True)
                else:
                    self._capture_value(raw, set())

    def _capture_function(self, function: types.FunctionType, *,
                          include_globals: bool) -> None:
        marker = (id(function), include_globals)
        if marker in self._function_seen:
            return
        self._function_seen.add(marker)
        record = {
            "function": function,
            "module": function.__module__,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "code": function.__code__,
            "defaults": _guard_snapshot(function.__defaults__),
            "keyword_defaults": _guard_snapshot(function.__kwdefaults__),
            "closure": tuple(_guard_cell_snapshot(cell)
                             for cell in (function.__closure__ or ())),
            "state": _guard_snapshot(vars(function)),
            "globals": [],
        }
        self.functions.append(record)
        names = set(function.__code__.co_names)
        for cell in function.__closure__ or ():
            try:
                cell_value = cell.cell_contents
            except ValueError:
                continue
            self._capture_value(cell_value, names)
        if not include_globals:
            return
        namespace = function.__globals__
        builtin_namespace = vars(builtins)
        for name in sorted(names):
            if name in namespace:
                value = namespace[name]
                source = "global"
            elif name in builtin_namespace:
                value = builtin_namespace[name]
                source = "builtin"
            else:
                record["globals"].append((name, "missing", None))
                continue
            record["globals"].append(
                (name, source, _guard_snapshot(value)))
            self._capture_value(value, names)

    def _capture_module(self, module: types.ModuleType,
                        names: set[str]) -> None:
        selected = tuple(sorted(name for name in names
                                if not name.startswith("__")))
        marker = (id(module), selected)
        if marker in self._module_seen:
            return
        self._module_seen.add(marker)
        namespace = vars(module)
        members = []
        for name in selected:
            if name not in namespace:
                members.append((name, False, None))
                continue
            value = namespace[name]
            members.append((name, True, _guard_snapshot(value)))
            if value is not module:
                self._capture_value(value, names)
        self.modules.append({
            "module": module,
            "name": module.__name__,
            "members": members,
        })

    def assert_unchanged(self, adapter: EndpointAdapter) -> None:
        if adapter is not self.adapter or type(adapter) is not self.root_class \
                or _guard_snapshot(vars(adapter)) != self.instance_state:
            raise ValueError("adapter instance state changed")
        for record in self.classes:
            cls = record["class"]
            if cls.__module__ != record["module"] \
                    or cls.__name__ != record["name"] \
                    or cls.__qualname__ != record["qualname"] \
                    or tuple(cls.__mro__) != record["mro"]:
                raise ValueError("adapter class hierarchy changed")
            for owner_record in record["owners"]:
                owner = owner_record["owner"]
                current = {
                    name: raw for name, raw in vars(owner).items()
                    if name not in
                    _ImplementationMaterializer._CLASS_METADATA
                }
                if tuple(sorted(current)) != owner_record["names"]:
                    raise ValueError("adapter class members changed")
                if any(
                        self._member_snapshot(current[name]) != expected
                        for name, expected in
                        owner_record["members"].items()):
                    raise ValueError("adapter class member changed")
        for record in self.functions:
            function = record["function"]
            if function.__module__ != record["module"] \
                    or function.__name__ != record["name"] \
                    or function.__qualname__ != record["qualname"] \
                    or function.__code__ is not record["code"] \
                    or _guard_snapshot(function.__defaults__) != \
                    record["defaults"] \
                    or _guard_snapshot(function.__kwdefaults__) != \
                    record["keyword_defaults"] \
                    or tuple(_guard_cell_snapshot(cell)
                             for cell in (function.__closure__ or ())) != \
                    record["closure"] \
                    or _guard_snapshot(vars(function)) != record["state"]:
                raise ValueError("adapter function changed")
            namespace = function.__globals__
            builtin_namespace = vars(builtins)
            for name, source, expected in record["globals"]:
                if source == "global":
                    if name not in namespace \
                            or _guard_snapshot(namespace[name]) != expected:
                        raise ValueError("adapter global changed")
                elif source == "builtin":
                    if name in namespace or name not in builtin_namespace \
                            or _guard_snapshot(
                                builtin_namespace[name]) != expected:
                        raise ValueError("adapter builtin binding changed")
                elif name in namespace or name in builtin_namespace:
                    raise ValueError("adapter unresolved global changed")
        for record in self.modules:
            if record["module"].__name__ != record["name"]:
                raise ValueError("adapter module identity changed")
            namespace = vars(record["module"])
            for name, existed, expected in record["members"]:
                if (name in namespace) is not existed:
                    raise ValueError("adapter module dependency changed")
                if existed and _guard_snapshot(namespace[name]) != expected:
                    raise ValueError("adapter module member changed")


def _implementation_fingerprint(adapter: EndpointAdapter) -> str:
    """Hash all supported behavior-bearing state without paths or addresses."""
    cls = type(adapter)
    builder = _ImplementationMaterializer(cls)
    material = {
        "schema": "endpoint-adapter-implementation/v2",
        "class": builder.class_value(
            cls, "adapter_class", frozenset(), frozenset(),
            include_external_bases=True),
        "instance_state": builder.value(
            dict(vars(adapter)), "adapter_instance", frozenset(),
            frozenset(), frozenset()),
    }
    raw = json.dumps(
        material, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"))
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _contract_snapshot(adapter: EndpointAdapter) -> tuple[dict, str]:
    contract = adapter.contract()
    if not isinstance(contract, dict) or not all(
            isinstance(key, str) for key in contract):
        raise ValueError(
            "endpoint adapter contract must be a JSON object with string keys")
    raw = json.dumps(
        contract, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode("ascii")
    # Never add provenance fields to an adapter-owned mutable dictionary.
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _catalog_contract(adapter: EndpointAdapter, registered_id: str) -> dict:
    with _REGISTRY_LOCK:
        expected = _ADAPTER_FINGERPRINTS.get(registered_id)
        expected_contract = _ADAPTER_CONTRACT_DIGESTS.get(registered_id)
        guard = _ADAPTER_GUARDS.get(registered_id)
    if expected is None or expected_contract is None \
            or not isinstance(guard, _ImplementationGuard):
        raise RuntimeError(
            f"registered endpoint adapter {registered_id!r} has incomplete "
            "registry evidence")
    try:
        guard.assert_unchanged(adapter)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RuntimeError(
            f"registered endpoint adapter {registered_id!r} changed "
            "implementation after registration") from exc
    try:
        contract, contract_digest = _contract_snapshot(adapter)
        guard.assert_unchanged(adapter)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RuntimeError(
            f"registered endpoint adapter {registered_id!r} changed its "
            "contract or implementation after registration") from exc
    # A contract() implementation is executable code.  Recheck after calling
    # it so a side effect cannot mutate behavior between validation and use.
    if contract_digest != expected_contract:
        raise RuntimeError(
            f"registered endpoint adapter {registered_id!r} changed "
            "its contract after registration")
    contract["implementation"] = (
        f"{type(adapter).__module__}.{type(adapter).__qualname__}")
    contract["implementation_sha256"] = expected
    return contract


class EndpointAdapterExecution:
    """One start-attested adapter reused for an execution's hot path.

    Registry attestation walks every behavior-bearing class/default/closure/
    helper dependency and is intentionally expensive. Doing that for every
    body serialization distorts high-rate client pacing. This lease performs
    the full check once, retains a defensive contract snapshot, and offers a
    direct serializer. Call :meth:`assert_unchanged` at the execution's final
    evidence boundary; a mutation then aborts sealing rather than publishing
    results produced by an unattested implementation.
    """

    __slots__ = ("_adapter_id", "_adapter", "_contract_json")

    def __init__(self, adapter_id: str, adapter: EndpointAdapter,
                 contract: dict) -> None:
        self._adapter_id = adapter_id
        self._adapter = adapter
        self._contract_json = json.dumps(
            contract, sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"))

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter(self) -> EndpointAdapter:
        return self._adapter

    @property
    def contract(self) -> dict:
        return json.loads(self._contract_json)

    def serialize_request(
        self, endpoint: Any, messages: list[dict], max_tokens: int,
        include_usage: bool,
    ) -> bytes:
        if getattr(endpoint, "adapter", None) != self._adapter_id:
            raise ValueError("endpoint adapter changed after execution start")
        # EndpointConfig is mutable for embedding callers. Revalidate request
        # controls on the cheap path; only registry implementation attestation
        # is deferred to the execution boundary.
        self._adapter.validate_endpoint(endpoint)
        return self._adapter.serialize_request(
            endpoint, messages, max_tokens, include_usage)

    def assert_unchanged(self) -> None:
        with _REGISTRY_LOCK:
            current = _ADAPTERS.get(self._adapter_id)
        if current is not self._adapter:
            raise RuntimeError(
                f"registered endpoint adapter {self._adapter_id!r} changed "
                "identity during execution")
        contract = _catalog_contract(self._adapter, self._adapter_id)
        current_json = json.dumps(
            contract, sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"))
        if current_json != self._contract_json:
            raise RuntimeError(
                f"registered endpoint adapter {self._adapter_id!r} changed "
                "contract during execution")


def _validate_adapter_contract(adapter: EndpointAdapter) -> None:
    if not isinstance(adapter, EndpointAdapter):
        raise TypeError("endpoint adapter must extend EndpointAdapter")
    adapter_id = getattr(adapter, "adapter_id", None)
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise ValueError("endpoint adapter_id must be a non-empty string")
    if len(adapter_id) > 128 or not _ADAPTER_ID_RE.fullmatch(adapter_id):
        raise ValueError(
            "endpoint adapter_id must be at most 128 characters and match "
            "name/vN with a positive numeric version")
    if getattr(adapter, "response_mode", None) not in {
        "streaming",
        "non_streaming",
    }:
        raise ValueError(
            "endpoint adapter response_mode must be streaming or "
            "non_streaming")
    media_types = getattr(adapter, "accepted_response_media_types", None)
    if not isinstance(media_types, tuple) or not media_types or not all(
        isinstance(value, str) and _MEDIA_TYPE_RE.fullmatch(value)
        for value in media_types
    ):
        raise ValueError(
            "endpoint adapter accepted_response_media_types must be a "
            "non-empty tuple of lowercase, parameter-free media types")
    if len(set(media_types)) != len(media_types):
        raise ValueError(
            "endpoint adapter accepted_response_media_types must not "
            "contain duplicates")
    for name in ("request_media_type", "accept_media_type"):
        value = getattr(adapter, name, None)
        if not isinstance(value, str) or not _MEDIA_TYPE_RE.fullmatch(value):
            raise ValueError(
                f"endpoint adapter {name} must be a lowercase, "
                "parameter-free media type")
    if getattr(adapter, "usage_request_mode", None) not in {
        "stream_options.include_usage",
        "intrinsic",
        "unavailable",
    }:
        raise ValueError("endpoint adapter has an invalid usage_request_mode")


def register_endpoint_adapter(adapter: EndpointAdapter) -> None:
    """Register one immutable, versioned adapter for this Python process.

    Replacing an identifier is forbidden.  This prevents benchmark behavior
    from depending on import order and preserves the meaning of adapter IDs in
    sealed artifacts.
    """
    _validate_adapter_contract(adapter)
    if vars(adapter):
        raise ValueError(
            "endpoint adapter instances must be stateless; place immutable "
            "contract constants on the adapter class")
    registered_id = adapter.adapter_id
    with _REGISTRY_LOCK:
        if registered_id in _ADAPTERS:
            raise ValueError(
                f"endpoint adapter {registered_id!r} is already "
                "registered")
        fingerprint = _implementation_fingerprint(adapter)
        _contract, contract_digest = _contract_snapshot(adapter)
        fingerprint_after_contract = _implementation_fingerprint(adapter)
        if adapter.adapter_id != registered_id \
                or fingerprint_after_contract != fingerprint:
            raise ValueError(
                "endpoint adapter implementation changed while it was being "
                "registered")
        guard = _ImplementationGuard(adapter)
        guard.assert_unchanged(adapter)
        # Publish all three records only after every validation succeeds.  A
        # rejected opaque default/closure/class value cannot leave a partial
        # registry entry that later appears available but has no evidence.
        _ADAPTERS[registered_id] = adapter
        _ADAPTER_FINGERPRINTS[registered_id] = fingerprint
        _ADAPTER_CONTRACT_DIGESTS[registered_id] = contract_digest
        _ADAPTER_GUARDS[registered_id] = guard


def _lookup_registered_adapter(adapter_id: str) -> EndpointAdapter:
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("endpoint adapter must be a non-empty string")
    with _REGISTRY_LOCK:
        adapter = _ADAPTERS.get(adapter_id)
        available = tuple(sorted(_ADAPTERS))
    if adapter is None:
        raise ValueError(
            f"unknown endpoint adapter {adapter_id!r}; available: "
            + ", ".join(available))
    return adapter


def resolve_endpoint_adapter(adapter_id: str) -> EndpointAdapterExecution:
    """Fully attest an adapter once and bind it to one execution.

    The returned execution object is the only supported way for a long-lived
    client to bypass per-request registry attestation.  Callers must perform
    its final integrity check before publishing execution evidence.
    """
    adapter = _lookup_registered_adapter(adapter_id)
    contract = _catalog_contract(adapter, adapter_id)
    return EndpointAdapterExecution(adapter_id, adapter, contract)


def get_endpoint_adapter(adapter_id: str) -> EndpointAdapter:
    return resolve_endpoint_adapter(adapter_id).adapter


def list_endpoint_adapters() -> list[dict]:
    with _REGISTRY_LOCK:
        adapters = [(key, _ADAPTERS[key]) for key in sorted(_ADAPTERS)]
    return [
        _catalog_contract(adapter, registered_id)
        for registered_id, adapter in adapters
    ]


def endpoint_adapter_contract(adapter_id: str) -> dict:
    """Return the immutable, provenance-bearing contract for one adapter."""
    return resolve_endpoint_adapter(adapter_id).contract


def _register_builtins() -> None:
    builtins = (OpenAIChatCompletionsSSEAdapter(),)
    for adapter in builtins:
        register_endpoint_adapter(adapter)


_register_builtins()


__all__ = [
    "DEFAULT_ENDPOINT_ADAPTER",
    "EndpointAdapter",
    "EndpointAdapterExecution",
    "EventMilestones",
    "OpenAIChatCompletionsSSEAdapter",
    "ResponseOutcome",
    "endpoint_adapter_contract",
    "get_endpoint_adapter",
    "list_endpoint_adapters",
    "register_endpoint_adapter",
    "resolve_endpoint_adapter",
]
