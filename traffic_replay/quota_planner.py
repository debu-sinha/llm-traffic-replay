"""Conservative, pre-traffic quota planning for pay-per-token runs.

The provider limits supported here are admission controls, not endpoint
capacity.  This module therefore answers one deliberately narrow question:
could traffic emitted by this harness alone cross the configured warning
budget?  It never claims that provider headroom exists, because unrelated
workspace traffic is not observable from a local load generator.

Input-token planning uses a tokenizer-independent upper bound: one token per
UTF-8 byte of the complete submitted JSON body plus a fixed chat-framing
allowance. This includes message roles and metadata, model, tool schemas,
provider controls, and wire-level JSON framing rather than counting only
message content. Synthetic replay is planned at the larger of its configured
chars-per-token value and the hard post-calibration ceiling, so calibration
cannot enlarge an already-authorized request beyond the pre-traffic plan.
Output planning uses the offered ``max_tokens`` reservation, which is the
conservative admission-time quantity for the Databricks FMAPI pay-per-token
accounting model.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import threading
import time
import uuid
from collections import deque
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:  # pragma: no cover - imports are runtime-local by design
    from .runner import RunConfig


class QuotaPlanError(ValueError):
    """A configured workload is not safe to start under its quota snapshot."""

    def __init__(self, plan: dict):
        self.plan = plan
        reasons = plan.get("refusal_reasons") or ["quota plan is incomplete"]
        if plan.get("refusal_stage") == "endpoint_binding":
            prefix = "endpoint binding refused before paid inference traffic: "
        else:
            prefix = "quota plan refused before endpoint traffic: "
        super().__init__(prefix + "; ".join(
            str(reason) for reason in reasons))


# ``calibrate_cpt`` in textgen.py clamps every measured value to 12.0.  Keep
# this explicit in the quota artifact: changing that runtime ceiling requires
# changing this planner and its regression test in the same review.
_CALIBRATED_CPT_HARD_MAX = 12.0

# Chat-template framing is provider-owned and is not present in the submitted
# JSON. Reserve this amount for every message plus one request-level block; a
# single fixed request allowance would become unsafe for long conversations.
# The complete JSON body itself is charged at the stricter tokenizer-
# independent limit of one token per UTF-8 byte.
_CHAT_FRAMING_TOKEN_ALLOWANCE = 64

# TextMaterializer's ASCII prose can require JSON escaping only for paragraph
# newlines. Six sentences of at least eight one-character words occupy more
# than 100 characters before the next two-newline paragraph marker. Reserve
# two escaped-newline expansion bytes per 100 content characters plus four for
# the suffix separator and rounding across prefix/suffix prose components.
# Regression tests compare this analytical bound with fully serialized bodies
# across seeds, prefix shapes, and every supported CPT extreme.
_SYNTHETIC_JSON_ESCAPE_BLOCK_CHARS = 100


_RUNTIME_QUOTA_DIMENSIONS = {
    "input_tokens_per_minute": ("input_tokens", 60.0),
    "output_tokens_per_minute": ("output_tokens", 60.0),
    "queries_per_hour": ("queries", 3600.0),
    # Accepted by the guard ahead of the configuration surface so the runtime
    # primitive does not need another algorithm when a verified QPS fact is
    # added.  RunConfig validation remains the authority on whether a command
    # may configure this field.
    "queries_per_second": ("queries", 1.0),
}

_RUNTIME_PER_REQUEST_LIMITS = {
    "request_bytes_max": "request_bytes",
}


def runtime_quota_scope_material(rate_limits: dict, endpoint: dict) -> dict:
    """Build the secret-free endpoint/accounting identity for one guard."""
    if not isinstance(rate_limits, dict) or not isinstance(endpoint, dict):
        raise ValueError("runtime quota scope needs rate limits and endpoint")
    base_url = endpoint.get("base_url")
    path = endpoint.get("path")
    if not isinstance(base_url, str) or not base_url \
            or not isinstance(path, str) or not path:
        raise ValueError("runtime quota scope needs endpoint base_url and path")
    identity_fields = (
        "provider", "deployment_mode", "workspace_tier", "model",
        "accounting_model", "scope", "source", "as_of", "verified_at",
        "max_age_days")
    return {
        "endpoint": {
            "base_url": base_url,
            "path": path,
            "request_model": endpoint.get("model"),
            "service_tier": (
                (endpoint.get("extra_body") or {}).get(
                    "service_tier", "default")
                if isinstance(endpoint.get("extra_body") or {}, dict)
                else None),
        },
        "rate_limit_identity": {
            name: rate_limits.get(name) for name in identity_fields},
    }

_RUNTIME_SCOPE_UNSET = object()


def _runtime_decimal(value: object, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{where} must be a finite number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite number") from exc
    if not number.is_finite():
        raise ValueError(f"{where} must be a finite number")
    return number


def _runtime_contract(rate_limits: dict) -> dict:
    """Return the exact enforcement subset used by a runtime guard."""
    if not isinstance(rate_limits, dict):
        raise ValueError("runtime quota rate_limits must be an object")
    warning = _runtime_decimal(
        rate_limits.get("warning_utilization"),
        "rate_limits.warning_utilization")
    if warning <= 0 or warning > 1:
        raise ValueError(
            "rate_limits.warning_utilization must be in (0, 1]")
    limits = {}
    for name in _RUNTIME_QUOTA_DIMENSIONS:
        if name not in rate_limits:
            continue
        limit = _runtime_decimal(rate_limits[name], f"rate_limits.{name}")
        if limit <= 0:
            raise ValueError(f"rate_limits.{name} must be greater than zero")
        limits[name] = format(limit, "f")
    per_request_limits = {}
    for name in _RUNTIME_PER_REQUEST_LIMITS:
        if name not in rate_limits:
            continue
        limit = _runtime_decimal(rate_limits[name], f"rate_limits.{name}")
        if limit <= 0:
            raise ValueError(f"rate_limits.{name} must be greater than zero")
        per_request_limits[name] = format(limit, "f")
    if not limits and not per_request_limits:
        raise ValueError(
            "runtime quota guard needs an input, output, query/hour, or "
            "query/second rolling limit, or a request byte limit")
    return {
        "warning_utilization": format(warning, "f"),
        "limits": limits,
        "per_request_limits": per_request_limits,
    }


def _runtime_scope_id(
        contract: dict, shard_total: int, scope_material: object | None) -> str:
    try:
        scope_bytes = json.dumps({
            "contract": contract,
            "shard_total": shard_total,
            "scope_material": scope_material,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
           allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "runtime quota scope_material must be finite JSON") from exc
    return "quota-scope-" + hashlib.sha256(scope_bytes).hexdigest()[:32]


def _runtime_integer_budget(limit: Decimal, warning: Decimal) -> int:
    """Largest integer strictly below ``limit * warning``.

    The static planner refuses contact with the warning threshold.  Expressing
    that policy as an integer budget avoids binary-float boundary drift.
    """
    threshold = limit * warning
    return max(
        int(threshold.to_integral_value(rounding=ROUND_CEILING)) - 1, 0)


class RuntimeQuotaGuardError(RuntimeError):
    """An internal admission uncertainty permanently tripped the guard."""


@dataclasses.dataclass
class AdmissionHandle:
    """Opaque runtime-admission handle plus its JSON-safe evidence event.

    Callers persist only ``event``.  The private token prevents a fabricated or
    foreign event dictionary from committing a reservation in this guard.
    """

    event: dict
    _guard_id: str = dataclasses.field(repr=False)
    _token: str | None = dataclasses.field(repr=False)
    _state: str = dataclasses.field(repr=False)


@dataclasses.dataclass
class _PendingRuntimeReservation:
    handle: AdmissionHandle
    amounts: dict[str, int]


class RuntimeQuotaGuard:
    """Fail-closed, non-waiting rolling admission for physical inference POSTs.

    A reservation is provisional until the caller observes response headers or
    an ambiguous transport failure.  Provisional reservations never age out.
    Committing at that later observation time is conservative because it cannot
    precede provider receipt; it prevents a scheduler, upload, or network pause
    from expiring a reservation before the physical POST's own rolling window.

    This guard covers one harness command only.  It deliberately makes no claim
    about unrelated workspace traffic or provider-side burst state.
    """

    schema_version = 1

    def __init__(
            self, rate_limits: dict, *, shard_index: int = 0,
            shard_total: int = 1, scope_material: object | None = None,
            guard_id: str | None = None,
            clock_ns: Callable[[], int] = time.monotonic_ns,
            wall_clock: Callable[[], float] = time.time):
        contract = _runtime_contract(rate_limits)
        if not isinstance(shard_total, int) or isinstance(shard_total, bool) \
                or shard_total <= 0:
            raise ValueError("shard_total must be a positive integer")
        if not isinstance(shard_index, int) or isinstance(shard_index, bool) \
                or not 0 <= shard_index < shard_total:
            raise ValueError("need 0 <= shard_index < shard_total")
        if not callable(clock_ns) or not callable(wall_clock):
            raise ValueError("runtime quota clocks must be callable")
        if guard_id is None:
            guard_id = "quota-guard-" + uuid.uuid4().hex
        if not isinstance(guard_id, str) or not guard_id.strip():
            raise ValueError("guard_id must be a non-empty string")
        if any(ord(char) < 0x21 or ord(char) > 0x7e for char in guard_id):
            raise ValueError(
                "guard_id must be printable ASCII without whitespace")

        scope_id = _runtime_scope_id(contract, shard_total, scope_material)

        created_ns = clock_ns()
        if isinstance(created_ns, bool) or not isinstance(created_ns, int) \
                or created_ns < 0:
            raise ValueError(
                "runtime quota monotonic clock must return a non-negative int")
        created_unix = wall_clock()
        if isinstance(created_unix, bool) \
                or not isinstance(created_unix, (int, float)) \
                or not math.isfinite(float(created_unix)):
            raise ValueError(
                "runtime quota wall clock must return a finite number")

        self._rate_contract = contract
        self._shard_index = shard_index
        self._shard_total = shard_total
        self._guard_id = guard_id
        self._scope_id = scope_id
        self._clock_ns = clock_ns
        self._wall_clock = wall_clock
        self._created_ns = created_ns
        self._created_unix = float(created_unix)
        self._last_clock_ns = created_ns
        self._last_wall = float(created_unix)
        self._lock = threading.Lock()

        warning = Decimal(contract["warning_utilization"])
        self._budgets: dict[str, dict] = {}
        self._per_request_limits: dict[str, dict] = {}
        self._committed: dict[str, deque[tuple[int, int, str]]] = {}
        self._committed_sums: dict[str, int] = {}
        self._provisional_sums: dict[str, int] = {}
        for name, text_limit in contract["limits"].items():
            limit = Decimal(text_limit)
            global_max = _runtime_integer_budget(limit, warning)
            base, remainder = divmod(global_max, shard_total)
            local_max = base + (1 if shard_index < remainder else 0)
            reservation_field, window_s = _RUNTIME_QUOTA_DIMENSIONS[name]
            self._budgets[name] = {
                "reservation_field": reservation_field,
                "window_seconds": window_s,
                "window_ns": int(window_s * 1_000_000_000),
                "configured_limit": text_limit,
                "warning_utilization": contract["warning_utilization"],
                "exclusive_warning_threshold": format(
                    limit * warning, "f"),
                "global_max_integer": global_max,
                "local_max_integer": local_max,
            }
            self._committed[name] = deque()
            self._committed_sums[name] = 0
            self._provisional_sums[name] = 0

        for name, text_limit in contract["per_request_limits"].items():
            limit = Decimal(text_limit)
            self._per_request_limits[name] = {
                "reservation_field": _RUNTIME_PER_REQUEST_LIMITS[name],
                "configured_limit": text_limit,
                # Byte usage is integral.  Flooring a fractional configured
                # maximum preserves the exact ``len(body) <= limit`` policy.
                "maximum_integer": int(limit.to_integral_value(
                    rounding=ROUND_FLOOR)),
                "measurement": "exact_serialized_request_body_bytes",
                "comparison": "less_than_or_equal",
                "allocation": "shard_independent_per_post",
            }

        self._pending: dict[str, _PendingRuntimeReservation] = {}
        self._external_prior_keys: set[tuple[str, int]] = set()
        self._sequence = 0
        self._tripped = False
        self._trip: dict | None = None
        self._counts = {
            "admission_decisions": 0,
            "admitted": 0,
            "denied": 0,
            "committed": 0,
            "cancelled_before_post": 0,
            "seeded_committed": 0,
            "seeded_nonconsuming": 0,
            "seeded_deduplicated": 0,
        }

    @property
    def guard_id(self) -> str:
        return self._guard_id

    @property
    def scope_id(self) -> str:
        return self._scope_id

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def matches(
            self, rate_limits: dict, shard_index: int, shard_total: int,
            scope_material: object = _RUNTIME_SCOPE_UNSET) -> bool:
        if not isinstance(shard_total, int) or isinstance(shard_total, bool) \
                or shard_total <= 0 \
                or not isinstance(shard_index, int) \
                or isinstance(shard_index, bool) \
                or not 0 <= shard_index < shard_total:
            return False
        try:
            contract = _runtime_contract(rate_limits)
            scope_matches = (
                True if scope_material is _RUNTIME_SCOPE_UNSET
                else _runtime_scope_id(
                    contract, shard_total, scope_material) == self._scope_id)
        except ValueError:
            return False
        return bool(
            contract == self._rate_contract
            and shard_index == self._shard_index
            and shard_total == self._shard_total
            and scope_matches)

    def _now_locked(self) -> tuple[int, float]:
        now_ns = self._clock_ns()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) \
                or now_ns < self._last_clock_ns:
            raise RuntimeError(
                "runtime quota monotonic clock moved backwards or became "
                "invalid; refusing admission")
        wall = self._wall_clock()
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
                or not math.isfinite(float(wall)):
            raise RuntimeError(
                "runtime quota wall clock became invalid; refusing admission")
        self._last_clock_ns = now_ns
        self._last_wall = float(wall)
        return now_ns, self._last_wall

    def _latch_internal_error_locked(
            self, exc: BaseException, *, operation: str) -> None:
        """Permanently fail closed without depending on another clock read."""
        if not self._tripped:
            self._trip_locked(
                now_ns=self._last_clock_ns, wall=self._last_wall,
                sequence=self._sequence or None,
                reason_code="guard_internal_error",
                denied_dimensions=[],
                prior_event={
                    "operation": operation,
                    "error_type": type(exc).__name__,
                })

    def _evict_locked(self, now_ns: int) -> None:
        for name, budget in self._budgets.items():
            active = self._committed[name]
            # Provider boundary inclusion is unpublished. Retain an event at
            # exactly the boundary and remove it only once strictly older.
            while active and now_ns - active[0][0] > budget["window_ns"]:
                _stamp, amount, _reservation_id = active.popleft()
                self._committed_sums[name] -= amount
                if self._committed_sums[name] < 0:
                    raise RuntimeError(
                        "runtime quota committed total became negative")

    def _transition_clock_locked(self, operation: str) -> tuple[int, float]:
        try:
            return self._now_locked()
        except Exception as exc:
            self._latch_internal_error_locked(exc, operation=operation)
            raise RuntimeQuotaGuardError(
                f"runtime quota guard tripped during {operation}") from exc

    def _active_before_locked(self) -> dict[str, int]:
        return {
            name: self._committed_sums[name] + self._provisional_sums[name]
            for name in self._budgets
        }

    def _projection_locked(self, amounts: dict[str, int]) -> dict:
        before = self._active_before_locked()
        return {
            name: {
                "window_seconds": budget["window_seconds"],
                "active_before": before[name],
                "reservation": amounts[budget["reservation_field"]],
                "projected_after": (
                    before[name]
                    + amounts[budget["reservation_field"]]),
                "local_max_integer": budget["local_max_integer"],
                "global_max_integer": budget["global_max_integer"],
            }
            for name, budget in self._budgets.items()
        }

    def _hard_limit_checks_locked(self, amounts: dict[str, int]) -> dict:
        return {
            name: {
                "reservation": amounts[limit["reservation_field"]],
                "configured_limit": limit["configured_limit"],
                "maximum_integer": limit["maximum_integer"],
                "measurement": limit["measurement"],
                "comparison": limit["comparison"],
                "allocation": limit["allocation"],
                "exceeded": (
                    amounts[limit["reservation_field"]]
                    > limit["maximum_integer"]),
            }
            for name, limit in self._per_request_limits.items()
        }

    def _trip_locked(self, *, now_ns: int, wall: float,
                     sequence: int | None, reason_code: str,
                     denied_dimensions: list[str],
                     prior_event: dict | None = None) -> None:
        if self._tripped:
            return
        self._tripped = True
        self._trip = {
            "reason_code": reason_code,
            "sequence": sequence,
            "at_elapsed_ns": now_ns - self._created_ns,
            "at_unix": wall,
            "denied_dimensions": list(denied_dimensions),
            "prior_event": copy.deepcopy(prior_event),
        }

    @staticmethod
    def _validate_reserve_args(
            body: bytes, message_count: int, max_tokens: int,
            request_id: str, attempt_ordinal: int,
            retry_trigger: str | None) -> None:
        if not isinstance(body, bytes):
            raise ValueError("runtime quota body must be bytes")
        if not isinstance(message_count, int) \
                or isinstance(message_count, bool) or message_count <= 0:
            raise ValueError("message_count must be a positive integer")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) \
                or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(attempt_ordinal, int) \
                or isinstance(attempt_ordinal, bool) or attempt_ordinal <= 0:
            raise ValueError("attempt_ordinal must be a positive integer")
        if retry_trigger is not None and (
                not isinstance(retry_trigger, str) or not retry_trigger):
            raise ValueError("retry_trigger must be null or a non-empty string")

    def _admission_event_locked(
            self, *, sequence: int, denied: bool, reason_code: str | None,
            request_id: str, attempt_ordinal: int,
            retry_trigger: str | None, now_ns: int, wall: float,
            amounts: dict[str, int], projection: dict,
            hard_limit_checks: dict,
            denied_dimensions: list[str]) -> dict:
        return {
            "schema_version": self.schema_version,
            "guard_id": self._guard_id,
            "scope_id": self._scope_id,
            "shard_index": self._shard_index,
            "shard_total": self._shard_total,
            "sequence": sequence,
            "decision": "denied" if denied else "admitted",
            "state": "denied" if denied else "provisional",
            "reason_code": reason_code,
            "request_id": request_id,
            "attempt_ordinal": attempt_ordinal,
            "retry_trigger": retry_trigger,
            "reserved_at_elapsed_ns": now_ns - self._created_ns,
            "reserved_at_unix": wall,
            "post_may_have_started": False,
            "post_started_at_elapsed_ns": None,
            "post_started_at_unix": None,
            "committed_at_elapsed_ns": None,
            "committed_at_unix": None,
            "cancelled_at_elapsed_ns": None,
            "cancelled_at_unix": None,
            "transition_reason": None,
            "reservation": copy.deepcopy(amounts),
            "projected": copy.deepcopy(projection),
            "hard_limit_checks": copy.deepcopy(hard_limit_checks),
            "denied_dimensions": list(denied_dimensions),
        }

    def reserve(
            self, body: bytes, message_count: int, max_tokens: int,
            request_id: str, attempt_ordinal: int,
            retry_trigger: str | None = None) -> AdmissionHandle:
        """Atomically admit or permanently trip without waiting or sleeping."""
        self._validate_reserve_args(
            body, message_count, max_tokens, request_id, attempt_ordinal,
            retry_trigger)
        amounts = {
            "request_bytes": len(body),
            "input_tokens": (
                len(body)
                + _CHAT_FRAMING_TOKEN_ALLOWANCE * (message_count + 1)),
            "output_tokens": max_tokens,
            "queries": 1,
        }
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            if self._tripped:
                projection = self._projection_locked(amounts)
                hard_limit_checks = self._hard_limit_checks_locked(amounts)
                event = self._admission_event_locked(
                    sequence=sequence, denied=True,
                    reason_code="guard_already_tripped",
                    request_id=request_id, attempt_ordinal=attempt_ordinal,
                    retry_trigger=retry_trigger,
                    now_ns=self._last_clock_ns, wall=self._last_wall,
                    amounts=amounts, projection=projection,
                    hard_limit_checks=hard_limit_checks,
                    denied_dimensions=list(
                        (self._trip or {}).get("denied_dimensions", [])))
                self._counts["admission_decisions"] += 1
                self._counts["denied"] += 1
                return AdmissionHandle(
                    event=event, _guard_id=self._guard_id,
                    _token=None, _state="denied")
            try:
                now_ns, wall = self._now_locked()
                self._evict_locked(now_ns)
                projection = self._projection_locked(amounts)
                hard_limit_checks = self._hard_limit_checks_locked(amounts)
            except Exception as exc:
                self._latch_internal_error_locked(exc, operation="reserve")
                event = self._admission_event_locked(
                    sequence=sequence, denied=True,
                    reason_code="guard_internal_error",
                    request_id=request_id, attempt_ordinal=attempt_ordinal,
                    retry_trigger=retry_trigger,
                    now_ns=self._last_clock_ns, wall=self._last_wall,
                    amounts=amounts, projection={}, hard_limit_checks={},
                    denied_dimensions=[])
                event["transition_reason"] = type(exc).__name__
                self._counts["admission_decisions"] += 1
                self._counts["denied"] += 1
                return AdmissionHandle(
                    event=event, _guard_id=self._guard_id,
                    _token=None, _state="denied")
            exceeded = [
                name for name, item in projection.items()
                if item["projected_after"] > item["local_max_integer"]
            ]
            exceeded.extend(
                name for name, item in hard_limit_checks.items()
                if item["exceeded"])
            denied = bool(exceeded)
            if any(name in self._per_request_limits for name in exceeded):
                reason_code = "hard_per_request_limit_exceeded"
            elif denied:
                reason_code = "warning_budget_would_be_reached"
            else:
                reason_code = None
            event = self._admission_event_locked(
                sequence=sequence, denied=denied,
                reason_code=reason_code,
                request_id=request_id, attempt_ordinal=attempt_ordinal,
                retry_trigger=retry_trigger, now_ns=now_ns, wall=wall,
                amounts=amounts, projection=projection,
                hard_limit_checks=hard_limit_checks,
                denied_dimensions=exceeded)
            self._counts["admission_decisions"] += 1
            if denied:
                self._counts["denied"] += 1
                self._trip_locked(
                    now_ns=now_ns, wall=wall, sequence=sequence,
                    reason_code=reason_code or "quota_admission_denied",
                    denied_dimensions=exceeded)
                return AdmissionHandle(
                    event=event, _guard_id=self._guard_id,
                    _token=None, _state="denied")

            token = uuid.uuid4().hex
            handle = AdmissionHandle(
                event=event, _guard_id=self._guard_id,
                _token=token, _state="provisional")
            self._pending[token] = _PendingRuntimeReservation(
                handle=handle, amounts=amounts)
            for name, budget in self._budgets.items():
                self._provisional_sums[name] += amounts[
                    budget["reservation_field"]]
            self._counts["admitted"] += 1
            return handle

    def _pending_for_locked(
            self, handle: AdmissionHandle) -> _PendingRuntimeReservation:
        if not isinstance(handle, AdmissionHandle) \
                or handle._guard_id != self._guard_id:
            raise ValueError("admission handle belongs to another guard")
        if handle._state == "denied":
            raise ValueError("a denied admission has no reservation")
        if handle._token is None:
            raise ValueError("admission handle has no reservation token")
        pending = self._pending.get(handle._token)
        if pending is None or pending.handle is not handle:
            raise ValueError("admission reservation is not active")
        return pending

    def mark_post_may_have_started(self, handle: AdmissionHandle) -> dict:
        """Mark the last safe point immediately before ``conn.request``."""
        with self._lock:
            if handle._state == "committed":
                return handle.event
            pending = self._pending_for_locked(handle)
            if pending.handle.event["post_may_have_started"]:
                return pending.handle.event
            now_ns, wall = self._transition_clock_locked("mark_post")
            pending.handle.event.update(
                post_may_have_started=True,
                post_started_at_elapsed_ns=now_ns - self._created_ns,
                post_started_at_unix=wall)
            return pending.handle.event

    def commit(self, handle: AdmissionHandle, *, reason: str | None = None) \
            -> dict:
        """Commit after headers or an ambiguous transport failure.

        The commit timestamp, rather than the earlier reserve timestamp, owns
        rolling-window expiry.
        """
        if reason is not None and (
                not isinstance(reason, str) or not reason):
            raise ValueError("commit reason must be null or a non-empty string")
        with self._lock:
            if isinstance(handle, AdmissionHandle) \
                    and handle._guard_id == self._guard_id \
                    and handle._state == "committed":
                return handle.event
            pending = self._pending_for_locked(handle)
            if not pending.handle.event["post_may_have_started"]:
                raise ValueError(
                    "cannot commit a reservation before the POST may have "
                    "started; cancel it instead")
            now_ns, wall = self._transition_clock_locked("commit")
            token = pending.handle._token
            assert token is not None  # validated by _pending_for_locked
            self._pending.pop(token)
            for name, budget in self._budgets.items():
                amount = pending.amounts[budget["reservation_field"]]
                self._provisional_sums[name] -= amount
                self._committed_sums[name] += amount
                self._committed[name].append((now_ns, amount, token))
            pending.handle._state = "committed"
            pending.handle.event.update(
                state="committed",
                committed_at_elapsed_ns=now_ns - self._created_ns,
                committed_at_unix=wall,
                transition_reason=reason)
            self._counts["committed"] += 1
            return pending.handle.event

    def cancel_before_post(
            self, handle: AdmissionHandle, *,
            reason: str | None = "cancelled_before_conn_request") -> dict:
        """Release only when the caller proves ``conn.request`` was not called."""
        if reason is not None and (
                not isinstance(reason, str) or not reason):
            raise ValueError(
                "cancellation reason must be null or a non-empty string")
        with self._lock:
            if isinstance(handle, AdmissionHandle) \
                    and handle._guard_id == self._guard_id \
                    and handle._state == "cancelled_before_post":
                return handle.event
            pending = self._pending_for_locked(handle)
            if pending.handle.event["post_may_have_started"]:
                raise ValueError(
                    "cannot release a reservation after the POST may have "
                    "started; commit it conservatively")
            now_ns, wall = self._transition_clock_locked(
                "cancel_before_post")
            token = pending.handle._token
            assert token is not None
            self._pending.pop(token)
            for name, budget in self._budgets.items():
                self._provisional_sums[name] -= pending.amounts[
                    budget["reservation_field"]]
            pending.handle._state = "cancelled_before_post"
            pending.handle.event.update(
                state="cancelled_before_post",
                cancelled_at_elapsed_ns=now_ns - self._created_ns,
                cancelled_at_unix=wall,
                transition_reason=reason)
            self._counts["cancelled_before_post"] += 1
            return pending.handle.event

    def _validate_prior_event(self, event: object) -> tuple[tuple[str, int],
                                                             bool, dict]:
        if not isinstance(event, dict):
            raise ValueError("prior runtime quota event must be an object")
        if event.get("schema_version") != self.schema_version:
            raise ValueError("prior runtime quota event schema is unsupported")
        prior_guard = event.get("guard_id")
        sequence = event.get("sequence")
        if not isinstance(prior_guard, str) or not prior_guard \
                or not isinstance(sequence, int) or isinstance(sequence, bool) \
                or sequence <= 0:
            raise ValueError("prior runtime quota event identity is invalid")
        if event.get("scope_id") != self._scope_id \
                or event.get("shard_index") != self._shard_index \
                or event.get("shard_total") != self._shard_total:
            raise ValueError(
                "prior runtime quota event belongs to a different scope or "
                "shard allocation")
        decision = event.get("decision")
        state = event.get("state")
        if decision not in {"admitted", "denied"} or state not in {
                "committed", "cancelled_before_post", "denied"}:
            raise ValueError(
                "prior runtime quota event is not terminal and cannot be "
                "seeded")
        post_may_have_started = event.get("post_may_have_started")
        valid_terminal_state = (
            (decision, state, post_may_have_started)
            in {
                ("admitted", "committed", True),
                ("admitted", "cancelled_before_post", False),
                ("denied", "denied", False),
            }
        )
        if not valid_terminal_state:
            raise ValueError(
                "prior runtime quota event terminal state is inconsistent")
        reservation = event.get("reservation")
        if not isinstance(reservation, dict) \
                or set(reservation) != {
                    "request_bytes", "input_tokens", "output_tokens",
                    "queries"}:
            raise ValueError(
                "prior runtime quota event reservation is invalid")
        for name, value in reservation.items():
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 0:
                raise ValueError(
                    f"prior runtime quota event {name} is invalid")
        if reservation["queries"] != 1:
            raise ValueError(
                "prior runtime quota event must represent one physical POST")
        hard_checks = event.get("hard_limit_checks")
        if not isinstance(hard_checks, dict) \
                or set(hard_checks) != set(self._per_request_limits):
            raise ValueError(
                "prior runtime quota event hard-limit evidence is invalid")
        for name, limit in self._per_request_limits.items():
            check = hard_checks[name]
            measured = reservation[limit["reservation_field"]]
            expected_exceeded = measured > limit["maximum_integer"]
            if not isinstance(check, dict) or check.get("reservation") != measured \
                    or check.get("configured_limit") != \
                    limit["configured_limit"] \
                    or check.get("maximum_integer") != \
                    limit["maximum_integer"] \
                    or check.get("measurement") != limit["measurement"] \
                    or check.get("comparison") != limit["comparison"] \
                    or check.get("allocation") != limit["allocation"] \
                    or check.get("exceeded") is not expected_exceeded:
                raise ValueError(
                    "prior runtime quota event hard-limit evidence is "
                    "inconsistent")
            if decision == "admitted" and expected_exceeded:
                raise ValueError(
                    "prior admitted runtime quota event violates a hard "
                    "per-request limit")
        consuming = bool(
            decision == "admitted" and state == "committed"
            and post_may_have_started is True)
        return (prior_guard, sequence), consuming, copy.deepcopy(reservation)

    def seed_prior_events(self, events: Iterable[dict]) -> dict:
        """Conservatively import terminal prior events, packed at ``now``.

        Wall-clock age is not trusted across processes. Packing every imported
        consuming event at the current monotonic instant can only reduce local
        headroom. Repeated imports and rows already accounted by this same guard
        are deduplicated by ``(guard_id, sequence)``.
        """
        if isinstance(events, (str, bytes, dict)):
            raise ValueError("prior runtime quota events must be an iterable")
        prepared = []
        for event in events:
            identity, consuming, amounts = self._validate_prior_event(event)
            prepared.append((
                identity, consuming, amounts,
                event.get("decision") == "denied",
                event.get("reason_code"),
                copy.deepcopy(event.get("denied_dimensions")),
            ))
        imported = deduplicated = nonconsuming = 0
        first_seeded_denial = None
        with self._lock:
            try:
                now_ns, wall = self._now_locked()
                self._evict_locked(now_ns)
            except Exception as exc:
                self._latch_internal_error_locked(
                    exc, operation="seed_prior_events")
                raise RuntimeQuotaGuardError(
                    "runtime quota guard tripped while seeding prior events") \
                    from exc
            for ((prior_guard, sequence), consuming, amounts, denied,
                 denial_reason, denied_dimensions) in prepared:
                key = (prior_guard, sequence)
                if prior_guard == self._guard_id:
                    if sequence <= self._sequence:
                        deduplicated += 1
                        continue
                    raise ValueError(
                        "prior event claims an unknown future sequence for "
                        "this live guard")
                if key in self._external_prior_keys:
                    deduplicated += 1
                    continue
                self._external_prior_keys.add(key)
                if not consuming:
                    nonconsuming += 1
                    if denied and first_seeded_denial is None:
                        first_seeded_denial = {
                            "source_guard_id": prior_guard,
                            "source_sequence": sequence,
                            "reason_code": (
                                denial_reason
                                if isinstance(denial_reason, str)
                                and denial_reason
                                else "seeded_prior_quota_denial"),
                            "denied_dimensions": (
                                denied_dimensions
                                if isinstance(denied_dimensions, list)
                                and all(isinstance(item, str)
                                        for item in denied_dimensions)
                                else []),
                        }
                    continue
                imported += 1
                reservation_id = f"prior:{prior_guard}:{sequence}"
                for name, budget in self._budgets.items():
                    amount = amounts[budget["reservation_field"]]
                    self._committed_sums[name] += amount
                    self._committed[name].append(
                        (now_ns, amount, reservation_id))

            self._counts["seeded_committed"] += imported
            self._counts["seeded_nonconsuming"] += nonconsuming
            self._counts["seeded_deduplicated"] += deduplicated
            active = self._active_before_locked()
            exceeded = [
                name for name, value in active.items()
                if value > self._budgets[name]["local_max_integer"]
            ]
            if exceeded:
                self._trip_locked(
                    now_ns=now_ns, wall=wall, sequence=None,
                    reason_code="seeded_prior_usage_exceeds_warning_budget",
                    denied_dimensions=exceeded)
            elif first_seeded_denial is not None:
                # A terminal denial is a command-level safety stop, not a
                # zero-cost historical observation.  Reconstructing a guard
                # for prior CLI traffic must never reset that trip and admit
                # later replay POSTs.
                self._trip_locked(
                    now_ns=now_ns, wall=wall, sequence=None,
                    reason_code=first_seeded_denial["reason_code"],
                    denied_dimensions=first_seeded_denial[
                        "denied_dimensions"],
                    prior_event={
                        "source_guard_id": first_seeded_denial[
                            "source_guard_id"],
                        "source_sequence": first_seeded_denial[
                            "source_sequence"],
                        "operation": "seed_prior_events",
                    })
        return {
            "imported": imported,
            "deduplicated": deduplicated,
            "nonconsuming": nonconsuming,
            "tripped": self.tripped,
        }

    def seed_prior_rows(self, rows: Iterable[dict], *,
                        event_field: str = "quota_guard_events") -> dict:
        """Seed exact row evidence and reject sent rows without guard events."""
        if isinstance(rows, (str, bytes, dict)):
            raise ValueError("prior runtime quota rows must be an iterable")
        all_events = []
        row_count = 0
        for position, row in enumerate(rows):
            row_count += 1
            if not isinstance(row, dict):
                raise ValueError(f"prior runtime quota row {position} is invalid")
            attempts = row.get("request_attempts")
            if not isinstance(attempts, int) or isinstance(attempts, bool) \
                    or attempts < 0:
                raise ValueError(
                    f"prior runtime quota row {position} has unknown attempts")
            events = row.get(event_field)
            if events is None:
                events = []
            if not isinstance(events, list):
                raise ValueError(
                    f"prior runtime quota row {position} events are invalid")
            for event in events:
                self._validate_prior_event(event)
            committed_posts = sum(
                isinstance(event, dict)
                and event.get("decision") == "admitted"
                and event.get("state") == "committed"
                and event.get("post_may_have_started") is True
                for event in events)
            committed_identities = [
                (event.get("guard_id"), event.get("sequence"))
                for event in events
                if isinstance(event, dict)
                and event.get("decision") == "admitted"
                and event.get("state") == "committed"
                and event.get("post_may_have_started") is True
            ]
            if committed_posts != attempts:
                raise ValueError(
                    f"prior runtime quota row {position} physical attempts "
                    "disagree with committed guard events")
            if len(set(committed_identities)) != len(committed_identities):
                raise ValueError(
                    f"prior runtime quota row {position} repeats committed "
                    "guard evidence for distinct physical attempts")
            all_events.extend(events)
        result = self.seed_prior_events(all_events)
        return {"rows": row_count, **result}

    def snapshot(self) -> dict:
        """Return a complete JSON-safe guard state without resetting it."""
        with self._lock:
            clock_status = "valid"
            if self._tripped:
                now_ns, wall = self._last_clock_ns, self._last_wall
                clock_status = "not_rechecked_after_permanent_trip"
            else:
                try:
                    now_ns, wall = self._now_locked()
                    self._evict_locked(now_ns)
                except Exception as exc:
                    self._latch_internal_error_locked(
                        exc, operation="snapshot")
                    now_ns, wall = self._last_clock_ns, self._last_wall
                    clock_status = "invalid_latched"
            dimensions = {}
            for name, budget in self._budgets.items():
                committed = self._committed_sums[name]
                provisional = self._provisional_sums[name]
                total = committed + provisional
                dimensions[name] = {
                    key: value for key, value in budget.items()
                    if key not in {"reservation_field", "window_ns"}
                } | {
                    "reservation_field": budget["reservation_field"],
                    "active_committed": committed,
                    "active_provisional": provisional,
                    "active_total": total,
                    "remaining_local_integer": max(
                        budget["local_max_integer"] - total, 0),
                }
            return {
                "schema_version": self.schema_version,
                "guard_id": self._guard_id,
                "scope_id": self._scope_id,
                "coverage": "one_harness_command_only",
                "provider_headroom_proven": False,
                "external_workspace_traffic_included": False,
                "created_at_unix": self._created_unix,
                "snapshot_at_unix": wall,
                "snapshot_at_elapsed_ns": now_ns - self._created_ns,
                "clock_status": clock_status,
                "shard_index": self._shard_index,
                "shard_total": self._shard_total,
                "allocation": "deterministic_static_shard_partition",
                "tripped": self._tripped,
                "trip": copy.deepcopy(self._trip),
                "sequence": self._sequence,
                "provisional_reservations": len(self._pending),
                "admitted_attempts": self._counts["admitted"],
                "committed_attempts": self._counts["committed"],
                "denied_attempts": self._counts["denied"],
                "counts": copy.deepcopy(self._counts),
                "seeded_guard_ids": sorted({
                    guard_id for guard_id, _sequence
                    in self._external_prior_keys}),
                "dimensions": dimensions,
                "hard_limits": copy.deepcopy(self._per_request_limits),
            }


def _synthetic_json_escape_overhead(content_chars: int) -> int:
    if content_chars <= 0:
        return 4
    return (2 * math.ceil(
        content_chars / _SYNTHETIC_JSON_ESCAPE_BLOCK_CHARS) + 4)


def _snapshot_freshness(rate_limits: dict, *, today: date | None = None) \
        -> dict:
    """Return an explicit, fail-closed freshness assessment.

    ``as_of`` describes the provider fact. ``verified_at`` records when the
    operator actually rechecked that fact. Paid traffic is allowed only while
    the latter remains inside the configured review window.
    """
    checked_on = today or date.today()
    verified_at = rate_limits.get("verified_at")
    max_age_days = rate_limits.get("max_age_days")
    out = {
        "status": "missing",
        "fresh": False,
        "checked_on": checked_on.isoformat(),
        "verified_at": verified_at,
        "max_age_days": max_age_days,
        "age_days": None,
        "source": rate_limits.get("source"),
        "source_as_of": rate_limits.get("as_of"),
    }
    if not isinstance(verified_at, str) \
            or isinstance(max_age_days, bool) \
            or not isinstance(max_age_days, int) \
            or max_age_days <= 0:
        return out
    try:
        verified_date = date.fromisoformat(verified_at)
    except ValueError:
        out["status"] = "invalid"
        return out
    age_days = (checked_on - verified_date).days
    out["age_days"] = age_days
    if age_days < 0:
        out["status"] = "invalid"
    elif age_days > max_age_days:
        out["status"] = "stale"
    else:
        out["status"] = "fresh"
        out["fresh"] = True
    return out


def _attempt_multiplier(rc: "RunConfig") -> int:
    """Worst configured number of POSTs one logical request can produce.

    ``max_retries`` covers transport retries.  The client can additionally
    retry once when ``stream_options`` is rejected and once after a proven
    credential refresh.  Counting both for every logical request is
    intentionally conservative; concurrent first requests can race before
    shared client state learns either result.
    """
    retries = int(rc.endpoint.get("max_retries", 0) or 0)
    return retries + 3


def _schedule(rc: "RunConfig") -> list[float]:
    from .schedule import load_trace, make_schedule

    if rc.timestamps_file:
        generated = load_trace(
            rc.timestamps_file, duration_cap_s=rc.duration_s)
    else:
        generated = make_schedule(
            duration_s=rc.duration_s,
            qps_base=rc.qps_base,
            qps_burst=rc.qps_burst,
            qps_min=rc.qps_min,
            qps_max=rc.qps_max,
            rate_scale=rc.rate_scale,
            seed=rc.seed + 16,
        )
    return [float(value) for value in generated["timestamps"]]


def _messages_input_upper_bound(endpoint, messages,
                                max_output: int) -> int | None:
    """Bound input from exact submitted JSON plus provider-owned framing."""
    if not isinstance(messages, list) or not messages:
        return None
    for message in messages:
        if not isinstance(message, dict):
            return None
    from .client import serialize_request_body

    body = serialize_request_body(
        endpoint, messages, max_output, endpoint.include_usage)
    return len(body) + _CHAT_FRAMING_TOKEN_ALLOWANCE * (len(messages) + 1)


def _prior_prompt_usage(row: dict) -> int | None:
    """Return only explicit, protocol-clean provider input usage.

    Prior rows describe setup POSTs which already happened. Their intended
    profile size is never a safe retrospective substitute, and a partial or
    malformed stream cannot be promoted to trusted token evidence merely
    because it happened to contain a usage-shaped object.
    """
    if row.get("status") != 200 or row.get("ok") is not True \
            or row.get("stream_complete") is not True:
        return None
    parse_errors = row.get("parse_errors")
    if not isinstance(parse_errors, int) or isinstance(parse_errors, bool) \
            or parse_errors != 0:
        return None
    prompt = row.get("prompt_tokens")
    completion = row.get("completion_tokens")
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt <= 0:
        return None
    if not isinstance(completion, int) or isinstance(completion, bool) \
            or completion < 0:
        return None
    cached = row.get("cached_tokens")
    if cached is not None and (
            not isinstance(cached, int) or isinstance(cached, bool)
            or cached < 0 or cached > prompt):
        return None
    reasoning = row.get("reasoning_tokens")
    if reasoning is not None and (
            not isinstance(reasoning, int) or isinstance(reasoning, bool)
            or reasoning < 0 or reasoning > completion):
        return None
    return prompt


def _prior_attempt_count(row: dict) -> int | None:
    """Validate the minimum cross-field evidence needed for POST counting."""
    attempts = row.get("request_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) \
            or attempts < 0:
        return None
    connections = row.get("connection_attempts")
    if not isinstance(connections, int) or isinstance(connections, bool) \
            or connections < attempts:
        return None
    retries = row.get("retries")
    retry_reasons = row.get("retry_reasons")
    if not isinstance(retries, int) or isinstance(retries, bool) \
            or retries < 0 or not isinstance(retry_reasons, list) \
            or any(not isinstance(reason, str) or not reason
                   for reason in retry_reasons) \
            or retries != len(retry_reasons):
        return None
    if attempts == 0:
        sent_evidence = (
            row.get("first_send_unix"), row.get("t_send_unix"),
            row.get("status"), row.get("prompt_tokens"),
            row.get("completion_tokens"), row.get("cached_tokens"),
            row.get("reasoning_tokens"),
            True if row.get("ok") is True else None,
            True if row.get("stream_complete") is True else None,
        )
        return 0 if all(value is None for value in sent_evidence) else None
    first_send = row.get("first_send_unix")
    if isinstance(first_send, bool) or not isinstance(first_send, (int, float)) \
            or not math.isfinite(float(first_send)) or first_send < 0:
        return None
    last_send = row.get("t_send_unix")
    if isinstance(last_send, bool) or not isinstance(last_send, (int, float)) \
            or not math.isfinite(float(last_send)) \
            or last_send < float(first_send):
        return None
    return attempts


def _prior_committed_reservations(
        row: dict, attempts: int) -> list[dict] | None:
    """Validate reservations for every physical POST in a prior row.

    Provider usage is optional on otherwise valid responses.  The runtime
    quota guard nevertheless records a conservative input reservation at the
    last safe point before each POST.  Accept that evidence only when there is
    one unique, admitted, committed, post-started event per observed physical
    attempt and every reservation has the exact current shape.  Denied and
    proven-unsent terminal events do not represent POSTs and are ignored.
    """
    events = row.get("quota_guard_events")
    if not isinstance(events, list):
        return None
    reservations = []
    identities = set()
    for event in events:
        if not isinstance(event, dict) \
                or event.get("decision") != "admitted" \
                or event.get("state") != "committed" \
                or event.get("post_may_have_started") is not True:
            continue
        guard_id = event.get("guard_id")
        sequence = event.get("sequence")
        identity = (guard_id, sequence)
        if not isinstance(guard_id, str) or not guard_id \
                or not isinstance(sequence, int) \
                or isinstance(sequence, bool) or sequence <= 0 \
                or identity in identities:
            return None
        reservation = event.get("reservation")
        if not isinstance(reservation, dict) or set(reservation) != {
                "request_bytes", "input_tokens", "output_tokens",
                "queries"}:
            return None
        if any(not isinstance(value, int) or isinstance(value, bool)
               or value < 0 for value in reservation.values()) \
                or reservation["queries"] != 1:
            return None
        identities.add(identity)
        reservations.append(reservation)
    if len(reservations) != attempts:
        return None
    return reservations


def _prior_request_bytes(row: dict, attempts: int) -> int | None:
    """Recover exact physical-body bytes from runtime-admission evidence."""
    reservations = _prior_committed_reservations(row, attempts)
    if reservations is None:
        return None
    return max(
        (reservation["request_bytes"] for reservation in reservations),
        default=0)


def _prior_reserved_input_usage(row: dict, attempts: int) -> int | None:
    """Sum conservative input reservations when provider usage is absent."""
    reservations = _prior_committed_reservations(row, attempts)
    if reservations is None:
        return None
    return sum(reservation["input_tokens"] for reservation in reservations)


def _plan_values(endpoint, plan: dict) -> tuple[int | None, int]:
    output_tokens = plan.get("max_output")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) \
            or output_tokens <= 0:
        raise ValueError("planned request needs a positive max_output")
    plan_endpoint = endpoint
    quota_extra_body = plan.get("_quota_extra_body")
    if quota_extra_body is not None:
        plan_endpoint = dataclasses.replace(
            endpoint, extra_body=copy.deepcopy(quota_extra_body))
    input_tokens = _messages_input_upper_bound(
        plan_endpoint, plan.get("messages"), output_tokens)
    return input_tokens, output_tokens


def _workload_values(endpoint, workload, index: int, *, post_calibration: bool) \
        -> tuple[int | None, int]:
    """Return a request's conservative input bound and output reservation."""
    if workload.prompts_mode:
        plan = workload.plan(index, f"quota-plan-prompt-{index}")
        return _plan_values(endpoint, plan)
    input_tokens = int(workload.draw["input_tokens"][index])
    cpt_ceiling = float(workload.rc.cpt)
    if post_calibration:
        cpt_ceiling = max(cpt_ceiling, _CALIBRATED_CPT_HARD_MAX)
    # Synthetic materialization is ASCII and targets round(tokens * cpt)
    # content characters.  ``ceil`` is a monotonic upper bound for that round,
    # and ASCII characters are one UTF-8 byte each.
    # TextMaterializer emits at most one system and one user message.
    output_tokens = min(
        int(workload.draw["output_tokens"][index]),
        int(workload.rc.max_output_tokens_cap),
    )
    # TextMaterializer emits at most one system and one user message. Serialize
    # that worst message shape with empty content using the exact client body
    # builder, then add the conservative ASCII content-byte ceiling. This
    # includes model, extra_body (including tools/schema), request controls,
    # roles, keys, and JSON syntax without materializing every large prompt.
    empty_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ]
    serialized_empty = _messages_input_upper_bound(
        endpoint, empty_messages, output_tokens)
    if serialized_empty is None:  # pragma: no cover - fixed local shape
        return None, output_tokens
    content_chars = int(math.ceil(input_tokens * cpt_ceiling))
    input_upper_bound = (
        content_chars
        + _synthetic_json_escape_overhead(content_chars)
        + serialized_empty)
    return input_upper_bound, output_tokens


def _logical_events(rc: "RunConfig", *, offset_s: float = 0.0,
                    setup_plans: Iterable[dict] = (),
                    prior_rows: Iterable[dict] = (),
                    prevalidated=None) \
        -> tuple[list[dict], list[str], int]:
    """Return weighted attempt events and any unplannable input evidence.

    CLI callers can supply the exact endpoint-free validation result used by
    their paid preflight.  Reusing its schedule and sampled workload prevents
    the quota gate from silently planning a second read or a second random
    realization of the inputs it is about to authorize.
    """
    from .client import EndpointConfig
    from .runner import _PreparedWorkload

    multiplier = _attempt_multiplier(rc)
    endpoint = EndpointConfig(**rc.endpoint)
    if prevalidated is not None:
        if prevalidated.rc != rc:
            raise ValueError(
                "prevalidated quota inputs do not match the run configuration")
        if prevalidated.full_schedule is None:
            raise ValueError(
                "prevalidated quota inputs have no fixed schedule")
        timestamps = [
            float(value)
            for value in prevalidated.full_schedule["timestamps"]
        ]
        workload = prevalidated.workload
        if workload is None or workload.total_n != len(timestamps):
            raise ValueError(
                "prevalidated quota workload does not match its schedule")
    else:
        timestamps = _schedule(rc)
        workload = _PreparedWorkload(rc, len(timestamps)) if timestamps else None
    if not timestamps:
        raise ValueError(
            "schedule produced zero arrivals; increase the fixed rate or "
            "duration")
    events: list[dict] = []
    unknowns: list[str] = []

    def add(timestamp: float, input_tokens: int | None,
            output_tokens: int, attempts: int, phase: str, *,
            request_bytes: int | None = None,
            input_tokens_total: int | None = None) -> None:
        if input_tokens is None and input_tokens_total is None:
            unknowns.append(
                f"{phase} input tokens are unknown without provider usage")
        events.append({
            "t": float(timestamp),
            "input_tokens": (
                int(input_tokens_total)
                if input_tokens_total is not None else
                None if input_tokens is None else
                int(input_tokens) * attempts),
            "output_tokens": int(output_tokens) * attempts,
            "queries": attempts,
            # One upper bound per physical POST, not the sum across retries.
            # Setup/replay planning supplies a conservative serialized-body
            # bound; observed prior rows supply exact runtime-guard evidence.
            "request_bytes": request_bytes,
            "logical_requests": 1,
            "phase": phase,
        })

    # Setup requests have no reliable duration before they are sent. Packing
    # them at the first replay instant is conservative for every rolling
    # window and prevents cooldown language from manufacturing headroom.
    for plan in setup_plans:
        inp, out = _plan_values(endpoint, plan)
        add(offset_s, inp, out, multiplier, "preflight_or_probe",
            request_bytes=inp)

    for position, row in enumerate(prior_rows):
        if not isinstance(row, dict):
            unknowns.append(f"prior request row {position} is not an object")
            continue
        attempts = _prior_attempt_count(row)
        if attempts is None:
            unknowns.append(
                f"prior request row {position} has unknown provider attempts "
                "because its send evidence is incomplete or inconsistent")
            continue
        if attempts == 0:
            continue
        # Only provider-reported input usage can retrospectively replace the
        # pre-traffic byte upper bound.  Falling back to an intended profile
        # target here would make an already-observed request less conservative.
        inp = _prior_prompt_usage(row)
        reserved_input_total = (
            _prior_reserved_input_usage(row, attempts)
            if inp is None else None)
        out = row.get("max_tokens_requested")
        if not isinstance(out, int) or isinstance(out, bool) or out <= 0:
            unknowns.append(
                f"prior request row {position} has unknown max_tokens")
            continue
        add(offset_s, inp, out, attempts, "prior_request",
            request_bytes=_prior_request_bytes(row, attempts),
            input_tokens_total=reserved_input_total)

    # Every shard performs its own calibration pass.  All shards can start
    # together, so place those requests at the same conservative instant.
    calibration_n = min(int(rc.calibrate_n), len(timestamps))
    for index in range(calibration_n):
        inp, out = _workload_values(
            endpoint, workload, index, post_calibration=False)
        for _shard in range(int(rc.shard_total)):
            add(offset_s, inp, out, multiplier, "calibration",
                request_bytes=inp)

    # A workspace quota is shared by the shards.  Plan the complete unsharded
    # schedule on every shard rather than blessing each fragment in isolation.
    for index, timestamp in enumerate(timestamps):
        inp, out = _workload_values(
            endpoint, workload, index, post_calibration=True)
        add(offset_s + timestamp, inp, out, multiplier, "replay",
            request_bytes=inp)
    return events, sorted(set(unknowns)), len(timestamps)


def _setup_events(rc: "RunConfig", plans: Iterable[dict], *,
                  offset_s: float = 0.0) -> tuple[list[dict], list[str]]:
    """Plan only CLI preflight/probe traffic without building a run twice."""
    from .client import EndpointConfig

    multiplier = _attempt_multiplier(rc)
    endpoint = EndpointConfig(**rc.endpoint)
    events = []
    unknowns = []
    for plan in plans:
        inp, out = _plan_values(endpoint, plan)
        if inp is None:
            unknowns.append(
                "preflight_or_probe input tokens are unknown without "
                "provider usage")
        events.append({
            "t": float(offset_s),
            "input_tokens": None if inp is None else inp * multiplier,
            "output_tokens": out * multiplier,
            "queries": multiplier,
            "request_bytes": inp,
            "logical_requests": 1,
            "phase": "preflight_or_probe",
        })
    return events, sorted(set(unknowns))


def _rolling_peak(events: list[dict], field: str, window_s: float) \
        -> int | None:
    if any(event[field] is None for event in events):
        return None
    ordered = sorted(events, key=lambda event: event["t"])
    active: deque[dict] = deque()
    total = 0
    peak = 0
    for event in ordered:
        timestamp = event["t"]
        # The provider does not publish whether an event exactly on the
        # rolling-window boundary is included.  Keep it in the local budget
        # rather than manufacturing headroom from a boundary convention that
        # this harness cannot prove.  It falls out only once it is strictly
        # older than the configured window.
        while active and timestamp - active[0]["t"] > window_s:
            total -= int(active.popleft()[field])
        active.append(event)
        total += int(event[field])
        peak = max(peak, total)
    return peak


def _evaluate(events: list[dict], rate_limits: dict, *,
              unknowns: Iterable[str], logical_replay_requests: int,
              attempt_multiplier: int, plan_kind: str,
              planned_rungs: list[float] | None = None) -> dict:
    warning = float(rate_limits["warning_utilization"])
    fields = (
        ("input_tokens_per_minute", "input_tokens", 60.0,
         "complete serialized JSON byte upper bound plus chat framing"),
        ("output_tokens_per_minute", "output_tokens", 60.0,
         "offered max_tokens reservations"),
        ("queries_per_hour", "queries", 3600.0,
         "worst-case physical POST attempts"),
        ("queries_per_second", "queries", 1.0,
         "worst-case physical POST attempts"),
    )
    windows = {}
    refusal_reasons: list[str] = []
    freshness = _snapshot_freshness(rate_limits)
    if not freshness["fresh"]:
        status = freshness["status"]
        if status == "stale":
            refusal_reasons.append(
                "rate-limit snapshot is stale: verified "
                f"{freshness['verified_at']} ({freshness['age_days']} days "
                f"old), exceeding max_age_days={freshness['max_age_days']}; "
                "recheck the cited provider source before paid traffic")
        elif status == "missing":
            refusal_reasons.append(
                "rate-limit snapshot has no verified_at/max_age_days "
                "freshness proof; recheck the cited provider source before "
                "paid traffic")
        else:
            refusal_reasons.append(
                "rate-limit snapshot freshness metadata is invalid; recheck "
                "the cited provider source before paid traffic")
    unknown_list = sorted(set(str(item) for item in unknowns))
    for limit_name, event_field, seconds, evidence in fields:
        if limit_name not in rate_limits:
            continue
        limit = float(rate_limits[limit_name])
        peak = _rolling_peak(events, event_field, seconds)
        ratio = None if peak is None else float(peak) / limit
        entry = {
            "window_seconds": seconds,
            "planned_peak": peak,
            "configured_limit": limit,
            "ratio_to_configured_limit": ratio,
            "warning_ratio": warning,
            "evidence": evidence,
        }
        windows[limit_name] = entry
        if peak is None:
            refusal_reasons.append(
                f"{limit_name} cannot be bounded from the configured input")
        elif ratio >= warning:
            refusal_reasons.append(
                f"planned {limit_name} peak {peak:,} is {ratio:.1%} of "
                f"the configured {limit:g} limit, at or above the "
                f"{warning:.1%} warning budget")

    for unknown in unknown_list:
        if "unknown provider attempts" in unknown \
                or "is not an object" in unknown:
            refusal_reasons.append(
                "provider-attempt count is unknown, so planned query and "
                "token demand cannot be bounded")
        elif "unknown max_tokens" in unknown \
                and "output_tokens_per_minute" in rate_limits:
            refusal_reasons.append(
                "offered output reservation is unknown, so planned output "
                "token demand cannot be bounded")

    hard_limits = {}
    if "request_bytes_max" in rate_limits:
        limit = int(rate_limits["request_bytes_max"])
        values = [event.get("request_bytes") for event in events]
        peak = (None if any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 0 for value in values) else max(values, default=0))
        hard_limits["request_bytes_max"] = {
            "planned_max": peak,
            "configured_limit": limit,
            "ratio_to_configured_limit": (
                None if peak is None else peak / limit),
            "comparison": "less_than_or_equal",
            "evidence": (
                "conservative serialized-body upper bound before traffic; "
                "runtime admission rechecks exact bytes for every POST"),
        }
        if peak is None:
            refusal_reasons.append(
                "per-request payload bytes cannot be bounded from the "
                "captured prior-request evidence")
        elif peak > limit:
            refusal_reasons.append(
                f"planned request payload upper bound {peak:,} bytes exceeds "
                f"the configured {limit:,}-byte per-request limit")
    refusal_reasons = list(dict.fromkeys(refusal_reasons))

    if any("input tokens are unknown" in item for item in unknown_list) \
            and "input_tokens_per_minute" not in rate_limits:
        # Keep the limitation visible, but it is not a blocker when no input
        # token policy was supplied.
        pass
    plan = {
        "schema_version": 1,
        "plan_kind": plan_kind,
        "status": ("refused" if refusal_reasons
                   else "within_configured_harness_warning_budget"),
        "may_start": not refusal_reasons,
        "provider_headroom_proven": False,
        "workspace_external_traffic_included": False,
        "rate_limit_snapshot_freshness": freshness,
        "logical_replay_requests": int(logical_replay_requests),
        "planned_physical_attempts_worst_case": (
            None if any("unknown provider attempts" in item
                        for item in unknown_list)
            else int(sum(event["queries"] for event in events))),
        "physical_attempts_per_logical_worst_case": attempt_multiplier,
        "planned_rungs_requests_per_second": planned_rungs,
        "windows": windows,
        "hard_limits": hard_limits,
        "unknowns": unknown_list,
        "refusal_reasons": refusal_reasons,
        "assumptions": [
            "Harness traffic is evaluated in isolation; unrelated workspace "
            "traffic can consume the same limits.",
            "Input demand is bounded at one token per UTF-8 byte of the "
            "complete serialized request JSON plus "
            f"{_CHAT_FRAMING_TOKEN_ALLOWANCE} tokens of chat framing for each "
            "message and one additional request-level block; roles, message "
            "metadata, model, tools, provider controls, and JSON syntax are "
            "included. This is intentionally stricter than intended profile "
            "token targets.",
            "Synthetic replay content is planned using the larger of its "
            "configured chars-per-token value and the calibrated hard maximum "
            f"of {_CALIBRATED_CPT_HARD_MAX:g}.",
            "Output uses max_tokens offered at admission, not eventual output "
            "consumption.",
            "Physical-attempt planning includes configured transport retries, "
            "one stream-options fallback, and one credential-refresh retry "
            "per logical request.",
            "Workspace QPS, when configured, uses the same conservative "
            "inclusive rolling-boundary policy as token and QPH windows.",
            "The per-request payload ceiling is checked against a conservative "
            "serialized-body upper bound before traffic and exact serialized "
            "bytes immediately before every physical POST.",
            "Setup traffic is packed against the first replay window; spacing "
            "is not treated as proof of quota reset.",
        ],
        "rate_limits": copy.deepcopy(rate_limits),
    }
    return plan


def plan_run_quota(rc: "RunConfig", *,
                   setup_plans: Iterable[dict] = (),
                   prior_rows: Iterable[dict] = (),
                   prevalidated=None) -> dict | None:
    """Plan one run without making any network call."""
    if rc.rate_limits is None:
        return None
    if rc.sizing_concurrency is not None:
        plan = _evaluate(
            [], rc.rate_limits,
            unknowns=[
                "sizing_concurrency derives the replay rate from paid endpoint "
                "traffic, so the schedule is unknowable before traffic starts"
            ],
            logical_replay_requests=0,
            attempt_multiplier=_attempt_multiplier(rc),
            plan_kind="run",
        )
        plan["refusal_reasons"].append(
            "quota-aware runs require a fixed rate or timestamp trace")
        plan["may_start"] = False
        plan["status"] = "refused"
        return plan
    events, unknowns, replay_n = _logical_events(
        rc, setup_plans=setup_plans, prior_rows=prior_rows,
        prevalidated=prevalidated)
    return _evaluate(
        events, rc.rate_limits, unknowns=unknowns,
        logical_replay_requests=replay_n,
        attempt_multiplier=_attempt_multiplier(rc), plan_kind="run")


def plan_sweep_quota(base_config: dict, rates: Iterable[float], *,
                     duration_s: int, cooldown_s: float,
                     setup_plans: Iterable[dict] = (),
                     prevalidated_rungs: Iterable | None = None) -> dict | None:
    """Plan the union of every requested sweep rung before preflight traffic."""
    from .runner import RunConfig

    base = copy.deepcopy(base_config)
    if base.get("rate_limits") is None:
        return None
    rate_values = [float(rate) for rate in rates]
    if not rate_values:
        raise ValueError("quota planner needs at least one sweep rung")
    validated = (None if prevalidated_rungs is None
                 else list(prevalidated_rungs))
    if validated is not None and len(validated) != len(rate_values):
        raise ValueError(
            "prevalidated sweep inputs must match every requested rung")
    setup = list(setup_plans)
    all_events: list[dict] = []
    unknowns: list[str] = []
    replay_total = 0
    offset = 0.0

    # Setup occurs once for the whole sweep. Use a valid first-rung config to
    # determine the endpoint retry contract and request budgets.
    if validated is not None:
        first_rc = validated[0].rc
    else:
        first_cfg = copy.deepcopy(base)
        first_cfg.update(
            qps_base=rate_values[0], qps_burst=rate_values[0],
            qps_min=rate_values[0], qps_max=rate_values[0], rate_scale=1.0,
            duration_s=duration_s,
        )
        first_rc = RunConfig(**first_cfg)
    if setup:
        # Cooldown is operational spacing, not proof that a token bucket or
        # provider accounting window reset. Pack setup against the first rung
        # even when the command sleeps between them.
        offset = float(cooldown_s)
        setup_events, setup_unknowns = _setup_events(
            first_rc, setup, offset_s=offset)
        all_events.extend(setup_events)
        unknowns.extend(setup_unknowns)

    for position, rate in enumerate(rate_values):
        if validated is not None:
            checked = validated[position]
            rc = checked.rc
            if rc.sizing_concurrency is not None \
                    or rc.timestamps_file is not None \
                    or rc.duration_s != duration_s \
                    or float(rc.rate_scale) != 1.0 \
                    or any(float(value) != rate for value in (
                        rc.qps_base, rc.qps_burst, rc.qps_min, rc.qps_max)):
                raise ValueError(
                    f"prevalidated sweep rung {position} does not match "
                    f"{rate:g} requests/second for {duration_s}s")
        else:
            checked = None
            cfg = copy.deepcopy(base)
            cfg.update(
                qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
                rate_scale=1.0, duration_s=duration_s,
                out_dir=f"quota-plan-rate-{position}",
                title=f"quota plan at {rate:g} requests/second",
            )
            rc = RunConfig(**cfg)
        events, rung_unknowns, replay_n = _logical_events(
            rc, offset_s=offset, prevalidated=checked)
        all_events.extend(events)
        unknowns.extend(rung_unknowns)
        replay_total += replay_n
        offset += float(duration_s)
        if position < len(rate_values) - 1:
            offset += float(cooldown_s)

    return _evaluate(
        all_events, first_rc.rate_limits, unknowns=unknowns,
        logical_replay_requests=replay_total,
        attempt_multiplier=_attempt_multiplier(first_rc), plan_kind="sweep",
        planned_rungs=rate_values)


def bind_quota_plan_to_endpoint(plan: dict | None, binding: dict) \
        -> dict | None:
    """Attach control-plane binding evidence and make the final gate decision.

    Offline schedule safety and endpoint identity are independent facts.  A
    plan may reach this function only after the harness-only windows pass; it
    may start paid inference only when both facts are true.  A fresh copy is
    returned so callers cannot accidentally mutate a plan already written to
    an audit record.
    """
    if plan is None:
        return None
    bound = copy.deepcopy(plan)
    schedule_may_start = bool(
        bound.get("schedule_may_start", bound.get("may_start")))
    schedule_reasons = list(bound.get("schedule_refusal_reasons",
                                      bound.get("refusal_reasons") or []))
    binding_copy = copy.deepcopy(binding)
    binding_complete = binding_copy.get("binding_complete") is True
    binding_reasons = [
        f"endpoint binding: {reason}"
        for reason in (binding_copy.get("reasons") or [])
    ]
    if not binding_complete and not binding_reasons:
        binding_reasons = ["endpoint binding: verification was incomplete"]

    bound.update(
        schedule_may_start=schedule_may_start,
        schedule_refusal_reasons=schedule_reasons,
        endpoint_binding_required=True,
        endpoint_binding=binding_copy,
        may_start=bool(schedule_may_start and binding_complete),
    )
    bound["refusal_reasons"] = list(dict.fromkeys(
        schedule_reasons + ([] if binding_complete else binding_reasons)))
    if bound["may_start"]:
        bound["status"] = "ready_for_paid_inference"
        bound["refusal_stage"] = None
    else:
        bound["status"] = "refused"
        bound["refusal_stage"] = (
            "endpoint_binding" if schedule_may_start else "schedule")
    return bound


def enforce_quota_plan(plan: dict | None) -> None:
    if plan is not None and not plan.get("may_start"):
        raise QuotaPlanError(plan)


def render_quota_plan(plan: dict | None) -> str:
    """Short terminal rendering; the complete plan remains machine-readable."""
    if plan is None:
        return ""
    binding = plan.get("endpoint_binding")
    binding_blocked = bool(
        isinstance(binding, dict)
        and not binding.get("binding_complete")
        and plan.get("schedule_may_start"))
    if binding_blocked:
        headline = (
            "REFUSED: harness schedule is within its configured warning "
            "budget, but endpoint binding blocked paid inference")
    else:
        headline = (
            ("PASS" if plan["may_start"] else "REFUSED")
            + ": harness-only worst-case schedule; provider headroom is not "
              "proven")
    lines = ["[quota-plan] " + headline]
    freshness = plan.get("rate_limit_snapshot_freshness") or {}
    lines.append(
        "[quota-plan] rate-limit snapshot: "
        f"{str(freshness.get('status') or 'unknown').upper()}; "
        f"source as-of={freshness.get('source_as_of')}; "
        f"verified={freshness.get('verified_at')}; "
        f"age={freshness.get('age_days')} days; "
        f"max-age={freshness.get('max_age_days')} days; "
        f"checked={freshness.get('checked_on')}")
    for name, evidence in plan["windows"].items():
        peak = evidence["planned_peak"]
        ratio = evidence["ratio_to_configured_limit"]
        shown_peak = "unknown" if peak is None else f"{peak:,}"
        shown_ratio = "unknown" if ratio is None else f"{ratio:.1%}"
        lines.append(
            f"[quota-plan] {name}: {shown_peak} / "
            f"{evidence['configured_limit']:g} ({shown_ratio}); "
            f"warning at {evidence['warning_ratio']:.1%}")
    for name, evidence in (plan.get("hard_limits") or {}).items():
        planned = evidence.get("planned_max")
        configured = evidence.get("configured_limit")
        shown = "unknown" if planned is None else f"{planned:,}"
        lines.append(
            f"[quota-plan] {name}: {shown} / {configured:,} bytes; "
            "hard per-request ceiling (exact bytes rechecked before POST)")
    if isinstance(binding, dict):
        lines.append(
            "[quota-plan] endpoint binding: "
            + ("VERIFIED" if binding.get("binding_complete") else "REFUSED")
            + f"; configured={binding.get('configured_model')}; "
            f"observed={binding.get('observed_endpoint_name')}")
        lines.append(
            "[quota-plan] workspace tier "
            f"{binding.get('configured_workspace_tier')!r} is a configured "
            "assertion, not verified by endpoint metadata")
    for reason in plan["refusal_reasons"]:
        lines.append(f"[quota-plan] STOP: {reason}")
    lines.append(
        "[quota-plan] unrelated workspace traffic is not visible; passing "
        "this gate is not a provider-capacity claim")
    return "\n".join(lines)
