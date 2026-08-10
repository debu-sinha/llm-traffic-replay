"""Runtime quota admission must fail closed before every physical POST."""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_CEILING

import pytest

from traffic_replay.quota_planner import (
    _CHAT_FRAMING_TOKEN_ALLOWANCE,
    _runtime_integer_budget,
    RuntimeQuotaGuard,
    RuntimeQuotaGuardError,
)


class _Clock:
    def __init__(self):
        self.ns = 1_000_000_000
        self.wall = 1_700_000_000.0

    def monotonic_ns(self):
        return self.ns

    def time(self):
        return self.wall

    def advance_ns(self, value: int):
        self.ns += value
        self.wall += value / 1_000_000_000

    def advance(self, seconds: float):
        self.advance_ns(int(seconds * 1_000_000_000))


def _limits(**overrides):
    value = {
        "input_tokens_per_minute": 1_000_000,
        "output_tokens_per_minute": 1_000_000,
        "queries_per_hour": 100_000,
        "warning_utilization": 1.0,
    }
    for name, item in overrides.items():
        if item is None:
            value.pop(name, None)
        else:
            value[name] = item
    return value


def _guard(limits=None, *, clock=None, **kwargs):
    clock = clock or _Clock()
    return RuntimeQuotaGuard(
        limits or _limits(), clock_ns=clock.monotonic_ns,
        wall_clock=clock.time, **kwargs)


def _reserve(guard, request_id="r", *, body=b"{}", messages=1,
             max_tokens=1, attempt=1, trigger=None):
    return guard.reserve(
        body, messages, max_tokens, request_id, attempt, trigger)


def _commit(guard, handle, reason="response_headers"):
    guard.mark_post_may_have_started(handle)
    return guard.commit(handle, reason=reason)


def test_integer_budget_is_the_largest_integer_strictly_below_threshold():
    for limit in range(1, 101):
        for numerator in range(1, 11):
            warning = Decimal(numerator) / Decimal(10)
            threshold = Decimal(limit) * warning
            budget = _runtime_integer_budget(Decimal(limit), warning)
            assert budget == max(
                int(threshold.to_integral_value(rounding=ROUND_CEILING)) - 1,
                0)
            assert budget < threshold
            assert budget + 1 >= threshold


def test_contact_with_warning_threshold_denies_and_permanently_trips():
    clock = _Clock()
    guard = _guard(_limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=10,
        warning_utilization=0.8), clock=clock)

    admitted = []
    for index in range(7):
        handle = _reserve(guard, f"ok-{index}")
        admitted.append(handle)
        assert handle.event["decision"] == "admitted"
        _commit(guard, handle)

    denied = _reserve(guard, "threshold")
    assert denied.event["decision"] == "denied"
    assert denied.event["reason_code"] == \
        "warning_budget_would_be_reached"
    assert denied.event["denied_dimensions"] == ["queries_per_hour"]
    assert guard.tripped is True

    clock.advance(7200)
    still_denied = _reserve(guard, "after-window")
    assert still_denied.event["decision"] == "denied"
    assert still_denied.event["reason_code"] == "guard_already_tripped"
    snap = guard.snapshot()
    assert snap["denied_attempts"] == 2
    assert snap["counts"]["denied"] == 2


def test_all_dimensions_admit_atomically():
    guard = _guard(_limits(
        input_tokens_per_minute=10_000,
        output_tokens_per_minute=3,
        queries_per_hour=100))
    first = _reserve(guard, "first", body=b"one", messages=1, max_tokens=2)
    _commit(guard, first)
    before = guard.snapshot()

    denied = _reserve(
        guard, "denied", body=b"a much larger second body", messages=2,
        max_tokens=1)
    assert denied.event["denied_dimensions"] == [
        "output_tokens_per_minute"]
    after = guard.snapshot()
    assert after["dimensions"]["queries_per_hour"]["active_total"] == 1
    assert after["dimensions"]["input_tokens_per_minute"][
        "active_total"] == before["dimensions"][
            "input_tokens_per_minute"]["active_total"]


def test_input_reservation_uses_exact_body_bytes_and_message_framing():
    body = '{"text":"é雪"}'.encode("utf-8")
    guard = _guard()
    handle = _reserve(
        guard, "utf8", body=body, messages=2, max_tokens=17,
        trigger="auth_token_refreshed", attempt=2)

    assert handle.event["reservation"] == {
        "request_bytes": len(body),
        "input_tokens": len(body) + _CHAT_FRAMING_TOKEN_ALLOWANCE * 3,
        "output_tokens": 17,
        "queries": 1,
    }
    assert handle.event["retry_trigger"] == "auth_token_refreshed"
    assert handle.event["attempt_ordinal"] == 2
    with pytest.raises(ValueError, match="body must be bytes"):
        _reserve(guard, "mutable", body=bytearray(body))


def test_provisional_reservation_never_ages_out_then_commit_owns_expiry():
    clock = _Clock()
    guard = _guard(_limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=2), clock=clock)
    handle = _reserve(guard, "slow-upload")

    clock.advance(7200)
    provisional = guard.snapshot()
    assert provisional["dimensions"]["queries_per_hour"][
        "active_provisional"] == 1
    assert provisional["dimensions"]["queries_per_hour"][
        "active_committed"] == 0

    guard.mark_post_may_have_started(handle)
    guard.commit(handle, reason="response_headers")
    clock.advance(3600)
    exact_boundary = guard.snapshot()
    assert exact_boundary["dimensions"]["queries_per_hour"][
        "active_committed"] == 1

    clock.advance_ns(1)
    expired = guard.snapshot()
    assert expired["dimensions"]["queries_per_hour"][
        "active_committed"] == 0


def test_cancel_releases_only_a_proven_unsent_provisional_reservation():
    guard = _guard()
    handle = _reserve(guard, "cancel")
    event = guard.cancel_before_post(handle, reason="operator_cancelled")
    assert event["state"] == "cancelled_before_post"
    assert event["post_may_have_started"] is False
    assert guard.snapshot()["dimensions"]["queries_per_hour"][
        "active_total"] == 0

    ambiguous = _reserve(guard, "ambiguous")
    guard.mark_post_may_have_started(ambiguous)
    with pytest.raises(ValueError, match="commit it conservatively"):
        guard.cancel_before_post(ambiguous)
    guard.commit(ambiguous, reason="transport_failure")
    assert guard.snapshot()["dimensions"]["queries_per_hour"][
        "active_total"] == 1


def test_commit_requires_explicit_post_start_mark_and_is_idempotent():
    guard = _guard()
    handle = _reserve(guard, "ordered")
    with pytest.raises(ValueError, match="before the POST may have started"):
        guard.commit(handle)
    first_mark = guard.mark_post_may_have_started(handle)
    second_mark = guard.mark_post_may_have_started(handle)
    assert second_mark is first_mark
    first_commit = guard.commit(handle, reason="headers")
    second_commit = guard.commit(handle, reason="ignored-idempotent-call")
    assert second_commit is first_commit
    assert first_commit["transition_reason"] == "headers"


def test_concurrent_admission_never_overshoots_and_trip_is_atomic():
    guard = _guard(_limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=10))
    events = []
    events_lock = threading.Lock()

    def worker(index):
        handle = _reserve(guard, f"concurrent-{index}")
        if handle.event["decision"] == "admitted":
            _commit(guard, handle)
        with events_lock:
            events.append(copy.deepcopy(handle.event))

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(worker, range(128)))

    assert sum(event["decision"] == "admitted" for event in events) == 9
    assert sum(event["decision"] == "denied" for event in events) == 119
    snap = guard.snapshot()
    assert snap["dimensions"]["queries_per_hour"]["active_total"] == 9
    assert snap["provisional_reservations"] == 0
    assert snap["sequence"] == 128
    assert snap["tripped"] is True


def test_guard_never_sleeps(monkeypatch):
    def forbidden_sleep(_seconds):
        raise AssertionError("runtime admission must never sleep")

    monkeypatch.setattr(time, "sleep", forbidden_sleep)
    guard = _guard()
    handle = _reserve(guard, "no-sleep")
    _commit(guard, handle)
    assert handle.event["state"] == "committed"


def test_shard_partition_is_deterministic_and_sums_to_global_safe_budget():
    limits = _limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=7200,
        warning_utilization=0.8)
    guards = [
        _guard(limits, shard_index=index, shard_total=3)
        for index in range(3)
    ]
    snapshots = [guard.snapshot() for guard in guards]
    allocations = [
        snap["dimensions"]["queries_per_hour"]["local_max_integer"]
        for snap in snapshots]

    assert allocations == [1920, 1920, 1919]
    assert sum(allocations) == 5759
    assert {guard.scope_id for guard in guards}.__len__() == 1
    assert len({guard.guard_id for guard in guards}) == 3
    assert guards[1].matches(limits, 1, 3)
    assert not guards[1].matches(limits, 0, 3)
    assert not guards[1].matches({**limits, "warning_utilization": 0.7}, 1, 3)


def test_match_can_bind_the_guard_to_exact_canonical_scope_material():
    limits = _limits()
    endpoint_a = {
        "command": "benchmark",
        "endpoint_base_url": "https://a.cloud.databricks.com",
        "endpoint_path": "/serving-endpoints/glm/invocations",
    }
    endpoint_b = {**endpoint_a,
                  "endpoint_base_url": "https://b.cloud.databricks.com"}
    guard = _guard(limits, scope_material=endpoint_a)

    assert guard.matches(limits, 0, 1, scope_material=endpoint_a)
    assert not guard.matches(limits, 0, 1, scope_material=endpoint_b)
    # Omission preserves compatibility for callers that did not bind a
    # command scope. Command paths with endpoint identity pass it explicitly.
    assert guard.matches(limits, 0, 1)
    assert not guard.matches(limits, False, 1, scope_material=endpoint_a)


def test_scope_material_binds_accounting_scope_and_freshness_contract():
    from traffic_replay.quota_planner import runtime_quota_scope_material

    endpoint = {
        "base_url": "https://a.cloud.databricks.com",
        "path": "/serving-endpoints/glm/invocations",
    }
    first = {
        **_limits(),
        "scope": "workspace A standard pay-per-token",
        "max_age_days": 7,
    }
    changed_scope = {**first, "scope": "workspace B standard pay-per-token"}
    changed_freshness = {**first, "max_age_days": 30}
    guard = _guard(
        first, scope_material=runtime_quota_scope_material(first, endpoint))

    assert guard.matches(
        first, 0, 1,
        scope_material=runtime_quota_scope_material(first, endpoint))
    assert not guard.matches(
        changed_scope, 0, 1,
        scope_material=runtime_quota_scope_material(
            changed_scope, endpoint))
    assert not guard.matches(
        changed_freshness, 0, 1,
        scope_material=runtime_quota_scope_material(
            changed_freshness, endpoint))


def test_event_and_snapshot_are_finite_json_without_opaque_handle_token():
    guard = _guard()
    handle = _reserve(guard, "json")
    event = _commit(guard, handle)
    encoded_event = json.dumps(event, allow_nan=False, sort_keys=True)
    encoded_snapshot = json.dumps(
        guard.snapshot(), allow_nan=False, sort_keys=True)

    def all_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from all_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from all_keys(item)

    assert "_token" not in set(all_keys(event))
    assert guard.guard_id in encoded_event
    assert guard.scope_id in encoded_snapshot
    assert event["state"] == "committed"
    assert event["post_may_have_started"] is True
    assert event["committed_at_elapsed_ns"] is not None


def test_request_bytes_hard_limit_is_exact_inclusive_and_shard_independent():
    limits = _limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=None,
        request_bytes_max=4.9)
    first = _guard(limits, shard_index=0, shard_total=3)
    last = _guard(limits, shard_index=2, shard_total=3)

    admitted = _reserve(first, "exact", body=b"1234")
    assert admitted.event["decision"] == "admitted"
    assert admitted.event["reservation"]["request_bytes"] == 4
    assert admitted.event["hard_limit_checks"]["request_bytes_max"] == {
        "reservation": 4,
        "configured_limit": "4.9",
        "maximum_integer": 4,
        "measurement": "exact_serialized_request_body_bytes",
        "comparison": "less_than_or_equal",
        "allocation": "shard_independent_per_post",
        "exceeded": False,
    }

    denied = _reserve(last, "oversize", body=b"12345")
    assert denied.event["decision"] == "denied"
    assert denied.event["reason_code"] == "hard_per_request_limit_exceeded"
    assert denied.event["denied_dimensions"] == ["request_bytes_max"]
    assert denied.event["hard_limit_checks"]["request_bytes_max"][
        "exceeded"] is True
    assert first.snapshot()["hard_limits"] == last.snapshot()["hard_limits"]
    assert first.scope_id == last.scope_id


def test_request_bytes_limit_can_be_the_only_configured_dimension():
    guard = _guard({"request_bytes_max": 2, "warning_utilization": 0.1})
    assert _reserve(guard, "two", body=b"12").event[
        "decision"] == "admitted"
    assert guard.snapshot()["dimensions"] == {}
    assert guard.snapshot()["hard_limits"]["request_bytes_max"][
        "maximum_integer"] == 2


def test_queries_per_second_uses_a_conservative_exact_one_second_window():
    clock = _Clock()
    guard = _guard(_limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=None,
        queries_per_second=3), clock=clock)
    first = _reserve(guard, "qps-first")
    _commit(guard, first)

    clock.advance(1)
    second = _reserve(guard, "qps-boundary")
    assert second.event["decision"] == "admitted"
    assert second.event["projected"]["queries_per_second"][
        "active_before"] == 1
    _commit(guard, second)

    separate_clock = _Clock()
    separate = _guard(_limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=None,
        queries_per_second=3), clock=separate_clock)
    committed = _reserve(separate, "qps-expiry")
    _commit(separate, committed)
    separate_clock.advance_ns(1_000_000_001)
    assert separate.snapshot()["dimensions"]["queries_per_second"][
        "active_committed"] == 0


def test_prior_row_seeding_packs_at_now_and_deduplicates_exact_events():
    limits = _limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=100)
    source = _guard(limits)
    event_handle = _reserve(source, "preflight")
    _commit(source, event_handle)
    row = {
        "request_attempts": 1,
        "quota_guard_events": [copy.deepcopy(event_handle.event)],
    }
    target = _guard(limits)

    first = target.seed_prior_rows([row])
    second = target.seed_prior_rows([row])
    assert first == {
        "rows": 1, "imported": 1, "deduplicated": 0,
        "nonconsuming": 0, "tripped": False}
    assert second == {
        "rows": 1, "imported": 0, "deduplicated": 1,
        "nonconsuming": 0, "tripped": False}
    assert target.snapshot()["dimensions"]["queries_per_hour"][
        "active_committed"] == 1

    own = source.seed_prior_rows([row])
    assert own["deduplicated"] == 1
    with pytest.raises(ValueError, match="disagree"):
        target.seed_prior_rows([{"request_attempts": 1}])

    duplicate = {
        "request_attempts": 2,
        "quota_guard_events": [
            copy.deepcopy(event_handle.event),
            copy.deepcopy(event_handle.event),
        ],
    }
    with pytest.raises(ValueError, match="repeats committed"):
        target.seed_prior_rows([duplicate])


def test_prior_hard_limit_evidence_must_be_exact_and_consistent():
    limits = _limits(request_bytes_max=1000)
    source = _guard(limits)
    handle = _reserve(source, "prior-hard", body=b"abc")
    _commit(source, handle)
    corrupted = copy.deepcopy(handle.event)
    corrupted["hard_limit_checks"]["request_bytes_max"][
        "reservation"] += 1

    target = _guard(limits)
    with pytest.raises(ValueError, match="hard-limit evidence"):
        target.seed_prior_events([corrupted])
    assert target.snapshot()["counts"]["seeded_committed"] == 0


def test_nonterminal_or_wrong_scope_prior_evidence_is_rejected_without_mutation():
    guard = _guard()
    source = _guard()
    provisional = _reserve(source, "still-running")
    before = guard.snapshot()

    with pytest.raises(ValueError, match="not terminal"):
        guard.seed_prior_events([provisional.event])
    wrong_scope = copy.deepcopy(provisional.event)
    wrong_scope["state"] = "committed"
    wrong_scope["post_may_have_started"] = True
    wrong_scope["scope_id"] = "quota-scope-wrong"
    with pytest.raises(ValueError, match="different scope"):
        guard.seed_prior_events([wrong_scope])
    after = guard.snapshot()
    assert after["dimensions"] == before["dimensions"]
    assert after["counts"] == before["counts"]


def test_seeded_usage_above_local_budget_trips_before_new_admission():
    limits = _limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=2)
    source = _guard(limits)
    handle = _reserve(source, "prior")
    _commit(source, handle)
    first = copy.deepcopy(handle.event)
    second = copy.deepcopy(handle.event)
    second["guard_id"] = "quota-guard-another-prior"
    second["sequence"] = 2
    target = _guard(limits)

    seeded = target.seed_prior_events([first, second])
    assert seeded["imported"] == 2
    assert seeded["tripped"] is True
    denied = _reserve(target, "after-seed")
    assert denied.event["decision"] == "denied"
    assert denied.event["reason_code"] == "guard_already_tripped"
    assert target.snapshot()["trip"]["reason_code"] == \
        "seeded_prior_usage_exceeds_warning_budget"


def test_seeded_terminal_denial_preserves_the_command_safety_stop():
    limits = _limits(
        input_tokens_per_minute=None,
        output_tokens_per_minute=None,
        queries_per_hour=None,
        request_bytes_max=3)
    source = _guard(limits)
    denied = _reserve(source, "oversize-preflight", body=b"1234")
    assert denied.event["decision"] == "denied"
    row = {
        "request_attempts": 0,
        "quota_guard_events": [copy.deepcopy(denied.event)],
    }

    reconstructed = _guard(limits)
    seeded = reconstructed.seed_prior_rows([row])

    assert seeded["tripped"] is True
    assert reconstructed.snapshot()["trip"]["reason_code"] == \
        "hard_per_request_limit_exceeded"
    later = _reserve(reconstructed, "small-replay", body=b"12")
    assert later.event["decision"] == "denied"
    assert later.event["reason_code"] == "guard_already_tripped"


def test_backward_monotonic_latches_internal_error_and_snapshot_stays_safe():
    clock = _Clock()
    guard = _guard(clock=clock)
    clock.ns -= 1

    denied = _reserve(guard, "clock-regressed")
    assert denied.event["decision"] == "denied"
    assert denied.event["reason_code"] == "guard_internal_error"
    assert guard.tripped is True
    snap = guard.snapshot()
    assert snap["tripped"] is True
    assert snap["trip"]["reason_code"] == "guard_internal_error"
    assert snap["trip"]["prior_event"] == {
        "operation": "reserve", "error_type": "RuntimeError"}
    assert snap["clock_status"] == "not_rechecked_after_permanent_trip"
    assert json.dumps(snap, allow_nan=False)

    again = _reserve(guard, "already-tripped")
    assert again.event["reason_code"] == "guard_already_tripped"


def test_invalid_wall_clock_latches_and_never_rechecks_for_later_denials():
    clock = _Clock()
    guard = _guard(clock=clock)
    clock.wall = math.nan

    denied = _reserve(guard, "bad-wall")
    assert denied.event["reason_code"] == "guard_internal_error"
    # A tripped guard uses the last valid evidence timestamps and therefore does
    # not need another read from the still-invalid clock.
    again = _reserve(guard, "no-recheck")
    assert again.event["reason_code"] == "guard_already_tripped"
    assert math.isfinite(again.event["reserved_at_unix"])
    assert guard.snapshot()["denied_attempts"] == 2


def test_clock_failure_during_transition_trips_and_keeps_reservation_provisional():
    clock = _Clock()
    guard = _guard(clock=clock)
    handle = _reserve(guard, "transition")
    clock.ns -= 1

    with pytest.raises(RuntimeQuotaGuardError, match="mark_post"):
        guard.mark_post_may_have_started(handle)
    snap = guard.snapshot()
    assert snap["tripped"] is True
    assert snap["provisional_reservations"] == 1
    assert snap["dimensions"]["queries_per_hour"][
        "active_provisional"] == 1


def test_foreign_transition_cannot_corrupt_or_trip_a_live_guard():
    first = _guard()
    second = _guard()
    handle = _reserve(first, "foreign")
    before = second.snapshot()

    with pytest.raises(ValueError, match="another guard"):
        second.mark_post_may_have_started(handle)
    after = second.snapshot()
    assert after["tripped"] is False
    assert after["counts"] == before["counts"]
    assert after["dimensions"] == before["dimensions"]
