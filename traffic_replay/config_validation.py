"""Strict validation for numeric policy configuration.

Acceptance and pricing values directly decide pass/fail and cost. Treating a
typo, NaN, Boolean, or negative rate as ordinary JSON can silently turn a
scorecard green or emit non-standard artifacts, so validation is centralized
and deliberately rejects unknown keys.
"""
from __future__ import annotations

import math
from typing import Any


_QUANTILES = {"p50", "p90", "p95", "p99"}


def _number(value: Any, where: str, *, positive: bool = False,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{where} must be greater than zero")
    if not positive and number < 0:
        raise ValueError(f"{where} must be non-negative")
    if maximum is not None and number > maximum:
        raise ValueError(f"{where} must be at most {maximum:g}")
    return number


def _keys(value: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"{where} has unknown field{'s' if len(unknown) != 1 else ''}: "
            + ", ".join(unknown))


def _latency_targets(value: Any, where: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where} must be a non-empty object")
    _keys(value, _QUANTILES, where)
    for quantile, target in value.items():
        _number(target, f"{where}.{quantile}", positive=True)


def validate_acceptance_targets(value: Any, where: str =
                                "acceptance_targets") -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    allowed = {
        "ttft_ms", "ttfg_ms", "hard_timeouts", "success_rate",
        "interchunk_ms", "targets_are", "priority", "note",
    }
    _keys(value, allowed, where)
    for name in ("ttft_ms", "ttfg_ms"):
        if name in value:
            _latency_targets(value[name], f"{where}.{name}")
    if "hard_timeouts" in value:
        hard = value["hard_timeouts"]
        if not isinstance(hard, dict) or not hard:
            raise ValueError(f"{where}.hard_timeouts must be a non-empty object")
        _keys(hard, {"ttft_s", "ttfg_s", "note"},
              f"{where}.hard_timeouts")
        limits = {name: limit for name, limit in hard.items()
                  if name != "note"}
        if not limits:
            raise ValueError(
                f"{where}.hard_timeouts needs ttft_s or ttfg_s")
        for name, limit in limits.items():
            _number(limit, f"{where}.hard_timeouts.{name}", positive=True)
        if "note" in hard and not isinstance(hard["note"], str):
            raise ValueError(f"{where}.hard_timeouts.note must be a string")
    if "success_rate" in value:
        _number(value["success_rate"], f"{where}.success_rate",
                positive=True, maximum=1.0)
    if "interchunk_ms" in value:
        _number(value["interchunk_ms"], f"{where}.interchunk_ms",
                positive=True)
    for name in ("targets_are", "priority", "note"):
        if name in value and not isinstance(value[name], str):
            raise ValueError(f"{where}.{name} must be a string")


def validate_pricing(value: Any, where: str = "pricing") -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    mode = value.get("mode")
    if mode not in {"per_token", "provisioned"}:
        raise ValueError(
            f"{where}.mode must be 'per_token' or 'provisioned'")
    common = {"mode", "usd_per_dbu"}
    if mode == "per_token":
        required = {"input_dbu_per_m", "output_dbu_per_m"}
        allowed = common | required | {"cache_read_dbu_per_m"}
    else:
        required = {"dbu_per_hour"}
        allowed = common | required
    _keys(value, allowed, where)
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"{where} is missing required field"
            f"{'s' if len(missing) != 1 else ''}: {', '.join(missing)}")
    for name, amount in value.items():
        if name != "mode":
            _number(amount, f"{where}.{name}",
                    positive=(name == "dbu_per_hour"))
