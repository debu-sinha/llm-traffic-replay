"""Strict validation for numeric policy configuration.

Acceptance and pricing values directly decide pass/fail and cost. Treating a
typo, NaN, Boolean, or negative rate as ordinary JSON can silently turn a
scorecard green or emit non-standard artifacts, so validation is centralized
and deliberately rejects unknown keys.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any
from urllib.parse import urlsplit


_QUANTILES = {"p50", "p90", "p95", "p99"}


def _number(value: Any, where: str, *, positive: bool = False,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite number") from exc
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
        if float(value["success_rate"]) >= 1.0:
            raise ValueError(
                f"{where}.success_rate must be less than 1.0; a finite "
                "sample can demonstrate a high reliability target with a "
                "one-sided confidence bound, but can never prove a true "
                "100% success probability")
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


def validate_rate_limits(value: Any, where: str = "rate_limits") -> None:
    """Validate an as-of provider quota snapshot used for run safety.

    Rate limits change independently of the harness.  Requiring both a source
    and an observation date keeps a sealed run from presenting an unattributed
    number as a timeless provider fact.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    allowed = {
        "input_tokens_per_minute", "output_tokens_per_minute",
        "queries_per_hour", "queries_per_second", "request_bytes_max",
        "warning_utilization", "source", "as_of",
        "scope", "note", "provider", "deployment_mode", "workspace_tier",
        "model", "accounting_model", "verified_at", "max_age_days",
    }
    _keys(value, allowed, where)
    limits = {
        name: value[name]
        for name in ("input_tokens_per_minute", "output_tokens_per_minute",
                     "queries_per_hour", "queries_per_second",
                     "request_bytes_max")
        if name in value
    }
    if not limits:
        raise ValueError(
            f"{where} needs input_tokens_per_minute, "
            "output_tokens_per_minute, queries_per_hour, "
            "queries_per_second, or request_bytes_max")
    for name, limit in limits.items():
        _number(limit, f"{where}.{name}", positive=True)
    if "request_bytes_max" in value and (
            isinstance(value["request_bytes_max"], bool)
            or not isinstance(value["request_bytes_max"], int)):
        raise ValueError(
            f"{where}.request_bytes_max must be a positive integer byte "
            "count")
    if "warning_utilization" not in value:
        raise ValueError(f"{where}.warning_utilization is required")
    _number(value["warning_utilization"],
            f"{where}.warning_utilization", positive=True, maximum=1.0)
    for name in (
            "source", "as_of", "scope", "provider", "deployment_mode",
            "workspace_tier", "model", "accounting_model"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f"{where}.{name} must be a non-empty string")
    if value["provider"] != "databricks":
        raise ValueError(
            f"{where}.provider must be 'databricks'; other accounting "
            "models are not implemented")
    if value["deployment_mode"] != "pay_per_token":
        raise ValueError(
            f"{where}.deployment_mode must be 'pay_per_token' for token/QPH "
            "accounting; provisioned endpoints do not use these TPM limits")
    if value["accounting_model"] != "databricks_fmapi_pay_per_token":
        raise ValueError(
            f"{where}.accounting_model must be "
            "'databricks_fmapi_pay_per_token'")
    source_url = urlsplit(value["source"])
    if source_url.scheme != "https" or not source_url.netloc:
        raise ValueError(f"{where}.source must be an https URL")
    try:
        parsed = date.fromisoformat(value["as_of"])
    except ValueError as exc:
        raise ValueError(f"{where}.as_of must be YYYY-MM-DD") from exc
    if parsed.isoformat() != value["as_of"]:
        raise ValueError(f"{where}.as_of must be YYYY-MM-DD")
    if parsed > date.today():
        raise ValueError(f"{where}.as_of cannot be in the future")
    freshness_fields = {"verified_at", "max_age_days"}.intersection(value)
    if freshness_fields and freshness_fields != {"verified_at", "max_age_days"}:
        missing = ({"verified_at", "max_age_days"} - freshness_fields).pop()
        raise ValueError(
            f"{where}.{missing} is required when snapshot freshness is set")
    if freshness_fields:
        verified_at = value["verified_at"]
        if not isinstance(verified_at, str):
            raise ValueError(f"{where}.verified_at must be YYYY-MM-DD")
        try:
            verified_date = date.fromisoformat(verified_at)
        except ValueError as exc:
            raise ValueError(
                f"{where}.verified_at must be YYYY-MM-DD") from exc
        if verified_date.isoformat() != verified_at:
            raise ValueError(f"{where}.verified_at must be YYYY-MM-DD")
        if verified_date > date.today():
            raise ValueError(f"{where}.verified_at cannot be in the future")
        if verified_date < parsed:
            raise ValueError(
                f"{where}.verified_at cannot be earlier than as_of")
        max_age = value["max_age_days"]
        if isinstance(max_age, bool) or not isinstance(max_age, int) \
                or max_age <= 0:
            raise ValueError(
                f"{where}.max_age_days must be a positive integer")
    for name in ("note",):
        if name in value and (
                not isinstance(value[name], str) or not value[name].strip()):
            raise ValueError(f"{where}.{name} must be a non-empty string")
