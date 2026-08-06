"""Policy inputs that drive verdicts and costs fail closed."""
from __future__ import annotations

import math

import pytest

from traffic_replay.config_validation import (validate_acceptance_targets,
                                               validate_pricing)


def test_valid_acceptance_and_pricing_schemas():
    validate_acceptance_targets({
        "ttft_ms": {"p50": 500, "p95": 900.0},
        "ttfg_ms": {"p99": 2000},
        "hard_timeouts": {"ttft_s": 15, "ttfg_s": 45},
        "interchunk_ms": 500,
        "success_rate": 0.999,
        "targets_are": "customer SLO",
        "note": "measured in production",
    })
    validate_pricing({
        "mode": "per_token", "input_dbu_per_m": 20,
        "output_dbu_per_m": 62.857, "cache_read_dbu_per_m": 2,
        "usd_per_dbu": 0.07,
    })
    validate_pricing({
        "mode": "provisioned", "dbu_per_hour": 85.714,
        "usd_per_dbu": 0.07,
    })


@pytest.mark.parametrize("value", [
    {"ttft_ms": {"p101": 1}},
    {"ttft_ms": {"p95": -5}},
    {"ttfg_ms": {"p99": math.nan}},
    {"hard_timeouts": {"ttft_s": 0}},
    {"hard_timeouts": {"unknown": 1}},
    {"success_rate": -1},
    {"success_rate": 1.01},
    {"success_rate": True},
    {"interchunk_ms": math.inf},
    {"unknown": 1},
])
def test_invalid_acceptance_values_are_rejected(value):
    with pytest.raises(ValueError):
        validate_acceptance_targets(value)


@pytest.mark.parametrize("value", [
    {"mode": "per_tokne", "input_dbu_per_m": 1,
     "output_dbu_per_m": 1},
    {"mode": "per_token", "input_dbu_per_m": -1,
     "output_dbu_per_m": 1},
    {"mode": "per_token", "input_dbu_per_m": 1,
     "output_dbu_per_m": math.nan},
    {"mode": "per_token", "input_dbu_per_m": True,
     "output_dbu_per_m": 1},
    {"mode": "per_token", "input_dbu_per_m": 1},
    {"mode": "provisioned", "dbu_per_hour": 0},
    {"mode": "provisioned", "dbu_per_hour": 1, "extra": 2},
])
def test_invalid_pricing_values_are_rejected(value):
    with pytest.raises(ValueError):
        validate_pricing(value)
