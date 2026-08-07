"""Unambiguous JSON parsing for configuration and workload inputs."""
from __future__ import annotations

import json


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def loads_strict(value: str | bytes):
    """Parse JSON while rejecting duplicate object keys at every depth."""
    return json.loads(value, object_pairs_hook=_object_without_duplicates)
