"""Best-effort capture of a Databricks serving endpoint's config.

A benchmark is only auditable if the report says what it ran against: the
GPU workload, provisioned size, and route. This reads the serving-endpoints
API for whatever endpoint name is in the run config, so it works with custom
endpoint names (no `databricks-` prefix assumed), and never breaks a run: any
failure returns None and the run proceeds without the metadata.

Databricks-specific by nature. Stdlib only.
"""
from __future__ import annotations

import http.client
import json
import ssl
import sys
import urllib.parse


def _note(msg: str) -> None:
    """Best-effort diagnostic. Metadata capture never fails a run, but a
    silent missing card is undebuggable, so say why on stderr."""
    print(f"[endpoint_meta] {msg}", file=sys.stderr)


def endpoint_name_from_path(path: str) -> str | None:
    """Pull the endpoint name out of `/serving-endpoints/<name>/invocations`.

    Works for any name, including a customer's custom one.
    """
    parts = [p for p in (path or "").split("/") if p]
    if "serving-endpoints" in parts:
        i = parts.index("serving-endpoints")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _summarize(doc: dict) -> dict:
    """Keep the customer-relevant fields, drop the noise."""
    # only the ACTIVE config served this run. pending_config carries the
    # new shape during an update, and naming it would describe capacity
    # that was never in the request path.
    cfg = doc.get("config") or {}
    entities = cfg.get("served_entities") or cfg.get("served_models") or []
    served = []
    for e in entities:
        # entity_name is the Unity Catalog three-level path. it identifies a
        # customer's catalog and schema, it adds nothing to "what was
        # measured", and this report is meant to be shared, so it is not kept.
        served.append({k: e.get(k) for k in (
            "name", "entity_version", "workload_type",
            "workload_size", "provisioned_model_units",
            "min_provisioned_throughput", "max_provisioned_throughput",
            "scale_to_zero_enabled") if e.get(k) is not None})
    return {
        "name": doc.get("name"),
        "task": doc.get("task"),
        "route_optimized": doc.get("route_optimized"),
        "ready": (doc.get("state") or {}).get("ready"),
        "served_entities": served,
        "note": "endpoint config read from the serving-endpoints API at run "
                "time, so the report states what was tested.",
    }


def fetch_endpoint_metadata(base_url: str, path: str, token: str | None,
                            timeout: float = 10.0) -> dict | None:
    """GET the serving endpoint config. Returns a compact summary, or None on
    any failure (missing name, no token, HTTP error, timeout, bad JSON)."""
    name = endpoint_name_from_path(path)
    if not name or not token:
        return None
    u = urllib.parse.urlparse(base_url)
    host = u.hostname
    if not host:
        return None
    port = u.port or (443 if (u.scheme or "https") == "https" else 80)
    api = f"/api/2.0/serving-endpoints/{urllib.parse.quote(name)}"
    conn = None
    try:
        if (u.scheme or "https") == "https":
            conn = http.client.HTTPSConnection(
                host, port, timeout=timeout,
                context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", api, headers={"Authorization": f"Bearer {token}"})
        resp = conn.getresponse()
        if resp.status != 200:
            _note(f"serving-endpoints API returned HTTP {resp.status} for "
                  f"'{name}', skipping the endpoint card")
            return None
        doc = json.loads(resp.read())
        return _summarize(doc)
    except Exception as exc:
        # never print the body or the token, only the failure class
        _note(f"could not read endpoint '{name}' ({type(exc).__name__}), "
              f"skipping the endpoint card")
        return None
    finally:
        if conn is not None:
            conn.close()
