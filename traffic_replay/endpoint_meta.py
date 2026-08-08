"""Best-effort capture of a Databricks serving endpoint's config.

A benchmark is only auditable if the report says what it ran against: the
GPU workload, provisioned size, and route. This reads the serving-endpoints
API for whatever endpoint name is in the run config, so it works with custom
endpoint names (no `databricks-` prefix assumed). The capture function is
best effort: any failure returns None. Ordinary runs proceed without the
metadata; quota-aware pay-per-token runs deliberately fail closed before
inference because they require this binding evidence.

Databricks-specific by nature. Stdlib only.
"""
from __future__ import annotations

import http.client
import math
import ssl
import sys
import urllib.parse

from .client import validate_bearer_transport
from .json_input import loads_strict


_MAX_RESPONSE_BYTES = 1024 * 1024
_PROVISIONED_ENTITY_FIELDS = frozenset({
    "workload_type",
    "workload_size",
    "provisioned_model_units",
    "min_provisioned_throughput",
    "max_provisioned_throughput",
})
_FOUNDATION_MODEL_PREFIX = "system.ai."


def _note(msg: str) -> None:
    """Best-effort diagnostic; say why capture returned no evidence."""
    print(f"[endpoint_meta] {msg}", file=sys.stderr)


def endpoint_name_from_path(path: str) -> str | None:
    """Pull the endpoint name out of `/serving-endpoints/<name>/invocations`.

    Works for any name, including a customer's custom one.  This parser is
    deliberately exact because quota-aware callers use a successful parse as
    evidence that inference and control-plane metadata refer to the same
    endpoint.  Query strings, fragments, alternate actions, repeated/trailing
    slashes, and extra path segments are therefore not accepted.
    """
    if not isinstance(path, str):
        return None
    # Direct Databricks invocation routes have no query contract.  Stripping a
    # query here would silently bind a different request target to the quota
    # snapshot, so fail closed even for an empty ``?`` suffix.
    if "?" in path or "#" in path:
        return None
    parts = path.split("/")
    if len(parts) == 4 and parts[0] == "" \
            and parts[1] == "serving-endpoints" \
            and parts[3] == "invocations":
        try:
            name = urllib.parse.unquote(parts[2], errors="strict")
        except (UnicodeDecodeError, ValueError):
            return None
        if name not in ("", ".", "..") and "/" not in name \
                and not any(char in name for char in ("\r", "\n", "\x00")):
            return name
    return None


def rate_limit_endpoint_binding(rate_limits: dict,
                                endpoint_metadata: dict | None,
                                endpoint_path: str | None = None) -> dict:
    """Bind a Databricks P2T quota snapshot to control-plane evidence.

    The serving-endpoints API does not expose the workspace product tier or
    workspace-wide quota counters.  This helper therefore verifies only what
    the captured endpoint document can prove: endpoint identity and the
    observed pay-per-token foundation-model entity shape.  The configured
    workspace tier remains an explicit assertion in the returned evidence.

    ``endpoint_metadata`` must be the compact value returned by
    :func:`fetch_endpoint_metadata`.  Missing or malformed evidence fails
    closed because a quota-aware run must not infer its deployment mode by
    sending paid inference traffic.
    """
    configured_model = rate_limits.get("model")
    configured_mode = rate_limits.get("deployment_mode")
    configured_route_name = (
        endpoint_name_from_path(endpoint_path)
        if endpoint_path is not None else None
    )
    reasons: list[str] = []

    if endpoint_path is not None and configured_route_name != configured_model:
        reasons.append(
            "request route endpoint does not match rate_limits.model")

    metadata_is_object = isinstance(endpoint_metadata, dict)
    observed_name = (
        endpoint_metadata.get("name") if metadata_is_object else None)
    endpoint_model_verified = bool(
        metadata_is_object and isinstance(observed_name, str)
        and observed_name == configured_model)
    if not metadata_is_object:
        reasons.append("serving endpoint metadata was not captured")
    elif not endpoint_model_verified:
        reasons.append(
            "captured endpoint name does not match rate_limits.model")
    observed_ready = (
        endpoint_metadata.get("ready") if metadata_is_object else None)
    endpoint_ready_verified = observed_ready == "READY"
    if metadata_is_object and not endpoint_ready_verified:
        reasons.append(
            "captured endpoint state is not exact READY")
    observed_route_optimized = (
        endpoint_metadata.get("route_optimized")
        if metadata_is_object else None)
    route_mode_verified = observed_route_optimized is False
    if metadata_is_object and not route_mode_verified:
        reasons.append(
            "captured endpoint route_optimized state is missing or not false")

    served_entities = (
        endpoint_metadata.get("served_entities")
        if metadata_is_object else None)
    entities_are_nonempty = bool(
        isinstance(served_entities, list) and served_entities)
    entity_names_verified = bool(
        entities_are_nonempty
        and all(isinstance(entity, dict)
                and entity.get("name") == configured_model
                for entity in served_entities))
    if metadata_is_object and not entities_are_nonempty:
        reasons.append(
            "captured endpoint metadata has no served entity evidence")
    elif metadata_is_object and not entity_names_verified:
        reasons.append(
            "captured served entity does not match rate_limits.model")

    expected_foundation_model_name = (
        _FOUNDATION_MODEL_PREFIX + configured_model
        if isinstance(configured_model, str) and configured_model else None)
    observed_foundation_model_names: list[str] = []
    missing_foundation_model_names = 0
    inspected_entities = (
        served_entities if isinstance(served_entities, list) else [])
    for entity in inspected_entities:
        foundation_model = (
            entity.get("foundation_model") if isinstance(entity, dict) else None)
        if not isinstance(foundation_model, dict) \
                or not isinstance(foundation_model.get("name"), str) \
                or not foundation_model["name"]:
            missing_foundation_model_names += 1
            continue
        observed_foundation_model_names.append(foundation_model["name"])
    foundation_model_names_verified = bool(
        entities_are_nonempty
        and expected_foundation_model_name is not None
        and not missing_foundation_model_names
        and len(observed_foundation_model_names) == len(served_entities)
        and all(name == expected_foundation_model_name
                for name in observed_foundation_model_names))
    if metadata_is_object and entities_are_nonempty:
        if missing_foundation_model_names:
            reasons.append(
                "captured served entity is missing foundation_model.name "
                "evidence")
        unexpected_foundation_models = sorted({
            str(name) for name in observed_foundation_model_names
            if name != expected_foundation_model_name
        })
        if unexpected_foundation_models:
            reasons.append(
                "captured foundation_model.name does not match expected "
                f"{expected_foundation_model_name}: "
                + ", ".join(unexpected_foundation_models))

    provisioned_fields = sorted({
        key
        for entity in (served_entities if isinstance(served_entities, list)
                       else [])
        if isinstance(entity, dict)
        for key in _PROVISIONED_ENTITY_FIELDS
        if key in entity
    })
    if provisioned_fields:
        reasons.append(
            "captured endpoint has provisioned-throughput entity fields: "
            + ", ".join(provisioned_fields))
    if configured_mode != "pay_per_token":
        reasons.append(
            "configured deployment mode is not pay_per_token")

    p2t_shape = bool(
        endpoint_model_verified
        and endpoint_ready_verified
        and route_mode_verified
        and entity_names_verified
        and foundation_model_names_verified
        and not provisioned_fields
        and configured_mode == "pay_per_token"
    )
    binding_complete = bool(
        p2t_shape
        and (endpoint_path is None
             or configured_route_name == configured_model)
    )
    # Preserve insertion order while suppressing duplicate diagnostics.
    reasons = list(dict.fromkeys(reasons))
    return {
        "status": "verified" if binding_complete else "refused",
        "configured_provider": rate_limits.get("provider"),
        "configured_model": configured_model,
        "configured_deployment_mode": configured_mode,
        "configured_workspace_tier": rate_limits.get("workspace_tier"),
        "workspace_tier_is_configured_assertion": True,
        "workspace_tier_verified": False,
        "configured_route_endpoint_name": configured_route_name,
        "observed_endpoint_name": observed_name,
        "endpoint_metadata_captured": metadata_is_object,
        "endpoint_model_verified": endpoint_model_verified,
        "observed_ready": observed_ready,
        "endpoint_ready_verified": endpoint_ready_verified,
        "expected_route_optimized": False,
        "observed_route_optimized": observed_route_optimized,
        "route_mode_verified": route_mode_verified,
        "served_entity_names_verified": entity_names_verified,
        "expected_foundation_model_name": expected_foundation_model_name,
        "observed_foundation_model_names": observed_foundation_model_names,
        "foundation_model_names_verified": foundation_model_names_verified,
        "provisioned_entity_fields_observed": provisioned_fields,
        "deployment_mode_evidence": (
            "every active served entity positively identified the expected "
            "system.ai foundation model and exposed no provisioned-throughput "
            "entity fields" if p2t_shape else None),
        "deployment_mode_verified": p2t_shape,
        "binding_complete": binding_complete,
        "reasons": reasons,
        "note": (
            "the expected provider identity is derived exactly as "
            "system.ai.<rate_limits.model>; this accounting binding also "
            "requires the captured direct workspace route to report "
            "route_optimized=false. workspace tier remains a configured "
            "assertion; confirm it and the workspace-wide quota counters in "
            "provider telemetry"),
    }


def _summarize(doc: dict) -> dict:
    """Keep the customer-relevant fields, drop the noise."""
    if not isinstance(doc, dict):
        raise ValueError("endpoint metadata response must be an object")
    # only the ACTIVE config served this run. pending_config carries the
    # new shape during an update, and naming it would describe capacity
    # that was never in the request path.
    cfg = doc.get("config")
    if cfg is None:
        cfg = {}
    elif not isinstance(cfg, dict):
        raise ValueError("endpoint metadata config must be an object")
    entities = cfg.get("served_entities")
    if entities is None:
        entities = cfg.get("served_models")
    if entities is None:
        entities = []
    if not isinstance(entities, list):
        raise ValueError("endpoint metadata served entities must be a list")
    served = []
    for e in entities:
        if not isinstance(e, dict):
            raise ValueError("endpoint metadata served entity must be an object")
        # entity_name is the Unity Catalog three-level path. it identifies a
        # customer's catalog and schema, it adds nothing to "what was
        # measured", and this report is meant to be shared, so it is not kept.
        compact = {k: e.get(k) for k in (
            "name", "entity_version", "workload_type",
            "workload_size", "provisioned_model_units",
            "min_provisioned_throughput", "max_provisioned_throughput",
            "scale_to_zero_enabled") if e.get(k) is not None}
        foundation_model = e.get("foundation_model")
        if foundation_model is not None:
            if not isinstance(foundation_model, dict):
                raise ValueError(
                    "endpoint metadata foundation_model must be an object")
            compact["foundation_model"] = {
                key: foundation_model.get(key)
                for key in ("name", "version")
                if foundation_model.get(key) is not None
            }
        served.append(compact)
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
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
            or not math.isfinite(float(timeout)) or timeout <= 0:
        return None
    try:
        scheme, host, port = validate_bearer_transport(base_url)
    except ValueError as exc:
        _note(f"unsafe or invalid endpoint origin ({type(exc).__name__}), "
              "skipping the endpoint card")
        return None
    api = ("/api/2.0/serving-endpoints/"
           f"{urllib.parse.quote(name, safe='')}")
    conn = None
    try:
        if scheme == "https":
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
        length = resp.getheader("Content-Length")
        if length is not None:
            try:
                if int(length) > _MAX_RESPONSE_BYTES:
                    _note(f"serving-endpoints API response for '{name}' was "
                          "too large, skipping the endpoint card")
                    return None
            except ValueError:
                pass
        raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            _note(f"serving-endpoints API response for '{name}' was too "
                  "large, skipping the endpoint card")
            return None
        doc = loads_strict(raw)
        return _summarize(doc)
    except Exception as exc:
        # never print the body or the token, only the failure class
        _note(f"could not read endpoint '{name}' ({type(exc).__name__}), "
              f"skipping the endpoint card")
        return None
    finally:
        if conn is not None:
            conn.close()
