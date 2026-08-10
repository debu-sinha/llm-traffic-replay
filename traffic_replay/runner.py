"""Run orchestration: schedule -> paced dispatch -> results.

Two input modes share the same dispatch and measurement path:
  profile mode  (profile_path): synthetic text generated to a statistical
                shape (sizes, cache structure).
  prompts mode  (prompts_file): the user's real prompts, replayed verbatim.

Pacing: open loop. Each request has an absolute scheduled time, and the
dispatcher thread sleeps until that timestamp and submits into a bounded
thread pool. It never waits for a response before firing the next request,
so a slow endpoint does not throttle the offered rate. That is the point: a
closed-loop generator quietly reduces load as the endpoint slows, and you
never find the knee.

Two different lateness numbers come out of this, and they answer different
questions. dispatch_lag_ms is stamped in the dispatcher just before the
submit, so it sees the dispatcher falling behind but NOT a saturated pool,
because ThreadPoolExecutor.submit() queues rather than blocking. HTTP
request-start lateness, computed from the exact monotonic clock immediately
before the first conn.request invocation, grows under either. It does not
observe upload completion or endpoint receipt; use it only to decide whether
the client began requests on schedule.

Warmup/calibration: the first `calibrate_n` requests run at low rate before
the schedule proper. In profile mode their endpoint-reported prompt_tokens
recalibrate the chars-per-token ratio used to build later request text; in
prompts mode the text is fixed, so the warmup only primes the endpoint.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
import stat
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import profile as prof
from .artifacts import (
    RunArtifacts,
    canonical_sha256,
    redact_secrets,
    sha256_bytes,
    snapshot_source_state,
)
from .adapters import (
    DEFAULT_ENDPOINT_ADAPTER,
    endpoint_adapter_contract,
    get_endpoint_adapter,
)
from .client import (EndpointClient, EndpointConfig, RequestResult,
                     normalized_origin, serialize_request_body,
                     validate_bearer_token, validate_bearer_transport)
from .config_validation import (validate_acceptance_targets,
                                validate_pricing, validate_rate_limits)
from .json_input import loads_strict
from .metrics import summarize, write_outputs
from .network import AbsoluteHTTPDeadline, bind_deadline_bounded_dns
from .prefix_pool import PrefixPool
from .schedule import (
    MAX_EXACT_ANALYSIS_REQUEST_ROWS,
    load_trace,
    make_schedule,
    schedule_report,
    shard,
    thin_schedule_ceiling,
    validate_exact_analysis_capacity,
    validate_schedule_capacity,
)
from .textgen import TextMaterializer, calibrate_cpt


_DEFAULT_MAX_CONCURRENCY = 256
_MAX_CONCURRENCY = 4096
_MAX_PENDING_REQUESTS = 100_000
_MAX_POOL_DOCS_PER_BUCKET = 10_000
_MAX_CALIBRATION_REQUESTS = 10_000
_MIN_SIZING_PROBES = 4
_MAX_SIZING_PROBES = 8
_AUTH_RESPONSE_MAX_BYTES = 64 * 1024
_AUTH_CREDENTIAL_MAX_BYTES = 8 * 1024
_AUTH_M2M_TIMEOUT_S = 15.0
_AUTH_CLI_TIMEOUT_S = 30.0
_AUTH_DISABLED_DEFAULT_SECTION = (
    "__traffic_replay_reserved_defaults_do_not_use__")
_CANCELLATION_DRAIN_TIMEOUT_S = 2.0

# Local workload files are intentionally snapshotted into memory exactly once
# before credentials or network access.  These byte/record bounds make that
# operation predictable and ensure sparse files, giant records, and hostile
# special files fail before allocation or blocking I/O.
_INPUT_LIMITS = {
    "profile": {
        "max_bytes": 16 * 1024 * 1024,
        "max_lines": 100_000,
        "max_line_bytes": 16 * 1024 * 1024,
    },
    "prompts": {
        "max_bytes": 64 * 1024 * 1024,
        "max_lines": MAX_EXACT_ANALYSIS_REQUEST_ROWS,
        "max_line_bytes": 4 * 1024 * 1024,
    },
    "prompts_json": {
        "max_bytes": 64 * 1024 * 1024,
        "max_lines": 100_000,
        "max_line_bytes": 64 * 1024 * 1024,
    },
    "timestamps": {
        "max_bytes": 16 * 1024 * 1024,
        "max_lines": MAX_EXACT_ANALYSIS_REQUEST_ROWS,
        "max_line_bytes": 64 * 1024,
    },
}
_MAX_PROMPT_RECORD_BYTES = 4 * 1024 * 1024


@dataclasses.dataclass
class RunConfig:
    endpoint: dict                    # EndpointConfig fields
    profile_path: str | None = None   # profile mode: synthetic text to a shape
    prompts_file: str | None = None   # prompts mode: replay real prompt text
    duration_s: int = 300
    qps_base: float = 25.0
    qps_burst: float = 350.0
    qps_min: float = 10.0
    qps_max: float = 500.0
    rate_scale: float = 1.0
    max_concurrency: int | None = None  # omission uses a 256-thread safety cap
    max_pending_requests: int | None = None  # running + queued client work
    sizing_concurrency: int | None = None
                                      # derives a FIXED open-loop arrival rate
                                      # from unloaded service time. It is a
                                      # sizing hint, not a held concurrency.
    concurrency: int | None = None    # legacy alias; normalized above at run
    seed: int = 7
    cpt: float = 4.0
    calibrate_n: int = 12
    shard_index: int = 0
    shard_total: int = 1
    run_id: str | None = None         # required/shared across multiple shards
    start_at_unix: float | None = None  # required/shared absolute replay epoch
    start_tolerance_s: float = 0.5    # refuse a stale synchronized start
    timestamps_file: str | None = None  # real arrival trace replaces synthetic
    pool_docs_per_bucket: int = 40      # cache-pool shape knobs (profile mode)
    pool_zipf_s: float = 1.1
    out_dir: str = "results"
    title: str = "traffic replay"
    label: str = ""
    max_output_tokens_cap: int = 512  # safety cap; full runs raise it
    acceptance_targets: dict | None = None  # SLA targets (either mode)
    pricing: dict | None = None              # DBU cost rates (see metrics)
    capture_endpoint_metadata: bool = True   # read serving-endpoint config
    measure_network_path: bool = True        # time the round trip to it
    ttft_definition: str = "first_content"   # or "first_visible"; sla scores it
    rate_limits: dict | None = None          # as-of quota snapshot + warning bar
    input_expectations: dict | None = None   # optional SHA-256/size replay guard

    def __post_init__(self) -> None:
        for name in ("profile_path", "prompts_file", "timestamps_file"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, (str, os.PathLike))
                    or not str(value).strip()):
                raise ValueError(f"{name} must be a non-empty path")
            if value is not None:
                setattr(self, name, str(value))
        if bool(self.profile_path) == bool(self.prompts_file):
            raise ValueError("set exactly one of profile_path or prompts_file")
        if not isinstance(self.endpoint, dict):
            raise ValueError("endpoint must be an object")
        try:
            endpoint_config = EndpointConfig(**self.endpoint)
        except TypeError as exc:
            raise ValueError(f"invalid endpoint configuration: {exc}") from exc
        if self.sizing_concurrency is not None and self.concurrency is not None:
            raise ValueError("set sizing_concurrency, not both it and legacy concurrency")
        if self.sizing_concurrency is None and self.concurrency is not None:
            self.sizing_concurrency = self.concurrency
            self.concurrency = None
        if self.sizing_concurrency is not None \
                and (not isinstance(self.sizing_concurrency, int)
                     or isinstance(self.sizing_concurrency, bool)
                     or self.sizing_concurrency <= 0):
            raise ValueError("sizing_concurrency must be a positive integer")
        if self.sizing_concurrency is not None and self.timestamps_file:
            raise ValueError(
                "sizing_concurrency cannot be combined with timestamps_file; "
                "the trace already defines the complete arrival schedule, so "
                "a derived fixed QPS would be ignored")
        if not isinstance(self.duration_s, int) \
                or isinstance(self.duration_s, bool) or self.duration_s <= 0:
            raise ValueError("duration_s must be a positive integer")
        rates = (self.qps_base, self.qps_burst, self.qps_min, self.qps_max)
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(float(x)) or float(x) <= 0 for x in rates):
            raise ValueError("qps_base/qps_burst/qps_min/qps_max must be positive and finite")
        if self.qps_min > self.qps_max:
            raise ValueError("qps_min cannot exceed qps_max")
        if not (self.qps_min <= self.qps_base <= self.qps_max):
            raise ValueError("qps_base must be between qps_min and qps_max")
        if not (self.qps_min <= self.qps_burst <= self.qps_max):
            raise ValueError("qps_burst must be between qps_min and qps_max")
        if isinstance(self.rate_scale, bool) \
                or not isinstance(self.rate_scale, (int, float)) \
                or not math.isfinite(float(self.rate_scale)) \
                or not (0 < self.rate_scale <= 1):
            raise ValueError("rate_scale must be in (0, 1]")
        if self.max_concurrency is None:
            if self.sizing_concurrency is None:
                self.max_concurrency = _DEFAULT_MAX_CONCURRENCY
        elif not isinstance(self.max_concurrency, int) \
                or isinstance(self.max_concurrency, bool) \
                or self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if self.max_concurrency is not None \
                and self.max_concurrency > _MAX_CONCURRENCY:
            raise ValueError(
                f"max_concurrency cannot exceed {_MAX_CONCURRENCY}; shard "
                "the load generator instead")
        if self.max_pending_requests is not None and (
                not isinstance(self.max_pending_requests, int)
                or isinstance(self.max_pending_requests, bool)
                or self.max_pending_requests <= 0):
            raise ValueError("max_pending_requests must be a positive integer")
        if self.max_pending_requests is not None \
                and self.max_pending_requests > _MAX_PENDING_REQUESTS:
            raise ValueError(
                f"max_pending_requests cannot exceed {_MAX_PENDING_REQUESTS}")
        if isinstance(self.cpt, bool) or not isinstance(self.cpt, (int, float)) \
                or not math.isfinite(float(self.cpt)) or self.cpt <= 0:
            raise ValueError("cpt must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.calibrate_n, int) \
                or isinstance(self.calibrate_n, bool) or self.calibrate_n < 0:
            raise ValueError("calibrate_n must be a non-negative integer")
        if self.calibrate_n > _MAX_CALIBRATION_REQUESTS:
            raise ValueError(
                f"calibrate_n cannot exceed {_MAX_CALIBRATION_REQUESTS}")
        if not isinstance(self.shard_total, int) \
                or isinstance(self.shard_total, bool) \
                or self.shard_total <= 0 \
                or not isinstance(self.shard_index, int) \
                or isinstance(self.shard_index, bool) \
                or not (0 <= self.shard_index < self.shard_total):
            raise ValueError("need 0 <= shard_index < shard_total")
        if not isinstance(self.pool_docs_per_bucket, int) \
                or isinstance(self.pool_docs_per_bucket, bool) \
                or self.pool_docs_per_bucket <= 0:
            raise ValueError("pool_docs_per_bucket must be a positive integer")
        if self.pool_docs_per_bucket > _MAX_POOL_DOCS_PER_BUCKET:
            raise ValueError(
                "pool_docs_per_bucket cannot exceed "
                f"{_MAX_POOL_DOCS_PER_BUCKET}")
        if isinstance(self.pool_zipf_s, bool) \
                or not isinstance(self.pool_zipf_s, (int, float)) \
                or not math.isfinite(float(self.pool_zipf_s)) \
                or self.pool_zipf_s <= 0:
            raise ValueError("pool_zipf_s must be positive and finite")
        if not isinstance(self.max_output_tokens_cap, int) \
                or isinstance(self.max_output_tokens_cap, bool) \
                or self.max_output_tokens_cap <= 0:
            raise ValueError("max_output_tokens_cap must be a positive integer")
        if self.ttft_definition not in ("first_content", "first_visible"):
            raise ValueError("ttft_definition must be first_content or first_visible")
        validate_acceptance_targets(self.acceptance_targets)
        validate_pricing(self.pricing)
        validate_rate_limits(self.rate_limits)
        if self.input_expectations is not None:
            if not isinstance(self.input_expectations, dict):
                raise ValueError("input_expectations must be an object")
            configured = {
                key for key, path in (
                    ("profile", self.profile_path),
                    ("prompts", self.prompts_file),
                    ("timestamps", self.timestamps_file))
                if path is not None
            }
            if set(self.input_expectations) != configured:
                raise ValueError(
                    "input_expectations must exactly match configured workload "
                    "inputs")
            normalized_expectations = {}
            for key in sorted(configured):
                item = self.input_expectations[key]
                if not isinstance(item, dict) \
                        or set(item) != {"sha256", "bytes"}:
                    raise ValueError(
                        f"input_expectations.{key} must contain exactly "
                        "sha256 and bytes")
                digest = item["sha256"]
                size = item["bytes"]
                if not isinstance(digest, str) or len(digest) != 64 \
                        or any(ch not in "0123456789abcdef" for ch in digest):
                    raise ValueError(
                        f"input_expectations.{key}.sha256 must be a lowercase "
                        "SHA-256 digest")
                if not isinstance(size, int) or isinstance(size, bool) \
                        or size < 0:
                    raise ValueError(
                        f"input_expectations.{key}.bytes must be a "
                        "non-negative integer")
                normalized_expectations[key] = {
                    "sha256": digest, "bytes": size}
            self.input_expectations = normalized_expectations
        for name in ("capture_endpoint_metadata", "measure_network_path"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.rate_limits is not None:
            extra_body = endpoint_config.extra_body or {}
            if "service_tier" in extra_body \
                    and extra_body["service_tier"] != "default":
                raise ValueError(
                    "rate_limits uses the standard pay-per-token accounting "
                    "model, so endpoint extra_body.service_tier must be "
                    "absent or the exact string 'default'; priority and "
                    "other tiers need their own limits and pricing evidence")
            if not self.capture_endpoint_metadata:
                raise ValueError(
                    "rate_limits requires capture_endpoint_metadata=true so "
                    "the configured model can be checked at run time")
            if self.pricing is not None \
                    and self.pricing.get("mode") != "per_token":
                raise ValueError(
                    "pay-per-token rate_limits cannot be combined with "
                    "provisioned pricing")
            _scheme, endpoint_host, _port = normalized_origin(
                endpoint_config.base_url)
            if not (endpoint_host.endswith(".databricks.com")
                    or endpoint_host.endswith(".azuredatabricks.net")):
                raise ValueError(
                    "databricks rate_limits requires a Databricks workspace "
                    "host")
            from .endpoint_meta import endpoint_name_from_path
            endpoint_name = endpoint_name_from_path(endpoint_config.path)
            if endpoint_name is None:
                raise ValueError(
                    "databricks rate_limits requires a direct "
                    "/serving-endpoints/<name>/invocations route")
            if endpoint_name != self.rate_limits["model"]:
                raise ValueError(
                    "rate_limits.model must match the serving endpoint name "
                    f"({endpoint_name})")
        if isinstance(self.start_tolerance_s, bool) \
                or not isinstance(self.start_tolerance_s, (int, float)) \
                or not math.isfinite(float(self.start_tolerance_s)) \
                or self.start_tolerance_s < 0:
            raise ValueError("start_tolerance_s must be non-negative and finite")
        if self.start_at_unix is not None \
                and (isinstance(self.start_at_unix, bool)
                     or not isinstance(self.start_at_unix, (int, float))
                     or not math.isfinite(float(self.start_at_unix))):
            raise ValueError("start_at_unix must be finite")
        if self.run_id is not None and (
                not isinstance(self.run_id, str) or not self.run_id.strip()):
            raise ValueError("run_id must be a non-empty string when set")
        if self.run_id is not None:
            self.run_id = self.run_id.strip()
        if not isinstance(self.out_dir, (str, os.PathLike)) \
                or not str(self.out_dir).strip():
            raise ValueError("out_dir must be a non-empty path")
        self.out_dir = str(self.out_dir)
        for name in ("title", "label"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.shard_total > 1:
            if self.sizing_concurrency is not None:
                raise ValueError(
                    "sharded runs cannot size independently; perform sizing "
                    "once, then put the resulting fixed QPS in every shard")
            if not isinstance(self.run_id, str) or not self.run_id.strip():
                raise ValueError("sharded runs require one shared non-empty run_id")
            if self.start_at_unix is None:
                raise ValueError("sharded runs require one shared start_at_unix")
        if self.sizing_concurrency is None and self.timestamps_file is None:
            validate_schedule_capacity(self.duration_s, self.qps_max)


def _shard_concurrency(rc) -> int | None:
    """Exact quotient/remainder share of the open-loop sizing hint.

    A share may legitimately be zero when the global hint is smaller than the
    shard count. Inflating every shard to one changes the requested total.
    """
    target = rc.sizing_concurrency
    if target is None:
        return None
    q, r = divmod(target, rc.shard_total)
    return q + (1 if rc.shard_index < r else 0)


def _input_limits(path: str | os.PathLike, input_kind: str | None) -> dict:
    kind = input_kind
    if kind is None:
        suffix = Path(path).suffix.lower()
        if suffix in {".txt", ".jsonl", ".ndjson"}:
            kind = "prompts"
        elif suffix in {".trace", ".timestamps"}:
            kind = "timestamps"
        else:
            kind = "profile"
    if kind == "prompts" and Path(path).suffix.lower() == ".json":
        kind = "prompts_json"
    try:
        return _INPUT_LIMITS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown workload input kind: {input_kind!r}") from exc


def _file_identity(path: str | None, *, input_kind: str | None = None) \
        -> str | None:
    if not path:
        return None
    try:
        raw, _info = _read_stable_bytes(path, input_kind=input_kind)
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        return f"unreadable:{path}"


def _read_stable_bytes(path: str, *, input_kind: str | None = None) \
        -> tuple[bytes, os.stat_result]:
    """Read one bounded immutable regular-file view without following links."""
    source = Path(path)
    limits = _input_limits(source, input_kind)
    try:
        path_info = source.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect input {source}: {exc}") from exc
    if not stat.S_ISREG(path_info.st_mode):
        raise ValueError(f"workload input is not a regular file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError(f"cannot snapshot input {source}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"workload input is not a regular file: {source}")
        if (path_info.st_dev, path_info.st_ino) != \
                (before.st_dev, before.st_ino):
            raise ValueError(
                f"input changed while it was being opened: {source}")
        if before.st_size > limits["max_bytes"]:
            raise ValueError(
                f"{input_kind or 'workload'} input {source} declares "
                f"{before.st_size:,} bytes, above its "
                f"{limits['max_bytes']:,}-byte snapshot limit")
        chunks = []
        remaining = before.st_size
        line_count = 0
        current_line_bytes = 0
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(
                    f"input was truncated while being snapshotted: {source}")
            chunks.append(chunk)
            remaining -= len(chunk)
            parts = chunk.split(b"\n")
            if len(parts) == 1:
                current_line_bytes += len(chunk)
                if current_line_bytes > limits["max_line_bytes"]:
                    raise ValueError(
                        f"{input_kind or 'workload'} input {source} contains "
                        f"a line above its {limits['max_line_bytes']:,}-byte "
                        "record limit")
            else:
                first_size = current_line_bytes + len(parts[0])
                if first_size > limits["max_line_bytes"] \
                        or any(len(part) > limits["max_line_bytes"]
                               for part in parts[1:-1]):
                    raise ValueError(
                        f"{input_kind or 'workload'} input {source} contains "
                        f"a line above its {limits['max_line_bytes']:,}-byte "
                        "record limit")
                line_count += len(parts) - 1
                if line_count > limits["max_lines"]:
                    raise ValueError(
                        f"{input_kind or 'workload'} input {source} exceeds "
                        f"its {limits['max_lines']:,}-line limit")
                current_line_bytes = len(parts[-1])
                if current_line_bytes > limits["max_line_bytes"]:
                    raise ValueError(
                        f"{input_kind or 'workload'} input {source} contains "
                        f"a line above its {limits['max_line_bytes']:,}-byte "
                        "record limit")
        extra = os.read(fd, 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_ctime_ns)
    raw = b"".join(chunks)
    if raw and not raw.endswith(b"\n"):
        line_count += 1
        if line_count > limits["max_lines"]:
            raise ValueError(
                f"{input_kind or 'workload'} input {source} exceeds its "
                f"{limits['max_lines']:,}-line limit")
    if identity_before != identity_after or len(raw) != before.st_size or extra:
        raise ValueError(
            f"input changed while it was being snapshotted: {source}")
    return raw, before


def _snapshot_run_inputs(rc: RunConfig, directory: Path) \
        -> tuple[RunConfig, dict]:
    """Copy workload inputs once; all later parsing uses these private bytes."""
    replacements = {}
    metadata = {}
    for field, key in (("profile_path", "profile"),
                       ("prompts_file", "prompts"),
                       ("timestamps_file", "timestamps")):
        original = getattr(rc, field)
        if not original:
            continue
        raw, info = _read_stable_bytes(original, input_kind=key)
        # Keep the original basename so effective configs and sweep identity
        # remain comparable while the private parent directory prevents name
        # collisions between profile/prompts/trace inputs.
        snapshot_parent = directory / key
        snapshot_parent.mkdir(mode=0o700)
        snapshot = snapshot_parent / Path(original).name
        fd = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(raw)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("short write")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        replacements[field] = str(snapshot)
        metadata[key] = {
            # The digest identifies the exact bytes. Persisting an absolute
            # local path adds no reproducibility after an artifact is moved,
            # but does expose usernames and customer directory names.
            "name": Path(original).name,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "captured_size": int(info.st_size),
            "captured_mtime_ns": int(info.st_mtime_ns),
            "snapshot_used_for_workload": True,
        }
    return dataclasses.replace(rc, **replacements), metadata


def _enforce_input_expectations(rc: RunConfig, captured: dict) -> None:
    """Fail a saved rerun closed when an external input changed."""
    if rc.input_expectations is None:
        return
    for key, expected in rc.input_expectations.items():
        actual = captured.get(key) or {}
        if any(actual.get(field) != expected[field]
               for field in ("sha256", "bytes")):
            raise ValueError(
                f"{key} input bytes changed since this config was saved; "
                "refusing to label a different workload as the same rerun. "
                "Create a new benchmark config intentionally.")


def _effective_config(original: RunConfig, effective: RunConfig) -> dict:
    """Persist resolved values without leaking private temporary paths."""
    value = dataclasses.asdict(effective)
    for field in ("profile_path", "prompts_file", "timestamps_file"):
        original_path = getattr(original, field)
        value[field] = (Path(original_path).name
                        if original_path is not None else None)
    value["out_dir"] = Path(original.out_dir).name
    return redact_secrets(value)


def _resolved_workload_id(rc: RunConfig, inputs: dict) -> str:
    """Deterministic identity of logical bodies and their global ordering."""
    material = {
        "schema": 1,
        "inputs": {key: {"sha256": value["sha256"],
                         "bytes": value["bytes"]}
                   for key, value in sorted(inputs.items())},
        "input_mode": "prompts" if rc.prompts_file else "profile",
        "seed": rc.seed,
        "cpt": rc.cpt,
        "calibrate_n": rc.calibrate_n,
        "pool_docs_per_bucket": rc.pool_docs_per_bucket,
        "pool_zipf_s": rc.pool_zipf_s,
        "max_output_tokens_cap": rc.max_output_tokens_cap,
        "schedule": {
            key: getattr(rc, key) for key in (
                "duration_s", "qps_base", "qps_burst", "qps_min",
                "qps_max", "rate_scale", "sizing_concurrency")
        },
        "request_shape": {
            "endpoint_adapter": endpoint_adapter_contract(
                rc.endpoint.get("adapter", DEFAULT_ENDPOINT_ADAPTER)),
            "model": rc.endpoint.get("model"),
            "temperature": rc.endpoint.get("temperature", 0.0),
            "extra_body": rc.endpoint.get("extra_body") or {},
        },
    }
    # Only the digest is persisted. Hash the real values so two payloads that
    # differ solely in a credential-like parameter do not collide, while the
    # effective configuration itself remains redacted.
    return "workload-" + canonical_sha256(material)[:24]


def _execution_ids(rc: RunConfig) -> tuple[str, str, str]:
    logical = (rc.run_id if rc.run_id
               else f"run-{uuid.uuid4().hex}")
    return logical, f"execution-{uuid.uuid4().hex}", \
        f"artifact-{uuid.uuid4().hex}"


def _schedule_identities(full: dict, selected: dict, rc: RunConfig) \
        -> tuple[dict, dict]:
    """Hash canonical binary schedule/index vectors without lossy JSON."""
    global_ts = np.asarray(full["timestamps"], dtype="<f8")
    shard_ts = np.asarray(selected["timestamps"], dtype="<f8")
    indices = np.asarray(selected.get("global_indices", []), dtype="<i8")

    def edge(values, which):
        if not len(values):
            return None
        return float(values.min() if which == "min" else values.max())

    schedule_identity = {
        "encoding": "float64-le-seconds-from-run-start",
        "global_timestamps_sha256": sha256_bytes(global_ts.tobytes()),
        "global_count": int(len(global_ts)),
        "global_min_s": edge(global_ts, "min"),
        "global_max_s": edge(global_ts, "max"),
        "shard_timestamps_sha256": sha256_bytes(shard_ts.tobytes()),
        "shard_count": int(len(shard_ts)),
        "shard_min_s": edge(shard_ts, "min"),
        "shard_max_s": edge(shard_ts, "max"),
    }
    index_identity = {
        "encoding": "int64-le",
        "global_indices_sha256": sha256_bytes(indices.tobytes()),
        "count": int(len(indices)),
        "min": int(indices.min()) if len(indices) else None,
        "max": int(indices.max()) if len(indices) else None,
        "global_count": int(len(global_ts)),
        "shard_index": rc.shard_index,
        "shard_total": rc.shard_total,
        "partition": ("round_robin_modulo" if rc.shard_total > 1
                      else "unsharded"),
    }
    return schedule_identity, index_identity


def _resolved_run_id(rc: RunConfig) -> str:
    """Stable identity shared by an unsharded run and all of its shards."""
    if rc.run_id:
        return rc.run_id
    material = {
        "seed": rc.seed,
        "profile": _file_identity(rc.profile_path, input_kind="profile"),
        "prompts": _file_identity(rc.prompts_file, input_kind="prompts"),
        "endpoint_path": rc.endpoint.get("path"),
        "endpoint_model": rc.endpoint.get("model"),
        "endpoint_adapter": endpoint_adapter_contract(
            rc.endpoint.get("adapter", DEFAULT_ENDPOINT_ADAPTER)),
        "extra_body": rc.endpoint.get("extra_body") or {},
        "cpt": rc.cpt,
        "pool_docs_per_bucket": rc.pool_docs_per_bucket,
        "pool_zipf_s": rc.pool_zipf_s,
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return "auto-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _stable_request_id(run_id: str, global_index: int,
                       namespace: str = "replay") -> str:
    return hashlib.sha256(
        f"{run_id}:{namespace}:{global_index}".encode()).hexdigest()[:16]


_SETUP_REQUEST_BINDING_SCHEMA = "setup-request-binding/v1"
_PREFLIGHT_BINDING_SCHEMA = "carried-preflight-binding/v1"
_SETUP_ARTIFACT_REFERENCE_SCHEMA = "setup-artifact-reference/v1"
_HEX_SHA256 = frozenset("0123456789abcdef")


def _binding_sha256(value: object) -> str:
    """Hash JSON semantics deterministically, independent of dict insertion."""
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(char in _HEX_SHA256 for char in value)


def _json_value_binding(value: object, *, field_present: bool) -> dict:
    """Bind a request value without copying customer/model content."""
    return {
        "field_present": bool(field_present),
        "value_sha256": _binding_sha256(value),
    }


def _endpoint_request_binding(endpoint: dict, max_output_tokens_cap: int) \
        -> dict:
    """Content-safe identity of every endpoint field affecting wire bytes."""
    if not isinstance(endpoint, dict):
        raise ValueError("endpoint request binding needs an endpoint object")
    ecfg = EndpointConfig(**copy.deepcopy(endpoint))
    scheme, host, port = normalized_origin(ecfg.base_url)
    adapter_contract = endpoint_adapter_contract(ecfg.adapter)
    return {
        "normalized_origin": {
            "scheme": scheme, "host": host, "port": port,
        },
        "path": ecfg.path,
        "adapter_id": ecfg.adapter,
        "adapter_contract_sha256": _binding_sha256(adapter_contract),
        "adapter_implementation_sha256": adapter_contract.get(
            "implementation_sha256"),
        "model": _json_value_binding(
            ecfg.model, field_present="model" in endpoint),
        "temperature": {
            "field_present": "temperature" in endpoint,
            "value": ecfg.temperature,
        },
        "include_usage": {
            "field_present": "include_usage" in endpoint,
            "value": ecfg.include_usage,
        },
        "extra_body": _json_value_binding(
            ecfg.extra_body or {}, field_present="extra_body" in endpoint),
        "max_output_tokens_cap": int(max_output_tokens_cap),
    }


def _representative_plan_binding(ecfg: EndpointConfig, plan: dict, *,
                                 position: int) -> dict:
    """Bind one deterministic representative without persisting its text."""
    max_tokens = int(plan["max_output"])
    logical = _payload_hash(ecfg, plan["messages"], max_tokens)
    return {
        "position": int(position),
        "representative_sha256": _binding_sha256(
            plan.get("representative")),
        "global_index": plan.get("global_index"),
        "sample_index": plan.get("sample_index"),
        "prompt_index": plan.get("prompt_index"),
        "body_request_id": plan.get("body_request_id"),
        "max_tokens": max_tokens,
        "logical_request_body_sha256": logical,
    }


def _preflight_execution_binding(rc: RunConfig,
                                 representative_plans: list[dict]) -> dict:
    """Bind the exact measured config/workload authorized by preflight."""
    ecfg = EndpointConfig(**copy.deepcopy(rc.endpoint))
    inputs = copy.deepcopy(rc.input_expectations or {})
    workload = {
        "input_mode": "prompts" if rc.prompts_file else "profile",
        "input_expectations_sha256": _binding_sha256(inputs),
        "seed": rc.seed,
        "cpt": rc.cpt,
        "pool_docs_per_bucket": rc.pool_docs_per_bucket,
        "pool_zipf_s": rc.pool_zipf_s,
    }
    return {
        "endpoint_request": _endpoint_request_binding(
            rc.endpoint, rc.max_output_tokens_cap),
        "workload": workload,
        "representatives": [
            _representative_plan_binding(ecfg, plan, position=position)
            for position, plan in enumerate(representative_plans, start=1)
        ],
    }


def _setup_request_binding(*, endpoint: dict,
                           max_output_tokens_cap: int, plan: dict,
                           phase: str, position: int, request_id: str,
                           row: dict) -> dict:
    """Build one row-level endpoint/body/trace binding after its attempts."""
    if phase not in {"preflight", "probe"}:
        raise ValueError("setup request binding phase is invalid")
    if not isinstance(request_id, str) or not request_id \
            or len(request_id) > 128 \
            or any(char not in "0123456789abcdef" for char in request_id):
        raise ValueError(
            "setup request trace IDs must be 1-128 lowercase hex characters")
    ecfg = EndpointConfig(**copy.deepcopy(endpoint))
    plan_binding = _representative_plan_binding(
        ecfg, {**plan, "max_output": int(plan["max_output"])},
        position=position)
    physical = list(row.get("physical_request_body_sha256s") or [])
    if not all(_is_sha256(item) for item in physical):
        raise ValueError("setup request physical body hashes are invalid")
    attempts = row.get("request_attempts")
    if attempts is not None and (
            isinstance(attempts, bool) or not isinstance(attempts, int)
            or attempts < 0 or len(physical) != attempts):
        raise ValueError(
            "setup request physical hashes disagree with request attempts")
    possible = {
        hashlib.sha256(serialize_request_body(
            ecfg, plan["messages"], int(plan["max_output"]),
            include_usage=include_usage)).hexdigest()
        for include_usage in ({False, True} if ecfg.include_usage else {False})
    }
    if any(item not in possible for item in physical):
        raise ValueError(
            "setup request physical body hash does not match the exact "
            "endpoint configuration and representative")
    binding = {
        "schema_version": _SETUP_REQUEST_BINDING_SCHEMA,
        "phase": phase,
        "position": int(position),
        "trace_request_id": request_id,
        "endpoint_request": _endpoint_request_binding(
            endpoint, max_output_tokens_cap),
        "representative": plan_binding,
        "physical_request_body_sha256s": physical,
        "physical_hash_status": (
            "exact" if attempts is not None else "outcome_unknown"),
    }
    if row.get("request_body_sha256") != \
            plan_binding["logical_request_body_sha256"]:
        raise ValueError(
            "setup row logical body hash disagrees with its exact request")
    return binding


def _attach_setup_request_binding(row: dict, *, endpoint: dict,
                                  max_output_tokens_cap: int, plan: dict,
                                  phase: str, position: int) -> dict:
    binding = _setup_request_binding(
        endpoint=endpoint, max_output_tokens_cap=max_output_tokens_cap,
        plan=plan, phase=phase, position=position,
        request_id=row.get("request_id"), row=row)
    row["setup_request_binding"] = binding
    row["setup_request_binding_sha256"] = _binding_sha256(binding)
    return row


def _rescope_representative_plans(plans: list[dict],
                                  execution_scope_id: str) -> list[dict]:
    """Give deterministic bodies fresh execution-scoped transport IDs."""
    if not isinstance(execution_scope_id, str) or not execution_scope_id \
            or len(execution_scope_id) > 128 \
            or any(ord(char) < 0x21 or ord(char) > 0x7e
                   for char in execution_scope_id):
        raise ValueError("execution request scope is invalid")
    scoped = []
    for position, original in enumerate(plans):
        plan = copy.deepcopy(original)
        # The synthetic text was constructed with the stable body ID. The
        # transport/journal ID is deliberately not embedded in that text.
        body_id = plan.get("body_request_id") or plan.get("request_id")
        plan["body_request_id"] = body_id
        plan["request_id"] = _stable_request_id(
            execution_scope_id, position, "preflight")
        plan["execution_scope_id"] = execution_scope_id
        scoped.append(plan)
    return scoped


def _payload_hash(ecfg: EndpointConfig, messages: list[dict],
                  max_tokens: int, *, adapter_execution=None) -> str:
    """Hash the deterministic logical body, excluding learned wire fallback."""
    if adapter_execution is None:
        # Standalone planning/binding calls retain full registry attestation.
        raw = serialize_request_body(
            ecfg, messages, int(max_tokens), include_usage=False)
    else:
        # A live EndpointClient already performed full start attestation and
        # rechecks at evidence sealing. Avoid distorting dispatcher pacing by
        # repeating the expensive implementation walk for every request.
        raw = adapter_execution.serialize_request(
            ecfg, messages, int(max_tokens), include_usage=False)
    return hashlib.sha256(raw).hexdigest()


def _client_payload_hash(client, fallback_ecfg: EndpointConfig,
                         messages: list[dict], max_tokens: int) -> str:
    """Use a real client's lease, retaining lightweight transport support."""
    adapter_execution = getattr(client, "adapter_execution", None)
    ecfg = getattr(client, "cfg", fallback_ecfg)
    if adapter_execution is None:
        # Injected transports used by embedding callers and tests predate the
        # execution lease. Keep their fully-attested standalone behavior.
        return _payload_hash(ecfg, messages, max_tokens)
    return _payload_hash(
        ecfg, messages, max_tokens,
        adapter_execution=adapter_execution)


class _PreparedWorkload:
    """One globally indexed workload, identical before and after sharding."""

    def __init__(self, rc: RunConfig, total_n: int, *,
                 loaded_profile: prof.Profile | None = None,
                 loaded_prompts: list[list[dict]] | None = None):
        if total_n <= 0:
            raise ValueError("workload needs at least one request")
        if loaded_profile is not None and loaded_prompts is not None:
            raise ValueError(
                "prevalidated workload cannot contain both profile and "
                "prompt inputs")
        self.rc = rc
        self.total_n = total_n
        self.prompts_mode = bool(rc.prompts_file)
        self.profile = None
        self.prompt_msgs = None
        self.mat = None
        if self.prompts_mode:
            if loaded_profile is not None:
                raise ValueError(
                    "profile input cannot prepare a prompts-mode workload")
            if loaded_prompts is None:
                from .prompts import load_prompts
                loaded_prompts = load_prompts(rc.prompts_file)
            self.prompt_msgs = loaded_prompts
        else:
            if loaded_prompts is not None:
                raise ValueError(
                    "prompt input cannot prepare a profile-mode workload")
            self.profile = (loaded_profile if loaded_profile is not None
                            else prof.Profile.from_json(rc.profile_path))
            self.mat = TextMaterializer(cpt=rc.cpt)
            self.draw = prof.sample(self.profile, total_n, seed=rc.seed)
            self.pool = PrefixPool(
                seed=rc.seed + 4, docs_per_bucket=rc.pool_docs_per_bucket,
                zipf_s=rc.pool_zipf_s)
            self.assignment = self.pool.assign(self.draw["prefix_tokens"])

    @property
    def prompts_count(self) -> int | None:
        return len(self.prompt_msgs) if self.prompt_msgs is not None else None

    def set_cpt(self, cpt: float) -> None:
        if not self.prompts_mode:
            self.mat = TextMaterializer(cpt=cpt)

    def plan(self, global_index: int, request_id: str) -> dict:
        if not 0 <= global_index < self.total_n:
            raise IndexError(f"global workload index {global_index} out of range")
        if self.prompts_mode:
            prompt_index = global_index % len(self.prompt_msgs)
            messages = self.prompt_msgs[prompt_index]
            chars = sum(len(x["content"]) for x in messages)
            return {
                "messages": messages,
                "max_output": self.rc.max_output_tokens_cap,
                "intended": (0, 0, None, prompt_index),
                "chars": chars,
                "global_index": global_index,
                "prompt_index": prompt_index,
                "sample_index": None,
                "construction": None,
                "body_request_id": request_id,
            }

        i = global_index
        input_tokens = int(self.draw["input_tokens"][i])
        prefix_tokens = int(self.assignment.prefix_tokens[i])
        # Assignment is the concrete cache structure. Keep total input fixed
        # even if a custom pool ever returns a shorter prefix.
        suffix_tokens = input_tokens - prefix_tokens
        if suffix_tokens < 0:
            raise ValueError("constructed prefix exceeds sampled input target")
        doc_id = int(self.assignment.doc_id[i])
        messages = self.mat.messages(
            request_id, doc_id, prefix_tokens,
            self.pool.doc_len.get(doc_id, 0), suffix_tokens)
        chars = sum(len(x["content"]) for x in messages)
        cache_fraction = prefix_tokens / input_tokens if input_tokens else 0.0
        return {
            "messages": messages,
            "max_output": min(int(self.draw["output_tokens"][i]),
                              self.rc.max_output_tokens_cap),
            "intended": (input_tokens, int(self.draw["output_tokens"][i]),
                         cache_fraction, doc_id),
            "chars": chars,
            "global_index": global_index,
            "prompt_index": None,
            "sample_index": global_index,
            "construction": self.mat.construction_report(messages, input_tokens),
            "body_request_id": request_id,
        }


def _representative_plans(
        rc: RunConfig, *, loaded_profile: prof.Profile | None = None,
        loaded_prompts: list[list[dict]] | None = None,
        resolved_run_id: str | None = None) -> list[dict]:
    """Concrete p50/p95 profile requests, or the first two real prompts."""
    run_id = (resolved_run_id if resolved_run_id is not None
              else _resolved_run_id(rc))
    if rc.prompts_file:
        if loaded_profile is not None:
            raise ValueError(
                "profile input cannot prepare prompts-mode representatives")
        if loaded_prompts is None:
            from .prompts import load_prompts
            loaded_prompts = load_prompts(rc.prompts_file)
        messages = loaded_prompts
        plans = []
        for i in range(2):
            prompt_index = i % len(messages)
            msgs = messages[prompt_index]
            body_rid = _stable_request_id(run_id, i, "preflight-body")
            plans.append({
                "messages": msgs, "max_output": rc.max_output_tokens_cap,
                "intended": (0, 0, None, prompt_index),
                "chars": sum(len(x["content"]) for x in msgs),
                "global_index": i, "prompt_index": prompt_index,
                "sample_index": None, "construction": None,
                "request_id": _stable_request_id(run_id, i, "preflight"),
                "body_request_id": body_rid,
                "representative": f"prompt {prompt_index}",
            })
        return plans

    if loaded_prompts is not None:
        raise ValueError(
            "prompt input cannot prepare profile-mode representatives")
    p = (loaded_profile if loaded_profile is not None
         else prof.Profile.from_json(rc.profile_path))
    mat = TextMaterializer(cpt=rc.cpt)
    inputs = np.asarray([
        int(round(float(p.input_tokens["p50"]))),
        int(round(float(p.input_tokens["p95"])))], dtype=int)
    outputs = np.asarray([
        int(round(float(p.output_tokens["p50"]))),
        int(round(float(p.output_tokens["p95"])))], dtype=int)
    cache = np.asarray([
        float(p.cache_fraction["p50"]),
        float(p.cache_fraction["p95"])], dtype=float)
    wanted_prefix = np.round(inputs * cache).astype(int)
    pool = PrefixPool(seed=rc.seed + 4,
                      docs_per_bucket=rc.pool_docs_per_bucket,
                      zipf_s=rc.pool_zipf_s)
    assignment = pool.assign(wanted_prefix)
    plans = []
    for i, quantile in enumerate(("p50", "p95")):
        body_rid = _stable_request_id(run_id, i, "preflight-body")
        prefix = int(assignment.prefix_tokens[i])
        suffix = int(inputs[i]) - prefix
        doc_id = int(assignment.doc_id[i])
        msgs = mat.messages(body_rid, doc_id, prefix,
                            pool.doc_len.get(doc_id, 0), suffix)
        plans.append({
            "messages": msgs,
            "max_output": min(int(outputs[i]), rc.max_output_tokens_cap),
            "intended": (int(inputs[i]), int(outputs[i]),
                         prefix / int(inputs[i]), doc_id),
            "chars": sum(len(x["content"]) for x in msgs),
            "global_index": i, "prompt_index": None, "sample_index": i,
            "construction": mat.construction_report(msgs, int(inputs[i])),
            "request_id": _stable_request_id(run_id, i, "preflight"),
            "body_request_id": body_rid,
            "representative": quantile,
        })
    return plans


def _validate_prompt_resources(prompts: list[list[dict]], path: str) -> None:
    """Bound decoded prompt count and any one replayable request record."""
    if len(prompts) > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
        raise ValueError(
            f"prompts input {path} contains {len(prompts):,} prompts, above "
            f"the exact-analysis limit of "
            f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}")
    for index, messages in enumerate(prompts):
        encoded = json.dumps(
            messages, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_PROMPT_RECORD_BYTES:
            raise ValueError(
                f"prompt item {index} in {path} is {len(encoded):,} bytes, "
                f"above the {_MAX_PROMPT_RECORD_BYTES:,}-byte per-request "
                "workload limit")


@dataclasses.dataclass
class PrevalidatedRunInputs:
    """Fully parsed, endpoint-free inputs reusable by preflight and runner.

    The object deliberately retains the parsed workload source and exact
    deterministic schedule so a caller does not validate one file view and
    later reread another. A sizing-derived final schedule cannot exist until
    unloaded service time is measured; its configured-qps ceiling schedule is
    materialized here so the final schedule is guaranteed to be a subset of a
    pretraffic-approved row population.
    """

    rc: RunConfig
    full_schedule: dict | None
    sizing_schedule_ceiling: dict | None
    workload: _PreparedWorkload | None
    profile: prof.Profile | None
    prompts: list[list[dict]] | None
    representative_plans: list[dict]
    schedule_kind: str


def exact_analysis_row_counts(
        prevalidated: PrevalidatedRunInputs) -> dict[str, int]:
    """Return the conservative rows one run can materialize exactly."""
    if not isinstance(prevalidated, PrevalidatedRunInputs):
        raise TypeError("expected PrevalidatedRunInputs")
    rc = prevalidated.rc
    if rc.sizing_concurrency is None:
        replay = len((prevalidated.full_schedule or {}).get("timestamps", ()))
        sizing = 0
    else:
        replay = len(
            (prevalidated.sizing_schedule_ceiling or {}).get("timestamps", ()))
        sizing = sizing_probe_row_count(rc.calibrate_n)
    return {
        "replay_rows": replay,
        "calibration_rows": min(rc.calibrate_n, replay),
        "sizing_rows": sizing,
    }


def sizing_probe_row_count(calibrate_n: int) -> int:
    """Return the exact number of paid rows used by concurrency sizing."""
    if not isinstance(calibrate_n, int) or isinstance(calibrate_n, bool) \
            or calibrate_n < 0:
        raise ValueError("calibrate_n must be a non-negative integer")
    return max(_MIN_SIZING_PROBES,
               min(calibrate_n, _MAX_SIZING_PROBES))


def enforce_exact_analysis_envelope(
        prevalidated: PrevalidatedRunInputs, *, setup_rows: int = 0,
        context: str = "run") -> int:
    """Enforce the exact-analysis row envelope for one validated workload."""
    return validate_exact_analysis_capacity(
        **exact_analysis_row_counts(prevalidated),
        setup_rows=setup_rows,
        context=context,
    )


def _loaded_source_run_id(
        rc: RunConfig, loaded_profile: prof.Profile | None,
        loaded_prompts: list[list[dict]] | None) -> str:
    """Resolve preflight IDs from the already parsed source, without rereads."""
    if rc.run_id:
        return rc.run_id
    material = {
        "seed": rc.seed,
        "profile": (dataclasses.asdict(loaded_profile)
                    if loaded_profile is not None else None),
        "prompts": loaded_prompts,
        "endpoint_path": rc.endpoint.get("path"),
        "endpoint_model": rc.endpoint.get("model"),
        "endpoint_adapter": endpoint_adapter_contract(
            rc.endpoint.get("adapter", DEFAULT_ENDPOINT_ADAPTER)),
        "extra_body": rc.endpoint.get("extra_body") or {},
        "cpt": rc.cpt,
        "pool_docs_per_bucket": rc.pool_docs_per_bucket,
        "pool_zipf_s": rc.pool_zipf_s,
    }
    raw = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "auto-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _reusable_source_matches(previous: PrevalidatedRunInputs,
                             current: RunConfig) -> bool:
    """Whether a parsed profile/prompt view belongs to the same source mode."""
    old = previous.rc
    return (old.profile_path == current.profile_path
            and old.prompts_file == current.prompts_file)


def _representative_settings_match(previous: PrevalidatedRunInputs,
                                   current: RunConfig) -> bool:
    """Whether an existing concrete representative plan can be reused."""
    old = previous.rc
    scalar_fields = (
        "profile_path", "prompts_file", "seed", "cpt",
        "pool_docs_per_bucket", "pool_zipf_s", "max_output_tokens_cap",
        "run_id",
    )
    if any(getattr(old, name) != getattr(current, name)
           for name in scalar_fields):
        return False
    body_endpoint_fields = (
        "path", "model", "adapter", "temperature", "extra_body")
    return all(old.endpoint.get(name) == current.endpoint.get(name)
               for name in body_endpoint_fields)


def prevalidate_run_inputs(
        rc: RunConfig, *,
        require_nonempty_schedule: bool = True,
        reuse_source: PrevalidatedRunInputs | None = None
        ) -> PrevalidatedRunInputs:
    """Parse and construct every locally knowable input without side effects.

    No credentials, environment tokens, endpoint clients, control-plane APIs,
    network probes, output files, or inference requests are touched.  Fixed
    synthetic schedules and timestamp traces are materialized exactly once.
    Profile sampling, prefix assignment, prompt parsing, and representative
    body construction are completed before the returned evidence can be used
    by a paid preflight or runner.
    """
    if not isinstance(rc, RunConfig):
        raise TypeError("prevalidate_run_inputs requires a RunConfig")
    if not isinstance(require_nonempty_schedule, bool):
        raise ValueError("require_nonempty_schedule must be boolean")
    if reuse_source is not None \
            and not isinstance(reuse_source, PrevalidatedRunInputs):
        raise TypeError("reuse_source must be PrevalidatedRunInputs")

    # Re-run dataclass validation and detach nested caller-owned policy/request
    # dictionaries. A caller mutating a previously constructed RunConfig must
    # not bypass the same fail-closed controls used by run().
    checked = dataclasses.replace(
        rc,
        endpoint=copy.deepcopy(rc.endpoint),
        acceptance_targets=copy.deepcopy(rc.acceptance_targets),
        pricing=copy.deepcopy(rc.pricing),
        rate_limits=copy.deepcopy(rc.rate_limits),
        input_expectations=copy.deepcopy(rc.input_expectations))

    if reuse_source is not None:
        if not _reusable_source_matches(reuse_source, checked):
            raise ValueError(
                "reused prevalidation source does not match workload "
                "construction settings")
        loaded_profile = reuse_source.profile
        loaded_prompts = reuse_source.prompts
    else:
        loaded_profile = None
        loaded_prompts = None
        if checked.profile_path:
            loaded_profile = prof.Profile.from_json(checked.profile_path)
            # Execute every schema/sampler-bound check without allocating the
            # eventual workload. The parsed object is reused below.
            prof.sample(loaded_profile, 0, seed=checked.seed)
        else:
            from .prompts import load_prompts
            loaded_prompts = load_prompts(checked.prompts_file)

    if loaded_prompts is not None:
        _validate_prompt_resources(
            loaded_prompts, str(checked.prompts_file))

    full_schedule = None
    sizing_schedule_ceiling = None
    if checked.sizing_concurrency is None:
        if checked.timestamps_file:
            full_schedule = load_trace(
                checked.timestamps_file, duration_cap_s=checked.duration_s,
                row_limit=MAX_EXACT_ANALYSIS_REQUEST_ROWS)
            schedule_kind = "timestamp_trace"
        else:
            full_schedule = make_schedule(
                duration_s=checked.duration_s,
                qps_base=checked.qps_base,
                qps_burst=checked.qps_burst,
                qps_min=checked.qps_min,
                qps_max=checked.qps_max,
                rate_scale=checked.rate_scale,
                seed=checked.seed + 16,
                request_limit=MAX_EXACT_ANALYSIS_REQUEST_ROWS)
            schedule_kind = "deterministic_synthetic"
        total_n = len(full_schedule["timestamps"])
        if require_nonempty_schedule and total_n == 0:
            raise RuntimeError(
                "schedule produced zero arrivals; raise rate_scale or duration")
    else:
        # The final rate depends on paid service-time probes, but it is capped
        # by qps_max. Materialize that maximum-rate Poisson schedule now; the
        # post-sizing schedule is an exact deterministic subset, so analysis
        # can never grow past this pretraffic-approved population.
        sizing_schedule_ceiling = make_schedule(
            duration_s=checked.duration_s,
            qps_base=checked.qps_max,
            qps_burst=checked.qps_max,
            qps_min=checked.qps_max,
            qps_max=checked.qps_max,
            rate_scale=1.0,
            seed=checked.seed + 16,
            request_limit=MAX_EXACT_ANALYSIS_REQUEST_ROWS)
        if not len(sizing_schedule_ceiling["timestamps"]):
            raise RuntimeError(
                "sizing qps_max ceiling produced zero arrivals; increase "
                "duration_s or qps_max")
        schedule_kind = "sizing_subset_of_prevalidated_qps_max_ceiling"
        total_n = sizing_probe_row_count(checked.calibrate_n)

    workload = None
    if total_n > 0:
        workload = _PreparedWorkload(
            checked, total_n, loaded_profile=loaded_profile,
            loaded_prompts=loaded_prompts)

        # Materialize representative actual sampled bodies as well as the
        # declared p50/p95 preflight bodies. This catches deterministic prefix,
        # suffix, and character-budget failures before endpoint access without
        # constructing every potentially large replay body at once.
        if not workload.prompts_mode:
            largest = int(np.argmax(workload.draw["input_tokens"]))
            indices = sorted({0, total_n - 1, largest})
        else:
            indices = sorted({0, total_n - 1})
        for index in indices:
            workload.plan(
                index, _stable_request_id(
                    "input-prevalidation", index, "body"))

    if reuse_source is not None \
            and _representative_settings_match(reuse_source, checked):
        representatives = reuse_source.representative_plans
    else:
        representatives = _representative_plans(
            checked, loaded_profile=loaded_profile,
            loaded_prompts=loaded_prompts,
            resolved_run_id=_loaded_source_run_id(
                checked, loaded_profile, loaded_prompts))
    result = PrevalidatedRunInputs(
        rc=checked,
        full_schedule=full_schedule,
        sizing_schedule_ceiling=sizing_schedule_ceiling,
        workload=workload,
        profile=loaded_profile,
        prompts=loaded_prompts,
        representative_plans=representatives,
        schedule_kind=schedule_kind,
    )
    enforce_exact_analysis_envelope(result, context="run prevalidation")
    return result


def _annotate_result(res, phase: str, plan: dict, body_hash: str) -> dict:
    row = dataclasses.asdict(res)
    row.update(phase=phase, global_index=plan["global_index"],
               sample_index=plan["sample_index"],
               prompt_index=plan["prompt_index"],
               body_request_id=plan.get("body_request_id"),
               request_body_sha256=body_hash)
    if plan["construction"]:
        row.update(
            constructed_target_chars=plan["construction"]["target_chars"],
            constructed_actual_chars=plan["construction"]["actual_chars"],
            constructed_error_chars=plan["construction"]["error_chars"])
    return row


def _clean_measurement_row(row: dict) -> bool:
    """Require a complete, parse-clean response for numeric calibration."""
    return bool(
        row.get("ok")
        and row.get("stream_complete") is True
        and isinstance(row.get("parse_errors", 0), int)
        and not isinstance(row.get("parse_errors", 0), bool)
        and row.get("parse_errors", 0) == 0)


def _send_request(client, messages, max_tokens, request_id, scheduled_s,
                  dispatch_lag_ms, intended, chars_sent, *,
                  scheduled_monotonic: float | None = None):
    """Call current clients with exact clocks, retaining old test adapters."""
    kwargs = {}
    if scheduled_monotonic is not None:
        try:
            parameters = inspect.signature(client.send).parameters.values()
            supports_clock = any(
                p.name == "scheduled_monotonic"
                or p.kind == inspect.Parameter.VAR_KEYWORD
                for p in parameters)
        except (TypeError, ValueError):
            supports_clock = True
        if supports_clock:
            kwargs["scheduled_monotonic"] = scheduled_monotonic
    return client.send(messages, max_tokens, request_id, scheduled_s,
                       dispatch_lag_ms, intended, chars_sent, **kwargs)


def _exception_result(request_id: str, phase: str, plan: dict,
                      body_hash: str, error: str,
                      scheduled_s: float = 0.0,
                      dispatch_lag_ms: float = 0.0, *,
                      known_not_sent: bool = False,
                      endpoint_adapter: str = DEFAULT_ENDPOINT_ADAPTER) -> dict:
    intended = plan["intended"]
    row = {
        "request_id": request_id, "scheduled_s": scheduled_s,
        "dispatch_lag_ms": dispatch_lag_ms, "t_send_unix": None,
        "first_send_unix": None, "ttfb_ms": None, "ttse_ms": None,
        "ttft_ms": None,
        "ttfr_ms": None, "ttfv_ms": None, "e2e_ms": None,
        "queue_wait_ms": None, "caller_ttfb_ms": None,
        "caller_ttse_ms": None,
        "caller_ttft_ms": None, "caller_ttfr_ms": None,
        "caller_ttfv_ms": None, "caller_ttf_tool_call_ms": None,
        "caller_send_ms": None,
        "caller_e2e_ms": None,
        "finished_unix": None,
        "status": None, "ok": False, "error": error,
        "content_chunks": 0, "interchunk_max_ms": None,
        "finish_reason": None, "prompt_tokens": None,
        "completion_tokens": None, "cached_tokens": None,
        "cached_tokens_source": None,
        "intended_input_tokens": intended[0],
        "intended_output_tokens": intended[1],
        "intended_cache_fraction": intended[2], "doc_id": intended[3],
        "chars_sent": plan["chars"], "retries": 0,
        "reasoning_tokens": None, "reasoning_tokens_source": None,
        "reasoning_chunks": 0, "connect_ms": None,
        "stream_complete": False, "visible_content_seen": False,
        "reasoning_seen": False, "truncated": False, "parse_errors": 0,
        "max_tokens_requested": plan["max_output"],
        "first_attempt_unix": None,
        "connection_attempts": 0 if known_not_sent else None,
        "request_attempts": 0 if known_not_sent else None,
        "retry_reasons": [],
        "tool_call_seen": False, "tool_call_chunks": 0,
        "ttf_tool_call_ms": None, "valid_tool_calls": 0,
        "endpoint_adapter": endpoint_adapter,
        "response_mode": get_endpoint_adapter(
            endpoint_adapter).response_mode,
        "physical_request_body_sha256s": [],
    }
    row.update(phase=phase, global_index=plan["global_index"],
               sample_index=plan["sample_index"],
               prompt_index=plan["prompt_index"],
               body_request_id=plan.get("body_request_id"),
               request_body_sha256=body_hash)
    if plan["construction"]:
        row.update(
            constructed_target_chars=plan["construction"]["target_chars"],
            constructed_actual_chars=plan["construction"]["actual_chars"],
            constructed_error_chars=plan["construction"]["error_chars"])
    return row


def _assert_row_adapter_contract(rows: list[dict], ecfg: EndpointConfig) -> None:
    """Require every request row to claim the configured wire contract.

    Carried setup rows were emitted by the same command immediately before
    measured execution.  Missing identity is therefore not a harmless legacy
    case: accepting it would let a preflight performed with a different wire
    dialect authorize paid traffic.  Historical artifacts remain readable by
    their versioned verifier, but they cannot be imported as current setup
    evidence without an exact adapter and response-mode claim.
    """
    expected_adapter = ecfg.adapter
    expected_mode = get_endpoint_adapter(expected_adapter).response_mode
    for index, row in enumerate(rows):
        recorded_adapter = row.get("endpoint_adapter")
        recorded_mode = row.get("response_mode")
        if recorded_adapter != expected_adapter:
            raise ValueError(
                f"request row {index} endpoint_adapter does not match the "
                "configured endpoint adapter")
        if recorded_mode != expected_mode:
            raise ValueError(
                f"request row {index} response_mode does not match the "
                "configured endpoint adapter")


def _size_for_concurrency(rc: "RunConfig", ecfg, client, record,
                          quiet: bool, workload_id: str,
                          execution_id: str, *,
                          prevalidated_workload: _PreparedWorkload | None = None
                          ) -> "RunConfig":
    """Derive a fixed open-loop rate from an unloaded concurrency hint.

    This does not hold concurrency. It measures unloaded service time once,
    computes ``rate = sizing_concurrency / mean(e2e)`` by Little's Law, and
    leaves that rate
    fixed while the endpoint slows or speeds up under load.
    """
    import numpy as _np

    probe_n = sizing_probe_row_count(rc.calibrate_n)
    workload = (prevalidated_workload if prevalidated_workload is not None
                else _PreparedWorkload(rc, probe_n))
    if prevalidated_workload is not None \
            and workload.total_n != probe_n:
        raise ValueError(
            "prevalidated sizing workload does not match the sizing probe "
            "count")

    e2e = []
    for i in range(probe_n):
        body_rid = _stable_request_id(workload_id, i, "sizing-body")
        rid = _stable_request_id(execution_id, i, "sizing")
        plan = workload.plan(i, body_rid)
        body_hash = _client_payload_hash(
            client, ecfg, plan["messages"], plan["max_output"])
        try:
            res = _send_request(
                client,
                plan["messages"], plan["max_output"], rid, scheduled_s=0.0,
                dispatch_lag_ms=0.0, intended=plan["intended"],
                chars_sent=plan["chars"])
            d = _annotate_result(res, "sizing", plan, body_hash)
        except Exception as exc:
            d = _exception_result(
                rid, "sizing", plan, body_hash,
                f"unexpected worker exception: {type(exc).__name__}: {exc}",
                endpoint_adapter=ecfg.adapter)
        record(d)
        if _clean_measurement_row(d) and d.get("e2e_ms"):
            e2e.append(d["e2e_ms"])

    if len(e2e) != probe_n:
        raise RuntimeError(
            f"sizing pass got {len(e2e)} clean, complete responses from "
            f"{probe_n} probes, so the arrival rate for "
            f"sizing_concurrency {rc.sizing_concurrency} cannot be derived. "
            "check auth and "
            "the endpoint path, or set qps_base and max_concurrency directly.")

    mean = float(_np.mean(e2e)) / 1000.0
    p50 = float(_np.percentile(e2e, 50)) / 1000.0
    p95 = float(_np.percentile(e2e, 95)) / 1000.0
    # L = lambda * W uses mean residence time, not median. A skewed service
    # distribution can have a p50 far below its mean; dividing by p50 would
    # systematically offer too much load and overshoot the sizing hint.
    uncapped_rate = rc.sizing_concurrency / max(mean, 1e-3)
    # qps_max is the pretraffic sizing ceiling. The final arrival process is
    # thinned from its already materialized schedule, so a very fast sizing
    # response cannot create an unbounded post-paid workload.
    rate = min(uncapped_rate, float(rc.qps_max))
    validate_schedule_capacity(rc.duration_s, rate)
    derived_pool_size = max(rc.sizing_concurrency * 2,
                            int(math.ceil(rate * p95 * 1.5)))
    pool_cap = (rc.max_concurrency if rc.max_concurrency is not None
                else _DEFAULT_MAX_CONCURRENCY)
    pool_size = min(derived_pool_size, pool_cap)
    if not quiet:
        print(f"[runner] sizing from {len(e2e)} probe requests: e2e mean "
              f"{mean * 1000:.0f} ms, p50 {p50 * 1000:.0f} ms, "
              f"p95 {p95 * 1000:.0f} ms")
        print(f"[runner] sizing hint {rc.sizing_concurrency}: offering a fixed "
              f"{rate:.2f} rps with pool {pool_size}"
              + (f" (derived {derived_pool_size}, capped by explicit "
                 "max_concurrency)" if pool_size < derived_pool_size
                 and rc.max_concurrency is not None else "")
              + (f" (derived {derived_pool_size}, capped by the default "
                 f"{_DEFAULT_MAX_CONCURRENCY}-thread safety limit)"
                 if pool_size < derived_pool_size
                 and rc.max_concurrency is None else "")
              + (f" (uncapped sizing rate {uncapped_rate:.2f} rps was "
                 f"limited by configured qps_max={rc.qps_max:g})"
                 if uncapped_rate > rate else "")
              + "; concurrency is measured, not held")
    return dataclasses.replace(
        rc, qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, max_concurrency=pool_size)


class AuthProfileError(RuntimeError):
    """A named Databricks profile could not be resolved safely."""


def _auth_bytes_fingerprint(value: bytes) -> str:
    """Describe an auth response without exposing any response content."""
    return f"bytes={len(value)}, sha256={hashlib.sha256(value).hexdigest()}"


def _validated_bearer_token(value, *, source: str) -> str:
    """Return one header-safe bearer token without ever echoing its value."""
    try:
        return validate_bearer_token(value, source=source)
    except ValueError as exc:
        raise AuthProfileError(str(exc)) from None


def _validated_m2m_credential(value: str, *, field: str,
                              allow_colon: bool) -> str:
    """Validate one Basic-auth credential without putting it in an error."""
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        raise AuthProfileError(
            f"Databricks profile field {field} must contain printable ASCII") \
            from None
    if len(encoded) > _AUTH_CREDENTIAL_MAX_BYTES:
        raise AuthProfileError(
            f"Databricks profile field {field} is too large "
            f"(bytes={len(encoded)})")
    if any(byte < 0x21 or byte > 0x7e for byte in encoded):
        raise AuthProfileError(
            f"Databricks profile field {field} must contain printable ASCII")
    if not allow_colon and ":" in value:
        raise AuthProfileError(
            "Databricks profile field client_id must not contain ':'")
    return value


def _read_bounded_auth_response(response) -> bytes:
    """Read one OAuth response with a hard accepted-size ceiling."""
    length = response.getheader("Content-Length")
    if length is not None:
        if not isinstance(length, str) or not length.isascii() \
                or not length.isdigit():
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned a malformed "
                "Content-Length header")
        if int(length) > _AUTH_RESPONSE_MAX_BYTES:
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint response exceeded the "
                f"{_AUTH_RESPONSE_MAX_BYTES}-byte safety limit")
    raw = response.read(_AUTH_RESPONSE_MAX_BYTES + 1)
    if not isinstance(raw, bytes):
        raise AuthProfileError(
            "Databricks OAuth M2M token endpoint returned a non-byte response")
    if len(raw) > _AUTH_RESPONSE_MAX_BYTES:
        raise AuthProfileError(
            "Databricks OAuth M2M token endpoint response exceeded the "
            f"{_AUTH_RESPONSE_MAX_BYTES}-byte safety limit")
    return raw


def _mint_workspace_m2m_token(origin: tuple[str, str, int],
                              client_id: str, client_secret: str,
                              *, profile_name: str) -> str:
    """Mint a standard workspace-origin OAuth M2M all-apis token.

    This deliberately does not implement route-optimized serving auth. That
    Databricks flow requires endpoint-scoped ``authorization_details`` in
    addition to ``scope=all-apis``.
    """
    import base64
    import http.client
    import socket
    import ssl
    import urllib.parse

    scheme, host, port = origin
    if scheme != "https":
        raise AuthProfileError(
            f"Databricks OAuth M2M profile {profile_name!r} requires an "
            "HTTPS workspace host")
    client_id = _validated_m2m_credential(
        client_id, field="client_id", allow_colon=False)
    client_secret = _validated_m2m_credential(
        client_secret, field="client_secret", allow_colon=True)
    basic = base64.b64encode(
        f"{client_id}:{client_secret}".encode("ascii")).decode("ascii")
    body = urllib.parse.urlencode((
        ("grant_type", "client_credentials"),
        ("scope", "all-apis"),
    )).encode("ascii")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Connection": "close",
    }
    conn = None
    try:
        conn = http.client.HTTPSConnection(
            host, port, timeout=_AUTH_M2M_TIMEOUT_S,
            context=ssl.create_default_context())
        bind_deadline_bounded_dns(conn)
        with AbsoluteHTTPDeadline(
                conn, _AUTH_M2M_TIMEOUT_S) as deadline:
            conn.request("POST", "/oidc/v1/token", body=body, headers=headers)
            deadline.raise_if_expired()
            response = conn.getresponse()
            deadline.raise_if_expired()
            status = response.status
            if not isinstance(status, int) or isinstance(status, bool) \
                    or not 100 <= status <= 599:
                raise AuthProfileError(
                    "Databricks OAuth M2M token endpoint returned an invalid "
                    "HTTP status")
            raw = _read_bounded_auth_response(response)
            deadline.raise_if_expired()
            content_type = response.getheader("Content-Type")
        fingerprint = _auth_bytes_fingerprint(raw)
        if status != 200:
            hint = {
                400: "check the OAuth client-credentials profile fields",
                401: "check the service-principal client ID and OAuth secret",
                403: "check workspace assignment and service-principal access",
            }.get(status, "check the workspace host and OAuth configuration")
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned HTTP "
                f"{status} ({fingerprint}); {hint}")
        media_type = (
            content_type.split(";", 1)[0].strip().lower()
            if isinstance(content_type, str) else "")
        if media_type != "application/json":
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned a non-JSON "
                f"Content-Type ({fingerprint})")
        try:
            envelope = loads_strict(raw)
        except (UnicodeError, ValueError):
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned invalid JSON "
                f"({fingerprint})") from None
        if not isinstance(envelope, dict):
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned a non-object "
                f"JSON value ({fingerprint})")
        token_type = envelope.get("token_type")
        if not isinstance(token_type, str) \
                or token_type.casefold() != "bearer":
            raise AuthProfileError(
                "Databricks OAuth M2M token response did not declare Bearer "
                f"token_type ({fingerprint})")
        scope = envelope.get("scope")
        if scope is not None and scope != "all-apis":
            raise AuthProfileError(
                "Databricks OAuth M2M token response returned an unexpected "
                f"scope ({fingerprint})")
        expires_in = envelope.get("expires_in")
        if expires_in is not None and (
                not isinstance(expires_in, int)
                or isinstance(expires_in, bool) or expires_in <= 0):
            raise AuthProfileError(
                "Databricks OAuth M2M token response returned an invalid "
                f"expires_in ({fingerprint})")
        return _validated_bearer_token(
            envelope.get("access_token"),
            source=f"Databricks OAuth M2M profile {profile_name!r}")
    except AuthProfileError:
        raise
    except (TimeoutError, socket.timeout):
        raise AuthProfileError(
            "Databricks OAuth M2M token request timed out after "
            f"{_AUTH_M2M_TIMEOUT_S:g} seconds; check workspace reachability") \
            from None
    except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
        raise AuthProfileError(
            "Databricks OAuth M2M token request failed "
            f"({type(exc).__name__}); check the HTTPS workspace host and "
            "network reachability") from None
    finally:
        if conn is not None:
            try:
                conn.close()
            except (OSError, http.client.HTTPException):
                pass


class _CLIOutputLimitError(RuntimeError):
    """A subprocess crossed its bounded-capture envelope."""

    def __init__(self, captured: bytes):
        super().__init__("bounded CLI output limit exceeded")
        self.captured = captured


def _run_cli_bounded(command: list[str], *, env: dict[str, str],
                     timeout_s: float, max_stdout_bytes: int) \
        -> tuple[int, bytes]:
    """Run a CLI while draining at most ``max_stdout_bytes + 1`` bytes.

    stderr is discarded at the file-descriptor boundary, so neither memory
    nor exception text can accumulate it. stdout is drained incrementally;
    crossing the cap kills the child immediately instead of waiting for an
    unbounded ``communicate()`` capture to return.
    """
    import selectors
    import subprocess
    process = None
    selector = selectors.DefaultSelector()
    raw = bytearray()
    deadline = time.monotonic() + timeout_s
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env)
        if process.stdout is None:
            raise OSError("Databricks CLI stdout pipe was not created")
        stdout_fd = process.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        selector.register(stdout_fd, selectors.EVENT_READ)
        stdout_eof = False
        while not stdout_eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            events = selector.select(min(remaining, 0.25))
            if not events:
                continue
            for key, _mask in events:
                chunk = os.read(
                    key.fd,
                    min(8192, max_stdout_bytes + 1 - len(raw)))
                if not chunk:
                    stdout_eof = True
                    selector.unregister(key.fd)
                    break
                raw.extend(chunk)
                if len(raw) > max_stdout_bytes:
                    raise _CLIOutputLimitError(bytes(raw))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_s)
        returncode = process.wait(timeout=remaining)
        return returncode, bytes(raw)
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise
    finally:
        selector.close()
        if process is not None and process.stdout is not None:
            process.stdout.close()


def _mint_cli_u2m_token(name: str, cfg_path) -> str:
    """Mint only the named CLI U2M profile, with auth env fallback removed."""
    import subprocess

    cli_env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("DATABRICKS_")
    }
    cli_env["DATABRICKS_CONFIG_FILE"] = str(cfg_path)
    # This selects only storage, not an identity or credential source.
    if os.environ.get("DATABRICKS_AUTH_STORAGE"):
        cli_env["DATABRICKS_AUTH_STORAGE"] = \
            os.environ["DATABRICKS_AUTH_STORAGE"]
    command = ["databricks", "auth", "token", "-p", name]
    try:
        returncode, raw = _run_cli_bounded(
            command, env=cli_env, timeout_s=_AUTH_CLI_TIMEOUT_S,
            max_stdout_bytes=_AUTH_RESPONSE_MAX_BYTES)
    except subprocess.TimeoutExpired:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} token mint timed out after "
            f"{_AUTH_CLI_TIMEOUT_S:g} seconds; run 'databricks auth login "
            f"--profile {name}' interactively") from None
    except _CLIOutputLimitError as exc:
        fingerprint = _auth_bytes_fingerprint(exc.captured)
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} returned an oversized token "
            f"response ({fingerprint})") from None
    except OSError as exc:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} could not invoke the "
            f"Databricks CLI ({type(exc).__name__}); install the CLI and run "
            f"'databricks auth login --profile {name}'") from None
    if returncode != 0:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} token mint exited with status "
            f"{returncode}; run 'databricks auth login --profile "
            f"{name}' interactively")
    fingerprint = _auth_bytes_fingerprint(raw)
    try:
        envelope = loads_strict(raw)
    except (UnicodeError, ValueError):
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} returned invalid token JSON "
            f"({fingerprint})") from None
    if not isinstance(envelope, dict):
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} returned non-object token JSON "
            f"({fingerprint})")
    return _validated_bearer_token(
        envelope.get("access_token"),
        source=f"Databricks U2M profile {name!r}")


def _token_from_profile(name: str, endpoint_base_url: str) -> str:
    """Resolve a ~/.databrickscfg profile to a bearer token.

    PAT values are read directly; ``databricks-cli`` explicitly selects U2M;
    complete ``client_id`` / ``client_secret`` profiles select workspace OAuth
    M2M. Before any credential value is read or used, the profile's configured
    origin is normalized and required to match the request endpoint. Named
    profiles fail closed: there is no environment credential fallback.
    """
    import configparser
    from pathlib import Path

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE",
                                   Path.home() / ".databrickscfg"))
    # Databricks calls [DEFAULT] a real named profile. Python ConfigParser
    # otherwise treats it as inherited defaults and can silently copy one
    # profile's credential into every other section.
    parser = configparser.ConfigParser(
        interpolation=None, default_section=_AUTH_DISABLED_DEFAULT_SECTION)
    try:
        read = parser.read(cfg_path)
    except (OSError, configparser.Error) as exc:
        raise AuthProfileError(
            f"could not read Databricks config {cfg_path} "
            f"({type(exc).__name__}); check its syntax and permissions") \
            from None
    if not read:
        raise AuthProfileError(f"Databricks config not found: {cfg_path}")
    if parser.defaults():
        raise AuthProfileError(
            "Databricks config uses a reserved defaults section; rename "
            f"[{_AUTH_DISABLED_DEFAULT_SECTION}]")
    if not parser.has_section(name):
        raise AuthProfileError(f"Databricks auth profile {name!r} does not exist")

    sect = parser[name]
    profile_host = (sect.get("host") or "").strip()
    if not profile_host:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} has no configured host")
    try:
        profile_origin = validate_bearer_transport(profile_host)
    except ValueError as exc:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} has an invalid or unsafe "
            f"host ({type(exc).__name__}); configure an HTTPS workspace "
            "origin without a path, query, fragment, or userinfo") from None
    try:
        endpoint_origin = validate_bearer_transport(endpoint_base_url)
    except ValueError as exc:
        raise AuthProfileError(
            "the endpoint has an invalid or unsafe base_url "
            f"({type(exc).__name__}); configure an HTTPS origin without a "
            "path, query, fragment, or userinfo") from None
    if profile_origin != endpoint_origin:
        def _display(origin):
            scheme, host, port = origin
            default = 443 if scheme == "https" else 80
            return f"{scheme}://{host}" + (f":{port}" if port != default else "")
        raise AuthProfileError(
            f"Databricks auth profile {name!r} is bound to "
            f"{_display(profile_origin)}, not {_display(endpoint_origin)}")

    def _field(key: str) -> str | None:
        if key not in sect:
            return None
        value = sect.get(key)
        if not isinstance(value, str) or not value:
            raise AuthProfileError(
                f"Databricks auth profile {name!r} field {key} must be "
                "non-empty")
        if any(char in value for char in ("\r", "\n", "\x00")):
            raise AuthProfileError(
                f"Databricks auth profile {name!r} field {key} contains "
                "control characters")
        return value

    auth_type = _field("auth_type")
    token = _field("token")
    client_id = _field("client_id")
    client_secret = _field("client_secret")
    unsupported_fields = tuple(
        field for field in (
            "account_id", "username", "password", "azure_client_id",
            "azure_client_secret", "azure_tenant_id", "azure_use_msi",
            "azure_workspace_resource_id", "google_credentials",
            "google_service_account", "oidc_token_env",
            "oidc_token_filepath",
        ) if field in sect and sect.get(field))
    if unsupported_fields:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} uses unsupported workspace "
            "authentication field(s): " + ", ".join(unsupported_fields))

    supported = {None, "pat", "databricks-cli", "oauth-m2m"}
    if auth_type not in supported:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} has unsupported auth_type; "
            "supported values are pat, databricks-cli, and oauth-m2m")
    has_token = token is not None
    has_client_id = client_id is not None
    has_client_secret = client_secret is not None
    has_any_client_credential = has_client_id or has_client_secret
    has_complete_client_credentials = has_client_id and has_client_secret
    if has_token and has_any_client_credential:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} mixes a PAT token with OAuth "
            "client credentials; use one authentication method per profile")
    if has_any_client_credential and not has_complete_client_credentials:
        missing = "client_secret" if has_client_id else "client_id"
        raise AuthProfileError(
            f"Databricks auth profile {name!r} has incomplete OAuth M2M "
            f"credentials; add {missing}")

    if auth_type == "pat":
        if not has_token:
            raise AuthProfileError(
                f"Databricks PAT profile {name!r} requires a token field")
        return _validated_bearer_token(
            token, source=f"Databricks PAT profile {name!r}")
    if auth_type == "databricks-cli":
        if has_token or has_any_client_credential:
            raise AuthProfileError(
                f"Databricks U2M profile {name!r} must not contain PAT or "
                "OAuth M2M credential fields")
        return _mint_cli_u2m_token(name, cfg_path)
    if auth_type == "oauth-m2m":
        if not has_complete_client_credentials:
            raise AuthProfileError(
                f"Databricks OAuth M2M profile {name!r} requires client_id "
                "and client_secret")
        return _mint_workspace_m2m_token(
            profile_origin, client_id, client_secret, profile_name=name)

    # Official M2M profile examples omit auth_type, while U2M is accepted only
    # when it explicitly declares databricks-cli. Host-only profiles must not
    # silently ask the CLI (and thereby select an environment credential).
    if has_token:
        return _validated_bearer_token(
            token, source=f"Databricks PAT profile {name!r}")
    if has_complete_client_credentials:
        return _mint_workspace_m2m_token(
            profile_origin, client_id, client_secret, profile_name=name)
    raise AuthProfileError(
        f"Databricks auth profile {name!r} has no supported credentials; "
        "add token, add client_id/client_secret, or explicitly set "
        "auth_type=databricks-cli for interactive U2M")


def _token(cfg: EndpointConfig) -> str | None:
    if cfg.auth_profile:
        return _token_from_profile(cfg.auth_profile, cfg.base_url)
    tok = os.environ.get(cfg.auth_token_env) or None
    if tok:
        validate_bearer_transport(cfg.base_url)
        return _validated_bearer_token(
            tok, source=f"environment variable {cfg.auth_token_env}")
    return None


_REASONING_PROBE_SCHEMA = "reasoning-control-probe-evidence/v1"
_REASONING_PROBE_REQUIRED_FIELDS = {
    "schema_version", "candidate_index", "candidate_redacted",
    "candidate_canonical_sha256", "disposition", "evidence_method",
    "effective_status", "effective_value", "request_id",
    "logical_request_body_sha256", "physical_request_body_sha256s",
}
_REASONING_PROBE_METHODS = {
    "accepted": {"single_request_behavior_observation"},
    "rejected": {"request_validation_response"},
    "unknown": {"non_validation_http_failure", "transport_outcome_unknown"},
}


def _reasoning_probe_candidate_sha256(value: dict) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validated_reasoning_probe_candidate(value: object, *, label: str) -> dict:
    """Independently bound persisted candidates without revealing content."""
    from .client import validate_extra_body_safety

    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label}.candidate_redacted must be a non-empty object")
    try:
        validate_extra_body_safety(value)
    except ValueError as exc:
        raise ValueError(
            f"{label}.candidate_redacted contains unsafe material") from exc
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 256 or depth > 8:
            raise ValueError(f"{label}.candidate_redacted exceeds v1 limits")
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, int):
            if not -(2 ** 63) <= item <= 2 ** 63 - 1:
                raise ValueError(
                    f"{label}.candidate_redacted integer exceeds int64")
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(
                    f"{label}.candidate_redacted number is not finite")
            continue
        if isinstance(item, str):
            if len(item.encode("utf-8")) > 4096 \
                    or any(ord(char) < 0x20 or ord(char) == 0x7f
                           for char in item):
                raise ValueError(
                    f"{label}.candidate_redacted string is invalid")
            continue
        if isinstance(item, list):
            if len(item) > 64:
                raise ValueError(
                    f"{label}.candidate_redacted array exceeds v1 limits")
            stack.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, dict):
            if len(item) > 64:
                raise ValueError(
                    f"{label}.candidate_redacted object exceeds v1 limits")
            for key, child in item.items():
                if not isinstance(key, str) or not key \
                        or len(key.encode("utf-8")) > 128 \
                        or any(ord(char) < 0x20 or ord(char) == 0x7f
                               for char in key):
                    raise ValueError(
                        f"{label}.candidate_redacted key is invalid")
                stack.append((child, depth + 1))
            continue
        raise ValueError(
            f"{label}.candidate_redacted contains a non-JSON value")
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    if len(raw) > 16 * 1024:
        raise ValueError(
            f"{label}.candidate_redacted exceeds v1 byte limit")
    return copy.deepcopy(value)


def _validated_reasoning_control_probe_evidence(
        value: object, *, row: dict | None = None,
        label: str = "reasoning_control_probe") -> dict:
    """Validate a v1 probe envelope and its logical/physical body links."""
    if not isinstance(value, dict) \
            or set(value) != _REASONING_PROBE_REQUIRED_FIELDS:
        raise ValueError(f"{label} has unknown or missing fields")
    if value.get("schema_version") != _REASONING_PROBE_SCHEMA:
        raise ValueError(f"{label}.schema_version is unsupported")
    index = value.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) \
            or not 1 <= index <= 16:
        raise ValueError(f"{label}.candidate_index is invalid")
    candidate = _validated_reasoning_probe_candidate(value.get(
        "candidate_redacted"), label=label)
    digest = value.get("candidate_canonical_sha256")
    if not isinstance(digest, str) or len(digest) != 64 \
            or any(char not in "0123456789abcdef" for char in digest) \
            or digest != _reasoning_probe_candidate_sha256(candidate):
        raise ValueError(f"{label}.candidate_canonical_sha256 is invalid")
    disposition = value.get("disposition")
    method = value.get("evidence_method")
    if disposition not in _REASONING_PROBE_METHODS \
            or method not in _REASONING_PROBE_METHODS[disposition]:
        raise ValueError(f"{label} disposition/evidence_method is invalid")
    expected_effective = (
        "not_applied_request_rejected"
        if disposition == "rejected" else "unknown")
    if value.get("effective_status") != expected_effective \
            or value.get("effective_value") is not None:
        raise ValueError(f"{label} makes an unsupported effective-value claim")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id \
            or len(request_id.encode("utf-8")) > 256 \
            or any(ord(char) < 0x21 or ord(char) > 0x7e
                   for char in request_id):
        raise ValueError(f"{label}.request_id is invalid")

    def digest_ok(item: object) -> bool:
        return isinstance(item, str) and len(item) == 64 \
            and all(char in "0123456789abcdef" for char in item)

    logical = value.get("logical_request_body_sha256")
    physical = value.get("physical_request_body_sha256s")
    if not digest_ok(logical) or not isinstance(physical, list) \
            or len(physical) > 5 or not all(digest_ok(item) for item in physical):
        raise ValueError(f"{label} body SHA-256 evidence is invalid")
    if row is not None:
        if row.get("phase") != "probe" \
                or row.get("request_id") != request_id \
                or row.get("request_body_sha256") != logical \
                or row.get("physical_request_body_sha256s", []) != physical:
            raise ValueError(f"{label} body/request links disagree with row")
        attempts = row.get("request_attempts")
        if attempts is not None and (
                isinstance(attempts, bool) or not isinstance(attempts, int)
                or attempts < 0 or len(physical) != attempts):
            raise ValueError(f"{label} physical hashes disagree with attempts")
    return copy.deepcopy(value)


def _prepare_prior_request_rows(value) -> list[dict]:
    """Validate metadata-only request rows produced by CLI preflight.

    These requests happened before ``run`` was entered, but they still used
    endpoint quota.  The rows are copied into the sealed journal so quota
    evidence never depends on an unauthenticated side channel.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("prior_request_rows must be a list of objects")
    allowed = {field.name for field in dataclasses.fields(RequestResult)} | {
        "phase", "global_index", "sample_index", "prompt_index",
        "body_request_id", "request_body_sha256",
        "constructed_target_chars", "constructed_actual_chars",
        "constructed_error_chars", "reasoning_control_probe",
        "setup_request_binding", "setup_request_binding_sha256",
    }
    timestamp_fields = {
        "first_send_unix", "t_send_unix", "first_attempt_unix",
        "finished_unix",
    }
    count_fields = {
        "request_attempts", "connection_attempts", "retries",
        "prompt_tokens", "completion_tokens", "max_tokens_requested",
        "cached_tokens", "reasoning_tokens", "parse_errors",
    }
    prepared = []
    for index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"prior_request_rows[{index}] must be an object")
        row = copy.deepcopy(candidate)
        if row.get("phase") not in {"preflight", "probe"}:
            raise ValueError(
                f"prior_request_rows[{index}].phase must be preflight or "
                "probe")
        if not isinstance(row.get("request_id"), str) \
                or not row["request_id"]:
            raise ValueError(
                f"prior_request_rows[{index}].request_id must be a "
                "non-empty string")
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(
                f"prior_request_rows[{index}] has unknown metadata field: "
                + ", ".join(unknown))
        probe_evidence = row.get("reasoning_control_probe")
        if probe_evidence is not None:
            if row.get("phase") != "probe":
                raise ValueError(
                    f"prior_request_rows[{index}].reasoning_control_probe "
                    "is only valid for a probe row")
            _validated_reasoning_control_probe_evidence(
                probe_evidence, row=row,
                label=f"prior_request_rows[{index}]")
        for name in timestamp_fields:
            item = row.get(name)
            if item is None:
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"prior_request_rows[{index}].{name} must be a finite "
                    "non-negative number or null")
            try:
                finite = math.isfinite(float(item))
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite or item < 0:
                raise ValueError(
                    f"prior_request_rows[{index}].{name} must be a finite "
                    "non-negative number or null")
        for name in count_fields:
            item = row.get(name)
            if item is None:
                continue
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise ValueError(
                    f"prior_request_rows[{index}].{name} must be a "
                    "non-negative integer or null")
            try:
                finite = math.isfinite(float(item))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(
                    f"prior_request_rows[{index}].{name} is too large")
        if row.get("max_tokens_requested") == 0:
            raise ValueError(
                f"prior_request_rows[{index}].max_tokens_requested must be "
                "positive or null")
        status = row.get("status")
        if status is not None and (
                not isinstance(status, int) or isinstance(status, bool)
                or not 100 <= status <= 599):
            raise ValueError(
                f"prior_request_rows[{index}].status must be an HTTP status "
                "integer or null")
        attempts = row.get("request_attempts")
        connections = row.get("connection_attempts")
        if attempts is not None and connections is None:
            raise ValueError(
                f"prior_request_rows[{index}].connection_attempts is required "
                "when request_attempts is known")
        if attempts is not None and connections is not None \
                and attempts > connections:
            raise ValueError(
                f"prior_request_rows[{index}].request_attempts cannot exceed "
                "connection_attempts")
        retry_reasons = row.get("retry_reasons")
        if retry_reasons is not None and (
                not isinstance(retry_reasons, list)
                or any(not isinstance(reason, str) or not reason
                       for reason in retry_reasons)):
            raise ValueError(
                f"prior_request_rows[{index}].retry_reasons must be a list "
                "of non-empty strings")
        retries = row.get("retries")
        if attempts is not None and (retries is None or retry_reasons is None):
            raise ValueError(
                f"prior_request_rows[{index}] needs retries and retry_reasons "
                "when request_attempts is known")
        if retries is not None and retry_reasons is not None \
                and retries != len(retry_reasons):
            raise ValueError(
                f"prior_request_rows[{index}].retries must equal the number "
                "of retry_reasons")
        body_hashes = row.get("physical_request_body_sha256s")
        if body_hashes is not None and (
                not isinstance(body_hashes, list)
                or any(not isinstance(digest, str) or len(digest) != 64
                       or any(char not in "0123456789abcdef"
                              for char in digest)
                       for digest in body_hashes)):
            raise ValueError(
                f"prior_request_rows[{index}]."
                "physical_request_body_sha256s must be a list of lowercase "
                "SHA-256 hex digests")
        if attempts is not None and body_hashes is not None \
                and len(body_hashes) != attempts:
            raise ValueError(
                f"prior_request_rows[{index}]."
                "physical_request_body_sha256s count must equal "
                "request_attempts")
        if attempts == 0:
            sent_only = {
                name: row.get(name) for name in (
                    "first_send_unix", "t_send_unix", "status",
                    "prompt_tokens", "completion_tokens", "cached_tokens",
                    "reasoning_tokens")
                if row.get(name) is not None
            }
            if row.get("ok") is True:
                sent_only["ok"] = True
            if row.get("stream_complete") is True:
                sent_only["stream_complete"] = True
            if sent_only:
                raise ValueError(
                    f"prior_request_rows[{index}] claims zero request_attempts "
                    "but contains sent-request evidence: "
                    + ", ".join(sorted(sent_only)))
        elif attempts is not None and attempts > 0:
            if row.get("first_send_unix") is None \
                    or row.get("t_send_unix") is None:
                raise ValueError(
                    f"prior_request_rows[{index}] with request_attempts > 0 "
                    "must include first_send_unix and t_send_unix")
        first_attempt = row.get("first_attempt_unix")
        if connections is not None and connections > 0 \
                and first_attempt is None:
            raise ValueError(
                f"prior_request_rows[{index}] with connection_attempts > 0 "
                "must include first_attempt_unix")
        if connections == 0 and first_attempt is not None:
            raise ValueError(
                f"prior_request_rows[{index}] with connection_attempts == 0 "
                "cannot include first_attempt_unix")
        prompt_tokens = row.get("prompt_tokens")
        completion_tokens = row.get("completion_tokens")
        cached_tokens = row.get("cached_tokens")
        reasoning_tokens = row.get("reasoning_tokens")
        if cached_tokens is not None and (
                prompt_tokens is None or cached_tokens > prompt_tokens):
            raise ValueError(
                f"prior_request_rows[{index}].cached_tokens cannot exceed "
                "prompt_tokens")
        if reasoning_tokens is not None and (
                completion_tokens is None
                or reasoning_tokens > completion_tokens):
            raise ValueError(
                f"prior_request_rows[{index}].reasoning_tokens cannot exceed "
                "completion_tokens")
        sent = row.get("first_send_unix")
        last_sent = row.get("t_send_unix")
        finished = row.get("finished_unix")
        if sent is not None and last_sent is not None and last_sent < sent:
            raise ValueError(
                f"prior_request_rows[{index}].t_send_unix cannot precede "
                "first_send_unix")
        if last_sent is not None and finished is not None \
                and finished < last_sent:
            raise ValueError(
                f"prior_request_rows[{index}].finished_unix cannot precede "
                "t_send_unix")
        if sent is not None and finished is not None and finished < sent:
            raise ValueError(
                f"prior_request_rows[{index}].finished_unix cannot precede "
                "first_send_unix")
        # Validate the row's own scalar and cross-field invariants before its
        # envelope.  The binding still fails closed, but malformed rows retain
        # precise and deterministic diagnostics instead of every error being
        # hidden behind a missing-binding message.
        binding = row.get("setup_request_binding")
        binding_digest = row.get("setup_request_binding_sha256")
        if not isinstance(binding, dict) or not _is_sha256(binding_digest) \
                or _binding_sha256(binding) != binding_digest:
            raise ValueError(
                f"prior_request_rows[{index}] lacks a valid exact setup "
                "request binding; legacy carried rows fail closed")
        required_binding = {
            "schema_version", "phase", "position", "trace_request_id",
            "endpoint_request", "representative",
            "physical_request_body_sha256s", "physical_hash_status",
        }
        if set(binding) != required_binding \
                or binding.get("schema_version") != \
                _SETUP_REQUEST_BINDING_SCHEMA \
                or binding.get("phase") != row.get("phase") \
                or binding.get("trace_request_id") != row.get("request_id") \
                or binding.get("physical_request_body_sha256s") != \
                list(row.get("physical_request_body_sha256s") or []) \
                or not isinstance(binding.get("endpoint_request"), dict) \
                or not isinstance(binding.get("representative"), dict):
            raise ValueError(
                f"prior_request_rows[{index}] setup request binding "
                "disagrees with its row")
        if binding["representative"].get(
                "logical_request_body_sha256") != \
                row.get("request_body_sha256"):
            raise ValueError(
                f"prior_request_rows[{index}] setup request binding has a "
                "different logical body hash")
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"prior_request_rows[{index}] must be finite JSON metadata") \
                from exc
        prepared.append(redact_secrets(row))
    return prepared


def _validated_preflight_gate(value, prior_rows: list[dict]) -> dict | None:
    """Validate command-level preflight state carried into a measured run."""
    if value is None:
        if prior_rows:
            raise ValueError(
                "carried setup rows require their exact preflight gate")
        return None
    if not isinstance(value, dict):
        raise ValueError("preflight_gate must be an object")
    required = {
        "skipped", "attempted", "reachable", "readable",
        "reasoning_probe_requests", "outcome", "force_requested",
        "gate_satisfied", "evidence_mode", "binding", "binding_sha256",
    }
    optional = {"reasoning_control_probes"}
    if not required.issubset(value) or set(value) - required - optional:
        raise ValueError("preflight_gate has unknown or missing fields")
    if value.get("skipped") is not False:
        raise ValueError("a carried preflight_gate cannot be skipped")
    if value.get("evidence_mode") not in {
            "carried_setup_rows", "inherited_setup_artifact"}:
        raise ValueError("preflight_gate evidence_mode is invalid")
    binding = value.get("binding")
    if not isinstance(binding, dict) \
            or binding.get("schema_version") != _PREFLIGHT_BINDING_SCHEMA \
            or not _is_sha256(value.get("binding_sha256")) \
            or _binding_sha256(binding) != value.get("binding_sha256"):
        raise ValueError(
            "preflight_gate lacks a valid cryptographic request binding; "
            "legacy carried gates fail closed")
    if not isinstance(value.get("force_requested"), bool) \
            or not isinstance(value.get("gate_satisfied"), bool):
        raise ValueError("preflight_gate boolean fields are invalid")
    counts = {}
    for field in ("attempted", "reachable", "readable",
                  "reasoning_probe_requests"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"preflight_gate.{field} must be non-negative")
        counts[field] = item
    if counts["attempted"] <= 0 \
            or counts["reachable"] > counts["attempted"] \
            or counts["readable"] > counts["reachable"]:
        raise ValueError("preflight_gate counts disagree")
    preflight_rows = sum(row.get("phase") == "preflight"
                         for row in prior_rows)
    probe_rows = sum(row.get("phase") == "probe" for row in prior_rows)
    carried = value["evidence_mode"] == "carried_setup_rows"
    if carried and (
            preflight_rows != counts["attempted"]
            or probe_rows != counts["reasoning_probe_requests"]):
        raise ValueError(
            "preflight_gate counts disagree with prior_request_rows")
    if not carried and prior_rows:
        raise ValueError(
            "an inherited preflight gate cannot duplicate setup rows")
    has_probe_evidence = "reasoning_control_probes" in value
    row_probe_evidence = [
        row.get("reasoning_control_probe")
        for row in prior_rows if row.get("phase") == "probe"
    ]
    if has_probe_evidence and carried:
        supplied = value.get("reasoning_control_probes")
        if not isinstance(supplied, list) \
                or len(supplied) != counts["reasoning_probe_requests"]:
            raise ValueError(
                "preflight_gate reasoning-control evidence count disagrees")
        validated = [
            _validated_reasoning_control_probe_evidence(
                item, row=row,
                label=f"preflight_gate.reasoning_control_probes[{index}]")
            for index, (item, row) in enumerate(
                zip(supplied,
                    [row for row in prior_rows
                     if row.get("phase") == "probe"], strict=True))
        ]
        if supplied != validated or row_probe_evidence != validated \
                or [item["candidate_index"] for item in validated] != \
                list(range(1, len(validated) + 1)):
            raise ValueError(
                "preflight_gate reasoning-control evidence is inconsistent")
    elif carried and any(item is not None for item in row_probe_evidence):
        raise ValueError(
            "preflight_gate is missing carried reasoning-control evidence")
    outcome = value.get("outcome")
    if outcome == "preflight_passed":
        valid = (counts["reachable"] == counts["attempted"]
                 and counts["readable"] == counts["attempted"]
                 and value["gate_satisfied"] is True)
    elif outcome == "preflight_forced_unreadable":
        valid = (value["force_requested"] is True
                 and value["gate_satisfied"] is False
                 and counts["reachable"] == counts["attempted"]
                 and counts["readable"] < counts["attempted"])
    else:
        # Reachability failures are refused even with --force. Unknown,
        # skipped, and refused states must never enter measured execution.
        valid = False
    if not valid:
        raise ValueError(
            "preflight_gate outcome does not authorize measured execution")
    return copy.deepcopy(value)


def _preflight_binding_for_rows(rc: RunConfig,
                                representative_plans: list[dict],
                                rows: list[dict]) -> tuple[dict, str]:
    """Create the gate-level digest over execution intent and setup rows."""
    setup_requests = []
    for row in rows:
        setup_requests.append({
            "phase": row.get("phase"),
            "trace_request_id": row.get("request_id"),
            "setup_request_binding_sha256": row.get(
                "setup_request_binding_sha256"),
            "logical_request_body_sha256": row.get(
                "request_body_sha256"),
            "physical_request_body_sha256s": list(
                row.get("physical_request_body_sha256s") or []),
        })
    binding = {
        "schema_version": _PREFLIGHT_BINDING_SCHEMA,
        "execution": _preflight_execution_binding(
            rc, representative_plans),
        "setup_requests": setup_requests,
    }
    return binding, _binding_sha256(binding)


def _validated_setup_artifact_reference(value: object,
                                        preflight_gate: dict | None) \
        -> dict | None:
    if value is None:
        if isinstance(preflight_gate, dict) and preflight_gate.get(
                "evidence_mode") == "inherited_setup_artifact":
            raise ValueError(
                "inherited preflight evidence requires a setup artifact "
                "reference")
        return None
    if not isinstance(value, dict):
        raise ValueError("setup_artifact_reference must be an object")
    required = {
        "schema_version", "artifact_id", "execution_id", "workload_id",
        "manifest_sha256", "manifest_bytes", "preflight_binding_sha256",
    }
    if set(value) != required \
            or value.get("schema_version") != \
            _SETUP_ARTIFACT_REFERENCE_SCHEMA \
            or any(not isinstance(value.get(field), str)
                   or not value[field]
                   or len(value[field]) > 160
                   or any(ord(char) < 0x21 or ord(char) > 0x7e
                          for char in value[field])
                   for field in ("artifact_id", "execution_id",
                                 "workload_id")) \
            or not _is_sha256(value.get("manifest_sha256")) \
            or not _is_sha256(value.get("preflight_binding_sha256")) \
            or not isinstance(value.get("manifest_bytes"), int) \
            or isinstance(value.get("manifest_bytes"), bool) \
            or value["manifest_bytes"] <= 0:
        raise ValueError("setup_artifact_reference is invalid")
    if not isinstance(preflight_gate, dict) \
            or value["preflight_binding_sha256"] != \
            preflight_gate.get("binding_sha256"):
        raise ValueError(
            "setup artifact reference does not bind this preflight gate")
    return copy.deepcopy(value)


def _merge_binding_candidate(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_binding_candidate(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _assert_carried_preflight_binding(
        rc: RunConfig, representative_plans: list[dict],
        prior_rows: list[dict], gate: dict | None) -> None:
    """Recompute the exact setup authorization before credentials or POSTs."""
    if gate is None:
        if prior_rows:
            raise ValueError("carried setup rows have no preflight binding")
        return
    binding = gate["binding"]
    expected_execution = _preflight_execution_binding(
        rc, representative_plans)
    if binding.get("execution") != expected_execution:
        raise ValueError(
            "carried preflight binding does not match this exact endpoint, "
            "request configuration, output cap, or workload")
    if gate["evidence_mode"] == "inherited_setup_artifact":
        return
    linked = binding.get("setup_requests")
    if not isinstance(linked, list) or len(linked) != len(prior_rows):
        raise ValueError(
            "carried preflight binding row population is inconsistent")
    plans_by_index = {
        (plan.get("global_index"), plan.get("sample_index"),
         plan.get("prompt_index")): plan
        for plan in representative_plans
    }
    base_endpoint = copy.deepcopy(rc.endpoint)
    for index, (row, link) in enumerate(zip(prior_rows, linked, strict=True)):
        expected_link = {
            "phase": row.get("phase"),
            "trace_request_id": row.get("request_id"),
            "setup_request_binding_sha256": row.get(
                "setup_request_binding_sha256"),
            "logical_request_body_sha256": row.get(
                "request_body_sha256"),
            "physical_request_body_sha256s": list(
                row.get("physical_request_body_sha256s") or []),
        }
        if link != expected_link:
            raise ValueError(
                f"carried preflight binding row {index} link disagrees")
        row_binding = row["setup_request_binding"]
        rep = row_binding["representative"]
        key = (rep.get("global_index"), rep.get("sample_index"),
               rep.get("prompt_index"))
        plan = plans_by_index.get(key)
        if plan is None:
            raise ValueError(
                f"carried preflight row {index} is not one of this "
                "workload's representatives")
        exact_endpoint = copy.deepcopy(base_endpoint)
        if row.get("phase") == "probe":
            evidence = row.get("reasoning_control_probe")
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"carried probe row {index} has no candidate evidence")
            exact_endpoint["extra_body"] = _merge_binding_candidate(
                exact_endpoint.get("extra_body") or {},
                evidence["candidate_redacted"])
        elif row_binding["endpoint_request"].get("include_usage", {}).get(
                "value") is True and EndpointConfig(
                    **exact_endpoint).include_usage is False:
            # The only accepted setup-to-execution request transform is the
            # adapter-observed include_usage rejection: the setup row binds
            # the attempted true value and the gate binds execution false.
            exact_endpoint["include_usage"] = True
        plan_for_row = copy.deepcopy(plan)
        plan_for_row["max_output"] = int(rep.get("max_tokens"))
        rebuilt = _setup_request_binding(
            endpoint=exact_endpoint,
            max_output_tokens_cap=rc.max_output_tokens_cap,
            plan=plan_for_row, phase=row["phase"],
            position=int(row_binding.get("position")),
            request_id=row["request_id"], row=row)
        if rebuilt != row_binding \
                or _binding_sha256(rebuilt) != \
                row["setup_request_binding_sha256"]:
            raise ValueError(
                f"carried preflight row {index} does not match the exact "
                "endpoint/config/workload binding")


def run(rc: RunConfig, token_override: str | None = None,
        quiet: bool = False, prior_request_rows=None,
        runtime_quota_guard=None, preflight_gate=None,
        setup_artifact_reference=None) -> dict:
    # Freeze all nested request/policy configuration and re-run validation in
    # case a caller mutated the dataclass after constructing it.
    rc = dataclasses.replace(
        rc,
        endpoint=copy.deepcopy(rc.endpoint),
        acceptance_targets=copy.deepcopy(rc.acceptance_targets),
        pricing=copy.deepcopy(rc.pricing),
        rate_limits=copy.deepcopy(rc.rate_limits),
        input_expectations=copy.deepcopy(rc.input_expectations))
    prior_rows = _prepare_prior_request_rows(prior_request_rows)
    preflight_gate = _validated_preflight_gate(preflight_gate, prior_rows)
    setup_artifact_reference = _validated_setup_artifact_reference(
        setup_artifact_reference, preflight_gate)
    prompts_mode = bool(rc.prompts_file)
    if prompts_mode and rc.profile_path:
        raise ValueError("set profile_path or prompts_file, not both")
    if not prompts_mode and not rc.profile_path:
        raise ValueError("set profile_path (synthetic shape) or "
                         "prompts_file (real prompt text)")
    if rc.start_at_unix is not None \
            and time.time() > rc.start_at_unix + rc.start_tolerance_s:
        raise ValueError(
            f"start_at_unix is stale by {time.time() - rc.start_at_unix:.3f}s; "
            "use a future shared start and synchronize shard clocks")
    ecfg = EndpointConfig(**rc.endpoint)
    # Carried CLI preflight/probe evidence must authorize this exact wire
    # contract.  Reject it before source snapshots, credential lookup,
    # metadata discovery, quota reservation, or any measured request.
    _assert_row_adapter_contract(prior_rows, ecfg)
    if token_override is not None and ecfg.auth_profile:
        raise AuthProfileError(
            "token_override cannot be combined with a named auth_profile")

    original_rc = rc
    sizing_requested = rc.sizing_concurrency
    sizing_local = _shard_concurrency(rc)
    load_mode = ("sizing_concurrency" if sizing_requested is not None
                 else "fixed_rate")
    run_started_at = time.time()
    source = snapshot_source_state(Path(__file__).parent)

    # A private snapshot is the only input parsed below. If the source profile,
    # prompts, or trace changes while a long run is active, request bodies and
    # schedule remain tied to the hashes captured in start.json.
    with tempfile.TemporaryDirectory(prefix="traffic-replay-inputs-") as tmp:
        work_rc, inputs = _snapshot_run_inputs(rc, Path(tmp))
        _enforce_input_expectations(rc, inputs)
        workload_id = _resolved_workload_id(original_rc, inputs)
        logical_run_id, execution_id, artifact_id = _execution_ids(original_rc)
        # Parse and construct the exact private snapshots once.  Quota planning
        # and execution below share these objects, so the safety gate cannot
        # authorize a different schedule or workload realization from the one
        # that is eventually sent.
        prevalidated = prevalidate_run_inputs(work_rc)
        # Recompute the complete setup authorization from the exact private
        # workload snapshot. This is deliberately before token lookup,
        # metadata/network probes, artifact creation, or measured inference.
        _assert_carried_preflight_binding(
            work_rc, prevalidated.representative_plans,
            prior_rows, preflight_gate)
        enforce_exact_analysis_envelope(
            prevalidated, setup_rows=len(prior_rows),
            context="run including carried setup traffic")
        # This gate is intentionally before token lookup, endpoint metadata,
        # network measurement, sizing, calibration, or replay.  A quota-aware
        # config that cannot be bounded must not spend inference traffic in
        # order to discover that fact.
        from .quota_planner import (
            enforce_quota_plan,
            plan_run_quota,
            render_quota_plan,
            runtime_quota_scope_material,
        )
        quota_plan = plan_run_quota(
            work_rc, prior_rows=prior_rows, prevalidated=prevalidated)
        if quota_plan is not None and not quota_plan.get("may_start") \
                and not quiet:
            print(render_quota_plan(quota_plan))
        enforce_quota_plan(quota_plan)
        if work_rc.rate_limits is not None:
            from .quota_planner import RuntimeQuotaGuard
            guard_scope = runtime_quota_scope_material(
                work_rc.rate_limits, work_rc.endpoint)
            if runtime_quota_guard is None:
                runtime_quota_guard = RuntimeQuotaGuard(
                    work_rc.rate_limits,
                    shard_index=work_rc.shard_index,
                    shard_total=work_rc.shard_total,
                    scope_material=guard_scope)
            elif not runtime_quota_guard.matches(
                    work_rc.rate_limits,
                    shard_index=work_rc.shard_index,
                    shard_total=work_rc.shard_total,
                    scope_material=guard_scope):
                raise ValueError(
                    "runtime quota guard does not match this run's rate "
                    "limits or shard allocation")
            if prior_rows:
                # A command-level guard already owns CLI preflight/probe
                # events; a newly constructed direct-run guard imports them
                # conservatively at the current instant. The guard
                # deduplicates same-command events and rejects any prior POST
                # whose physical-attempt evidence is missing.
                runtime_quota_guard.seed_prior_rows(prior_rows)
        elif runtime_quota_guard is not None:
            raise ValueError(
                "runtime_quota_guard requires rate_limits in the run config")
        runtime_quota_guard_baseline = (
            runtime_quota_guard.snapshot()
            if runtime_quota_guard is not None else None)
        started_utc = datetime.fromtimestamp(
            run_started_at, timezone.utc).isoformat()
        start_provenance = {
            "start_schema_version": 1,
            "status": "writing",
            "run_started_at_unix": run_started_at,
            "run_started_at_utc": started_utc,
            "logical_run_id": logical_run_id,
            "workload_id": workload_id,
            "execution_id": execution_id,
            "artifact_id": artifact_id,
            "effective_config": _effective_config(original_rc, original_rc),
            "inputs": inputs,
            "source": source,
            "token_override_supplied": token_override is not None,
            "quota_plan": quota_plan,
            "runtime_quota_guard": (
                copy.deepcopy(runtime_quota_guard_baseline)),
            "runtime_quota_guard_baseline": (
                copy.deepcopy(runtime_quota_guard_baseline)),
            "preflight_gate": copy.deepcopy(preflight_gate),
            "setup_artifact_reference": copy.deepcopy(
                setup_artifact_reference),
            "schedule_configuration": {
                key: getattr(original_rc, key) for key in (
                    "duration_s", "qps_base", "qps_burst", "qps_min",
                    "qps_max", "rate_scale", "sizing_concurrency",
                    "timestamps_file", "seed", "shard_index", "shard_total",
                    "start_at_unix")
            },
        }
        requested_out = (Path(original_rc.out_dir)
                         / time.strftime("%Y%m%d-%H%M%S",
                                         time.localtime(run_started_at)))
        # This exclusive, fsynced claim is deliberately before token lookup,
        # endpoint discovery, network measurement, sizing, or replay traffic.
        artifact = RunArtifacts.claim(
            requested_out, start_provenance, artifact_id=artifact_id)

        with artifact:
            for row in prior_rows:
                artifact.append(row)
            if prior_rows:
                artifact.sync()
                artifact.update_start(
                    prior_request_traffic={
                        "rows": len(prior_rows),
                        "phases": {
                            phase: sum(row.get("phase") == phase
                                       for row in prior_rows)
                            for phase in ("preflight", "probe")
                            if any(row.get("phase") == phase
                                   for row in prior_rows)
                        },
                        "note": (
                            "metadata-only rows for CLI traffic sent before "
                            "the measured runner; sealed here for quota "
                            "accounting"),
                    })
            token = (
                _validated_bearer_token(
                    token_override, source="explicit token override")
                if token_override is not None else _token(ecfg))
            # An explicit override is a caller-selected identity. Refreshing it
            # from the endpoint config could silently switch principals after
            # a 401, so only config-resolved credentials are refreshable.
            refresh = None if token_override is not None else lambda: _token(ecfg)
            client = EndpointClient(
                ecfg, token, refresh=refresh,
                runtime_quota_guard=runtime_quota_guard)
            req_params = {
                "endpoint_adapter": ecfg.adapter,
                # Use the same start-attested execution snapshot as request
                # serialization. transport_contract() re-attests this exact
                # adapter before the result package is sealed.
                "endpoint_adapter_contract": (
                    client.adapter_execution.contract
                    if getattr(client, "adapter_execution", None) is not None
                    else endpoint_adapter_contract(ecfg.adapter)),
                "response_mode": (
                    client.adapter.response_mode
                    if getattr(client, "adapter", None) is not None
                    else get_endpoint_adapter(ecfg.adapter).response_mode),
                "model": ecfg.model,
                "include_usage": ecfg.include_usage,
                "temperature": ecfg.temperature,
                "max_output_tokens_cap": original_rc.max_output_tokens_cap,
                "extra_body": ecfg.extra_body or {},
            }

            # Capture target and network evidence before the first inference
            # request. A sizing pass is real endpoint traffic; metadata read
            # after it could describe a different config than the one sized.
            net_path = None
            if original_rc.measure_network_path:
                from .netpath import measure_network_path
                net_path = measure_network_path(ecfg.base_url)
                if net_path and not quiet:
                    print(f"[runner] network: "
                          f"{net_path['tcp_connect_min_ms']:.0f} ms "
                          f"TCP-connect floor to {net_path['endpoint_host']} "
                          f"({', '.join(net_path['endpoint_ips'][:2])})")

            endpoint_meta = None
            endpoint_binding = None
            invocation_binding = None
            if original_rc.capture_endpoint_metadata:
                from .endpoint_meta import (
                    fetch_endpoint_metadata,
                    invocation_endpoint_binding,
                    rate_limit_endpoint_binding,
                )
                endpoint_meta = fetch_endpoint_metadata(
                    ecfg.base_url, ecfg.path, token, timeout=5.0)
                invocation_binding = invocation_endpoint_binding(
                    ecfg.path, endpoint_meta)
                if original_rc.rate_limits is not None:
                    endpoint_binding = rate_limit_endpoint_binding(
                        original_rc.rate_limits, endpoint_meta, ecfg.path)
            if original_rc.rate_limits is not None:
                from .quota_planner import bind_quota_plan_to_endpoint
                quota_plan = bind_quota_plan_to_endpoint(
                    quota_plan, endpoint_binding or {})
            artifact.update_start(
                status=("quota-binding-refused"
                        if quota_plan is not None
                        and not quota_plan.get("may_start")
                        else "target-snapshotted"),
                endpoint_metadata=endpoint_meta,
                invocation_binding=invocation_binding,
                endpoint_binding=endpoint_binding,
                quota_plan=quota_plan,
                network_path=net_path)
            if quota_plan is not None and not quiet:
                print(render_quota_plan(quota_plan))
            enforce_quota_plan(quota_plan)

            # ---- optional unloaded sizing pass ---------------------------
            effective_rc = work_rc
            if sizing_requested is not None:
                effective_rc = _size_for_concurrency(
                    work_rc, ecfg, client, artifact.append, quiet,
                    workload_id, execution_id,
                    prevalidated_workload=prevalidated.workload)
            derived_qps = (effective_rc.qps_base
                           if sizing_requested is not None else None)

            # Capture the complete unsharded schedule, then select this
            # process's globally indexed subset. Exact binary identities are
            # persisted before calibration and measured replay traffic.
            if sizing_requested is not None:
                ceiling = prevalidated.sizing_schedule_ceiling
                if ceiling is None:
                    raise RuntimeError(
                        "sizing prevalidation did not retain its qps_max "
                        "schedule ceiling")
                fraction = (float(effective_rc.qps_base)
                            / float(work_rc.qps_max))
                full_sched = thin_schedule_ceiling(
                    ceiling, fraction, seed=effective_rc.seed + 31)
            elif prevalidated.full_schedule is not None:
                full_sched = prevalidated.full_schedule
            elif effective_rc.timestamps_file:
                full_sched = load_trace(
                    effective_rc.timestamps_file,
                    duration_cap_s=effective_rc.duration_s,
                    row_limit=MAX_EXACT_ANALYSIS_REQUEST_ROWS)
            else:
                full_sched = make_schedule(
                    duration_s=effective_rc.duration_s,
                    qps_base=effective_rc.qps_base,
                    qps_burst=effective_rc.qps_burst,
                    qps_min=effective_rc.qps_min,
                    qps_max=effective_rc.qps_max,
                    rate_scale=effective_rc.rate_scale,
                    seed=effective_rc.seed + 16,
                    request_limit=MAX_EXACT_ANALYSIS_REQUEST_ROWS)
            total_n = len(full_sched["timestamps"])
            if total_n == 0:
                raise RuntimeError(
                    "schedule produced zero arrivals; raise rate_scale or duration")
            full_sched["global_indices"] = np.arange(total_n, dtype=int)
            full_sched["total_requests"] = total_n
            sched = (shard(full_sched, effective_rc.shard_index,
                           effective_rc.shard_total)
                     if effective_rc.shard_total > 1 else full_sched)
            schedule_identity, index_identity = _schedule_identities(
                full_sched, sched, original_rc)
            sched_meta = schedule_report(sched)
            if original_rc.timestamps_file:
                sched_meta["source"] = Path(original_rc.timestamps_file).name
            artifact.update_start(
                status="schedule-snapshotted",
                effective_config=_effective_config(original_rc, effective_rc),
                schedule_identity=schedule_identity,
                index_identity=index_identity,
                schedule=sched_meta,
                derived_qps=derived_qps)

            ts = sched["timestamps"]
            global_indices = sched["global_indices"]
            n = len(ts)
            if sizing_requested is None:
                workload = prevalidated.workload
                if workload is None or workload.total_n != total_n:
                    raise RuntimeError(
                        "prevalidated workload does not match the exact "
                        "schedule")
            else:
                workload = _PreparedWorkload(
                    effective_rc, total_n,
                    loaded_profile=prevalidated.profile,
                    loaded_prompts=prevalidated.prompts)
            m = workload.prompts_count
            p = workload.profile

            if not quiet:
                if prompts_mode:
                    print(f"[runner] {n} scheduled arrivals over "
                          f"{effective_rc.duration_s}s, replaying {m} real "
                          f"prompts from {original_rc.prompts_file}")
                else:
                    print(f"[runner] {n} scheduled arrivals over "
                          f"{effective_rc.duration_s}s (rate_scale "
                          f"{effective_rc.rate_scale}), profile '{p.name}'")
                    if p.label:
                        print(f"[runner] profile label: {p.label}")

            # ---- calibration / warmup pass ------------------------------
            calib_n = min(effective_rc.calibrate_n, total_n)
            chars_total = 0
            ptok_total = 0
            calibration_rows = []
            for i in range(calib_n):
                body_rid = _stable_request_id(
                    workload_id, i, "calibration-body")
                rid = _stable_request_id(
                    execution_id, i,
                    f"calibration-shard-{effective_rc.shard_index}")
                plan = workload.plan(i, body_rid)
                body_hash = _client_payload_hash(
                    client, ecfg, plan["messages"], plan["max_output"])
                if (runtime_quota_guard is not None
                        and runtime_quota_guard.tripped):
                    row = _exception_result(
                        rid, "calibration", plan, body_hash,
                        "request omitted after runtime quota admission "
                        "refusal; no HTTP POST was attempted",
                        known_not_sent=True,
                        endpoint_adapter=ecfg.adapter)
                    row.update(
                        quota_guard_id=runtime_quota_guard.guard_id,
                        quota_guard_denied=True,
                        quota_guard_events=[])
                else:
                    try:
                        res = _send_request(
                            client, plan["messages"], plan["max_output"], rid,
                            0.0, 0.0, plan["intended"], plan["chars"])
                        row = _annotate_result(
                            res, "calibration", plan, body_hash)
                    except Exception as exc:
                        row = _exception_result(
                            rid, "calibration", plan, body_hash,
                            "unexpected worker exception: "
                            f"{type(exc).__name__}: {exc}",
                            endpoint_adapter=ecfg.adapter)
                artifact.append(row)
                calibration_rows.append(row)
                if (_clean_measurement_row(row)
                        and row.get("prompt_tokens")):
                    chars_total += plan["chars"]
                    ptok_total += row["prompt_tokens"]

            # Recalibrate only synthetic material. The original input cannot
            # change this run: workload parsing is already on private bytes.
            calibration = {
                "requests": calib_n,
                "eligible_clean_usage_requests": sum(
                    1 for row in calibration_rows
                    if _clean_measurement_row(row)
                    and row.get("phase") == "calibration"
                    and row.get("prompt_tokens")),
                "reported_prompt_tokens": ptok_total,
                "cpt_initial": effective_rc.cpt,
                "cpt_final": effective_rc.cpt,
            }
            calibration_complete = (
                calibration["eligible_clean_usage_requests"] == calib_n)
            calibration["status"] = (
                "complete" if calibration_complete else
                "incomplete_cpt_unchanged")
            if not prompts_mode and ptok_total and calibration_complete:
                old_cpt = workload.mat.cpt
                new_cpt = calibrate_cpt(old_cpt, chars_total, ptok_total)
                calibration["cpt_final"] = new_cpt
                if not quiet:
                    print(f"[runner] cpt calibrated {old_cpt:.2f} -> "
                          f"{new_cpt:.2f} (from {ptok_total} reported "
                          "prompt tokens)")
                workload.set_cpt(new_cpt)
            elif not prompts_mode and calib_n and not calibration_complete \
                    and not quiet:
                print(
                    "[runner] calibration incomplete: only "
                    f"{calibration['eligible_clean_usage_requests']} of "
                    f"{calib_n} responses had clean, complete prompt usage; "
                    "cpt was left unchanged")
            artifact.update_start(
                status="replay-ready", calibration=calibration,
                endpoint_metadata=endpoint_meta, network_path=net_path,
                runtime_quota_guard=(
                    runtime_quota_guard.snapshot()
                    if runtime_quota_guard is not None else None))

            # ---- paced replay --------------------------------------------
            if effective_rc.start_at_unix is not None:
                until_start = effective_rc.start_at_unix - time.time()
                if until_start < -effective_rc.start_tolerance_s:
                    raise RuntimeError(
                        f"shared start_at_unix became stale by "
                        f"{-until_start:.3f}s during setup; choose a later "
                        "start and verify shard clocks")
                t0 = time.monotonic() + until_start
            else:
                t0 = time.monotonic() + 0.25

            from .progress import Progress
            prog = Progress(n, float(effective_rc.duration_s),
                            enabled=not quiet)
            pending_limit = (
                effective_rc.max_pending_requests
                if effective_rc.max_pending_requests is not None
                else max(effective_rc.max_concurrency * 2,
                         effective_rc.max_concurrency + 1))

            def _progress_done(fut):
                if fut.cancelled():
                    prog.done(None)
                    return
                try:
                    prog.done(fut.result())
                except Exception:
                    # Collection below persists the exception as an error row.
                    prog.done(None)

            def _collect(fut, context):
                rid, plan, body_hash, scheduled_s, lag_ms = context
                try:
                    return _annotate_result(
                        fut.result(), "replay", plan, body_hash)
                except Exception as exc:
                    return _exception_result(
                        rid, "replay", plan, body_hash,
                        "unexpected worker exception: "
                        f"{type(exc).__name__}: {exc}",
                        scheduled_s=scheduled_s, dispatch_lag_ms=lag_ms,
                        endpoint_adapter=ecfg.adapter)

            try:
                parameters = tuple(inspect.signature(
                    client.send).parameters.values())
                supports_scheduled_clock = any(
                    p.name == "scheduled_monotonic"
                    or p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in parameters)
                supports_cancellation = any(
                    p.name == "cancellation_event"
                    for p in parameters)
            except (TypeError, ValueError):
                supports_scheduled_clock = True
                supports_cancellation = True

            pending: dict = {}
            cancellation_event = threading.Event()
            ex = ThreadPoolExecutor(
                max_workers=effective_rc.max_concurrency)
            completed_normally = False
            try:
                for local_i in range(n):
                    target = t0 + float(ts[local_i])
                    now = time.monotonic()
                    while target > now and not (
                            runtime_quota_guard is not None
                            and runtime_quota_guard.tripped):
                        time.sleep(min(target - now, 0.1))
                        now = time.monotonic()
                    # Both bookkeeping and the executor queue stay bounded.
                    # Rows are journaled as soon as this dispatcher observes
                    # completion; no run-sized in-memory result list exists.
                    for done in [f for f in pending if f.done()]:
                        artifact.append(_collect(done, pending.pop(done)))

                    global_i = int(global_indices[local_i])
                    body_rid = _stable_request_id(
                        workload_id, global_i, "replay-body")
                    rid = _stable_request_id(
                        execution_id, global_i, "replay")
                    plan = workload.plan(global_i, body_rid)
                    body_hash = _client_payload_hash(
                        client, ecfg, plan["messages"], plan["max_output"])
                    # Dispatch lateness includes all synchronous dispatcher
                    # work required to make this request submit-ready.  A
                    # timestamp taken before plan construction/body hashing
                    # hid generator-side saturation as though the dispatcher
                    # were on time.
                    lag_ms = max(
                        (time.monotonic() - target) * 1000.0, 0.0)
                    prog.sent()
                    if (runtime_quota_guard is not None
                            and runtime_quota_guard.tripped):
                        row = _exception_result(
                            rid, "replay", plan, body_hash,
                            "request omitted after runtime quota admission "
                            "refusal; no HTTP POST was attempted",
                            scheduled_s=float(ts[local_i]),
                            dispatch_lag_ms=lag_ms,
                            known_not_sent=True,
                            endpoint_adapter=ecfg.adapter)
                        row.update(
                            quota_guard_id=runtime_quota_guard.guard_id,
                            quota_guard_denied=True,
                            quota_guard_events=[])
                        artifact.append(row)
                        prog.done(None)
                        prog.paint()
                        continue
                    if len(pending) >= pending_limit:
                        artifact.append(_exception_result(
                            rid, "replay", plan, body_hash,
                            f"client pending limit {pending_limit} reached; "
                            "request was not sent",
                            scheduled_s=float(ts[local_i]),
                            dispatch_lag_ms=lag_ms,
                            known_not_sent=True,
                            endpoint_adapter=ecfg.adapter))
                        prog.done(None)
                        prog.paint()
                        continue

                    send_kwargs = {}
                    if supports_scheduled_clock:
                        send_kwargs["scheduled_monotonic"] = target
                    if supports_cancellation:
                        send_kwargs["cancellation_event"] = cancellation_event
                    # Keep the persisted value adjacent to the actual queue
                    # handoff so quota checks and pending-pool bookkeeping are
                    # included too.
                    lag_ms = max(
                        (time.monotonic() - target) * 1000.0, 0.0)
                    send_args = (
                        plan["messages"], plan["max_output"], rid,
                        float(ts[local_i]), lag_ms, plan["intended"],
                        plan["chars"])
                    fut = ex.submit(client.send, *send_args, **send_kwargs)
                    fut.add_done_callback(_progress_done)
                    pending[fut] = (
                        rid, plan, body_hash, float(ts[local_i]), lag_ms)
                    prog.paint()

                for fut in as_completed(list(pending)):
                    artifact.append(_collect(fut, pending[fut]))
                    prog.paint()
                completed_normally = True
            except BaseException:
                # Set the cooperative guard before touching the executor
                # queue. A worker racing out of that queue will observe the
                # event immediately before POST, and a request already on the
                # wire will observe it before any retry.
                cancellation_event.set()
                cancel_active = getattr(client, "cancel_active_requests", None)
                if callable(cancel_active):
                    try:
                        cancel_active()
                    except Exception:
                        # Preserve the operator's BaseException. Cooperative
                        # cancellation still prevents new POSTs and retries
                        # even if a custom client cannot interrupt active I/O.
                        pass
                running = {}
                for fut, context in list(pending.items()):
                    if fut.cancel():
                        rid, plan, body_hash, scheduled_s, lag_ms = context
                        try:
                            artifact.append(_exception_result(
                                rid, "replay", plan, body_hash,
                                "operator cancellation before request send",
                                scheduled_s=scheduled_s,
                                dispatch_lag_ms=lag_ms,
                                known_not_sent=True,
                                endpoint_adapter=ecfg.adapter))
                        except Exception:
                            # Preserve the original BaseException. start.json
                            # and the append-only journal remain recoverable.
                            pass
                    else:
                        running[fut] = context

                # A running worker may already have emitted a physical POST.
                # Socket shutdown normally makes it return immediately; drain
                # for a short bounded interval and persist its exact result.
                # If a custom/non-cooperative transport remains stuck, write
                # one explicit unknown row so the journal never implies that
                # zero requests ran or were billable.
                drained, still_running = wait(
                    tuple(running), timeout=_CANCELLATION_DRAIN_TIMEOUT_S)
                for fut in drained:
                    try:
                        artifact.append(_collect(fut, running[fut]))
                    except Exception:
                        pass
                for fut in still_running:
                    rid, plan, body_hash, scheduled_s, lag_ms = running[fut]
                    try:
                        artifact.append(_exception_result(
                            rid, "replay", plan, body_hash,
                            "outcome unknown after operator cancellation; "
                            "the running worker may have emitted an HTTP POST",
                            scheduled_s=scheduled_s,
                            dispatch_lag_ms=lag_ms,
                            known_not_sent=False,
                            endpoint_adapter=ecfg.adapter))
                    except Exception:
                        pass
                ex.shutdown(wait=False, cancel_futures=True)
                try:
                    artifact.sync()
                except Exception:
                    pass
                raise
            finally:
                if completed_normally:
                    ex.shutdown(wait=True)
            prog.finish()
            artifact.sync()

            # A pre-run control-plane snapshot cannot prove the endpoint stayed
            # unchanged while traffic ran. Re-read only after every response
            # has drained. A changed document invalidates a single-config
            # benchmark; a failed second read remains explicit uncertainty.
            endpoint_meta_after = None
            endpoint_metadata_stability = "not_requested"
            endpoint_metadata_warning = None
            if original_rc.capture_endpoint_metadata:
                from .endpoint_meta import fetch_endpoint_metadata
                endpoint_meta_after = fetch_endpoint_metadata(
                    ecfg.base_url, ecfg.path, client.token, timeout=5.0)
                if endpoint_meta is None or endpoint_meta_after is None:
                    endpoint_metadata_stability = "unverified"
                    endpoint_metadata_warning = (
                        "serving endpoint metadata could not be captured both "
                        "before and after the replay, so configuration "
                        "stability was not established")
                elif canonical_sha256(endpoint_meta) != canonical_sha256(
                        endpoint_meta_after):
                    endpoint_metadata_stability = "changed"
                    endpoint_metadata_warning = (
                        "serving endpoint metadata changed between the "
                        "pre-run and post-drain snapshots")
                else:
                    endpoint_metadata_stability = "stable"
                artifact.update_start(
                    status="endpoint-post-snapshotted",
                    endpoint_metadata_after=endpoint_meta_after,
                    endpoint_metadata_stability=endpoint_metadata_stability,
                    endpoint_metadata_warning=endpoint_metadata_warning,
                    runtime_quota_guard=(
                        runtime_quota_guard.snapshot()
                        if runtime_quota_guard is not None else None))

            load_meta = {
                "load_mode": load_mode,
                "sizing_concurrency_requested": sizing_requested,
                "sizing_concurrency_local": sizing_local,
                "derived_qps": derived_qps,
                "run_id": logical_run_id,
                "logical_run_id": logical_run_id,
                "workload_id": workload_id,
                "execution_id": execution_id,
                "artifact_id": artifact_id,
                "schedule_identity": schedule_identity,
                "index_identity": index_identity,
                "start_at_unix": effective_rc.start_at_unix,
                "max_pending_requests": pending_limit,
                "global_index_start": index_identity["min"],
                "global_index_end": index_identity["max"],
                "global_index_range": [index_identity["min"],
                                       index_identity["max"]],
                # A fixed-rate open loop does not hold occupancy. Metrics
                # reports observed concurrency as an outcome instead.
                "concurrency_target": None,
            }
            common_meta = {
                "endpoint_path": ecfg.path,
                "label": original_rc.label,
                "title": original_rc.title,
                "request_params": req_params,
                "endpoint_metadata": endpoint_meta,
                "endpoint_metadata_after": endpoint_meta_after,
                "endpoint_metadata_stability": endpoint_metadata_stability,
                "endpoint_metadata_warning": endpoint_metadata_warning,
                "invocation_binding": invocation_binding,
                "network_path": net_path,
                # EndpointClient always exposes this contract.  Lightweight
                # injected transports used by library callers and tests may
                # not; absence remains explicit instead of crashing after all
                # paid traffic has already completed.
                "transport": (
                    client.transport_contract()
                    if callable(getattr(client, "transport_contract", None))
                    else None),
                "quota_plan": quota_plan,
                "endpoint_binding": endpoint_binding,
                "runtime_quota_guard": (
                    runtime_quota_guard.snapshot()
                    if runtime_quota_guard is not None else None),
                "runtime_quota_guard_baseline": (
                    copy.deepcopy(runtime_quota_guard_baseline)),
                "preflight_gate": copy.deepcopy(preflight_gate),
                "setup_artifact_reference": copy.deepcopy(
                    setup_artifact_reference),
                "shard": (f"{effective_rc.shard_index + 1}/"
                          f"{effective_rc.shard_total}"),
                "endpoint_base_url": ecfg.base_url,
                "endpoint_model": ecfg.model,
                "profile_path": (Path(original_rc.profile_path).name
                                 if original_rc.profile_path else None),
                "prompts_file": (Path(original_rc.prompts_file).name
                                  if original_rc.prompts_file else None),
                "seed": effective_rc.seed,
                "ttft_definition": effective_rc.ttft_definition,
                **load_meta,
            }
            if prompts_mode:
                meta = {
                    **common_meta,
                    "input_mode": "prompts",
                    "prompts_count": m,
                }
                acceptance = effective_rc.acceptance_targets
            else:
                meta = {
                    **common_meta,
                    "input_mode": "profile",
                    "profile": p.name,
                    "profile_provenance": p.provenance,
                    "profile_label": p.label,
                    "cpt_final": workload.mat.cpt,
                }
                acceptance = (
                    effective_rc.acceptance_targets
                    or (p.extra or {}).get("acceptance_targets"))

            if acceptance and "targets_are" not in acceptance:
                acceptance = {
                    **acceptance,
                    "targets_are": (
                        "the run config" if original_rc.acceptance_targets
                        else "this profile"),
                }

            # The journal is reread only after traffic has drained. During
            # generation memory is bounded by max_pending_requests; the final
            # exact percentile calculation uses the persisted replay rows.
            journal_rows = list(artifact.read_rows())
            _assert_row_adapter_contract(journal_rows, ecfg)
            replay_rows = [
                row for row in journal_rows if row.get("phase") == "replay"]
            summary = summarize(
                replay_rows, schedule_meta=sched_meta, run_meta=meta,
                acceptance=acceptance,
                ttft_definition=effective_rc.ttft_definition,
                pricing=effective_rc.pricing,
                rate_limits=effective_rc.rate_limits,
                rate_limit_results=journal_rows,
                concurrency_target=None)
            out = write_outputs(
                None, summary, artifact.path, original_rc.title,
                artifact_run=artifact,
                start_provenance=artifact.start_provenance)

        if not quiet:
            print(f"[runner] wrote {out}/report.html (open in a browser) "
                  f"and {out}/report.md")
        return {
            "summary": summary,
            "out_dir": str(out),
            "results_n": artifact.row_count,
        }
