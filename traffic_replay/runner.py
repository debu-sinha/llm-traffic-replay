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
because ThreadPoolExecutor.submit() queues rather than blocking. Wire
lateness, computed in metrics from first_send_unix against the schedule, is
when the client began sending, and it grows under either. Read wire lateness
to decide whether the client kept up.

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
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .client import (EndpointClient, EndpointConfig, RequestResult,
                     normalized_origin,
                     validate_bearer_transport)
from .config_validation import (validate_acceptance_targets,
                                validate_pricing, validate_rate_limits)
from .json_input import loads_strict
from .metrics import summarize, write_outputs
from .prefix_pool import PrefixPool
from .schedule import (load_trace, make_schedule, schedule_report, shard,
                       validate_schedule_capacity)
from .textgen import TextMaterializer, calibrate_cpt


_DEFAULT_MAX_CONCURRENCY = 256
_MAX_CONCURRENCY = 4096
_MAX_PENDING_REQUESTS = 100_000
_MAX_POOL_DOCS_PER_BUCKET = 10_000
_MAX_CALIBRATION_REQUESTS = 10_000
_AUTH_RESPONSE_MAX_BYTES = 64 * 1024
_AUTH_TOKEN_MAX_BYTES = 64 * 1024
_AUTH_CREDENTIAL_MAX_BYTES = 8 * 1024
_AUTH_M2M_TIMEOUT_S = 15.0
_AUTH_CLI_TIMEOUT_S = 30.0
_AUTH_DISABLED_DEFAULT_SECTION = (
    "__traffic_replay_reserved_defaults_do_not_use__")


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


def _file_identity(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return f"unreadable:{path}"


def _read_stable_bytes(path: str) -> tuple[bytes, os.stat_result]:
    """Read one immutable view of an input, rejecting concurrent mutation."""
    source = Path(path)
    try:
        fd = os.open(source, os.O_RDONLY)
    except OSError as exc:
        raise ValueError(f"cannot snapshot input {source}: {exc}") from exc
    try:
        before = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_ctime_ns)
    raw = b"".join(chunks)
    if identity_before != identity_after or len(raw) != before.st_size:
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
        raw, info = _read_stable_bytes(original)
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
        "profile": _file_identity(rc.profile_path),
        "prompts": _file_identity(rc.prompts_file),
        "endpoint_path": rc.endpoint.get("path"),
        "endpoint_model": rc.endpoint.get("model"),
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


def _payload_hash(ecfg: EndpointConfig, messages: list[dict],
                  max_tokens: int) -> str:
    """Hash the deterministic logical body, excluding learned wire fallback."""
    owned = {"messages", "max_tokens", "temperature", "stream", "model",
             "stream_options"}
    body = {k: v for k, v in (ecfg.extra_body or {}).items() if k not in owned}
    body.update(messages=messages, max_tokens=int(max_tokens),
                temperature=ecfg.temperature, stream=True)
    if ecfg.model:
        body["model"] = ecfg.model
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


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
            plans.append({
                "messages": msgs, "max_output": rc.max_output_tokens_cap,
                "intended": (0, 0, None, prompt_index),
                "chars": sum(len(x["content"]) for x in msgs),
                "global_index": i, "prompt_index": prompt_index,
                "sample_index": None, "construction": None,
                "request_id": _stable_request_id(run_id, i, "preflight"),
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
        rid = _stable_request_id(run_id, i, "preflight")
        prefix = int(assignment.prefix_tokens[i])
        suffix = int(inputs[i]) - prefix
        doc_id = int(assignment.doc_id[i])
        msgs = mat.messages(rid, doc_id, prefix,
                            pool.doc_len.get(doc_id, 0), suffix)
        plans.append({
            "messages": msgs,
            "max_output": min(int(outputs[i]), rc.max_output_tokens_cap),
            "intended": (int(inputs[i]), int(outputs[i]),
                         prefix / int(inputs[i]), doc_id),
            "chars": sum(len(x["content"]) for x in msgs),
            "global_index": i, "prompt_index": None, "sample_index": i,
            "construction": mat.construction_report(msgs, int(inputs[i])),
            "request_id": rid, "representative": quantile,
        })
    return plans


@dataclasses.dataclass
class PrevalidatedRunInputs:
    """Fully parsed, endpoint-free inputs reusable by preflight and runner.

    The object deliberately retains the parsed workload source and exact
    deterministic schedule so a caller does not validate one file view and
    later reread another.  A sizing-derived schedule is necessarily ``None``
    until the unloaded service-time pass determines its fixed rate; the
    sizing probe workload is still fully constructed here.
    """

    rc: RunConfig
    full_schedule: dict | None
    workload: _PreparedWorkload | None
    profile: prof.Profile | None
    prompts: list[list[dict]] | None
    representative_plans: list[dict]
    schedule_kind: str


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
    body_endpoint_fields = ("path", "model", "temperature", "extra_body")
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

    full_schedule = None
    if checked.sizing_concurrency is None:
        if checked.timestamps_file:
            full_schedule = load_trace(
                checked.timestamps_file, duration_cap_s=checked.duration_s)
            schedule_kind = "timestamp_trace"
        else:
            full_schedule = make_schedule(
                duration_s=checked.duration_s,
                qps_base=checked.qps_base,
                qps_burst=checked.qps_burst,
                qps_min=checked.qps_min,
                qps_max=checked.qps_max,
                rate_scale=checked.rate_scale,
                seed=checked.seed + 16)
            schedule_kind = "deterministic_synthetic"
        total_n = len(full_schedule["timestamps"])
        if require_nonempty_schedule and total_n == 0:
            raise RuntimeError(
                "schedule produced zero arrivals; raise rate_scale or duration")
    else:
        # The only schedule that cannot exist before endpoint traffic: sizing
        # derives its fixed rate from unloaded service time. RunConfig rejects
        # a timestamp trace combined with this mode.
        schedule_kind = "sizing_derived_after_prevalidation"
        total_n = max(4, min(checked.calibrate_n, 8))

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
    return PrevalidatedRunInputs(
        rc=checked,
        full_schedule=full_schedule,
        workload=workload,
        profile=loaded_profile,
        prompts=loaded_prompts,
        representative_plans=representatives,
        schedule_kind=schedule_kind,
    )


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
                      known_not_sent: bool = False) -> dict:
    intended = plan["intended"]
    row = {
        "request_id": request_id, "scheduled_s": scheduled_s,
        "dispatch_lag_ms": dispatch_lag_ms, "t_send_unix": None,
        "first_send_unix": None, "ttfb_ms": None, "ttft_ms": None,
        "ttfr_ms": None, "ttfv_ms": None, "e2e_ms": None,
        "queue_wait_ms": None, "caller_ttfb_ms": None,
        "caller_ttft_ms": None, "caller_ttfr_ms": None,
        "caller_ttfv_ms": None, "caller_ttf_tool_call_ms": None,
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


def _size_for_concurrency(rc: "RunConfig", ecfg, client, record,
                          quiet: bool, workload_id: str,
                          execution_id: str, *,
                          prevalidated_workload: _PreparedWorkload | None = None
                          ) -> "RunConfig":
    """Derive a fixed open-loop rate from an unloaded concurrency hint.

    This does not hold concurrency. It measures unloaded service time once,
    computes ``rate = sizing_concurrency / e2e_p50``, and leaves that rate
    fixed while the endpoint slows or speeds up under load.
    """
    import numpy as _np

    probe_n = max(4, min(rc.calibrate_n, 8))
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
        body_hash = _payload_hash(ecfg, plan["messages"], plan["max_output"])
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
                f"unexpected worker exception: {type(exc).__name__}: {exc}")
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

    p50 = float(_np.percentile(e2e, 50)) / 1000.0
    p95 = float(_np.percentile(e2e, 95)) / 1000.0
    rate = rc.sizing_concurrency / max(p50, 1e-3)
    validate_schedule_capacity(rc.duration_s, rate)
    derived_pool_size = max(rc.sizing_concurrency * 2,
                            int(math.ceil(rate * p95 * 1.5)))
    pool_cap = (rc.max_concurrency if rc.max_concurrency is not None
                else _DEFAULT_MAX_CONCURRENCY)
    pool_size = min(derived_pool_size, pool_cap)
    if not quiet:
        print(f"[runner] sizing from {len(e2e)} probe requests: e2e p50 "
              f"{p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms")
        print(f"[runner] sizing hint {rc.sizing_concurrency}: offering a fixed "
              f"{rate:.2f} rps with pool {pool_size}"
              + (f" (derived {derived_pool_size}, capped by explicit "
                 "max_concurrency)" if pool_size < derived_pool_size
                 and rc.max_concurrency is not None else "")
              + (f" (derived {derived_pool_size}, capped by the default "
                 f"{_DEFAULT_MAX_CONCURRENCY}-thread safety limit)"
                 if pool_size < derived_pool_size
                 and rc.max_concurrency is None else "")
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
    if not isinstance(value, str) or not value:
        raise AuthProfileError(
            f"{source} did not provide a non-empty access token")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        raise AuthProfileError(
            f"{source} returned a bearer token with non-ASCII characters") \
            from None
    if len(encoded) > _AUTH_TOKEN_MAX_BYTES:
        raise AuthProfileError(
            f"{source} returned an oversized bearer token "
            f"(bytes={len(encoded)})")
    if any(byte < 0x21 or byte > 0x7e for byte in encoded):
        raise AuthProfileError(
            f"{source} returned a bearer token with unsafe whitespace or "
            "control characters")
    return value


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
        conn.request("POST", "/oidc/v1/token", body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        if not isinstance(status, int) or isinstance(status, bool) \
                or not 100 <= status <= 599:
            raise AuthProfileError(
                "Databricks OAuth M2M token endpoint returned an invalid "
                "HTTP status")
        raw = _read_bounded_auth_response(response)
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
        content_type = response.getheader("Content-Type")
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
    try:
        result = subprocess.run(
            ["databricks", "auth", "token", "-p", name],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=_AUTH_CLI_TIMEOUT_S, env=cli_env, check=False)
    except subprocess.TimeoutExpired:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} token mint timed out after "
            f"{_AUTH_CLI_TIMEOUT_S:g} seconds; run 'databricks auth login "
            f"--profile {name}' interactively") from None
    except OSError as exc:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} could not invoke the "
            f"Databricks CLI ({type(exc).__name__}); install the CLI and run "
            f"'databricks auth login --profile {name}'") from None
    if result.returncode != 0:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} token mint exited with status "
            f"{result.returncode}; run 'databricks auth login --profile "
            f"{name}' interactively")
    raw = result.stdout
    if not isinstance(raw, bytes):
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} returned a non-byte token "
            "response")
    fingerprint = _auth_bytes_fingerprint(raw)
    if len(raw) > _AUTH_RESPONSE_MAX_BYTES:
        raise AuthProfileError(
            f"Databricks U2M profile {name!r} returned an oversized token "
            f"response ({fingerprint})")
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
    return tok


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
        "constructed_error_chars",
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
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"prior_request_rows[{index}] must be finite JSON metadata") \
                from exc
        prepared.append(redact_secrets(row))
    return prepared


def run(rc: RunConfig, token_override: str | None = None,
        quiet: bool = False, prior_request_rows=None) -> dict:
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
        # This gate is intentionally before token lookup, endpoint metadata,
        # network measurement, sizing, calibration, or replay.  A quota-aware
        # config that cannot be bounded must not spend inference traffic in
        # order to discover that fact.
        from .quota_planner import (
            enforce_quota_plan,
            plan_run_quota,
            render_quota_plan,
        )
        quota_plan = plan_run_quota(
            work_rc, prior_rows=prior_rows, prevalidated=prevalidated)
        if quota_plan is not None and not quota_plan.get("may_start") \
                and not quiet:
            print(render_quota_plan(quota_plan))
        enforce_quota_plan(quota_plan)
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
            token = (token_override if token_override is not None
                     else _token(ecfg))
            client = EndpointClient(ecfg, token,
                                    refresh=lambda: _token(ecfg))
            req_params = {
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
            if original_rc.capture_endpoint_metadata:
                from .endpoint_meta import (
                    fetch_endpoint_metadata,
                    rate_limit_endpoint_binding,
                )
                endpoint_meta = fetch_endpoint_metadata(
                    ecfg.base_url, ecfg.path, token, timeout=5.0)
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
            if prevalidated.full_schedule is not None:
                full_sched = prevalidated.full_schedule
            elif effective_rc.timestamps_file:
                full_sched = load_trace(
                    effective_rc.timestamps_file,
                    duration_cap_s=effective_rc.duration_s)
            else:
                full_sched = make_schedule(
                    duration_s=effective_rc.duration_s,
                    qps_base=effective_rc.qps_base,
                    qps_burst=effective_rc.qps_burst,
                    qps_min=effective_rc.qps_min,
                    qps_max=effective_rc.qps_max,
                    rate_scale=effective_rc.rate_scale,
                    seed=effective_rc.seed + 16)
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
                body_hash = _payload_hash(
                    ecfg, plan["messages"], plan["max_output"])
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
                        f"{type(exc).__name__}: {exc}")
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
                endpoint_metadata=endpoint_meta, network_path=net_path)

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
                        scheduled_s=scheduled_s, dispatch_lag_ms=lag_ms)

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
                    if target > now:
                        time.sleep(target - now)
                    lag_ms = max(
                        (time.monotonic() - target) * 1000.0, 0.0)

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
                    body_hash = _payload_hash(
                        ecfg, plan["messages"], plan["max_output"])
                    prog.sent()
                    if len(pending) >= pending_limit:
                        artifact.append(_exception_result(
                            rid, "replay", plan, body_hash,
                            f"client pending limit {pending_limit} reached; "
                            "request was not sent",
                            scheduled_s=float(ts[local_i]),
                            dispatch_lag_ms=lag_ms,
                            known_not_sent=True))
                        prog.done(None)
                        prog.paint()
                        continue

                    send_args = (
                        plan["messages"], plan["max_output"], rid,
                        float(ts[local_i]), lag_ms, plan["intended"],
                        plan["chars"])
                    send_kwargs = {}
                    if supports_scheduled_clock:
                        send_kwargs["scheduled_monotonic"] = target
                    if supports_cancellation:
                        send_kwargs["cancellation_event"] = cancellation_event
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
                for fut, context in list(pending.items()):
                    if not fut.cancel():
                        continue
                    rid, plan, body_hash, scheduled_s, lag_ms = context
                    try:
                        artifact.append(_exception_result(
                            rid, "replay", plan, body_hash,
                            "operator cancellation before request send",
                            scheduled_s=scheduled_s,
                            dispatch_lag_ms=lag_ms,
                            known_not_sent=True))
                    except Exception:
                        # Preserve the original BaseException. start.json and
                        # the append-only journal remain an incomplete,
                        # recoverable artifact even if this best-effort row
                        # cannot be written.
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
                "network_path": net_path,
                "quota_plan": quota_plan,
                "endpoint_binding": endpoint_binding,
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
