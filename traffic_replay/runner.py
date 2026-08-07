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
from .client import (EndpointClient, EndpointConfig,
                     validate_bearer_transport)
from .config_validation import (validate_acceptance_targets,
                                validate_pricing)
from .metrics import summarize, write_outputs
from .prefix_pool import PrefixPool
from .schedule import load_trace, make_schedule, schedule_report, shard
from .textgen import TextMaterializer, calibrate_cpt


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
    max_concurrency: int = 256
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
            EndpointConfig(**self.endpoint)
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
        if not isinstance(self.max_concurrency, int) \
                or isinstance(self.max_concurrency, bool) \
                or self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if self.max_pending_requests is not None and (
                not isinstance(self.max_pending_requests, int)
                or isinstance(self.max_pending_requests, bool)
                or self.max_pending_requests <= 0):
            raise ValueError("max_pending_requests must be a positive integer")
        if isinstance(self.cpt, bool) or not isinstance(self.cpt, (int, float)) \
                or not math.isfinite(float(self.cpt)) or self.cpt <= 0:
            raise ValueError("cpt must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.calibrate_n, int) \
                or isinstance(self.calibrate_n, bool) or self.calibrate_n < 0:
            raise ValueError("calibrate_n must be a non-negative integer")
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
        for name in ("capture_endpoint_metadata", "measure_network_path"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
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
            if not isinstance(self.run_id, str) or not self.run_id.strip():
                raise ValueError("sharded runs require one shared non-empty run_id")
            if self.start_at_unix is None:
                raise ValueError("sharded runs require one shared start_at_unix")


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
        suffix = "".join(Path(original).suffixes)
        snapshot = directory / f"{key}{suffix or '.snapshot'}"
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

    def __init__(self, rc: RunConfig, total_n: int):
        if total_n <= 0:
            raise ValueError("workload needs at least one request")
        self.rc = rc
        self.total_n = total_n
        self.prompts_mode = bool(rc.prompts_file)
        self.profile = None
        self.prompt_msgs = None
        self.mat = None
        if self.prompts_mode:
            from .prompts import load_prompts
            self.prompt_msgs = load_prompts(rc.prompts_file)
        else:
            self.profile = prof.Profile.from_json(rc.profile_path)
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


def _representative_plans(rc: RunConfig) -> list[dict]:
    """Concrete p50/p95 profile requests, or the first two real prompts."""
    run_id = _resolved_run_id(rc)
    if rc.prompts_file:
        from .prompts import load_prompts
        messages = load_prompts(rc.prompts_file)
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

    p = prof.Profile.from_json(rc.profile_path)
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
                      dispatch_lag_ms: float = 0.0) -> dict:
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
        "first_attempt_unix": None, "connection_attempts": 0,
        "request_attempts": 0, "retry_reasons": [],
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
                          execution_id: str) -> "RunConfig":
    """Derive a fixed open-loop rate from an unloaded concurrency hint.

    This does not hold concurrency. It measures unloaded service time once,
    computes ``rate = sizing_concurrency / e2e_p50``, and leaves that rate
    fixed while the endpoint slows or speeds up under load.
    """
    import numpy as _np

    probe_n = max(4, min(rc.calibrate_n, 8))
    workload = _PreparedWorkload(rc, probe_n)

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
        if d.get("ok") and d.get("e2e_ms"):
            e2e.append(d["e2e_ms"])

    if not e2e:
        raise RuntimeError(
            "sizing pass got no successful response, so the arrival rate for "
            f"sizing_concurrency {rc.sizing_concurrency} cannot be derived. "
            "check auth and "
            "the endpoint path, or set qps_base and max_concurrency directly.")

    p50 = float(_np.percentile(e2e, 50)) / 1000.0
    p95 = float(_np.percentile(e2e, 95)) / 1000.0
    rate = rc.sizing_concurrency / max(p50, 1e-3)
    pool_size = max(rc.sizing_concurrency * 2,
                    int(math.ceil(rate * p95 * 1.5)))
    if not quiet:
        print(f"[runner] sizing from {len(e2e)} probe requests: e2e p50 "
              f"{p50 * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms")
        print(f"[runner] sizing hint {rc.sizing_concurrency}: offering a fixed "
              f"{rate:.2f} rps with pool {pool_size}; concurrency is measured, "
              "not held")
    return dataclasses.replace(
        rc, qps_base=rate, qps_burst=rate, qps_min=rate, qps_max=rate,
        rate_scale=1.0, max_concurrency=pool_size)


class AuthProfileError(RuntimeError):
    """A named Databricks profile could not be resolved safely."""


def _token_from_profile(name: str, endpoint_base_url: str) -> str:
    """Resolve a ~/.databrickscfg profile to a bearer token.

    A PAT profile stores the token directly. An OAuth profile stores no
    usable bearer token, so the Databricks CLI is asked to mint one, which
    also refreshes it if it has expired. Before either token is read, the
    profile's configured origin is normalized and required to match the
    request endpoint. Named profiles fail closed: there is no environment
    fallback for a typo, an unavailable CLI, or a host mismatch.
    """
    import configparser
    import json as _json
    import subprocess
    from pathlib import Path

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE",
                                   Path.home() / ".databrickscfg"))
    parser = configparser.ConfigParser(interpolation=None)
    try:
        read = parser.read(cfg_path)
    except (OSError, configparser.Error) as exc:
        raise AuthProfileError(
            f"could not read Databricks config {cfg_path}: {exc}") from exc
    if not read:
        raise AuthProfileError(f"Databricks config not found: {cfg_path}")
    if not (parser.has_section(name) or name == "DEFAULT"):
        raise AuthProfileError(f"Databricks auth profile {name!r} does not exist")

    sect = parser[name]
    profile_host = (sect.get("host") or "").strip()
    if not profile_host:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} has no configured host")
    try:
        profile_origin = validate_bearer_transport(profile_host)
        endpoint_origin = validate_bearer_transport(endpoint_base_url)
    except ValueError as exc:
        raise AuthProfileError(str(exc)) from exc
    if profile_origin != endpoint_origin:
        def _display(origin):
            scheme, host, port = origin
            default = 443 if scheme == "https" else 80
            return f"{scheme}://{host}" + (f":{port}" if port != default else "")
        raise AuthProfileError(
            f"Databricks auth profile {name!r} is bound to "
            f"{_display(profile_origin)}, not {_display(endpoint_origin)}")

    tok = sect.get("token")
    # a PAT is usable as-is. an OAuth profile has auth_type set and either no
    # token or a stale one, so prefer the CLI there.
    if tok and not sect.get("auth_type"):
        return tok
    try:
        out = subprocess.run(["databricks", "auth", "token", "-p", name],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            cli_token = _json.loads(out.stdout).get("access_token")
            if cli_token:
                return cli_token
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise AuthProfileError(
            f"Databricks auth profile {name!r} could not mint a token: {exc}") \
            from exc
    raise AuthProfileError(
        f"Databricks auth profile {name!r} did not resolve to a token")


def _token(cfg: EndpointConfig) -> str | None:
    if cfg.auth_profile:
        return _token_from_profile(cfg.auth_profile, cfg.base_url)
    tok = os.environ.get(cfg.auth_token_env) or None
    if tok:
        validate_bearer_transport(cfg.base_url)
    return tok


def run(rc: RunConfig, token_override: str | None = None,
        quiet: bool = False) -> dict:
    # Freeze all nested request/policy configuration and re-run validation in
    # case a caller mutated the dataclass after constructing it.
    rc = dataclasses.replace(
        rc,
        endpoint=copy.deepcopy(rc.endpoint),
        acceptance_targets=copy.deepcopy(rc.acceptance_targets),
        pricing=copy.deepcopy(rc.pricing))
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
        workload_id = _resolved_workload_id(original_rc, inputs)
        logical_run_id, execution_id, artifact_id = _execution_ids(original_rc)
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
            # Parse policy-bearing workload input before auth lookup or any
            # endpoint call. Profile construction centrally validates embedded
            # acceptance targets; prompt loading likewise fails malformed
            # inputs before a benchmark can begin.
            if work_rc.profile_path:
                prof.Profile.from_json(work_rc.profile_path)
            else:
                from .prompts import load_prompts
                load_prompts(work_rc.prompts_file)
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
            if original_rc.capture_endpoint_metadata:
                from .endpoint_meta import fetch_endpoint_metadata
                endpoint_meta = fetch_endpoint_metadata(
                    ecfg.base_url, ecfg.path, token, timeout=5.0)
            artifact.update_start(
                status="target-snapshotted",
                endpoint_metadata=endpoint_meta, network_path=net_path)

            # ---- optional unloaded sizing pass ---------------------------
            effective_rc = work_rc
            if sizing_requested is not None:
                effective_rc = _size_for_concurrency(
                    work_rc, ecfg, client, artifact.append, quiet,
                    workload_id, execution_id)
            derived_qps = (effective_rc.qps_base
                           if sizing_requested is not None else None)

            # Capture the complete unsharded schedule, then select this
            # process's globally indexed subset. Exact binary identities are
            # persisted before calibration and measured replay traffic.
            if effective_rc.timestamps_file:
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
            workload = _PreparedWorkload(effective_rc, total_n)
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
                if row.get("ok") and row.get("prompt_tokens"):
                    chars_total += plan["chars"]
                    ptok_total += row["prompt_tokens"]

            # Recalibrate only synthetic material. The original input cannot
            # change this run: workload parsing is already on private bytes.
            calibration = {
                "requests": calib_n,
                "reported_prompt_tokens": ptok_total,
                "cpt_initial": effective_rc.cpt,
                "cpt_final": effective_rc.cpt,
            }
            if not prompts_mode and ptok_total:
                old_cpt = workload.mat.cpt
                new_cpt = calibrate_cpt(old_cpt, chars_total, ptok_total)
                calibration["cpt_final"] = new_cpt
                if not quiet:
                    print(f"[runner] cpt calibrated {old_cpt:.2f} -> "
                          f"{new_cpt:.2f} (from {ptok_total} reported "
                          "prompt tokens)")
                workload.set_cpt(new_cpt)
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
                parameters = inspect.signature(
                    client.send).parameters.values()
                supports_scheduled_clock = any(
                    p.name == "scheduled_monotonic"
                    or p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in parameters)
            except (TypeError, ValueError):
                supports_scheduled_clock = True

            pending: dict = {}
            with ThreadPoolExecutor(
                    max_workers=effective_rc.max_concurrency) as ex:
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
                            dispatch_lag_ms=lag_ms))
                        prog.done(None)
                        prog.paint()
                        continue

                    send_args = (
                        plan["messages"], plan["max_output"], rid,
                        float(ts[local_i]), lag_ms, plan["intended"],
                        plan["chars"])
                    if supports_scheduled_clock:
                        fut = ex.submit(
                            client.send, *send_args,
                            scheduled_monotonic=target)
                    else:  # compatibility for narrow third-party adapters
                        fut = ex.submit(client.send, *send_args)
                    fut.add_done_callback(_progress_done)
                    pending[fut] = (
                        rid, plan, body_hash, float(ts[local_i]), lag_ms)
                    prog.paint()

                for fut in as_completed(list(pending)):
                    artifact.append(_collect(fut, pending[fut]))
                    prog.paint()
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
            replay_rows = [
                row for row in artifact.read_rows()
                if row.get("phase") == "replay"]
            summary = summarize(
                replay_rows, schedule_meta=sched_meta, run_meta=meta,
                acceptance=acceptance,
                ttft_definition=effective_rc.ttft_definition,
                pricing=effective_rc.pricing,
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
