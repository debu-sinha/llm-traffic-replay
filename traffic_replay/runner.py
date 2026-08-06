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

import dataclasses
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import profile as prof
from .client import (EndpointClient, EndpointConfig,
                     validate_bearer_transport)
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
        if bool(self.profile_path) == bool(self.prompts_file):
            raise ValueError("set exactly one of profile_path or prompts_file")
        if not isinstance(self.endpoint, dict):
            raise ValueError("endpoint must be an object")
        if not str(self.endpoint.get("base_url") or "").strip() \
                or not str(self.endpoint.get("path") or "").strip():
            raise ValueError("endpoint needs non-empty base_url and path")
        from urllib.parse import urlparse
        parsed = urlparse(str(self.endpoint["base_url"]))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("endpoint base_url must be an http(s) URL with a host")
        if not str(self.endpoint["path"]).startswith("/"):
            raise ValueError("endpoint path must start with /")
        extra_body = self.endpoint.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError("endpoint extra_body must be an object")
        try:
            json.dumps(extra_body or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("endpoint extra_body must be JSON-serializable") from exc
        for timeout_name in ("connect_timeout_s", "read_timeout_s"):
            timeout = self.endpoint.get(timeout_name)
            if timeout is not None and (
                    not math.isfinite(float(timeout)) or float(timeout) <= 0):
                raise ValueError(f"endpoint {timeout_name} must be positive and finite")
        retries = self.endpoint.get("max_retries")
        if retries is not None and (
                not isinstance(retries, int) or isinstance(retries, bool)
                or retries < 0):
            raise ValueError("endpoint max_retries must be a non-negative integer")
        temperature = self.endpoint.get("temperature")
        if temperature is not None and not math.isfinite(float(temperature)):
            raise ValueError("endpoint temperature must be finite")
        if self.sizing_concurrency is not None and self.concurrency is not None:
            raise ValueError("set sizing_concurrency, not both it and legacy concurrency")
        if self.sizing_concurrency is None and self.concurrency is not None:
            self.sizing_concurrency = self.concurrency
            self.concurrency = None
        if self.sizing_concurrency is not None \
                and (not isinstance(self.sizing_concurrency, int)
                     or self.sizing_concurrency <= 0):
            raise ValueError("sizing_concurrency must be a positive integer")
        if not isinstance(self.duration_s, int) or self.duration_s <= 0:
            raise ValueError("duration_s must be a positive integer")
        rates = (self.qps_base, self.qps_burst, self.qps_min, self.qps_max)
        if any(not math.isfinite(float(x)) or float(x) <= 0 for x in rates):
            raise ValueError("qps_base/qps_burst/qps_min/qps_max must be positive and finite")
        if self.qps_min > self.qps_max:
            raise ValueError("qps_min cannot exceed qps_max")
        if not (self.qps_min <= self.qps_base <= self.qps_max):
            raise ValueError("qps_base must be between qps_min and qps_max")
        if not (self.qps_min <= self.qps_burst <= self.qps_max):
            raise ValueError("qps_burst must be between qps_min and qps_max")
        if not math.isfinite(float(self.rate_scale)) \
                or not (0 < self.rate_scale <= 1):
            raise ValueError("rate_scale must be in (0, 1]")
        if not isinstance(self.max_concurrency, int) or self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if self.max_pending_requests is not None and (
                not isinstance(self.max_pending_requests, int)
                or self.max_pending_requests <= 0):
            raise ValueError("max_pending_requests must be a positive integer")
        if not math.isfinite(float(self.cpt)) or self.cpt <= 0:
            raise ValueError("cpt must be positive and finite")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) \
                or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.calibrate_n, int) or self.calibrate_n < 0:
            raise ValueError("calibrate_n must be a non-negative integer")
        if not isinstance(self.shard_total, int) or self.shard_total <= 0 \
                or not isinstance(self.shard_index, int) \
                or not (0 <= self.shard_index < self.shard_total):
            raise ValueError("need 0 <= shard_index < shard_total")
        if not isinstance(self.pool_docs_per_bucket, int) \
                or self.pool_docs_per_bucket <= 0:
            raise ValueError("pool_docs_per_bucket must be a positive integer")
        if not math.isfinite(float(self.pool_zipf_s)) or self.pool_zipf_s <= 0:
            raise ValueError("pool_zipf_s must be positive and finite")
        if not isinstance(self.max_output_tokens_cap, int) \
                or self.max_output_tokens_cap <= 0:
            raise ValueError("max_output_tokens_cap must be a positive integer")
        if self.ttft_definition not in ("first_content", "first_visible"):
            raise ValueError("ttft_definition must be first_content or first_visible")
        if self.acceptance_targets is not None \
                and not isinstance(self.acceptance_targets, dict):
            raise ValueError("acceptance_targets must be an object")
        if self.pricing is not None and not isinstance(self.pricing, dict):
            raise ValueError("pricing must be an object")
        if not math.isfinite(float(self.start_tolerance_s)) \
                or self.start_tolerance_s < 0:
            raise ValueError("start_tolerance_s must be non-negative and finite")
        if self.start_at_unix is not None \
                and not math.isfinite(float(self.start_at_unix)):
            raise ValueError("start_at_unix must be finite")
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
               request_body_sha256=body_hash)
    if plan["construction"]:
        row.update(
            constructed_target_chars=plan["construction"]["target_chars"],
            constructed_actual_chars=plan["construction"]["actual_chars"],
            constructed_error_chars=plan["construction"]["error_chars"])
    return row


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
    }
    row.update(phase=phase, global_index=plan["global_index"],
               sample_index=plan["sample_index"],
               prompt_index=plan["prompt_index"],
               request_body_sha256=body_hash)
    if plan["construction"]:
        row.update(
            constructed_target_chars=plan["construction"]["target_chars"],
            constructed_actual_chars=plan["construction"]["actual_chars"],
            constructed_error_chars=plan["construction"]["error_chars"])
    return row


def _size_for_concurrency(rc: "RunConfig", ecfg, token, out_rows: list,
                          quiet: bool, run_id: str) -> "RunConfig":
    """Derive a fixed open-loop rate from an unloaded concurrency hint.

    This does not hold concurrency. It measures unloaded service time once,
    computes ``rate = sizing_concurrency / e2e_p50``, and leaves that rate
    fixed while the endpoint slows or speeds up under load.
    """
    import numpy as _np

    from .client import EndpointClient

    probe_n = max(4, min(rc.calibrate_n, 8))
    client = EndpointClient(ecfg, token,
                            refresh=lambda: _token(ecfg))
    workload = _PreparedWorkload(rc, probe_n)

    e2e = []
    for i in range(probe_n):
        rid = _stable_request_id(run_id, i, "sizing")
        plan = workload.plan(i, rid)
        body_hash = _payload_hash(ecfg, plan["messages"], plan["max_output"])
        try:
            res = client.send(
                plan["messages"], plan["max_output"], rid, scheduled_s=0.0,
                dispatch_lag_ms=0.0, intended=plan["intended"],
                chars_sent=plan["chars"])
            d = _annotate_result(res, "sizing", plan, body_hash)
        except Exception as exc:
            d = _exception_result(
                rid, "sizing", plan, body_hash,
                f"unexpected worker exception: {type(exc).__name__}: {exc}")
        out_rows.append(d)
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
    run_id = _resolved_run_id(rc)
    sizing_requested = rc.sizing_concurrency
    sizing_local = _shard_concurrency(rc)
    load_mode = "sizing_concurrency" if sizing_requested is not None \
        else "fixed_rate"
    if token_override is not None and ecfg.auth_profile:
        raise AuthProfileError(
            "token_override cannot be combined with a named auth_profile")
    token = token_override if token_override is not None else _token(ecfg)
    client = EndpointClient(ecfg, token,
                            refresh=lambda: _token(ecfg))
    req_params = {"temperature": ecfg.temperature,
                  "max_output_tokens_cap": rc.max_output_tokens_cap,
                  "extra_body": ecfg.extra_body or {}}
    # where the client sits relative to the endpoint. this is cheap, and
    # without it a run generated from the wrong region silently folds a
    # round trip into every latency number it prints.
    net_path = None
    if rc.measure_network_path:
        from .netpath import measure_network_path
        net_path = measure_network_path(ecfg.base_url)
        if net_path and not quiet:
            print(f"[runner] network: {net_path['rtt_ms']:.0f} ms round trip "
                  f"to {net_path['endpoint_host']} "
                  f"({', '.join(net_path['endpoint_ips'][:2])})")

    endpoint_meta = None
    if rc.capture_endpoint_metadata:
        from .endpoint_meta import fetch_endpoint_metadata
        endpoint_meta = fetch_endpoint_metadata(ecfg.base_url, ecfg.path,
                                                token, timeout=5.0)

    # ---- optional unloaded sizing pass -----------------------------------
    sizing_rows: list[dict] = []
    if rc.sizing_concurrency is not None:
        rc = _size_for_concurrency(
            rc, ecfg, token, sizing_rows, quiet, run_id)
    derived_qps = rc.qps_base if sizing_requested is not None else None

    # arrival schedule is shared by both modes
    if rc.timestamps_file:
        sched = load_trace(rc.timestamps_file, duration_cap_s=rc.duration_s)
    else:
        sched = make_schedule(
            duration_s=rc.duration_s, qps_base=rc.qps_base,
            qps_burst=rc.qps_burst, qps_min=rc.qps_min, qps_max=rc.qps_max,
            rate_scale=rc.rate_scale, seed=rc.seed + 16)
    total_n = len(sched["timestamps"])
    if total_n == 0:
        raise RuntimeError("schedule produced zero arrivals; "
                           "raise rate_scale or duration")
    sched["global_indices"] = np.arange(total_n, dtype=int)
    sched["total_requests"] = total_n
    if rc.shard_total > 1:
        sched = shard(sched, rc.shard_index, rc.shard_total)
    ts = sched["timestamps"]
    global_indices = sched.get("global_indices")
    n = len(ts)
    workload = _PreparedWorkload(rc, total_n)
    m = workload.prompts_count
    p = workload.profile

    if not quiet:
        if prompts_mode:
            print(f"[runner] {n} scheduled arrivals over {rc.duration_s}s, "
                  f"replaying {m} real prompts from {rc.prompts_file}")
        else:
            print(f"[runner] {n} scheduled arrivals over {rc.duration_s}s "
                  f"(rate_scale {rc.rate_scale}), profile '{p.name}'")
            if p.label:
                print(f"[runner] profile label: {p.label}")

    results: list[dict] = list(sizing_rows)

    # ---- calibration / warmup pass (sequential, low rate) --------------
    # Calibration is extra traffic, not arrivals removed from the measured
    # schedule. Every shard uses the same global indices so cpt calibration
    # cannot change replay bodies merely because the workload was partitioned.
    calib_n = min(rc.calibrate_n, total_n)
    chars_total = 0
    ptok_total = 0
    for i in range(calib_n):
        body_rid = _stable_request_id(run_id, i, "calibration")
        rid = _stable_request_id(
            run_id, i, f"calibration-shard-{rc.shard_index}")
        # Identical calibration bodies give every shard the same cpt update;
        # the transport request id remains shard-unique for tracing.
        plan = workload.plan(i, body_rid)
        body_hash = _payload_hash(ecfg, plan["messages"], plan["max_output"])
        try:
            res = client.send(
                plan["messages"], plan["max_output"], rid, scheduled_s=0.0,
                dispatch_lag_ms=0.0, intended=plan["intended"],
                chars_sent=plan["chars"])
            d = _annotate_result(res, "calibration", plan, body_hash)
        except Exception as exc:
            d = _exception_result(
                rid, "calibration", plan, body_hash,
                f"unexpected worker exception: {type(exc).__name__}: {exc}")
        results.append(d)
        if d.get("ok") and d.get("prompt_tokens"):
            chars_total += plan["chars"]
            ptok_total += d["prompt_tokens"]

    # recalibrate chars/token only in profile mode (real prompts are fixed)
    if not prompts_mode and ptok_total:
        old_cpt = workload.mat.cpt
        new_cpt = calibrate_cpt(old_cpt, chars_total, ptok_total)
        if not quiet:
            print(f"[runner] cpt calibrated {old_cpt:.2f} -> {new_cpt:.2f} "
                  f"(from {ptok_total} reported prompt tokens)")
        workload.set_cpt(new_cpt)

    # ---- paced replay ----------------------------------------------------
    if rc.start_at_unix is not None:
        until_start = rc.start_at_unix - time.time()
        if until_start < -rc.start_tolerance_s:
            raise RuntimeError(
                f"shared start_at_unix became stale by {-until_start:.3f}s "
                "during setup; choose a later start and verify shard clocks")
        t0 = time.monotonic() + until_start
    else:
        t0 = time.monotonic() + 0.25

    from .progress import Progress
    prog = Progress(n, float(rc.duration_s), enabled=not quiet)
    pending_limit = (rc.max_pending_requests
                     if rc.max_pending_requests is not None
                     else max(rc.max_concurrency * 2, rc.max_concurrency + 1))

    def _progress_done(fut):
        if fut.cancelled():
            prog.done(None)
            return
        try:
            prog.done(fut.result())
        except Exception:
            # Collection below persists the exception as an error row. A
            # callback must never re-raise on a worker thread.
            prog.done(None)

    def _collect(fut, context):
        rid, plan, body_hash, scheduled_s, lag_ms = context
        try:
            return _annotate_result(fut.result(), "replay", plan, body_hash)
        except Exception as exc:
            return _exception_result(
                rid, "replay", plan, body_hash,
                f"unexpected worker exception: {type(exc).__name__}: {exc}",
                scheduled_s=scheduled_s, dispatch_lag_ms=lag_ms)

    pending: dict = {}
    with ThreadPoolExecutor(max_workers=rc.max_concurrency) as ex:
        for local_i in range(n):
            target = t0 + float(ts[local_i])
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            lag_ms = max((time.monotonic() - target) * 1000.0, 0.0)

            # Keep both our bookkeeping and ThreadPoolExecutor's private queue
            # bounded. Completed work is drained without blocking the paced
            # dispatcher; if the bound is still full, record an explicit
            # client-side rejection instead of silently accumulating memory.
            for done in [f for f in pending if f.done()]:
                results.append(_collect(done, pending.pop(done)))

            global_i = int(global_indices[local_i])
            rid = _stable_request_id(run_id, global_i)
            plan = workload.plan(global_i, rid)
            body_hash = _payload_hash(
                ecfg, plan["messages"], plan["max_output"])
            prog.sent()
            if len(pending) >= pending_limit:
                results.append(_exception_result(
                    rid, "replay", plan, body_hash,
                    f"client pending limit {pending_limit} reached; request "
                    "was not sent", scheduled_s=float(ts[local_i]),
                    dispatch_lag_ms=lag_ms))
                prog.done(None)
                prog.paint()
                continue

            fut = ex.submit(
                client.send, plan["messages"], plan["max_output"], rid,
                float(ts[local_i]), lag_ms, plan["intended"], plan["chars"])
            fut.add_done_callback(_progress_done)
            pending[fut] = (rid, plan, body_hash, float(ts[local_i]), lag_ms)
            prog.paint()

        for fut in as_completed(list(pending)):
            results.append(_collect(fut, pending[fut]))
            prog.paint()
    prog.finish()

    load_meta = {
        "load_mode": load_mode,
        "sizing_concurrency_requested": sizing_requested,
        "sizing_concurrency_local": sizing_local,
        "derived_qps": derived_qps,
        "run_id": run_id,
        "start_at_unix": rc.start_at_unix,
        "max_pending_requests": pending_limit,
        # This is deliberately not `concurrency_target`: a fixed-rate open
        # loop does not hold an occupancy target. Metrics still reports the
        # concurrency that actually happened as an outcome.
        "concurrency_target": None,
    }
    if prompts_mode:
        meta = {
            "input_mode": "prompts",
            "prompts_file": rc.prompts_file, "prompts_count": m,
            "endpoint_path": ecfg.path, "label": rc.label, "title": rc.title,
            "request_params": req_params, "endpoint_metadata": endpoint_meta,
            "network_path": net_path,
            "shard": f"{rc.shard_index + 1}/{rc.shard_total}",
            **load_meta,
            # identity of the thing under test. without these, compare and
            # merge cannot tell two different providers apart when both sit
            # behind the same route.
            "endpoint_base_url": ecfg.base_url,
            "endpoint_model": ecfg.model,
            "profile_path": rc.profile_path,
            "seed": rc.seed,
        }
        acceptance = rc.acceptance_targets
    else:
        meta = {
            "input_mode": "profile",
            "profile": p.name, "profile_provenance": p.provenance,
            "profile_label": p.label, "cpt_final": workload.mat.cpt,
            "endpoint_path": ecfg.path, "label": rc.label, "title": rc.title,
            "request_params": req_params, "endpoint_metadata": endpoint_meta,
            "network_path": net_path,
            "shard": f"{rc.shard_index + 1}/{rc.shard_total}",
            **load_meta,
            # identity of the thing under test. without these, compare and
            # merge cannot tell two different providers apart when both sit
            # behind the same route.
            "endpoint_base_url": ecfg.base_url,
            "endpoint_model": ecfg.model,
            "profile_path": rc.profile_path,
            "prompts_file": rc.prompts_file,
            "seed": rc.seed,
        }
        acceptance = (rc.acceptance_targets
                      or (p.extra or {}).get("acceptance_targets"))

    # name the origin, so the scorecard cannot credit the profile for numbers
    # the run config supplied. the CLI stamps its own before we get here.
    if acceptance and "targets_are" not in acceptance:
        acceptance = {**acceptance,
                      "targets_are": ("the run config" if rc.acceptance_targets
                                      else "this profile")}

    summary = summarize([r for r in results if r.get("phase") == "replay"],
                        schedule_meta=schedule_report(sched), run_meta=meta,
                        acceptance=acceptance,
                        ttft_definition=rc.ttft_definition,
                        pricing=rc.pricing,
                        concurrency_target=None)
    out = write_outputs(results, summary,
                        Path(rc.out_dir) / time.strftime("%Y%m%d-%H%M%S"),
                        rc.title)
    if not quiet:
        print(f"[runner] wrote {out}/report.html (open in a browser) "
              f"and {out}/report.md")
    return {"summary": summary, "out_dir": str(out), "results_n": len(results)}
