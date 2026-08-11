"""Pool sharded runs (merge) and compare runs side by side (compare).

Both read the standard outputs write_outputs produced (summary.json,
requests.jsonl). Nothing here re-measures: merge re-summarizes the pooled
replay rows, compare tabulates existing summaries. Keeping them out of the
run path means a laptop can aggregate results a fleet of machines produced.
"""
from __future__ import annotations

import errno
from datetime import datetime, timezone
import hashlib
import html
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import time
from urllib.parse import quote
import uuid

from . import __version__
from .artifacts import (
    sanitize_display_text,
    snapshot_source_state,
    strict_json_dumps,
)
from .config_validation import validate_rate_limits
from .json_input import json_error_detail, loads_strict
from .markdown import markdown_plain_text
from .metrics import _pct_table, summarize, write_outputs
from .schedule import MAX_EXACT_ANALYSIS_REQUEST_ROWS


_WRITING_MARKER = ".traffic-replay-writing"
_COMPLETE_MARKER = ".traffic-replay-complete"
_SUPPORTED_MANIFEST_SCHEMAS = {3}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_SHARD_RE = re.compile(r"([1-9][0-9]*)/([1-9][0-9]*)")
_QUOTA_REQUEST_PHASES = frozenset({
    "preflight", "probe", "sizing", "calibration", "replay",
})
_MAX_METADATA_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_REQUEST_JOURNAL_BYTES = 256 * 1024 * 1024
_MAX_REQUEST_JSONL_LINE_BYTES = 256 * 1024


def _regular_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _read_regular_bytes(
        path: Path, *, max_bytes: int = _MAX_METADATA_ARTIFACT_BYTES) -> bytes:
    """Read one bounded stable artifact without following a symlink."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) \
            or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(
                f"artifact {path} declares {before.st_size:,} bytes, above "
                f"the {max_bytes:,}-byte metadata limit")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"artifact was truncated while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = os.read(fd, 1)
        after = os.fstat(fd)
        if extra or _regular_identity(before) != _regular_identity(after):
            raise ValueError(f"artifact changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _measure_regular(
        path: Path, *, max_bytes: int = _MAX_METADATA_ARTIFACT_BYTES,
        max_rows: int | None = None,
        max_line_bytes: int | None = None) -> tuple[str, int, int]:
    """Return SHA-256, byte count and newline count with bounded memory."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(
                f"artifact {path} declares {before.st_size:,} bytes, above "
                f"its {max_bytes:,}-byte resource limit")
        digest = hashlib.sha256()
        size = 0
        rows = 0
        current_line_bytes = 0
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(
                    f"artifact was truncated while measuring: {path}")
            digest.update(chunk)
            size += len(chunk)
            remaining -= len(chunk)
            parts = chunk.split(b"\n")
            if len(parts) == 1:
                current_line_bytes += len(chunk)
                if max_line_bytes is not None \
                        and current_line_bytes > max_line_bytes:
                    raise ValueError(
                        f"artifact line exceeds {max_line_bytes:,} bytes: "
                        f"{path}")
            else:
                if max_line_bytes is not None and (
                        current_line_bytes + len(parts[0]) > max_line_bytes
                        or any(len(part) > max_line_bytes
                               for part in parts[1:-1])):
                    raise ValueError(
                        f"artifact line exceeds {max_line_bytes:,} bytes: "
                        f"{path}")
                rows += len(parts) - 1
                if max_rows is not None and rows > max_rows:
                    raise ValueError(
                        f"artifact exceeds the {max_rows:,}-row resource "
                        f"limit: {path}")
                current_line_bytes = len(parts[-1])
                if max_line_bytes is not None \
                        and current_line_bytes > max_line_bytes:
                    raise ValueError(
                        f"artifact line exceeds {max_line_bytes:,} bytes: "
                        f"{path}")
        extra = os.read(fd, 1)
        after = os.fstat(fd)
        if extra or _regular_identity(before) != _regular_identity(after) \
                or size != before.st_size:
            raise ValueError(f"artifact changed while measuring: {path}")
        return digest.hexdigest(), size, rows
    finally:
        os.close(fd)


def _load_json_object(path: Path, label: str) -> dict:
    try:
        value = loads_strict(_read_regular_bytes(path))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"invalid {label} in {path}: {json_error_detail(exc)}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_summary(d: Path) -> dict:
    p = d / "summary.json"
    return _load_json_object(p, "summary.json")


def _load_manifest(d: Path) -> dict | None:
    p = d / "manifest.json"
    if not p.exists():
        return None
    return _load_json_object(p, "manifest.json")


def _stable(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _schedule_identity(schedule: dict, merging: bool) -> dict:
    """Comparable schedule fields, excluding shard-local bookkeeping."""
    out = dict(schedule or {})
    for key in ("shard", "rates_describe"):
        out.pop(key, None)
    if merging:
        # Each shard owns a subset of the same parent schedule.
        out.pop("requests", None)
    return out


def _global_schedule_identity(manifest: dict) -> dict | None:
    identity = manifest.get("schedule_identity")
    if not isinstance(identity, dict):
        return None
    return {
        key: identity.get(key)
        for key in ("encoding", "global_timestamps_sha256", "global_count",
                    "global_min_s", "global_max_s")
    }


def _local_replay_identity(manifest: dict) -> dict | None:
    """Exact shard-local schedule/index identity used by comparison.

    Two shards can truthfully share one parent/global schedule while covering
    disjoint requests.  Comparing only the parent digest would therefore make
    shard 1 look interchangeable with shard 2.  The values returned here are
    authenticated against ``requests.jsonl`` before comparison uses them.
    """
    schedule = manifest.get("schedule_identity")
    index = manifest.get("index_identity")
    if not isinstance(schedule, dict) or not isinstance(index, dict):
        return None
    return {
        "schedule": {
            key: schedule.get(key)
            for key in (
                "encoding", "shard_timestamps_sha256", "shard_count",
                "shard_min_s", "shard_max_s",
            )
        },
        "index": {
            key: index.get(key)
            for key in (
                "encoding", "global_indices_sha256", "count", "min", "max",
                "global_count", "shard_index", "shard_total", "partition",
            )
        },
    }


def _declared_ttft_definition(
        title: str, summary: dict, manifest: dict) -> tuple[str | None, list[str]]:
    """Return one canonical first-event definition or declaration issues.

    Do not use an ``or`` chain here: it hides an internally contradictory
    artifact by selecting whichever declaration happens to appear first.
    Current v3 artifacts carry the definition in both measurement output and
    immutable configuration provenance, and all present declarations must
    agree.
    """
    config_identity = manifest.get("config_identity")
    config_identity = config_identity if isinstance(config_identity, dict) else {}
    effective = manifest.get("effective_config")
    effective = effective if isinstance(effective, dict) else {}
    identity_effective = config_identity.get("effective_config")
    identity_effective = (identity_effective
                          if isinstance(identity_effective, dict) else {})
    sla_definition = config_identity.get("sla_definition")
    sla_definition = sla_definition if isinstance(sla_definition, dict) else {}
    summary_run = summary.get("run")
    summary_run = summary_run if isinstance(summary_run, dict) else {}
    summary_sla = summary.get("sla")
    summary_sla = summary_sla if isinstance(summary_sla, dict) else {}
    declarations = [
        ("manifest.config_identity.sla_definition",
         sla_definition.get("ttft_definition")),
        ("manifest.config_identity.effective_config",
         identity_effective.get("ttft_definition")),
        ("manifest.effective_config", effective.get("ttft_definition")),
        ("summary.run", summary_run.get("ttft_definition")),
        ("summary.sla", summary_sla.get("ttft_definition")),
    ]
    present = [(location, value) for location, value in declarations
               if value is not None]
    issues: list[str] = []
    invalid = [(location, value) for location, value in present
               if value not in {"first_content", "first_visible"}]
    if invalid:
        detail = ", ".join(
            f"{location}={value!r}" for location, value in invalid)
        issues.append(f"invalid TTFT definition declaration for {title}: {detail}")
        return None, issues
    groups: dict[str, list[str]] = {}
    for location, value in present:
        groups.setdefault(str(value), []).append(location)
    if len(groups) > 1:
        detail = "; ".join(
            f"{value} at {', '.join(locations)}"
            for value, locations in sorted(groups.items()))
        issues.append(
            f"conflicting TTFT definition declarations inside {title}: {detail}")
        return None, issues
    if not groups:
        issues.append(
            f"missing TTFT definition declaration for {title}; a v3 artifact "
            "must bind first_content or first_visible")
        return None, issues
    return next(iter(groups)), issues


def _normalized_acceptance_policy(value: object) -> dict | None:
    if not isinstance(value, dict) or not value:
        return None
    # targets_are says where a policy came from; it does not change a target.
    return {key: item for key, item in value.items() if key != "targets_are"}


def _production_transport_evidence(summary: dict) -> dict:
    """Normalize the artifact-safe actual-versus-production transport claim."""
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    transport = run.get("transport")
    transport = transport if isinstance(transport, dict) else {}

    def nonempty_string(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    actual = nonempty_string(transport.get("connection_policy_id"))
    declared = nonempty_string(
        transport.get("production_connection_policy_declared"))
    raw_match = transport.get("production_connection_policy_match")
    match = raw_match if isinstance(raw_match, bool) else None
    raw_warning = transport.get("production_comparability_warning")
    warning = nonempty_string(raw_warning)
    assurance = nonempty_string(
        transport.get("production_connection_policy_assurance"))
    exact = bool(
        actual is not None and declared == actual and match is True
        and raw_warning is None)
    if exact:
        status = "MATCH"
        note = assurance or "explicit exact connection-policy match recorded"
    elif match is True or (
            raw_warning is not None and not isinstance(raw_warning, str)):
        status = "INCONSISTENT"
        note = (
            warning or "transport parity fields are internally inconsistent")
    else:
        status = "UNVERIFIED"
        if warning:
            note = warning
        elif actual is None:
            note = "benchmark connection policy was not recorded"
        elif declared is None:
            note = (
                "production connection behavior was not declared, so it "
                f"cannot be compared with benchmark policy {actual}")
        else:
            note = (
                f"declared production policy {declared} does not have an "
                f"explicit exact match to benchmark policy {actual}")
    return {
        "status": status,
        "exact_match": exact,
        "actual_policy_id": actual,
        "declared_production_policy": declared,
        "recorded_match": match,
        "warning": warning,
        "assurance": assurance,
        "note": note,
    }


def _execution_contract(summary: dict, manifest: dict) -> dict | None:
    """Measurement-affecting client/runtime settings omitted by request body."""
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    transport = run.get("transport")
    if isinstance(transport, dict):
        # Production parity is an operator assertion about a different
        # client, not a behavior of this benchmark process. Compare the
        # benchmark's actual wire contract here, and qualify production
        # parity independently in the warning gate/matrix below.
        transport = {
            key: value for key, value in transport.items()
            if key not in {
                "production_connection_policy_declared",
                "production_connection_policy_match",
                "production_connection_policy_assurance",
                "production_comparability_warning",
            }
        }
    else:
        transport = None
    effective = manifest.get("effective_config")
    if not isinstance(effective, dict):
        config_identity = manifest.get("config_identity")
        config_identity = config_identity if isinstance(config_identity, dict) else {}
        effective = config_identity.get("effective_config")
    effective = effective if isinstance(effective, dict) else {}
    endpoint = effective.get("endpoint")
    endpoint = endpoint if isinstance(endpoint, dict) else {}
    value = {
        "transport": transport,
        "max_concurrency": effective.get("max_concurrency"),
        "max_pending_requests": (
            run.get("max_pending_requests")
            if run.get("max_pending_requests") is not None
            else effective.get("max_pending_requests")),
        "endpoint_client": {
            key: endpoint.get(key)
            for key in (
                "connect_timeout_s", "read_timeout_s", "total_timeout_s",
                "max_retries", "include_usage",
            )
            if endpoint.get(key) is not None
        },
    }
    return value if any(item not in (None, {}) for item in value.values()) else None


def _declared_acceptance_policy(
        title: str, summary: dict, manifest: dict) -> tuple[dict | None, list[str]]:
    """Reconcile every persisted declaration of a source SLA policy."""
    config_identity = manifest.get("config_identity")
    config_identity = config_identity if isinstance(config_identity, dict) else {}
    effective = manifest.get("effective_config")
    effective = effective if isinstance(effective, dict) else {}
    identity_effective = config_identity.get("effective_config")
    identity_effective = (identity_effective
                          if isinstance(identity_effective, dict) else {})
    sla_definition = config_identity.get("sla_definition")
    sla_definition = sla_definition if isinstance(sla_definition, dict) else {}
    summary_sla = summary.get("sla")
    summary_sla = summary_sla if isinstance(summary_sla, dict) else {}
    declarations = [
        ("manifest.config_identity.sla_definition.acceptance_config",
         sla_definition.get("acceptance_config")),
        ("manifest.config_identity.effective_config.acceptance_targets",
         identity_effective.get("acceptance_targets")),
        ("manifest.effective_config.acceptance_targets",
         effective.get("acceptance_targets")),
        ("summary.sla.acceptance_config",
         summary_sla.get("acceptance_config")),
    ]
    present = [
        (location, _normalized_acceptance_policy(value))
        for location, value in declarations
        if value is not None
    ]
    malformed = [location for location, value in present if value is None]
    if malformed:
        return None, [
            f"invalid acceptance policy declaration for {title}: "
            + ", ".join(malformed) + " must be a non-empty object"
        ]
    groups: dict[str, tuple[dict, list[str]]] = {}
    for location, value in present:
        assert value is not None
        key = _stable(value)
        if key not in groups:
            groups[key] = (value, [])
        groups[key][1].append(location)
    if len(groups) > 1:
        detail = "; ".join(
            f"{policy} at {', '.join(locations)}"
            for policy, (_value, locations) in sorted(groups.items()))
        return None, [
            f"conflicting acceptance policy declarations inside {title}: "
            f"{detail}"
        ]
    return (next(iter(groups.values()))[0] if groups else None), []


def _compatibility_issues(dirs: list[Path], summaries: list[dict],
                          manifests: list[dict | None], *,
                          merging: bool) -> list[str]:
    """Facts that make pooled or side-by-side latency incomparable.

    Compare deliberately allows different endpoints; merge does not. Both
    require immutable code provenance and the same workload definition.
    Missing provenance is an incompatibility, not evidence that values match.
    """
    titles = [_run_title(d, s) for d, s in zip(dirs, summaries)]
    issues: list[str] = []
    missing = [t for t, m in zip(titles, manifests) if m is None]
    if missing:
        issues.append(
            f"missing manifest.json for {', '.join(missing)}; workload and "
            "code identity cannot be proven")

    present = [(t, s, m) for t, s, m in zip(titles, summaries, manifests)
               if m is not None]
    dirty = [t for t, _s, m in present if m.get("git_dirty") is not False]
    if dirty:
        issues.append(
            f"{', '.join(dirty)} has dirty or unknown Git state; its source "
            "cannot be reconstructed from a commit")
    invalid_aggregates = [
        t for t, source_summary, _m in present
        if (source_summary.get("run") or {}).get("aggregation_valid") is False]
    if invalid_aggregates:
        issues.append(
            f"{', '.join(invalid_aggregates)} is an explicitly INVALID "
            "aggregate and cannot be treated as benchmark evidence")
    for title, source_summary, manifest in present:
        run = source_summary.get("run") or {}
        for label, summary_value, manifest_value in (
                ("harness version", source_summary.get("harness_version"),
                 manifest.get("harness_version")),
                ("latency basis", source_summary.get("latency_basis"),
                 manifest.get("latency_basis")),
                ("endpoint path", run.get("endpoint_path"),
                 manifest.get("endpoint_path")),
                ("endpoint model", run.get("endpoint_model"),
                 manifest.get("endpoint_model")),
                ("input mode", run.get("input_mode"),
                 manifest.get("input_mode"))):
            if (summary_value is not None and manifest_value is not None
                    and summary_value != manifest_value):
                issues.append(
                    f"{title} summary and manifest disagree on {label} "
                    f"({_stable(summary_value)} vs {_stable(manifest_value)})")

    def check(label, getter, *, required=True, detail=None):
        values = [(t, getter(s, m)) for t, s, m in present]
        absent = [t for t, v in values if v is None]
        have = [(t, v) for t, v in values if v is not None]
        if (required or have) and absent:
            issues.append(f"missing {label} for {', '.join(absent)}")
        groups = {}
        for title, value in have:
            groups.setdefault(_stable(value), []).append(title)
        if len(groups) > 1:
            desc = "; ".join(f"{', '.join(ts)}={value}"
                             for value, ts in groups.items())
            issues.append((detail or f"different {label}") + f": {desc}")

    check("Git commit", lambda _s, m: m.get("git_commit"))
    check("harness version",
          lambda s, m: m.get("harness_version") or s.get("harness_version"),
          detail=("different harness versions; latency definitions can change "
                  "between releases, including whether TCP/TLS is measured"))
    check("latency basis",
          lambda s, m: m.get("latency_basis") or s.get("latency_basis"))
    check("input mode", lambda _s, m: m.get("input_mode"))
    check("profile or prompts SHA-256",
          lambda _s, m: m.get("profile_sha256")
          or m.get("profile_sha256_16"))
    check("workload identity", lambda _s, m: m.get("workload_id"))
    check("sampling seed", lambda _s, m: m.get("seed"))
    check("request parameters", lambda _s, m: m.get("request_params"))
    check("arrival schedule",
          lambda s, m: (_global_schedule_identity(m)
                        or _schedule_identity(m.get("schedule")
                                              or s.get("schedule") or {},
                                              merging)
                        or None))
    check("load mode", lambda _s, m: m.get("load_mode"), required=False)
    check("client execution contract", _execution_contract, required=False,
          detail=("different client execution contracts; timeout, retry, "
                  "stream-usage, transport, or client saturation settings "
                  "can change measured latency and failure behavior"))
    definitions: list[tuple[str, str | None]] = []
    policies: list[tuple[str, dict | None]] = []
    for title, source_summary, manifest in present:
        definition, declaration_issues = _declared_ttft_definition(
            title, source_summary, manifest)
        definitions.append((title, definition))
        issues.extend(declaration_issues)
        policy, policy_issues = _declared_acceptance_policy(
            title, source_summary, manifest)
        policies.append((title, policy))
        issues.extend(policy_issues)
    definition_groups: dict[str, list[str]] = {}
    for title, definition in definitions:
        if definition is not None:
            definition_groups.setdefault(definition, []).append(title)
    if len(definition_groups) > 1:
        detail = "; ".join(
            f"{', '.join(group_titles)}={definition}"
            for definition, group_titles in sorted(definition_groups.items()))
        issues.append("different TTFT definitions: " + detail)
    if merging:
        present_policies = [(title, value) for title, value in policies
                            if value is not None]
        absent_policies = [title for title, value in policies
                           if value is None]
        if present_policies and absent_policies:
            issues.append(
                "missing acceptance policy for " + ", ".join(absent_policies))
        groups: dict[str, list[str]] = {}
        for title, value in present_policies:
            groups.setdefault(_stable(value), []).append(title)
        if len(groups) > 1:
            detail = "; ".join(
                f"{', '.join(group_titles)}={value}"
                for value, group_titles in groups.items())
            issues.append("different acceptance policies: " + detail)
    if not merging:
        check("local replay schedule/index identity",
              lambda _s, m: _local_replay_identity(m))
    if merging:
        check("endpoint identity", lambda _s, m: ({
            "base_url": m.get("endpoint_base_url"),
            "model": m.get("endpoint_model"),
            "path": m.get("endpoint_path"),
        } if any((m.get("endpoint_base_url"), m.get("endpoint_model"),
                  m.get("endpoint_path"))) else None))
    return issues


def _run_title(d: Path, summ: dict) -> str:
    run = summ.get("run")
    title = run.get("title") if isinstance(run, dict) else None
    return str(title) if title not in (None, "") else d.name


def _has_path(path: Path) -> bool:
    """Like lexists(): broken symlinks are still security-relevant paths."""
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _require_regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _artifact_declarations(manifest: dict, d: Path) -> dict[str, dict]:
    """Normalize supported artifact-integrity declarations.

    Earlier producers in the field used both a digest-only mapping and the
    richer ``artifacts`` mapping. The current shape is per filename with
    ``sha256``, ``bytes`` and (for JSONL) ``row_count``. If more than one
    representation is present they must agree rather than silently choosing
    one.
    """
    declarations: dict[str, dict] = {}

    def add(name, metadata, source):
        if not isinstance(name, str) or not name or name in (".", "..") \
                or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(
                f"unsafe artifact name in {source} for {d}: {name!r}")
        if isinstance(metadata, str):
            metadata = {"sha256": metadata}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"invalid artifact metadata for {name!r} in {source} for {d}")
        digest = metadata.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"invalid SHA-256 for artifact {name!r} in {source} for {d}")
        normalized = {"sha256": digest.lower()}
        size = metadata.get("bytes", metadata.get("size_bytes"))
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"invalid byte count for artifact {name!r} in {source} "
                    f"for {d}")
            normalized["bytes"] = size
        rows = metadata.get("row_count")
        if rows is not None:
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
                raise ValueError(
                    f"invalid row_count for artifact {name!r} in {source} "
                    f"for {d}")
            normalized["row_count"] = rows
        if name == "requests.jsonl":
            if size is not None and size > _MAX_REQUEST_JOURNAL_BYTES:
                raise ValueError(
                    f"artifact {name!r} in {source} for {d} declares "
                    f"{size:,} bytes, above the "
                    f"{_MAX_REQUEST_JOURNAL_BYTES:,}-byte journal limit")
            if rows is not None and rows > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
                raise ValueError(
                    f"artifact {name!r} in {source} for {d} declares "
                    f"{rows:,} rows, above the exact-analysis limit of "
                    f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}")
        elif size is not None and size > _MAX_METADATA_ARTIFACT_BYTES:
            raise ValueError(
                f"artifact {name!r} in {source} for {d} declares {size:,} "
                f"bytes, above the {_MAX_METADATA_ARTIFACT_BYTES:,}-byte "
                "metadata limit")
        old = declarations.get(name)
        if old is not None and old != normalized:
            raise ValueError(
                f"conflicting integrity metadata for artifact {name!r} in {d}")
        declarations[name] = normalized

    for field in ("artifact_sha256", "artifact_hashes"):
        if field not in manifest:
            continue
        block = manifest[field]
        if not isinstance(block, dict):
            raise ValueError(f"{field} must be an object in {d / 'manifest.json'}")
        for name, metadata in block.items():
            add(name, metadata, field)

    if "artifacts" in manifest:
        block = manifest["artifacts"]
        if not isinstance(block, dict):
            raise ValueError(
                f"artifacts must be an object in {d / 'manifest.json'}")
        for name, metadata in block.items():
            add(name, metadata, "artifacts")
    return declarations


def _verify_artifacts(d: Path, manifest: dict,
                      required: tuple[str, ...]) -> None:
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError(
            f"manifest for {d} must contain a v3 artifacts object")
    declarations = _artifact_declarations(manifest, d)
    missing = [name for name in required if name not in declarations]
    if missing:
        raise ValueError(
            f"manifest for {d} is missing required artifact integrity "
            f"entries: {', '.join(missing)}")
    if "requests.jsonl" in required \
            and "row_count" not in declarations["requests.jsonl"]:
        raise ValueError(
            f"manifest for {d} must declare requests.jsonl row_count")
    without_sizes = [name for name, metadata in declarations.items()
                     if "bytes" not in metadata]
    if without_sizes:
        raise ValueError(
            f"manifest for {d} must declare artifact byte counts for: "
            + ", ".join(without_sizes))
    for name, expected in declarations.items():
        path = d / name
        request_journal = name == "requests.jsonl"
        actual, actual_bytes, actual_rows = _measure_regular(
            path,
            max_bytes=(_MAX_REQUEST_JOURNAL_BYTES if request_journal
                       else _MAX_METADATA_ARTIFACT_BYTES),
            max_rows=(MAX_EXACT_ANALYSIS_REQUEST_ROWS
                      if request_journal else None),
            max_line_bytes=(_MAX_REQUEST_JSONL_LINE_BYTES
                            if request_journal else None),
        )
        if not hmac.compare_digest(actual, expected["sha256"]):
            raise ValueError(
                f"artifact SHA-256 mismatch for {path}: expected "
                f"{expected['sha256']}, got {actual}")
        if actual_bytes != expected["bytes"]:
            raise ValueError(
                f"artifact byte count mismatch for {path}: expected "
                f"{expected['bytes']}, got {actual_bytes}")
        if "row_count" in expected:
            if actual_rows != expected["row_count"]:
                raise ValueError(
                    f"artifact row count mismatch for {path}: expected "
                    f"{expected['row_count']}, got {actual_rows}")


def _identity_count(value, label: str, d: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {label} for {d}: {value!r}")
    return value


def _identity_float(value, label: str, d: Path, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"invalid {label} for {d}: {value!r}")
    return float(value)


def _identity_digest(value, label: str, d: Path) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid {label} for {d}: {value!r}")
    return value.lower()


def _validate_identity_shapes(d: Path, manifest: dict) -> None:
    schedule_meta = manifest.get("schedule")
    if not isinstance(schedule_meta, dict):
        raise ValueError(f"manifest for {d} is missing schedule object")
    schedule = manifest.get("schedule_identity")
    if not isinstance(schedule, dict):
        raise ValueError(f"manifest for {d} is missing schedule_identity")
    if schedule.get("encoding") != "float64-le-seconds-from-run-start":
        raise ValueError(f"invalid schedule_identity.encoding for {d}")
    _identity_digest(schedule.get("global_timestamps_sha256"),
                     "schedule_identity.global_timestamps_sha256", d)
    _identity_digest(schedule.get("shard_timestamps_sha256"),
                     "schedule_identity.shard_timestamps_sha256", d)
    global_count = _identity_count(
        schedule.get("global_count"), "schedule_identity.global_count", d)
    shard_count = _identity_count(
        schedule.get("shard_count"), "schedule_identity.shard_count", d)
    if shard_count > global_count:
        raise ValueError(f"schedule_identity shard_count exceeds global_count for {d}")
    global_min = _identity_float(
        schedule.get("global_min_s"), "schedule_identity.global_min_s", d,
        allow_none=True)
    global_max = _identity_float(
        schedule.get("global_max_s"), "schedule_identity.global_max_s", d,
        allow_none=True)
    shard_min = _identity_float(
        schedule.get("shard_min_s"), "schedule_identity.shard_min_s", d,
        allow_none=True)
    shard_max = _identity_float(
        schedule.get("shard_max_s"), "schedule_identity.shard_max_s", d,
        allow_none=True)
    for label, count, low, high in (
            ("global", global_count, global_min, global_max),
            ("shard", shard_count, shard_min, shard_max)):
        if ((count == 0 and (low is not None or high is not None))
                or (count > 0 and (low is None or high is None))):
            raise ValueError(
                f"schedule_identity {label} count/min/max disagree for {d}")
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"schedule_identity {label}_min_s exceeds {label}_max_s for {d}")

    index = manifest.get("index_identity")
    if not isinstance(index, dict):
        raise ValueError(f"manifest for {d} is missing index_identity")
    if index.get("encoding") != "int64-le":
        raise ValueError(f"invalid index_identity.encoding for {d}")
    _identity_digest(index.get("global_indices_sha256"),
                     "index_identity.global_indices_sha256", d)
    count = _identity_count(index.get("count"), "index_identity.count", d)
    index_global_count = _identity_count(
        index.get("global_count"), "index_identity.global_count", d)
    shard_index = _identity_count(
        index.get("shard_index"), "index_identity.shard_index", d)
    shard_total = _identity_count(
        index.get("shard_total"), "index_identity.shard_total", d)
    if shard_total <= 0 or shard_index >= shard_total:
        raise ValueError(f"invalid index_identity shard index/total for {d}")
    expected_partition = "unsharded" if shard_total == 1 \
        else "round_robin_modulo"
    diagnostic_subset = index.get("partition") == "diagnostic_observed_subset"
    aggregation = manifest.get("aggregation")
    forced_diagnostic = isinstance(aggregation, dict) \
        and aggregation.get("forced") is True
    if diagnostic_subset and not forced_diagnostic:
        raise ValueError(
            f"diagnostic_observed_subset is allowed only on an explicitly "
            f"forced aggregate for {d}")
    if not diagnostic_subset and index.get("partition") != expected_partition:
        raise ValueError(f"invalid index_identity.partition for {d}")
    low = index.get("min")
    high = index.get("max")
    if count == 0:
        if low is not None or high is not None:
            raise ValueError(f"index_identity count/min/max disagree for {d}")
    else:
        low = _identity_count(low, "index_identity.min", d)
        high = _identity_count(high, "index_identity.max", d)
        if low > high:
            raise ValueError(f"index_identity min exceeds max for {d}")
    if count != shard_count or index_global_count != global_count:
        raise ValueError(
            f"schedule_identity and index_identity counts disagree for {d}")
    parsed_index, parsed_total = _parse_shard(manifest, d)
    if parsed_index != shard_index or parsed_total != shard_total:
        raise ValueError(
            f"shard i/n metadata and index_identity disagree for {d}")


def _validate_manifest_identity(d: Path, manifest: dict) -> None:
    try:
        _logical_run_id(manifest)
    except ValueError as exc:
        raise ValueError(f"{exc} in {d}") from exc
    required = {
        "workload_id": manifest.get("workload_id"),
        "logical_run_id": manifest.get("logical_run_id"),
        "execution_id": manifest.get("execution_id"),
        "artifact_id": manifest.get("artifact_id"),
    }
    missing = [name for name, value in required.items()
               if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError(
            f"manifest for {d} is missing required non-empty identity fields: "
            + ", ".join(missing))
    _validate_identity_shapes(d, manifest)


def _verify_run_completion_marker(d: Path, manifest: dict) -> None:
    """Require the v3 marker to bind the manifest and request journal."""
    completion = _load_json_object(
        d / _COMPLETE_MARKER, "completion marker")
    if completion.get("status") != "complete":
        raise ValueError(f"completion marker status is not complete for {d}")
    if completion.get("artifact_id") != manifest["artifact_id"]:
        raise ValueError(
            f"completion marker artifact_id disagrees with manifest for {d}")
    actual_manifest, actual_bytes, _rows = _measure_regular(
        d / "manifest.json")
    expected_manifest = _identity_digest(
        completion.get("manifest_sha256"),
        "completion marker manifest_sha256", d)
    if not hmac.compare_digest(actual_manifest, expected_manifest):
        raise ValueError(f"completion marker manifest SHA-256 mismatch for {d}")
    declared_bytes = completion.get("manifest_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) \
            or declared_bytes != actual_bytes:
        raise ValueError(
            f"completion marker manifest byte count mismatch for {d}")
    request_metadata = _artifact_declarations(
        manifest, d)["requests.jsonl"]
    declared_rows = completion.get("request_rows")
    if isinstance(declared_rows, bool) or not isinstance(declared_rows, int) \
            or declared_rows != request_metadata["row_count"]:
        raise ValueError(
            f"completion marker request_rows disagrees with manifest-bound "
            f"requests.jsonl for {d}")


def _require_run_dir(d: Path, need: str) -> dict:
    try:
        info = d.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"input run dir not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"input run dir not found: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(
            f"input run is still being written and cannot be trusted: {d}")
    _require_regular(d / _COMPLETE_MARKER, "completion marker")
    _require_regular(d / need, need)
    _require_regular(d / "manifest.json", "manifest.json")
    _require_regular(d / "requests.jsonl", "requests.jsonl")
    manifest = _load_manifest(d)
    assert manifest is not None
    schema = manifest.get("manifest_schema_version")
    if isinstance(schema, bool) or schema not in _SUPPORTED_MANIFEST_SCHEMAS:
        supported = ", ".join(str(x) for x in sorted(
            _SUPPORTED_MANIFEST_SCHEMAS))
        raise ValueError(
            f"unsupported manifest schema {schema!r} in {d}; supported: "
            f"{supported}")
    _validate_manifest_identity(d, manifest)
    required_artifacts = ("summary.json", "requests.jsonl")
    _verify_artifacts(d, manifest, required_artifacts)
    _verify_run_completion_marker(d, manifest)
    if need == "summary.json":
        local_requests = manifest["schedule"].get("requests")
        shard_count = manifest["schedule_identity"]["shard_count"]
        if local_requests is not None and (
                isinstance(local_requests, bool)
                or not isinstance(local_requests, int)
                or local_requests != shard_count):
            raise ValueError(
                f"schedule.requests and exact shard identity count disagree "
                f"for {d}")
    return manifest


def _validated_input_dirs(input_dirs, need: str, operation: str) \
        -> tuple[list[Path], list[dict]]:
    dirs = [Path(value) for value in input_dirs]
    if len(dirs) < 2:
        raise ValueError(f"{operation} requires at least two distinct run dirs")
    manifests = []
    seen: dict[tuple[int, int], Path] = {}
    seen_artifacts: dict[str, Path] = {}
    for d in dirs:
        manifest = _require_run_dir(d, need)
        identity = d.stat()
        key = (identity.st_dev, identity.st_ino)
        if key in seen:
            raise ValueError(
                f"duplicate input run dir: {d} is the same directory as "
                f"{seen[key]}")
        seen[key] = d
        artifact_id = manifest["artifact_id"]
        if artifact_id in seen_artifacts:
            raise ValueError(
                f"duplicate input artifact_id {artifact_id!r}: {d} and "
                f"{seen_artifacts[artifact_id]}")
        seen_artifacts[artifact_id] = d
        manifests.append(manifest)
    return dirs, manifests


def _scan_request_journal(path: Path, expected: dict, visitor) -> None:
    """Strictly stream one bounded manifest-bound JSONL journal."""
    expected_bytes = expected.get("bytes")
    expected_rows = expected.get("row_count")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) \
            or not 0 <= expected_bytes <= _MAX_REQUEST_JOURNAL_BYTES:
        raise ValueError(f"invalid bounded request byte declaration for {path}")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) \
            or not 0 <= expected_rows <= MAX_EXACT_ANALYSIS_REQUEST_ROWS:
        raise ValueError(f"invalid bounded request row declaration for {path}")
    expected_digest = expected.get("sha256")
    if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(
            expected_digest):
        raise ValueError(f"invalid request SHA-256 declaration for {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    rows = 0
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        if before.st_size != expected_bytes:
            raise ValueError(
                f"artifact byte count mismatch for {path}: expected "
                f"{expected_bytes}, got {before.st_size}")
        with os.fdopen(fd, "rb") as handle:
            fd = -1                 # fdopen owns it from here
            while True:
                raw = handle.readline(_MAX_REQUEST_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                rows += 1
                if rows > expected_rows \
                        or rows > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
                    raise ValueError(
                        f"request journal exceeds its {expected_rows:,}-row "
                        f"declaration in {path}")
                if len(raw) > _MAX_REQUEST_JSONL_LINE_BYTES:
                    raise ValueError(
                        f"request journal line {rows} exceeds the "
                        f"{_MAX_REQUEST_JSONL_LINE_BYTES:,}-byte limit in "
                        f"{path}")
                if not raw.endswith(b"\n"):
                    raise ValueError(
                        f"request journal line {rows} is not newline "
                        f"terminated in {path}")
                digest.update(raw)
                size += len(raw)
                if size > expected_bytes:
                    raise ValueError(
                        f"request journal exceeds its declared byte count in "
                        f"{path}")
                if not raw.strip():
                    raise ValueError(
                        f"blank JSONL record in {path} line {rows}")
                try:
                    row = loads_strict(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {rows}: "
                        f"{json_error_detail(exc)}") from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"requests.jsonl line {rows} is not an object in "
                        f"{path.parent}")
                visitor(row, rows)
            after = os.fstat(handle.fileno())
        if _regular_identity(before) != _regular_identity(after):
            raise ValueError(f"request journal changed while reading: {path}")
        if rows != expected_rows:
            raise ValueError(
                f"artifact row count mismatch for {path}: expected "
                f"{expected_rows}, got {rows}")
        if size != expected_bytes:
            raise ValueError(
                f"artifact byte count mismatch for {path}: expected "
                f"{expected_bytes}, got {size}")
        actual_digest = digest.hexdigest()
        if not hmac.compare_digest(actual_digest, expected_digest.lower()):
            raise ValueError(
                f"artifact SHA-256 mismatch for {path}: expected "
                f"{expected_digest}, got {actual_digest}")
    finally:
        if fd >= 0:
            os.close(fd)


def _request_rows(d: Path, manifest: dict) -> list[dict]:
    """Materialize only a pre-bounded, manifest-bound request journal.

    Merge latency/SLA integrity is checked against the replay subset, but
    setup traffic is still real workspace demand. Keeping the full journal
    here lets rolling token/query windows union every phase. The combined
    input population is rejected before this function is called.
    """
    rows: list[dict] = []
    expected = _artifact_declarations(manifest, d)["requests.jsonl"]
    _scan_request_journal(
        d / "requests.jsonl", expected,
        lambda row, _line_number: rows.append(row))
    return rows


def _rate_limit_merge_context(
        dirs: list[Path], summaries: list[dict], manifests: list[dict],
) -> tuple[dict | None, dict | None, list[str]]:
    """Return a quota snapshot and endpoint binding safe to pool.

    A provider limit is workspace/model/deployment policy, not a shard-local
    measurement setting.  A merged report may compare the epoch-unioned
    traffic with that policy only when every sealed source carries the same
    valid snapshot in both its effective configuration and its summary, and
    every source captured the same endpoint metadata with a complete binding.
    Missing or conflicting evidence becomes an ordinary merge compatibility
    issue so ``--force`` can still emit an explicitly INVALID diagnostic, but
    the diagnostic receives no configured quota comparison.
    """
    snapshots: list[tuple[str, dict]] = []
    metadata: list[tuple[str, dict]] = []
    issues: list[str] = []
    quota_seen = False

    for d, summary, manifest in zip(dirs, summaries, manifests):
        title = _run_title(d, summary)
        effective = manifest.get("effective_config")
        effective_has = isinstance(effective, dict) \
            and "rate_limits" in effective \
            and effective.get("rate_limits") is not None
        effective_limits = effective.get("rate_limits") \
            if effective_has else None

        summary_block = summary.get("rate_limits")
        summary_has = isinstance(summary_block, dict) \
            and "configured" in summary_block \
            and summary_block.get("configured") is not None
        summary_limits = summary_block.get("configured") \
            if summary_has else None
        malformed_summary_block = summary_block is not None \
            and not isinstance(summary_block, dict)
        quota_seen = quota_seen or effective_has or summary_block is not None

        if malformed_summary_block:
            issues.append(f"invalid rate-limit evidence for {title}: "
                          "summary.rate_limits must be an object")
            continue
        if isinstance(summary_block, dict) and not summary_has:
            issues.append(f"invalid rate-limit evidence for {title}: "
                          "summary.rate_limits.configured is missing")
            continue
        if effective_has != summary_has:
            missing = ("summary configured snapshot" if effective_has else
                       "manifest effective-config snapshot")
            issues.append(
                f"incomplete rate-limit evidence for {title}: missing {missing}")
            continue
        if not effective_has:
            continue

        valid = True
        for label, value in (
                ("manifest effective-config", effective_limits),
                ("summary configured", summary_limits)):
            try:
                validate_rate_limits(
                    value, f"{title} {label} rate_limits")
            except ValueError as exc:
                issues.append(f"invalid rate-limit evidence for {title}: {exc}")
                valid = False
        if not valid:
            continue
        if _stable(effective_limits) != _stable(summary_limits):
            issues.append(
                f"rate-limit snapshot disagrees between the manifest and "
                f"summary for {title}")
            continue

        manifest_meta = manifest.get("endpoint_metadata")
        summary_meta = (summary.get("run") or {}).get("endpoint_metadata")
        if not isinstance(manifest_meta, dict) \
                or not isinstance(summary_meta, dict):
            issues.append(
                f"incomplete rate-limit endpoint binding for {title}: "
                "captured endpoint metadata is missing")
            continue
        if _stable(manifest_meta) != _stable(summary_meta):
            issues.append(
                f"rate-limit endpoint metadata disagrees between the manifest "
                f"and summary for {title}")
            continue
        provisioned_fields = {
            "workload_type", "workload_size", "provisioned_model_units",
            "min_provisioned_throughput", "max_provisioned_throughput",
        }
        entities = manifest_meta.get("served_entities")
        independently_bound = bool(
            manifest_meta.get("name") == effective_limits.get("model")
            and isinstance(entities, list) and entities
            and all(isinstance(entity, dict)
                    and entity.get("name") == effective_limits.get("model")
                    and not any(field in entity for field in provisioned_fields)
                    for entity in entities))
        if not independently_bound:
            issues.append(
                f"incomplete rate-limit endpoint binding for {title}: "
                "captured metadata does not independently bind the configured "
                "pay-per-token model/deployment")
            continue
        binding = summary_block.get("binding") or {}
        if not isinstance(binding, dict) \
                or binding.get("binding_complete") is not True:
            issues.append(
                f"incomplete rate-limit endpoint binding for {title}: the "
                "source run did not verify its configured model/deployment")
            continue
        snapshots.append((title, effective_limits))
        metadata.append((title, manifest_meta))

    if not quota_seen:
        return None, None, issues
    if len(snapshots) != len(dirs):
        issues.append(
            "rate-limit snapshot and endpoint binding are not complete for "
            "every merge source")

    snapshot_groups: dict[str, list[str]] = {}
    for title, value in snapshots:
        snapshot_groups.setdefault(_stable(value), []).append(title)
    if len(snapshot_groups) > 1:
        detail = "; ".join(
            f"{', '.join(titles)}={value}"
            for value, titles in snapshot_groups.items())
        issues.append("different rate-limit snapshots: " + detail)

    metadata_groups: dict[str, list[str]] = {}
    for title, value in metadata:
        metadata_groups.setdefault(_stable(value), []).append(title)
    if len(metadata_groups) > 1:
        detail = "; ".join(
            f"{', '.join(titles)}={value}"
            for value, titles in metadata_groups.items())
        issues.append("different rate-limit endpoint metadata: " + detail)

    if issues or len(snapshots) != len(dirs) or len(metadata) != len(dirs):
        return None, None, issues
    return snapshots[0][1], metadata[0][1], issues


def _parse_shard(manifest: dict, d: Path) -> tuple[int, int]:
    """Return a zero-based shard index and total from ``i/n`` metadata."""
    schedule = manifest.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError(f"manifest for {d} is missing schedule object")
    candidates = []
    for location, value in (
            ("manifest.shard", manifest.get("shard")),
            ("manifest.schedule.shard",
             schedule.get("shard"))):
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"invalid {location} for {d}: expected i/n")
        match = _SHARD_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"invalid {location} for {d}: expected i/n")
        shown, total = (int(x) for x in match.groups())
        if shown > total:
            raise ValueError(f"invalid {location} for {d}: {value!r}")
        candidates.append((location, (shown - 1, total)))
    if not candidates:
        raise ValueError(f"missing shard i/n metadata for merge input {d}")
    values = {value for _location, value in candidates}
    if len(values) != 1:
        detail = ", ".join(f"{location}={value[0] + 1}/{value[1]}"
                           for location, value in candidates)
        raise ValueError(f"inconsistent shard metadata for {d}: {detail}")
    return candidates[0][1]


def _logical_run_id(manifest: dict):
    current = manifest.get("logical_run_id")
    legacy = manifest.get("run_id")
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(
            "manifest logical_run_id and legacy run_id aliases disagree")
    return current if current is not None else legacy


def _declared_total_requests(manifest: dict, d: Path) -> int | None:
    schedule = manifest.get("schedule") or {}
    schedule_identity = manifest.get("schedule_identity") or {}
    index_identity = manifest.get("index_identity") or {}
    values = []
    for label, value in (
            ("manifest.total_requests", manifest.get("total_requests")),
            ("manifest.global_request_count",
             manifest.get("global_request_count")),
            ("manifest.schedule.total_requests",
             schedule.get("total_requests")),
            ("manifest.schedule.global_requests",
             schedule.get("global_requests")),
            ("manifest.schedule_identity.global_count",
             schedule_identity.get("global_count")),
            ("manifest.index_identity.global_count",
             index_identity.get("global_count"))):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {label} for {d}: {value!r}")
        values.append((label, value))
    distinct = {value for _label, value in values}
    if len(distinct) > 1:
        detail = ", ".join(f"{label}={value}" for label, value in values)
        raise ValueError(f"inconsistent total request metadata for {d}: {detail}")
    return values[0][1] if values else None


def _packed_sha256(values, encoding: str) -> str:
    pack = "<q" if encoding == "int64-le" else "<d"
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack(pack, value))
    return digest.hexdigest()


def _merge_integrity(dirs: list[Path], manifests: list[dict],
                     rows_by_dir: list[list[dict]]) -> list[str]:
    """Validate that inputs form one non-overlapping logical shard set.

    Corrupt identities and duplicate evidence are rejected unconditionally.
    Missing expected shards/indices are returned as compatibility issues so
    the existing ``--force`` path can retain an explicitly INVALID diagnostic
    artifact without ever labelling a partial aggregation valid.
    """
    parsed = [_parse_shard(manifest, d)
              for d, manifest in zip(dirs, manifests)]
    totals = {total for _index, total in parsed}
    if len(totals) != 1:
        detail = ", ".join(
            f"{d}={index + 1}/{total}"
            for d, (index, total) in zip(dirs, parsed))
        raise ValueError(f"inconsistent shard totals: {detail}")
    shard_total = next(iter(totals))
    indices = [index for index, _total in parsed]
    duplicate_indices = sorted(
        index for index in set(indices) if indices.count(index) > 1)
    if duplicate_indices:
        shown = ", ".join(str(index + 1) for index in duplicate_indices)
        raise ValueError(f"duplicate shard indices: {shown}/{shard_total}")

    if shard_total > 1:
        run_ids = []
        starts = []
        for d, manifest in zip(dirs, manifests):
            try:
                run_id = _logical_run_id(manifest)
            except ValueError as exc:
                raise ValueError(f"{exc} in {d}") from exc
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError(
                    f"multi-shard merge requires a non-empty logical_run_id "
                    f"for {d}")
            start = manifest.get("start_at_unix")
            if isinstance(start, bool) or not isinstance(start, (int, float)) \
                    or not math.isfinite(float(start)):
                raise ValueError(
                    f"multi-shard merge requires a finite shared "
                    f"start_at_unix for {d}")
            run_ids.append(run_id)
            starts.append(float(start))
        if len(set(run_ids)) != 1:
            raise ValueError(
                "multi-shard inputs have inconsistent logical_run_id values")
        if len(set(starts)) != 1:
            raise ValueError(
                "multi-shard inputs have inconsistent shared start_at_unix "
                "values")

    request_owner: dict[str, Path] = {}
    index_owner: dict[int, Path] = {}
    local_expected: dict[int, int] = {}
    declared_totals = []
    issues = []
    for d, manifest, rows, (shard_index, _total) in zip(
            dirs, manifests, rows_by_dir, parsed):
        schedule = manifest.get("schedule") or {}
        schedule_identity = manifest["schedule_identity"]
        index_identity = manifest["index_identity"]
        if index_identity["shard_index"] != shard_index \
                or index_identity["shard_total"] != shard_total:
            raise ValueError(
                f"index_identity shard index/total disagrees with shard i/n "
                f"metadata for {d}")
        scheduled = schedule.get("requests")
        if isinstance(scheduled, bool) or not isinstance(scheduled, int) \
                or scheduled < 0:
            issues.append(
                f"missing or invalid local schedule.requests for shard "
                f"{shard_index + 1}/{shard_total} ({d})")
        else:
            local_expected[shard_index] = scheduled
            if len(rows) != scheduled:
                issues.append(
                    f"shard {shard_index + 1}/{shard_total} has {len(rows)} "
                    f"replay rows but schedule.requests declares {scheduled}")
        declared_total = _declared_total_requests(manifest, d)
        if declared_total is not None:
            declared_totals.append((d, declared_total))

        for row_number, row in enumerate(rows, 1):
            request_id = row.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError(
                    f"replay row {row_number} in {d} has no valid request_id")
            if request_id in request_owner:
                raise ValueError(
                    f"duplicate replay request_id {request_id!r} in {d} and "
                    f"{request_owner[request_id]}")
            request_owner[request_id] = d

            global_index = row.get("global_index")
            if isinstance(global_index, bool) \
                    or not isinstance(global_index, int) or global_index < 0:
                raise ValueError(
                    f"replay row {row_number} in {d} has no valid "
                    "non-negative global_index")
            if global_index in index_owner:
                raise ValueError(
                    f"overlapping replay global_index {global_index} in {d} "
                    f"and {index_owner[global_index]}")
            if global_index % shard_total != shard_index:
                raise ValueError(
                    f"global_index {global_index} in {d} belongs to shard "
                    f"{global_index % shard_total + 1}/{shard_total}, not "
                    f"declared shard {shard_index + 1}/{shard_total}")
            index_owner[global_index] = d

        ordered = sorted(rows, key=lambda row: row["global_index"])
        ordered_indices = [row["global_index"] for row in ordered]
        actual_index_hash = _packed_sha256(ordered_indices, "int64-le")
        if not hmac.compare_digest(
                actual_index_hash,
                index_identity["global_indices_sha256"].lower()):
            raise ValueError(
                f"index_identity SHA-256 disagrees with replay global_index "
                f"values for {d}")
        actual_min = ordered_indices[0] if ordered_indices else None
        actual_max = ordered_indices[-1] if ordered_indices else None
        if (index_identity["count"] != len(ordered_indices)
                or index_identity.get("min") != actual_min
                or index_identity.get("max") != actual_max):
            raise ValueError(
                f"index_identity count/min/max disagrees with replay rows for {d}")

        ordered_timestamps = []
        for row_number, row in enumerate(ordered, 1):
            value = row.get("scheduled_s")
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise ValueError(
                    f"replay row {row_number} in {d} has no valid scheduled_s")
            ordered_timestamps.append(float(value))
        actual_shard_schedule_hash = _packed_sha256(
            ordered_timestamps, "float64-le")
        if not hmac.compare_digest(
                actual_shard_schedule_hash,
                schedule_identity["shard_timestamps_sha256"].lower()):
            raise ValueError(
                f"schedule_identity shard SHA-256 disagrees with replay "
                f"scheduled_s values for {d}")
        actual_schedule_min = min(ordered_timestamps) \
            if ordered_timestamps else None
        actual_schedule_max = max(ordered_timestamps) \
            if ordered_timestamps else None
        if (schedule_identity["shard_count"] != len(ordered_timestamps)
                or schedule_identity.get("shard_min_s") != actual_schedule_min
                or schedule_identity.get("shard_max_s") != actual_schedule_max):
            raise ValueError(
                f"schedule_identity shard count/min/max disagrees with replay "
                f"rows for {d}")

    if declared_totals:
        total_values = {value for _d, value in declared_totals}
        if len(total_values) != 1:
            detail = ", ".join(f"{d}={value}"
                               for d, value in declared_totals)
            raise ValueError(f"inconsistent declared total requests: {detail}")
    if len(declared_totals) != len(manifests):
        declared_dirs = {d for d, _value in declared_totals}
        missing = [str(d) for d in dirs if d not in declared_dirs]
        issues.append(
            "missing declared global total request coverage for "
            + ", ".join(missing))

    expected_shards = set(range(shard_total))
    actual_shards = set(indices)
    missing_shards = sorted(expected_shards - actual_shards)
    if missing_shards:
        issues.append(
            "missing expected shard indices: "
            + ", ".join(f"{index + 1}/{shard_total}"
                        for index in missing_shards))

    expected_total = None
    if declared_totals:
        expected_total = declared_totals[0][1]
    elif actual_shards == expected_shards \
            and set(local_expected) == expected_shards:
        expected_total = sum(local_expected.values())

    if expected_total is None:
        issues.append(
            "expected global request/index coverage cannot be proven from "
            "the shard manifests")
    elif actual_shards == expected_shards:
        missing_text, extra_text = _coverage_gaps(
            sorted(index_owner), expected_total)
        if missing_text or extra_text:
            detail = []
            if missing_text:
                detail.append("missing " + missing_text)
            if extra_text:
                detail.append("unexpected " + extra_text)
            issues.append(
                "global_index coverage is incomplete or out of range: "
                + "; ".join(detail))
        q, r = divmod(expected_total, shard_total)
        for shard_index in sorted(actual_shards):
            expected_local = q + (1 if shard_index < r else 0)
            declared_local = local_expected.get(shard_index)
            if declared_local is not None and declared_local != expected_local:
                issues.append(
                    f"shard {shard_index + 1}/{shard_total} declares "
                    f"{declared_local} requests; global coverage requires "
                    f"{expected_local}")
        if not missing_text and not extra_text:
            ordered_global_rows = sorted(
                (row for rows in rows_by_dir for row in rows),
                key=lambda row: row["global_index"])
            global_timestamps = [float(row["scheduled_s"])
                                 for row in ordered_global_rows]
            global_hash = _packed_sha256(global_timestamps, "float64-le")
            global_min = min(global_timestamps) if global_timestamps else None
            global_max = max(global_timestamps) if global_timestamps else None
            for d, manifest in zip(dirs, manifests):
                identity = manifest["schedule_identity"]
                if (not hmac.compare_digest(
                        identity["global_timestamps_sha256"].lower(),
                        global_hash)
                        or identity["global_count"] != len(global_timestamps)
                        or identity.get("global_min_s") != global_min
                        or identity.get("global_max_s") != global_max):
                    raise ValueError(
                        f"schedule_identity global schedule disagrees with "
                        f"the complete replay coverage for {d}")
    return issues


def _coverage_gaps(actual: list[int], expected_total: int,
                   preview_limit: int = 12) -> tuple[str | None, str | None]:
    """Describe missing/extra indices without materializing ``range(total)``."""
    missing_preview = []
    missing_count = 0
    extra_preview = []
    extra_count = 0
    next_expected = 0
    for value in actual:
        if value >= expected_total:
            extra_count += 1
            if len(extra_preview) < preview_limit:
                extra_preview.append(value)
            continue
        if value > next_expected:
            gap = value - next_expected
            missing_count += gap
            room = preview_limit - len(missing_preview)
            if room > 0:
                missing_preview.extend(range(
                    next_expected, min(value, next_expected + room)))
        next_expected = value + 1
    if next_expected < expected_total:
        gap = expected_total - next_expected
        missing_count += gap
        room = preview_limit - len(missing_preview)
        if room > 0:
            missing_preview.extend(range(
                next_expected, min(expected_total, next_expected + room)))

    def describe(preview, count):
        if not count:
            return None
        shown = ",".join(str(value) for value in preview)
        return shown + (f",... ({count} total)" if count > len(preview) else "")

    return describe(missing_preview, missing_count), describe(
        extra_preview, extra_count)


def merge_runs(out_dir, input_dirs, title=None, acceptance=None,
               force=False) -> Path:
    """Pool concurrent shard evidence and re-summarize the epoch union.

    Replay rows alone feed latency and SLA metrics. Every sealed request row
    feeds rolling quota windows so setup traffic cannot disappear at merge.
    """
    dirs, manifests = _validated_input_dirs(
        input_dirs, "requests.jsonl", "merge")
    combined_request_rows = sum(
        _artifact_declarations(manifest, d)["requests.jsonl"]["row_count"]
        for d, manifest in zip(dirs, manifests))
    if combined_request_rows > MAX_EXACT_ANALYSIS_REQUEST_ROWS:
        raise ValueError(
            f"merge inputs declare {combined_request_rows:,} total request "
            f"rows, above the exact-analysis resource envelope of "
            f"{MAX_EXACT_ANALYSIS_REQUEST_ROWS:,}; merge smaller compatible "
            "sets or implement bounded streaming aggregation")
    summaries = [_load_summary(d) for d in dirs]
    request_rows_by_dir = [
        _request_rows(d, manifest)
        for d, manifest in zip(dirs, manifests)
    ]
    rows_by_dir = [
        [row for row in source_rows if row.get("phase") == "replay"]
        for source_rows in request_rows_by_dir
    ]
    coverage_issues = _merge_integrity(dirs, manifests, rows_by_dir)
    compatibility_issues = _compatibility_issues(
        dirs, summaries, manifests, merging=True)
    compatibility_issues.extend(coverage_issues)
    rate_limits, quota_endpoint_meta, quota_issues = \
        _rate_limit_merge_context(dirs, summaries, manifests)
    compatibility_issues.extend(quota_issues)
    quota_rows = [
        row for source_rows in request_rows_by_dir for row in source_rows
    ]
    observed_quota_phases = {
        phase: sum(str(row.get("phase") or "unlabeled") == phase
                   for row in quota_rows)
        for phase in sorted({
            str(row.get("phase") or "unlabeled") for row in quota_rows
        })
    }
    if rate_limits is not None:
        unsupported_phases = sorted({
            str(row.get("phase")) for row in quota_rows
            if row.get("phase") not in _QUOTA_REQUEST_PHASES
        })
        if unsupported_phases:
            compatibility_issues.append(
                "configured rate-limit evidence contains unsupported request "
                "phases: " + ", ".join(unsupported_phases))
    if compatibility_issues and not force:
        raise ValueError(
            "refusing to merge inputs that are not proven compatible: "
            + "; ".join(compatibility_issues)
            + ". pass force=True only to create an explicitly INVALID "
              "diagnostic aggregate.")
    endpoints, rows = set(), []
    for d, source_summary, source_rows in zip(dirs, summaries, rows_by_dir):
        run = source_summary.get("run") or {}
        # identity is host plus model plus route. comparing the route alone
        # pooled two different providers whenever both served
        # /v1/chat/completions, which is most of them.
        ident = (run.get("endpoint_base_url"), run.get("endpoint_model"),
                 run.get("endpoint_path"))
        if any(x is not None for x in ident):
            endpoints.add(ident)
        rows += source_rows
    rows.sort(key=lambda row: row["global_index"])
    # prompts-mode shards each cycled the same prompt file, so the pooled
    # cache fraction is still replay behavior. carry the fields summarize()
    # needs, otherwise the merged report shows the cache number with no note.
    counts = {(s.get("run") or {}).get("prompts_count") for s in summaries}
    declared_source_policies = [
        _declared_acceptance_policy(_run_title(d, summary), summary, manifest)[0]
        for d, summary, manifest in zip(dirs, summaries, manifests)
    ]
    policy_values = {
        _stable(policy) for policy in declared_source_policies
        if policy is not None
    }
    shared_source_policy = (
        next(policy for policy in declared_source_policies if policy is not None)
        if len(policy_values) == 1
        and all(policy is not None for policy in declared_source_policies)
        else None
    )
    supplied_policy = _normalized_acceptance_policy(acceptance)
    if acceptance is None:
        effective_acceptance = shared_source_policy
        acceptance_mode = (
            "source_policy_propagated" if shared_source_policy is not None
            else "not_configured")
    else:
        effective_acceptance = acceptance
        acceptance_mode = (
            "explicit_policy_matches_sources"
            if shared_source_policy is not None
            and supplied_policy == shared_source_policy
            else "post_hoc_override")
    acceptance_provenance = {
        "mode": acceptance_mode,
        "source_policy_coverage": sum(
            policy is not None for policy in declared_source_policies),
        "source_count": len(declared_source_policies),
        "source_policy": shared_source_policy,
        "applied_policy": _normalized_acceptance_policy(effective_acceptance),
        "post_hoc": acceptance_mode == "post_hoc_override",
        "note": (
            "acceptance thresholds were supplied at merge time and differ "
            "from, or were absent from, source-run policy; this is transparent "
            "post-hoc rescoring of sealed rows"
            if acceptance_mode == "post_hoc_override" else
            "the common source-run acceptance policy was retained"
            if acceptance_mode == "source_policy_propagated" else
            "the explicit merge policy matches every source-run policy"
            if acceptance_mode == "explicit_policy_matches_sources" else
            "no acceptance policy was configured on the source runs or merge"),
    }
    source_provenance = []
    for d, manifest, policy in zip(
            dirs, manifests, declared_source_policies):
        source_provenance.append({
            "run_dir": str(d),
            "logical_run_id": (_logical_run_id(manifest)
                               if manifest else None),
            "execution_id": (manifest or {}).get("execution_id"),
            "artifact_id": (manifest or {}).get("artifact_id"),
            "workload_id": (manifest or {}).get("workload_id"),
            "shard": (manifest or {}).get("shard"),
            "start_at_unix": (manifest or {}).get("start_at_unix"),
            "git_commit": (manifest or {}).get("git_commit"),
            "profile_sha256": ((manifest or {}).get("profile_sha256")
                               or (manifest or {}).get("profile_sha256_16")),
            "config_sha256": (manifest or {}).get("config_sha256"),
            "acceptance_policy": policy,
        })
    workload_ids = {manifest.get("workload_id") for manifest in manifests}
    workload_id = (next(iter(workload_ids)) if len(workload_ids) == 1 else
                   "invalid-mixed-" + hashlib.sha256(
                       _stable(sorted(str(value) for value in workload_ids)).encode()
                   ).hexdigest()[:16])
    logical_run_id = _logical_run_id(manifests[0])
    shared_start_at = manifests[0].get("start_at_unix")
    input_modes = {manifest.get("input_mode") for manifest in manifests}
    input_mode = next(iter(input_modes)) if len(input_modes) == 1 else None
    request_params_values = {
        _stable(manifest.get("request_params")) for manifest in manifests
    }
    request_params = (manifests[0].get("request_params")
                      if len(request_params_values) == 1 else None)
    seed_values = {manifest.get("seed") for manifest in manifests}
    seed = next(iter(seed_values)) if len(seed_values) == 1 else None
    load_modes = {manifest.get("load_mode") for manifest in manifests}
    load_mode = next(iter(load_modes)) if len(load_modes) == 1 else None
    transport_values = {
        _stable((summary.get("run") or {}).get("transport"))
        for summary in summaries
    }
    transport = ((summaries[0].get("run") or {}).get("transport")
                 if len(transport_values) == 1 else None)
    expected_total = _declared_total_requests(manifests[0], dirs[0])
    merged_indices = [row["global_index"] for row in rows]
    observed_total = len(merged_indices)
    observed_indices_dense = all(
        value == position for position, value in enumerate(merged_indices))
    index_identity = {
        "encoding": "int64-le",
        "global_indices_sha256": _packed_sha256(merged_indices, "int64-le"),
        "count": len(merged_indices),
        "min": merged_indices[0] if merged_indices else None,
        "max": merged_indices[-1] if merged_indices else None,
        # A forced incomplete merge remains a self-consistent sealed artifact.
        # The parent expectation is retained separately as diagnostic lineage.
        "global_count": observed_total,
        "shard_index": 0,
        "shard_total": 1,
        "partition": (
            "diagnostic_observed_subset"
            if not observed_indices_dense else "unsharded"),
    }
    source_schedule_identity = manifests[0].get("schedule_identity") or {}
    merged_timestamps = [float(row["scheduled_s"]) for row in rows]
    observed_schedule_hash = _packed_sha256(
        merged_timestamps, "float64-le")
    merged_schedule_identity = {
        "encoding": (source_schedule_identity.get("encoding")
                     or "float64-le-seconds-from-run-start"),
        "global_timestamps_sha256": observed_schedule_hash,
        "global_count": observed_total,
        "global_min_s": min(merged_timestamps) if merged_timestamps else None,
        "global_max_s": max(merged_timestamps) if merged_timestamps else None,
        "shard_timestamps_sha256": observed_schedule_hash,
        "shard_count": observed_total,
        "shard_min_s": min(merged_timestamps) if merged_timestamps else None,
        "shard_max_s": max(merged_timestamps) if merged_timestamps else None,
    }

    definitions = {
        definition
        for d, summary, manifest in zip(dirs, summaries, manifests)
        for definition in [_declared_ttft_definition(
            _run_title(d, summary), summary, manifest)[0]]
        if definition is not None
    }
    # A non-forced merge reaches here only with one canonical declaration.
    # Forced diagnostics use first_content solely to render their withheld
    # metrics; compatibility_issues remains the authoritative invalid state.
    ttft_definition = (next(iter(definitions)) if len(definitions) == 1
                       else "first_content")

    source_schedules = [manifest.get("schedule") or {} for manifest in manifests]

    def common_schedule_value(field: str):
        values = [schedule.get(field) for schedule in source_schedules]
        stable = {_stable(value) for value in values if value is not None}
        return values[0] if len(stable) == 1 and all(
            value is not None for value in values) else None

    merged_schedule_meta = {
        "requests": observed_total,
        "total_requests": observed_total,
        "source_expected_total_requests": expected_total,
        "observed_replay_requests": observed_total,
        "coverage_complete": expected_total == observed_total,
        "shard": "1/1",
        "source": common_schedule_value("source") or "merged proven schedule",
    }
    for schedule_field in (
            "seconds", "rate_min", "rate_p50", "rate_p95", "rate_max",
            "arrival_mode", "trace_sha256"):
        value = common_schedule_value(schedule_field)
        if value is not None:
            merged_schedule_meta[schedule_field] = value
    meta = {
        "merged_from": [str(d) for d in dirs],
        **({"endpoint_base_url": next(iter(endpoints))[0],
            "endpoint_model": next(iter(endpoints))[1]}
           if len(endpoints) == 1 else
           {"endpoint_base_url": "MIXED", "endpoint_model": "MIXED"}),
        "endpoint_path": (next(iter(endpoints))[2] if len(endpoints) == 1
                          else "MIXED"),
        "label": f"merged from {len(dirs)} runs",
        "logical_run_id": logical_run_id,
        "run_id": logical_run_id,
        "workload_id": workload_id,
        "start_at_unix": shared_start_at,
        "shard": "1/1",
        "input_mode": input_mode,
        "request_params": request_params,
        "seed": seed,
        "load_mode": load_mode,
        **({"transport": transport} if transport is not None else {}),
        "ttft_definition": ttft_definition,
        "index_identity": index_identity,
        "schedule_identity": merged_schedule_identity,
        "aggregation_valid": not compatibility_issues,
        "compatibility_issues": compatibility_issues,
        "aggregation": {
            "kind": "merge",
            "forced": bool(force),
            "sources": source_provenance,
            "acceptance_policy_provenance": acceptance_provenance,
            "source_schedule_expectation": {
                "expected_total_requests": expected_total,
                "observed_replay_requests": observed_total,
                "complete": expected_total == observed_total,
                "global_schedule_identities": [
                    _global_schedule_identity(manifest)
                    for manifest in manifests
                ],
            },
        },
        **({"prompts_count": counts.pop()}
           if input_mode == "prompts" and len(counts) == 1
           and None not in counts else {}),
        "merge_note": (f"pooled from {len(dirs)} run dirs. throughput is over "
                       "the union wall-clock window, so it is the aggregate "
                       "rate only when the shards ran concurrently."),
        **({"endpoint_metadata": quota_endpoint_meta}
           if rate_limits is not None and not compatibility_issues else {}),
        "quota_merge": {
            "traffic_population": "all_sealed_request_phases",
            "sla_population": "replay_only",
            "eligible_phase_kinds": sorted(_QUOTA_REQUEST_PHASES),
            "observed_phase_rows": observed_quota_phases,
            "sealed_rows": len(quota_rows),
            "configured_snapshot_status": (
                "compatible" if rate_limits is not None
                and not compatibility_issues else
                "withheld_invalid_inputs" if compatibility_issues else
                "not_configured"),
            "note": (
                "rolling quota windows pool manifest-bound shard rows by epoch; "
                "latency and acceptance-target metrics remain replay-only"),
        },
    }
    # cost is a per-run figure (rates can differ across pooled runs), so
    # it is not recomputed here; read each run report for its own cost.
    # Legacy rows reconstruct caller delay from one epoch offset shared by the
    # input list. That is invalid when runs began at different wall-clock
    # times. Mark queue wait unavailable before summarize() so a merge pools
    # only exact monotonic caller clocks already recorded on each row.
    temporary_caller_send_markers: set[int] = set()
    for row in rows:
        row["queue_wait_ms"] = None
        if "caller_send_ms" not in row:
            # summarize() reconstructs only legacy rows where this field is
            # absent. A pooled merge has no valid cross-run epoch offset, so
            # use an explicit temporary None to suppress reconstruction.
            row["caller_send_ms"] = None
            temporary_caller_send_markers.add(id(row))
    summary = summarize(
        rows, run_meta=meta, acceptance=effective_acceptance,
        schedule_meta=merged_schedule_meta,
        ttft_definition=ttft_definition,
        rate_limits=(rate_limits if not compatibility_issues else None),
        rate_limit_results=quota_rows)
    if compatibility_issues:
        # ``--force`` is diagnostic only.  In particular, do not display a
        # workspace rolling-rate union when the inputs may cover different
        # policies, endpoints, workloads, or an incomplete shard set.
        summary.pop("rate_limits", None)
        summary["observed_rate_windows"] = {
            "withheld": True,
            "note": (
                "workspace rolling-rate evidence is withheld because the "
                "forced merge inputs were not proven compatible"),
        }
    # drift buckets on absolute send time from the pooled minimum. shards that
    # ran at different times produce windows spanning the gap between them, so
    # a trend across pooled rows would describe the schedule, not the endpoint.
    # same hazard as drift below: shards start at different wall-clock times,
    # so a single schedule-vs-send offset across pooled rows reads the gap
    # between shards as lateness.
    summary["arrivals"]["http_request_start_lateness_ms"] = _pct_table([])
    summary["arrivals"]["wire_lateness_ms"] = _pct_table([])
    summary["arrivals"]["wire_lateness_note"] = (
        "HTTP request-start lateness is not computed for a merged run, "
        "because pooled rows "
        "come from separate runs and the offset between them would read as "
        "lateness. read each run's own report. dispatch lag below is pooled "
        "and still meaningful, since it is measured within each run.")
    summary.pop("client", None)
    # A merged row must not imply that one cross-run epoch offset established
    # its queue delay. Exact caller clocks remain on their own fields.
    for _r in rows:
        _r.pop("queue_wait_ms", None)
        if id(_r) in temporary_caller_send_markers:
            _r.pop("caller_send_ms", None)
    corrected_fields = (
        "ttft_corrected_ms", "ttfv_corrected_ms",
        "ttf_tool_call_corrected_ms", "e2e_corrected_ms")
    if any((summary.get(key) or {}).get("n") for key in corrected_fields):
        summary["latency_correction_note"] = (
            "merged caller-experienced latency pools only exact monotonic "
            "durations recorded by each source row. Legacy schedule/send "
            "timestamps are not reconstructed across runs because their "
            "wall-clock offsets are not comparable.")
    else:
        summary.pop("latency_correction_provenance", None)
        summary["latency_correction_note"] = (
            "caller-experienced latency is unavailable for this merged run: "
            "the source rows did not carry exact monotonic caller clocks, and "
            "legacy schedule/send timestamps cannot be reconstructed across "
            "different run epochs. Service-time latency remains available.")
    # concurrency is interval overlap across pooled rows. shards that never
    # ran at the same time have no overlap, so a merged run would report a
    # p50 of 0 in flight. same reason wire lateness and drift are blanked.
    if summary.pop("concurrency", None) is not None:
        summary["concurrency_note"] = (
            "concurrency in flight is not computed for a merged run, because "
            "it is measured by interval overlap and shards that ran at "
            "different times do not overlap. read each run's own report.")
    summary["drift"] = {
        "windows": [], "window_seconds": 60,
        "note": "stability over time is not computed for a merged run. the "
                "pooled rows come from separate runs, so time windows would "
                "span the gaps between them. that also means a merged run "
                "cannot report a breaking point, so if any shard was shedding "
                "requests, read its own report. the pooled error rate below "
                "still counts every failure.",
    }
    created_at = time.time()
    input_hash = (manifests[0].get("profile_sha256")
                  or manifests[0].get("profile_sha256_16"))
    input_key = "prompts" if input_mode == "prompts" else "profile"
    start_provenance = {
        "start_schema_version": 1,
        "status": "aggregation",
        "run_started_at_unix": created_at,
        "run_started_at_utc": datetime.fromtimestamp(
            created_at, timezone.utc).isoformat(),
        "logical_run_id": logical_run_id,
        "workload_id": workload_id,
        "execution_id": f"execution-{uuid.uuid4().hex}",
        "effective_config": {
            "operation": "merge",
            "forced": bool(force),
            "sources": source_provenance,
            "ttft_definition": ttft_definition,
            "schedule": merged_schedule_meta,
            "acceptance_targets": effective_acceptance,
            "acceptance_policy_provenance": acceptance_provenance,
            **({"rate_limits": rate_limits}
               if rate_limits is not None and not compatibility_issues else {}),
        },
        "inputs": ({input_key: {"sha256": input_hash}}
                   if input_hash else {}),
        "schedule_identity": merged_schedule_identity,
        "index_identity": index_identity,
    }
    return write_outputs(
        quota_rows, summary, out_dir, title or f"merged: {len(dirs)} runs",
        start_provenance=start_provenance)


def _cell(v, fmt="{:.0f}") -> str:
    return fmt.format(v) if v is not None else "-"


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        # macOS exposes /tmp as a symlink to /private/tmp. Resolve a directory
        # alias only for this read-only fsync, then keep O_NOFOLLOW on the
        # resolved final component and prove it is the same directory inode.
        if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
            raise
        expected = path.stat()
        resolved = path.resolve(strict=True)
        fd = os.open(resolved, flags)
        actual = os.fstat(fd)
        if not stat.S_ISDIR(expected.st_mode) \
                or (actual.st_dev, actual.st_ino) != (
                    expected.st_dev, expected.st_ino):
            os.close(fd)
            raise OSError(errno.ESTALE, "directory alias changed during fsync",
                          str(path))
    try:
        _fsync_fd(fd)
    finally:
        os.close(fd)


def _fsync_fd(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems/platforms do not support directory fsync. Real I/O
        # failures must still fail the write rather than claim durability.
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def _write_compare_fd(fd: int, raw: bytes, name: str) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
            raise OSError(f"short write while creating {name}")
        offset += written


def _claim_compare_dir(requested: Path, artifact_id: str,
                       created_at: float) -> tuple[Path, int]:
    """Exclusively claim a fresh directory and return an open directory fd."""
    requested.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10_000):
        candidate = (requested if attempt == 0 else requested.with_name(
            f"{requested.name}-{uuid.uuid4().hex[:12]}"))
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            # Never enter or reuse an existing path, including an empty dir or
            # a symlink. That makes both repeated and adversarial claims safe.
            continue
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        try:
            dir_fd = os.open(candidate, flags)
            marker_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
                | getattr(os, "O_NOFOLLOW", 0)
            marker_fd = os.open(
                _WRITING_MARKER, marker_flags, 0o644, dir_fd=dir_fd)
            try:
                marker = strict_json_dumps({
                    "artifact_id": artifact_id,
                    "artifact_type": "comparison",
                    "status": "writing",
                    "created_at_unix": created_at,
                }).encode("utf-8") + b"\n"
                _write_compare_fd(marker_fd, marker, _WRITING_MARKER)
                os.fsync(marker_fd)
            finally:
                os.close(marker_fd)
            _fsync_fd(dir_fd)
            _fsync_directory(candidate.parent)
            return candidate, dir_fd
        except Exception:
            if "dir_fd" in locals():
                os.close(dir_fd)
            raise
    raise RuntimeError(f"could not claim a unique comparison directory: {requested}")


def _atomic_compare_text(dir_fd: int, name: str, value: str) -> dict:
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o644, dir_fd=dir_fd)
    raw = value.encode("utf-8")
    try:
        _write_compare_fd(fd, raw, name)
        os.fsync(fd)
    except Exception:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    _fsync_fd(dir_fd)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _verified_comparison_summary(d: Path, manifest: dict) -> dict:
    """Read exactly the summary bytes bound by the input manifest."""
    expected = _artifact_declarations(manifest, d)["summary.json"]
    raw = _read_regular_bytes(d / "summary.json")
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected["sha256"]):
        raise ValueError(
            f"artifact SHA-256 mismatch for {d / 'summary.json'}: expected "
            f"{expected['sha256']}, got {actual}")
    if len(raw) != expected["bytes"]:
        raise ValueError(
            f"artifact byte count mismatch for {d / 'summary.json'}: expected "
            f"{expected['bytes']}, got {len(raw)}")
    try:
        value = loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"invalid summary.json in {d}: {json_error_detail(exc)}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"summary.json must contain a JSON object: {d}")
    return value


def _verified_comparison_request_evidence(d: Path, manifest: dict) -> dict:
    """Scan the exact manifest-bound journal bytes for HTTP 429 evidence.

    Comparison must include setup traffic, not only replay rows. Reading and
    hashing through one descriptor also closes the verify-then-read race: the
    429 verdict is derived from the same bytes bound by the source manifest.
    """
    expected = _artifact_declarations(manifest, d)["requests.jsonl"]
    path = d / "requests.jsonl"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    total = 0
    count = 0
    phases: dict[str, int] = {}
    phase_totals: dict[str, int] = {}
    http_status_observed_for = 0
    replay_records: list[tuple[int, float, str]] = []
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        if before.st_size != expected["bytes"]:
            raise ValueError(
                f"artifact byte count mismatch for {path}: expected "
                f"{expected['bytes']}, got {before.st_size}")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            line_no = 0
            while True:
                raw = handle.readline(_MAX_REQUEST_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                line_no += 1
                if line_no > expected["row_count"]:
                    raise ValueError(
                        f"request journal exceeds its declared row count in "
                        f"{path}")
                if len(raw) > _MAX_REQUEST_JSONL_LINE_BYTES:
                    raise ValueError(
                        f"request journal line {line_no} exceeds the "
                        f"{_MAX_REQUEST_JSONL_LINE_BYTES:,}-byte limit in "
                        f"{path}")
                if not raw.endswith(b"\n"):
                    raise ValueError(
                        f"request journal line {line_no} is not newline "
                        f"terminated in {path}")
                digest.update(raw)
                size += len(raw)
                if size > expected["bytes"]:
                    raise ValueError(
                        f"request journal exceeds its declared byte count in "
                        f"{path}")
                if not raw.strip():
                    raise ValueError(
                        f"blank JSONL record in {path} line {line_no}")
                try:
                    row = loads_strict(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSON in {path} line {line_no}: "
                        f"{json_error_detail(exc)}") from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"requests.jsonl line {line_no} is not an object in {d}")
                total += 1
                phase = str(row.get("phase") or "unlabeled")
                phase_totals[phase] = phase_totals.get(phase, 0) + 1
                if phase == "replay":
                    global_index = row.get("global_index")
                    if isinstance(global_index, bool) \
                            or not isinstance(global_index, int) \
                            or global_index < 0:
                        raise ValueError(
                            f"replay row {line_no} in {path} has no valid "
                            "non-negative global_index")
                    scheduled_s = row.get("scheduled_s")
                    if isinstance(scheduled_s, bool) \
                            or not isinstance(scheduled_s, (int, float)) \
                            or not math.isfinite(float(scheduled_s)):
                        raise ValueError(
                            f"replay row {line_no} in {path} has no valid "
                            "scheduled_s")
                    request_id = row.get("request_id")
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError(
                            f"replay row {line_no} in {path} has no valid "
                            "request_id")
                    replay_records.append((
                        global_index, float(scheduled_s), request_id))
                status = row.get("status")
                if isinstance(status, int) and not isinstance(status, bool):
                    http_status_observed_for += 1
                    if status == 429:
                        count += 1
                        phases[phase] = phases.get(phase, 0) + 1
            after = os.fstat(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    if _regular_identity(before) != _regular_identity(after):
        raise ValueError(f"request journal changed while reading: {path}")
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected["sha256"]):
        raise ValueError(
            f"artifact SHA-256 mismatch for {path}: expected "
            f"{expected['sha256']}, got {actual}")
    if size != expected["bytes"]:
        raise ValueError(
            f"artifact byte count mismatch for {path}: expected "
            f"{expected['bytes']}, got {size}")
    if total != expected["row_count"]:
        raise ValueError(
            f"artifact row count mismatch for {path}: expected "
            f"{expected['row_count']}, got {total}")

    ordered = sorted(replay_records, key=lambda item: item[0])
    indices = [item[0] for item in ordered]
    timestamps = [item[1] for item in ordered]
    request_ids = [item[2] for item in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"duplicate replay global_index values in manifest-bound {path}")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(
            f"duplicate replay request_id values in manifest-bound {path}")

    index_identity = manifest["index_identity"]
    schedule_identity = manifest["schedule_identity"]
    actual_index_hash = _packed_sha256(indices, "int64-le")
    actual_schedule_hash = _packed_sha256(timestamps, "float64-le")
    actual_index_min = indices[0] if indices else None
    actual_index_max = indices[-1] if indices else None
    actual_schedule_min = min(timestamps) if timestamps else None
    actual_schedule_max = max(timestamps) if timestamps else None
    if (not hmac.compare_digest(
            actual_index_hash,
            str(index_identity["global_indices_sha256"]).lower())
            or index_identity["count"] != len(indices)
            or index_identity.get("min") != actual_index_min
            or index_identity.get("max") != actual_index_max):
        raise ValueError(
            f"index_identity disagrees with replay global_index values in "
            f"manifest-bound {path}")
    if (not hmac.compare_digest(
            actual_schedule_hash,
            str(schedule_identity["shard_timestamps_sha256"]).lower())
            or schedule_identity["shard_count"] != len(timestamps)
            or schedule_identity.get("shard_min_s") != actual_schedule_min
            or schedule_identity.get("shard_max_s") != actual_schedule_max):
        raise ValueError(
            f"schedule_identity disagrees with replay scheduled_s values in "
            f"manifest-bound {path}")
    shard_index = index_identity["shard_index"]
    shard_total = index_identity["shard_total"]
    misplaced = [index for index in indices
                 if index % shard_total != shard_index]
    if misplaced:
        raise ValueError(
            f"replay global_index {misplaced[0]} in {path} does not belong "
            f"to declared shard {shard_index + 1}/{shard_total}")
    if shard_total == 1 and (
            not hmac.compare_digest(
                actual_schedule_hash,
                str(schedule_identity["global_timestamps_sha256"]).lower())
            or schedule_identity["global_count"] != len(timestamps)
            or schedule_identity.get("global_min_s") != actual_schedule_min
            or schedule_identity.get("global_max_s") != actual_schedule_max
            or index_identity["global_count"] != len(indices)):
        raise ValueError(
            f"global schedule/index identity disagrees with complete unsharded "
            f"replay evidence in manifest-bound {path}")
    return {
        "count": count,
        "total": total,
        "phases": phases,
        "phase_totals": phase_totals,
        "http_status_observed_for": http_status_observed_for,
    }


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _comparison_http_429_issues(
        title: str, summary: dict, journal: dict) -> list[str]:
    """Fail closed on direct or summarized, manifest-bound 429 evidence."""
    issues: list[str] = []
    block = summary.get("http_429")
    block = block if isinstance(block, dict) else {}
    failure_counts = summary.get("failures_by_http_status")
    failure_counts = failure_counts if isinstance(failure_counts, dict) else {}
    reported = {
        "summary.http_429_count": _nonnegative_int(
            summary.get("http_429_count")),
        "summary.http_429.count": _nonnegative_int(block.get("count")),
        "summary.failures_by_http_status[429]": _nonnegative_int(
            failure_counts.get("429")),
    }
    positive = {name: value for name, value in reported.items()
                if value is not None and value > 0}
    journal_count = journal["count"]

    if journal_count > 0:
        phases = ", ".join(
            f"{name or 'unlabeled'}={value}"
            for name, value in sorted(journal["phases"].items()))
        issues.append(
            f"{title}: quota-limited; {journal_count}/{journal['total']} "
            "manifest-bound request rows returned HTTP 429; phases: "
            f"{phases}. This source supports no endpoint-capacity conclusion")
        disagreements = [
            f"{name}={value}" for name, value in reported.items()
            if value is not None and value != journal_count
        ]
        if summary.get("quota_limited") is False:
            disagreements.append("summary.quota_limited=false")
        if disagreements:
            issues.append(
                f"{title}: manifest-bound 429 summary evidence disagrees with "
                "the sealed request journal (" + ", ".join(disagreements) + ")")
        return issues

    if positive:
        counts = sorted(set(positive.values()))
        count = counts[-1]
        denominator = _nonnegative_int(block.get("request_rows_examined"))
        if denominator is None or denominator < count:
            denominator = _nonnegative_int(summary.get("requests_total"))
        shown_denominator = str(denominator) if denominator is not None \
            and denominator >= count else "?"
        reported_phases = block.get("phases")
        if isinstance(reported_phases, dict):
            phase_items = [
                (str(name), value)
                for name, raw_value in reported_phases.items()
                if (value := _nonnegative_int(raw_value)) is not None
                and value > 0
            ]
        else:
            phase_items = []
        phases = ", ".join(
            f"{name or 'unlabeled'}={value}"
            for name, value in sorted(phase_items)) or "not recorded"
        issues.append(
            f"{title}: quota-limited; the manifest-bound summary reports "
            f"{count}/{shown_denominator} request rows returned HTTP 429; "
            f"phases: {phases}. The sealed journal contains no matching 429, "
            "so the source evidence is inconsistent and supports no "
            "endpoint-capacity conclusion")
        if len(counts) > 1:
            detail = ", ".join(
                f"{name}={value}" for name, value in positive.items())
            issues.append(
                f"{title}: manifest-bound 429 summary counts disagree "
                f"internally ({detail})")
    elif summary.get("quota_limited") is True:
        issues.append(
            f"{title}: the manifest-bound summary marks this run "
            "quota-limited, but exact HTTP 429 count, denominator, and phases "
            "are unavailable; this source is invalid as comparison evidence")
    return issues


def _explicit_measurement_issues(title: str, summary: dict) -> list[str]:
    """Recognize manifest-bound source invalidity before showing deltas."""
    issues: list[str] = []
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    answers = summary.get("answers")
    answers = answers if isinstance(answers, dict) else {}
    validity = summary.get("validity")
    validity = validity if isinstance(validity, dict) else {}
    decision = summary.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    measurement = decision.get("measurement_validity")
    measurement = measurement if isinstance(measurement, dict) else {}

    decision_code = measurement.get("code")
    if decision_code == "INVALID":
        issues.append(
            f"{title}: canonical measurement state is INVALID: "
            f"{measurement.get('reason') or 'no reason recorded'}")
    elif decision_code is not None and decision_code not in {"VALID", "CAUTION"}:
        issues.append(
            f"{title}: canonical measurement state is unrecognized "
            f"({decision_code!r})")

    invalid_reason = answers.get("invalid")
    if invalid_reason:
        issues.append(
            f"{title}: source measurement is explicitly INVALID: "
            f"{invalid_reason}")
    flags = []
    if summary.get("measurement_valid") is False:
        flags.append("summary.measurement_valid=false")
    if run.get("measurement_valid") is False:
        flags.append("summary.run.measurement_valid=false")
    if validity.get("valid") is False:
        flags.append("summary.validity.valid=false")
    status = validity.get("status")
    if isinstance(status, str) and status.strip().lower() in {
            "invalid", "inconclusive"}:
        flags.append(f"summary.validity.status={status}")
    if flags:
        issues.append(
            f"{title}: source measurement carries explicit invalidity "
            "evidence (" + ", ".join(flags) + ")")

    response_identity = summary.get("response_identity")
    response_identity = (response_identity
                         if isinstance(response_identity, dict) else {})
    if response_identity.get("status") == "invalid":
        issues.append(
            f"{title}: response-model identity is invalid: "
            f"{response_identity.get('invalid') or 'no reason recorded'}")

    runtime_quota = summary.get("runtime_quota_admission")
    runtime_quota = runtime_quota if isinstance(runtime_quota, dict) else {}
    runtime_status = runtime_quota.get("status")
    if runtime_status == "denied":
        issues.append(
            f"{title}: the local runtime quota guard denied a physical POST; "
            "the requested load was not fully delivered")
    elif runtime_status == "invalid_evidence":
        details = runtime_quota.get("invariant_errors")
        rendered = "; ".join(str(item) for item in details) \
            if isinstance(details, list) and details else "no reason recorded"
        issues.append(
            f"{title}: runtime quota-admission evidence is invalid: "
            f"{rendered}")

    metadata_state = run.get("endpoint_metadata_stability")
    if metadata_state == "changed":
        issues.append(
            f"{title}: serving-endpoint metadata changed between the pre-run "
            "and post-drain snapshots")
    return issues


def _explicit_measurement_warnings(title: str, summary: dict) -> list[str]:
    """Authenticated cautions that make relative judgment diagnostic-only."""
    warnings: list[str] = []
    decision = summary.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    measurement = decision.get("measurement_validity")
    measurement = measurement if isinstance(measurement, dict) else {}
    if measurement.get("code") == "CAUTION":
        warnings.append(
            f"{title}: canonical measurement state is CAUTION: "
            f"{measurement.get('reason') or 'no reason recorded'}")

    direct_paths = (
        ("cache fidelity", ("cache_fidelity", "warning")),
        ("token-shape fidelity", ("token_targeting", "warning")),
        ("token-usage coverage", ("throughput", "coverage_warning")),
        ("latency population", ("latency_population", "warning")),
        ("load delivery", ("client", "warning")),
        ("concurrency fidelity", ("concurrency", "warning")),
        ("rate-limit evidence", ("rate_limits", "warning")),
        ("Acceptance-target coverage", ("sla", "coverage_warning")),
        ("caller-latency coverage", ("sla", "caller_latency_warning")),
    )
    for label, path in direct_paths:
        value: object = summary
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and value.strip():
            rendered = f"{title}: {label}: {value.strip()}"
            if not any(value.strip() in existing for existing in warnings):
                warnings.append(rendered)

    response_identity = summary.get("response_identity")
    response_identity = (response_identity
                         if isinstance(response_identity, dict) else {})
    identity_warning = response_identity.get("warning")
    if isinstance(identity_warning, str) and identity_warning.strip():
        warnings.append(
            f"{title}: response-model identity: {identity_warning.strip()}")

    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    metadata_state = run.get("endpoint_metadata_stability")
    if metadata_state in {"unverified", "not_requested"}:
        detail = run.get("endpoint_metadata_warning") or (
            "pre-run/post-drain endpoint stability was not established")
        warnings.append(
            f"{title}: endpoint metadata stability is {metadata_state}: "
            f"{detail}")
    transport = _production_transport_evidence(summary)
    if not transport["exact_match"]:
        transport_warning = (
            f"{title}: production transport parity is "
            f"{str(transport['status']).lower()}: {transport['note']}. "
            "Connection pooling, HTTP protocol, and fresh-connection "
            "behavior can change DNS/TCP/TLS pressure; relative production "
            "performance claims are withheld")
        if not any(transport["note"] in existing for existing in warnings):
            warnings.append(transport_warning)
    return warnings


def _comparison_source_reference(position: int, d: Path,
                                 manifest: dict) -> dict:
    """Bind the exact source manifest plus its manifest-bound summary."""
    raw = _read_regular_bytes(d / "manifest.json")
    try:
        current = loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"invalid manifest.json in {d}: {json_error_detail(exc)}") from exc
    if current != manifest:
        raise ValueError(
            f"input manifest changed while constructing comparison: {d}")
    summary = _artifact_declarations(manifest, d)["summary.json"]
    return {
        "position": position,
        "artifact_id": manifest["artifact_id"],
        "logical_run_id": manifest["logical_run_id"],
        "execution_id": manifest["execution_id"],
        "workload_id": manifest["workload_id"],
        "manifest": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "summary": {
            "sha256": summary["sha256"],
            "bytes": summary["bytes"],
        },
    }


def _verify_comparison_source(source: object, position: int, d: Path) -> None:
    if not isinstance(source, dict) or source.get("position") != position:
        raise ValueError(f"invalid source position in comparison manifest for {d}")
    for field in ("artifact_id", "logical_run_id", "execution_id", "workload_id"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"invalid source {field} in comparison manifest for {d}")
    for field in ("manifest", "summary"):
        metadata = source.get(field)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"invalid source {field} metadata in comparison manifest for {d}")
        _identity_digest(metadata.get("sha256"),
                         f"sources[{position}].{field}.sha256", d)
        size = metadata.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"invalid source {field} byte count in comparison manifest for {d}")


def _html_text(value: object) -> str:
    """Escape untrusted text and remove controls that can spoof report UI."""
    return html.escape(sanitize_display_text(value), quote=True)


def _html_code(value: object) -> str:
    return f"<code>{_html_text(value)}</code>"


def _html_number(value: object, *, scale: float = 1.0,
                 decimals: int = 1, unit: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        return "<span class='na'>not reported</span>"
    number = float(value) * scale
    shown = f"{number:,.{decimals}f}"
    suffix = f" {_html_text(unit)}" if unit else ""
    return f"{shown}{suffix}"


def _comparison_endpoint_value(summary: dict, manifest: dict) -> str:
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    metadata = manifest.get("endpoint_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    base = manifest.get("endpoint_base_url") or run.get("endpoint_base_url")
    path = manifest.get("endpoint_path") or run.get("endpoint_path")
    model = (manifest.get("endpoint_model") or run.get("endpoint_model")
             or metadata.get("name"))
    parts = []
    if base or path:
        parts.append(f"route={base or ''}{path or ''}")
    if model:
        parts.append(f"model={model}")
    return "; ".join(parts) or "not recorded"


def _comparison_utc_instant(
        manifest: dict, iso_field: str, unix_field: str) -> str | None:
    raw_iso = manifest.get(iso_field)
    if isinstance(raw_iso, str) and raw_iso.strip():
        try:
            parsed = datetime.fromisoformat(
                raw_iso.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z")
    raw_unix = manifest.get(unix_field)
    if isinstance(raw_unix, (int, float)) \
            and not isinstance(raw_unix, bool) \
            and math.isfinite(float(raw_unix)):
        try:
            return datetime.fromtimestamp(
                float(raw_unix), timezone.utc).isoformat().replace(
                    "+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _comparison_utc_window(manifest: dict) -> str:
    start = _comparison_utc_instant(
        manifest, "run_started_at_utc", "run_started_at_unix")
    end = _comparison_utc_instant(
        manifest, "run_ended_at_utc", "run_ended_at_unix")
    if start and end:
        return f"{start} → {end}"
    if start:
        return f"{start} → end not recorded"
    if end:
        return f"start not recorded → {end}"
    return "not recorded"


def _comparison_endpoint_metadata(summary: dict, manifest: dict) -> dict:
    metadata = manifest.get("endpoint_metadata")
    if isinstance(metadata, dict):
        return metadata
    run = summary.get("run")
    run = run if isinstance(run, dict) else {}
    metadata = run.get("endpoint_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _comparison_deployment_value(summary: dict, manifest: dict) -> str:
    """Render only deployment facts actually recorded by the source run."""
    metadata = _comparison_endpoint_metadata(summary, manifest)
    parts = []
    for field, label in (
            ("name", "endpoint"), ("task", "task"),
            ("route_optimized", "route optimized"), ("ready", "ready")):
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{label}={value}")
    entities = metadata.get("served_entities")
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            facts = []
            for field, label in (
                    ("name", "name"), ("entity_version", "version"),
                    ("workload_type", "workload"),
                    ("workload_size", "size"),
                    ("provisioned_model_units", "PMUs"),
                    ("min_provisioned_throughput", "min throughput"),
                    ("max_provisioned_throughput", "max throughput"),
                    ("scale_to_zero_enabled", "scale to zero")):
                value = entity.get(field)
                if value is not None:
                    facts.append(f"{label}={value}")
            if facts:
                parts.append("served entity: " + ", ".join(facts))
    rate_limits = summary.get("rate_limits")
    configured = (rate_limits.get("configured")
                  if isinstance(rate_limits, dict) else None)
    if isinstance(configured, dict) and configured.get("deployment_mode"):
        parts.append(
            "configured deployment mode="
            + str(configured["deployment_mode"]))
    return "; ".join(parts) or "not recorded"


def _comparison_workload_digest(manifest: dict) -> str:
    digest = manifest.get("profile_sha256") \
        or manifest.get("profile_sha256_16")
    if isinstance(digest, str) and digest:
        return digest
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        for name in ("profile", "prompts"):
            entry = inputs.get(name)
            value = entry.get("sha256") if isinstance(entry, dict) else None
            if isinstance(value, str) and value:
                return value
    return "not recorded"


def _comparison_sample_count(summary: dict) -> str:
    sample = summary.get("sample")
    value = sample.get("n") if isinstance(sample, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return f"{value:,}"
    return "not recorded"


def _comparison_summary_429_count(summary: dict) -> int:
    block = summary.get("http_429")
    block = block if isinstance(block, dict) else {}
    statuses = summary.get("failures_by_http_status")
    statuses = statuses if isinstance(statuses, dict) else {}
    values = [
        _nonnegative_int(summary.get("http_429_count")),
        _nonnegative_int(block.get("count")),
        _nonnegative_int(statuses.get("429")),
    ]
    return max((value for value in values if value is not None), default=0)


def _comparison_relative_report_link(
        source_dir: Path, out_dir: Path, manifest: dict) -> str:
    declarations = manifest.get("artifacts")
    if not isinstance(declarations, dict) or "report.html" not in declarations:
        return "<span class='muted'>No sealed source report</span>"
    relative = os.path.relpath(source_dir / "report.html", start=out_dir)
    relative = relative.replace(os.sep, "/")
    href = quote(relative, safe="/._~-")
    return (
        f"<a href='{_html_text(href)}'>Open sealed source report</a>"
        f"<span class='link-path'>{_html_text(relative)}</span>"
    )


def _comparison_metric_value(summary: dict, path: tuple[str, ...]):
    value: object = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _comparison_ttft_definition(summaries: list[dict]) -> str:
    definitions = {
        ((summary.get("sla") or {}).get("ttft_definition")
         or (summary.get("run") or {}).get("ttft_definition"))
        for summary in summaries
    }
    return "first_visible" if definitions == {"first_visible"} \
        else "first_content"


def _comparison_directional_coverage_issues(
        summaries: list[dict], titles: list[str]) -> list[str]:
    """Require complete raw-event and caller-event populations for labels.

    Percentile arithmetic remains useful for diagnosis when an event is
    missing, but calling the surviving subset numerically preferred/adverse is
    a directional claim.  The comparison may make that claim only when every
    replay outcome contributed to both the service-time and caller-experienced
    primary/E2E populations.
    """
    first_visible = _comparison_ttft_definition(summaries) == "first_visible"
    required = (
        (("service TTFV", "ttfv_ms"),
         ("caller TTFV", "ttfv_corrected_ms"))
        if first_visible else
        (("service TTFT", "ttft_ms"),
         ("caller TTFT", "ttft_corrected_ms"))
    ) + (("service E2E", "e2e_ms"),
         ("caller E2E", "e2e_corrected_ms"))
    incomplete = []
    for title, summary in zip(titles, summaries):
        expected = _nonnegative_int(summary.get("requests_total"))
        for label, key in required:
            block = summary.get(key)
            observed = _nonnegative_int(
                block.get("n") if isinstance(block, dict) else None)
            if expected is None or expected == 0 or observed != expected:
                incomplete.append(
                    f"{title} {label} "
                    f"({observed if observed is not None else 'not recorded'}/"
                    f"{expected if expected is not None else 'not recorded'})")
    if not incomplete:
        return []
    return [
        "complete caller/event latency coverage is not established: "
        + ", ".join(incomplete)
        + ". Arithmetic percentiles remain diagnostic, but directional "
          "preference labels are withheld because event survivorship could "
          "change the ordering."
    ]


def _comparison_metric_table(
        summaries: list[dict], titles: list[str], valid: bool) -> str:
    """Render arithmetic deltas without claiming statistical improvement."""
    first_visible = _comparison_ttft_definition(summaries) == "first_visible"
    caller_first_metrics = (
        (("Caller TTFV p50", ("ttfv_corrected_ms", "p50"),
          "lower", "ms", 1.0, 1, "ms"),
         ("Caller TTFV p95", ("ttfv_corrected_ms", "p95"),
          "lower", "ms", 1.0, 1, "ms"))
        if first_visible else
        (("Caller TTFT p50", ("ttft_corrected_ms", "p50"),
          "lower", "ms", 1.0, 1, "ms"),
         ("Caller TTFT p95", ("ttft_corrected_ms", "p95"),
          "lower", "ms", 1.0, 1, "ms"))
    )
    service_first_metrics = (
        (("TTFV p50", ("ttfv_ms", "p50"), "lower", "ms", 1.0, 1, "ms"),
         ("TTFV p95", ("ttfv_ms", "p95"), "lower", "ms", 1.0, 1, "ms"),
         ("TTFV p99", ("ttfv_ms", "p99"), "lower", "ms", 1.0, 1, "ms"),
         ("TTFT (reasoning/visible/refusal onset) p95",
          ("ttft_ms", "p95"), "context", "ms", 1.0, 1, "ms"))
        if first_visible else
        (("TTFT p50", ("ttft_ms", "p50"), "lower", "ms", 1.0, 1, "ms"),
         ("TTFT p95", ("ttft_ms", "p95"), "lower", "ms", 1.0, 1, "ms"),
         ("TTFT p99", ("ttft_ms", "p99"), "lower", "ms", 1.0, 1, "ms"))
    )
    # label, path, direction, value unit, scale, decimals, delta unit
    metrics = caller_first_metrics + (
        ("Caller E2E p95", ("e2e_corrected_ms", "p95"),
         "lower", "ms", 1.0, 1, "ms"),
    ) + service_first_metrics + (
        ("TTFB p50", ("ttfb_ms", "p50"), "lower", "ms", 1.0, 1, "ms"),
        ("TTFB p95", ("ttfb_ms", "p95"), "lower", "ms", 1.0, 1, "ms"),
        ("TTFB p99", ("ttfb_ms", "p99"), "lower", "ms", 1.0, 1, "ms"),
        ("E2E p50", ("e2e_ms", "p50"), "lower", "ms", 1.0, 1, "ms"),
        ("E2E p95", ("e2e_ms", "p95"), "lower", "ms", 1.0, 1, "ms"),
        ("E2E p99", ("e2e_ms", "p99"), "lower", "ms", 1.0, 1, "ms"),
        ("Interchunk max p95", ("interchunk_max_ms", "p95"),
         "lower", "ms", 1.0, 1, "ms"),
        ("Interchunk max p99", ("interchunk_max_ms", "p99"),
         "lower", "ms", 1.0, 1, "ms"),
        ("Error rate", ("error_rate",), "lower", "%", 100.0, 2, "pp"),
        ("Input tokens / min", ("throughput", "input_tokens_per_min"),
         "context", "", 1.0, 0, ""),
        ("Output tokens / min", ("throughput", "output_tokens_per_min"),
         "context", "", 1.0, 0, ""),
        ("Dispatch lag p95", ("arrivals", "dispatch_lag_ms", "p95"),
         "lower", "ms", 1.0, 1, "ms"),
        ("HTTP request-start lateness p95",
         # Compatibility alias exists in both old and new summaries.
         ("arrivals", "wire_lateness_ms", "p95"),
         "lower", "ms", 1.0, 1, "ms"),
    )
    candidate_count = len(summaries) - 1
    head = [
        "<thead>",
        "<tr><th scope='col' rowspan='2' class='sticky-col'>Metric</th>",
        "<th scope='col' rowspan='2'>Direction</th>",
        "<th scope='col' rowspan='2'>Baseline absolute</th>",
    ]
    for title in titles[1:]:
        head.append(
            f"<th scope='colgroup' colspan='3'>{_html_text(title)}</th>")
    head.extend(["</tr><tr>"])
    for _ in range(candidate_count):
        head.extend([
            "<th scope='col'>Candidate absolute</th>",
            "<th scope='col'>Absolute delta</th>",
            "<th scope='col'>Percent delta</th>",
        ])
    head.extend(["</tr></thead>"])

    body = ["<tbody>"]
    for label, path, direction, unit, scale, decimals, delta_unit in metrics:
        baseline = _comparison_metric_value(summaries[0], path)
        baseline_number = (float(baseline) if isinstance(
            baseline, (int, float)) and not isinstance(baseline, bool) else None)
        direction_text = (
            {
                "lower": "lower preferred; untested",
                "higher": "higher preferred; untested",
                "context": "context only",
            }[direction]
            if valid else
            ("context only" if direction == "context" else
             "direction withheld"))
        body.extend([
            "<tr>",
            f"<th scope='row' class='sticky-col'>{_html_text(label)}</th>",
            f"<td class='direction'>{direction_text}</td>",
            f"<td>{_html_number(baseline, scale=scale, decimals=decimals, unit=unit)}</td>",
        ])
        for candidate in summaries[1:]:
            value = _comparison_metric_value(candidate, path)
            candidate_number = (float(value) if isinstance(
                value, (int, float)) and not isinstance(value, bool) else None)
            body.append(
                f"<td>{_html_number(value, scale=scale, decimals=decimals, unit=unit)}</td>")
            if baseline_number is None or candidate_number is None:
                body.extend([
                    "<td class='delta'>not available</td>",
                    "<td class='delta'>not available</td>",
                ])
                continue
            delta = (candidate_number - baseline_number) * scale
            sign = "+" if delta > 0 else ""
            delta_suffix = f" {_html_text(delta_unit)}" if delta_unit else ""
            assessment = ""
            signal_class = ""
            if valid and direction in {"lower", "higher"} and delta != 0:
                preferred_direction = (
                    (direction == "lower" and delta < 0)
                    or (direction == "higher" and delta > 0))
                assessment = (
                    "numerically preferred" if preferred_direction else
                    "numerically adverse")
                signal_class = " signal-change"
            assessment_html = (
                f"<span class='assessment'>{assessment}</span>"
                if assessment else "")
            body.append(
                f"<td class='delta{signal_class}'>{sign}{delta:,.{decimals}f}"
                f"{delta_suffix}{assessment_html}</td>")
            if baseline_number == 0:
                body.append(
                    "<td class='delta'>not defined (baseline is zero)</td>")
            else:
                percent = (candidate_number - baseline_number) \
                    / abs(baseline_number) * 100.0
                pct_sign = "+" if percent > 0 else ""
                body.append(
                    f"<td class='delta{signal_class}'>{pct_sign}{percent:,.1f}%"
                    f"{assessment_html}</td>")
        body.append("</tr>")
    body.append("</tbody>")
    return "".join(head + body)


def _render_comparison_html(
        out_dir: Path, dirs: list[Path], summaries: list[dict],
        manifests: list[dict], request_evidence: list[dict],
        titles: list[str], compatibility_issues: list[str],
        warnings: list[str], artifact_id: str) -> str:
    """Create a sealed, dependency-free decision and diagnostic surface."""
    comparison_state = (
        "invalid" if compatibility_issues else
        "qualified" if warnings else "valid")
    arithmetic_labels_allowed = comparison_state == "valid"
    status = {
        "valid": "VALID COMPARISON",
        "qualified": "QUALIFIED COMPARISON",
        "invalid": "INVALID COMPARISON",
    }[comparison_state]
    status_class = comparison_state
    disposition = {
        "valid": (
            "Compatibility and measurement-quality checks passed. Deltas are "
            "arithmetic observations relative to the first input, not "
            "statistically demonstrated improvements or regressions."),
        "qualified": (
            "Diagnostic-only while measurement warnings remain. Do not quote "
            "relative performance, rank candidates, or use directional delta "
            "judgments until every warning is resolved and the runs repeat."),
        "invalid": (
            "Diagnostic-only. Do not quote relative results, rank candidates, "
            "or draw endpoint-capacity conclusions until every issue is "
            "resolved and the runs are repeated."),
    }[comparison_state]

    source_cards = []
    for position, (title, source_dir, summary, manifest) in enumerate(
            zip(titles, dirs, summaries, manifests)):
        role = "Baseline · first input" if position == 0 else \
            f"Candidate {position}"
        endpoint = _comparison_endpoint_value(summary, manifest)
        deployment = _comparison_deployment_value(summary, manifest)
        workload_digest = _comparison_workload_digest(manifest)
        transport = _production_transport_evidence(summary)
        source_cards.append(
            "<article class='source-card'>"
            f"<div class='eyebrow'>{_html_text(role)}</div>"
            f"<h3>{_html_text(title)}</h3>"
            "<dl class='source-meta'>"
            f"<dt>Artifact ID</dt><dd>{_html_code(manifest['artifact_id'])}</dd>"
            f"<dt>UTC window</dt><dd>"
            f"{_html_text(_comparison_utc_window(manifest))}</dd>"
            f"<dt>Endpoint identity</dt><dd>{_html_code(endpoint)}</dd>"
            f"<dt>Deployment context</dt><dd>{_html_text(deployment)}</dd>"
            f"<dt>Workload ID</dt><dd>"
            f"{_html_code(manifest.get('workload_id') or 'not recorded')}</dd>"
            f"<dt>Workload digest</dt><dd>"
            f"{_html_code(workload_digest)}</dd>"
            f"<dt>Sample count</dt><dd>"
            f"{_html_text(_comparison_sample_count(summary))}</dd>"
            f"<dt>Transport parity</dt><dd>"
            f"{_html_text(transport['status'])} · "
            f"{_html_text(transport['note'])}</dd>"
            "</dl>"
            f"<div class='source-link'>{_comparison_relative_report_link(source_dir, out_dir, manifest)}</div>"
            "</article>"
        )

    issue_blocks = []
    if compatibility_issues:
        items = "".join(
            f"<li>{_html_text(issue)}</li>" for issue in compatibility_issues)
        issue_blocks.append(
            "<section class='callout invalid-callout' aria-labelledby='issues-heading'>"
            "<h2 id='issues-heading'>Why this comparison is invalid</h2>"
            f"<details><summary>Show {len(compatibility_issues)} technical "
            f"compatibility issues</summary><ol>{items}</ol></details></section>"
        )
    if warnings:
        items = "".join(
            f"<li>{_html_text(warning)}</li>" for warning in warnings)
        issue_blocks.append(
            "<section id='warnings' class='callout warning-callout' "
            "aria-labelledby='first-warning-heading'>"
            "<h2 id='first-warning-heading'>Why this comparison is "
            "diagnostic-only</h2>"
            f"<p>{len(warnings)} measurement warning(s) block arithmetic "
            "preference labels and relative performance claims:</p>"
            f"<details><summary>Show {len(warnings)} measurement warnings"
            f"</summary><ol>{items}</ol></details>"
            "</section>"
        )
    issue_block = "".join(issue_blocks)

    def same(values: list[object]) -> bool:
        return bool(values) and all(value is not None for value in values) \
            and len({_stable(value) for value in values}) == 1

    harness_values = [
        (manifest.get("harness_version") or summary.get("harness_version"),
         manifest.get("latency_basis") or summary.get("latency_basis"))
        for summary, manifest in zip(summaries, manifests)
    ]
    workload_values = [
        (manifest.get("workload_id"), manifest.get("profile_sha256")
         or manifest.get("profile_sha256_16"))
        for manifest in manifests
    ]
    parameter_values = [manifest.get("request_params") for manifest in manifests]
    sample_ready = all(
        not (summary.get("sample") or {}).get("warning")
        and (summary.get("drift") or {}).get("drift_kind") == "stable"
        for summary in summaries
    )

    def runtime_quota_status(summary: dict) -> str | None:
        block = summary.get("runtime_quota_admission")
        return block.get("status") if isinstance(block, dict) else None

    any_quota_issue = any(
        journal["count"] > 0 or summary.get("quota_limited") is True
        or _comparison_summary_429_count(summary) > 0
        or runtime_quota_status(summary) in {"denied", "invalid_evidence"}
        for summary, journal in zip(summaries, request_evidence)
    )
    any_request_rows = any(journal["total"] > 0 for journal in request_evidence)

    matrix_rows: list[tuple[str, str, list[str]]] = []
    identity_cells = []
    for manifest in manifests:
        clean = "clean" if manifest.get("git_dirty") is False else \
            "dirty or unknown"
        identity_cells.append(
            f"Artifact {_html_code(manifest.get('artifact_id'))}<br>"
            f"source {_html_code(manifest.get('git_commit'))} · {_html_text(clean)}")
    matrix_rows.append(("Artifact / source identity", "Bound", identity_cells))
    matrix_rows.append((
        "Harness / latency basis",
        "Match" if same(harness_values) else "Invalid",
        [f"harness {_html_code(value[0])}<br>{_html_text(value[1] or 'not recorded')}"
         for value in harness_values],
    ))
    transport_values = [
        _production_transport_evidence(summary) for summary in summaries]
    transport_state = (
        "Match" if all(value["exact_match"] for value in transport_values)
        else "Review")
    transport_cells = []
    for value in transport_values:
        exact = "yes" if value["exact_match"] else "no"
        recorded = value["recorded_match"]
        recorded_text = (
            "true" if recorded is True else "false" if recorded is False
            else "not recorded")
        transport_cells.append(
            f"benchmark policy: "
            f"{_html_code(value['actual_policy_id'] or 'not recorded')}"
            f"<br>declared production policy: "
            f"{_html_code(value['declared_production_policy'] or 'not recorded')}"
            f"<br>recorded match: {_html_text(recorded_text)}"
            f"<br>exact match: {_html_text(exact)}"
            f"<br>{_html_text(value['note'])}")
    matrix_rows.append((
        "Production transport parity", transport_state, transport_cells,
    ))
    matrix_rows.append((
        "Workload / profile hash",
        "Match" if same(workload_values) else "Invalid",
        [f"workload {_html_code(value[0])}<br>profile {_html_code(value[1])}"
         for value in workload_values],
    ))
    matrix_rows.append((
        "Request parameters",
        "Match" if same(parameter_values) else "Invalid",
        [_html_code(_stable(value)) if value is not None
         else "<span class='na'>not recorded</span>"
         for value in parameter_values],
    ))
    matrix_rows.append((
        "Endpoint",
        "Context",
        [_html_code(_comparison_endpoint_value(summary, manifest))
         for summary, manifest in zip(summaries, manifests)],
    ))

    response_identity_cells = []
    response_identity_statuses = []
    for summary in summaries:
        identity = summary.get("response_identity")
        identity = identity if isinstance(identity, dict) else {}
        identity_status = identity.get("status") or "not recorded"
        response_identity_statuses.append(identity_status)
        models = identity.get("models")
        models = models if isinstance(models, dict) else {}
        expected = identity.get("expected_models")
        expected = expected if isinstance(expected, list) else []
        response_identity_cells.append(
            f"status: {_html_text(identity_status)}"
            f"<br>observed: {_html_code(_stable(models.get('counts') or {}))}"
            f"<br>expected: {_html_code(_stable(expected))}"
        )
    if any(status == "invalid" for status in response_identity_statuses):
        response_identity_state = "Invalid"
    elif response_identity_statuses and all(
            status == "bound" for status in response_identity_statuses):
        response_identity_state = "Bound"
    else:
        response_identity_state = "Review"
    matrix_rows.append((
        "Response-model identity", response_identity_state,
        response_identity_cells,
    ))

    endpoint_stability_cells = []
    endpoint_stability_states = []
    for summary in summaries:
        run = summary.get("run")
        run = run if isinstance(run, dict) else {}
        state = run.get("endpoint_metadata_stability") or "not recorded"
        endpoint_stability_states.append(state)
        warning = run.get("endpoint_metadata_warning")
        endpoint_stability_cells.append(
            _html_text(state)
            + (f"<br>{_html_text(warning)}" if warning else ""))
    if any(state == "changed" for state in endpoint_stability_states):
        endpoint_stability_state = "Invalid"
    elif endpoint_stability_states and all(
            state == "stable" for state in endpoint_stability_states):
        endpoint_stability_state = "Ready"
    else:
        endpoint_stability_state = "Review"
    matrix_rows.append((
        "Endpoint metadata: pre-run vs post-drain",
        endpoint_stability_state, endpoint_stability_cells,
    ))

    sample_cells = []
    for summary in summaries:
        sample = summary.get("sample")
        sample = sample if isinstance(sample, dict) else {}
        drift = summary.get("drift")
        drift = drift if isinstance(drift, dict) else {}
        sample_cells.append(
            f"n={_html_text(sample.get('n', 'not recorded'))}<br>"
            f"stability={_html_text(drift.get('drift_kind') or 'not established')}"
            + (f"<br>{_html_text(sample['warning'])}"
               if sample.get("warning") else "")
        )
    matrix_rows.append((
        "Sample / stability", "Ready" if sample_ready else "Review",
        sample_cells,
    ))
    quota_cells = []
    for summary, journal in zip(summaries, request_evidence):
        phases = ", ".join(
            f"{name or 'unlabeled'}={count}"
            for name, count in sorted(journal["phases"].items())) or "none"
        quota_flag = summary.get("quota_limited")
        quota_label = "yes" if quota_flag is True else \
            "no" if quota_flag is False else "not recorded"
        runtime_quota = summary.get("runtime_quota_admission")
        runtime_quota = runtime_quota \
            if isinstance(runtime_quota, dict) else {}
        guard_status = runtime_quota.get("status") or "not recorded"
        guard_id = runtime_quota.get("guard_id") or "not recorded"
        denied_attempts = runtime_quota.get(
            "denied_attempts_in_captured_rows", "not recorded")
        quota_cells.append(
            f"HTTP 429: <strong>{journal['count']}/{journal['total']}</strong>"
            f"<br>phases: {_html_text(phases)}"
            f"<br>summary quota-limited: {_html_text(quota_label)}"
            f"<br>local guard: {_html_text(guard_status)}"
            f"<br>guard id: {_html_code(guard_id)}"
            f"<br>locally denied attempts: {_html_text(denied_attempts)}"
        )
    quota_status = "Invalid" if any_quota_issue else \
        "Clear" if any_request_rows else "No request rows"
    matrix_rows.append((
        "Provider HTTP 429 / local quota admission", quota_status,
        quota_cells,
    ))

    matrix_head = "".join(
        f"<th scope='col'>{_html_text(title)}</th>" for title in titles)
    matrix_body = []
    for dimension, state, cells in matrix_rows:
        state_class = (
            "state-pass" if state in {"Bound", "Match", "Ready", "Clear"}
            else "state-invalid" if state == "Invalid" else "state-review"
        )
        matrix_body.append(
            f"<tr><th scope='row' class='sticky-col'>"
            f"{_html_text(dimension)}</th>"
            f"<td><span class='matrix-state {state_class}'>{_html_text(state)}</span></td>"
            + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    metric_table = _comparison_metric_table(
        summaries, titles, arithmetic_labels_allowed)
    color_note = (
        "Numeric direction labels are shown because the comparison is valid, "
        "but no repeat-run uncertainty or practical-effect threshold was "
        "configured. They are not improvement/regression verdicts. Positive "
        "deltas mean the candidate value is numerically higher."
        if arithmetic_labels_allowed else
        "All deltas are neutral diagnostic values. Arithmetic preference "
        "labels and performance judgments are intentionally suppressed for this "
        f"{comparison_state} comparison."
    )
    baseline_title = titles[0]
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<meta name='referrer' content='no-referrer'>
<meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; font-src 'none'; script-src 'none'; connect-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'">
<title>{_html_text(status)} · endpoint comparison</title>
<style>
:root{{--ink:#122033;--muted:#5d6b7c;--line:#dce3ec;--soft:#f4f7fb;--navy:#172f52;--blue:#2f67d8;--green:#117a55;--green-soft:#e8f7f0;--red:#b42318;--red-soft:#fff0ef;--amber:#8a5700;--amber-soft:#fff7df;--white:#fff}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#edf2f7;color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
a{{color:#174ea6;text-underline-offset:3px}}code{{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}}.shell{{max-width:1500px;margin:auto;background:var(--white);min-height:100vh;box-shadow:0 0 45px #10233a1a}}
header{{padding:32px 42px 0}}.eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}}h1{{font-size:clamp(30px,4vw,50px);line-height:1.05;letter-spacing:-.035em;margin:8px 0 20px}}h2{{font-size:24px;line-height:1.2;margin:3px 0 0}}h3{{font-size:17px;margin:4px 0 9px}}
.hero{{border:1px solid var(--line);border-left:8px solid var(--green);border-radius:14px;padding:22px 24px;background:linear-gradient(135deg,#fff,#f3fbf7);display:grid;grid-template-columns:minmax(230px,.75fr) 2fr;gap:26px;align-items:center}}.hero.invalid{{border-left-color:var(--red);background:linear-gradient(135deg,#fff,#fff4f3)}}.hero.qualified{{border-left-color:var(--amber);background:linear-gradient(135deg,#fff,#fffaf0)}}.status{{font-size:22px;font-weight:850;color:var(--green)}}.invalid .status{{color:var(--red)}}.qualified .status{{color:var(--amber)}}.disposition{{font-size:17px;margin:4px 0 10px;max-width:860px}}.hero-facts{{display:flex;flex-wrap:wrap;gap:8px 22px;color:var(--muted)}}
.callout{{margin-top:16px;border-radius:12px;padding:16px 20px}}.callout h2{{font-size:18px}}.callout ol{{margin:8px 0 0;padding-left:22px}}.invalid-callout{{border:1px solid #fac5c1;background:var(--red-soft)}}.warning-callout{{border:1px solid #f1d58a;background:var(--amber-soft)}}
nav{{margin-top:18px;border-block:1px solid var(--line);display:flex;gap:22px;padding:12px 42px;overflow:auto;background:#fff;position:sticky;top:0;z-index:2}}nav a{{white-space:nowrap;font-weight:700;text-decoration:none}}main{{padding:0 42px 48px}}section{{padding:31px 0;border-bottom:1px solid var(--line)}}.source-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;margin-top:16px}}.source-card{{border:1px solid var(--line);border-radius:12px;padding:16px;background:var(--soft)}}.source-meta{{display:grid;grid-template-columns:minmax(112px,.38fr) minmax(0,1fr);gap:6px 10px;margin:10px 0 0;font-size:12px}}.source-meta dt{{font-weight:800;color:#435168}}.source-meta dd{{margin:0;min-width:0;overflow-wrap:anywhere}}.source-link{{margin-top:12px}}.link-path{{display:block;color:var(--muted);font-size:11px;overflow-wrap:anywhere}}.muted,.na{{color:var(--muted)}}
.section-head{{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:14px}}.count{{display:inline-grid;place-items:center;min-width:35px;height:35px;padding:0 9px;border-radius:20px;background:var(--soft);font-weight:800}}.scroll-hint{{display:none}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;overscroll-behavior-inline:contain;scrollbar-gutter:stable}}table{{border-collapse:separate;border-spacing:0;width:100%;min-width:900px}}caption{{text-align:left;padding:12px 14px;background:var(--soft);font-weight:700;color:var(--muted)}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);border-right:1px solid var(--line);text-align:right;vertical-align:top}}th:last-child,td:last-child{{border-right:0}}thead th{{background:var(--navy);color:#fff;font-size:12px;letter-spacing:.02em}}tbody th{{text-align:left;background:#f8fafc;min-width:160px}}.compat td{{text-align:left;min-width:210px}}.compat td:nth-child(2){{min-width:115px}}.matrix-state{{display:inline-block;border-radius:20px;padding:3px 9px;font-size:12px;font-weight:800}}.state-pass{{color:var(--green);background:var(--green-soft)}}.state-invalid{{color:var(--red);background:var(--red-soft)}}.state-review{{color:var(--amber);background:var(--amber-soft)}}.direction{{color:var(--muted);font-size:12px;white-space:nowrap}}.delta{{white-space:nowrap}}.signal-change{{color:#174ea6;background:#eef4ff;font-weight:750}}.assessment{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}.warning-list{{padding-left:22px}}.warning-list li+li{{margin-top:10px}}.method-note{{color:var(--muted);max-width:1000px}}.method-card{{border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:11px;background:var(--soft);padding:14px 16px;margin:14px 0}}.method-card h3{{margin:0 0 5px}}.method-card p{{margin:0}}
footer{{padding:22px 42px;background:var(--navy);color:#dce7f8}}footer code{{color:#fff}}
.print-stamp{{display:none}}
@media(max-width:720px){{header,main{{padding-left:18px;padding-right:18px}}header{{padding-top:22px}}nav{{padding-left:18px;padding-right:18px}}.hero{{grid-template-columns:1fr;padding:18px}}h1{{font-size:34px}}section{{padding:24px 0}}.source-grid{{grid-template-columns:1fr}}.source-meta{{grid-template-columns:1fr;gap:2px;font-size:13px}}.source-meta dt{{margin-top:7px}}.source-meta dd,.source-meta code{{font-size:12px}}.scroll-hint{{display:flex;align-items:center;gap:7px;margin:0 0 7px;padding:7px 9px;border-radius:8px;background:#eef4ff;color:#174ea6;font-size:12px;font-weight:750}}.table-wrap{{box-shadow:inset -12px 0 12px -14px #122033;-webkit-overflow-scrolling:touch}}.table-wrap:focus-visible{{outline:3px solid #155eef;outline-offset:3px}}.table-wrap .sticky-col{{position:sticky;inset-inline-start:0;z-index:2;box-shadow:5px 0 7px -7px #122033}}.table-wrap thead .sticky-col{{z-index:4;background:var(--navy)}}.table-wrap tbody .sticky-col{{background:#f8fafc}}th,td{{padding:9px 10px}}footer{{padding:20px 18px}}}}
@media print{{@page{{size:landscape;margin:10mm}}body{{background:#fff;font-size:10px}}.shell{{box-shadow:none;max-width:none}}nav{{display:none}}header,main{{padding-left:0;padding-right:0}}section{{break-inside:auto;padding:14px 0}}.hero,.source-card,.callout,.method-card{{break-inside:avoid;print-color-adjust:exact;-webkit-print-color-adjust:exact}}#method{{break-inside:avoid-page;break-after:avoid-page;margin-bottom:6px}}.source-meta{{font-size:9px;gap:3px 7px}}.source-link{{margin-top:6px}}.print-stamp{{display:block;border:1px solid #98a2b3;padding:2.5mm 3mm;margin:4mm 0 2mm;background:#fff;color:#344054;text-align:center;font-size:8pt;line-height:1.25;break-inside:avoid}}.scroll-hint{{display:none}}.table-wrap{{overflow:visible;box-shadow:none}}.table-wrap .sticky-col{{position:static;box-shadow:none}}table{{min-width:0}}th,td{{padding:5px 6px}}thead{{display:table-header-group}}tr{{break-inside:avoid}}a::after{{content:" (" attr(href) ")";font-size:9px}}footer{{display:none}}}}
.callout{{min-width:0}}.callout li{{overflow-wrap:anywhere;word-break:break-word}}
.callout details summary{{cursor:pointer;font-weight:800;margin-top:9px}}
</style>
</head>
<body><div class='shell'>
<header>
<div class='eyebrow'>Sealed endpoint comparison</div>
<h1>Benchmark comparison</h1>
<div class='hero {status_class}' role='status' aria-live='off'>
<div><div class='status'>{status}</div><div>{len(summaries)} internally hash-verified inputs</div></div>
<div><p class='disposition'>{_html_text(disposition)}</p><div class='hero-facts'>
<span><strong>Baseline:</strong> {_html_text(baseline_title)} (first input)</span>
<span><strong>Compatibility issues:</strong> {len(compatibility_issues)}</span>
<span><strong>Warnings:</strong> {len(warnings)}</span>
</div></div></div>
{issue_block}
<div class='print-stamp' role='note'>UNSEALED PRINT/PDF DERIVATIVE: verify the comparison manifest · artifact {_html_text(artifact_id)} · internal hashes are not a digital signature</div>
<div class='source-grid'>{''.join(source_cards)}</div>
</header>
<nav aria-label='Report sections'><a href='#compatibility'>Compatibility</a><a href='#metrics'>Metrics and deltas</a>{"<a href='#warnings'>Warnings</a>" if warnings else ""}<a href='#method'>How to read</a></nav>
<main>
<section id='compatibility' aria-labelledby='compatibility-heading'>
<div class='section-head'><div><div class='eyebrow'>Evidence gate</div><h2 id='compatibility-heading'>Compatibility matrix</h2></div></div>
<div class='scroll-hint' id='compatibility-scroll-hint' role='note'><span aria-hidden='true'>↔</span> Scroll horizontally; the Dimension column stays visible.</div>
<div class='table-wrap' tabindex='0' role='region' aria-labelledby='compatibility-heading' aria-describedby='compatibility-scroll-hint'><table class='compat'><caption>Each cell comes from a manifest-bound source manifest, summary, or request journal. Internal hashes are not a digital signature.</caption><thead><tr><th scope='col' class='sticky-col'>Dimension</th><th scope='col'>State</th>{matrix_head}</tr></thead><tbody>{''.join(matrix_body)}</tbody></table></div>
</section>
<section id='method' class='method-card' aria-labelledby='method-heading'><div class='eyebrow'>Interpretation contract</div><h2 id='method-heading'>How to read this report</h2><p class='method-note'>Latency percentiles describe successful requests and can be biased when errors occur. Throughput and token counts are context, not an automatic quality ranking. Production transport parity requires an explicit exact actual-versus-declared connection-policy match for every source. HTTP 429 evidence includes setup and replay phases from each sealed journal. This file contains no scripts, remote assets, remote fonts, or network requests.</p></section>
<section id='metrics' aria-labelledby='metrics-heading'>
<div class='section-head'><div><div class='eyebrow'>First-input baseline</div><h2 id='metrics-heading'>Absolute values and deltas</h2></div></div>
<p class='method-note'>Baseline is explicitly the first input: <strong>{_html_text(baseline_title)}</strong>. Absolute delta is candidate minus baseline. {_html_text(color_note)}</p>
<div class='scroll-hint' id='metrics-scroll-hint' role='note'><span aria-hidden='true'>↔</span> Scroll horizontally; the Metric column stays visible.</div>
<div class='table-wrap' tabindex='0' role='region' aria-labelledby='metrics-heading' aria-describedby='metrics-scroll-hint'><table><caption>Candidate values and deltas relative to the first input baseline.</caption>{metric_table}</table></div>
</section>
</main>
<footer>Generated by traffic-replay {_html_code(__version__)} · comparison artifact is complete only when this file and <code>comparison.md</code> match <code>manifest.json</code>.</footer>
</div></body></html>
"""


def verify_comparison_output(out_dir: str | Path) -> dict:
    """Verify the completion chain and rendered artifact of a comparison."""
    d = Path(out_dir)
    try:
        info = d.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"comparison directory not found: {d}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"comparison directory is not a regular directory: {d}")
    if _has_path(d / _WRITING_MARKER):
        raise ValueError(f"comparison is still being written: {d}")
    _require_regular(d / _COMPLETE_MARKER, "completion marker")
    _require_regular(d / "manifest.json", "manifest.json")
    _require_regular(d / "comparison.md", "comparison.md")
    _require_regular(d / "comparison.html", "comparison.html")
    completion = _load_json_object(d / _COMPLETE_MARKER, "completion marker")
    manifest = _load_json_object(d / "manifest.json", "manifest.json")
    if manifest.get("manifest_schema_version") != 3 \
            or manifest.get("artifact_type") != "comparison":
        raise ValueError(f"unsupported comparison manifest in {d}")
    artifact_id = manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError(f"invalid comparison artifact_id in {d}")
    if completion.get("status") != "complete" \
            or completion.get("artifact_type") != "comparison" \
            or completion.get("artifact_id") != artifact_id:
        raise ValueError(f"completion marker and comparison manifest disagree in {d}")
    actual_manifest, actual_bytes, _rows = _measure_regular(d / "manifest.json")
    expected_manifest = _identity_digest(
        completion.get("manifest_sha256"),
        "completion marker manifest_sha256", d)
    if not hmac.compare_digest(actual_manifest, expected_manifest):
        raise ValueError(f"manifest SHA-256 mismatch for comparison {d}")
    declared_bytes = completion.get("manifest_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) \
            or declared_bytes != actual_bytes:
        raise ValueError(f"manifest byte count mismatch for comparison {d}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) < 2 \
            or manifest.get("input_count") != len(sources):
        raise ValueError(f"invalid sources in comparison manifest for {d}")
    for position, source in enumerate(sources):
        _verify_comparison_source(source, position, d)
    _verify_artifacts(d, manifest, ("comparison.md", "comparison.html"))
    return manifest


def compare_runs(out_dir, input_dirs) -> Path:
    """Tabulate several runs one column each, on identical measurement, and
    invalidate the comparison when their per-request cached prompt-token
    fractions or provenance diverge enough to make latency incomparable."""
    dirs, manifests = _validated_input_dirs(
        input_dirs, "summary.json", "compare")
    summ = [_verified_comparison_summary(d, manifest)
            for d, manifest in zip(dirs, manifests)]
    request_evidence = [
        _verified_comparison_request_evidence(d, manifest)
        for d, manifest in zip(dirs, manifests)
    ]
    source_state = snapshot_source_state(Path(__file__).parent)
    source_commit = source_state.get("git_commit")
    source_tree = source_state.get("source_tree_sha256")
    generator_source_reconstructible = (
        source_state.get("git_dirty") is False
        and isinstance(source_commit, str) and bool(source_commit.strip())
        and isinstance(source_tree, str) and bool(_SHA256_RE.fullmatch(source_tree))
    )
    raw_titles = [_run_title(d, s) for d, s in zip(dirs, summ)]
    titles = [markdown_plain_text(title) or f"run {position + 1}"
              for position, title in enumerate(raw_titles)]
    n = len(titles)
    hdr = "| metric / quantile | " + " | ".join(titles) + " |"
    sep = "|---" * (n + 1) + "|"
    L = ["# endpoint comparison", "",
         "Runs measured on the same instrument. Read the warnings and the "
         "believability section before trusting the latency tables.", ""]
    L += ["## source runs", ""]
    for position, (title, summary, manifest) in enumerate(
            zip(titles, summ, manifests)):
        role = "Baseline (first input)" if position == 0 else \
            f"Candidate {position}"
        transport = _production_transport_evidence(summary)
        L += [
            f"### {role}: {title}",
            "",
            "- Artifact ID: "
            + markdown_plain_text(manifest.get("artifact_id")
                                  or "not recorded"),
            "- UTC window: "
            + markdown_plain_text(_comparison_utc_window(manifest)),
            "- Endpoint identity: "
            + markdown_plain_text(
                _comparison_endpoint_value(summary, manifest)),
            "- Deployment context: "
            + markdown_plain_text(
                _comparison_deployment_value(summary, manifest)),
            "- Workload ID: "
            + markdown_plain_text(manifest.get("workload_id")
                                  or "not recorded"),
            "- Workload digest: "
            + markdown_plain_text(_comparison_workload_digest(manifest)),
            "- Sample count: "
            + markdown_plain_text(_comparison_sample_count(summary)),
            "- Production transport parity: "
            + markdown_plain_text(
                f"{transport['status']}; benchmark policy="
                f"{transport['actual_policy_id'] or 'not recorded'}; "
                f"declared production policy="
                f"{transport['declared_production_policy'] or 'not recorded'}; "
                f"{transport['note']}"),
            "",
        ]

    compatibility_issues = _compatibility_issues(
        dirs, summ, manifests, merging=False)
    for title, summary, journal in zip(raw_titles, summ, request_evidence):
        compatibility_issues.extend(
            _comparison_http_429_issues(title, summary, journal))
        compatibility_issues.extend(
            _explicit_measurement_issues(title, summary))
    if not generator_source_reconstructible:
        if source_state.get("git_dirty") is not False:
            reason = "dirty or unknown Git state"
        elif not isinstance(source_commit, str) or not source_commit.strip():
            reason = "no source commit"
        else:
            reason = "no valid source-tree digest"
        compatibility_issues.append(
            f"the comparison generator has {reason}; the code that rendered "
            "this table is not reconstructible")
    if compatibility_issues:
        L += ["## INVALID COMPARISON / INCONCLUSIVE: diagnostic-only", "",
              "The tables below are retained for diagnosis only. Do not quote "
              "a winner or a relative latency until every incompatibility is "
              "resolved and the runs are repeated.", ""]
        for issue in compatibility_issues:
            L += [f"> INVALID: {markdown_plain_text(issue)}", ""]

    # Everything that can make a side-by-side dishonest goes ABOVE the tables.
    # A reader who stops after the first screen still sees the disqualifiers.
    warns: list[str] = []
    for title, summary in zip(raw_titles, summ):
        warns.extend(_explicit_measurement_warnings(title, summary))
    warns.extend(_comparison_directional_coverage_issues(summ, titles))

    # Arithmetic can still be rendered when provenance is incomplete, but it
    # must not receive a green comparison state. Unknown endpoint identity or
    # an absent/partial request journal cannot establish which system was
    # exercised or that HTTP 429 was absent.
    missing_endpoint_identity = [
        title for title, summary, manifest in zip(titles, summ, manifests)
        if _comparison_endpoint_value(summary, manifest) == "not recorded"
    ]
    if missing_endpoint_identity:
        warns.append(
            "endpoint identity is not recorded for "
            f"{', '.join(missing_endpoint_identity)}; endpoint-under-test "
            "provenance is incomplete, so the columns cannot support a "
            "relative performance claim")

    no_request_rows = [
        title for title, journal in zip(titles, request_evidence)
        if journal["total"] == 0
    ]
    if no_request_rows:
        warns.append(
            "no manifest-bound request rows are available for "
            f"{', '.join(no_request_rows)}; absence of HTTP 429 and request "
            "outcome evidence is not established")

    incomplete_status_evidence = [
        (title, journal["http_status_observed_for"], journal["total"])
        for title, journal in zip(titles, request_evidence)
        if journal["total"] > 0
        and journal["http_status_observed_for"] != journal["total"]
    ]
    if incomplete_status_evidence:
        detail = ", ".join(
            f"{title} ({observed}/{total} rows)"
            for title, observed, total in incomplete_status_evidence)
        warns.append(
            "HTTP status is not recorded for every manifest-bound request "
            f"row: {detail}. Absence of HTTP 429 is not established")

    incomplete_replay_evidence = []
    for title, summary, journal in zip(titles, summ, request_evidence):
        expected = _nonnegative_int(summary.get("requests_total"))
        observed = journal["phase_totals"].get("replay", 0)
        if expected is not None and observed != expected:
            incomplete_replay_evidence.append((title, observed, expected))
    if incomplete_replay_evidence:
        detail = ", ".join(
            f"{title} ({observed}/{expected} replay rows)"
            for title, observed, expected in incomplete_replay_evidence)
        warns.append(
            "the manifest-bound request journal does not cover the complete "
            f"reported replay population: {detail}. Outcome and HTTP 429 "
            "evidence is incomplete")

    # cache parity. one endpoint reporting no cache at all is the common case
    # when putting Databricks next to a provider that does not report cached
    # tokens, and it is the most misleading comparison the tool can produce,
    # so it has to be louder than a missing cell in a table.
    def _cache_cell(s, q):
        """A missing cache value means the endpoint never reported the field.
        A dash reads like a formatting gap, so say what it actually is."""
        acf = s.get("achieved_cache_fraction") or {}
        v = acf.get(q)
        return "NOT REPORTED" if v is None else f"{v:.3f}"

    caches = [(s.get("achieved_cache_fraction") or {}).get("p50") for s in summ]
    missing = [t for t, c in zip(titles, caches) if c is None]
    have = [c for c in caches if c is not None]
    # a missing value means the endpoint did not report the field, NOT that it
    # had zero cached prompt tokens. a reported zero comes through as 0.0.
    if missing and have:
        warns.append(
            f"{', '.join(missing)} did not report cached tokens, so its cache "
            f"usage is unknown, while another run measured a cache p50 of "
            f"{max(have):.3f}. Serving cached prompt tokens is far cheaper "
            "than serving cold ones, so unless you can establish the unknown side "
            "independently these latency columns may not be measuring the "
            "same work. Do not present this as a like-for-like result.")
    elif missing and not have:
        warns.append(
            "no run reported cached tokens, so cache usage is unknown for "
            "every column. Cached prompt-token fraction is usually a major "
            "biggest driver of the latency you are about to compare. Confirm "
            "how each endpoint handles caching before quoting these numbers.")
    if len(have) >= 2 and (max(have) - min(have)) > 0.10:
        warns.append(
            f"cached prompt-token fraction p50 spans {min(have):.3f} to "
            f"{max(have):.3f}, a gap over 0.10. Comparing latency at different "
            "cached-token fractions is not fair. Match them before quoting "
            "numbers.")
    cache_p95 = [
        (s.get("achieved_cache_fraction") or {}).get("p95") for s in summ]
    cache_p95_have = [value for value in cache_p95
                      if isinstance(value, (int, float))
                      and not isinstance(value, bool)]
    if len(cache_p95_have) >= 2 \
            and max(cache_p95_have) - min(cache_p95_have) > 0.10:
        warns.append(
            "cached prompt-token fraction p95 spans "
            f"{min(cache_p95_have):.3f} to {max(cache_p95_have):.3f}, a gap "
            "over 0.10. Tail latency is not like-for-like until that cache "
            "shape is matched.")

    # Identical intended profiles do not guarantee that different endpoint
    # tokenizers or early-stop behavior produced identical work. Compare the
    # endpoint-reported achieved/input and output ratios, not only the profile
    # hash. Missing achieved evidence is itself a qualification.
    for side, label in (("input", "input-token"), ("output", "output-token")):
        for quantile in ("p50", "p95"):
            field = f"{side}_reported_over_intended"
            values = [
                ((summary.get("token_targeting") or {}).get(field) or {}).get(
                    quantile)
                for summary in summ
            ]
            missing_titles = [
                title for title, value in zip(titles, values)
                if isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ]
            have_values = [float(value) for value in values
                           if isinstance(value, (int, float))
                           and not isinstance(value, bool)
                           and math.isfinite(float(value))]
            if missing_titles:
                warns.append(
                    f"achieved {label} shape {quantile} is not reported for "
                    f"{', '.join(missing_titles)}. Matching intended workload "
                    "identity alone does not prove equal realized token work.")
            elif len(have_values) >= 2 \
                    and max(have_values) - min(have_values) > 0.10:
                warns.append(
                    f"achieved {label} reported/intended {quantile} spans "
                    f"{min(have_values):.3f} to {max(have_values):.3f}, a gap "
                    "over 0.10. Match realized token shape before quoting "
                    "relative latency.")

    weak_answer_coverage = []
    for title, summary in zip(titles, summ):
        answers = summary.get("answers")
        answers = answers if isinstance(answers, dict) else {}
        rate = answers.get("answer_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) \
                and math.isfinite(float(rate)) and float(rate) < 0.99:
            weak_answer_coverage.append((title, float(rate)))
    if weak_answer_coverage:
        detail = ", ".join(
            f"{title} at {rate:.1%}" for title, rate in weak_answer_coverage)
        warns.append(
            f"acceptable-answer coverage is below 99%: {detail}. Latency "
            "percentiles exclude unacceptable outcomes and are subject to "
            "survivorship bias.")

    # error rates. percentiles over a run that dropped requests carry
    # survivorship bias, and the failures are often the slow ones.
    bad = [(t, s.get("error_rate") or 0.0) for t, s in zip(titles, summ)
           if (s.get("error_rate") or 0.0) > 0.01]
    if bad:
        detail = ", ".join(f"{t} at {r * 100:.1f} percent" for t, r in bad)
        warns.append(
            f"these runs failed requests: {detail}. Latency percentiles only "
            "cover requests that succeeded, so a run that dropped its slowest "
            "requests can look faster than one that served them. Read the "
            "error rate next to every latency number below.")

    # sample size. a tail number needs requests behind it.
    thin = [(t, (s.get("sample") or {}).get("n"))
            for t, s in zip(titles, summ)
            if (s.get("sample") or {}).get("warning")]
    if thin:
        detail = ", ".join(
            f"{t} ({markdown_plain_text(n)} requests)" for t, n in thin)
        warns.append(
            f"small samples: {detail}. p99 is indicative below 1000 "
            "requests. Run longer before quoting a tail.")

    # stability. a run still warming up is not a steady-state number.
    moving = [(t, (s.get("drift") or {}).get("drift_kind"))
              for t, s in zip(titles, summ)
              if (s.get("drift") or {}).get("drift_flag")]
    if moving:
        detail = ", ".join(
            f"{t} ({markdown_plain_text(k)})" for t, k in moving)
        broke = [t for t, k in moving if k == "failing"]
        one = len(broke) == 1
        extra = (f" {', '.join(broke)} {'was' if one else 'were'} shedding "
                 f"requests, which {'is a breaking point' if one else 'are breaking points'} "
                 f"rather than {'a latency result' if one else 'latency results'}, "
                 f"so {'its' if one else 'their'} "
                 "surviving percentiles are not comparable to anything."
                 if broke else "")
        warns.append(
            f"these runs were not in steady state: {detail}. Read each run's "
            "stability card. A warming endpoint compared against a warm one "
            "is a measurement artifact, not a difference between "
            f"providers.{extra}")
    # no verdict at all is not the same as passing. a run too short to bucket,
    # or whose windows were too thin to count, was never checked.
    unjudged = [t for t, s in zip(titles, summ)
                if (s.get("drift") or {}).get("drift_kind") is None]
    if unjudged:
        why = [
            (t, markdown_plain_text(
                (s.get("drift") or {}).get("note") or "no stability data"))
            for t, s in zip(titles, summ)
            if (s.get("drift") or {}).get("drift_kind") is None
        ]
        detail = " ".join(f"{t}: {w}" for t, w in why)
        warns.append(
            f"stability was never established for {', '.join(unjudged)}, so "
            "these columns were not checked for warmup or degradation. "
            f"Reported reason per run. {detail}")

    comparison_state = (
        "invalid" if compatibility_issues else
        "qualified" if warns else "valid")
    if warns:
        if comparison_state == "qualified":
            L.extend([
                "## QUALIFIED COMPARISON: diagnostic-only",
                "",
                "Compatibility checks passed, but the measurement warnings "
                "below block relative performance claims, candidate ranking, "
                "and directional judgment. Resolve every warning and repeat "
                "the runs before quoting a winner or latency delta.",
                "",
            ])
        L.append("## Read this before the tables")
        L.append("")
        for w in warns:
            L.append(f"> WARNING: {w}")
            L.append("")
    elif not compatibility_issues:
        L += ["Comparability checks (harness version, cache reporting and "
              "parity, production transport parity, error rate, sample size, "
              "steady state) all passed on these runs.", ""]

    def pct(name, key):
        L.extend([f"## {name}", hdr, sep])
        for q in ("p50", "p90", "p95", "p99"):
            cells = [_cell((s.get(key) or {}).get(q)) for s in summ]
            L.append(f"| {q} | " + " | ".join(cells) + " |")
        L.append("")

    if _comparison_ttft_definition(summ) == "first_visible":
        pct("TTFV: first visible content (ms)", "ttfv_ms")
        pct("TTFT: reasoning/visible/refusal onset, diagnostic (ms)",
            "ttft_ms")
    else:
        pct("TTFT (ms)", "ttft_ms")
    pct("TTFG / E2E (ms)", "e2e_ms")
    pct("interchunk max (ms)", "interchunk_max_ms")

    def scalar(label, fn, fmt="{:.0f}"):
        return f"| {label} | " + " | ".join(_cell(fn(s), fmt)
                                             for s in summ) + " |"

    def _reported_reasoning_tokens(s):
        source = str(s.get("reasoning_tokens_source") or "").lower()
        return (None if "stream-counted" in source
                else s.get("reasoning_tokens_total"))

    def _reasoning_deltas(s):
        if s.get("reasoning_stream_deltas_total") is not None:
            return s["reasoning_stream_deltas_total"]
        source = str(s.get("reasoning_tokens_source") or "").lower()
        return (s.get("reasoning_tokens_total")
                if "stream-counted" in source else None)

    L.extend(["## rates and throughput", hdr, sep,
              scalar("error rate", lambda s: s.get("error_rate"), "{:.4f}"),
              "| cached prompt-token fraction p50 | " + " | ".join(
                  _cache_cell(s, "p50") for s in summ) + " |",
              scalar("input tokens/min",
                     lambda s: (s.get("throughput") or {}).get("input_tokens_per_min"),
                     "{:,.0f}"),
              scalar("output tokens/min",
                     lambda s: (s.get("throughput") or {}).get("output_tokens_per_min"),
                     "{:,.0f}"),
              scalar("endpoint-reported reasoning tokens (total)",
                     _reported_reasoning_tokens,
                     "{:,.0f}"),
              scalar("reasoning stream deltas (total; not tokens)",
                     _reasoning_deltas, "{:,.0f}"),
              scalar("DBU per 1k requests",
                     lambda s: (s.get("cost") or {}).get("dbu_per_1k_requests"),
                     "{:,.2f}"), ""])

    L.extend(["## believability (read before trusting the latency tables)",
              hdr, sep,
              "| cached prompt-token fraction p50 | " + " | ".join(
                  _cache_cell(s, "p50") for s in summ) + " |",
              "| cached prompt-token fraction p95 | " + " | ".join(
                  _cache_cell(s, "p95") for s in summ) + " |",
              scalar("dispatch lag p95 (ms)",
                     lambda s: ((s.get("arrivals") or {}).get("dispatch_lag_ms")
                                or {}).get("p95")),
              scalar("HTTP request-start lateness p95 (ms)",
                     lambda s: (((s.get("arrivals") or {}).get(
                         "http_request_start_lateness_ms")
                         or (s.get("arrivals") or {}).get("wire_lateness_ms")
                         or {}).get("p95"))), ""])

    comparison_text = "\n".join(L) + "\n"
    sources = [
        _comparison_source_reference(position, d, manifest)
        for position, (d, manifest) in enumerate(zip(dirs, manifests))
    ]
    created_at = time.time()
    artifact_id = f"comparison-{uuid.uuid4().hex}"
    requested = Path(out_dir)
    out, dir_fd = _claim_compare_dir(
        requested, artifact_id, created_at)
    try:
        comparison_metadata = _atomic_compare_text(
            dir_fd, "comparison.md", comparison_text)
        comparison_html = _render_comparison_html(
            out, dirs, summ, manifests, request_evidence, raw_titles,
            compatibility_issues, warns, artifact_id)
        comparison_html_metadata = _atomic_compare_text(
            dir_fd, "comparison.html", comparison_html)
        manifest = {
            "manifest_schema_version": 3,
            "artifact_type": "comparison",
            "artifact_id": artifact_id,
            "artifact_created_at_utc": datetime.fromtimestamp(
                created_at, timezone.utc).isoformat(),
            "artifact_created_at_unix": created_at,
            "operation": "compare",
            "harness_version": __version__,
            "git_commit": source_state.get("git_commit"),
            "git_dirty": source_state.get("git_dirty"),
            "source": source_state,
            "source_tree_sha256": source_state.get("source_tree_sha256"),
            "generator_source_reconstructible":
                generator_source_reconstructible,
            "input_count": len(sources),
            "sources": sources,
            "comparison_state": comparison_state,
            "comparison_valid": comparison_state == "valid",
            "numeric_direction_labels_allowed": comparison_state == "valid",
            "directional_judgment_allowed": False,
            "performance_judgment_basis": "not configured; no repeat-run uncertainty or practical-effect threshold",
            "compatibility_issue_count": len(compatibility_issues),
            "warning_count": len(warns),
            "artifacts": {
                "comparison.md": comparison_metadata,
                "comparison.html": comparison_html_metadata,
            },
        }
        manifest_text = strict_json_dumps(manifest, indent=2) + "\n"
        manifest_metadata = _atomic_compare_text(
            dir_fd, "manifest.json", manifest_text)
        completed_at = time.time()
        completion_text = strict_json_dumps({
            "artifact_id": artifact_id,
            "artifact_type": "comparison",
            "status": "complete",
            "completed_at_unix": completed_at,
            "manifest_sha256": manifest_metadata["sha256"],
            "manifest_bytes": manifest_metadata["bytes"],
        }) + "\n"
        _atomic_compare_text(dir_fd, _WRITING_MARKER, completion_text)
        os.replace(_WRITING_MARKER, _COMPLETE_MARKER,
                   src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        _fsync_fd(dir_fd)
    finally:
        os.close(dir_fd)
    _fsync_directory(out.parent)
    verify_comparison_output(out)
    return out
