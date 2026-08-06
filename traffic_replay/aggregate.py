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
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import time
import uuid

from .metrics import _pct_table, summarize, write_outputs


_WRITING_MARKER = ".traffic-replay-writing"
_COMPLETE_MARKER = ".traffic-replay-complete"
_SUPPORTED_MANIFEST_SCHEMAS = {3}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_SHARD_RE = re.compile(r"([1-9][0-9]*)/([1-9][0-9]*)")


def _read_regular_bytes(path: Path) -> bytes:
    """Read one artifact without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _measure_regular(path: Path) -> tuple[str, int, int]:
    """Return SHA-256, byte count and newline count with bounded memory."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        rows = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
        return digest.hexdigest(), size, rows
    finally:
        os.close(fd)


def _load_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(_read_regular_bytes(path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid {label} in {path}: {exc}") from exc
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
    check("TTFT definition", lambda s, m: (
        (((m.get("config_identity") or {}).get("sla_definition") or {}).get(
            "ttft_definition"))
        or (s.get("sla") or {}).get("ttft_definition")), required=False)
    if merging:
        check("endpoint identity", lambda _s, m: ({
            "base_url": m.get("endpoint_base_url"),
            "model": m.get("endpoint_model"),
            "path": m.get("endpoint_path"),
        } if any((m.get("endpoint_base_url"), m.get("endpoint_model"),
                  m.get("endpoint_path"))) else None))
    return issues


def _run_title(d: Path, summ: dict) -> str:
    return (summ.get("run") or {}).get("title") or d.name


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
        actual, actual_bytes, actual_rows = _measure_regular(path)
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
    if index.get("partition") != expected_partition:
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
    required_artifacts = (("summary.json", "requests.jsonl")
                          if need == "requests.jsonl" else ("summary.json",))
    _verify_artifacts(d, manifest, required_artifacts)
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


def _replay_rows(d: Path) -> list[dict]:
    rows = []
    path = d / "requests.jsonl"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot read regular artifact {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1                 # fdopen owns it from here
            try:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(
                            f"blank JSONL record in {path} line {line_no}")
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON in {path} line {line_no}: "
                            f"{exc}") from exc
                    if not isinstance(r, dict):
                        raise ValueError(
                            f"requests.jsonl line {line_no} is not an object "
                            f"in {d}")
                    if r.get("phase") == "replay":
                        rows.append(r)
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"requests.jsonl is not UTF-8 in {d}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    return rows


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
    """Concatenate replay rows from each run dir and re-summarize the union."""
    dirs, manifests = _validated_input_dirs(
        input_dirs, "requests.jsonl", "merge")
    summaries = [_load_summary(d) for d in dirs]
    rows_by_dir = [_replay_rows(d) for d in dirs]
    coverage_issues = _merge_integrity(dirs, manifests, rows_by_dir)
    compatibility_issues = _compatibility_issues(
        dirs, summaries, manifests, merging=True)
    compatibility_issues.extend(coverage_issues)
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
    source_provenance = []
    for d, manifest in zip(dirs, manifests):
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
    expected_total = _declared_total_requests(manifests[0], dirs[0])
    merged_indices = [row["global_index"] for row in rows]
    index_identity = {
        "encoding": "int64-le",
        "global_indices_sha256": _packed_sha256(merged_indices, "int64-le"),
        "count": len(merged_indices),
        "min": merged_indices[0] if merged_indices else None,
        "max": merged_indices[-1] if merged_indices else None,
        "global_count": expected_total,
        "shard_index": 0,
        "shard_total": 1,
        "partition": "unsharded",
    }
    schedule_identities = {
        _stable(_global_schedule_identity(manifest))
        for manifest in manifests if manifest.get("schedule_identity") is not None
    }
    source_schedule_identity = manifests[0].get("schedule_identity") or {}
    merged_timestamps = [float(row["scheduled_s"]) for row in rows]
    merged_schedule_identity = {
        "encoding": source_schedule_identity.get("encoding"),
        "global_timestamps_sha256": source_schedule_identity.get(
            "global_timestamps_sha256"),
        "global_count": source_schedule_identity.get("global_count"),
        "global_min_s": source_schedule_identity.get("global_min_s"),
        "global_max_s": source_schedule_identity.get("global_max_s"),
        "shard_timestamps_sha256": _packed_sha256(
            merged_timestamps, "float64-le"),
        "shard_count": len(merged_timestamps),
        "shard_min_s": min(merged_timestamps) if merged_timestamps else None,
        "shard_max_s": max(merged_timestamps) if merged_timestamps else None,
    }
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
        "index_identity": index_identity,
        **({"schedule_identity": merged_schedule_identity}
           if len(schedule_identities) == 1 else {}),
        "aggregation_valid": not compatibility_issues,
        "compatibility_issues": compatibility_issues,
        "aggregation": {
            "kind": "merge",
            "forced": bool(force),
            "sources": source_provenance,
        },
        **({"prompts_count": counts.pop()}
           if input_mode == "prompts" and len(counts) == 1
           and None not in counts else {}),
        "merge_note": (f"pooled from {len(dirs)} run dirs. throughput is over "
                       "the union wall-clock window, so it is the aggregate "
                       "rate only when the shards ran concurrently."),
    }
    # cost is a per-run figure (rates can differ across pooled runs), so
    # it is not recomputed here; read each run report for its own cost.
    summary = summarize(rows, run_meta=meta, acceptance=acceptance)
    # drift buckets on absolute send time from the pooled minimum. shards that
    # ran at different times produce windows spanning the gap between them, so
    # a trend across pooled rows would describe the schedule, not the endpoint.
    # same hazard as drift below: shards start at different wall-clock times,
    # so a single schedule-vs-send offset across pooled rows reads the gap
    # between shards as lateness.
    summary["arrivals"]["wire_lateness_ms"] = _pct_table([])
    summary["arrivals"]["wire_lateness_note"] = (
        "wire lateness is not computed for a merged run, because pooled rows "
        "come from separate runs and the offset between them would read as "
        "lateness. read each run's own report. dispatch lag below is pooled "
        "and still meaningful, since it is measured within each run.")
    summary.pop("client", None)
    # summarize() stamps queue_wait_ms on each row against one schedule
    # offset. across runs that started at different times that number is
    # meaningless, and leaving it on the rows would contradict the note
    # below in the same output directory.
    for _r in rows:
        _r.pop("queue_wait_ms", None)
    # corrected latency is computed against one schedule offset. pooling rows
    # from runs that started at different wall-clock times makes that offset
    # meaningless: two 200 ms runs an hour apart would report a corrected p95
    # of an hour. same reason wire lateness is blanked.
    for k in ("ttft_corrected_ms", "e2e_corrected_ms",
              "latency_correction_note"):
        summary.pop(k, None)
    summary["latency_correction_note"] = (
        "caller-experienced latency is not computed for a merged run, "
        "because it measures against each run's own schedule and pooled "
        "rows come from different ones. read each run's own report.")
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
        },
        "inputs": ({input_key: {"sha256": input_hash}}
                   if input_hash else {}),
        "schedule_identity": merged_schedule_identity,
        "index_identity": index_identity,
    }
    return write_outputs(
        rows, summary, out_dir, title or f"merged: {len(dirs)} runs",
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
        raise
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


def _claim_compare_dir(requested: Path) -> tuple[Path, int]:
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
            os.close(marker_fd)
            _fsync_fd(dir_fd)
            _fsync_directory(candidate.parent)
            return candidate, dir_fd
        except Exception:
            if "dir_fd" in locals():
                os.close(dir_fd)
            raise
    raise RuntimeError(f"could not claim a unique comparison directory: {requested}")


def _atomic_compare_text(dir_fd: int, name: str, value: str) -> None:
    tmp = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o644, dir_fd=dir_fd)
    try:
        raw = value.encode("utf-8")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise OSError(f"short write while creating {name}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    _fsync_fd(dir_fd)


def compare_runs(out_dir, input_dirs) -> Path:
    """Tabulate several runs one column each, on identical measurement, and
    invalidate the comparison when their per-request cached prompt-token
    fractions or provenance diverge enough to make latency incomparable."""
    dirs, manifests = _validated_input_dirs(
        input_dirs, "summary.json", "compare")
    summ = [_load_summary(d) for d in dirs]
    titles = [_run_title(d, s) for d, s in zip(dirs, summ)]
    n = len(titles)
    hdr = "| metric / quantile | " + " | ".join(titles) + " |"
    sep = "|---" * (n + 1) + "|"
    L = ["# endpoint comparison", "",
         "Runs measured on the same instrument. Read the warnings and the "
         "believability section before trusting the latency tables.", ""]

    compatibility_issues = _compatibility_issues(
        dirs, summ, manifests, merging=False)
    if compatibility_issues:
        L += ["## INVALID COMPARISON — inputs are not proven like-for-like", "",
              "The tables below are retained for diagnosis only. Do not quote "
              "a winner or a relative latency until every incompatibility is "
              "resolved and the runs are repeated.", ""]
        for issue in compatibility_issues:
            L += [f"> INVALID: {issue}", ""]

    # Everything that can make a side-by-side dishonest goes ABOVE the tables.
    # A reader who stops after the first screen still sees the disqualifiers.
    warns: list[str] = []

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
        detail = ", ".join(f"{t} ({n} requests)" for t, n in thin)
        warns.append(
            f"small samples: {detail}. p99 is unstable below about 100 "
            "requests. Run longer before quoting a tail.")

    # stability. a run still warming up is not a steady-state number.
    moving = [(t, (s.get("drift") or {}).get("drift_kind"))
              for t, s in zip(titles, summ)
              if (s.get("drift") or {}).get("drift_flag")]
    if moving:
        detail = ", ".join(f"{t} ({k})" for t, k in moving)
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
        why = {t: ((s.get("drift") or {}).get("note") or "no stability data")
               for t, s in zip(titles, summ)
               if (s.get("drift") or {}).get("drift_kind") is None}
        detail = " ".join(f"{t}: {w}" for t, w in why.items())
        warns.append(
            f"stability was never established for {', '.join(unjudged)}, so "
            "these columns were not checked for warmup or degradation. "
            f"Reported reason per run. {detail}")

    if warns:
        L.append("## Read this before the tables")
        L.append("")
        for w in warns:
            L.append(f"> WARNING: {w}")
            L.append("")
    elif not compatibility_issues:
        L += ["Comparability checks (harness version, cache reporting and "
              "parity, error rate, sample size, steady state) all passed on "
              "these runs.", ""]

    def pct(name, key):
        L.extend([f"## {name}", hdr, sep])
        for q in ("p50", "p90", "p95", "p99"):
            cells = [_cell((s.get(key) or {}).get(q)) for s in summ]
            L.append(f"| {q} | " + " | ".join(cells) + " |")
        L.append("")

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
              scalar("wire lateness p95 (ms)",
                     lambda s: ((s.get("arrivals") or {}).get("wire_lateness_ms")
                                or {}).get("p95")), ""])

    requested = Path(out_dir)
    out, dir_fd = _claim_compare_dir(requested)
    try:
        _atomic_compare_text(dir_fd, "comparison.md", "\n".join(L) + "\n")
        os.replace(_WRITING_MARKER, _COMPLETE_MARKER,
                   src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        _fsync_fd(dir_fd)
    finally:
        os.close(dir_fd)
    _fsync_directory(out.parent)
    return out
