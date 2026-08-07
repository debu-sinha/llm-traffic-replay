"""Crash-safe lifecycle for benchmark evidence.

The load generator must reserve and validate its destination before it sends a
request.  During the run each completed request is appended to a durable JSONL
journal.  Final reports are written by same-directory atomic replacement and a
completion marker is promoted only after the manifest has bound every artifact.

An interrupted directory intentionally remains useful: it keeps
``.traffic-replay-writing``, ``start.json`` and ``requests.jsonl.partial``.
Readers may recover every newline-terminated JSON object and ignore at most one
truncated final record.  They must never mistake that directory for a completed
run.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Iterator


WRITING_MARKER = ".traffic-replay-writing"
COMPLETE_MARKER = ".traffic-replay-complete"
PARTIAL_REQUESTS = "requests.jsonl.partial"
FINAL_REQUESTS = "requests.jsonl"


class ArtifactError(RuntimeError):
    """The requested artifact destination cannot be used safely."""


_SECRET_EXACT = {
    "authorization", "proxyauthorization", "apikey", "xapikey",
    "accesskey", "secretkey", "clientsecret", "password", "passwd",
    "credential", "credentials", "cookie", "setcookie", "token",
    "accesstoken", "authtoken", "bearertoken", "refreshtoken", "idtoken",
    "jwt", "assertion", "clientassertion", "signature", "sig", "sas",
    "sastoken", "sharedaccesssignature", "privatekey", "privatekeydata",
    "authprofile",
}
_SECRET_SUFFIXES = (
    "apikey", "accesskey", "secretkey", "clientsecret", "password",
    "credential", "credentials", "accesstoken", "authtoken",
    "bearertoken", "refreshtoken", "idtoken", "clientassertion",
    "privatekey", "sharedaccesssignature", "signature", "sastoken",
)
_NON_SECRET_TOKEN_KEYS = {
    # Model/request controls and usage counters. Keep this allowlist explicit:
    # an unknown singular/plural token key is safer to treat as a credential.
    "mintokens", "maxtokens", "maxnewtokens", "maxinputtokens",
    "maxoutputtokens", "maxcompletiontokens", "budgettokens",
    "inputtokens", "outputtokens", "prompttokens", "completiontokens",
    "cachedtokens", "reasoningtokens", "totaltokens", "numtokens",
    "tokencount", "tokencounts", "tokenlimit", "tokenbudget", "tokenids",
}
_HEADER_KEYS = {"header", "headers", "httpheader", "httpheaders",
                "requestheader", "requestheaders"}
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_VALUE = re.compile(
    r"(?i)\bbasic\s+([A-Za-z0-9+/]+={0,2})(?![A-Za-z0-9+/=])")
_TOKEN_VALUE = re.compile(
    r"\b(?:dapi[A-Za-z0-9._-]{8,}|sk-[A-Za-z0-9._-]{8,}|"
    r"ghp_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|AKIA[A-Z0-9]{12,})\b")
_JWT_VALUE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\b")
_HEADER_VALUE = re.compile(
    r"(?i)\b(authorization|proxy-authorization|x-api-key|api-key)"
    r"\s*:\s*[^\r\n,;]+")
_INLINE_SECRET = re.compile(
    r"(?i)\b(access[_-]?token|api[_-]?key|client[_-]?assertion|jwt|"
    r"signature|sig|sas|password|secret)\s*[:=]\s*([^&\s,;]+)")
_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@",
                              re.IGNORECASE)
_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL)


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _secret_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized in _NON_SECRET_TOKEN_KEYS:
        return False
    plural_credential_tokens = normalized.endswith("tokens") and any(
        normalized[:-6].endswith(prefix) for prefix in (
            "api", "service", "auth", "access", "bearer", "refresh",
            "session", "oauth", "credential", "secret", "client"))
    return (normalized in _SECRET_EXACT
            or any(normalized.endswith(suffix)
                   for suffix in _SECRET_SUFFIXES)
            or normalized.endswith("token")
            or plural_credential_tokens)


def _header_container_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return (normalized in _HEADER_KEYS
            or normalized.endswith("header")
            or normalized.endswith("headers"))


def _redact_url(value: str) -> str:
    """Redact credentials and secret-valued query parameters in URLs/paths."""
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    absolute = parsed.scheme.lower() in {"http", "https"} and parsed.netloc
    relative = not parsed.scheme and not parsed.netloc and bool(parsed.query)
    if not absolute and not relative:
        return value

    query = []
    changed = False
    for key, item in urllib.parse.parse_qsl(
            parsed.query, keep_blank_values=True, strict_parsing=False):
        secret = _secret_key(key)
        query.append((key, "<redacted>" if secret else item))
        changed = changed or secret
    if relative:
        if not changed:
            return value
        return urllib.parse.urlunsplit((
            "", "", parsed.path, urllib.parse.urlencode(query, doseq=True),
            parsed.fragment))

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    has_userinfo = parsed.username is not None or parsed.password is not None
    userinfo = "<redacted>@" if has_userinfo else ""
    changed = changed or has_userinfo
    if not changed:
        return value
    netloc = f"{userinfo}{host}{port}"
    return urllib.parse.urlunsplit((
        parsed.scheme, netloc, parsed.path,
        urllib.parse.urlencode(query, doseq=True), parsed.fragment))


def _redact_string(value: str, *, header_context: bool = False) -> str:
    if header_context:
        return "<redacted>" if value else value
    value = _redact_url(value)
    value = _PEM_PRIVATE_KEY.sub("<redacted>", value)
    value = _URL_CREDENTIALS.sub(r"\1<redacted>@", value)
    value = _HEADER_VALUE.sub(lambda m: f"{m.group(1)}: <redacted>", value)
    value = _INLINE_SECRET.sub(lambda m: f"{m.group(1)}=<redacted>", value)
    value = _JWT_VALUE.sub("<redacted>", value)
    value = _TOKEN_VALUE.sub("<redacted>", value)
    value = _BEARER_VALUE.sub("<redacted>", value)
    def redact_basic(match: re.Match) -> str:
        encoded = match.group(1)
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            decoded = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            return match.group(0)
        return "<redacted>" if b":" in decoded else match.group(0)

    value = _BASIC_VALUE.sub(redact_basic, value)
    return value


def redact_secrets(value, key: str | None = None, *, header_context=False):
    """Return a JSON-safe copy with credentials removed.

    Matching is semantic rather than a broad ``"token" in key`` test.  Model
    controls such as ``min_tokens``, ``max_tokens`` and ``token_limit`` are
    behavioral configuration and must remain visible and comparable.
    """
    if key is not None and _secret_key(key):
        return "<redacted>"
    child_header_context = header_context or (
        key is not None and _header_container_key(key))
    if header_context and not isinstance(value, (dict, list, tuple, str)):
        return "<redacted>" if value is not None else None
    if isinstance(value, dict):
        return {str(k): redact_secrets(v, str(k),
                                       header_context=child_header_context)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v, header_context=child_header_context)
                for v in value]
    if isinstance(value, str):
        return _redact_string(value, header_context=child_header_context)
    return value


def sanitize_title(value: object) -> str:
    """A one-line, credential-redacted report title."""
    safe = str(redact_secrets(str(value)))
    return re.sub(r"[\x00-\x1f\x7f]+", " ", safe).strip()[:500]


def strict_json_dumps(value, *, indent: int | None = None) -> str:
    """Standards-compliant JSON; NaN and infinities are configuration errors."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent,
                      separators=None if indent is not None else (",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value) -> str:
    raw = strict_json_dumps(value).encode("utf-8")
    return sha256_bytes(raw)


def _fsync_dir_fd(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems do not support directory fsync. That means they
        # cannot provide the durability contract this harness promises.
        raise ArtifactError(f"cannot fsync artifact directory: {exc}") from exc


def _write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise ArtifactError("short write while persisting benchmark evidence")
        view = view[written:]


def _regular_metadata(path: Path, *, row_count: int | None = None) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactError(f"artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        newline_count = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            newline_count += chunk.count(b"\n")
            digest.update(chunk)
    finally:
        os.close(fd)
    out = {"sha256": digest.hexdigest(), "bytes": size}
    if row_count is not None:
        if newline_count != row_count:
            raise ArtifactError(
                f"requests row count changed while finalizing: expected "
                f"{row_count}, found {newline_count}")
        out["row_count"] = row_count
    return out


def snapshot_source_state(package_dir: str | Path) -> dict:
    """Snapshot source bytes and Git identity before the output tree exists."""
    root = Path(package_dir).resolve()
    digest = hashlib.sha256()
    files = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8") + b"\0" + raw + b"\0")
        files.append({"path": rel, "sha256": sha256_bytes(raw), "bytes": len(raw)})

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, timeout=10)
        except Exception:
            return None
        return (result.stdout.decode("utf-8", "replace").strip()
                if result.returncode == 0 else None)

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "captured_at_unix": time.time(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status_sha256": (sha256_bytes(status.encode("utf-8"))
                               if status is not None else None),
        "source_tree_sha256": digest.hexdigest(),
        "source_files": files,
    }


class RunArtifacts:
    """Exclusive run directory plus an incrementally durable request journal."""

    def __init__(self, path: Path, dir_fd: int, partial_fd: int,
                 start_provenance: dict, *, sync_every_rows: int,
                 artifact_id: str):
        self.path = path
        self._dir_fd = dir_fd
        self._partial_fd = partial_fd
        self._start = redact_secrets(start_provenance)
        self.sync_every_rows = max(int(sync_every_rows), 1)
        self.artifact_id = artifact_id
        self.row_count = 0
        self._rows_since_sync = 0
        self._requests_finalized = False
        self._complete = False
        self._closed = False
        self._io_lock = threading.RLock()

    @classmethod
    def claim(cls, out_dir: str | Path, start_provenance: dict, *,
              sync_every_rows: int = 16,
              artifact_id: str | None = None) -> "RunArtifacts":
        requested = Path(out_dir)
        requested.parent.mkdir(parents=True, exist_ok=True)
        artifact_id = artifact_id or f"artifact-{uuid.uuid4().hex}"
        candidate = requested
        first = True
        while True:
            created = False
            try:
                candidate.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                info = candidate.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise ArtifactError(
                        f"refusing symlink artifact directory: {candidate}")
                if not stat.S_ISDIR(info.st_mode):
                    if first:
                        raise ArtifactError(
                            f"artifact path is not a directory: {candidate}")
                    candidate = requested.with_name(
                        f"{requested.name}-{uuid.uuid4().hex[:12]}")
                    first = False
                    continue
                try:
                    next(candidate.iterdir())
                except StopIteration:
                    pass                    # explicit caller-supplied empty dir
                else:
                    candidate = requested.with_name(
                        f"{requested.name}-{uuid.uuid4().hex[:12]}")
                    first = False
                    continue

            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
                | getattr(os, "O_NOFOLLOW", 0)
            try:
                dir_fd = os.open(candidate, flags)
            except OSError as exc:
                if created:
                    try:
                        candidate.rmdir()
                    except OSError:
                        pass
                raise ArtifactError(
                    f"cannot open artifact directory safely {candidate}: {exc}") from exc
            try:
                marker_fd = os.open(
                    WRITING_MARKER,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600, dir_fd=dir_fd)
            except FileExistsError:
                os.close(dir_fd)
                candidate = requested.with_name(
                    f"{requested.name}-{uuid.uuid4().hex[:12]}")
                first = False
                continue
            except OSError as exc:
                os.close(dir_fd)
                raise ArtifactError(
                    f"artifact directory is not writable {candidate}: {exc}") from exc
            try:
                try:
                    marker_value = strict_json_dumps({
                        "artifact_id": artifact_id,
                        "status": "writing",
                        "created_at_unix": time.time(),
                    }).encode("utf-8") + b"\n"
                    _write_all(marker_fd, marker_value)
                    os.fsync(marker_fd)
                finally:
                    os.close(marker_fd)
            except Exception:
                try:
                    os.unlink(WRITING_MARKER, dir_fd=dir_fd)
                except OSError:
                    pass
                os.close(dir_fd)
                raise
            try:
                partial_fd = -1
                partial_fd = os.open(
                    PARTIAL_REQUESTS,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600, dir_fd=dir_fd)
                obj = cls(candidate, dir_fd, partial_fd, start_provenance,
                          sync_every_rows=sync_every_rows,
                          artifact_id=artifact_id)
                obj._atomic_json("start.json", obj._start)
                os.fsync(partial_fd)
                _fsync_dir_fd(dir_fd)
                return obj
            except Exception:
                if partial_fd >= 0:
                    try:
                        os.close(partial_fd)
                    except OSError:
                        pass
                try:
                    os.unlink(PARTIAL_REQUESTS, dir_fd=dir_fd)
                except OSError:
                    pass
                try:
                    os.unlink("start.json", dir_fd=dir_fd)
                except OSError:
                    pass
                try:
                    os.unlink(WRITING_MARKER, dir_fd=dir_fd)
                except OSError:
                    pass
                os.close(dir_fd)
                raise

    def _atomic_bytes(self, name: str, value: bytes) -> None:
        if Path(name).name != name or name in {".", ".."}:
            raise ArtifactError(f"unsafe artifact name: {name!r}")
        tmp = f".{name}.{uuid.uuid4().hex}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0),
                     0o600, dir_fd=self._dir_fd)
        try:
            _write_all(fd, value)
            os.fsync(fd)
        except Exception:
            try:
                os.unlink(tmp, dir_fd=self._dir_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)
        try:
            os.replace(tmp, name, src_dir_fd=self._dir_fd,
                       dst_dir_fd=self._dir_fd)
        except Exception:
            try:
                os.unlink(tmp, dir_fd=self._dir_fd)
            except OSError:
                pass
            raise
        _fsync_dir_fd(self._dir_fd)

    def _atomic_json(self, name: str, value) -> None:
        raw = strict_json_dumps(redact_secrets(value), indent=2).encode("utf-8")
        self._atomic_bytes(name, raw + b"\n")

    def atomic_text(self, name: str, value: str) -> None:
        self._atomic_bytes(name, value.encode("utf-8"))

    def atomic_json(self, name: str, value) -> None:
        self._atomic_json(name, value)

    def update_start(self, **fields) -> None:
        self._start.update(redact_secrets(fields))
        self._atomic_json("start.json", self._start)

    @property
    def start_provenance(self) -> dict:
        return dict(self._start)

    @property
    def complete(self) -> bool:
        return self._complete

    def append(self, row: dict) -> None:
        with self._io_lock:
            if self._partial_fd < 0 or self._requests_finalized:
                raise ArtifactError("request journal is already finalized")
            raw = strict_json_dumps(redact_secrets(row)).encode("utf-8") + b"\n"
            _write_all(self._partial_fd, raw)
            self.row_count += 1
            self._rows_since_sync += 1
            if self._rows_since_sync >= self.sync_every_rows:
                os.fsync(self._partial_fd)
                self._rows_since_sync = 0

    def sync(self) -> None:
        with self._io_lock:
            if self._partial_fd >= 0:
                os.fsync(self._partial_fd)
                self._rows_since_sync = 0

    def abort(self, error: object | None = None) -> None:
        if self._complete or self._closed:
            return
        persistence_error = None
        try:
            self.sync()
            self._atomic_json("failure.json", {
                "status": "incomplete",
                "failed_at_unix": time.time(),
                "error": str(error) if error is not None else None,
                "durable_rows": self.row_count,
            })
        except Exception as exc:
            # Never mask the exception that aborted the benchmark. The writing
            # marker itself remains the durable incomplete-run signal when a
            # full disk also prevents failure.json from being written.
            persistence_error = exc
        finally:
            self.close()
        if error is None and persistence_error is not None:
            raise persistence_error

    def finalize_requests(self) -> None:
        with self._io_lock:
            if self._requests_finalized:
                return
            self.sync()
            os.close(self._partial_fd)
            self._partial_fd = -1
            os.replace(PARTIAL_REQUESTS, FINAL_REQUESTS,
                       src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
            _fsync_dir_fd(self._dir_fd)
            self._requests_finalized = True

    def read_rows(self, *, include_truncated_final=False) -> Iterator[dict]:
        """Read durable rows; an incomplete final fragment is recoverable."""
        self.sync()
        path = self.path / (FINAL_REQUESTS if self._requests_finalized
                            else PARTIAL_REQUESTS)
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.endswith(b"\n") and not include_truncated_final:
                    break
                try:
                    value = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not raw.endswith(b"\n"):
                        break
                    raise ArtifactError(
                        f"invalid durable JSON row {line_number} in {path}")
                if not isinstance(value, dict):
                    raise ArtifactError(
                        f"durable JSON row {line_number} is not an object")
                yield value

    def metadata(self, names: list[str]) -> dict[str, dict]:
        out = {}
        for name in names:
            rows = self.row_count if name == FINAL_REQUESTS else None
            out[name] = _regular_metadata(self.path / name, row_count=rows)
        return out

    def mark_complete(self) -> None:
        if not self._requests_finalized:
            raise ArtifactError("cannot complete a run before requests are finalized")
        manifest = _regular_metadata(self.path / "manifest.json")
        self._atomic_json(WRITING_MARKER, {
            "artifact_id": self.artifact_id,
            "status": "complete",
            "completed_at_unix": time.time(),
            "manifest_sha256": manifest["sha256"],
            "manifest_bytes": manifest["bytes"],
            "request_rows": self.row_count,
        })
        os.replace(WRITING_MARKER, COMPLETE_MARKER,
                   src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
        _fsync_dir_fd(self._dir_fd)
        self._complete = True
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self._partial_fd >= 0:
            os.close(self._partial_fd)
            self._partial_fd = -1
        if self._dir_fd >= 0:
            os.close(self._dir_fd)
            self._dir_fd = -1
        self._closed = True

    def __enter__(self) -> "RunArtifacts":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._complete:
            self.abort(exc)
        return False
