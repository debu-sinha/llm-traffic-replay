"""Race-safe immutable persistence for generated profiles and run configs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import uuid

from .artifacts import _fsync_dir_fd, _write_all, strict_json_dumps


class ImmutableConfigError(RuntimeError):
    """A generated configuration path cannot satisfy the integrity contract."""


def _open_safe_dir(path: Path) -> int:
    """Create one directory if needed, then open it without following links."""
    created = False
    try:
        path.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    try:
        info = path.lstat()
    except OSError as exc:
        raise ImmutableConfigError(
            f"cannot inspect generated-config directory {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ImmutableConfigError(
            f"generated-config path is not a regular directory: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ImmutableConfigError(
            f"cannot open generated-config directory safely {path}: {exc}") from exc
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise ImmutableConfigError(
            f"generated-config path is not a directory: {path}")
    if created:
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            parent_fd = os.open(path.parent, parent_flags)
            try:
                _fsync_dir_fd(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError as exc:
            os.close(fd)
            raise ImmutableConfigError(
                f"cannot make generated-config directory durable {path}: "
                f"{exc}") from exc
    return fd


def _ensure_safe_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = _open_safe_dir(path)
    try:
        _fsync_dir_fd(fd)
    finally:
        os.close(fd)


def _read_regular_at(dir_fd: int, name: str, path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise ImmutableConfigError(
            f"cannot read generated config safely {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ImmutableConfigError(
                f"generated config is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(fd)


def _publish_once(directory: Path, name: str, raw: bytes, *,
                  immutable: bool) -> bool:
    """Publish bytes without replacement; return whether an existing file agrees."""
    dir_fd = _open_safe_dir(directory)
    temp = f".{name}.{uuid.uuid4().hex}.tmp"
    fd = -1
    matches = True
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
            | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp, flags, 0o600, dir_fd=dir_fd)
        _write_all(fd, raw)
        if immutable:
            os.fchmod(fd, 0o400)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd,
                    follow_symlinks=False)
        except FileExistsError:
            existing, info = _read_regular_at(
                dir_fd, name, directory / name)
            if immutable and info.st_mode & 0o222:
                raise ImmutableConfigError(
                    f"immutable generated config is writable: {directory / name}")
            if existing != raw:
                if immutable:
                    raise ImmutableConfigError(
                        f"content-addressed generated config disagrees with its "
                        f"digest path: {directory / name}")
                matches = False
        finally:
            try:
                os.unlink(temp, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
        _fsync_dir_fd(dir_fd)
        return matches
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(dir_fd)


def _store_root(out_dir: str | Path) -> Path:
    requested = Path(out_dir)
    return requested.parent / ".traffic-replay-configs"


def write_immutable_json(out_dir: str | Path, kind: str, value) -> Path:
    """Write canonical JSON below a content-addressed, read-only path."""
    if kind not in {"profile", "run-config"}:
        raise ValueError(f"unsupported generated config kind: {kind!r}")
    raw = (strict_json_dumps(value, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    root = _store_root(out_dir)
    section = root / ("profiles" if kind == "profile" else "runs")
    leaf = section / digest
    for directory in (root, section, leaf):
        _ensure_safe_dir(directory)
    name = f"{kind}.json"
    _publish_once(leaf, name, raw, immutable=True)
    return (leaf / name).resolve(strict=True)


def write_immutable_text(out_dir: str | Path, kind: str, text: str) -> Path:
    """Write a content-addressed text input used by a generated run config."""
    if kind != "timestamps":
        raise ValueError(f"unsupported generated text kind: {kind!r}")
    if not isinstance(text, str) or not text:
        raise ValueError("generated timestamps text must be non-empty")
    raw = text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    root = _store_root(out_dir)
    section = root / "timestamps"
    leaf = section / digest
    for directory in (root, section, leaf):
        _ensure_safe_dir(directory)
    name = "timestamps.txt"
    _publish_once(leaf, name, raw, immutable=True)
    return (leaf / name).resolve(strict=True)


def publish_legacy_copy(source: str | Path, destination: str | Path) -> bool:
    """Create an old well-known filename once, without ever replacing it.

    ``False`` means a different regular file already occupies the legacy name.
    The caller must continue to advertise the immutable source path in that
    case. Symlinks and non-regular destinations fail closed.
    """
    src = Path(source)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(src, flags)
    except OSError as exc:
        raise ImmutableConfigError(
            f"cannot read immutable generated config {src}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ImmutableConfigError(
                f"immutable generated config is not a regular file: {src}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    dest = Path(destination)
    _ensure_safe_dir(dest.parent)
    return _publish_once(dest.parent, dest.name, raw, immutable=False)
