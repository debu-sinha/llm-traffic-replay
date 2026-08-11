"""Versioned, data-only model capability profiles.

This module deliberately contains no model-name conditionals.  A model is
known only when a reviewed profile binds an exact provider/model/route/API
tuple to an endpoint adapter and cites the evidence for that binding.  New
models using an existing wire adapter can therefore be added as data.

The SHA-256 digest is an integrity check, not a signature or an assertion that
the cited source is correct.  Callers must still decide which source kinds and
publishers they trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


CAPABILITY_SCHEMA_VERSION = "llm-capability-profile/v1"
MAX_PROFILE_BYTES = 1024 * 1024


class CapabilityProfileError(ValueError):
    """A capability profile violates the versioned data contract."""


class CapabilityIntegrityError(CapabilityProfileError):
    """A profile or its backing file no longer matches its integrity record."""


class CapabilityDuplicateError(CapabilityProfileError):
    """A catalog contains duplicate identities."""


class CapabilityAmbiguityError(CapabilityProfileError):
    """More than one active profile claims the same exact target."""


class CapabilityNotFoundError(LookupError):
    """No active profile describes the requested exact target."""


_VERSIONED_ID = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,95}/v[1-9][0-9]*$"
)
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_BEHAVIOR = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REQUEST_PATH_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$")
_ACCEPTANCE = {"accepted", "rejected", "transformed", "ignored", "unknown"}
_REASONING_MODES = {"configurable", "fixed", "unsupported", "unknown"}
_SOURCE_KINDS = {
    "customer_record",
    "owner_confirmation",
    "provider_documentation",
    "runtime_probe",
    "vendor_release_note",
}


def _reject_json_constant(value: str) -> None:
    raise CapabilityProfileError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapabilityProfileError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw: str | bytes) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except UnicodeDecodeError as exc:
        raise CapabilityProfileError("profile must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise CapabilityProfileError(f"invalid capability profile JSON: {exc}") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapabilityProfileError(
            "profile must contain only finite JSON-compatible values"
        ) from exc


def _json_clone(value: Any) -> Any:
    return _strict_json_loads(_canonical_json(value))


def compute_profile_digest(document: Mapping[str, Any]) -> str:
    """Return the canonical digest, excluding the top-level ``digest`` field."""
    if not isinstance(document, Mapping):
        raise CapabilityProfileError("profile must be an object")
    clone = _json_clone(document)
    if not isinstance(clone, dict):
        raise CapabilityProfileError("profile must be an object")
    clone.pop("digest", None)
    return "sha256:" + hashlib.sha256(_canonical_json(clone)).hexdigest()


def seal_capability_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    """Defensively copy an unsealed document and attach its canonical digest."""
    clone = _json_clone(document)
    if not isinstance(clone, dict):
        raise CapabilityProfileError("profile must be an object")
    if "digest" in clone:
        raise CapabilityProfileError("refusing to overwrite an existing digest")
    clone["digest"] = compute_profile_digest(clone)
    return clone


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapabilityProfileError(f"{where} must be an object")
    return value


def _require_list(value: Any, where: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        raise CapabilityProfileError(f"{where} must be an array")
    if nonempty and not value:
        raise CapabilityProfileError(f"{where} must not be empty")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise CapabilityProfileError(
            f"{where} is missing required field"
            f"{'s' if len(missing) != 1 else ''}: {', '.join(missing)}"
        )
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise CapabilityProfileError(
            f"{where} has unknown field"
            f"{'s' if len(unknown) != 1 else ''}: {', '.join(unknown)}"
        )


def _text(value: Any, where: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CapabilityProfileError(f"{where} must be a non-empty trimmed string")
    if len(value) > maximum or any(ord(char) < 32 for char in value):
        raise CapabilityProfileError(f"{where} is too long or contains control characters")
    return value


def _slug(value: Any, where: str) -> str:
    result = _text(value, where, maximum=96)
    if not _SLUG.fullmatch(result):
        raise CapabilityProfileError(f"{where} must be a lowercase identifier")
    return result


def _versioned_id(value: Any, where: str) -> str:
    result = _text(value, where, maximum=128)
    if not _VERSIONED_ID.fullmatch(result):
        raise CapabilityProfileError(
            f"{where} must be a lowercase versioned ID such as name/v1"
        )
    return result


def _iso_date(value: Any, where: str, *, future_ok: bool = False) -> date:
    text = _text(value, where, maximum=10)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise CapabilityProfileError(f"{where} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != text:
        raise CapabilityProfileError(f"{where} must be YYYY-MM-DD")
    if not future_ok and parsed > date.today():
        raise CapabilityProfileError(f"{where} cannot be in the future")
    return parsed


def _unique_text_list(
    value: Any,
    where: str,
    *,
    parser=_text,
) -> tuple[str, ...]:
    raw = _require_list(value, where)
    parsed = tuple(parser(item, f"{where}[{index}]") for index, item in enumerate(raw))
    if len(set(parsed)) != len(parsed):
        raise CapabilityProfileError(f"{where} must not contain duplicates")
    return parsed


def _source_refs(value: Any, where: str, source_ids: set[str]) -> tuple[str, ...]:
    refs = _unique_text_list(value, where, parser=_slug)
    missing = sorted(set(refs) - source_ids)
    if missing:
        raise CapabilityProfileError(
            f"{where} references unknown source{'s' if len(missing) != 1 else ''}: "
            + ", ".join(missing)
        )
    return refs


def _validate_source(source: Any, index: int) -> tuple[str, date]:
    where = f"sources[{index}]"
    item = _require_object(source, where)
    _exact_keys(
        item,
        {"source_id", "kind", "uri", "retrieved_at"},
        {"note"},
        where,
    )
    source_id = _slug(item["source_id"], f"{where}.source_id")
    kind = _slug(item["kind"], f"{where}.kind")
    if kind not in _SOURCE_KINDS:
        raise CapabilityProfileError(
            f"{where}.kind must be one of: {', '.join(sorted(_SOURCE_KINDS))}"
        )
    uri = _text(item["uri"], f"{where}.uri", maximum=2048)
    parsed_uri = urlsplit(uri)
    if parsed_uri.scheme == "https":
        if not parsed_uri.netloc or parsed_uri.username or parsed_uri.password:
            raise CapabilityProfileError(
                f"{where}.uri must be an https URL without embedded credentials"
            )
    elif parsed_uri.scheme == "urn":
        if not parsed_uri.path or any(char.isspace() for char in uri):
            raise CapabilityProfileError(f"{where}.uri is not a valid URN")
    else:
        raise CapabilityProfileError(f"{where}.uri must use https or urn")
    retrieved_at = _iso_date(item["retrieved_at"], f"{where}.retrieved_at")
    if "note" in item:
        _text(item["note"], f"{where}.note", maximum=2048)
    return source_id, retrieved_at


def _validate_value_contract(
    contract: Any,
    where: str,
    source_ids: set[str],
    *,
    require_value: bool,
) -> bytes | None:
    item = _require_object(contract, where)
    required = {"acceptance", "effective_behavior", "source_refs"}
    if require_value:
        required.add("value")
    _exact_keys(item, required, set(), where)
    value_key: bytes | None = None
    if require_value:
        candidate = item["value"]
        if isinstance(candidate, (dict, list)) or not (
            candidate is None or isinstance(candidate, (str, bool, int, float))
        ):
            raise CapabilityProfileError(f"{where}.value must be a JSON scalar")
        if isinstance(candidate, float) and not math.isfinite(candidate):
            raise CapabilityProfileError(f"{where}.value must be finite")
        value_key = _canonical_json(candidate)
    acceptance = _slug(item["acceptance"], f"{where}.acceptance")
    if acceptance not in _ACCEPTANCE:
        raise CapabilityProfileError(
            f"{where}.acceptance must be one of: {', '.join(sorted(_ACCEPTANCE))}"
        )
    behavior = _text(
        item["effective_behavior"], f"{where}.effective_behavior", maximum=128
    )
    if not _BEHAVIOR.fullmatch(behavior):
        raise CapabilityProfileError(
            f"{where}.effective_behavior must be a lowercase identifier"
        )
    _source_refs(item["source_refs"], f"{where}.source_refs", source_ids)
    return value_key


def _validate_control(
    control: Any,
    where: str,
    source_ids: set[str],
) -> tuple[str, ...]:
    item = _require_object(control, where)
    _exact_keys(
        item,
        {"request_path", "values", "omitted"},
        set(),
        where,
    )
    path_items = _require_list(item["request_path"], f"{where}.request_path")
    path: list[str] = []
    for index, part in enumerate(path_items):
        component = _text(part, f"{where}.request_path[{index}]", maximum=128)
        if not _REQUEST_PATH_PART.fullmatch(component):
            raise CapabilityProfileError(
                f"{where}.request_path[{index}] is not a valid request key"
            )
        path.append(component)
    values = _require_list(item["values"], f"{where}.values")
    seen_values: set[bytes] = set()
    for index, value in enumerate(values):
        key = _validate_value_contract(
            value,
            f"{where}.values[{index}]",
            source_ids,
            require_value=True,
        )
        assert key is not None
        if key in seen_values:
            raise CapabilityProfileError(f"{where}.values has duplicate request values")
        seen_values.add(key)
    _validate_value_contract(
        item["omitted"],
        f"{where}.omitted",
        source_ids,
        require_value=False,
    )
    return tuple(path)


def _validate_reasoning(reasoning: Any, where: str, source_ids: set[str]) -> None:
    item = _require_object(reasoning, where)
    _exact_keys(
        item,
        {"mode", "effective_behavior", "controls", "source_refs"},
        set(),
        where,
    )
    mode = _slug(item["mode"], f"{where}.mode")
    if mode not in _REASONING_MODES:
        raise CapabilityProfileError(
            f"{where}.mode must be one of: {', '.join(sorted(_REASONING_MODES))}"
        )
    behavior = _text(
        item["effective_behavior"], f"{where}.effective_behavior", maximum=128
    )
    if not _BEHAVIOR.fullmatch(behavior):
        raise CapabilityProfileError(
            f"{where}.effective_behavior must be a lowercase identifier"
        )
    controls = _require_list(
        item["controls"], f"{where}.controls", nonempty=False
    )
    if mode == "configurable" and not controls:
        raise CapabilityProfileError(
            f"{where}.controls must not be empty when mode is configurable"
        )
    if mode != "configurable" and controls:
        raise CapabilityProfileError(
            f"{where}.controls must be empty unless mode is configurable"
        )
    seen_paths: set[tuple[str, ...]] = set()
    for index, control in enumerate(controls):
        path = _validate_control(
            control, f"{where}.controls[{index}]", source_ids
        )
        if path in seen_paths:
            raise CapabilityProfileError(f"{where}.controls has duplicate request paths")
        seen_paths.add(path)
    _source_refs(item["source_refs"], f"{where}.source_refs", source_ids)


def _validate_model(model: Any, index: int) -> tuple[str, str]:
    where = f"models[{index}]"
    item = _require_object(model, where)
    _exact_keys(item, {"provider", "model_id"}, set(), where)
    provider = _slug(item["provider"], f"{where}.provider")
    model_id = _text(item["model_id"], f"{where}.model_id", maximum=256)
    return provider, model_id


def _validate_route(
    route: Any,
    index: int,
    models: set[tuple[str, str]],
    source_ids: set[str],
) -> tuple[str, list[tuple[str, str, str, str]]]:
    where = f"routes[{index}]"
    item = _require_object(route, where)
    _exact_keys(
        item,
        {
            "route_id",
            "provider",
            "route",
            "api",
            "endpoint_adapter",
            "model_ids",
            "stream_modes",
            "reasoning",
            "source_refs",
        },
        set(),
        where,
    )
    route_id = _versioned_id(item["route_id"], f"{where}.route_id")
    provider = _slug(item["provider"], f"{where}.provider")
    route_name = _text(item["route"], f"{where}.route", maximum=256)
    api = _slug(item["api"], f"{where}.api")
    _versioned_id(item["endpoint_adapter"], f"{where}.endpoint_adapter")
    model_ids = _unique_text_list(
        item["model_ids"], f"{where}.model_ids", parser=_text
    )
    for model_id in model_ids:
        if (provider, model_id) not in models:
            raise CapabilityProfileError(
                f"{where}.model_ids contains {model_id!r}, which is not declared "
                f"for provider {provider!r}"
            )
    _unique_text_list(
        item["stream_modes"], f"{where}.stream_modes", parser=_slug
    )
    _validate_reasoning(item["reasoning"], f"{where}.reasoning", source_ids)
    _source_refs(item["source_refs"], f"{where}.source_refs", source_ids)
    keys = [(provider, model_id, route_name, api) for model_id in model_ids]
    return route_id, keys


def _validate_document(document: Any) -> tuple[date, date | None]:
    root = _require_object(document, "profile")
    _exact_keys(
        root,
        {
            "schema_version",
            "profile_id",
            "effective_from",
            "verified_at",
            "sources",
            "models",
            "routes",
            "digest",
        },
        {"effective_until"},
        "profile",
    )
    digest = root["digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise CapabilityIntegrityError(
            "profile.digest must be sha256 followed by 64 lowercase hex digits"
        )
    expected = compute_profile_digest(root)
    if digest != expected:
        raise CapabilityIntegrityError(
            f"profile.digest mismatch: expected {expected}, found {digest}"
        )
    if root["schema_version"] != CAPABILITY_SCHEMA_VERSION:
        raise CapabilityProfileError(
            "profile.schema_version must be " + repr(CAPABILITY_SCHEMA_VERSION)
        )
    _versioned_id(root["profile_id"], "profile.profile_id")
    effective_from = _iso_date(root["effective_from"], "profile.effective_from")
    verified_at = _iso_date(root["verified_at"], "profile.verified_at")
    if verified_at < effective_from:
        raise CapabilityProfileError(
            "profile.verified_at cannot be earlier than effective_from"
        )
    effective_until = None
    if "effective_until" in root:
        effective_until = _iso_date(
            root["effective_until"], "profile.effective_until", future_ok=True
        )
        if effective_until < effective_from:
            raise CapabilityProfileError(
                "profile.effective_until cannot be earlier than effective_from"
            )

    sources = _require_list(root["sources"], "profile.sources")
    source_ids: set[str] = set()
    retrieved_dates: list[date] = []
    for index, source in enumerate(sources):
        source_id, retrieved_at = _validate_source(source, index)
        if source_id in source_ids:
            raise CapabilityDuplicateError(
                f"profile.sources has duplicate source_id {source_id!r}"
            )
        source_ids.add(source_id)
        retrieved_dates.append(retrieved_at)
    if max(retrieved_dates) > verified_at:
        raise CapabilityProfileError(
            "profile.verified_at cannot be earlier than a source retrieval date"
        )

    model_items = _require_list(root["models"], "profile.models")
    models: set[tuple[str, str]] = set()
    for index, model in enumerate(model_items):
        selector = _validate_model(model, index)
        if selector in models:
            raise CapabilityDuplicateError(
                "profile.models has duplicate provider/model selector"
            )
        models.add(selector)

    route_items = _require_list(root["routes"], "profile.routes")
    route_ids: set[str] = set()
    route_keys: set[tuple[str, str, str, str]] = set()
    for index, route in enumerate(route_items):
        route_id, keys = _validate_route(route, index, models, source_ids)
        if route_id in route_ids:
            raise CapabilityDuplicateError(
                f"profile.routes has duplicate route_id {route_id!r}"
            )
        route_ids.add(route_id)
        overlap = route_keys.intersection(keys)
        if overlap:
            raise CapabilityAmbiguityError(
                "profile.routes describes the same provider/model/route/API "
                "target more than once"
            )
        route_keys.update(keys)

    return effective_from, effective_until


@dataclass(frozen=True, slots=True)
class ModelSelector:
    """An exact provider/model identity; no substring or name inference."""

    provider: str
    model_id: str


@dataclass(frozen=True, slots=True)
class RouteCapability:
    """One exact route/API binding returned as an immutable snapshot."""

    route_id: str
    provider: str
    route: str
    api: str
    endpoint_adapter: str
    model_ids: tuple[str, ...]
    stream_modes: tuple[str, ...]
    _canonical_document: bytes = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a new mutable JSON object on every call."""
        result = _strict_json_loads(self._canonical_document)
        assert isinstance(result, dict)
        return result


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Validated immutable capability-profile snapshot."""

    schema_version: str
    profile_id: str
    effective_from: date
    effective_until: date | None
    verified_at: date
    digest: str
    models: tuple[ModelSelector, ...]
    routes: tuple[RouteCapability, ...]
    _canonical_document: bytes = field(repr=False)
    _source_path: str | None = field(default=None, repr=False)
    _source_raw_sha256: str | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CapabilityProfile":
        """Validate and copy a mapping without retaining caller-owned objects."""
        clone = _json_clone(document)
        effective_from, effective_until = _validate_document(clone)
        return cls._from_validated(clone, effective_from, effective_until)

    @classmethod
    def _from_validated(
        cls,
        document: dict[str, Any],
        effective_from: date,
        effective_until: date | None,
        *,
        source_path: str | None = None,
        source_raw_sha256: str | None = None,
    ) -> "CapabilityProfile":
        routes = tuple(
            RouteCapability(
                route_id=item["route_id"],
                provider=item["provider"],
                route=item["route"],
                api=item["api"],
                endpoint_adapter=item["endpoint_adapter"],
                model_ids=tuple(item["model_ids"]),
                stream_modes=tuple(item["stream_modes"]),
                _canonical_document=_canonical_json(item),
            )
            for item in document["routes"]
        )
        return cls(
            schema_version=document["schema_version"],
            profile_id=document["profile_id"],
            effective_from=effective_from,
            effective_until=effective_until,
            verified_at=date.fromisoformat(document["verified_at"]),
            digest=document["digest"],
            models=tuple(
                ModelSelector(item["provider"], item["model_id"])
                for item in document["models"]
            ),
            routes=routes,
            _canonical_document=_canonical_json(document),
            _source_path=source_path,
            _source_raw_sha256=source_raw_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new mutable JSON object on every call."""
        result = _strict_json_loads(self._canonical_document)
        assert isinstance(result, dict)
        return result

    @property
    def source_path(self) -> Path | None:
        return Path(self._source_path) if self._source_path is not None else None

    def is_active(self, as_of: date) -> bool:
        return self.effective_from <= as_of and (
            self.effective_until is None or as_of <= self.effective_until
        )

    def assert_source_unchanged(self) -> None:
        """Fail if a file-backed profile changed after this snapshot was loaded."""
        if self._source_path is None or self._source_raw_sha256 is None:
            return
        raw = _read_regular_file(Path(self._source_path))
        observed = hashlib.sha256(raw).hexdigest()
        if observed != self._source_raw_sha256:
            raise CapabilityIntegrityError(
                f"capability profile source changed after load: {self._source_path}"
            )


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    """The one unambiguous profile and route for an exact target."""

    profile: CapabilityProfile
    route: RouteCapability


def _read_regular_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CapabilityIntegrityError(f"cannot inspect profile {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise CapabilityIntegrityError(
            f"capability profile must be a regular non-symlink file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CapabilityIntegrityError(f"cannot open profile safely {path}: {exc}") from exc
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise CapabilityIntegrityError(f"capability profile changed while opening: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, MAX_PROFILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PROFILE_BYTES:
                raise CapabilityIntegrityError(
                    f"capability profile exceeds {MAX_PROFILE_BYTES} bytes: {path}"
                )
    finally:
        os.close(fd)
    try:
        after = path.lstat()
    except OSError as exc:
        raise CapabilityIntegrityError(f"profile disappeared while reading: {path}") from exc
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise CapabilityIntegrityError(f"capability profile changed while reading: {path}")
    return b"".join(chunks)


def load_capability_profile(path: str | Path) -> CapabilityProfile:
    """Load one regular JSON file as a validated immutable snapshot."""
    # ``abspath`` normalizes a relative path without resolving a final symlink;
    # the lstat/O_NOFOLLOW checks in ``_read_regular_file`` must see that link.
    absolute = Path(os.path.abspath(path))
    raw = _read_regular_file(absolute)
    document = _strict_json_loads(raw)
    effective_from, effective_until = _validate_document(document)
    return CapabilityProfile._from_validated(
        document,
        effective_from,
        effective_until,
        source_path=str(absolute),
        source_raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


class CapabilityCatalog:
    """Immutable, ambiguity-free collection of capability profiles."""

    __slots__ = ("_profiles", "_by_id", "_sealed")

    def __init__(self, profiles: Iterable[CapabilityProfile] = ()) -> None:
        snapshot = tuple(profiles)
        if any(not isinstance(profile, CapabilityProfile) for profile in snapshot):
            raise TypeError("catalog entries must be CapabilityProfile objects")
        by_id: dict[str, CapabilityProfile] = {}
        claims: dict[
            tuple[str, str, str, str],
            list[tuple[date, date | None, str]],
        ] = {}
        for profile in snapshot:
            if profile.profile_id in by_id:
                raise CapabilityDuplicateError(
                    f"duplicate capability profile_id {profile.profile_id!r}"
                )
            by_id[profile.profile_id] = profile
            for route in profile.routes:
                for model_id in route.model_ids:
                    key = (route.provider, model_id, route.route, route.api)
                    for start, end, owner in claims.get(key, []):
                        if _periods_overlap(
                            profile.effective_from,
                            profile.effective_until,
                            start,
                            end,
                        ):
                            raise CapabilityAmbiguityError(
                                "overlapping capability profiles claim "
                                f"{key!r}: {owner!r} and {profile.profile_id!r}"
                            )
                    claims.setdefault(key, []).append(
                        (
                            profile.effective_from,
                            profile.effective_until,
                            profile.profile_id,
                        )
                    )
        object.__setattr__(self, "_profiles", snapshot)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CapabilityCatalog is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "CapabilityCatalog":
        """Load every top-level ``*.json`` profile from one regular directory."""
        root = Path(directory)
        try:
            info = root.lstat()
        except OSError as exc:
            raise CapabilityIntegrityError(
                f"cannot inspect capability directory {root}: {exc}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise CapabilityIntegrityError(
                f"capability catalog path must be a non-symlink directory: {root}"
            )
        paths = sorted(path for path in root.iterdir() if path.suffix == ".json")
        if not paths:
            raise CapabilityProfileError(
                f"capability directory contains no JSON profiles: {root}"
            )
        return cls(load_capability_profile(path) for path in paths)

    @property
    def profiles(self) -> tuple[CapabilityProfile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> CapabilityProfile:
        try:
            return self._by_id[profile_id]
        except KeyError as exc:
            raise CapabilityNotFoundError(
                f"unknown capability profile_id {profile_id!r}"
            ) from exc

    def with_profile(self, profile: CapabilityProfile) -> "CapabilityCatalog":
        """Return a new catalog; the original catalog is never mutated."""
        return CapabilityCatalog((*self._profiles, profile))

    def resolve(
        self,
        *,
        provider: str,
        model_id: str,
        route: str,
        api: str,
        as_of: date | None = None,
    ) -> CapabilityResolution:
        """Resolve one exact active target; fuzzy or default matching is forbidden."""
        target_date = as_of or date.today()
        if not isinstance(target_date, date):
            raise TypeError("as_of must be a datetime.date")
        matches: list[CapabilityResolution] = []
        for profile in self._profiles:
            if not profile.is_active(target_date):
                continue
            if ModelSelector(provider, model_id) not in profile.models:
                continue
            for candidate in profile.routes:
                if (
                    candidate.provider == provider
                    and model_id in candidate.model_ids
                    and candidate.route == route
                    and candidate.api == api
                ):
                    matches.append(CapabilityResolution(profile, candidate))
        if not matches:
            raise CapabilityNotFoundError(
                "no active capability profile for exact target "
                f"provider={provider!r}, model_id={model_id!r}, "
                f"route={route!r}, api={api!r}, as_of={target_date.isoformat()}"
            )
        if len(matches) != 1:
            # Constructor validation should make this unreachable, but retain a
            # fail-closed guard if catalog internals are ever changed.
            raise CapabilityAmbiguityError(
                "multiple active capability profiles matched an exact target"
            )
        return matches[0]

    def assert_sources_unchanged(self) -> None:
        for profile in self._profiles:
            profile.assert_source_unchanged()


def _periods_overlap(
    first_start: date,
    first_end: date | None,
    second_start: date,
    second_end: date | None,
) -> bool:
    return first_start <= (second_end or date.max) and second_start <= (
        first_end or date.max
    )
