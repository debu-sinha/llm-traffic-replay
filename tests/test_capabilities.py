"""Contract tests for the data-only model capability catalog."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from traffic_replay.capabilities import (
    CAPABILITY_SCHEMA_VERSION,
    MAX_PROFILE_BYTES,
    CapabilityAmbiguityError,
    CapabilityCatalog,
    CapabilityDuplicateError,
    CapabilityIntegrityError,
    CapabilityNotFoundError,
    CapabilityProfile,
    CapabilityProfileError,
    compute_profile_digest,
    load_capability_profile,
    seal_capability_profile,
)


def _profile_document(
    *,
    profile_id: str = "future-model-profile/v1",
    provider: str = "future-provider",
    model_id: str = "frontier/model-next-2030",
    route: str = "managed-model-route",
    api: str = "chat_completions",
    adapter: str = "openai.chat_completions.sse/v1",
    effective_from: str = "2024-01-01",
    effective_until: str | None = None,
) -> dict:
    document = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "profile_id": profile_id,
        "effective_from": effective_from,
        "verified_at": effective_from,
        "sources": [
            {
                "source_id": "provider-doc",
                "kind": "provider_documentation",
                "uri": "https://docs.example.test/models/model-next",
                "retrieved_at": effective_from,
            },
            {
                "source_id": "route-probe",
                "kind": "runtime_probe",
                "uri": "urn:benchmark-probe:future-model:2024-01-01",
                "retrieved_at": effective_from,
            },
        ],
        "models": [{"provider": provider, "model_id": model_id}],
        "routes": [
            {
                "route_id": "future-provider.managed-chat/v1",
                "provider": provider,
                "route": route,
                "api": api,
                "endpoint_adapter": adapter,
                "model_ids": [model_id],
                "stream_modes": ["streaming", "non_streaming"],
                "reasoning": {
                    "mode": "configurable",
                    "effective_behavior": "value_dependent",
                    "controls": [
                        {
                            "request_path": ["reasoning_effort"],
                            "values": [
                                {
                                    "value": "off",
                                    "acceptance": "accepted",
                                    "effective_behavior": "thinking_disabled",
                                    "source_refs": ["route-probe"],
                                },
                                {
                                    "value": "low",
                                    "acceptance": "accepted",
                                    "effective_behavior": "lower_effort",
                                    "source_refs": ["provider-doc", "route-probe"],
                                },
                            ],
                            "omitted": {
                                "acceptance": "accepted",
                                "effective_behavior": "provider_default",
                                "source_refs": ["provider-doc"],
                            },
                        }
                    ],
                    "source_refs": ["provider-doc", "route-probe"],
                },
                "source_refs": ["provider-doc", "route-probe"],
            }
        ],
    }
    if effective_until is not None:
        document["effective_until"] = effective_until
    return seal_capability_profile(document)


def _reseal(document: dict) -> dict:
    clone = json.loads(json.dumps(document))
    clone.pop("digest", None)
    return seal_capability_profile(clone)


def _write_profile(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def test_valid_profile_is_data_only_and_resolves_arbitrary_future_model():
    profile = CapabilityProfile.from_dict(_profile_document())
    catalog = CapabilityCatalog([profile])

    resolution = catalog.resolve(
        provider="future-provider",
        model_id="frontier/model-next-2030",
        route="managed-model-route",
        api="chat_completions",
        as_of=date(2025, 1, 1),
    )

    assert resolution.profile is profile
    assert resolution.route.endpoint_adapter == "openai.chat_completions.sse/v1"
    assert resolution.route.stream_modes == ("streaming", "non_streaming")
    reasoning = resolution.route.to_dict()["reasoning"]
    assert reasoning["controls"][0]["request_path"] == ["reasoning_effort"]
    assert reasoning["controls"][0]["values"][0] == {
        "value": "off",
        "acceptance": "accepted",
        "effective_behavior": "thinking_disabled",
        "source_refs": ["route-probe"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "Future-Provider"),
        ("model_id", "model-next"),
        ("route", "other-route"),
        ("api", "responses"),
    ],
)
def test_resolution_is_exact_and_never_fuzzy(field: str, value: str):
    catalog = CapabilityCatalog(
        [CapabilityProfile.from_dict(_profile_document())]
    )
    target = {
        "provider": "future-provider",
        "model_id": "frontier/model-next-2030",
        "route": "managed-model-route",
        "api": "chat_completions",
        "as_of": date(2025, 1, 1),
    }
    target[field] = value
    with pytest.raises(CapabilityNotFoundError, match="exact target"):
        catalog.resolve(**target)


def test_profile_defensively_copies_input_and_output():
    source = _profile_document()
    profile = CapabilityProfile.from_dict(source)
    source["models"][0]["model_id"] = "mutated"
    first = profile.to_dict()
    first["models"][0]["model_id"] = "also-mutated"
    route = profile.routes[0].to_dict()
    route["reasoning"]["mode"] = "mutated"

    assert profile.models[0].model_id == "frontier/model-next-2030"
    assert profile.to_dict()["models"][0]["model_id"] == \
        "frontier/model-next-2030"
    assert profile.routes[0].to_dict()["reasoning"]["mode"] == "configurable"


def test_profile_objects_and_catalog_membership_are_immutable():
    profile = CapabilityProfile.from_dict(_profile_document())
    catalog = CapabilityCatalog([profile])

    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "changed/v1"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        profile.routes[0].api = "changed"  # type: ignore[misc]
    assert isinstance(catalog.profiles, tuple)
    with pytest.raises(TypeError):
        catalog._by_id[profile.profile_id] = profile  # type: ignore[index]
    with pytest.raises(AttributeError, match="immutable"):
        catalog._profiles = ()  # type: ignore[misc]


def test_with_profile_returns_new_catalog_without_mutating_original():
    first = CapabilityProfile.from_dict(_profile_document())
    second = CapabilityProfile.from_dict(
        _profile_document(
            profile_id="another-model-profile/v1",
            model_id="frontier/another-model",
        )
    )
    original = CapabilityCatalog([first])
    extended = original.with_profile(second)

    assert original.profiles == (first,)
    assert extended.profiles == (first, second)
    assert extended.get(second.profile_id) is second


def test_digest_is_canonical_across_object_key_order():
    document = _profile_document()
    reversed_document = dict(reversed(list(document.items())))
    assert compute_profile_digest(document) == compute_profile_digest(
        reversed_document
    )


def test_seal_refuses_to_hide_existing_digest():
    with pytest.raises(CapabilityProfileError, match="overwrite"):
        seal_capability_profile(_profile_document())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["models"][0].update(model_id="changed"),
        lambda value: value["routes"][0].update(api="responses"),
        lambda value: value["sources"][0].update(retrieved_at="2024-01-02"),
    ],
)
def test_any_claim_mutation_without_resealing_fails_digest(mutation):
    document = _profile_document()
    mutation(document)
    with pytest.raises(CapabilityIntegrityError, match="digest mismatch"):
        CapabilityProfile.from_dict(document)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(schema_version="v2"), "schema_version"),
        (lambda value: value.update(extra=True), "unknown field"),
        (lambda value: value.pop("verified_at"), "missing required field"),
        (lambda value: value.update(profile_id="Unversioned"), "versioned ID"),
        (lambda value: value.update(effective_from="2024-1-1"), "YYYY-MM-DD"),
    ],
)
def test_schema_version_keys_ids_and_dates_are_strict(change, message: str):
    document = _profile_document()
    change(document)
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match=message):
        CapabilityProfile.from_dict(document)


def test_profile_dates_cannot_claim_verification_before_effective_or_source():
    document = _profile_document()
    document["effective_from"] = "2024-01-02"
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="verified_at"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    document["sources"][0]["retrieved_at"] = "2024-01-02"
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="retrieval date"):
        CapabilityProfile.from_dict(document)


def test_future_verification_and_source_dates_are_rejected():
    future = (date.today() + timedelta(days=1)).isoformat()
    document = _profile_document()
    document["effective_from"] = future
    document["verified_at"] = future
    for source in document["sources"]:
        source["retrieved_at"] = future
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="future"):
        CapabilityProfile.from_dict(document)


def test_effective_until_may_be_future_but_must_not_precede_start():
    future = (date.today() + timedelta(days=365)).isoformat()
    assert CapabilityProfile.from_dict(
        _profile_document(effective_until=future)
    ).effective_until == date.fromisoformat(future)

    document = _profile_document(effective_until="2023-12-31")
    with pytest.raises(CapabilityProfileError, match="effective_until"):
        CapabilityProfile.from_dict(document)


@pytest.mark.parametrize("uri", ["http://example.test/doc", "file:///tmp/doc"])
def test_sources_require_https_or_urn(uri: str):
    document = _profile_document()
    document["sources"][0]["uri"] = uri
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="https or urn"):
        CapabilityProfile.from_dict(document)


def test_source_credentials_unknown_kinds_and_unknown_refs_fail_closed():
    document = _profile_document()
    document["sources"][0]["uri"] = "https://user:secret@example.test/doc"
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="credentials"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    document["sources"][0]["kind"] = "internet-rumor"
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="kind must be"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    document["routes"][0]["source_refs"] = ["missing-source"]
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="unknown source"):
        CapabilityProfile.from_dict(document)


def test_duplicate_sources_models_routes_and_target_claims_fail_closed():
    document = _profile_document()
    document["sources"].append(dict(document["sources"][0]))
    document = _reseal(document)
    with pytest.raises(CapabilityDuplicateError, match="source_id"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    document["models"].append(dict(document["models"][0]))
    document = _reseal(document)
    with pytest.raises(CapabilityDuplicateError, match="models"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    duplicate = json.loads(json.dumps(document["routes"][0]))
    duplicate["route_id"] = "future-provider.other-route-id/v1"
    document["routes"].append(duplicate)
    document = _reseal(document)
    with pytest.raises(CapabilityAmbiguityError, match="same provider"):
        CapabilityProfile.from_dict(document)


def test_route_must_reference_a_declared_exact_model():
    document = _profile_document()
    document["routes"][0]["model_ids"] = ["frontier/undeclared"]
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="not declared"):
        CapabilityProfile.from_dict(document)


def test_reasoning_mode_controls_and_omitted_contract_are_strict():
    document = _profile_document()
    document["routes"][0]["reasoning"]["controls"] = []
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="must not be empty"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    reasoning = document["routes"][0]["reasoning"]
    reasoning["mode"] = "unknown"
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="must be empty unless"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    del document["routes"][0]["reasoning"]["controls"][0]["omitted"]
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="missing required field"):
        CapabilityProfile.from_dict(document)


def test_reasoning_values_cannot_be_duplicate_or_structured():
    document = _profile_document()
    values = document["routes"][0]["reasoning"]["controls"][0]["values"]
    values.append(dict(values[0]))
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="duplicate request values"):
        CapabilityProfile.from_dict(document)

    document = _profile_document()
    values = document["routes"][0]["reasoning"]["controls"][0]["values"]
    values[0]["value"] = {"nested": "ambiguous"}
    document = _reseal(document)
    with pytest.raises(CapabilityProfileError, match="JSON scalar"):
        CapabilityProfile.from_dict(document)


def test_overlapping_catalog_claims_fail_at_construction():
    first = CapabilityProfile.from_dict(_profile_document())
    second = CapabilityProfile.from_dict(
        _profile_document(profile_id="replacement-profile/v2")
    )
    with pytest.raises(CapabilityAmbiguityError, match="overlapping"):
        CapabilityCatalog([first, second])


def test_disjoint_profile_revisions_resolve_by_effective_date():
    old = CapabilityProfile.from_dict(
        _profile_document(effective_until="2024-12-31")
    )
    new = CapabilityProfile.from_dict(
        _profile_document(
            profile_id="future-model-profile/v2",
            effective_from="2025-01-01",
        )
    )
    catalog = CapabilityCatalog([old, new])

    assert catalog.resolve(
        provider="future-provider",
        model_id="frontier/model-next-2030",
        route="managed-model-route",
        api="chat_completions",
        as_of=date(2024, 12, 31),
    ).profile.profile_id == "future-model-profile/v1"
    assert catalog.resolve(
        provider="future-provider",
        model_id="frontier/model-next-2030",
        route="managed-model-route",
        api="chat_completions",
        as_of=date(2025, 1, 1),
    ).profile.profile_id == "future-model-profile/v2"


def test_duplicate_profile_id_and_unknown_id_fail_closed():
    profile = CapabilityProfile.from_dict(_profile_document())
    with pytest.raises(CapabilityDuplicateError, match="profile_id"):
        CapabilityCatalog([profile, profile])
    with pytest.raises(CapabilityNotFoundError, match="unknown"):
        CapabilityCatalog([profile]).get("absent/v1")


def test_catalog_rejects_wrong_entry_type_and_as_of_type():
    with pytest.raises(TypeError, match="CapabilityProfile"):
        CapabilityCatalog([{}])  # type: ignore[list-item]
    catalog = CapabilityCatalog(
        [CapabilityProfile.from_dict(_profile_document())]
    )
    with pytest.raises(TypeError, match="datetime.date"):
        catalog.resolve(
            provider="future-provider",
            model_id="frontier/model-next-2030",
            route="managed-model-route",
            api="chat_completions",
            as_of="2025-01-01",  # type: ignore[arg-type]
        )


def test_file_loader_detects_duplicate_json_keys_and_non_finite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(CapabilityProfileError, match="duplicate JSON object key"):
        load_capability_profile(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(CapabilityProfileError, match="non-finite"):
        load_capability_profile(nonfinite)


def test_file_backed_profile_detects_post_load_mutation(tmp_path):
    path = tmp_path / "profile.json"
    _write_profile(path, _profile_document())
    profile = load_capability_profile(path)
    catalog = CapabilityCatalog([profile])
    catalog.assert_sources_unchanged()

    changed = _profile_document(model_id="frontier/replacement")
    _write_profile(path, changed)
    with pytest.raises(CapabilityIntegrityError, match="changed after load"):
        profile.assert_source_unchanged()
    with pytest.raises(CapabilityIntegrityError, match="changed after load"):
        catalog.assert_sources_unchanged()


def test_loader_rejects_symlinks_and_oversize_files(tmp_path):
    real = tmp_path / "real.json"
    _write_profile(real, _profile_document())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(CapabilityIntegrityError, match="non-symlink"):
        load_capability_profile(link)

    huge = tmp_path / "huge.json"
    huge.write_bytes(b" " * (MAX_PROFILE_BYTES + 1))
    with pytest.raises(CapabilityIntegrityError, match="exceeds"):
        load_capability_profile(huge)


def test_directory_catalog_loads_profiles_and_rejects_empty_or_symlink(tmp_path):
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    _write_profile(catalog_dir / "one.json", _profile_document())
    (catalog_dir / "README.md").write_text("not a profile")

    catalog = CapabilityCatalog.from_directory(catalog_dir)
    assert len(catalog.profiles) == 1
    assert catalog.profiles[0].source_path == catalog_dir / "one.json"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CapabilityProfileError, match="no JSON"):
        CapabilityCatalog.from_directory(empty)

    linked_dir = tmp_path / "linked-catalog"
    linked_dir.symlink_to(catalog_dir, target_is_directory=True)
    with pytest.raises(CapabilityIntegrityError, match="non-symlink directory"):
        CapabilityCatalog.from_directory(linked_dir)


def test_directory_catalog_fails_on_overlapping_files(tmp_path):
    _write_profile(tmp_path / "one.json", _profile_document())
    _write_profile(
        tmp_path / "two.json",
        _profile_document(profile_id="other-profile/v1"),
    )
    with pytest.raises(CapabilityAmbiguityError, match="overlapping"):
        CapabilityCatalog.from_directory(tmp_path)
