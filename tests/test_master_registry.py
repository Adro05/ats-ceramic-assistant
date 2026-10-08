"""Focused Golden Master Registry tests using synthetic values only."""

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from pydantic import ValidationError

from ats_ceramic.master_registry import (
    GoldenMasterRecord,
    GoldenMasterRegistry,
    MasterKind,
    content_sha256,
)
from ats_ceramic.schemas import DataOrigin, MasterMeasurement, ToleranceSpec

NOW = datetime(2026, 1, 4, 10, 0, tzinfo=UTC)
CONTENT_A = b"synthetic-master-artwork-v1"
CONTENT_B = b"synthetic-master-artwork-v2"
HASH_A = sha256(CONTENT_A).hexdigest()


def _measurement(*, sku_id="SKU_SYN_01", data_origin=DataOrigin.SYNTHETIC):
    from ats_ceramic.schemas import MeasurementMetadata, RegionMeasurement

    metadata = MeasurementMetadata(
        instrument_id="SPEC_SYN_1",
        geometry="d/8",
        illuminant="D65",
        observer="2deg",
        specular_mode="SCI",
        aperture_mm=8.0,
        measured_at=NOW,
        data_origin=data_origin,
    )
    return MasterMeasurement(
        sku_id=sku_id,
        regions=(RegionMeasurement(region_id="R1", L=55.0, a=12.0, b=-4.0, gloss=60.0),),
        metadata=metadata,
    )


def _record(
    *,
    master_id="M_SYN_01",
    version=1,
    content=CONTENT_A,
    sku_id="SKU_SYN_01",
    context_id="CTX_SYN_01",
    kind=MasterKind.CANONICAL_MASTER,
    parent_version=None,
    parent_master_id=None,
    canonical_master_id=None,
    tolerance=None,
    data_origin=DataOrigin.SYNTHETIC,
):
    return GoldenMasterRecord(
        master_id=master_id,
        version=version,
        sku_id=sku_id,
        context_id=context_id,
        master_kind=kind,
        content_bytes=content,
        content_hash=content_sha256(content),
        registered_at=NOW,
        parent_version=parent_version,
        parent_master_id=parent_master_id,
        canonical_master_id=canonical_master_id,
        tolerance=tolerance,
        data_origin=data_origin,
        provenance="synthetic registry fixture",
    )


def _placeholder_tolerance():
    return ToleranceSpec(
        acceptance_metric="max_de00",
        delta_e00_threshold=1.0,
        gloss_abs_threshold=5.0,
        is_placeholder=True,
        source="synthetic placeholder only",
    )


def test_registering_canonical_master_succeeds():
    registry = GoldenMasterRegistry()
    record = _record(tolerance=_placeholder_tolerance())
    assert registry.register(record) == record
    assert registry.get("M_SYN_01", 1) == record


def test_same_deterministic_content_has_same_sha256():
    assert content_sha256(CONTENT_A) == HASH_A
    assert content_sha256(bytes(CONTENT_A)) == HASH_A


def test_changing_content_changes_hash():
    assert content_sha256(CONTENT_A) != content_sha256(CONTENT_B)


def test_registered_master_preserves_content_hash():
    record = _record()
    registry = GoldenMasterRegistry()
    registry.register(record)
    assert registry.get("M_SYN_01", 1).content_hash == HASH_A


def test_golden_master_is_immutable():
    record = _record()
    with pytest.raises(ValidationError):
        record.content_hash = "b" * 64


def test_duplicate_version_registration_is_rejected():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_record())


def test_existing_version_cannot_be_overwritten():
    registry = GoldenMasterRegistry()
    first = _record()
    registry.register(first)
    replacement = _record(content=CONTENT_B)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(replacement)
    assert registry.get("M_SYN_01", 1) == first


def test_new_version_preserves_previous_version_unchanged():
    registry = GoldenMasterRegistry()
    first = _record()
    registry.register(first)
    second = _record(version=2, content=CONTENT_B, parent_version=1, parent_master_id="M_SYN_01")
    registry.register(second)
    assert registry.get("M_SYN_01", 1) == first
    assert registry.get("M_SYN_01", 2) == second


def test_version_lineage_is_recorded_correctly():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    child = _record(version=2, content=CONTENT_B, parent_version=1, parent_master_id="M_SYN_01")
    registry.register(child)
    assert child.parent_master_id == "M_SYN_01"
    assert child.parent_version == 1


def test_invalid_parent_lineage_is_rejected():
    registry = GoldenMasterRegistry()
    child = _record(version=2, content=CONTENT_B, parent_version=1, parent_master_id="M_SYN_01")
    with pytest.raises(ValueError, match="parent master version is not registered"):
        registry.register(child)


def test_canonical_and_legacy_kinds_are_distinguishable():
    registry = GoldenMasterRegistry()
    canonical = _record()
    registry.register(canonical)
    legacy = _record(
        master_id="LEGACY_SYN_01",
        kind=MasterKind.LEGACY_WORKING_REFERENCE,
        canonical_master_id="M_SYN_01",
        content=CONTENT_B,
    )
    registry.register(legacy)
    assert canonical.master_kind is MasterKind.CANONICAL_MASTER
    assert registry.get("LEGACY_SYN_01", 1).master_kind is MasterKind.LEGACY_WORKING_REFERENCE


def test_legacy_master_does_not_replace_canonical_master():
    registry = GoldenMasterRegistry()
    canonical = _record()
    registry.register(canonical)
    legacy = _record(
        master_id="LEGACY_SYN_01",
        kind=MasterKind.LEGACY_WORKING_REFERENCE,
        canonical_master_id="M_SYN_01",
        content=CONTENT_B,
    )
    registry.register(legacy)
    assert registry.get("M_SYN_01", 1) == canonical
    assert registry.get("M_SYN_01", 1).master_kind is MasterKind.CANONICAL_MASTER


def test_reanchoring_is_not_automatic():
    registry = GoldenMasterRegistry()
    canonical = _record()
    registry.register(canonical)
    legacy = _record(
        master_id="LEGACY_SYN_01",
        kind=MasterKind.LEGACY_WORKING_REFERENCE,
        canonical_master_id="M_SYN_01",
        content=CONTENT_B,
    )
    registry.register(legacy)
    assert registry.get("M_SYN_01", 1).content_hash == HASH_A
    assert registry.get("M_SYN_01", 1).content_bytes == CONTENT_A


def test_sku_mismatch_is_rejected_in_lineage():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    child = _record(
        version=2,
        content=CONTENT_B,
        parent_version=1,
        parent_master_id="M_SYN_01",
        sku_id="SKU_OTHER_SYN",
    )
    with pytest.raises(ValueError, match="parent SKU"):
        registry.register(child)


def test_data_origin_mismatch_is_rejected_in_lineage():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    child = _record(
        version=2,
        content=CONTENT_B,
        parent_version=1,
        parent_master_id="M_SYN_01",
        data_origin=DataOrigin.CLIENT,
    )
    with pytest.raises(ValueError, match="parent data origin"):
        registry.register(child)


def test_synthetic_provenance_is_explicit():
    record = _record()
    assert record.data_origin is DataOrigin.SYNTHETIC
    assert record.provenance == "synthetic registry fixture"


def test_instrument_specific_measurement_can_be_associated():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    measurement = _measurement()
    registry.associate_measurement("M_SYN_01", 1, measurement)
    assert registry.measurements_for("M_SYN_01", 1) == (measurement,)


def test_measurement_sku_mismatch_is_rejected():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    with pytest.raises(ValueError, match="measurement SKU"):
        registry.associate_measurement("M_SYN_01", 1, _measurement(sku_id="SKU_OTHER_SYN"))


def test_measurement_data_origin_mismatch_is_rejected():
    registry = GoldenMasterRegistry()
    registry.register(_record())
    with pytest.raises(ValueError, match="measurement data origin"):
        registry.associate_measurement("M_SYN_01", 1, _measurement(data_origin=DataOrigin.CLIENT))


def test_invalid_instrument_identity_is_rejected_at_schema_level():
    from ats_ceramic.schemas import MeasurementMetadata, RegionMeasurement

    registry = GoldenMasterRegistry()
    registry.register(_record())
    metadata = MeasurementMetadata(
        instrument_id="SPEC_SYN_1",
        geometry="d/8",
        illuminant="D65",
        observer="2deg",
        specular_mode="SCI",
        aperture_mm=8.0,
        measured_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
    )
    invalid = MasterMeasurement(
        sku_id="SKU_SYN_01",
        regions=(RegionMeasurement(region_id="R1", L=55.0, a=12.0, b=-4.0, gloss=60.0),),
        metadata=metadata,
    )
    # MasterMeasurement itself requires a valid instrument-bearing metadata object;
    # a missing instrument identity must therefore fail at metadata construction.
    with pytest.raises(ValidationError):
        MeasurementMetadata(
            instrument_id="",
            geometry="d/8",
            illuminant="D65",
            observer="2deg",
            specular_mode="SCI",
            aperture_mm=8.0,
            measured_at=NOW,
            data_origin=DataOrigin.SYNTHETIC,
        )
    assert invalid.metadata.instrument_id == "SPEC_SYN_1"


def test_optional_tolerance_does_not_invent_client_values():
    record = _record(tolerance=None)
    assert record.tolerance is None


def test_placeholder_tolerance_is_explicitly_labelled():
    record = _record(tolerance=_placeholder_tolerance())
    assert record.tolerance is not None
    assert record.tolerance.is_placeholder is True
    assert "synthetic" in record.tolerance.source


def test_client_placeholder_tolerance_is_rejected():
    with pytest.raises(ValidationError, match="client tolerance cannot be a placeholder"):
        _record(tolerance=_placeholder_tolerance(), data_origin=DataOrigin.CLIENT)


def test_registry_retrieval_is_deterministic():
    registry = GoldenMasterRegistry()
    record = _record()
    registry.register(record)
    assert registry.get("M_SYN_01", 1) == registry.get("M_SYN_01", 1)
    assert registry.list_versions(master_id="M_SYN_01") == (record,)


def test_repeated_registration_retrieval_is_deterministic():
    first = GoldenMasterRegistry()
    second = GoldenMasterRegistry()
    record = _record()
    first.register(record)
    second.register(record)
    assert first.get("M_SYN_01", 1) == second.get("M_SYN_01", 1)
    assert first.list_versions() == second.list_versions()


def test_existing_records_remain_unchanged_after_new_version():
    registry = GoldenMasterRegistry()
    first = _record()
    registry.register(first)
    before = first.model_dump()
    registry.register(
    _record(
        version=2,
        content=CONTENT_B,
        parent_version=1,
        parent_master_id="M_SYN_01",
    )
)
    assert first.model_dump() == before
    assert registry.get("M_SYN_01", 1).model_dump() == before


def test_registry_does_not_modify_master_content():
    content = bytearray(CONTENT_A)
    original = bytes(content)
    record = _record(content=content)
    registry = GoldenMasterRegistry()
    registry.register(record)
    assert bytes(content) == original
    assert registry.get("M_SYN_01", 1).content_bytes == original


def test_invalid_sha256_is_rejected():
    with pytest.raises(ValidationError):
        GoldenMasterRecord(
            **_record().model_dump(exclude={"content_hash"}),
            content_hash="not-a-sha256",
        )
