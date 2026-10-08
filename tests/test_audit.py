"""Focused tests for the immutable audit-log boundary."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ats_ceramic.audit import AuditDetail, AuditEvent, AuditEventType, AuditLog, create_audit_event
from ats_ceramic.schemas import DataOrigin

TIMESTAMP = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)


def _event(
    event_id: str,
    event_type: AuditEventType = AuditEventType.OPTIMIZATION_COMPLETED,
    *,
    batch_id: str | None = "B-SYN-1",
    master_id: str | None = "M-SYN-1",
    correction_spec_id: str | None = None,
) -> AuditEvent:
    return create_audit_event(
        event_id=event_id,
        event_type=event_type,
        occurred_at=TIMESTAMP,
        batch_id=batch_id,
        sku_id="SKU-SYN-1",
        context_id="CTX-SYN-1",
        region_id="R1",
        master_id=master_id,
        correction_spec_id=correction_spec_id,
        data_origin=DataOrigin.SYNTHETIC,
        provenance="synthetic_generator:v1:test",
        status="completed",
        result="recorded",
        details=(AuditDetail(key="objective_after", value=1.25),),
    )


def test_audit_event_is_immutable():
    event = _event("E1")
    with pytest.raises((ValidationError, TypeError)):
        event.status = "changed"


def test_required_fields_and_references_validate():
    event = _event("E1")
    assert event.event_id == "E1"
    assert event.event_type is AuditEventType.OPTIMIZATION_COMPLETED
    assert event.batch_id == "B-SYN-1"
    assert event.master_id == "M-SYN-1"


def test_timezone_aware_timestamp_is_accepted():
    assert _event("E1").occurred_at == TIMESTAMP


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        create_audit_event(
            event_id="E1",
            event_type=AuditEventType.MASTER_REGISTERED,
            occurred_at=datetime(2026, 10, 9),
            data_origin=DataOrigin.SYNTHETIC,
            provenance="synthetic_generator:v1:test",
        )


def test_factory_rejects_naive_timestamp_directly():
    with pytest.raises(ValidationError):
        create_audit_event(
            event_id="E1",
            event_type=AuditEventType.MASTER_REGISTERED,
            occurred_at=datetime(2026, 10, 9),
            data_origin=DataOrigin.SYNTHETIC,
            provenance="synthetic_generator:v1:test",
        )


def test_synthetic_event_requires_synthetic_origin():
    with pytest.raises(ValidationError, match="synthetic generator"):
        create_audit_event(
            event_id="E1",
            event_type=AuditEventType.MASTER_REGISTERED,
            occurred_at=TIMESTAMP,
            data_origin=DataOrigin.SYNTHETIC,
            provenance="client:source",
        )


def test_synthetic_provenance_is_preserved():
    event = _event("E1")
    assert event.data_origin is DataOrigin.SYNTHETIC
    assert event.provenance == "synthetic_generator:v1:test"


def test_client_provenance_cannot_be_synthetic_generator():
    with pytest.raises(ValidationError, match="cannot identify the synthetic generator"):
        create_audit_event(
            event_id="E1",
            event_type=AuditEventType.MEASUREMENT_ACCEPTED,
            occurred_at=TIMESTAMP,
            data_origin=DataOrigin.CLIENT,
            provenance="synthetic_generator:v1:test",
        )


def test_audit_log_append_works():
    log = AuditLog()
    event = _event("E1")
    log.append(event)
    assert log.events() == (event,)


def test_audit_log_preserves_append_order():
    log = AuditLog()
    first = _event("E1")
    second = _event("E2")
    log.append(first)
    log.append(second)
    assert log.events() == (first, second)


def test_duplicate_event_id_is_rejected():
    log = AuditLog()
    log.append(_event("E1"))
    with pytest.raises(ValueError, match="already present"):
        log.append(_event("E1", AuditEventType.QC_DECISION_RECORDED))


def test_events_returns_immutable_tuple():
    log = AuditLog()
    log.append(_event("E1"))
    events = log.events()
    assert isinstance(events, tuple)
    with pytest.raises(TypeError):
        events[0] = _event("E2")  # type: ignore[index]


def test_events_for_batch_filters_without_reordering():
    log = AuditLog()
    first = _event("E1", batch_id="B1")
    second = _event("E2", batch_id="B2")
    third = _event("E3", batch_id="B1")
    for event in (first, second, third):
        log.append(event)
    assert log.events_for_batch("B1") == (first, third)


def test_events_for_correction_spec_filters_correctly():
    log = AuditLog()
    first = _event("E1", correction_spec_id="CS1")
    second = _event("E2", correction_spec_id="CS2")
    third = _event("E3", correction_spec_id="CS1")
    for event in (first, second, third):
        log.append(event)
    assert log.events_for_correction_spec("CS1") == (first, third)


def test_events_for_master_filters_correctly():
    log = AuditLog()
    first = _event("E1", master_id="M1")
    second = _event("E2", master_id="M2")
    third = _event("E3", master_id="M1")
    for event in (first, second, third):
        log.append(event)
    assert log.events_for_master("M1") == (first, third)


def test_returned_events_cannot_mutate_stored_events():
    log = AuditLog()
    event = _event("E1")
    log.append(event)
    returned = log.events()
    with pytest.raises((ValidationError, TypeError)):
        returned[0].provenance = "changed"
    assert log.events() == (event,)


def test_factory_has_no_hidden_random_ids_or_timestamps():
    first = _event("E1")
    second = _event("E1")
    assert first == second
    assert first.event_id == "E1"
    assert first.occurred_at == TIMESTAMP


def test_audit_event_references_only_structured_scalar_details():
    event = _event("E1")
    assert event.details[0].key == "objective_after"
    assert event.details[0].value == 1.25
    assert not hasattr(event, "source_object")
    assert not hasattr(event, "correction_spec")


def test_equivalent_inputs_produce_equivalent_events():
    assert _event("E1") == _event("E1")


def test_correction_spec_audit_does_not_mutate_source():
    source = {
        "correction_spec_id": "CS-SYN-1",
        "batch_id": "B-SYN-1",
        "master_id": "M-SYN-1",
        "status": "approved",
    }
    before = source.copy()
    event = create_audit_event(
        event_id="E-CS-1",
        event_type=AuditEventType.CORRECTION_SPEC_CREATED,
        occurred_at=TIMESTAMP,
        batch_id=source["batch_id"],
        master_id=source["master_id"],
        correction_spec_id=source["correction_spec_id"],
        data_origin=DataOrigin.SYNTHETIC,
        provenance="synthetic_generator:v1:test",
        status=source["status"],
    )
    log = AuditLog()
    log.append(event)
    assert source == before
    assert log.events() == (event,)


def test_all_required_event_types_are_constrained():
    assert len(AuditEventType) == 11
    assert AuditEventType.LEARNING_EVIDENCE_CREATED.value == "learning_evidence_created"


def test_optional_model_and_solver_versions_are_absent_by_default():
    event = _event("E1")
    assert event.model_version is None
    assert event.configuration_version is None
    assert event.solver_version is None


def test_detail_float_must_be_finite():
    with pytest.raises(ValidationError):
        AuditDetail(key="bad", value=float("inf"))
