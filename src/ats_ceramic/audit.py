"""Immutable audit-event boundary for traceability across the correction pipeline.

The audit layer records references and structured scalar details only.  It does not
make decisions, mutate domain objects, rerun computation, or persist to an external
system.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, FiniteFloat, field_validator, model_validator

from ats_ceramic.schemas import DataOrigin, FrozenModel, Identifier


class AuditEventType(StrEnum):
    """Small, fixed taxonomy of important correction-pipeline boundaries."""

    MASTER_REGISTERED = "master_registered"
    MEASUREMENT_ACCEPTED = "measurement_accepted"
    MEASUREMENT_REJECTED = "measurement_rejected"
    TRIAGE_COMPLETED = "triage_completed"
    RESPONSE_MODEL_FIT = "response_model_fit"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    CONFIDENCE_ASSESSED = "confidence_assessed"
    QC_DECISION_RECORDED = "qc_decision_recorded"
    CORRECTION_SPEC_CREATED = "correction_spec_created"
    VERIFICATION_COMPLETED = "verification_completed"
    LEARNING_EVIDENCE_CREATED = "learning_evidence_created"


class AuditDetail(FrozenModel):
    """One named scalar detail attached to an audit event."""

    key: Identifier
    value: str | int | bool | FiniteFloat | None


class AuditEvent(FrozenModel):
    """Immutable trace record containing references, provenance, and scalar details."""

    event_id: Identifier
    event_type: AuditEventType
    occurred_at: AwareDatetime
    batch_id: Identifier | None = None
    sku_id: Identifier | None = None
    context_id: Identifier | None = None
    region_id: Identifier | None = None
    master_id: Identifier | None = None
    correction_spec_id: Identifier | None = None
    decision_id: Identifier | None = None
    verification_id: Identifier | None = None
    learning_evidence_id: Identifier | None = None
    model_id: Identifier | None = None
    model_version: Identifier | None = None
    configuration_id: Identifier | None = None
    configuration_version: Identifier | None = None
    solver_id: Identifier | None = None
    solver_version: Identifier | None = None
    status: Identifier | None = None
    result: Identifier | None = None
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)
    details: tuple[AuditDetail, ...] = ()

    @field_validator("occurred_at")
    @classmethod
    def _require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_provenance(self) -> Self:
        synthetic_prefix = "synthetic_generator:"
        if self.data_origin is DataOrigin.SYNTHETIC:
            if not self.provenance.startswith(synthetic_prefix):
                raise ValueError("synthetic audit provenance must identify the synthetic generator")
        elif self.provenance.startswith(synthetic_prefix):
            raise ValueError("client audit provenance cannot identify the synthetic generator")
        return self


class AuditLog:
    """Append-only in-memory audit boundary preserving insertion order."""

    def __init__(self) -> None:
        self._events: tuple[AuditEvent, ...] = ()
        self._event_ids: frozenset[str] = frozenset()

    def append(self, event: AuditEvent) -> None:
        """Append one immutable event, rejecting duplicate event IDs."""
        if not isinstance(event, AuditEvent):
            raise ValueError("append requires an AuditEvent")
        if event.event_id in self._event_ids:
            raise ValueError("event_id is already present in the audit log")
        self._events = (*self._events, event)
        self._event_ids = self._event_ids | {event.event_id}

    def events(self) -> tuple[AuditEvent, ...]:
        """Return all events in append order as an immutable tuple."""
        return self._events

    def events_for_batch(self, batch_id: str) -> tuple[AuditEvent, ...]:
        """Return events referencing ``batch_id`` in append order."""
        return tuple(event for event in self._events if event.batch_id == batch_id)

    def events_for_correction_spec(self, correction_spec_id: str) -> tuple[AuditEvent, ...]:
        """Return events referencing ``correction_spec_id`` in append order."""
        return tuple(
            event for event in self._events if event.correction_spec_id == correction_spec_id
        )

    def events_for_master(self, master_id: str) -> tuple[AuditEvent, ...]:
        """Return events referencing ``master_id`` in append order."""
        return tuple(event for event in self._events if event.master_id == master_id)


def create_audit_event(
    *,
    event_id: str,
    event_type: AuditEventType,
    occurred_at: datetime,
    data_origin: DataOrigin,
    provenance: str,
    batch_id: str | None = None,
    sku_id: str | None = None,
    context_id: str | None = None,
    region_id: str | None = None,
    master_id: str | None = None,
    correction_spec_id: str | None = None,
    decision_id: str | None = None,
    verification_id: str | None = None,
    learning_evidence_id: str | None = None,
    model_id: str | None = None,
    model_version: str | None = None,
    configuration_id: str | None = None,
    configuration_version: str | None = None,
    solver_id: str | None = None,
    solver_version: str | None = None,
    status: str | None = None,
    result: str | None = None,
    details: tuple[AuditDetail, ...] = (),
) -> AuditEvent:
    """Create one deterministic audit event from explicitly supplied values.

    No timestamp, identifier, or version is generated by this factory.
    """
    return AuditEvent(
        event_id=event_id,
        event_type=event_type,
        occurred_at=occurred_at,
        batch_id=batch_id,
        sku_id=sku_id,
        context_id=context_id,
        region_id=region_id,
        master_id=master_id,
        correction_spec_id=correction_spec_id,
        decision_id=decision_id,
        verification_id=verification_id,
        learning_evidence_id=learning_evidence_id,
        model_id=model_id,
        model_version=model_version,
        configuration_id=configuration_id,
        configuration_version=configuration_version,
        solver_id=solver_id,
        solver_version=solver_version,
        status=status,
        result=result,
        data_origin=data_origin,
        provenance=provenance,
        details=details,
    )
