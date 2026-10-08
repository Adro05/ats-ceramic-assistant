"""Deterministic human-readable adapter for approved CorrectionSpec artifacts.

Pipeline position::

    CorrectionSpec -> human-readable instruction sheet

This module is a presentation/serialization boundary only. It does not mutate the
CorrectionSpec or Golden Master, perform optimization or confidence calculations,
make QC decisions, modify artwork, or interact with printer/RIP systems.
"""

from typing import Self

from pydantic import Field, model_validator

from ats_ceramic.correction import CorrectionSpec, QCDecision
from ats_ceramic.schemas import (
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
)


class CorrectionInstructionSheet(FrozenModel):
    """Immutable, deterministic human-readable representation of a CorrectionSpec."""

    correction_spec_id: Identifier
    batch_id: Identifier
    master_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    recommended_adjustments: tuple[ChannelAdjustment, ...] = Field(min_length=1)
    current_adjustment: tuple[ChannelAdjustment, ...]
    confidence_status: str
    ood_status: str
    qc_decision: QCDecision
    reviewer_id: Identifier
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)
    instruction_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if self.qc_decision is not QCDecision.APPROVE:
            raise ValueError("instruction sheet requires an approved QC decision")
        if self.channel_ids != tuple(
            item.channel_id for item in self.recommended_adjustments
        ):
            raise ValueError("channel_ids do not match recommended adjustments")
        return self


def _adjustment_lines(adjustments: tuple[ChannelAdjustment, ...]) -> str:
    return "\n".join(
        f"Channel {item.channel_id}: {item.delta}" for item in adjustments
    )


def _current_adjustment_lines(adjustments: tuple[ChannelAdjustment, ...]) -> str:
    if not adjustments:
        return "Not specified"
    return "\n".join(
        f"Channel {item.channel_id}: {item.delta}" for item in adjustments
    )


def create_correction_instruction_sheet(
    correction_spec: CorrectionSpec,
) -> CorrectionInstructionSheet:
    """Adapt an approved CorrectionSpec into a deterministic operator instruction sheet."""
    if not isinstance(correction_spec, CorrectionSpec):
        raise ValueError("a valid CorrectionSpec is required")
    if correction_spec.qc_decision.decision is not QCDecision.APPROVE:
        raise ValueError("instruction sheet requires an approved QC decision")

    confidence = correction_spec.confidence_assessment
    recommended = tuple(correction_spec.recommended_adjustments)
    current = tuple(correction_spec.current_adjustment)
    channels = tuple(item.channel_id for item in recommended)

    instruction_text = "\n".join(
        (
            "CORRECTION INSTRUCTION SHEET",
            "============================",
            "",
            f"Correction Spec: {correction_spec.correction_spec_id}",
            f"Batch: {correction_spec.batch_id}",
            f"SKU: {correction_spec.sku_id}",
            f"Context: {correction_spec.context_id}",
            f"Region: {correction_spec.region_id}",
            "",
            "MASTER REFERENCE",
            "----------------",
            f"Master ID: {correction_spec.master_id}",
            "Golden Master status: UNCHANGED",
            "",
            "QC",
            "--",
            f"Decision: {correction_spec.qc_decision.decision.value.upper()}",
            f"Reviewer: {correction_spec.qc_decision.reviewer_id}",
            "",
            "RECOMMENDED BATCH CORRECTION",
            "----------------------------",
            _adjustment_lines(recommended),
            "",
            "CURRENT ADJUSTMENT",
            "------------------",
            _current_adjustment_lines(current),
            "",
            "CONFIDENCE",
            "----------",
            f"Status: {confidence.status.value}",
            f"OOD: {confidence.ood_status.value}",
            "",
            "PROVENANCE",
            "----------",
            f"Data origin: {correction_spec.data_origin.value}",
            f"Provenance: {correction_spec.provenance}",
            "",
            "OPERATIONAL NOTE",
            "----------------",
            "This recommendation is a batch-specific production correction.",
            "The Golden Master remains unchanged.",
            "This is not a modification of the canonical master.",
            "Apply the correction and use the existing verification workflow",
            "to assess the resulting batch measurement.",
        )
    )

    return CorrectionInstructionSheet(
        correction_spec_id=correction_spec.correction_spec_id,
        batch_id=correction_spec.batch_id,
        master_id=correction_spec.master_id,
        sku_id=correction_spec.sku_id,
        context_id=correction_spec.context_id,
        region_id=correction_spec.region_id,
        channel_ids=channels,
        recommended_adjustments=recommended,
        current_adjustment=current,
        confidence_status=confidence.status.value,
        ood_status=confidence.ood_status.value,
        qc_decision=correction_spec.qc_decision.decision,
        reviewer_id=correction_spec.qc_decision.reviewer_id,
        data_origin=correction_spec.data_origin,
        provenance=correction_spec.provenance,
        instruction_text=instruction_text,
    )