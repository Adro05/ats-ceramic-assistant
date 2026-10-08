"""Human QC approval and auditable correction specification.

Pipeline position::

    optimizer -> confidence / OOD -> human QC -> CorrectionSpec -> verification

This module is an approval and handoff boundary only.  It does not modify the
Golden Master, alter optimizer recommendations, release production, integrate with
RIP/printer systems, edit artwork, or update any model.
"""

from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from ats_ceramic.confidence import ConfidenceAssessment, ConfidenceStage, ConfidenceStatus
from ats_ceramic.optimizer import OptimizationResult, OptimizerStatus
from ats_ceramic.schemas import (
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    Sha256Hex,
)


class QCDecision(StrEnum):
    """Human QC decision on a recommendation."""

    APPROVE = "approve"
    REJECT = "reject"


class HumanQCDecision(FrozenModel):
    """Immutable, auditable record of a human QC decision."""

    decision_id: Identifier
    correction_spec_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    decision: QCDecision
    reason_comment: str | None = None
    reviewer_id: Identifier
    decided_at: AwareDatetime
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)


class CorrectionSpec(FrozenModel):
    """Approved, batch-specific correction handoff for downstream verification.

    The master relationship is represented by the existing correction-layer identity
    convention (batch id, master id, and master content hash).  The embedded
    optimizer result, confidence assessment, and QC decision preserve the exact
    recommendation and approval evidence; no master values are copied or modified.
    """

    correction_spec_id: Identifier
    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    master_id: Identifier
    master_content_hash: Sha256Hex
    recommended_adjustments: tuple[ChannelAdjustment, ...] = Field(min_length=1)
    current_adjustment: tuple[ChannelAdjustment, ...]
    optimizer_result: OptimizationResult
    confidence_assessment: ConfidenceAssessment
    qc_decision: HumanQCDecision
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_approval_record(self) -> Self:
        result = self.optimizer_result
        confidence = self.confidence_assessment
        decision = self.qc_decision

        if decision.decision is not QCDecision.APPROVE:
            raise ValueError("CorrectionSpec requires an approved QC decision")
        if result.status is not OptimizerStatus.RECOMMEND:
            raise ValueError("CorrectionSpec requires an optimizer recommendation")
        if confidence.stage is not ConfidenceStage.POST_SOLVE:
            raise ValueError("CorrectionSpec requires a post-solve confidence assessment")
        if confidence.status is ConfidenceStatus.RED:
            raise ValueError("RED confidence cannot produce a CorrectionSpec")
        if decision.correction_spec_id != self.correction_spec_id:
            raise ValueError("QC decision does not match correction_spec_id")
        if decision.sku_id != self.sku_id or result.sku_id != self.sku_id:
            raise ValueError("SKU does not match across approval records")
        if decision.context_id != self.context_id or result.context_id != self.context_id:
            raise ValueError("context does not match across approval records")
        if decision.region_id != self.region_id or result.region_id != self.region_id:
            raise ValueError("region does not match across approval records")
        if confidence.sku_id != self.sku_id:
            raise ValueError("confidence SKU does not match")
        if confidence.context_id != self.context_id:
            raise ValueError("confidence context does not match")
        if confidence.region_id != self.region_id:
            raise ValueError("confidence region does not match")
        if confidence.provenance != self.provenance:
            raise ValueError("CorrectionSpec provenance does not match confidence provenance")
        if result.data_origin is not self.data_origin:
            raise ValueError("optimizer data origin does not match CorrectionSpec")
        if confidence.data_origin is not self.data_origin:
            raise ValueError("confidence data origin does not match CorrectionSpec")
        if decision.data_origin is not self.data_origin:
            raise ValueError("QC data origin does not match CorrectionSpec")
        if result.channel_ids != confidence.channel_ids:
            raise ValueError("optimizer and confidence channels do not match")
        if tuple(self.recommended_adjustments) != result.recommended_adjustments:
            raise ValueError("recommended adjustments do not match optimizer result")
        if tuple(self.current_adjustment) != result.current_adjustment:
            raise ValueError("current adjustment does not match optimizer result")
        return self


def validate_qc_decision(
    result: OptimizationResult,
    confidence: ConfidenceAssessment,
    decision: HumanQCDecision,
) -> None:
    """Validate a human QC decision against the recommendation and confidence gate.

    Rejection remains recordable even when the recommendation is invalid or RED.
    Approval is allowed for GREEN or AMBER only; no QC override path exists for RED.
    """
    if decision.sku_id != result.sku_id or decision.sku_id != confidence.sku_id:
        raise ValueError("SKU mismatch between QC decision and recommendation")
    if decision.context_id != result.context_id or decision.context_id != confidence.context_id:
        raise ValueError("context mismatch between QC decision and recommendation")
    if decision.region_id != result.region_id or decision.region_id != confidence.region_id:
        raise ValueError("region mismatch between QC decision and recommendation")
    if (
        decision.data_origin is not result.data_origin
        or decision.data_origin is not confidence.data_origin
    ):
        raise ValueError("provenance/data origin mismatch")
    if confidence.stage is not ConfidenceStage.POST_SOLVE:
        raise ValueError("QC requires a post-solve confidence assessment")
    if decision.decision is QCDecision.APPROVE and decision.provenance != confidence.provenance:
        raise ValueError("provenance mismatch between QC decision and recommendation")
    if confidence.channel_ids != result.channel_ids:
        raise ValueError("channel mismatch between confidence and recommendation")

    if decision.decision is QCDecision.APPROVE:
        if result.status is not OptimizerStatus.RECOMMEND:
            raise ValueError("only a valid optimizer recommendation can be approved")
        if confidence.status is ConfidenceStatus.RED:
            raise ValueError("RED confidence cannot be approved")


def create_correction_spec(
    result: OptimizationResult,
    confidence: ConfidenceAssessment,
    decision: HumanQCDecision,
    *,
    batch_id: Identifier,
    master_id: Identifier,
    master_content_hash: Sha256Hex,
) -> CorrectionSpec | None:
    """Create an immutable CorrectionSpec after human QC approval.

    A rejection is an auditable terminal QC decision for this handoff and returns
    ``None`` rather than creating an executable correction artifact.
    """
    validate_qc_decision(result, confidence, decision)
    if decision.decision is QCDecision.REJECT:
        return None

    return CorrectionSpec(
        correction_spec_id=decision.correction_spec_id,
        batch_id=batch_id,
        sku_id=result.sku_id,
        context_id=result.context_id,
        region_id=result.region_id,
        master_id=master_id,
        master_content_hash=master_content_hash,
        recommended_adjustments=result.recommended_adjustments,
        current_adjustment=result.current_adjustment,
        optimizer_result=result,
        confidence_assessment=confidence,
        qc_decision=decision,
        data_origin=result.data_origin,
        provenance=confidence.provenance,
    )
