"""Bounded post-approval verification and learning evidence.

Pipeline position::

    CorrectionSpec -> verification measurement -> bounded assessment ->
    next correction / escalation -> gated learning evidence

This module is deliberately a verification boundary. It does not modify the Golden
Master, CorrectionSpec, QC decision, response model, artwork, printer/RIP settings,
or production state. Acceptance criteria are explicit caller-supplied configuration;
no client colour or gloss tolerance is invented here.
"""

from enum import StrEnum
from math import isfinite
from typing import Self

from pydantic import Field, FiniteFloat, model_validator

from ats_ceramic.calibration_workflow import ValidatedOperatingEnvelope
from ats_ceramic.color import ComparisonResult, compare_measurements
from ats_ceramic.correction import CorrectionSpec
from ats_ceramic.schemas import (
    AcceptanceMetric,
    BatchMeasurement,
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    MasterMeasurement,
    NonNegativeFinite,
    PositiveFinite,
    RegionMeasurement,
)


class VerificationStatus(StrEnum):
    """Deterministic outcome of one bounded verification assessment."""

    VERIFIED = "verified"
    ITERATION_ALLOWED = "iteration_allowed"
    ESCALATE = "escalate"
    NOT_ASSESSABLE = "not_assessable"


class VerificationReason(StrEnum):
    """Auditable verification decision reasons."""

    ACCEPTED = "accepted"
    COLOUR_OUTSIDE_CRITERION = "colour_outside_criterion"
    GLOSS_OUTSIDE_CRITERION = "gloss_outside_criterion"
    MISSING_CRITERION = "missing_criterion"
    INVALID_MEASUREMENT = "invalid_measurement"
    ITERATION_LIMIT_REACHED = "iteration_limit_reached"
    ITERATION_AVAILABLE = "iteration_available"
    BASELINE_NOT_AVAILABLE = "baseline_not_available"


class VerificationAcceptanceCriterion(FrozenModel):
    """Explicit colour and gloss acceptance criteria supplied by the caller.

    The thresholds are configuration, not client requirements. Synthetic prototype
    criteria must be labelled as placeholders; client criteria must be explicitly
    marked as CLIENT provenance and non-placeholder.
    """

    acceptance_metric: AcceptanceMetric = AcceptanceMetric.MAX_DE00
    delta_e00_threshold: PositiveFinite
    gloss_abs_threshold: NonNegativeFinite
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)
    is_placeholder: bool

    @model_validator(mode="after")
    def _provenance_policy(self) -> Self:
        if self.data_origin is DataOrigin.SYNTHETIC and not self.is_placeholder:
            raise ValueError("synthetic acceptance criteria must be marked as placeholders")
        if self.data_origin is DataOrigin.CLIENT and self.is_placeholder:
            raise ValueError("CLIENT acceptance criteria cannot be marked as placeholders")
        return self


class VerificationMeasurement(FrozenModel):
    """Immutable post-correction measurement and per-region comparison evidence."""

    correction_spec_id: Identifier
    master_id: Identifier
    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    master: MasterMeasurement
    measurement: BatchMeasurement
    comparison: ComparisonResult
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        measurement = self.measurement
        if measurement.batch_id != self.batch_id:
            raise ValueError("measurement.batch_id does not match verification batch_id")
        if measurement.sku_id != self.sku_id:
            raise ValueError("measurement.sku_id does not match verification SKU")
        if measurement.context_id != self.context_id:
            raise ValueError("measurement.context_id does not match verification context")
        if measurement.iteration < 1:
            raise ValueError("verification measurement must have iteration >= 1")
        if measurement.applied_correction_id != self.correction_spec_id:
            raise ValueError("measurement.applied_correction_id does not match correction spec")
        if measurement.metadata.data_origin is not self.data_origin:
            raise ValueError("measurement data origin does not match verification")
        if self.master.sku_id != self.sku_id:
            raise ValueError("master SKU does not match verification")
        if self.master.metadata.data_origin is not self.data_origin:
            raise ValueError("master data origin does not match verification")
        if self.region_id not in tuple(r.region_id for r in measurement.regions):
            raise ValueError("verification region is missing from measurement")
        try:
            self.comparison.region(self.region_id)
        except KeyError as exc:
            raise ValueError("verification region is missing from comparison") from exc
        if compare_measurements(self.master, measurement) != self.comparison:
            raise ValueError("verification comparison does not match master and measurement")
        return self


class VerificationAssessment(FrozenModel):
    """Immutable, per-region decision for one verification iteration."""

    correction_spec_id: Identifier
    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    iteration: int = Field(ge=1)
    status: VerificationStatus
    reason: VerificationReason
    region_delta_e00: NonNegativeFinite
    region_abs_gloss_delta: NonNegativeFinite
    improved: bool | None = None
    improvement_delta_e00: FiniteFloat | None = None
    max_iterations: int = Field(ge=1)
    criterion: VerificationAcceptanceCriterion | None
    verification: VerificationMeasurement
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.iteration != self.verification.measurement.iteration:
            raise ValueError("assessment iteration does not match verification measurement")
        if self.region_id != self.verification.region_id:
            raise ValueError("assessment region does not match verification region")
        if self.data_origin is not self.verification.data_origin:
            raise ValueError("assessment data origin does not match verification")
        if self.criterion is not None and self.criterion.data_origin is not self.data_origin:
            raise ValueError("acceptance criterion data origin does not match verification")
        if (
            self.status is VerificationStatus.VERIFIED
            and self.reason is not VerificationReason.ACCEPTED
        ):
            raise ValueError("verified assessment must have accepted reason")
        return self


class LearningEvidence(FrozenModel):
    """Immutable evidence for a future controlled response-model update.

    This record is evidence only. Creating it never mutates or replaces a
    ``ResponseModel``.
    """

    correction_spec_id: Identifier
    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    prior_adjustment: tuple[ChannelAdjustment, ...]
    current_adjustment: tuple[ChannelAdjustment, ...]
    observed_delta_L: FiniteFloat
    observed_delta_a: FiniteFloat
    observed_delta_b: FiniteFloat
    resulting_measurement: RegionMeasurement
    successful_verification: bool
    verification_iteration: int = Field(ge=1)
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.resulting_measurement.region_id != self.region_id:
            raise ValueError("learning evidence region does not match measurement")
        return self


class LocalSecantUpdate(FrozenModel):
    """Bounded one-channel local secant evidence proposal.

    A full multi-channel Broyden update is intentionally not attempted: one
    before/after observation cannot identify a unique multi-channel Jacobian. This
    proposal is therefore limited to a single changed channel and is not a model
    replacement.
    """

    correction_spec_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_id: Identifier
    input_delta: FiniteFloat
    observed_delta_L: FiniteFloat
    observed_delta_a: FiniteFloat
    observed_delta_b: FiniteFloat
    sensitivity_L_per_unit: FiniteFloat
    sensitivity_a_per_unit: FiniteFloat
    sensitivity_b_per_unit: FiniteFloat
    prior_adjustment: tuple[ChannelAdjustment, ...]
    proposed_adjustment: tuple[ChannelAdjustment, ...]
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)


def _validate_region_measurement(measurement: RegionMeasurement) -> None:
    values = (measurement.L, measurement.a, measurement.b, measurement.gloss)
    if not all(isfinite(value) for value in values):
        raise ValueError(VerificationReason.INVALID_MEASUREMENT.value)
    if not 0.0 <= measurement.L <= 100.0 or measurement.gloss < 0.0:
        raise ValueError(VerificationReason.INVALID_MEASUREMENT.value)


def _validate_master_and_batch_identity(
    correction_spec: CorrectionSpec,
    master: MasterMeasurement,
    measurement: BatchMeasurement,
    master_id: str,
    region_id: str,
) -> None:
    if master_id != correction_spec.master_id:
        raise ValueError("master identity does not match correction spec")
    if master.sku_id != correction_spec.sku_id or measurement.sku_id != correction_spec.sku_id:
        raise ValueError("SKU mismatch between correction spec and verification")
    if measurement.context_id != correction_spec.context_id:
        raise ValueError("context mismatch between correction spec and verification")
    if measurement.batch_id != correction_spec.batch_id:
        raise ValueError("batch mismatch between correction spec and verification")
    if correction_spec.data_origin is not measurement.metadata.data_origin:
        raise ValueError("data-origin mismatch between correction spec and verification")
    if correction_spec.data_origin is not master.metadata.data_origin:
        raise ValueError("data-origin mismatch between correction spec and master")
    if region_id != correction_spec.region_id:
        raise ValueError("region mismatch between correction spec and verification")
    if region_id not in tuple(r.region_id for r in master.regions):
        raise ValueError("verification region is missing from master")
    if region_id not in tuple(r.region_id for r in measurement.regions):
        raise ValueError("region mismatch between correction spec and verification")
    for region in master.regions:
        _validate_region_measurement(region)
    for region in measurement.regions:
        _validate_region_measurement(region)


def create_verification_measurement(
    correction_spec: CorrectionSpec,
    master: MasterMeasurement,
    measurement: BatchMeasurement,
    *,
    master_id: str,
    region_id: str,
    provenance: str,
) -> VerificationMeasurement:
    """Create deterministic post-correction measurement evidence for one region."""
    _validate_master_and_batch_identity(correction_spec, master, measurement, master_id, region_id)
    comparison = compare_measurements(master, measurement)
    return VerificationMeasurement(
        correction_spec_id=correction_spec.correction_spec_id,
        master_id=master_id,
        master=master,
        batch_id=measurement.batch_id,
        sku_id=measurement.sku_id,
        context_id=measurement.context_id,
        region_id=region_id,
        measurement=measurement,
        comparison=comparison,
        data_origin=measurement.metadata.data_origin,
        provenance=provenance,
    )


def assess_verification(
    correction_spec: CorrectionSpec,
    verification: VerificationMeasurement,
    criterion: VerificationAcceptanceCriterion | None,
    *,
    max_iterations: int,
    baseline_measurement: BatchMeasurement | None = None,
) -> VerificationAssessment:
    """Assess one verified region without changing any upstream record."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if verification.correction_spec_id != correction_spec.correction_spec_id:
        raise ValueError("correction spec identity mismatch")
    if verification.sku_id != correction_spec.sku_id:
        raise ValueError("SKU mismatch between correction spec and verification")
    if verification.context_id != correction_spec.context_id:
        raise ValueError("context mismatch between correction spec and verification")
    if verification.region_id != correction_spec.region_id:
        raise ValueError("region mismatch between correction spec and verification")
    if verification.data_origin is not correction_spec.data_origin:
        raise ValueError("data-origin mismatch between correction spec and verification")
    if verification.provenance != correction_spec.provenance:
        raise ValueError("provenance mismatch between correction spec and verification")
    if criterion is not None:
        if criterion.data_origin is not verification.data_origin:
            raise ValueError("acceptance criterion data origin mismatch")
        if criterion.provenance == "":
            raise ValueError("acceptance criterion provenance is required")

    difference = verification.comparison.region(verification.region_id)
    improved = None
    improvement = None
    if baseline_measurement is not None:
        if baseline_measurement.batch_id != verification.batch_id:
            raise ValueError("baseline batch does not match verification batch")
        if baseline_measurement.sku_id != verification.sku_id:
            raise ValueError("baseline SKU does not match verification")
        if baseline_measurement.context_id != verification.context_id:
            raise ValueError("baseline context does not match verification")
        if baseline_measurement.iteration >= verification.measurement.iteration:
            raise ValueError("baseline iteration must precede verification iteration")
        if baseline_measurement.metadata.data_origin is not verification.data_origin:
            raise ValueError("baseline data origin mismatch")
        baseline = compare_measurements(verification.master, baseline_measurement)
        baseline_difference = baseline.region(verification.region_id)
        improved = difference.delta_E00 < baseline_difference.delta_E00
        improvement = baseline_difference.delta_E00 - difference.delta_E00

    if criterion is None:
        status = VerificationStatus.NOT_ASSESSABLE
        reason = VerificationReason.MISSING_CRITERION
    else:
        colour_ok = difference.delta_E00 <= criterion.delta_e00_threshold
        gloss_ok = abs(difference.delta_gloss) <= criterion.gloss_abs_threshold
        if colour_ok and gloss_ok:
            status = VerificationStatus.VERIFIED
            reason = VerificationReason.ACCEPTED
        elif verification.measurement.iteration >= max_iterations:
            status = VerificationStatus.ESCALATE
            reason = (
                VerificationReason.GLOSS_OUTSIDE_CRITERION
                if colour_ok
                else VerificationReason.COLOUR_OUTSIDE_CRITERION
            )
        else:
            status = VerificationStatus.ITERATION_ALLOWED
            reason = (
                VerificationReason.GLOSS_OUTSIDE_CRITERION
                if colour_ok
                else VerificationReason.COLOUR_OUTSIDE_CRITERION
            )

    if (
        criterion is not None
        and status is not VerificationStatus.VERIFIED
        and verification.measurement.iteration >= max_iterations
        and reason
        in {
            VerificationReason.COLOUR_OUTSIDE_CRITERION,
            VerificationReason.GLOSS_OUTSIDE_CRITERION,
        }
    ):
        reason = VerificationReason.ITERATION_LIMIT_REACHED

    return VerificationAssessment(
        correction_spec_id=correction_spec.correction_spec_id,
        batch_id=verification.batch_id,
        sku_id=verification.sku_id,
        context_id=verification.context_id,
        region_id=verification.region_id,
        iteration=verification.measurement.iteration,
        status=status,
        reason=reason,
        region_delta_e00=difference.delta_E00,
        region_abs_gloss_delta=abs(difference.delta_gloss),
        improved=improved,
        improvement_delta_e00=improvement,
        max_iterations=max_iterations,
        criterion=criterion,
        verification=verification,
        data_origin=verification.data_origin,
        provenance=verification.provenance,
    )


def create_learning_evidence(
    correction_spec: CorrectionSpec,
    assessment: VerificationAssessment,
    before_measurement: BatchMeasurement,
) -> LearningEvidence:
    """Record successful before/after evidence without updating the response model."""
    if assessment.status is not VerificationStatus.VERIFIED:
        raise ValueError("learning evidence requires successful verification")
    verification = assessment.verification.measurement
    if before_measurement.batch_id != verification.batch_id:
        raise ValueError("before measurement batch mismatch")
    if before_measurement.sku_id != correction_spec.sku_id:
        raise ValueError("before measurement SKU mismatch")
    if before_measurement.context_id != correction_spec.context_id:
        raise ValueError("before measurement context mismatch")
    if before_measurement.iteration >= verification.iteration:
        raise ValueError("before measurement must precede verification")
    if before_measurement.metadata.data_origin is not correction_spec.data_origin:
        raise ValueError("before measurement data origin mismatch")

    before = next(
        (
            region
            for region in before_measurement.regions
            if region.region_id == assessment.region_id
        ),
        None,
    )
    after = next(
        (region for region in verification.regions if region.region_id == assessment.region_id),
        None,
    )
    if before is None or after is None:
        raise ValueError("learning evidence region is missing")

    return LearningEvidence(
        correction_spec_id=correction_spec.correction_spec_id,
        batch_id=verification.batch_id,
        sku_id=verification.sku_id,
        context_id=verification.context_id,
        region_id=assessment.region_id,
        channel_ids=tuple(item.channel_id for item in correction_spec.recommended_adjustments),
        prior_adjustment=correction_spec.current_adjustment,
        current_adjustment=correction_spec.optimizer_result.proposed_total_adjustment,
        observed_delta_L=after.L - before.L,
        observed_delta_a=after.a - before.a,
        observed_delta_b=after.b - before.b,
        resulting_measurement=after,
        successful_verification=True,
        verification_iteration=verification.iteration,
        data_origin=verification.metadata.data_origin,
        provenance=assessment.provenance,
    )


def propose_local_secant_update(
    correction_spec: CorrectionSpec,
    assessment: VerificationAssessment,
    before_measurement: BatchMeasurement,
    *,
    channel_id: str,
    operating_envelope: ValidatedOperatingEnvelope,
) -> LocalSecantUpdate:
    """Propose bounded one-channel secant evidence for future controlled learning."""
    if assessment.status is not VerificationStatus.VERIFIED:
        raise ValueError("secant evidence requires successful verification")
    if operating_envelope.validation_status.value != "validated":
        raise ValueError("operating envelope must be validated")
    if operating_envelope.sku_id != correction_spec.sku_id:
        raise ValueError("SKU mismatch with operating envelope")
    if operating_envelope.context_id != correction_spec.context_id:
        raise ValueError("context mismatch with operating envelope")
    if operating_envelope.region_id != correction_spec.region_id:
        raise ValueError("region mismatch with operating envelope")
    if operating_envelope.data_origin is not correction_spec.data_origin:
        raise ValueError("data-origin mismatch with operating envelope")
    if channel_id not in correction_spec.optimizer_result.channel_ids:
        raise ValueError("unsupported channel for secant evidence")
    if operating_envelope.channel_ids != correction_spec.optimizer_result.channel_ids:
        raise ValueError("operating envelope channels do not match correction")

    prior = {item.channel_id: item.delta for item in correction_spec.current_adjustment}
    recommended = {item.channel_id: item.delta for item in correction_spec.recommended_adjustments}
    known = set(correction_spec.optimizer_result.channel_ids)
    if set(prior) - known or set(recommended) - known:
        raise ValueError("secant adjustment contains unsupported channel")

    changed = tuple(
        channel
        for channel in correction_spec.optimizer_result.channel_ids
        if channel in recommended
    )
    if changed != (channel_id,):
        raise ValueError("local secant evidence requires exactly one changed channel")
    input_delta = recommended[channel_id]
    if input_delta == 0.0:
        raise ValueError("secant denominator cannot be zero")

    proposed = tuple(
        ChannelAdjustment(
            channel_id=channel,
            delta=prior.get(channel, 0.0) + recommended.get(channel, 0.0),
        )
        for channel in correction_spec.optimizer_result.channel_ids
    )
    if not operating_envelope.contains(proposed):
        raise ValueError("proposed secant update is outside operating envelope")

    verification_region = next(
        (
            region
            for region in assessment.verification.measurement.regions
            if region.region_id == assessment.region_id
        ),
        None,
    )
    before_region = next(
        (
            region
            for region in before_measurement.regions
            if region.region_id == assessment.region_id
        ),
        None,
    )
    if verification_region is None or before_region is None:
        raise ValueError("secant evidence region is missing")
    if before_measurement.metadata.data_origin is not correction_spec.data_origin:
        raise ValueError("before measurement data origin mismatch")
    if before_measurement.iteration >= assessment.iteration:
        raise ValueError("before measurement must precede verification")

    observed_l = verification_region.L - before_region.L
    observed_a = verification_region.a - before_region.a
    observed_b = verification_region.b - before_region.b
    return LocalSecantUpdate(
        correction_spec_id=correction_spec.correction_spec_id,
        sku_id=correction_spec.sku_id,
        context_id=correction_spec.context_id,
        region_id=assessment.region_id,
        channel_id=channel_id,
        input_delta=input_delta,
        observed_delta_L=observed_l,
        observed_delta_a=observed_a,
        observed_delta_b=observed_b,
        sensitivity_L_per_unit=observed_l / input_delta,
        sensitivity_a_per_unit=observed_a / input_delta,
        sensitivity_b_per_unit=observed_b / input_delta,
        prior_adjustment=correction_spec.current_adjustment,
        proposed_adjustment=proposed,
        data_origin=correction_spec.data_origin,
        provenance=assessment.provenance,
    )
