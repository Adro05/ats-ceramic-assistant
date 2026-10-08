"""Deterministic confidence and out-of-domain gate for optimizer recommendations.

Pipeline position::

    constrained optimizer -> confidence / OOD -> human QC approval

This module is a trustworthiness gate only.  It does not decide client colour
acceptance, mutate corrections, release production, or update the response model.
It consumes the existing Bayesian response-model uncertainty, optimizer result,
and validated operating envelope.
"""

from collections.abc import Iterable
from enum import StrEnum
from math import isfinite
from typing import Self

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from ats_ceramic.calibration_workflow import ControllableInput, ValidatedOperatingEnvelope
from ats_ceramic.optimizer import (
    OptimizationResult,
    OptimizationTarget,
    OptimizerStatus,
)
from ats_ceramic.response_model import ResponseModel
from ats_ceramic.schemas import (
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    PositiveFinite,
)


class ConfidenceStage(StrEnum):
    """Checkpoint at which confidence was assessed."""

    PREFLIGHT = "preflight"
    POST_SOLVE = "post_solve"


class ConfidenceStatus(StrEnum):
    """Trustworthiness tier for human QC review."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class OODStatus(StrEnum):
    """Deterministic position relative to a validated operating domain."""

    IN_DOMAIN = "in_domain"
    NEAR_BOUNDARY = "near_boundary"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class ModelFreshness(StrEnum):
    """Freshness state supported by available repository metadata."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class ConfidenceReason(StrEnum):
    """Auditable reasons contributing to a confidence tier."""

    VALIDATED_DOMAIN = "validated_domain"
    NEAR_DOMAIN_BOUNDARY = "near_domain_boundary"
    OUTSIDE_OPERATING_ENVELOPE = "outside_operating_envelope"
    MISSING_OPERATING_ENVELOPE = "missing_operating_envelope"
    UNVALIDATED_OPERATING_ENVELOPE = "unvalidated_operating_envelope"
    MODEL_CONTEXT_MISMATCH = "model_context_mismatch"
    REGION_MISMATCH = "region_mismatch"
    SKU_MISMATCH = "sku_mismatch"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    CHANNEL_MISMATCH = "channel_mismatch"
    INSUFFICIENT_MODEL_SUPPORT = "insufficient_model_support"
    HIGH_PREDICTION_UNCERTAINTY = "high_prediction_uncertainty"
    ELEVATED_PREDICTION_UNCERTAINTY = "elevated_prediction_uncertainty"
    INVALID_PREDICTION = "invalid_prediction"
    OPTIMIZER_FAILURE = "optimizer_failure"
    NO_RECOMMENDATION = "no_recommendation"
    NO_MEANINGFUL_IMPROVEMENT = "no_meaningful_improvement"
    TRUST_REGION_VIOLATION = "trust_region_violation"
    HARD_BOUND_VIOLATION = "hard_bound_violation"
    MODEL_FRESHNESS_UNKNOWN = "model_freshness_unknown"
    OPERATING_ENVELOPE_MISMATCH = "operating_envelope_mismatch"
    VALID_MODEL_SUPPORT = "valid_model_support"


class ConfidenceSpec(FrozenModel):
    """Explicit prototype confidence thresholds, not client colour tolerances.

    ``near_boundary_fraction`` is the fraction of an envelope span treated as the
    boundary band.  The two uncertainty thresholds apply to the Euclidean norm of
    the existing L/a/b predictive standard deviations from ``SensitivityPrediction``.
    They are prototype trust thresholds and must not be interpreted as client
    acceptance limits.
    """

    near_boundary_fraction: float = Field(gt=0.0, lt=0.5)
    uncertainty_warn_norm: PositiveFinite
    uncertainty_red_norm: PositiveFinite
    provenance: str = Field(min_length=1)
    is_placeholder: bool

    @model_validator(mode="after")
    def _uncertainty_order(self) -> Self:
        if self.uncertainty_warn_norm >= self.uncertainty_red_norm:
            raise ValueError("uncertainty_warn_norm must be below uncertainty_red_norm")
        return self


class ConfidenceAssessment(FrozenModel):
    """Immutable, auditable trustworthiness assessment for human QC review."""

    stage: ConfidenceStage
    status: ConfidenceStatus
    ood_status: OODStatus
    model_freshness: ModelFreshness
    reason_codes: tuple[ConfidenceReason, ...] = Field(min_length=1)
    prediction_uncertainty: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    uncertainty_norm: FiniteFloat | None = None
    operating_envelope_status: OODStatus
    model_support_observations: int
    model_min_observations: int
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _uncertainty_shape(self) -> Self:
        if (self.prediction_uncertainty is None) != (self.uncertainty_norm is None):
            raise ValueError(
                "prediction uncertainty and uncertainty_norm must be supplied together"
            )
        return self


def _reason_once(reasons: list[ConfidenceReason], reason: ConfidenceReason) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _finite_uncertainty(values: tuple[float, float, float]) -> bool:
    return all(isfinite(value) and value > 0.0 for value in values)


def _uncertainty_values(result: OptimizationResult) -> tuple[float, float, float] | None:
    values = (
        result.candidate.prediction.uncertainty_std_L,
        result.candidate.prediction.uncertainty_std_a,
        result.candidate.prediction.uncertainty_std_b,
    )
    return values if _finite_uncertainty(values) else None


def _uncertainty_norm(values: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


def _envelope_matches(
    envelope: ValidatedOperatingEnvelope,
    model: ResponseModel,
    channel_ids: tuple[str, ...],
) -> bool:
    return (
        envelope.sku_id == model.sku_id
        and envelope.context_id == model.context_id
        and envelope.region_id == model.region_id
        and envelope.channel_ids == channel_ids
        and envelope.data_origin is model.data_origin
    )


def _envelope_position(
    envelope: ValidatedOperatingEnvelope | None,
    adjustments: tuple[ChannelAdjustment, ...],
    spec: ConfidenceSpec,
) -> OODStatus:
    if envelope is None:
        return OODStatus.UNKNOWN
    if envelope.validation_status.value != "validated":
        return OODStatus.UNKNOWN
    values = {item.channel_id: item.delta for item in adjustments}
    if set(values) - set(envelope.channel_ids):
        return OODStatus.OUT_OF_DOMAIN

    boundary = False
    for index, channel_id in enumerate(envelope.channel_ids):
        value = values.get(channel_id, 0.0)
        lower = envelope.minimum_adjustment[index]
        upper = envelope.maximum_adjustment[index]
        if value < lower or value > upper:
            return OODStatus.OUT_OF_DOMAIN
        span = upper - lower
        if span == 0.0:
            boundary = True
            continue
        margin = min(value - lower, upper - value) / span
        if margin <= spec.near_boundary_fraction:
            boundary = True
    return OODStatus.NEAR_BOUNDARY if boundary else OODStatus.IN_DOMAIN


def _freshness(envelope: ValidatedOperatingEnvelope | None) -> ModelFreshness:
    # ResponseModel has no timestamp or freshness policy.  An envelope timestamp is
    # calibration provenance, not enough evidence to declare a model current/stale.
    if envelope is None:
        return ModelFreshness.UNKNOWN
    return ModelFreshness.UNKNOWN


def preflight_confidence(
    model: ResponseModel,
    target: OptimizationTarget,
    controllable_inputs: Iterable[ControllableInput],
    spec: ConfidenceSpec,
    *,
    expected_sku_id: str | None = None,
    expected_context_id: str | None = None,
    channel_ids: tuple[str, ...] | None = None,
    current_adjustment: Iterable[ChannelAdjustment] = (),
    operating_envelope: ValidatedOperatingEnvelope | None = None,
) -> ConfidenceAssessment:
    """Assess whether a request is sufficiently supported before solving."""
    inputs = tuple(controllable_inputs)
    reasons: list[ConfidenceReason] = []
    selected = (
        tuple(channel_ids)
        if channel_ids is not None
        else tuple(
            channel_id
            for channel_id in model.channel_ids
            if any(
                item.channel_id == channel_id and item.controllable and item.client_confirmed
                for item in inputs
            )
        )
    )
    status = ConfidenceStatus.GREEN
    ood_status = OODStatus.UNKNOWN

    if target.data_origin is not model.data_origin:
        _reason_once(reasons, ConfidenceReason.PROVENANCE_MISMATCH)
        status = ConfidenceStatus.RED
    if expected_sku_id is not None and expected_sku_id != model.sku_id:
        _reason_once(reasons, ConfidenceReason.SKU_MISMATCH)
        status = ConfidenceStatus.RED
    if expected_context_id is not None and expected_context_id != model.context_id:
        _reason_once(reasons, ConfidenceReason.MODEL_CONTEXT_MISMATCH)
        status = ConfidenceStatus.RED
    if target.master.region_id != model.region_id or target.batch.region_id != model.region_id:
        _reason_once(reasons, ConfidenceReason.REGION_MISMATCH)
        status = ConfidenceStatus.RED
    if model.n_observations < model.spec.min_observations:
        _reason_once(reasons, ConfidenceReason.INSUFFICIENT_MODEL_SUPPORT)
        status = ConfidenceStatus.RED

    model_channels = set(model.channel_ids)
    for channel_id in selected:
        item = next((value for value in inputs if value.channel_id == channel_id), None)
        if (
            channel_id not in model_channels
            or item is None
            or not item.controllable
            or not item.client_confirmed
        ):
            _reason_once(reasons, ConfidenceReason.CHANNEL_MISMATCH)
            status = ConfidenceStatus.RED
    if not selected:
        _reason_once(reasons, ConfidenceReason.CHANNEL_MISMATCH)
        status = ConfidenceStatus.RED

    current = tuple(current_adjustment)
    current_map = {item.channel_id: item.delta for item in current}
    input_by_id = {item.channel_id: item for item in inputs}
    for channel_id in selected:
        item = input_by_id.get(channel_id)
        if item is None or item.lower_bound is None or item.upper_bound is None:
            _reason_once(reasons, ConfidenceReason.HARD_BOUND_VIOLATION)
            status = ConfidenceStatus.RED
            continue
        value = current_map.get(channel_id, 0.0)
        if value < item.lower_bound or value > item.upper_bound:
            _reason_once(reasons, ConfidenceReason.HARD_BOUND_VIOLATION)
            status = ConfidenceStatus.RED

    if operating_envelope is not None and not _envelope_matches(
        operating_envelope, model, selected
    ):
        _reason_once(reasons, ConfidenceReason.OPERATING_ENVELOPE_MISMATCH)
        status = ConfidenceStatus.RED
    ood_status = _envelope_position(operating_envelope, current, spec)
    if operating_envelope is None:
        _reason_once(reasons, ConfidenceReason.MISSING_OPERATING_ENVELOPE)
        if status is ConfidenceStatus.GREEN:
            status = ConfidenceStatus.AMBER
    elif operating_envelope.validation_status.value != "validated":
        _reason_once(reasons, ConfidenceReason.UNVALIDATED_OPERATING_ENVELOPE)
        status = ConfidenceStatus.RED
    elif ood_status is OODStatus.OUT_OF_DOMAIN:
        _reason_once(reasons, ConfidenceReason.OUTSIDE_OPERATING_ENVELOPE)
        status = ConfidenceStatus.RED
    elif ood_status is OODStatus.NEAR_BOUNDARY:
        _reason_once(reasons, ConfidenceReason.NEAR_DOMAIN_BOUNDARY)
        if status is ConfidenceStatus.GREEN:
            status = ConfidenceStatus.AMBER
    else:
        _reason_once(reasons, ConfidenceReason.VALIDATED_DOMAIN)

    if status is ConfidenceStatus.GREEN and not reasons:
        _reason_once(reasons, ConfidenceReason.VALID_MODEL_SUPPORT)
    _reason_once(reasons, ConfidenceReason.MODEL_FRESHNESS_UNKNOWN)
    if not reasons:
        _reason_once(reasons, ConfidenceReason.VALID_MODEL_SUPPORT)
    return ConfidenceAssessment(
        stage=ConfidenceStage.PREFLIGHT,
        status=status,
        ood_status=ood_status,
        model_freshness=_freshness(operating_envelope),
        reason_codes=tuple(reasons),
        prediction_uncertainty=None,
        uncertainty_norm=None,
        operating_envelope_status=ood_status,
        model_support_observations=model.n_observations,
        model_min_observations=model.spec.min_observations,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=selected,
        data_origin=model.data_origin,
        provenance=model.spec.source,
    )


def post_solve_confidence(
    model: ResponseModel,
    target: OptimizationTarget,
    result: OptimizationResult,
    controllable_inputs: Iterable[ControllableInput],
    spec: ConfidenceSpec,
    *,
    operating_envelope: ValidatedOperatingEnvelope | None = None,
) -> ConfidenceAssessment:
    """Assess the trustworthiness of an already-computed optimizer result."""
    reasons: list[ConfidenceReason] = []
    status = ConfidenceStatus.GREEN
    selected = result.channel_ids

    if result.sku_id != model.sku_id:
        _reason_once(reasons, ConfidenceReason.SKU_MISMATCH)
        status = ConfidenceStatus.RED
    if result.context_id != model.context_id:
        _reason_once(reasons, ConfidenceReason.MODEL_CONTEXT_MISMATCH)
        status = ConfidenceStatus.RED
    if result.region_id != model.region_id or result.region_id != target.batch.region_id:
        _reason_once(reasons, ConfidenceReason.REGION_MISMATCH)
        status = ConfidenceStatus.RED
    if result.data_origin is not model.data_origin or target.data_origin is not model.data_origin:
        _reason_once(reasons, ConfidenceReason.PROVENANCE_MISMATCH)
        status = ConfidenceStatus.RED
    if model.n_observations < model.spec.min_observations:
        _reason_once(reasons, ConfidenceReason.INSUFFICIENT_MODEL_SUPPORT)
        status = ConfidenceStatus.RED

    input_by_id = {item.channel_id: item for item in controllable_inputs}
    for channel_id in selected:
        item = input_by_id.get(channel_id)
        if (
            channel_id not in model.channel_ids
            or item is None
            or not item.controllable
            or not item.client_confirmed
        ):
            _reason_once(reasons, ConfidenceReason.CHANNEL_MISMATCH)
            status = ConfidenceStatus.RED

    uncertainty = _uncertainty_values(result)
    if uncertainty is None:
        _reason_once(reasons, ConfidenceReason.INVALID_PREDICTION)
        status = ConfidenceStatus.RED
        uncertainty_norm = None
    else:
        uncertainty_norm = _uncertainty_norm(uncertainty)
        if uncertainty_norm >= spec.uncertainty_red_norm:
            _reason_once(reasons, ConfidenceReason.HIGH_PREDICTION_UNCERTAINTY)
            status = ConfidenceStatus.RED
        elif uncertainty_norm >= spec.uncertainty_warn_norm:
            _reason_once(reasons, ConfidenceReason.ELEVATED_PREDICTION_UNCERTAINTY)
            if status is ConfidenceStatus.GREEN:
                status = ConfidenceStatus.AMBER

    if result.status is not OptimizerStatus.RECOMMEND:
        if result.status is OptimizerStatus.NO_CHANGE:
            _reason_once(reasons, ConfidenceReason.NO_MEANINGFUL_IMPROVEMENT)
        elif result.status is OptimizerStatus.NUMERICAL_FAILURE:
            _reason_once(reasons, ConfidenceReason.OPTIMIZER_FAILURE)
        _reason_once(reasons, ConfidenceReason.NO_RECOMMENDATION)
        status = ConfidenceStatus.RED
    elif not result.solver_success:
        _reason_once(reasons, ConfidenceReason.OPTIMIZER_FAILURE)
        status = ConfidenceStatus.RED
    else:
        proposed = result.proposed_total_adjustment
        current = result.current_adjustment
        current_map = {item.channel_id: item.delta for item in current}
        proposed_map = {item.channel_id: item.delta for item in proposed}
        incremental = np.asarray(
            [
                proposed_map.get(channel_id, 0.0) - current_map.get(channel_id, 0.0)
                for channel_id in selected
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(incremental))
            or float(np.linalg.norm(incremental)) > result.objective_spec.trust_region_radius + 1e-8
        ):
            _reason_once(reasons, ConfidenceReason.TRUST_REGION_VIOLATION)
            status = ConfidenceStatus.RED

        for channel_id in selected:
            item = input_by_id.get(channel_id)
            if item is None or item.lower_bound is None or item.upper_bound is None:
                _reason_once(reasons, ConfidenceReason.HARD_BOUND_VIOLATION)
                status = ConfidenceStatus.RED
                continue
            value = proposed_map.get(channel_id, 0.0)
            if value < item.lower_bound - 1e-8 or value > item.upper_bound + 1e-8:
                _reason_once(reasons, ConfidenceReason.HARD_BOUND_VIOLATION)
                status = ConfidenceStatus.RED

    if operating_envelope is not None and not _envelope_matches(
        operating_envelope, model, selected
    ):
        _reason_once(reasons, ConfidenceReason.OPERATING_ENVELOPE_MISMATCH)
        status = ConfidenceStatus.RED
    ood_status = _envelope_position(operating_envelope, result.proposed_total_adjustment, spec)
    if operating_envelope is None:
        _reason_once(reasons, ConfidenceReason.MISSING_OPERATING_ENVELOPE)
        if status is ConfidenceStatus.GREEN:
            status = ConfidenceStatus.AMBER
    elif operating_envelope.validation_status.value != "validated":
        _reason_once(reasons, ConfidenceReason.UNVALIDATED_OPERATING_ENVELOPE)
        status = ConfidenceStatus.RED
    elif ood_status is OODStatus.OUT_OF_DOMAIN:
        _reason_once(reasons, ConfidenceReason.OUTSIDE_OPERATING_ENVELOPE)
        status = ConfidenceStatus.RED
    elif ood_status is OODStatus.NEAR_BOUNDARY:
        _reason_once(reasons, ConfidenceReason.NEAR_DOMAIN_BOUNDARY)
        if status is ConfidenceStatus.GREEN:
            status = ConfidenceStatus.AMBER
    else:
        _reason_once(reasons, ConfidenceReason.VALIDATED_DOMAIN)

    _reason_once(reasons, ConfidenceReason.MODEL_FRESHNESS_UNKNOWN)
    return ConfidenceAssessment(
        stage=ConfidenceStage.POST_SOLVE,
        status=status,
        ood_status=ood_status,
        model_freshness=_freshness(operating_envelope),
        reason_codes=tuple(reasons),
        prediction_uncertainty=uncertainty,
        uncertainty_norm=uncertainty_norm,
        operating_envelope_status=ood_status,
        model_support_observations=model.n_observations,
        model_min_observations=model.spec.min_observations,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=selected,
        data_origin=model.data_origin,
        provenance=model.spec.source,
    )
