"""Deterministic bounded optimizer for the local ceramic printer response model.

Pipeline position::

    response model -> constrained optimizer -> confidence / recommendation

The optimizer is deliberately per-region.  It consumes the existing Bayesian
response model and never fits, updates, or extrapolates that model.  Printer
channels remain opaque identifiers; only inputs explicitly marked controllable
and client-confirmed by the calibration/discovery record may be optimized.

All objective weights, bounds, trust-region limits, and provenance are explicit
inputs.  No client acceptance tolerance is invented here.  The optimizer returns a
recommendation only; it never mutates a master, writes a correction file, or
releases production output.
"""

from collections.abc import Callable, Iterable
from enum import StrEnum
from math import sqrt
from typing import Self

import numpy as np
from pydantic import Field, FiniteFloat, PositiveFloat, model_validator
from scipy.optimize import OptimizeResult, minimize

from ats_ceramic.calibration_workflow import (
    ControllableInput,
    ValidatedOperatingEnvelope,
)
from ats_ceramic.color import delta_e_2000
from ats_ceramic.response_model import ResponseModel, SensitivityPrediction, predict_response
from ats_ceramic.schemas import (
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    RegionMeasurement,
    require_unique,
)


class OptimizerStatus(StrEnum):
    """Deterministic outcome of one regional optimization attempt."""

    RECOMMEND = "recommend"
    NO_CHANGE = "no_change"
    INFEASIBLE = "infeasible"
    NUMERICAL_FAILURE = "numerical_failure"


class OptimizerReason(StrEnum):
    """Auditable reason codes; these are not client acceptance thresholds."""

    IMPROVED = "improved"
    NO_MEANINGFUL_IMPROVEMENT = "no_meaningful_improvement"
    NO_CONTROLLABLE_CHANNELS = "no_controllable_channels"
    UNCONFIRMED_CHANNEL = "unconfirmed_channel"
    UNKNOWN_CHANNEL = "unknown_channel"
    MISSING_BOUNDS = "missing_bounds"
    INCONSISTENT_BOUNDS = "inconsistent_bounds"
    CURRENT_ADJUSTMENT_OUTSIDE_BOUNDS = "current_adjustment_outside_bounds"
    REGION_MISMATCH = "region_mismatch"
    SKU_MISMATCH = "sku_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    EMPTY_TARGET = "empty_target"
    INVALID_TARGET = "invalid_target"
    ENVELOPE_MISMATCH = "envelope_mismatch"
    OUTSIDE_OPERATING_ENVELOPE = "outside_operating_envelope"
    TRUST_REGION = "trust_region"
    MODEL_TARGET_PROVENANCE_MISMATCH = "model_target_provenance_mismatch"
    SOLVER_FAILURE = "solver_failure"


class OptimizerObjectiveSpec(FrozenModel):
    """Explicit objective weights and solve thresholds.

    The weights are prototype configuration, not client tolerances.  A correction
    is accepted only when it lowers the configured objective by at least
    ``minimum_objective_improvement`` relative to the do-nothing baseline.
    """

    delta_e_weight: PositiveFloat = 1.0
    correction_size_weight: FiniteFloat = 0.0
    uncertainty_weight: FiniteFloat = 0.0
    minimum_objective_improvement: FiniteFloat = 0.0
    trust_region_radius: PositiveFloat
    provenance: str = Field(min_length=1)
    is_placeholder: bool

    @model_validator(mode="after")
    def _non_negative_penalties(self) -> Self:
        if self.correction_size_weight < 0.0:
            raise ValueError("correction_size_weight must be >= 0")
        if self.uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be >= 0")
        if self.minimum_objective_improvement < 0.0:
            raise ValueError("minimum_objective_improvement must be >= 0")
        return self


class OptimizationTarget(FrozenModel):
    """Immutable master target and current measured batch for one named region."""

    master: RegionMeasurement
    batch: RegionMeasurement
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _matching_region(self) -> Self:
        if not self.master.region_id or not self.batch.region_id:
            raise ValueError("target region_id cannot be empty")
        if self.master.region_id != self.batch.region_id:
            raise ValueError("master and batch region_id must match")
        return self


class OptimizationEvaluation(FrozenModel):
    """Auditable objective evaluation for one candidate adjustment."""

    adjustments: tuple[ChannelAdjustment, ...]
    total_adjustment: tuple[ChannelAdjustment, ...]
    prediction: SensitivityPrediction
    predicted_L: FiniteFloat
    predicted_a: FiniteFloat
    predicted_b: FiniteFloat
    predicted_delta_E00: FiniteFloat
    correction_size_penalty: FiniteFloat
    uncertainty_penalty: FiniteFloat
    objective: FiniteFloat


class OptimizationResult(FrozenModel):
    """Recommendation-only result for one region; no production release is implied."""

    status: OptimizerStatus
    reason: OptimizerReason
    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    recommended_adjustments: tuple[ChannelAdjustment, ...]
    current_adjustment: tuple[ChannelAdjustment, ...]
    proposed_total_adjustment: tuple[ChannelAdjustment, ...]
    baseline: OptimizationEvaluation
    candidate: OptimizationEvaluation
    improvement: FiniteFloat
    solver_success: bool
    solver_message: str
    objective_spec: OptimizerObjectiveSpec
    provenance: str = Field(min_length=1)
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _candidate_consistency(self) -> Self:
        if self.status is OptimizerStatus.RECOMMEND and self.reason is not OptimizerReason.IMPROVED:
            raise ValueError("a recommendation must have an improved reason")
        if self.status is not OptimizerStatus.RECOMMEND and self.recommended_adjustments:
            raise ValueError("non-recommendation results cannot contain a correction")
        return self


Solver = Callable[..., OptimizeResult]


def _channel_map(inputs: Iterable[ControllableInput]) -> dict[str, ControllableInput]:
    values = tuple(inputs)
    require_unique((item.channel_id for item in values), "channel_id")
    return {item.channel_id: item for item in values}


def _adjustment_map(adjustments: Iterable[ChannelAdjustment], *, label: str) -> dict[str, float]:
    values = tuple(adjustments)
    require_unique((item.channel_id for item in values), f"{label} channel_id")
    return {item.channel_id: float(item.delta) for item in values}


def _ordered_adjustments(
    channel_ids: tuple[str, ...], values: np.ndarray
) -> tuple[ChannelAdjustment, ...]:
    return tuple(
        ChannelAdjustment(channel_id=channel_id, delta=float(values[index]))
        for index, channel_id in enumerate(channel_ids)
    )


def _validate_channel_contract(
    model: ResponseModel,
    controllable_inputs: tuple[ControllableInput, ...],
    requested_channel_ids: tuple[str, ...],
) -> tuple[dict[str, ControllableInput], str | None]:
    input_by_id = _channel_map(controllable_inputs)
    if not requested_channel_ids:
        return input_by_id, OptimizerReason.NO_CONTROLLABLE_CHANNELS.value
    require_unique(requested_channel_ids, "optimization channel_id")
    model_channels = set(model.channel_ids)
    for channel_id in requested_channel_ids:
        if channel_id not in model_channels:
            return input_by_id, f"{OptimizerReason.UNKNOWN_CHANNEL.value}:{channel_id}"
        item = input_by_id.get(channel_id)
        if item is None:
            return input_by_id, f"{OptimizerReason.UNKNOWN_CHANNEL.value}:{channel_id}"
        if not item.controllable or not item.client_confirmed:
            return input_by_id, f"{OptimizerReason.UNCONFIRMED_CHANNEL.value}:{channel_id}"
        if item.lower_bound is None or item.upper_bound is None:
            return input_by_id, f"{OptimizerReason.MISSING_BOUNDS.value}:{channel_id}"
        if item.lower_bound > item.upper_bound:
            return input_by_id, f"{OptimizerReason.INCONSISTENT_BOUNDS.value}:{channel_id}"
    return input_by_id, None


def _target_residual(
    target: OptimizationTarget, prediction: SensitivityPrediction
) -> tuple[float, float, float]:
    return (
        target.batch.L + prediction.predicted_delta_L,
        target.batch.a + prediction.predicted_delta_a,
        target.batch.b + prediction.predicted_delta_b,
    )


def _evaluate(
    *,
    model: ResponseModel,
    target: OptimizationTarget,
    current: np.ndarray,
    incremental: np.ndarray,
    channel_ids: tuple[str, ...],
    objective: OptimizerObjectiveSpec,
) -> OptimizationEvaluation:
    adjustment = _ordered_adjustments(channel_ids, incremental)
    total = current + incremental
    total_adjustment = _ordered_adjustments(channel_ids, total)
    prediction = predict_response(model, adjustment)
    predicted_lab = _target_residual(target, prediction)
    de00 = delta_e_2000(
        (target.master.L, target.master.a, target.master.b),
        predicted_lab,
    )
    correction_norm = float(np.linalg.norm(incremental))
    uncertainty_norm = sqrt(
        prediction.uncertainty_std_L**2
        + prediction.uncertainty_std_a**2
        + prediction.uncertainty_std_b**2
    )
    size_penalty = correction_norm**2
    uncertainty_penalty = uncertainty_norm
    objective_value = (
        objective.delta_e_weight * de00
        + objective.correction_size_weight * size_penalty
        + objective.uncertainty_weight * uncertainty_penalty
    )
    return OptimizationEvaluation(
        adjustments=adjustment,
        total_adjustment=total_adjustment,
        prediction=prediction,
        predicted_L=predicted_lab[0],
        predicted_a=predicted_lab[1],
        predicted_b=predicted_lab[2],
        predicted_delta_E00=de00,
        correction_size_penalty=size_penalty,
        uncertainty_penalty=uncertainty_penalty,
        objective=objective_value,
    )


def _zero_baseline(
    *,
    model: ResponseModel,
    target: OptimizationTarget,
    current: np.ndarray,
    channel_ids: tuple[str, ...],
    objective: OptimizerObjectiveSpec,
) -> OptimizationEvaluation:
    return _evaluate(
        model=model,
        target=target,
        current=current,
        incremental=np.zeros(len(channel_ids), dtype=float),
        channel_ids=channel_ids,
        objective=objective,
    )


def _envelope_bounds(
    envelope: ValidatedOperatingEnvelope | None,
    channel_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    if envelope is None:
        return None
    if envelope.validation_status.value != "validated":
        raise ValueError("operating envelope is not validated")
    if envelope.channel_ids != channel_ids:
        raise ValueError("operating envelope channel_ids must match optimization channel_ids")
    return (
        np.asarray(envelope.minimum_adjustment, dtype=float),
        np.asarray(envelope.maximum_adjustment, dtype=float),
    )


def optimize_region(
    model: ResponseModel,
    target: OptimizationTarget,
    controllable_inputs: Iterable[ControllableInput],
    objective: OptimizerObjectiveSpec,
    *,
    current_adjustment: Iterable[ChannelAdjustment] = (),
    channel_ids: tuple[str, ...] | None = None,
    operating_envelope: ValidatedOperatingEnvelope | None = None,
    provenance: str = "synthetic optimizer test configuration",
    solver: Solver = minimize,
) -> OptimizationResult:
    """Solve one named region with hard bounds and an explicit trust region.

    ``current_adjustment`` is the already-applied total correction relative to the
    response-model operating point.  The optimizer variable is the *incremental*
    correction to apply now.  Hard bounds and an optional empirical envelope apply
    to the proposed total adjustment, while the trust-region radius limits the
    incremental correction around the current point.

    The do-nothing baseline is always evaluated first.  A numerical candidate is
    returned only when it beats that baseline by the explicitly configured objective
    improvement.  No client colour tolerance is consulted or invented.
    """
    if not target.master.region_id or not target.batch.region_id:
        raise ValueError(OptimizerReason.EMPTY_TARGET.value)
    if target.master.region_id != model.region_id or target.batch.region_id != model.region_id:
        raise ValueError(OptimizerReason.REGION_MISMATCH.value)
    if model.sku_id == "" or model.context_id == "":
        raise ValueError(OptimizerReason.INVALID_TARGET.value)
    if target.data_origin is not model.data_origin:
        raise ValueError(OptimizerReason.MODEL_TARGET_PROVENANCE_MISMATCH.value)
    data_origin = model.data_origin

    inputs = tuple(controllable_inputs)
    input_by_id = _channel_map(inputs)
    if channel_ids is None:
        selected = tuple(
            channel_id
            for channel_id in model.channel_ids
            if (item := input_by_id.get(channel_id)) is not None
            and item.controllable
            and item.client_confirmed
        )
    else:
        selected = channel_ids
    input_by_id, contract_error = _validate_channel_contract(model, inputs, selected)
    if contract_error is not None:
        reason_text, _, detail = contract_error.partition(":")
        reason = OptimizerReason(reason_text)
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=reason,
            detail=detail,
            provenance=provenance,
            data_origin=data_origin,
        )

    current_map = _adjustment_map(current_adjustment, label="current")
    unknown_current = set(current_map) - set(model.channel_ids)
    if unknown_current:
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.UNKNOWN_CHANNEL,
            detail=sorted(unknown_current)[0],
            provenance=provenance,
            data_origin=data_origin,
        )
    current = np.asarray([current_map.get(channel_id, 0.0) for channel_id in selected], dtype=float)
    lower = np.asarray(
        [input_by_id[channel_id].lower_bound for channel_id in selected], dtype=float
    )
    upper = np.asarray(
        [input_by_id[channel_id].upper_bound for channel_id in selected], dtype=float
    )
    if np.any(current < lower) or np.any(current > upper):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.CURRENT_ADJUSTMENT_OUTSIDE_BOUNDS,
            detail="current adjustment is outside a hard channel bound",
            provenance=provenance,
            data_origin=data_origin,
        )

    try:
        envelope_bounds = _envelope_bounds(operating_envelope, selected)
    except ValueError as exc:
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.ENVELOPE_MISMATCH,
            detail=str(exc),
            provenance=provenance,
            data_origin=data_origin,
        )

    if envelope_bounds is not None:
        envelope_lower, envelope_upper = envelope_bounds
        if np.any(current < envelope_lower) or np.any(current > envelope_upper):
            return _infeasible_result(
                model=model,
                target=target,
                selected=selected,
                current_adjustment=tuple(current_adjustment),
                objective=objective,
                reason=OptimizerReason.OUTSIDE_OPERATING_ENVELOPE,
                detail="current adjustment is outside the supplied operating envelope",
                provenance=provenance,
                data_origin=data_origin,
            )
        lower = np.maximum(lower, envelope_lower)
        upper = np.minimum(upper, envelope_upper)

    if np.any(lower > upper):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.INCONSISTENT_BOUNDS,
            detail="hard and operating-envelope bounds have no intersection",
            provenance=provenance,
            data_origin=data_origin,
        )

    baseline = _zero_baseline(
        model=model,
        target=target,
        current=current,
        channel_ids=selected,
        objective=objective,
    )

    incremental_lower = lower - current
    incremental_upper = upper - current
    trust_radius = objective.trust_region_radius
    if np.any(incremental_lower > incremental_upper):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.TRUST_REGION,
            detail="no feasible incremental adjustment remains",
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
        )

    bounds = tuple(
        (
            max(float(incremental_lower[index]), -trust_radius),
            min(float(incremental_upper[index]), trust_radius),
        )
        for index in range(len(selected))
    )
    if any(lo > hi for lo, hi in bounds):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.TRUST_REGION,
            detail="trust-region and channel bounds have no intersection",
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
        )

    def objective_function(x: np.ndarray) -> float:
        evaluation = _evaluate(
            model=model,
            target=target,
            current=current,
            incremental=x,
            channel_ids=selected,
            objective=objective,
        )
        return evaluation.objective

    constraints = {
        "type": "ineq",
        "fun": lambda x: trust_radius**2 - float(np.dot(x, x)),
    }
    initial = np.zeros(len(selected), dtype=float)
    result = solver(
        objective_function,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    if not bool(getattr(result, "success", False)) or not hasattr(result, "x"):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.SOLVER_FAILURE,
            detail=str(getattr(result, "message", "optimizer did not converge")),
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
            status=OptimizerStatus.NUMERICAL_FAILURE,
        )

    candidate_x = np.asarray(result.x, dtype=float)
    if candidate_x.shape != initial.shape or not np.all(np.isfinite(candidate_x)):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.SOLVER_FAILURE,
            detail="optimizer returned a non-finite or incorrectly shaped solution",
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
            status=OptimizerStatus.NUMERICAL_FAILURE,
        )

    if float(np.linalg.norm(candidate_x)) > trust_radius + 1e-8:
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.TRUST_REGION,
            detail="solver returned a candidate outside the trust region",
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
        )

    proposed_total = current + candidate_x
    if np.any(proposed_total < lower - 1e-8) or np.any(proposed_total > upper + 1e-8):
        return _infeasible_result(
            model=model,
            target=target,
            selected=selected,
            current_adjustment=tuple(current_adjustment),
            objective=objective,
            reason=OptimizerReason.CURRENT_ADJUSTMENT_OUTSIDE_BOUNDS,
            detail="solver returned a candidate outside hard bounds",
            provenance=provenance,
            data_origin=data_origin,
            baseline=baseline,
        )
    if envelope_bounds is not None:
        envelope_lower, envelope_upper = envelope_bounds
        if np.any(proposed_total < envelope_lower - 1e-8) or np.any(
            proposed_total > envelope_upper + 1e-8
        ):
            return _infeasible_result(
                model=model,
                target=target,
                selected=selected,
                current_adjustment=tuple(current_adjustment),
                objective=objective,
                reason=OptimizerReason.OUTSIDE_OPERATING_ENVELOPE,
                detail="solver returned a candidate outside the operating envelope",
                provenance=provenance,
                data_origin=data_origin,
                baseline=baseline,
            )

    candidate = _evaluate(
        model=model,
        target=target,
        current=current,
        incremental=candidate_x,
        channel_ids=selected,
        objective=objective,
    )
    improvement = baseline.objective - candidate.objective
    if improvement < objective.minimum_objective_improvement:
        return OptimizationResult(
            status=OptimizerStatus.NO_CHANGE,
            reason=OptimizerReason.NO_MEANINGFUL_IMPROVEMENT,
            sku_id=model.sku_id,
            context_id=model.context_id,
            region_id=model.region_id,
            channel_ids=selected,
            recommended_adjustments=(),
            current_adjustment=baseline.total_adjustment,
            proposed_total_adjustment=baseline.total_adjustment,
            baseline=baseline,
            candidate=candidate,
            improvement=improvement,
            solver_success=True,
            solver_message=str(getattr(result, "message", "success")),
            objective_spec=objective,
            provenance=provenance,
            data_origin=data_origin,
        )

    return OptimizationResult(
        status=OptimizerStatus.RECOMMEND,
        reason=OptimizerReason.IMPROVED,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=selected,
        recommended_adjustments=candidate.adjustments,
        current_adjustment=baseline.total_adjustment,
        proposed_total_adjustment=candidate.total_adjustment,
        baseline=baseline,
        candidate=candidate,
        improvement=improvement,
        solver_success=True,
        solver_message=str(getattr(result, "message", "success")),
        objective_spec=objective,
        provenance=provenance,
        data_origin=data_origin,
    )


def _infeasible_result(
    *,
    model: ResponseModel,
    target: OptimizationTarget,
    selected: tuple[str, ...],
    current_adjustment: tuple[ChannelAdjustment, ...],
    objective: OptimizerObjectiveSpec,
    reason: OptimizerReason,
    detail: str,
    provenance: str,
    data_origin: DataOrigin,
    baseline: OptimizationEvaluation | None = None,
    status: OptimizerStatus = OptimizerStatus.INFEASIBLE,
) -> OptimizationResult:
    if not selected or any(channel_id not in model.channel_ids for channel_id in selected):
        selected = model.channel_ids
    current_map = _adjustment_map(current_adjustment, label="current")
    current = np.asarray([current_map.get(channel_id, 0.0) for channel_id in selected], dtype=float)
    if baseline is None:
        baseline = _zero_baseline(
            model=model,
            target=target,
            current=current,
            channel_ids=selected,
            objective=objective,
        )
    return OptimizationResult(
        status=status,
        reason=reason,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=selected,
        recommended_adjustments=(),
        current_adjustment=baseline.total_adjustment,
        proposed_total_adjustment=baseline.total_adjustment,
        baseline=baseline,
        candidate=baseline,
        improvement=0.0,
        solver_success=False,
        solver_message=detail,
        objective_spec=objective,
        provenance=provenance,
        data_origin=data_origin,
    )
