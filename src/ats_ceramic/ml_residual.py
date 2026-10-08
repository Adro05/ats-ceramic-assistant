"""Gated ML residual correction layered on the empirical response model.

Pipeline position::

    empirical response model -> optional residual -> constrained optimizer

The empirical Bayesian response model is always the primary prediction.  This module
can learn only the held-out residual around that prediction and can activate it only
when explicit data, provenance, validation, operating-envelope, and safety gates pass.
The module contains no production release, retraining, or autonomous decision logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from math import isfinite

import numpy as np
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor

from ats_ceramic.calibration_workflow import (
    CalibrationObservation,
    ValidatedOperatingEnvelope,
    split_held_out_by_run,
)
from ats_ceramic.response_model import ResponseModel, SensitivityPrediction, predict_response
from ats_ceramic.schemas import (
    ChannelAdjustment,
    DataOrigin,
    FiniteFloat,
    FrozenModel,
    Identifier,
    NonNegativeFinite,
    PositiveFinite,
)


class ResidualGateStatus(StrEnum):
    """Deterministic outcome of the optional residual gate."""

    ENABLED = "enabled"
    INSUFFICIENT_DATA = "insufficient_data"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    CHANNEL_MISMATCH = "channel_mismatch"
    SKU_CONTEXT_REGION_MISMATCH = "sku_context_region_mismatch"
    VALIDATION_FAILED = "validation_failed"
    OUT_OF_DOMAIN = "out_of_domain"
    RESIDUAL_TOO_LARGE = "residual_too_large"
    DISABLED = "disabled"


class ResidualConfig(FrozenModel):
    """Explicit engineering controls for the optional residual model.

    These are model-selection and safety settings, not client colour tolerances.
    """

    min_observations: int = Field(default=8, ge=2)
    min_relative_improvement: float = Field(default=0.05, ge=0.0)
    max_residual_abs: PositiveFinite = 0.75
    random_state: int = Field(default=0, ge=0)
    n_estimators: int = Field(default=32, ge=1, le=128)
    max_depth: int = Field(default=3, ge=1, le=8)
    min_samples_leaf: int = Field(default=2, ge=1)
    provenance: str = Field(min_length=1)


class AblationReport(FrozenModel):
    """Held-out comparison of empirical baseline versus empirical-plus-residual."""

    training_observation_count: int = Field(ge=0)
    held_out_observation_count: int = Field(ge=0)
    baseline_mse: NonNegativeFinite = 0.0
    residual_mse: NonNegativeFinite = 0.0
    absolute_improvement: FiniteFloat
    relative_improvement: FiniteFloat
    safety_bound_passed: bool
    acceptance_passed: bool
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)


class ResidualTrainingResult(FrozenModel):
    """Immutable public training outcome; no mutable estimator is exposed."""

    status: ResidualGateStatus
    model_ready: bool
    feature_names: tuple[Identifier, ...]
    training_observation_count: int = Field(ge=0)
    held_out_observation_count: int = Field(ge=0)
    ablation: AblationReport | None = None
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)


class ResidualPrediction(FrozenModel):
    """Immutable prediction produced by the gated residual layer."""

    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    empirical_prediction: tuple[float, float, float]
    residual_prediction: tuple[float, float, float]
    combined_prediction: tuple[float, float, float]
    residual_enabled: bool
    gate_status: ResidualGateStatus
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)


class ResidualModel:
    """Immutable-after-fit internal estimator boundary.

    The fitted sklearn estimators are deliberately private.  Callers receive the
    frozen training result and prediction records, not the mutable estimators.
    """

    __slots__ = (
        "__estimators",
        "__result",
        "__config",
        "__channel_ids",
        "__sku_id",
        "__context_id",
        "__region_id",
        "__data_origin",
        "__provenance",
        "__training_keys",
        "__frozen",
    )

    def __init__(
        self,
        estimators: tuple[
            RandomForestRegressor, RandomForestRegressor, RandomForestRegressor
        ] | None,
        result: ResidualTrainingResult,
        config: ResidualConfig,
        channel_ids: tuple[str, ...],
        sku_id: str,
        context_id: str,
        region_id: str,
        data_origin: DataOrigin,
        provenance: str,
        training_keys: frozenset[tuple[str, str]],
    ) -> None:
        object.__setattr__(self, "_ResidualModel__estimators", estimators)
        object.__setattr__(self, "_ResidualModel__result", result)
        object.__setattr__(self, "_ResidualModel__config", config)
        object.__setattr__(self, "_ResidualModel__channel_ids", channel_ids)
        object.__setattr__(self, "_ResidualModel__sku_id", sku_id)
        object.__setattr__(self, "_ResidualModel__context_id", context_id)
        object.__setattr__(self, "_ResidualModel__region_id", region_id)
        object.__setattr__(self, "_ResidualModel__data_origin", data_origin)
        object.__setattr__(self, "_ResidualModel__provenance", provenance)
        object.__setattr__(self, "_ResidualModel__training_keys", training_keys)
        object.__setattr__(self, "_ResidualModel__frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_ResidualModel__frozen", False):
            raise AttributeError("ResidualModel is immutable after fitting")
        object.__setattr__(self, name, value)

    @property
    def result(self) -> ResidualTrainingResult:
        return self.__result

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return self.__channel_ids

    def _predict_residual(self, features: tuple[float, ...]) -> tuple[float, float, float]:
        if self.__estimators is None or not self.result.model_ready:
            raise ValueError("residual model is not enabled")
        vector = np.asarray(features, dtype=float).reshape(1, -1)
        values = tuple(float(estimator.predict(vector)[0]) for estimator in self.__estimators)
        if not all(isfinite(value) for value in values):
            raise ValueError("residual model produced a non-finite prediction")
        return values

    def was_training_observation(self, observation: CalibrationObservation) -> bool:
        key = (observation.experiment_id, observation.production_run_id)
        return key in self.__training_keys


def _observed_delta(
    observation: CalibrationObservation, model: ResponseModel
) -> tuple[float, float, float]:
    comparison = observation.response_observation.comparison.region(model.region_id)
    return comparison.delta_L, comparison.delta_a, comparison.delta_b


def _features(
    observation: CalibrationObservation,
    model: ResponseModel,
) -> tuple[float, ...]:
    response = observation.response_observation
    adjustment_map = {item.channel_id: item.delta for item in response.adjustments}
    if set(adjustment_map) - set(model.channel_ids):
        raise ValueError("observation contains a channel outside the empirical model")
    adjustments = tuple(adjustment_map.get(channel_id, 0.0) for channel_id in model.channel_ids)
    empirical = predict_response(
        model,
        tuple(
            ChannelAdjustment(channel_id=cid, delta=value)
            for cid, value in zip(model.channel_ids, adjustments, strict=True)
        ),
    )
    return adjustments + (
        empirical.predicted_delta_L,
        empirical.predicted_delta_a,
        empirical.predicted_delta_b,
    )


def _residual_target(
    observation: CalibrationObservation,
    model: ResponseModel,
) -> tuple[float, float, float]:
    observed = _observed_delta(observation, model)
    empirical = _features(observation, model)[-3:]
    return tuple(observed[index] - empirical[index] for index in range(3))


def _validate_training_context(
    observations: tuple[CalibrationObservation, ...],
    model: ResponseModel,
    config: ResidualConfig,
) -> DataOrigin:
    if not observations:
        raise ValueError("at least one calibration observation is required")
    origin = observations[0].data_origin
    provenance = observations[0].provenance
    for observation in observations:
        response = observation.response_observation
        if observation.data_origin is not origin or response.data_origin is not origin:
            raise ValueError("calibration observations must use one data origin")
        if observation.provenance != provenance:
            raise ValueError("calibration observations must use one provenance")
        if response.sku_id != model.sku_id or response.context_id != model.context_id:
            raise ValueError(
                "calibration observations do not match the empirical model SKU/context"
            )
        if response.region_id != model.region_id:
            raise ValueError("calibration observations do not match the empirical model region")
        if set(item.channel_id for item in response.adjustments) - set(model.channel_ids):
            raise ValueError("calibration observation channel IDs do not match the empirical model")
    if provenance != config.provenance:
        raise ValueError("calibration provenance does not match residual configuration provenance")
    if model.data_origin is not origin:
        raise ValueError("empirical model and residual observations must use one data origin")
    return origin


def _train_estimators(
    training: tuple[CalibrationObservation, ...],
    model: ResponseModel,
    config: ResidualConfig,
) -> tuple[RandomForestRegressor, RandomForestRegressor, RandomForestRegressor]:
    X = np.asarray([_features(observation, model) for observation in training], dtype=float)
    Y = np.asarray([_residual_target(observation, model) for observation in training], dtype=float)
    estimators = []
    for output_index in range(3):
        estimator = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=1,
        )
        estimator.fit(X, Y[:, output_index])
        estimators.append(estimator)
    return tuple(estimators)  # type: ignore[return-value]


def _prediction_for_observation(
    observation: CalibrationObservation,
    model: ResponseModel,
    residual_model: ResidualModel,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    features = _features(observation, model)
    empirical = features[-3:]
    residual = residual_model._predict_residual(features)
    return tuple(float(value) for value in empirical), residual


def fit_residual_model(
    training: Iterable[CalibrationObservation],
    held_out: Iterable[CalibrationObservation],
    empirical_model: ResponseModel | None,
    config: ResidualConfig,
) -> ResidualModel:
    """Fit a residual model from explicit training data and evaluate it on held-out data.

    Held-out observations are never passed to estimator fitting.  The empirical model
    must already exist; this function never creates or replaces the primary model.
    """
    training_tuple = tuple(training)
    held_out_tuple = tuple(held_out)
    if empirical_model is None:
        result = ResidualTrainingResult(
            status=ResidualGateStatus.BASELINE_UNAVAILABLE,
            model_ready=False,
            feature_names=(),
            training_observation_count=len(training_tuple),
            held_out_observation_count=len(held_out_tuple),
            ablation=None,
            data_origin=DataOrigin.SYNTHETIC,
            provenance=config.provenance,
        )
        return ResidualModel(
            None, result, config, (), "UNAVAILABLE", "UNAVAILABLE", "UNAVAILABLE",
            DataOrigin.SYNTHETIC, config.provenance, frozenset(),
        )

    if len(training_tuple) < config.min_observations:
        origin = training_tuple[0].data_origin if training_tuple else empirical_model.data_origin
        provenance = training_tuple[0].provenance if training_tuple else config.provenance
        result = ResidualTrainingResult(
            status=ResidualGateStatus.INSUFFICIENT_DATA,
            model_ready=False,
            feature_names=tuple(
                (
            *empirical_model.channel_ids,
            "empirical_delta_L",
            "empirical_delta_a",
            "empirical_delta_b",
        )
            ),
            training_observation_count=len(training_tuple),
            held_out_observation_count=len(held_out_tuple),
            data_origin=origin,
            provenance=provenance,
        )
        return ResidualModel(
            None, result, config, empirical_model.channel_ids, empirical_model.sku_id,
            empirical_model.context_id, empirical_model.region_id, origin, provenance, frozenset(),
        )

    origin = _validate_training_context(training_tuple, empirical_model, config)
    _validate_training_context(held_out_tuple, empirical_model, config)
    if not held_out_tuple:
        raise ValueError("held-out observations are required for residual ablation")

    training_runs = {item.production_run_id for item in training_tuple}
    held_out_runs = {item.production_run_id for item in held_out_tuple}
    if training_runs & held_out_runs:
        raise ValueError("training and held-out production/kiln runs must not overlap")

    feature_names = tuple(
        (
            *empirical_model.channel_ids,
            "empirical_delta_L",
            "empirical_delta_a",
            "empirical_delta_b",
        )
    )
    estimators = _train_estimators(training_tuple, empirical_model, config)
    training_keys = frozenset(
        (item.experiment_id, item.production_run_id) for item in training_tuple
    )

    internal = ResidualModel(
        estimators,
        ResidualTrainingResult(
            status=ResidualGateStatus.DISABLED,
            model_ready=True,
            feature_names=feature_names,
            training_observation_count=len(training_tuple),
            held_out_observation_count=len(held_out_tuple),
            ablation=None,
            data_origin=origin,
            provenance=config.provenance,
        ),
        config,
        empirical_model.channel_ids,
        empirical_model.sku_id,
        empirical_model.context_id,
        empirical_model.region_id,
        origin,
        config.provenance,
        training_keys,
    )

    baseline_values = []
    combined_values = []
    predicted_residuals = []
    for observation in held_out_tuple:
        empirical, residual = _prediction_for_observation(observation, empirical_model, internal)
        observed = _observed_delta(observation, empirical_model)
        baseline_values.append([observed[index] - empirical[index] for index in range(3)])
        combined_values.append(
            [observed[index] - (empirical[index] + residual[index]) for index in range(3)]
        )
        predicted_residuals.append(residual)

    baseline_array = np.asarray(baseline_values, dtype=float)
    combined_array = np.asarray(combined_values, dtype=float)
    baseline_mse = float(np.mean(np.square(baseline_array)))
    residual_mse = float(np.mean(np.square(combined_array)))
    absolute_improvement = baseline_mse - residual_mse
    relative_improvement = absolute_improvement / baseline_mse if baseline_mse > 0.0 else 0.0
    safety_bound_passed = all(
        abs(value) <= config.max_residual_abs
        for residual in predicted_residuals
        for value in residual
    )
    acceptance_passed = (
        relative_improvement >= config.min_relative_improvement and safety_bound_passed
    )
    ablation = AblationReport(
        training_observation_count=len(training_tuple),
        held_out_observation_count=len(held_out_tuple),
        baseline_mse=baseline_mse,
        residual_mse=residual_mse,
        absolute_improvement=absolute_improvement,
        relative_improvement=relative_improvement,
        safety_bound_passed=safety_bound_passed,
        acceptance_passed=acceptance_passed,
        data_origin=origin,
        provenance=config.provenance,
    )
    if acceptance_passed:
        status = ResidualGateStatus.ENABLED
    elif not safety_bound_passed:
        status = ResidualGateStatus.RESIDUAL_TOO_LARGE
    else:
        status = ResidualGateStatus.VALIDATION_FAILED
    result = ResidualTrainingResult(
        status=status,
        model_ready=acceptance_passed,
        feature_names=feature_names,
        training_observation_count=len(training_tuple),
        held_out_observation_count=len(held_out_tuple),
        ablation=ablation,
        data_origin=origin,
        provenance=config.provenance,
    )
    return ResidualModel(
        estimators if acceptance_passed else None,
        result,
        config,
        empirical_model.channel_ids,
        empirical_model.sku_id,
        empirical_model.context_id,
        empirical_model.region_id,
        origin,
        config.provenance,
        training_keys,
    )


def split_and_fit_residual_model(
    observations: Iterable[CalibrationObservation],
    validation_run_ids: tuple[str, ...],
    empirical_model: ResponseModel | None,
    config: ResidualConfig,
) -> ResidualModel:
    """Use the existing grouped-by-run calibration split before residual fitting."""
    training, held_out = split_held_out_by_run(observations, validation_run_ids)
    return fit_residual_model(training, held_out, empirical_model, config)


def _inside_envelope(
    *,
    sku_id: str,
    context_id: str,
    region_id: str,
    channel_ids: tuple[str, ...],
    data_origin: DataOrigin,
    adjustments: tuple[ChannelAdjustment, ...],
    envelope: ValidatedOperatingEnvelope | None,
) -> bool:
    if envelope is None or envelope.validation_status.value != "validated":
        return False
    if (
        envelope.sku_id != sku_id
        or envelope.context_id != context_id
        or envelope.region_id != region_id
        or envelope.channel_ids != channel_ids
        or envelope.data_origin is not data_origin
    ):
        return False
    values = {item.channel_id: item.delta for item in adjustments}
    if set(values) - set(envelope.channel_ids):
        return False
    for index, channel_id in enumerate(envelope.channel_ids):
        value = values.get(channel_id, 0.0)
        if value < envelope.minimum_adjustment[index] or value > envelope.maximum_adjustment[index]:
            return False
    return True


def apply_residual(
    residual_model: ResidualModel,
    empirical_prediction: SensitivityPrediction | None,
    *,
    sku_id: str,
    context_id: str,
    region_id: str,
    channel_ids: tuple[str, ...],
    adjustments: tuple[ChannelAdjustment, ...],
    data_origin: DataOrigin,
    provenance: str,
    operating_envelope: ValidatedOperatingEnvelope | None,
) -> ResidualPrediction:
    """Apply the residual only when every gate passes; otherwise return empirical only."""
    if empirical_prediction is None:
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.BASELINE_UNAVAILABLE,
        )
    empirical = (
        empirical_prediction.predicted_delta_L,
        empirical_prediction.predicted_delta_a,
        empirical_prediction.predicted_delta_b,
    )
    status = residual_model.result.status
    if not residual_model.result.model_ready:
        if status is ResidualGateStatus.ENABLED:
            status = ResidualGateStatus.DISABLED
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance, status, empirical
        )
    if (
        sku_id != residual_model._ResidualModel__sku_id
        or context_id != residual_model._ResidualModel__context_id
        or region_id != residual_model._ResidualModel__region_id
    ):
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.SKU_CONTEXT_REGION_MISMATCH, empirical,
        )
    if (
        tuple(channel_ids) != residual_model.channel_ids
        or empirical_prediction.channel_ids != channel_ids
        or empirical_prediction.region_id != region_id
        or tuple(empirical_prediction.adjustment)
        != tuple(item.delta for item in adjustments)
    ):
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.CHANNEL_MISMATCH, empirical,
        )
    if (
        data_origin is not residual_model._ResidualModel__data_origin
        or provenance != residual_model._ResidualModel__provenance
    ):
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.PROVENANCE_MISMATCH, empirical,
        )
    if not _inside_envelope(
        sku_id=sku_id,
        context_id=context_id,
        region_id=region_id,
        channel_ids=channel_ids,
        data_origin=data_origin,
        adjustments=adjustments,
        envelope=operating_envelope,
    ):
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.OUT_OF_DOMAIN, empirical,
        )

    features = tuple(
        {item.channel_id: item.delta for item in adjustments}.get(channel_id, 0.0)
        for channel_id in residual_model.channel_ids
    ) + empirical
    residual = residual_model._predict_residual(features)
    if any(
        abs(value) > residual_model._ResidualModel__config.max_residual_abs
        for value in residual
    ):
        return _disabled_prediction(
            residual_model, region_id, channel_ids, data_origin, provenance,
            ResidualGateStatus.RESIDUAL_TOO_LARGE, empirical,
        )
    combined = tuple(empirical[index] + residual[index] for index in range(3))
    return ResidualPrediction(
        region_id=region_id,
        channel_ids=channel_ids,
        empirical_prediction=empirical,
        residual_prediction=residual,
        combined_prediction=combined,
        residual_enabled=True,
        gate_status=ResidualGateStatus.ENABLED,
        data_origin=data_origin,
        provenance=provenance,
    )


def _disabled_prediction(
    residual_model: ResidualModel,
    region_id: str,
    channel_ids: tuple[str, ...],
    data_origin: DataOrigin,
    provenance: str,
    status: ResidualGateStatus,
    empirical: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ResidualPrediction:
    return ResidualPrediction(
        region_id=region_id,
        channel_ids=channel_ids,
        empirical_prediction=empirical,
        residual_prediction=(0.0, 0.0, 0.0),
        combined_prediction=empirical,
        residual_enabled=False,
        gate_status=status,
        data_origin=data_origin,
        provenance=provenance,
    )
