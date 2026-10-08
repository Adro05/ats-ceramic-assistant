"""Calibration experiment and validation workflow around the local response model.

Pipeline position::

    controllable-input discovery
        -> gauge/noise study
        -> perturbation design
        -> multi-region / multi-run calibration
        -> Bayesian response-model fitting
        -> held-out validation
        -> validated operating envelope
        -> production use

This module only represents and validates calibration. It does not generate
corrections, optimize inputs, update models online, or model kiln/raw-material
causality. Printer channels remain opaque identifiers until client discovery
confirms what is actually controllable.
"""

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from math import sqrt
from typing import Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from ats_ceramic.response_model import (
    BayesianSensitivitySpec,
    ResponseModel,
    ResponseObservation,
    fit_response_model,
    predict_response,
)
from ats_ceramic.schemas import (
    CalibrationContext,
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    MeasurementMetadata,
    NonNegativeFinite,
    PositiveFinite,
    RegionMeasurement,
    require_unique,
)


class ControllableInput(FrozenModel):
    """Discovery record for one opaque controllable input."""

    channel_id: Identifier
    controllable: bool
    lower_bound: FiniteFloat | None = None
    upper_bound: FiniteFloat | None = None
    nominal_value: FiniteFloat | None = None
    unit: Identifier | None = None
    source: Identifier
    client_confirmed: bool
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound cannot exceed upper_bound")
        if self.nominal_value is not None:
            if self.lower_bound is not None and self.nominal_value < self.lower_bound:
                raise ValueError("nominal_value is below lower_bound")
            if self.upper_bound is not None and self.nominal_value > self.upper_bound:
                raise ValueError("nominal_value is above upper_bound")
        if self.client_confirmed and self.data_origin is not DataOrigin.CLIENT:
            raise ValueError("client_confirmed inputs must have CLIENT provenance")
        return self


class Perturbation(FrozenModel):
    """One deliberate change to one abstract channel in a calibration experiment."""

    perturbation_id: Identifier
    experiment_id: Identifier
    production_run_id: Identifier
    channel_id: Identifier
    delta: FiniteFloat
    is_baseline: bool = False

    @model_validator(mode="after")
    def _baseline_or_nonzero(self) -> Self:
        if self.is_baseline and self.delta != 0.0:
            raise ValueError("a baseline perturbation must have delta = 0")
        if not self.is_baseline and self.delta == 0.0:
            raise ValueError("non-baseline perturbation must have non-zero delta")
        return self


class CalibrationExperiment(FrozenModel):
    """Definition of one designed calibration experiment for one production run."""

    experiment_id: Identifier
    sku_id: Identifier
    calibration_context: CalibrationContext
    region_ids: tuple[Identifier, ...] = Field(min_length=1)
    controllable_inputs: tuple[ControllableInput, ...] = Field(min_length=1)
    nominal_operating_point: tuple[ChannelAdjustment, ...]
    perturbations: tuple[Perturbation, ...] = Field(min_length=1)
    production_run_id: Identifier
    measurement_metadata: MeasurementMetadata | None = None
    created_at: AwareDatetime
    provenance: str = Field(min_length=1)
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _validate_design(self) -> Self:
        require_unique(self.region_ids, "region_id")
        require_unique(
            (channel.channel_id for channel in self.controllable_inputs),
            "channel_id",
        )
        require_unique(
            (adjustment.channel_id for adjustment in self.nominal_operating_point),
            "channel_id",
        )
        require_unique(
            (perturbation.perturbation_id for perturbation in self.perturbations),
            "perturbation_id",
        )
        channel_ids = {channel.channel_id for channel in self.controllable_inputs}
        for adjustment in self.nominal_operating_point:
            if adjustment.channel_id not in channel_ids:
                raise ValueError(f"unknown nominal channel {adjustment.channel_id!r}")
        for perturbation in self.perturbations:
            if perturbation.experiment_id != self.experiment_id:
                raise ValueError("perturbation experiment_id does not match experiment")
            if perturbation.production_run_id != self.production_run_id:
                raise ValueError("perturbation production_run_id does not match experiment")
            if perturbation.channel_id not in channel_ids:
                raise ValueError(f"unknown perturbation channel {perturbation.channel_id!r}")
        return self


class CalibrationObservation(FrozenModel):
    """A calibration observation with experiment/run provenance around ResponseObservation."""

    experiment_id: Identifier
    production_run_id: Identifier
    measured_at: AwareDatetime
    response_observation: ResponseObservation
    provenance: str = Field(min_length=1)
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _provenance_matches(self) -> Self:
        if self.data_origin is not self.response_observation.data_origin:
            raise ValueError("calibration and response observation provenance must match")
        return self


class NoiseFloorEstimate(FrozenModel):
    """Deterministic replicate-based estimate of measurement variation for one region."""

    region_id: Identifier
    instrument_id: Identifier
    n_measurements: int = Field(ge=1)
    replicate_count_total: int = Field(ge=2)
    noise_std_L: NonNegativeFinite
    noise_std_a: NonNegativeFinite
    noise_std_b: NonNegativeFinite
    noise_std_gloss: NonNegativeFinite | None = None
    estimation_method: Identifier
    estimated_at: AwareDatetime
    provenance: str = Field(min_length=1)
    data_origin: DataOrigin


class NoiseResponseAssessment(FrozenModel):
    """Deterministic comparison of an observed Lab response with measurement noise."""

    region_id: Identifier
    observed_delta_L: FiniteFloat
    observed_delta_a: FiniteFloat
    observed_delta_b: FiniteFloat
    noise_std_L: NonNegativeFinite
    noise_std_a: NonNegativeFinite
    noise_std_b: NonNegativeFinite
    above_noise_floor_L: bool
    above_noise_floor_a: bool
    above_noise_floor_b: bool


class ValidationCriterion(FrozenModel):
    """Explicit workflow criteria for deciding whether held-out validation passes.

    These are workflow-level acceptance criteria, not client-confirmed ceramic
    tolerances. No performance threshold is invented by this module.
    """

    min_training_runs: int = Field(ge=1)
    min_validation_runs: int = Field(ge=1)
    min_validation_observations: int = Field(ge=1)
    max_mean_abs_error_L: NonNegativeFinite | None = None
    max_mean_abs_error_a: NonNegativeFinite | None = None
    max_mean_abs_error_b: NonNegativeFinite | None = None
    min_interval_coverage_L: NonNegativeFinite | None = Field(default=None, le=1.0)
    min_interval_coverage_a: NonNegativeFinite | None = Field(default=None, le=1.0)
    min_interval_coverage_b: NonNegativeFinite | None = Field(default=None, le=1.0)


class ValidationStatus(StrEnum):
    """Outcome of deterministic held-out validation and explicit criteria."""

    VALIDATED = "validated"
    FAILED = "failed"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    EVALUATED_NO_CRITERION = "evaluated_no_criterion"


class ValidationObservationResult(FrozenModel):
    """Prediction error and Gaussian predictive-interval coverage for one held-out sample."""

    region_id: Identifier
    production_run_id: Identifier
    observed_delta_L: FiniteFloat
    observed_delta_a: FiniteFloat
    observed_delta_b: FiniteFloat
    predicted_delta_L: FiniteFloat
    predicted_delta_a: FiniteFloat
    predicted_delta_b: FiniteFloat
    error_L: FiniteFloat
    error_a: FiniteFloat
    error_b: FiniteFloat
    uncertainty_std_L: PositiveFinite
    uncertainty_std_a: PositiveFinite
    uncertainty_std_b: PositiveFinite
    within_95_interval_L: bool
    within_95_interval_a: bool
    within_95_interval_b: bool


class HeldOutValidationResult(FrozenModel):
    """Deterministic report for training/held-out calibration observations."""

    sku_id: Identifier
    context_id: Identifier
    training_observation_count: int = Field(ge=0)
    validation_observation_count: int = Field(ge=0)
    training_run_ids: tuple[Identifier, ...]
    validation_run_ids: tuple[Identifier, ...]
    region_results: tuple[ValidationObservationResult, ...]
    interval_z: PositiveFinite
    interval_coverage_L: NonNegativeFinite
    interval_coverage_a: NonNegativeFinite
    interval_coverage_b: NonNegativeFinite
    mean_abs_error_L: NonNegativeFinite
    mean_abs_error_a: NonNegativeFinite
    mean_abs_error_b: NonNegativeFinite
    status: ValidationStatus
    failure_reasons: tuple[str, ...]
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _coverage_bounds(self) -> Self:
        for name in (
            "interval_coverage_L",
            "interval_coverage_a",
            "interval_coverage_b",
        ):
            value = getattr(self, name)
            if value > 1.0:
                raise ValueError(f"{name} cannot exceed 1")
        return self


class ValidatedOperatingEnvelope(FrozenModel):
    """Empirically observed adjustment envelope for one SKU/context/region."""

    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    minimum_adjustment: tuple[float, ...] = Field(min_length=1)
    maximum_adjustment: tuple[float, ...] = Field(min_length=1)
    n_observations: int = Field(ge=1)
    validation_status: ValidationStatus
    production_run_ids: tuple[Identifier, ...] = Field(min_length=1)
    calibrated_at: AwareDatetime
    data_origin: DataOrigin
    model_version: Identifier | None = None

    @model_validator(mode="after")
    def _shape_and_ranges(self) -> Self:
        if len(self.minimum_adjustment) != len(self.channel_ids):
            raise ValueError("minimum_adjustment length must match channel_ids")
        if len(self.maximum_adjustment) != len(self.channel_ids):
            raise ValueError("maximum_adjustment length must match channel_ids")
        for minimum, maximum in zip(
            self.minimum_adjustment,
            self.maximum_adjustment,
            strict=True,
        ):
            if minimum > maximum:
                raise ValueError("minimum_adjustment cannot exceed maximum_adjustment")
        require_unique(self.channel_ids, "channel_id")
        require_unique(self.production_run_ids, "production_run_id")
        return self

    def contains(self, adjustments: tuple[ChannelAdjustment, ...]) -> bool:
        """Return whether a proposed adjustment lies inside this empirical envelope."""
        values = {adjustment.channel_id: adjustment.delta for adjustment in adjustments}
        if set(values) - set(self.channel_ids):
            return False
        for index, channel_id in enumerate(self.channel_ids):
            value = values.get(channel_id, 0.0)
            if value < self.minimum_adjustment[index] or value > self.maximum_adjustment[index]:
                return False
        return True


class RecalibrationTrigger(StrEnum):
    """Reason codes for requesting recalibration without performing it."""

    OUTSIDE_VALIDATED_ENVELOPE = "outside_validated_envelope"
    INSUFFICIENT_RECENT_COVERAGE = "insufficient_recent_coverage"
    REPEATED_VALIDATION_FAILURE = "repeated_validation_failure"
    MODEL_STALE = "model_stale"
    MEASUREMENT_CONTEXT_CHANGED = "measurement_context_changed"
    PRINTER_CONTEXT_CHANGED = "printer_context_changed"
    REPEATED_VERIFICATION_RESIDUAL = "repeated_verification_residual"


class RecalibrationAssessment(FrozenModel):
    """Deterministic trigger assessment; no automatic retraining is performed."""

    triggered: bool
    reasons: tuple[RecalibrationTrigger, ...]

    @model_validator(mode="after")
    def _trigger_matches_reasons(self) -> Self:
        if self.triggered != bool(self.reasons):
            raise ValueError("triggered must match whether any reason is present")
        return self


def _require_same_context(
    observations: tuple[CalibrationObservation, ...],
) -> tuple[str, str, DataOrigin]:
    if not observations:
        raise ValueError("at least one calibration observation is required")
    first = observations[0].response_observation
    for observation in observations:
        response = observation.response_observation
        if response.sku_id != first.sku_id:
            raise ValueError("calibration observations must use one SKU")
        if response.context_id != first.context_id:
            raise ValueError("calibration observations must use one calibration context")
        if observation.data_origin is not first.data_origin:
            raise ValueError("calibration observations must use one data origin")
    return first.sku_id, first.context_id, first.data_origin


def to_response_observations(
    observations: Iterable[CalibrationObservation],
) -> tuple[ResponseObservation, ...]:
    """Return the existing response-model observations without changing their identity."""
    observations_tuple = tuple(observations)
    _require_same_context(observations_tuple)
    return tuple(observation.response_observation for observation in observations_tuple)


def estimate_noise_floor(
    measurements: Iterable[RegionMeasurement],
    metadata: MeasurementMetadata,
    estimated_at: datetime,
    provenance: str,
    data_origin: DataOrigin,
) -> NoiseFloorEstimate:
    """Estimate region-specific replicate noise using pooled within-sample variance."""
    measurements_tuple = tuple(measurements)
    if not measurements_tuple:
        raise ValueError("at least one region measurement is required")
    region_ids = tuple(measurement.region_id for measurement in measurements_tuple)
    require_unique((region_ids[0],), "region_id")
    if any(region_id != region_ids[0] for region_id in region_ids):
        raise ValueError("noise-floor estimation requires one region at a time")
    if any(
        measurement.n_replicates < 2 or measurement.replicate_spread is None
        for measurement in measurements_tuple
    ):
        raise ValueError("noise-floor estimation requires replicate_spread with n_replicates >= 2")

    total_degrees = sum(measurement.n_replicates - 1 for measurement in measurements_tuple)

    def pooled(quantity: str) -> float:
        numerator = sum(
            (measurement.n_replicates - 1) * getattr(measurement.replicate_spread, quantity) ** 2
            for measurement in measurements_tuple
        )
        return sqrt(numerator / total_degrees)

    gloss_values = [measurement.replicate_spread.gloss for measurement in measurements_tuple]
    return NoiseFloorEstimate(
        region_id=measurements_tuple[0].region_id,
        instrument_id=metadata.instrument_id,
        n_measurements=len(measurements_tuple),
        replicate_count_total=sum(measurement.n_replicates for measurement in measurements_tuple),
        noise_std_L=pooled("L"),
        noise_std_a=pooled("a"),
        noise_std_b=pooled("b"),
        noise_std_gloss=pooled("gloss") if gloss_values else None,
        estimation_method="pooled_within_replicate_std",
        estimated_at=estimated_at,
        provenance=provenance,
        data_origin=data_origin,
    )


def assess_response_against_noise_floor(
    noise_floor: NoiseFloorEstimate,
    *,
    observed_delta_L: float,
    observed_delta_a: float,
    observed_delta_b: float,
    region_id: str | None = None,
) -> NoiseResponseAssessment:
    """Report whether an observed Lab response is above each estimated noise floor."""
    assessment_region_id = noise_floor.region_id if region_id is None else region_id
    if assessment_region_id != noise_floor.region_id:
        raise ValueError("noise-floor region and observed response region must match")
    return NoiseResponseAssessment(
        region_id=assessment_region_id,
        observed_delta_L=observed_delta_L,
        observed_delta_a=observed_delta_a,
        observed_delta_b=observed_delta_b,
        noise_std_L=noise_floor.noise_std_L,
        noise_std_a=noise_floor.noise_std_a,
        noise_std_b=noise_floor.noise_std_b,
        above_noise_floor_L=abs(observed_delta_L) > noise_floor.noise_std_L,
        above_noise_floor_a=abs(observed_delta_a) > noise_floor.noise_std_a,
        above_noise_floor_b=abs(observed_delta_b) > noise_floor.noise_std_b,
    )


def fit_calibration_models(
    context: CalibrationContext,
    sku_id: str,
    observations: Iterable[CalibrationObservation],
    channel_ids: tuple[str, ...],
    spec: BayesianSensitivitySpec,
) -> tuple[ResponseModel, ...]:
    """Fit the existing Bayesian response model independently for each observed region."""
    observations_tuple = tuple(observations)
    if not observations_tuple:
        raise ValueError("at least one calibration observation is required")
    _require_same_context(observations_tuple)
    if any(
        observation.response_observation.sku_id != sku_id
        or observation.response_observation.context_id != context.context_id
        for observation in observations_tuple
    ):
        raise ValueError("calibration observations do not match requested SKU/context")

    region_ids = tuple(
        dict.fromkeys(
            observation.response_observation.region_id for observation in observations_tuple
        )
    )
    models = []
    for region_id in region_ids:
        region_observations = tuple(
            observation.response_observation
            for observation in observations_tuple
            if observation.response_observation.region_id == region_id
        )
        models.append(
            fit_response_model(
                context,
                sku_id,
                region_id,
                region_observations,
                channel_ids,
                spec,
            )
        )
    return tuple(models)


def split_held_out_by_run(
    observations: Iterable[CalibrationObservation],
    validation_run_ids: tuple[str, ...],
) -> tuple[
    tuple[CalibrationObservation, ...],
    tuple[CalibrationObservation, ...],
]:
    """Split observations by explicit production/kiln run IDs with no run leakage."""
    observations_tuple = tuple(observations)
    if not observations_tuple:
        raise ValueError("at least one calibration observation is required")
    require_unique(validation_run_ids, "validation_run_id")
    requested = set(validation_run_ids)
    observed_runs = {observation.production_run_id for observation in observations_tuple}
    unknown_runs = requested - observed_runs
    if unknown_runs:
        raise ValueError(f"unknown validation run IDs: {sorted(unknown_runs)}")

    training = tuple(
        observation
        for observation in observations_tuple
        if observation.production_run_id not in requested
    )
    validation = tuple(
        observation
        for observation in observations_tuple
        if observation.production_run_id in requested
    )
    if not training:
        raise ValueError("held-out split requires at least one training production/kiln run")
    if not validation:
        raise ValueError("held-out split requires at least one validation production/kiln run")
    return training, validation


def validate_held_out(
    models: Iterable[ResponseModel],
    training: Iterable[CalibrationObservation],
    validation: Iterable[CalibrationObservation],
    *,
    interval_z: float = 1.96,
    criterion: ValidationCriterion | None = None,
) -> HeldOutValidationResult:
    """Evaluate grouped held-out data against explicitly supplied workflow criteria."""
    if interval_z <= 0.0:
        raise ValueError("interval_z must be positive")
    models_tuple = tuple(models)
    training_tuple = tuple(training)
    validation_tuple = tuple(validation)
    if not models_tuple:
        raise ValueError("at least one response model is required")
    if not training_tuple or not validation_tuple:
        raise ValueError("training and validation observations are both required")

    sku_id, context_id, data_origin = _require_same_context(training_tuple)
    validation_sku, validation_context, validation_origin = _require_same_context(validation_tuple)
    if (validation_sku, validation_context) != (sku_id, context_id):
        raise ValueError("training and validation must use the same SKU/context")
    if validation_origin is not data_origin:
        raise ValueError("training and validation must use the same data origin")

    training_runs = tuple(
        dict.fromkeys(observation.production_run_id for observation in training_tuple)
    )
    validation_runs = tuple(
        dict.fromkeys(observation.production_run_id for observation in validation_tuple)
    )
    if set(training_runs) & set(validation_runs):
        raise ValueError("training and validation production/kiln runs must not overlap")

    model_by_region = {model.region_id: model for model in models_tuple}
    require_unique(model_by_region.keys(), "region_id")
    results: list[ValidationObservationResult] = []
    for observation in validation_tuple:
        response = observation.response_observation
        try:
            model = model_by_region[response.region_id]
        except KeyError as exc:
            raise ValueError(
                f"no response model for validation region {response.region_id!r}"
            ) from exc
        if model.sku_id != sku_id or model.context_id != context_id:
            raise ValueError("response models must match the validation SKU/context")
        if model.data_origin is not data_origin:
            raise ValueError("response models must match the validation data origin")
        prediction = predict_response(model, response.adjustments)
        observed = response.comparison.region(response.region_id)
        errors = (
            observed.delta_L - prediction.predicted_delta_L,
            observed.delta_a - prediction.predicted_delta_a,
            observed.delta_b - prediction.predicted_delta_b,
        )
        results.append(
            ValidationObservationResult(
                region_id=response.region_id,
                production_run_id=observation.production_run_id,
                observed_delta_L=observed.delta_L,
                observed_delta_a=observed.delta_a,
                observed_delta_b=observed.delta_b,
                predicted_delta_L=prediction.predicted_delta_L,
                predicted_delta_a=prediction.predicted_delta_a,
                predicted_delta_b=prediction.predicted_delta_b,
                error_L=errors[0],
                error_a=errors[1],
                error_b=errors[2],
                uncertainty_std_L=prediction.uncertainty_std_L,
                uncertainty_std_a=prediction.uncertainty_std_a,
                uncertainty_std_b=prediction.uncertainty_std_b,
                within_95_interval_L=abs(errors[0]) <= interval_z * prediction.uncertainty_std_L,
                within_95_interval_a=abs(errors[1]) <= interval_z * prediction.uncertainty_std_a,
                within_95_interval_b=abs(errors[2]) <= interval_z * prediction.uncertainty_std_b,
            )
        )

    def coverage(attribute: str) -> float:
        return sum(bool(getattr(result, attribute)) for result in results) / len(results)

    interval_coverage_L = coverage("within_95_interval_L")
    interval_coverage_a = coverage("within_95_interval_a")
    interval_coverage_b = coverage("within_95_interval_b")
    mean_abs_error_L = sum(abs(result.error_L) for result in results) / len(results)
    mean_abs_error_a = sum(abs(result.error_a) for result in results) / len(results)
    mean_abs_error_b = sum(abs(result.error_b) for result in results) / len(results)

    failures: list[str] = []
    if criterion is not None:
        if len(training_runs) < criterion.min_training_runs:
            failures.append(
                "training run coverage below criterion "
                f"({len(training_runs)} < {criterion.min_training_runs})"
            )
        if len(validation_runs) < criterion.min_validation_runs:
            failures.append(
                "validation run coverage below criterion "
                f"({len(validation_runs)} < {criterion.min_validation_runs})"
            )
        if len(validation_tuple) < criterion.min_validation_observations:
            failures.append(
                "validation observation coverage below criterion "
                f"({len(validation_tuple)} < {criterion.min_validation_observations})"
            )

    coverage_sufficient = (
        bool(training_runs)
        and bool(validation_runs)
        and bool(results)
        and (
            criterion is None
            or (
                len(training_runs) >= criterion.min_training_runs
                and len(validation_runs) >= criterion.min_validation_runs
                and len(validation_tuple) >= criterion.min_validation_observations
            )
        )
    )

    if not coverage_sufficient:
        status = ValidationStatus.INSUFFICIENT_COVERAGE
    elif criterion is None:
        status = ValidationStatus.EVALUATED_NO_CRITERION
    else:
        performance_failures: list[str] = []
        for name, value, maximum in (
            ("L", mean_abs_error_L, criterion.max_mean_abs_error_L),
            ("a", mean_abs_error_a, criterion.max_mean_abs_error_a),
            ("b", mean_abs_error_b, criterion.max_mean_abs_error_b),
        ):
            if maximum is not None and value > maximum:
                performance_failures.append(
                    f"mean_abs_error_{name} exceeds criterion ({value} > {maximum})"
                )
        for name, value, minimum in (
            ("L", interval_coverage_L, criterion.min_interval_coverage_L),
            ("a", interval_coverage_a, criterion.min_interval_coverage_a),
            ("b", interval_coverage_b, criterion.min_interval_coverage_b),
        ):
            if minimum is not None and value < minimum:
                performance_failures.append(
                    f"interval_coverage_{name} below criterion ({value} < {minimum})"
                )
        failures.extend(performance_failures)
        status = ValidationStatus.FAILED if failures else ValidationStatus.VALIDATED

    return HeldOutValidationResult(
        sku_id=sku_id,
        context_id=context_id,
        training_observation_count=len(training_tuple),
        validation_observation_count=len(validation_tuple),
        training_run_ids=training_runs,
        validation_run_ids=validation_runs,
        region_results=tuple(results),
        interval_z=interval_z,
        interval_coverage_L=interval_coverage_L,
        interval_coverage_a=interval_coverage_a,
        interval_coverage_b=interval_coverage_b,
        mean_abs_error_L=mean_abs_error_L,
        mean_abs_error_a=mean_abs_error_a,
        mean_abs_error_b=mean_abs_error_b,
        status=status,
        failure_reasons=tuple(failures),
        data_origin=data_origin,
    )


def build_operating_envelopes(
    models: Iterable[ResponseModel],
    training: Iterable[CalibrationObservation],
    validation_result: HeldOutValidationResult,
    calibrated_at: datetime,
) -> tuple[ValidatedOperatingEnvelope, ...]:
    """Build empirical min/max channel envelopes from training observations only."""
    models_tuple = tuple(models)
    training_tuple = tuple(training)
    if not training_tuple:
        raise ValueError("at least one training observation is required")

    envelopes = []
    for model in models_tuple:
        observations = tuple(
            observation
            for observation in training_tuple
            if observation.response_observation.region_id == model.region_id
        )
        if not observations:
            raise ValueError(f"no training observations for region {model.region_id!r}")

        vectors = []
        for observation in observations:
            values = {
                adjustment.channel_id: adjustment.delta
                for adjustment in observation.response_observation.adjustments
            }
            vectors.append(tuple(values.get(channel_id, 0.0) for channel_id in model.channel_ids))
        minimum = tuple(
            min(vector[index] for vector in vectors) for index in range(len(model.channel_ids))
        )
        maximum = tuple(
            max(vector[index] for vector in vectors) for index in range(len(model.channel_ids))
        )
        runs = tuple(dict.fromkeys(observation.production_run_id for observation in observations))
        envelopes.append(
            ValidatedOperatingEnvelope(
                sku_id=model.sku_id,
                context_id=model.context_id,
                region_id=model.region_id,
                channel_ids=model.channel_ids,
                minimum_adjustment=minimum,
                maximum_adjustment=maximum,
                n_observations=len(observations),
                validation_status=validation_result.status,
                production_run_ids=runs,
                calibrated_at=calibrated_at,
                data_origin=model.data_origin,
                model_version=None,
            )
        )
    return tuple(envelopes)


def assess_recalibration(
    *,
    outside_validated_envelope: bool = False,
    insufficient_recent_coverage: bool = False,
    repeated_validation_failure: bool = False,
    model_stale: bool = False,
    measurement_context_changed: bool = False,
    printer_context_changed: bool = False,
    repeated_verification_residual: bool = False,
) -> RecalibrationAssessment:
    """Return configurable reason codes; never retrain or recalibrate automatically."""
    reasons = tuple(
        reason
        for condition, reason in (
            (
                outside_validated_envelope,
                RecalibrationTrigger.OUTSIDE_VALIDATED_ENVELOPE,
            ),
            (
                insufficient_recent_coverage,
                RecalibrationTrigger.INSUFFICIENT_RECENT_COVERAGE,
            ),
            (
                repeated_validation_failure,
                RecalibrationTrigger.REPEATED_VALIDATION_FAILURE,
            ),
            (
                model_stale,
                RecalibrationTrigger.MODEL_STALE,
            ),
            (
                measurement_context_changed,
                RecalibrationTrigger.MEASUREMENT_CONTEXT_CHANGED,
            ),
            (
                printer_context_changed,
                RecalibrationTrigger.PRINTER_CONTEXT_CHANGED,
            ),
            (
                repeated_verification_residual,
                RecalibrationTrigger.REPEATED_VERIFICATION_RESIDUAL,
            ),
        )
        if condition
    )
    return RecalibrationAssessment(triggered=bool(reasons), reasons=reasons)
