"""Local Bayesian linear response model for per-region colour sensitivity.

Pipeline position::

    triage -> response model -> constrained optimiser

The model is deliberately local to one SKU, calibration context, and named region.
Printer channels remain opaque identifiers supplied by the caller; this module does
not discover or invent controllable channels. Gloss is excluded from the response
model and remains a separate diagnostic signal.

All fitting inputs are expected to be labelled with their ``DataOrigin``. The shipped
prototype contains no client calibration data; tests use explicitly synthetic values.
"""

from collections.abc import Iterable
from math import sqrt

import numpy as np
from pydantic import Field, model_validator

from ats_ceramic.color import ComparisonResult
from ats_ceramic.schemas import (
    CalibrationContext,
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    Identifier,
    PositiveFinite,
    require_unique,
)


class BayesianSensitivitySpec(FrozenModel):
    """Explicit modelling assumptions for the local Bayesian linear model.

    These values are not client settings. The caller must provide them explicitly so
    this prototype never silently invents a printer-specific prior or noise level.
    """

    prior_variance: PositiveFinite
    observation_noise_std: tuple[
        PositiveFinite,
        PositiveFinite,
        PositiveFinite,
    ]
    min_observations: int = Field(default=2, ge=1)
    is_placeholder: bool
    source: str = Field(min_length=1)


class ResponseObservation(FrozenModel):
    """One calibration observation for one named region."""

    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    adjustments: tuple[ChannelAdjustment, ...]
    comparison: ComparisonResult
    data_origin: DataOrigin


class ResponseModel(FrozenModel):
    """Posterior parameters of a local, intercept-free Bayesian sensitivity model.

    The model is local to one ``sku_id``, calibration ``context_id``, and
    ``region_id``. ``coefficients`` contains one row per abstract channel and three
    columns for ``(delta_L, delta_a, delta_b)``. ``posterior_covariance`` contains
    one parameter covariance matrix for each Lab output dimension.
    """

    sku_id: Identifier
    context_id: Identifier
    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    coefficients: tuple[tuple[float, float, float], ...] = Field(min_length=1)
    posterior_covariance: tuple[tuple[tuple[float, ...], ...], ...] = Field(min_length=1)
    spec: BayesianSensitivitySpec
    n_observations: int = Field(ge=1)
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _validate_shape(self) -> "ResponseModel":
        p = len(self.channel_ids)
        if len(self.coefficients) != p:
            raise ValueError("coefficients must contain one row per channel")
        if any(len(row) != 3 for row in self.coefficients):
            raise ValueError("each coefficient row must contain L, a and b")
        if len(self.posterior_covariance) != 3 or any(
            len(matrix) != p or any(len(row) != p for row in matrix)
            for matrix in self.posterior_covariance
        ):
            raise ValueError(
                "posterior_covariance must contain three square matrices with one row per channel"
            )
        require_unique(self.channel_ids, "channel_id")
        return self


class SensitivityPrediction(FrozenModel):
    """Posterior prediction and uncertainty for one proposed abstract adjustment."""

    region_id: Identifier
    channel_ids: tuple[Identifier, ...] = Field(min_length=1)
    adjustment: tuple[float, ...] = Field(min_length=1)
    predicted_delta_L: float
    predicted_delta_a: float
    predicted_delta_b: float
    uncertainty_std_L: PositiveFinite
    uncertainty_std_a: PositiveFinite
    uncertainty_std_b: PositiveFinite

    @model_validator(mode="after")
    def _matching_shape(self) -> "SensitivityPrediction":
        if len(self.adjustment) != len(self.channel_ids):
            raise ValueError("adjustment length must match channel_ids")
        return self


def _region_signed_colour_delta(
    comparison: ComparisonResult, region_id: str
) -> tuple[float, float, float]:
    for region in comparison.regions:
        if region.region_id == region_id:
            return region.delta_L, region.delta_a, region.delta_b
    raise ValueError(f"region {region_id!r} is not present in comparison")


def _design_row(
    adjustments: tuple[ChannelAdjustment, ...], channel_index: dict[str, int], p: int
) -> tuple[float, ...]:
    row = [0.0] * p
    for adjustment in adjustments:
        try:
            index = channel_index[adjustment.channel_id]
        except KeyError as exc:
            raise ValueError(f"unknown channel {adjustment.channel_id!r} in observation") from exc
        row[index] = adjustment.delta
    return tuple(row)


def _invert(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("response-model posterior precision matrix is singular") from exc


def fit_response_model(
    context: CalibrationContext,
    sku_id: str,
    region_id: str,
    observations: Iterable[ResponseObservation],
    channel_ids: tuple[str, ...],
    spec: BayesianSensitivitySpec,
) -> ResponseModel:
    """Fit a local Bayesian linear sensitivity model for one named region.

    Only observations for the requested SKU, calibration context, and region are
    accepted. The model has no intercept: a zero channel adjustment represents zero
    predicted colour response. This prevents the response model from becoming a
    hidden baseline or causal model.
    """
    observations_tuple = tuple(observations)
    if len(observations_tuple) < spec.min_observations:
        raise ValueError(
            f"at least {spec.min_observations} observations are required; "
            f"received {len(observations_tuple)}"
        )
    if not channel_ids:
        raise ValueError("at least one channel_id is required")
    require_unique(channel_ids, "channel_id")
    if context.context_id == "":
        raise ValueError("context_id must be non-empty")

    for observation in observations_tuple:
        if observation.sku_id != sku_id:
            raise ValueError("all observations must match the requested sku_id")
        if observation.context_id != context.context_id:
            raise ValueError("all observations must match the requested calibration context")
        if observation.region_id != region_id:
            raise ValueError("all observations must match the requested region_id")
        if observation.data_origin is not DataOrigin.SYNTHETIC and spec.is_placeholder:
            raise ValueError("placeholder response-model assumptions cannot fit non-synthetic data")
        _region_signed_colour_delta(observation.comparison, region_id)

    channel_index = {channel_id: i for i, channel_id in enumerate(channel_ids)}
    X = np.asarray(
        [
            _design_row(observation.adjustments, channel_index, len(channel_ids))
            for observation in observations_tuple
        ],
        dtype=float,
    )
    Y = np.asarray(
        [
            _region_signed_colour_delta(observation.comparison, region_id)
            for observation in observations_tuple
        ],
        dtype=float,
    )

    prior_precision = np.eye(len(channel_ids), dtype=float) / spec.prior_variance
    posterior_covariances: list[np.ndarray] = []
    coefficients = np.empty((len(channel_ids), 3), dtype=float)

    for output_index, noise_std in enumerate(spec.observation_noise_std):
        noise_precision = 1.0 / (noise_std * noise_std)
        precision = prior_precision + noise_precision * (X.T @ X)
        covariance = _invert(precision)
        posterior_covariances.append(covariance)
        coefficients[:, output_index] = covariance @ (noise_precision * X.T @ Y[:, output_index])

    return ResponseModel(
        sku_id=sku_id,
        context_id=context.context_id,
        region_id=region_id,
        channel_ids=channel_ids,
        coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
        posterior_covariance=tuple(
            tuple(tuple(float(value) for value in row) for row in matrix)
            for matrix in posterior_covariances
        ),
        spec=spec,
        n_observations=len(observations_tuple),
        data_origin=DataOrigin.SYNTHETIC
        if all(
            observation.data_origin is DataOrigin.SYNTHETIC for observation in observations_tuple
        )
        else DataOrigin.CLIENT,
    )


def predict_response(
    model: ResponseModel,
    adjustments: tuple[ChannelAdjustment, ...],
) -> SensitivityPrediction:
    """Predict Lab response and posterior predictive uncertainty for the model region.

    The uncertainty is the posterior predictive standard deviation: parameter
    uncertainty plus the configured observation-noise variance. No gloss response is
    predicted here.
    """
    channel_index = {channel_id: i for i, channel_id in enumerate(model.channel_ids)}
    x = np.asarray(_design_row(adjustments, channel_index, len(model.channel_ids)), dtype=float)
    beta = np.asarray(model.coefficients, dtype=float)
    covariances = tuple(np.asarray(matrix, dtype=float) for matrix in model.posterior_covariance)

    predicted = x @ beta
    parameter_variances = []
    for covariance in covariances:
        variance = float(x @ covariance @ x)
        if variance < 0.0 and variance > -1e-12:
            variance = 0.0
        if variance < 0.0:
            raise ValueError("posterior parameter variance became negative")
        parameter_variances.append(variance)

    uncertainties = tuple(
        sqrt(variance + noise_std * noise_std)
        for variance, noise_std in zip(
            parameter_variances,
            model.spec.observation_noise_std,
            strict=True,
        )
    )
    return SensitivityPrediction(
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustment=tuple(float(value) for value in x),
        predicted_delta_L=float(predicted[0]),
        predicted_delta_a=float(predicted[1]),
        predicted_delta_b=float(predicted[2]),
        uncertainty_std_L=uncertainties[0],
        uncertainty_std_a=uncertainties[1],
        uncertainty_std_b=uncertainties[2],
    )
