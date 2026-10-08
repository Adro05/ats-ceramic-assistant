"""Tests for the local Bayesian per-region sensitivity response model.

All calibration observations and modelling assumptions in this file are explicitly
synthetic test values; they are not client printer settings or measured data.
"""

import pytest
from pydantic import ValidationError

from ats_ceramic.color import ComparisonResult, RegionDifference
from ats_ceramic.response_model import (
    BayesianSensitivitySpec,
    ResponseObservation,
    SensitivityPrediction,
    fit_response_model,
    predict_response,
)
from ats_ceramic.schemas import CalibrationContext, ChannelAdjustment, DataOrigin


def _context() -> CalibrationContext:
    return CalibrationContext(
        context_id="ctx_test_syn",
        printer_id="printer_test_syn",
        glaze_context="glaze_test_syn",
        ink_set_id="ink_test_syn",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _comparison(
    delta_l: float,
    delta_a: float,
    delta_b: float,
    *,
    region_id: str = "R1",
    gloss_delta: float = 0.0,
) -> ComparisonResult:
    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id=region_id,
                delta_L=delta_l,
                delta_a=delta_a,
                delta_b=delta_b,
                delta_E00=1.0,
                delta_gloss=gloss_delta,
            ),
        ),
        mean_delta_E00=1.0,
        max_delta_E00=1.0,
        worst_region_id=region_id,
        mean_abs_delta_gloss=abs(gloss_delta),
        max_abs_delta_gloss=abs(gloss_delta),
    )


def _comparison_two_regions(
    r1: tuple[float, float, float], r2: tuple[float, float, float], *, gloss_delta: float = 0.0
) -> ComparisonResult:
    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id="R1",
                delta_L=r1[0],
                delta_a=r1[1],
                delta_b=r1[2],
                delta_E00=1.0,
                delta_gloss=gloss_delta,
            ),
            RegionDifference(
                region_id="R2",
                delta_L=r2[0],
                delta_a=r2[1],
                delta_b=r2[2],
                delta_E00=1.0,
                delta_gloss=gloss_delta,
            ),
        ),
        mean_delta_E00=1.0,
        max_delta_E00=1.0,
        worst_region_id="R1",
        mean_abs_delta_gloss=abs(gloss_delta),
        max_abs_delta_gloss=abs(gloss_delta),
    )


def _observation(
    channel: str,
    delta: float,
    response: tuple[float, float, float],
    *,
    region_id: str = "R1",
    comparison: ComparisonResult | None = None,
    data_origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> ResponseObservation:
    return ResponseObservation(
        sku_id="SKU_SYN_TEST",
        context_id="ctx_test_syn",
        region_id=region_id,
        adjustments=(ChannelAdjustment(channel_id=channel, delta=delta),),
        comparison=comparison or _comparison(*response, region_id=region_id),
        data_origin=data_origin,
    )


def _spec() -> BayesianSensitivitySpec:
    return BayesianSensitivitySpec(
        prior_variance=100.0,
        observation_noise_std=(0.01, 0.01, 0.01),
        min_observations=2,
        is_placeholder=True,
        source="synthetic test assumption",
    )


def test_multiple_regions_are_kept_separate():
    r1_observations = (
        _observation(
            "channel_a",
            1.0,
            (4.0, 1.0, -1.0),
            region_id="R1",
            comparison=_comparison_two_regions((4.0, 1.0, -1.0), (-4.0, -1.0, 1.0)),
        ),
        _observation(
            "channel_a",
            2.0,
            (8.0, 2.0, -2.0),
            region_id="R1",
            comparison=_comparison_two_regions((8.0, 2.0, -2.0), (-8.0, -2.0, 2.0)),
        ),
    )
    r2_observations = (
        _observation(
            "channel_a",
            1.0,
            (-4.0, -1.0, 1.0),
            region_id="R2",
            comparison=_comparison_two_regions((4.0, 1.0, -1.0), (-4.0, -1.0, 1.0)),
        ),
        _observation(
            "channel_a",
            2.0,
            (-8.0, -2.0, 2.0),
            region_id="R2",
            comparison=_comparison_two_regions((8.0, 2.0, -2.0), (-8.0, -2.0, 2.0)),
        ),
    )

    r1_model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", r1_observations, ("channel_a",), _spec()
    )
    r2_model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R2", r2_observations, ("channel_a",), _spec()
    )

    assert r1_model.region_id == "R1"
    assert r2_model.region_id == "R2"
    assert r1_model.coefficients != r2_model.coefficients


def test_positive_and_negative_regions_are_not_averaged_to_zero():
    r1_observations = (
        _observation(
            "channel_a",
            1.0,
            (4.0, 0.0, 0.0),
            region_id="R1",
            comparison=_comparison_two_regions((4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
        ),
        _observation(
            "channel_a",
            2.0,
            (8.0, 0.0, 0.0),
            region_id="R1",
            comparison=_comparison_two_regions((8.0, 0.0, 0.0), (-8.0, 0.0, 0.0)),
        ),
    )
    r2_observations = (
        _observation(
            "channel_a",
            1.0,
            (-4.0, 0.0, 0.0),
            region_id="R2",
            comparison=_comparison_two_regions((4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
        ),
        _observation(
            "channel_a",
            2.0,
            (-8.0, 0.0, 0.0),
            region_id="R2",
            comparison=_comparison_two_regions((8.0, 0.0, 0.0), (-8.0, 0.0, 0.0)),
        ),
    )

    r1_model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", r1_observations, ("channel_a",), _spec()
    )
    r2_model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R2", r2_observations, ("channel_a",), _spec()
    )
    r1_prediction = predict_response(
        r1_model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),)
    )
    r2_prediction = predict_response(
        r2_model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),)
    )

    assert r1_prediction.predicted_delta_L > 0.0
    assert r2_prediction.predicted_delta_L < 0.0
    assert abs(r1_prediction.predicted_delta_L) > 1.0
    assert abs(r2_prediction.predicted_delta_L) > 1.0


def test_prediction_can_be_requested_for_a_specific_region():
    observations = (
        _observation(
            "channel_a",
            1.0,
            (2.0, 1.0, -1.0),
            region_id="R1",
            comparison=_comparison_two_regions((2.0, 1.0, -1.0), (-2.0, -1.0, 1.0)),
        ),
        _observation(
            "channel_a",
            2.0,
            (4.0, 2.0, -2.0),
            region_id="R1",
            comparison=_comparison_two_regions((4.0, 2.0, -2.0), (-4.0, -2.0, 2.0)),
        ),
    )
    model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a",), _spec()
    )
    prediction = predict_response(model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),))

    assert isinstance(prediction, SensitivityPrediction)
    assert prediction.region_id == "R1"
    assert prediction.predicted_delta_L == pytest.approx(2.0, abs=0.05)


def test_fit_and_predict_are_deterministic():
    observations = (
        _observation("channel_a", 1.0, (2.0, 1.0, -1.0)),
        _observation("channel_a", 2.0, (4.0, 2.0, -2.0)),
        _observation("channel_b", 1.0, (-1.0, 3.0, 2.0)),
        _observation("channel_b", 2.0, (-2.0, 6.0, 4.0)),
    )
    model_a = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a", "channel_b"), _spec()
    )
    model_b = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a", "channel_b"), _spec()
    )

    assert model_a == model_b
    prediction = predict_response(
        model_a,
        (
            ChannelAdjustment(channel_id="channel_a", delta=1.5),
            ChannelAdjustment(channel_id="channel_b", delta=0.5),
        ),
    )
    assert isinstance(prediction, SensitivityPrediction)
    assert prediction.predicted_delta_L == pytest.approx(2.5, abs=0.05)
    assert prediction.predicted_delta_a == pytest.approx(3.0, abs=0.05)
    assert prediction.predicted_delta_b == pytest.approx(-0.5, abs=0.05)
    assert prediction.uncertainty_std_L > 0.0
    assert prediction.uncertainty_std_a > 0.0
    assert prediction.uncertainty_std_b > 0.0


def test_model_is_local_to_sku_context_and_region():
    observation = _observation("channel_a", 1.0, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="sku_id"):
        fit_response_model(
            _context(), "OTHER_SKU", "R1", (observation, observation), ("channel_a",), _spec()
        )

    other_context = _context().model_copy(update={"context_id": "other_context"})
    with pytest.raises(ValueError, match="calibration context"):
        fit_response_model(
            other_context, "SKU_SYN_TEST", "R1", (observation, observation), ("channel_a",), _spec()
        )

    with pytest.raises(ValueError, match="region_id"):
        fit_response_model(
            _context(), "SKU_SYN_TEST", "R2", (observation, observation), ("channel_a",), _spec()
        )


def test_insufficient_observations_are_rejected():
    with pytest.raises(ValueError, match="at least 2 observations"):
        fit_response_model(
            _context(),
            "SKU_SYN_TEST",
            "R1",
            (_observation("channel_a", 1.0, (1.0, 0.0, 0.0)),),
            ("channel_a",),
            _spec(),
        )


def test_unknown_channel_in_observation_is_rejected():
    observations = (
        _observation("channel_a", 1.0, (1.0, 0.0, 0.0)),
        _observation("channel_b", 1.0, (0.0, 1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="unknown channel"):
        fit_response_model(_context(), "SKU_SYN_TEST", "R1", observations, ("channel_a",), _spec())


def test_duplicate_channel_ids_are_rejected():
    observations = (
        _observation("channel_a", 1.0, (1.0, 0.0, 0.0)),
        _observation("channel_a", 2.0, (2.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="duplicate channel_id"):
        fit_response_model(
            _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a", "channel_a"), _spec()
        )


def test_gloss_is_not_modelled():
    comparison = _comparison(1.0, 0.0, 0.0, gloss_delta=50.0)
    first = _observation("channel_a", 1.0, (1.0, 0.0, 0.0), comparison=comparison)
    second = _observation("channel_a", 2.0, (2.0, 0.0, 0.0))
    model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", (first, second), ("channel_a",), _spec()
    )
    prediction = predict_response(model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),))
    assert prediction.predicted_delta_L == pytest.approx(1.0, abs=0.05)
    assert prediction.predicted_delta_a == pytest.approx(0.0, abs=0.05)
    assert prediction.predicted_delta_b == pytest.approx(0.0, abs=0.05)


def test_zero_adjustment_predicts_zero_mean_response():
    observations = (
        _observation("channel_a", 1.0, (1.0, 2.0, 3.0)),
        _observation("channel_a", 2.0, (2.0, 4.0, 6.0)),
    )
    model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a",), _spec()
    )
    prediction = predict_response(model, ())
    assert prediction.adjustment == (0.0,)
    assert prediction.predicted_delta_L == 0.0
    assert prediction.predicted_delta_a == 0.0
    assert prediction.predicted_delta_b == 0.0


def test_non_synthetic_data_requires_non_placeholder_model_assumptions():
    observation = _observation("channel_a", 1.0, (1.0, 0.0, 0.0), data_origin=DataOrigin.CLIENT)
    with pytest.raises(ValueError, match="placeholder"):
        fit_response_model(
            _context(), "SKU_SYN_TEST", "R1", (observation, observation), ("channel_a",), _spec()
        )


def test_bayesian_uncertainty_is_positive_and_includes_observation_noise():
    observations = (
        _observation("channel_a", 1.0, (1.0, 0.0, 0.0)),
        _observation("channel_a", 2.0, (2.0, 0.0, 0.0)),
    )
    model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a",), _spec()
    )
    prediction = predict_response(model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),))

    assert prediction.uncertainty_std_L > 0.01
    assert prediction.uncertainty_std_a > 0.01
    assert prediction.uncertainty_std_b > 0.01


def test_result_models_are_immutable():
    observations = (
        _observation("channel_a", 1.0, (1.0, 0.0, 0.0)),
        _observation("channel_a", 2.0, (2.0, 0.0, 0.0)),
    )
    model = fit_response_model(
        _context(), "SKU_SYN_TEST", "R1", observations, ("channel_a",), _spec()
    )
    prediction = predict_response(model, (ChannelAdjustment(channel_id="channel_a", delta=1.0),))

    with pytest.raises(ValidationError):
        model.sku_id = "OTHER"
    with pytest.raises(ValidationError):
        prediction.predicted_delta_L = 99.0


def test_response_model_requires_positive_noise_and_prior():
    with pytest.raises(ValidationError):
        BayesianSensitivitySpec(
            prior_variance=0.0,
            observation_noise_std=(0.1, 0.1, 0.1),
            is_placeholder=True,
            source="synthetic test assumption",
        )

    with pytest.raises(ValidationError):
        BayesianSensitivitySpec(
            prior_variance=1.0,
            observation_noise_std=(0.1, 0.0, 0.1),
            is_placeholder=True,
            source="synthetic test assumption",
        )
