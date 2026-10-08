"""Focused tests for the optional gated ML residual layer.

All fixtures are explicitly synthetic and are not client evidence.
"""

from datetime import UTC, datetime

import pytest

from ats_ceramic.calibration_workflow import (
    CalibrationObservation,
    ValidatedOperatingEnvelope,
)
from ats_ceramic.color import ComparisonResult, RegionDifference
from ats_ceramic.ml_residual import (
    AblationReport,
    ResidualConfig,
    ResidualGateStatus,
    ResidualPrediction,
    apply_residual,
    fit_residual_model,
    split_and_fit_residual_model,
)
from ats_ceramic.response_model import (
    BayesianSensitivitySpec,
    ResponseObservation,
    fit_response_model,
    predict_response,
)
from ats_ceramic.schemas import CalibrationContext, ChannelAdjustment, DataOrigin

NOW = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
PROVENANCE = "synthetic_generator:v1:ml_residual_test"


def _context() -> CalibrationContext:
    return CalibrationContext(
        context_id="CTX_SYN_ML",
        printer_id="PRINTER_SYN_ML",
        glaze_context="GLAZE_SYN_ML",
        ink_set_id="INK_SYN_ML",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _comparison(x: float, region_id: str = "R1") -> ComparisonResult:
    # Deliberately nonlinear observed response: empirical model learns the
    # linear component while the residual layer sees the remaining curvature.
    delta_l = 1.5 * x + 0.35 * x * x
    delta_a = -0.5 * x + 0.10 * x * x
    delta_b = 0.7 * x - 0.12 * x * x

    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id=region_id,
                delta_L=delta_l,
                delta_a=delta_a,
                delta_b=delta_b,
                delta_E00=abs(delta_l),
                delta_gloss=0.0,
            ),
        ),
        mean_delta_E00=abs(delta_l),
        max_delta_E00=abs(delta_l),
        worst_region_id=region_id,
        mean_abs_delta_gloss=0.0,
        max_abs_delta_gloss=0.0,
    )


def _observation(
    x: float,
    run: str,
    experiment: str | None = None,
    *,
    origin=DataOrigin.SYNTHETIC,
    provenance=PROVENANCE,
):
    response = ResponseObservation(
        sku_id="SKU_SYN_ML",
        context_id="CTX_SYN_ML",
        region_id="R1",
        adjustments=(ChannelAdjustment(channel_id="channel_1", delta=x),),
        comparison=_comparison(x),
        data_origin=origin,
    )

    return CalibrationObservation(
        experiment_id=experiment or f"EXP_{run}_{x}",
        production_run_id=run,
        measured_at=NOW,
        response_observation=response,
        provenance=provenance,
        data_origin=origin,
    )


def _observation_with_deltas(
    x: float,
    run: str,
    deltas: tuple[float, float, float],
) -> CalibrationObservation:
    response = ResponseObservation(
        sku_id="SKU_SYN_ML",
        context_id="CTX_SYN_ML",
        region_id="R1",
        adjustments=(ChannelAdjustment(channel_id="channel_1", delta=x),),
        comparison=ComparisonResult(
            regions=(
                RegionDifference(
                    region_id="R1",
                    delta_L=deltas[0],
                    delta_a=deltas[1],
                    delta_b=deltas[2],
                    delta_E00=abs(deltas[0]),
                    delta_gloss=0.0,
                ),
            ),
            mean_delta_E00=abs(deltas[0]),
            max_delta_E00=abs(deltas[0]),
            worst_region_id="R1",
            mean_abs_delta_gloss=0.0,
            max_abs_delta_gloss=0.0,
        ),
        data_origin=DataOrigin.SYNTHETIC,
    )

    return CalibrationObservation(
        experiment_id=f"EXP_{run}_{x}",
        production_run_id=run,
        measured_at=NOW,
        response_observation=response,
        provenance=PROVENANCE,
        data_origin=DataOrigin.SYNTHETIC,
    )


def _empirical_model(observations: tuple[CalibrationObservation, ...]):
    spec = BayesianSensitivitySpec(
        prior_variance=100.0,
        observation_noise_std=(0.1, 0.1, 0.1),
        min_observations=2,
        is_placeholder=True,
        source="synthetic empirical model test",
    )

    return fit_response_model(
        _context(),
        "SKU_SYN_ML",
        "R1",
        tuple(item.response_observation for item in observations),
        ("channel_1",),
        spec,
    )


def _envelope(model, observations):
    values = [
        item.response_observation.adjustments[0].delta
        for item in observations
    ]

    return ValidatedOperatingEnvelope(
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        minimum_adjustment=(min(values),),
        maximum_adjustment=(max(values),),
        n_observations=len(observations),
        validation_status="validated",
        production_run_ids=tuple(
            dict.fromkeys(item.production_run_id for item in observations)
        ),
        calibrated_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
        model_version=None,
    )


def _dataset():
    training = tuple(
        _observation(x, f"RUN_{i // 2}")
        for i, x in enumerate(
            (-1.4, -1.1, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.1, 1.4)
        )
    )

    held_out = tuple(
        _observation(x, f"HOLD_{i}")
        for i, x in enumerate((-1.25, -0.75, 0.35, 1.25))
    )

    model = _empirical_model(training)

    return training, held_out, model


def _config(**overrides):
    values = {
        "provenance": PROVENANCE,
        "min_observations": 8,
        "min_relative_improvement": 0.0,
        "max_residual_abs": 1.0,
        "random_state": 7,
        "n_estimators": 32,
        "max_depth": 3,
        "min_samples_leaf": 1,
    }

    values.update(overrides)

    return ResidualConfig(**values)


def _enabled_result():
    training, held_out, model = _dataset()

    residual = fit_residual_model(
        training,
        held_out,
        model,
        _config(),
    )

    assert residual.result.ablation is not None
    assert residual.result.ablation.acceptance_passed is True

    return training, held_out, model, residual


def test_residual_target_is_observed_minus_empirical_prediction():
    training, _, model = _dataset()
    observation = training[-1]

    observed = observation.response_observation.comparison.region("R1")
    prediction = predict_response(
        model,
        observation.response_observation.adjustments,
    )

    expected = (
        observed.delta_L - prediction.predicted_delta_L,
        observed.delta_a - prediction.predicted_delta_a,
        observed.delta_b - prediction.predicted_delta_b,
    )

    from ats_ceramic.ml_residual import _residual_target

    assert _residual_target(observation, model) == pytest.approx(expected)


def test_insufficient_observations_do_not_train():
    training, held_out, model = _dataset()

    result = fit_residual_model(
        training[:4],
        held_out,
        model,
        _config(min_observations=5),
    )

    assert result.result.status is ResidualGateStatus.INSUFFICIENT_DATA
    assert result.result.model_ready is False


def test_baseline_is_required():
    training, held_out, _ = _dataset()

    result = fit_residual_model(
        training,
        held_out,
        None,
        _config(),
    )

    assert result.result.status is ResidualGateStatus.BASELINE_UNAVAILABLE
    assert result.result.model_ready is False


def test_synthetic_origin_and_provenance_are_preserved():
    training, held_out, model = _dataset()

    result = fit_residual_model(
        training,
        held_out,
        model,
        _config(),
    )

    assert result.result.data_origin is DataOrigin.SYNTHETIC
    assert result.result.provenance == PROVENANCE
    assert result.result.ablation is not None
    assert result.result.ablation.data_origin is DataOrigin.SYNTHETIC


def test_mixed_data_origins_are_rejected():
    training, held_out, model = _dataset()

    mixed = training[:-1] + (
        _observation(
            1.5,
            "RUN_MIX",
            origin=DataOrigin.CLIENT,
            provenance="client-calibration-source",
        ),
    )

    with pytest.raises(ValueError, match="one data origin"):
        fit_residual_model(
            mixed,
            held_out,
            model,
            _config(),
        )


def test_provenance_mismatch_disables_residual():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance="synthetic_generator:v1:other",
        operating_envelope=_envelope(model, training),
    )

    assert result.gate_status is ResidualGateStatus.PROVENANCE_MISMATCH
    assert result.residual_enabled is False
    assert result.combined_prediction == result.empirical_prediction


def test_channel_mismatch_disables_residual():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=("other_channel",),
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.gate_status is ResidualGateStatus.CHANNEL_MISMATCH


def test_sku_context_region_mismatch_disables_residual():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id="OTHER_SKU",
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert (
        result.gate_status
        is ResidualGateStatus.SKU_CONTEXT_REGION_MISMATCH
    )


def test_out_of_domain_disables_residual():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        (ChannelAdjustment(channel_id="channel_1", delta=2.0),),
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=(
            ChannelAdjustment(channel_id="channel_1", delta=2.0),
        ),
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.gate_status is ResidualGateStatus.OUT_OF_DOMAIN


def test_residual_safety_bound_disables_without_clipping():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    unsafe = _config(max_residual_abs=1e-6)

    unsafe_model = fit_residual_model(
        training,
        held_out,
        model,
        unsafe,
    )

    # The tight bound may fail during ablation; either way a disabled fit
    # cannot activate.
    result = apply_residual(
        unsafe_model,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.residual_enabled is False
    assert result.combined_prediction == result.empirical_prediction
    assert result.residual_prediction == (0.0, 0.0, 0.0)


def test_successful_gate_returns_empirical_plus_residual():
    training, held_out, model, residual = _enabled_result()
    observation = held_out[1]

    prediction = predict_response(
        model,
        observation.response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=observation.response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.gate_status is ResidualGateStatus.ENABLED
    assert result.residual_enabled is True
    assert result.combined_prediction == pytest.approx(
        tuple(
            result.empirical_prediction[i] + result.residual_prediction[i]
            for i in range(3)
        )
    )


def test_disabled_gate_falls_back_exactly_to_empirical_prediction():
    training, held_out, model = _dataset()

    residual = fit_residual_model(
        training[:4],
        held_out,
        model,
        _config(min_observations=8),
    )

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.residual_enabled is False
    assert result.combined_prediction == result.empirical_prediction


def test_original_empirical_prediction_is_not_mutated():
    training, held_out, model, residual = _enabled_result()

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    before = prediction

    apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert prediction == before


def test_ablation_uses_held_out_data_and_no_training_overlap():
    training, held_out, model = _dataset()

    residual = fit_residual_model(
        training,
        held_out,
        model,
        _config(),
    )

    assert residual.result.training_observation_count == len(training)
    assert residual.result.held_out_observation_count == len(held_out)
    assert not any(
        residual.was_training_observation(item)
        for item in held_out
    )
    assert all(
        residual.was_training_observation(item)
        for item in training
    )


def test_ablation_reports_required_metrics():
    training, held_out, model, residual = _enabled_result()

    report = residual.result.ablation

    assert isinstance(report, AblationReport)
    assert report.baseline_mse >= 0.0
    assert report.residual_mse >= 0.0
    assert report.absolute_improvement == pytest.approx(
        report.baseline_mse - report.residual_mse
    )


def test_worse_held_out_ablation_disables_residual_and_keeps_baseline():
    training, _, model = _dataset()

    held_out = tuple(
        _observation_with_deltas(
            x,
            f"WORSE_{i}",
            (-0.6, 0.2, -0.2),
        )
        for i, x in enumerate((-1.2, -0.6, 0.4, 1.2))
    )

    residual = fit_residual_model(
        training,
        held_out,
        model,
        _config(min_relative_improvement=0.0),
    )

    report = residual.result.ablation

    assert report is not None
    assert report.residual_mse > report.baseline_mse
    assert residual.result.status is ResidualGateStatus.VALIDATION_FAILED

    prediction = predict_response(
        model,
        held_out[0].response_observation.adjustments,
    )

    result = apply_residual(
        residual,
        prediction,
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        adjustments=held_out[0].response_observation.adjustments,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
        operating_envelope=_envelope(model, training),
    )

    assert result.residual_enabled is False
    assert result.combined_prediction == result.empirical_prediction


def test_failed_ablation_disables_residual():
    training, held_out, model = _dataset()

    residual = fit_residual_model(
        training,
        held_out,
        model,
        _config(min_relative_improvement=10.0),
    )

    assert residual.result.status is ResidualGateStatus.VALIDATION_FAILED
    assert residual.result.model_ready is False


def test_deterministic_random_state_produces_repeatable_ablation():
    training, held_out, model = _dataset()

    first = fit_residual_model(
        training,
        held_out,
        model,
        _config(random_state=13),
    )

    second = fit_residual_model(
        training,
        held_out,
        model,
        _config(random_state=13),
    )

    assert first.result == second.result


def test_feature_set_contains_only_observed_numeric_features():
    training, held_out, model = _dataset()

    residual = fit_residual_model(
        training,
        held_out,
        model,
        _config(),
    )

    assert residual.result.feature_names == (
        "channel_1",
        "empirical_delta_L",
        "empirical_delta_a",
        "empirical_delta_b",
    )

    assert all(
        "scenario" not in name
        for name in residual.result.feature_names
    )

    assert all(
        "oracle" not in name
        for name in residual.result.feature_names
    )

    assert all(
        "process" not in name
        for name in residual.result.feature_names
    )


def test_model_and_result_are_immutable():
    training, held_out, model, residual = _enabled_result()

    with pytest.raises(AttributeError):
        residual.result = residual.result  # type: ignore[misc]

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        residual.result.model_ready = False  # type: ignore[misc]

    with pytest.raises(AttributeError):
        residual.random_state = 123  # type: ignore[attr-defined]


def test_no_client_tolerance_is_introduced():
    assert "delta_e00_threshold" not in ResidualConfig.model_fields
    assert "gloss_abs_threshold" not in ResidualConfig.model_fields
    assert "tolerance" not in ResidualConfig.model_fields


def test_split_helper_uses_existing_grouped_run_split():
    training, held_out, model = _dataset()
    all_observations = training + held_out

    residual = split_and_fit_residual_model(
        all_observations,
        ("HOLD_0", "HOLD_1", "HOLD_2", "HOLD_3"),
        model,
        _config(),
    )

    assert residual.result.training_observation_count == len(training)
    assert residual.result.held_out_observation_count == len(held_out)


def test_no_automatic_retraining_occurs():
    training, held_out, model, residual = _enabled_result()

    before = residual.result

    result = residual._predict_residual((0.25, 0.1, -0.1, 0.05))

    assert result
    assert residual.result == before


def test_residual_prediction_is_immutable():
    prediction = ResidualPrediction(
        region_id="R1",
        channel_ids=("channel_1",),
        empirical_prediction=(1.0, 2.0, 3.0),
        residual_prediction=(0.1, 0.2, 0.3),
        combined_prediction=(1.1, 2.2, 3.3),
        residual_enabled=True,
        gate_status=ResidualGateStatus.ENABLED,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=PROVENANCE,
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        prediction.residual_enabled = False  # type: ignore[misc]
