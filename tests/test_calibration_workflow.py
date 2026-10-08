"""Tests for the Chunk 5B calibration workflow.

Every value in this file is explicitly synthetic and is not a client printer,
measurement instrument, perturbation range, tolerance, or production result.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ats_ceramic.calibration_workflow import (
    CalibrationExperiment,
    CalibrationObservation,
    ControllableInput,
    HeldOutValidationResult,
    NoiseFloorEstimate,
    NoiseResponseAssessment,
    Perturbation,
    RecalibrationTrigger,
    ValidatedOperatingEnvelope,
    ValidationCriterion,
    ValidationStatus,
    assess_recalibration,
    assess_response_against_noise_floor,
    build_operating_envelopes,
    estimate_noise_floor,
    fit_calibration_models,
    split_held_out_by_run,
    to_response_observations,
    validate_held_out,
)
from ats_ceramic.color import ComparisonResult, RegionDifference
from ats_ceramic.response_model import BayesianSensitivitySpec, ResponseObservation
from ats_ceramic.schemas import (
    CalibrationContext,
    ChannelAdjustment,
    DataOrigin,
    MeasurementMetadata,
    RegionMeasurement,
    ReplicateSpread,
)

NOW = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def _context() -> CalibrationContext:
    return CalibrationContext(
        context_id="ctx_test_syn",
        printer_id="printer_test_syn",
        glaze_context="glaze_test_syn",
        ink_set_id="ink_test_syn",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _metadata() -> MeasurementMetadata:
    return MeasurementMetadata(
        instrument_id="instrument_test_syn",
        geometry="d/8",
        illuminant="D65",
        observer="2deg",
        specular_mode="SCI",
        aperture_mm=8.0,
        measured_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
    )


def _spec() -> BayesianSensitivitySpec:
    return BayesianSensitivitySpec(
        prior_variance=100.0,
        observation_noise_std=(0.01, 0.01, 0.01),
        min_observations=2,
        is_placeholder=True,
        source="synthetic test assumption",
    )


def _comparison(
    region_id: str,
    delta_l: float,
    delta_a: float = 0.0,
    delta_b: float = 0.0,
) -> ComparisonResult:
    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id=region_id,
                delta_L=delta_l,
                delta_a=delta_a,
                delta_b=delta_b,
                delta_E00=1.0,
                delta_gloss=0.0,
            ),
        ),
        mean_delta_E00=1.0,
        max_delta_E00=1.0,
        worst_region_id=region_id,
        mean_abs_delta_gloss=0.0,
        max_abs_delta_gloss=0.0,
    )


def _calibration_observation(
    *,
    region_id: str,
    run_id: str,
    delta_u: float,
    delta_l: float,
    experiment_id: str = "exp_syn_1",
) -> CalibrationObservation:
    response = ResponseObservation(
        sku_id="SKU_SYN_TEST",
        context_id="ctx_test_syn",
        region_id=region_id,
        adjustments=(ChannelAdjustment(channel_id="channel_a", delta=delta_u),),
        comparison=_comparison(region_id, delta_l),
        data_origin=DataOrigin.SYNTHETIC,
    )
    return CalibrationObservation(
        experiment_id=experiment_id,
        production_run_id=run_id,
        measured_at=NOW,
        response_observation=response,
        provenance="synthetic calibration test",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _input(channel_id: str = "channel_a") -> ControllableInput:
    return ControllableInput(
        channel_id=channel_id,
        controllable=True,
        lower_bound=None,
        upper_bound=None,
        nominal_value=None,
        unit=None,
        source="synthetic discovery test",
        client_confirmed=False,
        data_origin=DataOrigin.SYNTHETIC,
    )


def _experiment() -> CalibrationExperiment:
    return CalibrationExperiment(
        experiment_id="exp_syn_1",
        sku_id="SKU_SYN_TEST",
        calibration_context=_context(),
        region_ids=("R1", "R2"),
        controllable_inputs=(_input("channel_a"), _input("channel_b")),
        nominal_operating_point=(
            ChannelAdjustment(channel_id="channel_a", delta=0.0),
            ChannelAdjustment(channel_id="channel_b", delta=0.0),
        ),
        perturbations=(
            Perturbation(
                perturbation_id="p0",
                experiment_id="exp_syn_1",
                production_run_id="run_syn_1",
                channel_id="channel_a",
                delta=0.0,
                is_baseline=True,
            ),
            Perturbation(
                perturbation_id="p+",
                experiment_id="exp_syn_1",
                production_run_id="run_syn_1",
                channel_id="channel_a",
                delta=0.1,
            ),
            Perturbation(
                perturbation_id="p-",
                experiment_id="exp_syn_1",
                production_run_id="run_syn_1",
                channel_id="channel_a",
                delta=-0.1,
            ),
        ),
        production_run_id="run_syn_1",
        measurement_metadata=_metadata(),
        created_at=NOW,
        provenance="synthetic calibration experiment",
        data_origin=DataOrigin.SYNTHETIC,
    )


def test_experiment_represents_multiple_regions_runs_channels_and_provenance():
    experiment = _experiment()
    assert experiment.region_ids == ("R1", "R2")
    assert experiment.production_run_id == "run_syn_1"
    assert tuple(channel.channel_id for channel in experiment.controllable_inputs) == (
        "channel_a",
        "channel_b",
    )
    assert experiment.data_origin is DataOrigin.SYNTHETIC
    assert experiment.perturbations[0].is_baseline is True


def test_controllable_input_allows_unknown_bounds():
    channel = _input()
    assert channel.lower_bound is None
    assert channel.upper_bound is None
    assert channel.client_confirmed is False


def test_client_confirmed_input_requires_client_provenance():
    with pytest.raises(ValueError, match="CLIENT provenance"):
        ControllableInput(
            channel_id="channel_a",
            controllable=True,
            source="synthetic discovery test",
            client_confirmed=True,
            data_origin=DataOrigin.SYNTHETIC,
        )


def test_invalid_perturbations_are_rejected():
    with pytest.raises(ValidationError):
        Perturbation(
            perturbation_id="p_bad",
            experiment_id="exp_syn_1",
            production_run_id="run_syn_1",
            channel_id="channel_a",
            delta=0.0,
        )

    with pytest.raises(ValidationError):
        Perturbation(
            perturbation_id="p_bad",
            experiment_id="exp_syn_1",
            production_run_id="run_syn_1",
            channel_id="channel_a",
            delta=0.1,
            is_baseline=True,
        )


def test_experiment_rejects_unknown_perturbation_channel():
    experiment = _experiment()
    with pytest.raises(ValidationError, match="unknown perturbation channel"):
        CalibrationExperiment(
            **experiment.model_dump(exclude={"perturbations"}),
            perturbations=experiment.perturbations
            + (
                Perturbation(
                    perturbation_id="p_unknown",
                    experiment_id="exp_syn_1",
                    production_run_id="run_syn_1",
                    channel_id="channel_unknown",
                    delta=0.1,
                ),
            ),
        )


def test_calibration_observations_preserve_runs_and_convert_to_existing_response_observations():
    observations = (
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_1",
            delta_u=0.1,
            delta_l=1.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_2",
            delta_u=-0.1,
            delta_l=-1.0,
        ),
    )
    converted = to_response_observations(observations)
    assert len(converted) == 2
    assert converted[0].region_id == "R1"
    assert observations[0].production_run_id != observations[1].production_run_id


def test_noise_floor_is_region_specific_and_deterministic():
    measurements = (
        RegionMeasurement(
            region_id="R1",
            L=50.0,
            a=10.0,
            b=-2.0,
            gloss=40.0,
            n_replicates=3,
            replicate_spread=ReplicateSpread(
                L=0.2,
                a=0.1,
                b=0.3,
                gloss=0.4,
            ),
        ),
        RegionMeasurement(
            region_id="R1",
            L=51.0,
            a=11.0,
            b=-1.0,
            gloss=41.0,
            n_replicates=3,
            replicate_spread=ReplicateSpread(
                L=0.4,
                a=0.3,
                b=0.5,
                gloss=0.6,
            ),
        ),
    )
    first = estimate_noise_floor(
        measurements,
        _metadata(),
        NOW,
        "synthetic noise test",
        DataOrigin.SYNTHETIC,
    )
    second = estimate_noise_floor(
        measurements,
        _metadata(),
        NOW,
        "synthetic noise test",
        DataOrigin.SYNTHETIC,
    )
    assert isinstance(first, NoiseFloorEstimate)
    assert first == second
    assert first.region_id == "R1"
    assert first.noise_std_L == pytest.approx((0.2**2 + 0.4**2) ** 0.5 / 2**0.5)
    assert first.noise_std_gloss is not None


def test_noise_floor_requires_replicates():
    measurement = RegionMeasurement(
        region_id="R1",
        L=50.0,
        a=10.0,
        b=-2.0,
        gloss=40.0,
        n_replicates=1,
    )
    with pytest.raises(ValueError, match="replicate_spread"):
        estimate_noise_floor(
            (measurement,),
            _metadata(),
            NOW,
            "synthetic noise test",
            DataOrigin.SYNTHETIC,
        )


def test_noise_floor_rejects_mixed_regions():
    measurements = (
        RegionMeasurement(
            region_id="R1",
            L=50.0,
            a=10.0,
            b=-2.0,
            gloss=40.0,
            n_replicates=2,
            replicate_spread=ReplicateSpread(
                L=0.2,
                a=0.1,
                b=0.3,
                gloss=0.4,
            ),
        ),
        RegionMeasurement(
            region_id="R2",
            L=51.0,
            a=11.0,
            b=-1.0,
            gloss=41.0,
            n_replicates=2,
            replicate_spread=ReplicateSpread(
                L=0.2,
                a=0.1,
                b=0.3,
                gloss=0.4,
            ),
        ),
    )
    with pytest.raises(ValueError, match="one region at a time"):
        estimate_noise_floor(
            measurements,
            _metadata(),
            NOW,
            "synthetic noise test",
            DataOrigin.SYNTHETIC,
        )


def test_noise_response_assessment_reports_below_and_above_floor_without_significance_claims():
    noise_floor = NoiseFloorEstimate(
        region_id="R1",
        instrument_id="instrument_test_syn",
        n_measurements=2,
        replicate_count_total=4,
        noise_std_L=0.10,
        noise_std_a=0.10,
        noise_std_b=0.10,
        noise_std_gloss=0.20,
        estimation_method="synthetic test noise floor",
        estimated_at=NOW,
        provenance="synthetic noise response test",
        data_origin=DataOrigin.SYNTHETIC,
    )
    assessment = assess_response_against_noise_floor(
        noise_floor,
        observed_delta_L=0.05,
        observed_delta_a=-0.50,
        observed_delta_b=0.50,
    )
    assert isinstance(assessment, NoiseResponseAssessment)
    assert assessment.above_noise_floor_L is False
    assert assessment.above_noise_floor_a is True
    assert assessment.above_noise_floor_b is True


def test_noise_response_assessment_uses_absolute_effect_and_rejects_wrong_region():
    noise_floor = NoiseFloorEstimate(
        region_id="R1",
        instrument_id="instrument_test_syn",
        n_measurements=1,
        replicate_count_total=2,
        noise_std_L=0.10,
        noise_std_a=0.10,
        noise_std_b=0.10,
        estimation_method="synthetic test noise floor",
        estimated_at=NOW,
        provenance="synthetic noise response test",
        data_origin=DataOrigin.SYNTHETIC,
    )
    assessment = assess_response_against_noise_floor(
        noise_floor,
        observed_delta_L=-0.50,
        observed_delta_a=0.0,
        observed_delta_b=0.0,
    )
    assert assessment.above_noise_floor_L is True
    with pytest.raises(ValueError, match="region"):
        assess_response_against_noise_floor(
            noise_floor,
            observed_delta_L=0.5,
            observed_delta_a=0.0,
            observed_delta_b=0.0,
            region_id="R2",
        )


def test_fit_calibration_models_uses_existing_response_model_per_region():
    observations = (
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_1",
            delta_u=1.0,
            delta_l=4.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_2",
            delta_u=2.0,
            delta_l=8.0,
        ),
        _calibration_observation(
            region_id="R2",
            run_id="run_syn_1",
            delta_u=1.0,
            delta_l=-4.0,
        ),
        _calibration_observation(
            region_id="R2",
            run_id="run_syn_2",
            delta_u=2.0,
            delta_l=-8.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        observations,
        ("channel_a",),
        _spec(),
    )
    assert tuple(model.region_id for model in models) == ("R1", "R2")
    assert models[0].coefficients != models[1].coefficients
    assert all(model.data_origin is DataOrigin.SYNTHETIC for model in models)


def test_fit_calibration_models_rejects_mixed_sku_or_context():
    observation = _calibration_observation(
        region_id="R1",
        run_id="run_syn_1",
        delta_u=1.0,
        delta_l=4.0,
    )
    other_response = observation.response_observation.model_copy(update={"sku_id": "OTHER_SKU"})
    other = observation.model_copy(update={"response_observation": other_response})
    with pytest.raises(ValueError, match="one SKU"):
        fit_calibration_models(
            _context(),
            "SKU_SYN_TEST",
            (observation, other),
            ("channel_a",),
            _spec(),
        )


def test_held_out_split_is_grouped_by_production_run_without_leakage():
    observations = (
        _calibration_observation(
            region_id="R1",
            run_id="run_train",
            delta_u=1.0,
            delta_l=4.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_validation",
            delta_u=2.0,
            delta_l=8.0,
        ),
    )
    training, validation = split_held_out_by_run(
        observations,
        ("run_validation",),
    )
    assert tuple(observation.production_run_id for observation in training) == ("run_train",)
    assert tuple(observation.production_run_id for observation in validation) == ("run_validation",)


def test_held_out_split_rejects_unknown_or_all_runs():
    observation = _calibration_observation(
        region_id="R1",
        run_id="run_train",
        delta_u=1.0,
        delta_l=4.0,
    )
    with pytest.raises(ValueError, match="unknown validation run IDs"):
        split_held_out_by_run(
            (observation,),
            ("unknown",),
        )
    with pytest.raises(ValueError, match="at least one training"):
        split_held_out_by_run(
            (observation,),
            ("run_train",),
        )


def test_held_out_validation_is_deterministic_and_reports_prediction_uncertainty():
    training = (
        _calibration_observation(
            region_id="R1",
            run_id="run_train_1",
            delta_u=1.0,
            delta_l=4.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_train_2",
            delta_u=2.0,
            delta_l=8.0,
        ),
    )
    validation = (
        _calibration_observation(
            region_id="R1",
            run_id="run_validation",
            delta_u=1.5,
            delta_l=6.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        training,
        ("channel_a",),
        _spec(),
    )
    first = validate_held_out(models, training, validation)
    second = validate_held_out(models, training, validation)
    assert isinstance(first, HeldOutValidationResult)
    assert first == second
    assert first.status is ValidationStatus.EVALUATED_NO_CRITERION
    assert first.training_run_ids == ("run_train_1", "run_train_2")
    assert first.validation_run_ids == ("run_validation",)
    assert first.region_results[0].uncertainty_std_L > 0.0
    assert first.interval_coverage_L == 1.0
    assert first.mean_abs_error_L == pytest.approx(0.0, abs=2e-6)


def test_held_out_validation_rejects_run_leakage():
    observation = _calibration_observation(
        region_id="R1",
        run_id="run_same",
        delta_u=1.0,
        delta_l=4.0,
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        (
            observation,
            observation.model_copy(update={"experiment_id": "exp_2"}),
        ),
        ("channel_a",),
        _spec(),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        validate_held_out(models, (observation,), (observation,))


def _validation_fixture():
    training = (
        _calibration_observation(
            region_id="R1",
            run_id="run_train_1",
            delta_u=1.0,
            delta_l=4.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_train_2",
            delta_u=2.0,
            delta_l=8.0,
        ),
    )
    validation = (
        _calibration_observation(
            region_id="R1",
            run_id="run_validation_1",
            delta_u=1.5,
            delta_l=6.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_validation_2",
            delta_u=1.0,
            delta_l=4.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        training,
        ("channel_a",),
        _spec(),
    )
    return models, training, validation


def test_held_out_validation_with_explicit_criterion_passes():
    models, training, validation = _validation_fixture()
    criterion = ValidationCriterion(
        min_training_runs=2,
        min_validation_runs=2,
        min_validation_observations=2,
        max_mean_abs_error_L=0.01,
        min_interval_coverage_L=1.0,
    )
    result = validate_held_out(
        models,
        training,
        validation,
        criterion=criterion,
    )
    assert result.status is ValidationStatus.VALIDATED
    assert result.failure_reasons == ()
    assert result.mean_abs_error_L == pytest.approx(0.0, abs=2e-6)


def test_held_out_validation_with_explicit_criterion_fails_on_large_residual():
    models, training, validation = _validation_fixture()
    failing_validation = validation[0].model_copy(
        update={
            "response_observation": validation[0].response_observation.model_copy(
                update={"comparison": _comparison("R1", 20.0)}
            )
        }
    )
    criterion = ValidationCriterion(
        min_training_runs=2,
        min_validation_runs=2,
        min_validation_observations=2,
        max_mean_abs_error_L=0.5,
    )
    result = validate_held_out(
        models,
        training,
        (failing_validation, validation[1]),
        criterion=criterion,
    )
    assert result.status is ValidationStatus.FAILED
    assert len(result.failure_reasons) == 1
    assert result.failure_reasons[0].startswith("mean_abs_error_L exceeds criterion (")
    assert result.failure_reasons[0].endswith(" > 0.5)")


def test_held_out_validation_reports_insufficient_grouped_coverage_from_criterion():
    models, training, validation = _validation_fixture()
    criterion = ValidationCriterion(
        min_training_runs=3,
        min_validation_runs=3,
        min_validation_observations=3,
    )
    result = validate_held_out(
        models,
        training,
        validation,
        criterion=criterion,
    )
    assert result.status is ValidationStatus.INSUFFICIENT_COVERAGE
    assert result.failure_reasons == (
        "training run coverage below criterion (2 < 3)",
        "validation run coverage below criterion (2 < 3)",
        "validation observation coverage below criterion (2 < 3)",
    )


def test_held_out_validation_does_not_use_hidden_client_thresholds():
    models, training, validation = _validation_fixture()
    shifted = validation[0].model_copy(
        update={
            "response_observation": validation[0].response_observation.model_copy(
                update={"comparison": _comparison("R1", 6.2)}
            )
        }
    )
    permissive = ValidationCriterion(
        min_training_runs=1,
        min_validation_runs=1,
        min_validation_observations=1,
        max_mean_abs_error_L=0.2,
    )
    strict = ValidationCriterion(
        min_training_runs=1,
        min_validation_runs=1,
        min_validation_observations=1,
        max_mean_abs_error_L=0.05,
    )
    permissive_result = validate_held_out(
        models,
        training,
        (shifted, validation[1]),
        criterion=permissive,
    )
    strict_result = validate_held_out(
        models,
        training,
        (shifted, validation[1]),
        criterion=strict,
    )
    assert permissive_result.status is ValidationStatus.VALIDATED
    assert strict_result.status is ValidationStatus.FAILED


def test_operating_envelope_is_built_from_training_data_only():
    training = (
        _calibration_observation(
            region_id="R1",
            run_id="run_train_1",
            delta_u=-0.2,
            delta_l=-1.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_train_2",
            delta_u=0.3,
            delta_l=1.5,
        ),
    )
    validation = (
        _calibration_observation(
            region_id="R1",
            run_id="run_validation",
            delta_u=0.8,
            delta_l=4.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        training,
        ("channel_a",),
        _spec(),
    )
    validation_result = validate_held_out(
        models,
        training,
        validation,
        criterion=ValidationCriterion(
            min_training_runs=2,
            min_validation_runs=1,
            min_validation_observations=1,
            max_mean_abs_error_L=0.01,
        ),
    )
    assert validation_result.status is ValidationStatus.VALIDATED
    envelopes = build_operating_envelopes(
        models,
        training,
        validation_result,
        NOW,
    )
    envelope = envelopes[0]
    assert isinstance(envelope, ValidatedOperatingEnvelope)
    assert envelope.minimum_adjustment == (-0.2,)
    assert envelope.maximum_adjustment == (0.3,)
    assert envelope.contains((ChannelAdjustment(channel_id="channel_a", delta=0.0),))
    assert not envelope.contains((ChannelAdjustment(channel_id="channel_a", delta=0.8),))


def test_operating_envelope_preserves_nonvalidated_validation_statuses():
    training = (
        _calibration_observation(
            region_id="R1",
            run_id="run_train_1",
            delta_u=-0.2,
            delta_l=-1.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_train_2",
            delta_u=0.3,
            delta_l=1.5,
        ),
    )
    validation = (
        _calibration_observation(
            region_id="R1",
            run_id="run_validation",
            delta_u=0.8,
            delta_l=4.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        training,
        ("channel_a",),
        _spec(),
    )
    result = validate_held_out(models, training, validation)
    assert result.status is ValidationStatus.EVALUATED_NO_CRITERION

    for status in (
        ValidationStatus.INSUFFICIENT_COVERAGE,
        ValidationStatus.FAILED,
        ValidationStatus.EVALUATED_NO_CRITERION,
    ):
        carried = result.model_copy(update={"status": status})
        envelope = build_operating_envelopes(
            models,
            training,
            carried,
            NOW,
        )[0]
        assert envelope.validation_status is status
        assert envelope.validation_status is not ValidationStatus.VALIDATED


def test_recalibration_triggers_are_deterministic_and_do_not_retrain():
    assessment = assess_recalibration(
        outside_validated_envelope=True,
        model_stale=True,
        repeated_validation_failure=False,
    )
    assert assessment.triggered is True
    assert assessment.reasons == (
        RecalibrationTrigger.OUTSIDE_VALIDATED_ENVELOPE,
        RecalibrationTrigger.MODEL_STALE,
    )

    healthy = assess_recalibration()
    assert healthy.triggered is False
    assert healthy.reasons == ()


def test_synthetic_provenance_is_preserved_through_workflow():
    observations = (
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_1",
            delta_u=1.0,
            delta_l=4.0,
        ),
        _calibration_observation(
            region_id="R1",
            run_id="run_syn_2",
            delta_u=2.0,
            delta_l=8.0,
        ),
    )
    models = fit_calibration_models(
        _context(),
        "SKU_SYN_TEST",
        observations,
        ("channel_a",),
        _spec(),
    )
    training, validation = split_held_out_by_run(
        observations,
        ("run_syn_2",),
    )
    result = validate_held_out(
        models,
        training,
        validation,
        criterion=ValidationCriterion(
            min_training_runs=1,
            min_validation_runs=1,
            min_validation_observations=1,
            max_mean_abs_error_L=0.01,
        ),
    )
    envelope = build_operating_envelopes(models, training, result, NOW)[0]

    assert all(model.data_origin is DataOrigin.SYNTHETIC for model in models)
    assert result.data_origin is DataOrigin.SYNTHETIC
    assert envelope.data_origin is DataOrigin.SYNTHETIC
