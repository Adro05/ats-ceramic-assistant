"""Focused tests for the deterministic synthetic generator."""

import pytest

from ats_ceramic.schemas import DataOrigin
from ats_ceramic.synthetic import (
    SyntheticConfig,
    SyntheticScenario,
    _oracle_delta,
    generate,
)

SCENARIOS = tuple(SyntheticScenario)


def test_every_required_scenario_can_be_generated():
    for scenario in SCENARIOS:
        dataset = generate(SyntheticConfig(seed=7), scenario)
        assert dataset.scenario is scenario
        assert dataset.target_observation.scenario is scenario


def test_every_generated_artifact_is_synthetic_and_provenance_is_scenario_specific():
    for scenario in SCENARIOS:
        dataset = generate(SyntheticConfig(seed=7), scenario)
        assert dataset.data_origin is DataOrigin.SYNTHETIC
        assert dataset.target_observation.data_origin is DataOrigin.SYNTHETIC
        assert all(
            item.data_origin is DataOrigin.SYNTHETIC
            for item in dataset.calibration_observations
        )
        assert dataset.provenance == f"synthetic_generator:v1:{scenario.value}"


def test_same_seed_and_config_produce_identical_output():
    config = SyntheticConfig(seed=123)
    assert generate(config, SyntheticScenario.CORRECTABLE) == generate(
        config, SyntheticScenario.CORRECTABLE
    )


def test_different_seed_can_produce_different_observations():
    first = generate(SyntheticConfig(seed=1), SyntheticScenario.CORRECTABLE)
    second = generate(SyntheticConfig(seed=2), SyntheticScenario.CORRECTABLE)
    assert first != second
    assert (
        first.calibration_observations[0].observation_id
        == second.calibration_observations[0].observation_id
    )
    assert (
        first.calibration_observations[0].observed_L
        != second.calibration_observations[0].observed_L
    )


def test_generated_ids_are_deterministic_under_same_seed():
    first = generate(SyntheticConfig(seed=19), SyntheticScenario.CORRECTABLE)
    second = generate(SyntheticConfig(seed=19), SyntheticScenario.CORRECTABLE)
    assert tuple(x.observation_id for x in first.calibration_observations) == tuple(
        x.observation_id for x in second.calibration_observations
    )
    assert first.target_observation.batch_id == second.target_observation.batch_id


def test_correctable_scenario_has_controllable_colour_deviation():
    dataset = generate(SyntheticConfig(seed=4), SyntheticScenario.CORRECTABLE)
    target = dataset.target_observation
    master = dataset.master_measurement.regions[0]
    assert abs(target.observed_L - master.L) > 0.1
    assert any(abs(value) > 0.0 for value in dataset.target_required_adjustment)


def test_near_boundary_scenario_requires_adjustment_close_to_configured_bounds():
    config = SyntheticConfig(seed=4)
    dataset = generate(config, SyntheticScenario.NEAR_BOUNDARY)
    for required, upper in zip(
        dataset.target_required_adjustment,
        config.channel_upper_bounds,
        strict=True
    ):
        assert required == pytest.approx(0.95 * upper)
        assert abs(upper - required) <= 0.10 * abs(upper)


def test_process_side_gloss_scenario_has_separate_process_gloss_component():
    dataset = generate(
        SyntheticConfig(seed=4),
        SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE,
    )
    target = dataset.target_observation
    assert target.process_state.gloss_process_variable != 0.0
    assert target.observed_gloss > 60.0
    assert "process-side" in target.expected_condition


def test_unreachable_scenario_requires_correction_outside_allowed_range():
    config = SyntheticConfig(seed=4)
    dataset = generate(config, SyntheticScenario.UNREACHABLE)
    for required, lower, upper in zip(
        dataset.target_required_adjustment,
        config.channel_lower_bounds,
        config.channel_upper_bounds,
        strict=True
    ):
        assert required > upper or required < lower


def test_unseen_sku_ood_is_absent_from_calibration_training_records():
    dataset = generate(
        SyntheticConfig(seed=4),
        SyntheticScenario.UNSEEN_SKU_OOD,
    )
    calibration_skus = {
        item.sku_id for item in dataset.calibration_observations
    }
    calibration_contexts = {
        item.context_id for item in dataset.calibration_observations
    }
    assert dataset.target_observation.sku_id not in calibration_skus
    assert dataset.target_observation.context_id not in calibration_contexts


def test_noisy_corrupted_scenario_contains_explicitly_invalid_measurement_without_clamping():
    dataset = generate(
        SyntheticConfig(seed=4),
        SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT,
    )
    target = dataset.target_observation
    assert target.corrupted is True
    assert target.observed_L == 105.0
    assert target.observed_L > 100.0
    with pytest.raises(ValueError, match="corrupted"):
        target.as_region_measurement()


def test_sparse_calibration_has_fewer_observations_than_normal():
    config = SyntheticConfig(seed=4, calibration_observations=8)
    normal = generate(config, SyntheticScenario.CORRECTABLE)
    sparse = generate(config, SyntheticScenario.SPARSE_CALIBRATION)
    assert len(sparse.calibration_observations) < len(
        normal.calibration_observations
    )
    assert len(sparse.calibration_observations) == 2 * len(config.region_ids)


def test_valid_generated_lab_values_are_physically_sensible():
    for scenario in SCENARIOS:
        dataset = generate(SyntheticConfig(seed=8), scenario)
        records = dataset.calibration_observations + (
            dataset.target_observation,
        )
        for record in records:
            if record.corrupted:
                continue
            assert 0.0 <= record.observed_L <= 100.0
            from math import isfinite

            assert isfinite(record.observed_a)
            assert isfinite(record.observed_b)


def test_valid_generated_gloss_is_finite_and_non_negative():
    for scenario in SCENARIOS:
        dataset = generate(SyntheticConfig(seed=8), scenario)
        records = dataset.calibration_observations + (
            dataset.target_observation,
        )
        for record in records:
            assert record.observed_gloss >= 0.0
            assert record.observed_gloss == pytest.approx(record.observed_gloss)


def test_scenario_ground_truth_is_separate_from_observed_pipeline_values():
    dataset = generate(
        SyntheticConfig(seed=10),
        SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE,
    )
    observation = dataset.target_observation
    assert observation.scenario is SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE
    assert observation.expected_condition
    assert observation.observed_gloss is not None
    assert observation.process_state.gloss_process_variable != 0.0


def test_generator_never_produces_client_origin():
    for scenario in SCENARIOS:
        dataset = generate(SyntheticConfig(seed=11), scenario)
        assert dataset.data_origin is not DataOrigin.CLIENT
        assert dataset.context.data_origin is DataOrigin.SYNTHETIC
        assert all(
            item.data_origin is not DataOrigin.CLIENT
            for item in dataset.calibration_observations
        )


def test_synthetic_configuration_has_no_client_tolerance_fields():
    fields = SyntheticConfig.model_fields
    assert "tolerance" not in fields
    assert "delta_e00_threshold" not in fields
    assert "gloss_abs_threshold" not in fields


def test_hidden_oracle_is_nonlinear_and_saturating():
    low = _oracle_delta((1.0, 1.0), (0.0, 0.0, 0.0))
    high = _oracle_delta((20.0, 20.0), (0.0, 0.0, 0.0))
    much_higher = _oracle_delta((200.0, 200.0), (0.0, 0.0, 0.0))
    assert high[0] < 4.0
    assert much_higher[0] == pytest.approx(high[0], rel=0.01)
    assert high[0] < 20.0 * low[0]


def test_generated_models_are_immutable():
    dataset = generate(
        SyntheticConfig(seed=5),
        SyntheticScenario.CORRECTABLE,
    )
    with pytest.raises((TypeError, ValueError)):
        dataset.target_observation.observed_L = 1.0


def test_generation_does_not_modify_global_random_state():
    import random

    random.seed(999)
    before = random.getstate()
    generate(SyntheticConfig(seed=12), SyntheticScenario.CORRECTABLE)
    after = random.getstate()
    assert before == after


def test_valid_observation_maps_into_existing_batch_measurement_schema():
    dataset = generate(
        SyntheticConfig(seed=5),
        SyntheticScenario.CORRECTABLE,
    )
    batch = dataset.target_observation.as_batch_measurement()
    assert batch.batch_id == dataset.target_observation.batch_id
    assert batch.sku_id == dataset.target_observation.sku_id
    assert batch.context_id == dataset.target_observation.context_id
    assert batch.regions[0].region_id == dataset.target_observation.region_id


def test_config_is_immutable_and_generation_does_not_mutate_it():
    config = SyntheticConfig(seed=5)
    before = config
    generate(config, SyntheticScenario.CORRECTABLE)
    assert config == before