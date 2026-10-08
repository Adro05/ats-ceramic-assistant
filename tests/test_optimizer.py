"""Tests for Chunk 6's constrained optimizer.

All measurement, response, objective-weight, bound, and envelope numbers below are
explicitly synthetic fixtures.  The CLIENT provenance used on controllable inputs
only exercises the existing schema's client-confirmation state; it is not client
data or a client printer setting.
"""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from ats_ceramic.calibration_workflow import ControllableInput, ValidatedOperatingEnvelope
from ats_ceramic.color import delta_e_2000
from ats_ceramic.optimizer import (
    OptimizationTarget,
    OptimizerObjectiveSpec,
    OptimizerReason,
    OptimizerStatus,
    optimize_region,
)
from ats_ceramic.response_model import (
    BayesianSensitivitySpec,
    ResponseModel,
    ResponseObservation,
    fit_response_model,
)
from ats_ceramic.schemas import (
    CalibrationContext,
    ChannelAdjustment,
    DataOrigin,
    RegionMeasurement,
)

NOW = "2026-01-01T08:00:00+00:00"


def _context() -> CalibrationContext:
    return CalibrationContext(
        context_id="ctx_test_syn",
        printer_id="printer_test_syn",
        glaze_context="glaze_test_syn",
        ink_set_id="ink_test_syn",
        data_origin=DataOrigin.SYNTHETIC,
    )


def _comparison(region_id: str, delta_l: float, delta_a: float = 0.0, delta_b: float = 0.0):
    from ats_ceramic.color import ComparisonResult, RegionDifference

    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id=region_id,
                delta_L=delta_l,
                delta_a=delta_a,
                delta_b=delta_b,
                delta_E00=delta_e_2000((50.0, 0.0, 0.0), (50.0 + delta_l, delta_a, delta_b)),
                delta_gloss=0.0,
            ),
        ),
        mean_delta_E00=1.0,
        max_delta_E00=1.0,
        worst_region_id=region_id,
        mean_abs_delta_gloss=0.0,
        max_abs_delta_gloss=0.0,
    )


def _observation(
    channel_ids: tuple[str, ...],
    deltas: tuple[tuple[float, float, float], ...],
    scale: float,
):
    adjustments = tuple(
        ChannelAdjustment(channel_id=channel_id, delta=scale) for channel_id in channel_ids
    )
    total = tuple(sum(values[index] * scale for values in deltas) for index in range(3))
    return ResponseObservation(
        sku_id="SKU_SYN_TEST",
        context_id="ctx_test_syn",
        region_id="R1",
        adjustments=adjustments,
        comparison=_comparison("R1", total[0], total[1], total[2]),
        data_origin=DataOrigin.SYNTHETIC,
    )


def _model(
    *,
    region_id: str = "R1",
    coefficients: tuple[tuple[float, float, float], ...] = ((2.0, 0.0, 0.0),),
) -> ResponseModel:
    channel_ids = tuple(f"channel_{index + 1}" for index in range(len(coefficients)))
    observations = (
        ResponseObservation(
            sku_id="SKU_SYN_TEST",
            context_id="ctx_test_syn",
            region_id=region_id,
            adjustments=tuple(ChannelAdjustment(channel_id=cid, delta=1.0) for cid in channel_ids),
            comparison=_comparison(
                region_id,
                sum(row[0] for row in coefficients),
                sum(row[1] for row in coefficients),
                sum(row[2] for row in coefficients),
            ),
            data_origin=DataOrigin.SYNTHETIC,
        ),
        ResponseObservation(
            sku_id="SKU_SYN_TEST",
            context_id="ctx_test_syn",
            region_id=region_id,
            adjustments=tuple(ChannelAdjustment(channel_id=cid, delta=2.0) for cid in channel_ids),
            comparison=_comparison(
                region_id,
                2.0 * sum(row[0] for row in coefficients),
                2.0 * sum(row[1] for row in coefficients),
                2.0 * sum(row[2] for row in coefficients),
            ),
            data_origin=DataOrigin.SYNTHETIC,
        ),
    )
    return fit_response_model(
        _context(),
        "SKU_SYN_TEST",
        region_id,
        observations,
        channel_ids,
        BayesianSensitivitySpec(
            prior_variance=100.0,
            observation_noise_std=(0.01, 0.01, 0.01),
            min_observations=2,
            is_placeholder=True,
            source="synthetic optimizer test assumption",
        ),
    )


def _target(
    region_id: str = "R1", batch_L: float = 54.0, master_L: float = 50.0
) -> OptimizationTarget:
    return OptimizationTarget(
        master=RegionMeasurement(region_id=region_id, L=master_L, a=0.0, b=0.0, gloss=40.0),
        batch=RegionMeasurement(region_id=region_id, L=batch_L, a=0.0, b=0.0, gloss=40.0),
        data_origin=DataOrigin.SYNTHETIC,
    )


def _input(
    channel_id: str, *, lower: float = -5.0, upper: float = 5.0, confirmed: bool = True
) -> ControllableInput:
    return ControllableInput(
        channel_id=channel_id,
        controllable=True,
        lower_bound=lower,
        upper_bound=upper,
        nominal_value=0.0,
        unit="synthetic_unit",
        source="test-only client-confirmation state; not client data",
        client_confirmed=confirmed,
        data_origin=DataOrigin.CLIENT,
    )


def _objective(
    *,
    size: float = 0.0,
    uncertainty: float = 0.0,
    improvement: float = 0.0,
    trust: float = 5.0,
) -> OptimizerObjectiveSpec:
    return OptimizerObjectiveSpec(
        delta_e_weight=1.0,
        correction_size_weight=size,
        uncertainty_weight=uncertainty,
        minimum_objective_improvement=improvement,
        trust_region_radius=trust,
        provenance="synthetic optimizer test configuration",
        is_placeholder=True,
    )


def _optimize(model, target, inputs, objective=None, **kwargs):
    return optimize_region(
        model,
        target,
        inputs,
        objective or _objective(),
        provenance="synthetic optimizer test",
        **kwargs,
    )


def test_zero_change_baseline_is_always_reported_and_simple_one_channel_direction_is_correct():
    model = _model()
    result = _optimize(model, _target(), (_input("channel_1"),))

    assert result.baseline.predicted_delta_E00 > 0.0
    assert result.status is OptimizerStatus.RECOMMEND
    assert result.recommended_adjustments[0].delta < 0.0
    assert result.candidate.predicted_L < result.baseline.predicted_L
    assert result.candidate.predicted_delta_E00 < result.baseline.predicted_delta_E00


def test_lab_sign_convention_is_batch_minus_master_and_optimizer_moves_back_toward_master():
    model = _model()
    target = _target(batch_L=54.0, master_L=50.0)
    result = _optimize(model, target, (_input("channel_1"),))

    assert target.batch.L - target.master.L == 4.0
    assert result.recommended_adjustments[0].delta < 0.0


def test_multiple_channels_are_optimized_without_averaging_regions():
    model = _model(coefficients=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    target = OptimizationTarget(
        master=RegionMeasurement(region_id="R1", L=50.0, a=0.0, b=0.0, gloss=40.0),
        batch=RegionMeasurement(region_id="R1", L=54.0, a=4.0, b=0.0, gloss=40.0),
        data_origin=DataOrigin.SYNTHETIC,
    )
    result = _optimize(model, target, (_input("channel_1"), _input("channel_2")))

    deltas = {item.channel_id: item.delta for item in result.recommended_adjustments}
    assert deltas["channel_1"] < 0.0
    assert deltas["channel_2"] < 0.0


def test_per_region_isolation_uses_region_specific_model():
    r1 = _model(region_id="R1", coefficients=((2.0, 0.0, 0.0),))
    r2 = _model(region_id="R2", coefficients=((-2.0, 0.0, 0.0),))
    inputs = (_input("channel_1"),)

    result_r1 = _optimize(r1, _target("R1"), inputs)
    result_r2 = _optimize(r2, _target("R2"), inputs)

    assert result_r1.recommended_adjustments[0].delta < 0.0
    assert result_r2.recommended_adjustments[0].delta > 0.0
    assert result_r1.region_id != result_r2.region_id


def test_hard_lower_and_upper_bounds_are_respected():
    model = _model()
    lower_result = _optimize(
        model, _target(batch_L=70.0), (_input("channel_1", lower=-1.0, upper=1.0),)
    )
    upper_result = _optimize(
        model, _target(batch_L=30.0), (_input("channel_1", lower=-1.0, upper=1.0),)
    )

    assert lower_result.proposed_total_adjustment[0].delta >= -1.0 - 1e-8
    assert upper_result.proposed_total_adjustment[0].delta <= 1.0 + 1e-8


def test_trust_region_bounds_incremental_correction():
    model = _model()
    result = _optimize(
        model,
        _target(batch_L=70.0),
        (_input("channel_1", lower=-5.0, upper=5.0),),
        _objective(trust=0.25),
    )

    assert abs(result.recommended_adjustments[0].delta) <= 0.25 + 1e-8


def test_unconfirmed_channel_is_rejected_when_explicitly_requested():
    result = _optimize(
        model=_model(),
        target=_target(),
        inputs=(_input("channel_1", confirmed=False),),
        channel_ids=("channel_1",),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.UNCONFIRMED_CHANNEL
    assert result.recommended_adjustments == ()


def test_default_channel_selection_uses_only_controllable_client_confirmed_model_channels():
    model = _model(coefficients=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    result = _optimize(
        model=model,
        target=_target(),
        inputs=(
            _input("channel_1", confirmed=True),
            _input("channel_2", confirmed=False),
        ),
    )

    assert result.channel_ids == ("channel_1",)


def test_default_channel_selection_rejects_when_no_model_channel_is_controllable_and_confirmed():
    result = _optimize(
        model=_model(),
        target=_target(),
        inputs=(_input("channel_1", confirmed=False),),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.NO_CONTROLLABLE_CHANNELS


def test_no_controllable_channels_are_rejected():
    result = _optimize(
        model=_model(),
        target=_target(),
        inputs=(_input("channel_1", confirmed=False),),
        channel_ids=(),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.NO_CONTROLLABLE_CHANNELS


def test_unknown_channel_is_rejected():
    result = _optimize(
        model=_model(),
        target=_target(),
        inputs=(_input("channel_1"),),
        channel_ids=("channel_unknown",),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.UNKNOWN_CHANNEL


def test_model_channel_mismatch_is_rejected():
    result = _optimize(
        model=_model(),
        target=_target(),
        inputs=(_input("channel_2"),),
        channel_ids=("channel_2",),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.UNKNOWN_CHANNEL


def test_missing_bounds_are_rejected():
    result = _optimize(model=_model(), target=_target(), inputs=(_input("channel_1", lower=None),))

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.MISSING_BOUNDS


def test_current_adjustment_outside_hard_bounds_is_rejected():
    result = _optimize(
        _model(),
        _target(),
        (_input("channel_1", lower=-1.0, upper=1.0),),
        current_adjustment=(ChannelAdjustment(channel_id="channel_1", delta=2.0),),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.CURRENT_ADJUSTMENT_OUTSIDE_BOUNDS


def test_current_adjustment_is_the_reference_point_for_incremental_correction():
    model = _model()
    result = _optimize(
        model,
        _target(batch_L=54.0),
        (_input("channel_1", lower=-5.0, upper=5.0),),
        current_adjustment=(ChannelAdjustment(channel_id="channel_1", delta=-1.0),),
    )

    assert result.current_adjustment[0].delta == pytest.approx(-1.0)
    assert result.proposed_total_adjustment[0].delta == pytest.approx(
        -1.0 + result.recommended_adjustments[0].delta
    )


def test_repeatability_is_deterministic():
    args = (_model(), _target(), (_input("channel_1"),), _objective(size=0.01, uncertainty=0.02))
    first = _optimize(*args)
    second = _optimize(*args)

    assert first == second


def test_uncertainty_penalty_changes_objective_and_can_change_ranking():
    model = _model()
    target = _target()
    low_penalty = _optimize(model, target, (_input("channel_1"),), _objective(uncertainty=0.0))
    high_penalty = _optimize(model, target, (_input("channel_1"),), _objective(uncertainty=100.0))

    assert high_penalty.candidate.uncertainty_penalty > 0.0
    assert high_penalty.candidate.objective > high_penalty.candidate.predicted_delta_E00
    assert high_penalty.candidate.objective != pytest.approx(low_penalty.candidate.objective)


def test_correction_size_penalty_discourages_large_corrections():
    model = _model()
    target = _target(batch_L=53.0)
    unpenalized = _optimize(model, target, (_input("channel_1"),), _objective(size=0.0))
    penalized = _optimize(model, target, (_input("channel_1"),), _objective(size=1.0))

    assert (
        abs(penalized.recommended_adjustments[0].delta)
        <= abs(unpenalized.recommended_adjustments[0].delta) + 1e-8
    )


def test_candidate_worse_than_baseline_is_not_recommended():
    model = _model()

    def deliberately_bad_solver(*args, **kwargs):
        return OptimizeResult(success=True, x=np.array([1.0]), message="synthetic worse candidate")

    result = optimize_region(
        model,
        _target(batch_L=54.0),
        (_input("channel_1"),),
        _objective(),
        solver=deliberately_bad_solver,
        provenance="synthetic optimizer test",
    )

    assert result.status is OptimizerStatus.NO_CHANGE
    assert result.reason is OptimizerReason.NO_MEANINGFUL_IMPROVEMENT
    assert result.recommended_adjustments == ()


def test_numerical_failure_is_handled_safely():
    def failing_solver(*args, **kwargs):
        return OptimizeResult(success=False, x=np.array([0.0]), message="synthetic solver failure")

    result = optimize_region(
        _model(),
        _target(),
        (_input("channel_1"),),
        _objective(),
        solver=failing_solver,
        provenance="synthetic optimizer test",
    )

    assert result.status is OptimizerStatus.NUMERICAL_FAILURE
    assert result.reason is OptimizerReason.SOLVER_FAILURE
    assert result.recommended_adjustments == ()


def test_optional_operating_envelope_constrains_proposed_total_adjustment():
    model = _model()
    envelope = ValidatedOperatingEnvelope(
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        minimum_adjustment=(-0.5,),
        maximum_adjustment=(0.5,),
        n_observations=2,
        validation_status="validated",
        production_run_ids=("run_syn_1",),
        calibrated_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
        model_version=None,
    )
    result = _optimize(
        model,
        _target(batch_L=70.0),
        (_input("channel_1"),),
        operating_envelope=envelope,
    )

    assert result.proposed_total_adjustment[0].delta >= -0.5 - 1e-8
    assert result.proposed_total_adjustment[0].delta <= 0.5 + 1e-8


def test_operating_envelope_channel_mismatch_is_rejected():
    model = _model()
    envelope = ValidatedOperatingEnvelope(
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=("other_channel",),
        minimum_adjustment=(-1.0,),
        maximum_adjustment=(1.0,),
        n_observations=2,
        validation_status="validated",
        production_run_ids=("run_syn_1",),
        calibrated_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
        model_version=None,
    )
    result = _optimize(
        model,
        _target(),
        (_input("channel_1"),),
        operating_envelope=envelope,
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.ENVELOPE_MISMATCH


def test_nonvalidated_operating_envelope_is_rejected():
    model = _model()
    envelope = ValidatedOperatingEnvelope(
        sku_id=model.sku_id,
        context_id=model.context_id,
        region_id=model.region_id,
        channel_ids=model.channel_ids,
        minimum_adjustment=(-1.0,),
        maximum_adjustment=(1.0,),
        n_observations=2,
        validation_status="failed",
        production_run_ids=("run_syn_1",),
        calibrated_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
        model_version=None,
    )
    result = _optimize(
        model,
        _target(),
        (_input("channel_1"),),
        operating_envelope=envelope,
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert result.reason is OptimizerReason.ENVELOPE_MISMATCH


def test_no_client_tolerance_is_part_of_optimizer_contract():
    assert "tolerance" not in OptimizerObjectiveSpec.model_fields
    assert "delta_e_tolerance" not in OptimizerObjectiveSpec.model_fields


def test_synthetic_model_and_client_target_provenance_mismatch_is_rejected():
    target = OptimizationTarget(
        master=RegionMeasurement(region_id="R1", L=50.0, a=0.0, b=0.0, gloss=40.0),
        batch=RegionMeasurement(region_id="R1", L=54.0, a=0.0, b=0.0, gloss=40.0),
        data_origin=DataOrigin.CLIENT,
    )

    with pytest.raises(ValueError, match="model_target_provenance_mismatch"):
        _optimize(_model(), target, (_input("channel_1"),))


def test_synthetic_provenance_is_preserved_and_objective_is_explicitly_placeholder():
    result = _optimize(_model(), _target(), (_input("channel_1"),))

    assert result.provenance == "synthetic optimizer test"
    assert result.objective_spec.is_placeholder is True
    assert result.objective_spec.provenance.startswith("synthetic")
    assert result.data_origin is DataOrigin.SYNTHETIC


def test_optimizer_does_not_mutate_model_target_or_inputs():
    model = _model()
    target = _target()
    inputs = (_input("channel_1"),)
    model_before = model
    target_before = target
    inputs_before = inputs

    _optimize(model, target, inputs)

    assert model == model_before
    assert target == target_before
    assert inputs == inputs_before


def test_target_region_mismatch_is_rejected():
    with pytest.raises(ValueError, match="region_mismatch"):
        _optimize(_model(region_id="R1"), _target(region_id="R2"), (_input("channel_1"),))
