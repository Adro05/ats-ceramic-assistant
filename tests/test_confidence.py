"""Tests for Chunk 7 confidence / OOD gating.

All numerical fixtures are explicitly synthetic prototype assumptions.  None are
client acceptance tolerances or confirmed printer settings.
"""
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from test_optimizer import _input, _model, _target

from ats_ceramic.calibration_workflow import (
    ControllableInput,
    ValidatedOperatingEnvelope,
    ValidationStatus,
)
from ats_ceramic.confidence import (
    ConfidenceReason,
    ConfidenceSpec,
    ConfidenceStage,
    ConfidenceStatus,
    ModelFreshness,
    OODStatus,
    post_solve_confidence,
    preflight_confidence,
)
from ats_ceramic.optimizer import (
    OptimizationTarget,
    OptimizerObjectiveSpec,
    OptimizerStatus,
    optimize_region,
)
from ats_ceramic.response_model import BayesianSensitivitySpec
from ats_ceramic.schemas import ChannelAdjustment, DataOrigin, RegionMeasurement

NOW = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def _confidence_spec(*, warn: float = 0.1, red: float = 0.5) -> ConfidenceSpec:
    return ConfidenceSpec(
        near_boundary_fraction=0.1,
        uncertainty_warn_norm=warn,
        uncertainty_red_norm=red,
        provenance="synthetic confidence test configuration",
        is_placeholder=True,
    )


def _envelope(
    *,
    minimum: tuple[float, ...] = (-5.0,),
    maximum: tuple[float, ...] = (5.0,),
    status: ValidationStatus = ValidationStatus.VALIDATED,
    data_origin: DataOrigin = DataOrigin.SYNTHETIC,
) -> ValidatedOperatingEnvelope:
    return ValidatedOperatingEnvelope(
        sku_id="SKU_SYN_TEST",
        context_id="ctx_test_syn",
        region_id="R1",
        channel_ids=("channel_1",),
        minimum_adjustment=minimum,
        maximum_adjustment=maximum,
        n_observations=4,
        validation_status=status,
        production_run_ids=("run_1", "run_2"),
        calibrated_at=NOW,
        data_origin=data_origin,
        model_version=None,
    )


def _optimize(**kwargs):
    model = kwargs.pop("model", _model())
    target = kwargs.pop("target", _target())
    inputs = kwargs.pop("inputs", (_input("channel_1"),))
    objective = kwargs.pop(
        "objective",
        OptimizerObjectiveSpec(
            delta_e_weight=1.0,
            correction_size_weight=0.0,
            uncertainty_weight=0.0,
            minimum_objective_improvement=0.0,
            trust_region_radius=5.0,
            provenance="synthetic confidence test configuration",
            is_placeholder=True,
        ),
    )
    return optimize_region(
        model,
        target,
        inputs,
        objective,
        provenance="synthetic confidence test",
        **kwargs,
    )


def test_preflight_green_for_validated_in_domain_request():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert assessment.stage is ConfidenceStage.PREFLIGHT
    assert assessment.status is ConfidenceStatus.GREEN
    assert assessment.ood_status is OODStatus.IN_DOMAIN
    assert ConfidenceReason.VALIDATED_DOMAIN in assessment.reason_codes


def test_preflight_near_boundary_is_amber():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        current_adjustment=(ChannelAdjustment(channel_id="channel_1", delta=4.5),),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.AMBER
    assert assessment.ood_status is OODStatus.NEAR_BOUNDARY
    assert ConfidenceReason.NEAR_DOMAIN_BOUNDARY in assessment.reason_codes


def test_preflight_current_adjustment_outside_hard_bounds_is_red():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1", lower=-2.0, upper=2.0),),
        _confidence_spec(),
        current_adjustment=(ChannelAdjustment(channel_id="channel_1", delta=3.0),),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.HARD_BOUND_VIOLATION in assessment.reason_codes


def test_preflight_outside_domain_is_red():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        current_adjustment=(ChannelAdjustment(channel_id="channel_1", delta=6.0),),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert assessment.ood_status is OODStatus.OUT_OF_DOMAIN
    assert ConfidenceReason.OUTSIDE_OPERATING_ENVELOPE in assessment.reason_codes


def test_missing_envelope_is_unknown_and_reduces_confidence_without_inventing_domain():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
    )

    assert assessment.status is ConfidenceStatus.AMBER
    assert assessment.ood_status is OODStatus.UNKNOWN
    assert assessment.operating_envelope_status is OODStatus.UNKNOWN
    assert ConfidenceReason.MISSING_OPERATING_ENVELOPE in assessment.reason_codes


def test_unvalidated_envelope_is_red():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(status=ValidationStatus.FAILED),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert assessment.ood_status is OODStatus.UNKNOWN
    assert ConfidenceReason.UNVALIDATED_OPERATING_ENVELOPE in assessment.reason_codes


def test_preflight_sku_and_context_mismatch_are_red_when_request_metadata_is_supplied():
    assessment = preflight_confidence(
        _model(),
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        expected_sku_id="OTHER_SKU",
        expected_context_id="OTHER_CONTEXT",
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.SKU_MISMATCH in assessment.reason_codes
    assert ConfidenceReason.MODEL_CONTEXT_MISMATCH in assessment.reason_codes


def test_assessment_preserves_model_evidence_provenance():
    model = _model()
    result = _optimize(model=model)

    preflight = preflight_confidence(
        model,
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )
    post_solve = post_solve_confidence(
        model,
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert model.spec.source == "synthetic optimizer test assumption"
    assert preflight.provenance == model.spec.source
    assert post_solve.provenance == model.spec.source
    assert preflight.provenance != _confidence_spec().provenance
    assert post_solve.provenance != _confidence_spec().provenance


def test_provenance_mismatch_is_red_and_never_promoted():
    target = OptimizationTarget(
        master=RegionMeasurement(region_id="R1", L=50.0, a=0.0, b=0.0, gloss=40.0),
        batch=RegionMeasurement(region_id="R1", L=54.0, a=0.0, b=0.0, gloss=40.0),
        data_origin=DataOrigin.CLIENT,
    )

    assessment = preflight_confidence(
        _model(),
        target,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert assessment.data_origin is DataOrigin.SYNTHETIC
    assert ConfidenceReason.PROVENANCE_MISMATCH in assessment.reason_codes


def test_insufficient_model_support_is_red():
    model = _model().model_copy(
        update={
            "n_observations": 1,
            "spec": BayesianSensitivitySpec(
                prior_variance=100.0,
                observation_noise_std=(0.01, 0.01, 0.01),
                min_observations=2,
                is_placeholder=True,
                source="synthetic confidence test assumption",
            ),
        }
    )
    assessment = preflight_confidence(
        model,
        _target(),
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.INSUFFICIENT_MODEL_SUPPORT in assessment.reason_codes


def test_post_solve_green_uses_existing_sensitivity_prediction_uncertainty():
    model = _model()
    result = _optimize(model=model)
    assessment = post_solve_confidence(
        model,
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(minimum=(-5.0,), maximum=(5.0,)),
    )

    assert result.status is OptimizerStatus.RECOMMEND
    assert assessment.stage is ConfidenceStage.POST_SOLVE
    assert assessment.status is ConfidenceStatus.GREEN
    assert assessment.prediction_uncertainty == (
        result.candidate.prediction.uncertainty_std_L,
        result.candidate.prediction.uncertainty_std_a,
        result.candidate.prediction.uncertainty_std_b,
    )
    assert assessment.uncertainty_norm is not None


def test_post_solve_near_boundary_is_amber():
    result = _optimize()
    proposed = result.proposed_total_adjustment[0].delta
    if proposed < 0.0:
        minimum, maximum = proposed / 0.9, 5.0
    else:
        minimum, maximum = -5.0, proposed / 0.9
    envelope = _envelope(minimum=(minimum,), maximum=(maximum,))

    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=envelope,
    )

    assert assessment.status is ConfidenceStatus.AMBER
    assert assessment.ood_status is OODStatus.NEAR_BOUNDARY


def test_post_solve_proposed_total_outside_validated_envelope_is_red():
    result = _optimize()
    proposed = result.proposed_total_adjustment[0].delta
    if proposed >= 0.0:
        minimum, maximum = (-0.01, 0.01)
    else:
        minimum, maximum = (0.01, 0.02)

    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(minimum=(minimum,), maximum=(maximum,)),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert assessment.ood_status is OODStatus.OUT_OF_DOMAIN
    assert ConfidenceReason.OUTSIDE_OPERATING_ENVELOPE in assessment.reason_codes


def test_post_solve_missing_envelope_is_unknown_and_amber():
    result = _optimize()
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
    )

    assert assessment.status is ConfidenceStatus.AMBER
    assert assessment.ood_status is OODStatus.UNKNOWN


def test_post_solve_unvalidated_envelope_is_red():
    result = _optimize()
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(status=ValidationStatus.FAILED),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.UNVALIDATED_OPERATING_ENVELOPE in assessment.reason_codes


def test_elevated_uncertainty_is_amber_using_model_prediction():
    model = _model().model_copy(
        update={
            "spec": BayesianSensitivitySpec(
                prior_variance=100.0,
                observation_noise_std=(0.08, 0.08, 0.08),
                min_observations=2,
                is_placeholder=True,
                source="synthetic confidence test assumption",
            )
        }
    )
    result = _optimize(model=model)
    assessment = post_solve_confidence(
        model,
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(warn=0.1, red=1.0),
        operating_envelope=_envelope(),
    )

    assert assessment.uncertainty_norm is not None
    assert assessment.status is ConfidenceStatus.AMBER
    assert ConfidenceReason.ELEVATED_PREDICTION_UNCERTAINTY in assessment.reason_codes


def test_high_uncertainty_is_red():
    model = _model().model_copy(
        update={
            "spec": BayesianSensitivitySpec(
                prior_variance=100.0,
                observation_noise_std=(1.0, 1.0, 1.0),
                min_observations=2,
                is_placeholder=True,
                source="synthetic confidence test assumption",
            )
        }
    )
    result = _optimize(model=model)
    assessment = post_solve_confidence(
        model,
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(warn=0.1, red=0.5),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.HIGH_PREDICTION_UNCERTAINTY in assessment.reason_codes


def test_invalid_uncertainty_fails_safe(monkeypatch):
    result = _optimize()
    monkeypatch.setattr("ats_ceramic.confidence._uncertainty_values", lambda _: None)

    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.INVALID_PREDICTION in assessment.reason_codes


def test_optimizer_no_change_is_red():
    result = _optimize(
        objective=OptimizerObjectiveSpec(
            delta_e_weight=1.0,
            correction_size_weight=0.0,
            uncertainty_weight=0.0,
            minimum_objective_improvement=100.0,
            trust_region_radius=5.0,
            provenance="synthetic confidence test configuration",
            is_placeholder=True,
        )
    )
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert result.status is OptimizerStatus.NO_CHANGE
    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.NO_MEANINGFUL_IMPROVEMENT in assessment.reason_codes


def test_optimizer_infeasible_is_red():
    infeasible_input = ControllableInput(
        channel_id="channel_1",
        controllable=True,
        lower_bound=1.0,
        upper_bound=2.0,
        nominal_value=1.5,
        unit="synthetic_unit",
        source="test-only client-confirmation state; not client data",
        client_confirmed=True,
        data_origin=DataOrigin.CLIENT,
    )
    result = _optimize(inputs=(infeasible_input,))
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (infeasible_input,),
        _confidence_spec(),
        operating_envelope=_envelope(minimum=(1.0,), maximum=(2.0,)),
    )

    assert result.status is OptimizerStatus.INFEASIBLE
    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.NO_RECOMMENDATION in assessment.reason_codes


def test_optimizer_numerical_failure_is_red():
    def failing_solver(*args, **kwargs):
        from scipy.optimize import OptimizeResult

        return OptimizeResult(success=False, message="synthetic solver failure")

    result = _optimize(solver=failing_solver)
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert result.status is OptimizerStatus.NUMERICAL_FAILURE
    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.OPTIMIZER_FAILURE in assessment.reason_codes


def test_hard_bound_violation_is_red_without_reimplementing_optimizer():
    result = _optimize()
    restrictive_input = _input("channel_1", lower=-0.1, upper=0.1)

    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (restrictive_input,),
        _confidence_spec(),
        operating_envelope=_envelope(minimum=(-5.0,), maximum=(5.0,)),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.HARD_BOUND_VIOLATION in assessment.reason_codes


def test_trust_region_violation_is_red():
    result = _optimize()
    tampered = result.model_copy(
        update={
            "proposed_total_adjustment": (
                ChannelAdjustment(
                    channel_id="channel_1",
                    delta=result.current_adjustment[0].delta + 6.0,
                ),
            )
        }
    )

    assessment = post_solve_confidence(
        _model(),
        _target(),
        tampered,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(minimum=(-5.0,), maximum=(5.0,)),
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.TRUST_REGION_VIOLATION in assessment.reason_codes


def test_envelope_identity_mismatch_is_red():
    result = _optimize()
    envelope = _envelope().model_copy(update={"context_id": "OTHER_CONTEXT"})

    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=envelope,
    )

    assert assessment.status is ConfidenceStatus.RED
    assert ConfidenceReason.OPERATING_ENVELOPE_MISMATCH in assessment.reason_codes


def test_freshness_is_unknown_instead_of_invented():
    result = _optimize()
    assessment = post_solve_confidence(
        _model(),
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert assessment.model_freshness is ModelFreshness.UNKNOWN
    assert ConfidenceReason.MODEL_FRESHNESS_UNKNOWN in assessment.reason_codes


def test_provenance_remains_synthetic_end_to_end():
    model = _model()
    result = _optimize(model=model)
    assessment = post_solve_confidence(
        model,
        _target(),
        result,
        (_input("channel_1"),),
        _confidence_spec(),
        operating_envelope=_envelope(),
    )

    assert model.data_origin is DataOrigin.SYNTHETIC
    assert result.data_origin is DataOrigin.SYNTHETIC
    assert assessment.data_origin is DataOrigin.SYNTHETIC


def test_assessment_is_immutable_and_deterministic():
    model = _model()
    result = _optimize(model=model)
    kwargs = dict(
        model=model,
        target=_target(),
        result=result,
        controllable_inputs=(_input("channel_1"),),
        spec=_confidence_spec(),
        operating_envelope=_envelope(),
    )
    first = post_solve_confidence(**kwargs)
    second = post_solve_confidence(**kwargs)

    assert first == second
    with pytest.raises(ValidationError):
        first.status = ConfidenceStatus.RED


def test_confidence_contains_no_client_delta_e_threshold():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/ats_ceramic/confidence.py").read_text()
    assert "delta_e00_threshold" not in source
