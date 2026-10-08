"""Focused tests for Chunk 9 bounded verification and learning evidence.

All measurements, thresholds, provenance strings, channels, and operating-envelope
values in this module are explicitly synthetic prototype fixtures. They are not
client tolerances, client printer settings, or production records.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from test_confidence import _envelope
from test_correction import HASH, _approved_case
from test_optimizer import _model

from ats_ceramic.correction import create_correction_spec
from ats_ceramic.schemas import BatchMeasurement, DataOrigin, MasterMeasurement, RegionMeasurement
from ats_ceramic.verification import (
    LocalSecantUpdate,
    VerificationAcceptanceCriterion,
    VerificationReason,
    VerificationStatus,
    assess_verification,
    create_learning_evidence,
    create_verification_measurement,
    propose_local_secant_update,
)

NOW = datetime(2026, 1, 3, 10, 0, tzinfo=UTC)


def _spec():
    result, confidence, decision = _approved_case()
    return create_correction_spec(
        result,
        confidence,
        decision,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    )


def _master(*, sku_id="SKU_SYN_TEST", region_id="R1", L=50.0, gloss=40.0):
    from ats_ceramic.schemas import MeasurementMetadata

    metadata = MeasurementMetadata(
        instrument_id="SPEC_SYN_1",
        geometry="d/8",
        illuminant="D65",
        observer="2deg",
        specular_mode="SCI",
        aperture_mm=8.0,
        measured_at=NOW,
        data_origin=DataOrigin.SYNTHETIC,
    )
    return MasterMeasurement(
        sku_id=sku_id,
        regions=(RegionMeasurement(region_id=region_id, L=L, a=0.0, b=0.0, gloss=gloss),),
        metadata=metadata,
    )


def _batch(
    *,
    iteration=1,
    correction_id="CS-1",
    sku_id="SKU_SYN_TEST",
    context_id="ctx_test_syn",
    region_id="R1",
    L=50.1,
    gloss=40.1,
    origin=DataOrigin.SYNTHETIC,
):
    from ats_ceramic.schemas import MeasurementMetadata

    metadata = MeasurementMetadata(
        instrument_id="SPEC_SYN_1",
        geometry="d/8",
        illuminant="D65",
        observer="2deg",
        specular_mode="SCI",
        aperture_mm=8.0,
        measured_at=NOW,
        data_origin=origin,
    )
    return BatchMeasurement(
        batch_id="B-SYN-1",
        sku_id=sku_id,
        context_id=context_id,
        iteration=iteration,
        applied_correction_id=correction_id,
        regions=(RegionMeasurement(region_id=region_id, L=L, a=0.0, b=0.0, gloss=gloss),),
        metadata=metadata,
    )


def _criterion(*, de=1.0, gloss=1.0, origin=DataOrigin.SYNTHETIC):
    return VerificationAcceptanceCriterion(
        delta_e00_threshold=de,
        gloss_abs_threshold=gloss,
        data_origin=origin,
        provenance="synthetic verification acceptance criterion",
        is_placeholder=True,
    )


def _verification(spec, *, L=50.1, gloss=40.1, region_id="R1", iteration=1):
    master = _master(region_id=region_id)
    batch = _batch(
        iteration=iteration,
        correction_id=spec.correction_spec_id,
        L=L,
        gloss=gloss,
        region_id=region_id,
    )
    return create_verification_measurement(
        spec,
        master,
        batch,
        master_id=spec.master_id,
        region_id=region_id,
        provenance=spec.provenance,
    )


def test_valid_verification_meets_synthetic_criterion_and_is_verified():
    spec = _spec()
    verification = _verification(spec, L=50.1, gloss=40.1)
    assessment = assess_verification(
        spec, verification, _criterion(de=0.5, gloss=0.5), max_iterations=3
    )

    assert assessment.status is VerificationStatus.VERIFIED
    assert assessment.reason is VerificationReason.ACCEPTED


def test_verification_outside_criterion_allows_bounded_next_iteration():
    spec = _spec()
    verification = _verification(spec, L=55.0, gloss=45.0)
    assessment = assess_verification(
        spec, verification, _criterion(de=0.5, gloss=0.5), max_iterations=3
    )

    assert assessment.status is VerificationStatus.ITERATION_ALLOWED
    assert assessment.reason in {
        VerificationReason.COLOUR_OUTSIDE_CRITERION,
        VerificationReason.GLOSS_OUTSIDE_CRITERION,
    }


def test_missing_criterion_cannot_claim_verified():
    spec = _spec()
    verification = _verification(spec)
    assessment = assess_verification(spec, verification, None, max_iterations=3)

    assert assessment.status is VerificationStatus.NOT_ASSESSABLE
    assert assessment.reason is VerificationReason.MISSING_CRITERION


def test_maximum_iteration_bound_is_respected():
    spec = _spec()
    verification = _verification(spec, L=55.0, gloss=45.0, iteration=2)
    assessment = assess_verification(
        spec, verification, _criterion(de=0.5, gloss=0.5), max_iterations=3
    )

    assert assessment.status is VerificationStatus.ITERATION_ALLOWED


def test_exhausting_iteration_limit_escalates():
    spec = _spec()
    verification = _verification(spec, L=55.0, gloss=45.0, iteration=3)
    assessment = assess_verification(
        spec, verification, _criterion(de=0.5, gloss=0.5), max_iterations=3
    )

    assert assessment.status is VerificationStatus.ESCALATE
    assert assessment.reason is VerificationReason.ITERATION_LIMIT_REACHED


def test_invalid_measurement_fails_safely():
    spec = _spec()
    master = _master()
    batch = _batch(correction_id=spec.correction_spec_id)
    invalid_region = batch.regions[0].model_copy(update={"L": 101.0})
    invalid_batch = batch.model_copy(update={"regions": (invalid_region,)})

    with pytest.raises(ValueError, match="invalid_measurement"):
        create_verification_measurement(
            spec,
            master,
            invalid_batch,
            master_id=spec.master_id,
            region_id="R1",
            provenance=spec.provenance,
        )


@pytest.mark.parametrize(
    "field, expected",
    [
        ("region_id", "region mismatch"),
        ("sku_id", "SKU mismatch"),
        ("context_id", "context mismatch"),
    ],
)
def test_identity_mismatches_are_rejected(field, expected):
    spec = _spec()
    master = _master()
    batch = _batch(correction_id=spec.correction_spec_id, **{field: "OTHER"})
    with pytest.raises(ValueError, match=expected):
        create_verification_measurement(
            spec,
            master,
            batch,
            master_id=spec.master_id,
            region_id="R1",
            provenance=spec.provenance,
        )


def test_channel_mismatch_is_rejected_for_secant_update():
    spec = _spec()
    verification = _verification(spec)
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)

    with pytest.raises(ValueError, match="unsupported channel"):
        propose_local_secant_update(
            spec,
            assessment,
            _batch(iteration=0, correction_id=None, L=50.5, gloss=40.5),
            channel_id="unknown_channel",
            operating_envelope=_envelope(),
        )


def test_provenance_mismatch_is_rejected():
    spec = _spec()
    verification = _verification(spec)
    tampered = verification.model_copy(update={"data_origin": DataOrigin.CLIENT})
    with pytest.raises(ValueError, match="data-origin"):
        assess_verification(spec, tampered, _criterion(), max_iterations=2)


def test_provenance_mismatch_is_rejected_without_changing_data_origin():
    spec = _spec()
    verification = _verification(spec).model_copy(
        update={"provenance": "different synthetic provenance"}
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        assess_verification(spec, verification, _criterion(), max_iterations=2)


def test_synthetic_client_data_origin_mismatch_is_rejected():
    spec = _spec()
    master = _master()
    with pytest.raises(ValueError, match="data-origin mismatch"):
        create_verification_measurement(
            spec,
            master,
            _batch(correction_id=spec.correction_spec_id, origin=DataOrigin.CLIENT),
            master_id=spec.master_id,
            region_id="R1",
            provenance=spec.provenance,
        )


def test_verification_records_are_immutable():
    spec = _spec()
    verification = _verification(spec)
    with pytest.raises(ValidationError):
        verification.region_id = "R2"
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)
    with pytest.raises(ValidationError):
        assessment.status = VerificationStatus.ESCALATE


def test_correction_qc_and_master_remain_unchanged_after_verification():
    result, confidence, decision = _approved_case()
    spec = _spec()
    master = _master()
    spec_before = spec.model_dump()
    decision_before = decision.model_dump()
    master_before = master.model_dump()

    _ = assess_verification(spec, _verification(spec), _criterion(), max_iterations=2)

    assert spec.model_dump() == spec_before
    assert decision.model_dump() == decision_before
    assert master.model_dump() == master_before
    assert result.model_dump() == spec.optimizer_result.model_dump()
    assert confidence.model_dump() == spec.confidence_assessment.model_dump()


def test_successful_verification_produces_immutable_learning_evidence():
    spec = _spec()
    verification = _verification(spec, L=50.1, gloss=40.1)
    assessment = assess_verification(
        spec,
        verification,
        _criterion(de=0.5, gloss=0.5),
        max_iterations=2,
        baseline_measurement=_batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
    )
    evidence = create_learning_evidence(
        spec,
        assessment,
        _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
    )

    assert evidence.successful_verification is True
    assert evidence.data_origin is DataOrigin.SYNTHETIC
    with pytest.raises(ValidationError):
        evidence.region_id = "R2"


def test_learning_evidence_does_not_mutate_response_model():
    spec = _spec()
    model = _model()
    before = model.model_dump()
    verification = _verification(spec, L=50.1, gloss=40.1)
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)
    create_learning_evidence(
        spec,
        assessment,
        _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
    )
    assert model.model_dump() == before


def test_secant_update_rejects_zero_denominator():
    spec = _spec()
    result = spec.optimizer_result.model_copy(
        update={
            "recommended_adjustments": tuple(
                item.model_copy(update={"delta": 0.0}) for item in spec.recommended_adjustments
            )
        }
    )
    spec_zero = spec.model_copy(
        update={
            "optimizer_result": result,
            "recommended_adjustments": result.recommended_adjustments,
        }
    )
    verification = _verification(spec_zero)
    assessment = assess_verification(spec_zero, verification, _criterion(), max_iterations=2)
    with pytest.raises(ValueError, match="denominator"):
        propose_local_secant_update(
            spec_zero,
            assessment,
            _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
            channel_id="channel_1",
            operating_envelope=_envelope(),
        )


def test_secant_update_respects_channel_identity_and_locality():
    spec = _spec()
    verification = _verification(spec, L=50.1, gloss=40.1)
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)
    before = _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0)
    update = propose_local_secant_update(
        spec,
        assessment,
        before,
        channel_id=spec.optimizer_result.channel_ids[0],
        operating_envelope=_envelope(),
    )

    assert isinstance(update, LocalSecantUpdate)
    assert update.channel_id == spec.optimizer_result.channel_ids[0]
    assert update.region_id == spec.region_id
    assert update.context_id == spec.context_id
    assert update.sku_id == spec.sku_id


def test_secant_update_rejects_multi_channel_extrapolation():
    spec = _spec()
    second = spec.optimizer_result.recommended_adjustments[0].model_copy(
        update={"channel_id": "channel_2", "delta": 0.1}
    )
    result = spec.optimizer_result.model_copy(
        update={
            "channel_ids": ("channel_1", "channel_2"),
            "recommended_adjustments": (spec.recommended_adjustments[0], second),
        }
    )
    spec_multi = spec.model_copy(
        update={
            "optimizer_result": result,
            "recommended_adjustments": result.recommended_adjustments,
        }
    )
    verification = _verification(spec_multi)
    assessment = assess_verification(spec_multi, verification, _criterion(), max_iterations=2)
    with pytest.raises(ValueError, match="exactly one changed channel"):
        propose_local_secant_update(
            spec_multi,
            assessment,
            _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
            channel_id="channel_1",
            operating_envelope=_envelope().model_copy(
                update={
                    "channel_ids": ("channel_1", "channel_2"),
                    "minimum_adjustment": (-5.0, -5.0),
                    "maximum_adjustment": (5.0, 5.0),
                }
            ),
        )


def test_secant_update_rejects_outside_operating_envelope():
    spec = _spec()
    verification = _verification(spec)
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)
    narrow = _envelope().model_copy(
        update={"minimum_adjustment": (-0.00001,), "maximum_adjustment": (0.00001,)}
    )
    with pytest.raises(ValueError, match="outside operating envelope"):
        propose_local_secant_update(
            spec,
            assessment,
            _batch(iteration=0, correction_id=None, L=54.0, gloss=44.0),
            channel_id="channel_1",
            operating_envelope=narrow,
        )


def test_repeated_verification_is_deterministic():
    spec = _spec()
    verification = _verification(spec)
    criterion = _criterion()
    first = assess_verification(spec, verification, criterion, max_iterations=2)
    second = assess_verification(spec, verification, criterion, max_iterations=2)
    assert first == second


def test_synthetic_provenance_is_explicit():
    spec = _spec()
    verification = _verification(spec)
    assessment = assess_verification(spec, verification, _criterion(), max_iterations=2)
    assert verification.data_origin is DataOrigin.SYNTHETIC
    assert assessment.data_origin is DataOrigin.SYNTHETIC
    assert assessment.criterion is not None
    assert assessment.criterion.is_placeholder is True


def test_gloss_is_separate_from_colour_acceptance_logic():
    spec = _spec()
    colour_ok_gloss_bad = _verification(spec, L=50.1, gloss=45.0)
    assessment = assess_verification(
        spec, colour_ok_gloss_bad, _criterion(de=0.5, gloss=0.5), max_iterations=2
    )
    assert assessment.status is VerificationStatus.ITERATION_ALLOWED
    assert assessment.reason is VerificationReason.GLOSS_OUTSIDE_CRITERION
    assert assessment.region_delta_e00 < 0.5
    assert assessment.region_abs_gloss_delta > 0.5


def test_no_client_delta_e_tolerance_is_invented():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/ats_ceramic/verification.py").read_text()
    assert "delta_e00_threshold = 1.0" not in source
    assert "client_delta_e00" not in source
