"""Tests for Chunk 8 human QC and CorrectionSpec.

All fixtures are explicitly synthetic prototype records.  They are not client
printer settings, client colour tolerances, or production records.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from test_confidence import _envelope, _optimize
from test_optimizer import _input, _model, _target

from ats_ceramic.confidence import ConfidenceSpec, post_solve_confidence
from ats_ceramic.correction import (
    CorrectionSpec,
    HumanQCDecision,
    QCDecision,
    create_correction_spec,
    validate_qc_decision,
)
from ats_ceramic.optimizer import OptimizerStatus
from ats_ceramic.schemas import DataOrigin, MasterMeasurement

NOW = datetime(2026, 1, 3, 10, 0, tzinfo=UTC)
HASH = "b" * 64


def _confidence(result):
    model = _model()
    target = _target()
    return post_solve_confidence(
        model,
        target,
        result,
        (_input("channel_1"),),
        ConfidenceSpec(
            near_boundary_fraction=0.1,
            uncertainty_warn_norm=0.1,
            uncertainty_red_norm=0.5,
            provenance="synthetic confidence configuration",
            is_placeholder=True,
        ),
        operating_envelope=_envelope(),
    )


def _decision(
    *,
    decision=QCDecision.APPROVE,
    correction_spec_id="CS-1",
    sku_id="SKU_SYN_TEST",
    context_id="ctx_test_syn",
    region_id="R1",
    data_origin=DataOrigin.SYNTHETIC,
    reason_comment=None,
):
    return HumanQCDecision(
        decision_id="QC-1",
        correction_spec_id=correction_spec_id,
        sku_id=sku_id,
        context_id=context_id,
        region_id=region_id,
        decision=decision,
        reason_comment=reason_comment,
        reviewer_id="reviewer-synthetic",
        decided_at=NOW,
        data_origin=data_origin,
        provenance="synthetic optimizer test assumption",
    )


def _approved_case():
    result = _optimize()
    confidence = _confidence(result)
    assert confidence.status.value in {"green", "amber"}
    decision = _decision()
    return result, confidence, decision


def test_valid_green_recommendation_can_be_approved_and_creates_spec():
    result, confidence, decision = _approved_case()
    spec = create_correction_spec(
        result,
        confidence,
        decision,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    )

    assert isinstance(spec, CorrectionSpec)
    assert spec.qc_decision.decision is QCDecision.APPROVE


def test_red_confidence_cannot_be_approved():
    result, _, decision = _approved_case()
    model = _model()
    target = _target()
    red_confidence = post_solve_confidence(
        model,
        target,
        result,
        (_input("channel_1"),),
        ConfidenceSpec(
            near_boundary_fraction=0.1,
            uncertainty_warn_norm=0.000001,
            uncertainty_red_norm=0.000002,
            provenance="synthetic confidence configuration",
            is_placeholder=True,
        ),
        operating_envelope=_envelope(),
    )
    assert red_confidence.status.value == "red"
    with pytest.raises(ValueError, match="RED confidence"):
        validate_qc_decision(result, red_confidence, decision)
    with pytest.raises(ValueError, match="RED confidence"):
        create_correction_spec(
            result,
            red_confidence,
            decision,
            batch_id="B-SYN-1",
            master_id="M-SYN-1",
            master_content_hash=HASH,
        )


def test_matching_qc_provenance_allows_approval():
    result, confidence, decision = _approved_case()

    validate_qc_decision(result, confidence, decision)


def test_mismatched_qc_provenance_rejects_approval_without_mutation():
    result, confidence, decision = _approved_case()
    decision = decision.model_copy(update={"provenance": "different QC provenance"})
    before = decision.model_dump()

    with pytest.raises(
        ValueError, match="provenance mismatch between QC decision and recommendation"
    ):
        validate_qc_decision(result, confidence, decision)

    assert decision.model_dump() == before


def test_qc_provenance_validation_is_deterministic():
    result, confidence, decision = _approved_case()
    decision = decision.model_copy(update={"provenance": "different QC provenance"})
    before = decision.model_dump()

    messages = []
    for _ in range(2):
        with pytest.raises(ValueError) as exc_info:
            validate_qc_decision(result, confidence, decision)
        messages.append(str(exc_info.value))

    assert messages == [
        "provenance mismatch between QC decision and recommendation",
        "provenance mismatch between QC decision and recommendation",
    ]
    assert decision.model_dump() == before


def test_qc_validation_is_deterministic():
    result, confidence, decision = _approved_case()
    validate_qc_decision(result, confidence, decision)
    first = decision.model_dump()
    validate_qc_decision(result, confidence, decision)
    assert decision.model_dump() == first


def test_rejection_is_recorded_and_does_not_create_correction_spec():
    result, confidence, _ = _approved_case()
    decision = _decision(
        decision=QCDecision.REJECT,
        reason_comment="Synthetic QC rejection for audit test",
    )

    validate_qc_decision(result, confidence, decision)
    assert (
        create_correction_spec(
            result,
            confidence,
            decision,
            batch_id="B-SYN-1",
            master_id="M-SYN-1",
            master_content_hash=HASH,
        )
        is None
    )
    assert decision.reason_comment == "Synthetic QC rejection for audit test"


@pytest.mark.parametrize(
    "field",
    ["sku_id", "context_id", "region_id"],
)
def test_identifier_mismatch_fails(field):
    result, confidence, _ = _approved_case()
    decision = _decision(**{field: "mismatch"})
    with pytest.raises(ValueError, match="mismatch"):
        validate_qc_decision(result, confidence, decision)


def test_provenance_data_origin_mismatch_fails():
    result, confidence, _ = _approved_case()
    decision = _decision(data_origin=DataOrigin.CLIENT)
    with pytest.raises(ValueError, match="provenance/data origin"):
        validate_qc_decision(result, confidence, decision)


def test_invalid_optimizer_result_cannot_create_spec():
    result, confidence, decision = _approved_case()
    bad = result.model_copy(
        update={"status": OptimizerStatus.NO_CHANGE, "recommended_adjustments": ()}
    )
    with pytest.raises(ValueError, match="valid optimizer recommendation"):
        validate_qc_decision(bad, confidence, decision)


def test_recommendation_adjustments_and_audit_metadata_are_preserved_exactly():
    result, confidence, decision = _approved_case()
    spec = create_correction_spec(
        result,
        confidence,
        decision,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    )

    assert spec is not None
    assert spec.recommended_adjustments == result.recommended_adjustments
    assert spec.current_adjustment == result.current_adjustment
    assert spec.optimizer_result == result
    assert spec.confidence_assessment == confidence
    assert spec.qc_decision == decision
    assert spec.batch_id == "B-SYN-1"
    assert spec.master_id == "M-SYN-1"
    assert spec.master_content_hash == HASH
    assert spec.data_origin is DataOrigin.SYNTHETIC
    assert spec.provenance == confidence.provenance


def test_qc_decision_and_correction_spec_are_immutable():
    result, confidence, decision = _approved_case()
    with pytest.raises(ValidationError):
        decision.decision = QCDecision.REJECT

    spec = create_correction_spec(
        result,
        confidence,
        decision,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    )
    assert spec is not None
    with pytest.raises(ValidationError):
        spec.sku_id = "MUTATED"


def test_master_is_not_mutated_by_correction_spec_creation(make_metadata):
    result, confidence, decision = _approved_case()
    master = MasterMeasurement(
        sku_id=result.sku_id,
        regions=(_target().master,),
        metadata=make_metadata(),
    )
    before = master.model_dump()

    create_correction_spec(
        result,
        confidence,
        decision,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    )

    assert master.model_dump() == before


def test_amber_confidence_can_be_reviewed_without_new_policy():
    result = _optimize()
    model = _model()
    target = _target()
    confidence = post_solve_confidence(
        model,
        target,
        result,
        (_input("channel_1"),),
        ConfidenceSpec(
            near_boundary_fraction=0.49,
            uncertainty_warn_norm=0.000001,
            uncertainty_red_norm=0.5,
            provenance="synthetic confidence configuration",
            is_placeholder=True,
        ),
        operating_envelope=_envelope(),
    )
    assert confidence.status.value == "amber"
    decision = _decision()
    validate_qc_decision(result, confidence, decision)


def test_correction_spec_rejects_mismatched_model_provenance():
    result, confidence, decision = _approved_case()
    with pytest.raises(ValidationError, match="provenance"):
        CorrectionSpec(
            correction_spec_id="CS-1",
            batch_id="B-SYN-1",
            sku_id=result.sku_id,
            context_id=result.context_id,
            region_id=result.region_id,
            master_id="M-SYN-1",
            master_content_hash=HASH,
            recommended_adjustments=result.recommended_adjustments,
            current_adjustment=result.current_adjustment,
            optimizer_result=result,
            confidence_assessment=confidence,
            qc_decision=decision,
            data_origin=result.data_origin,
            provenance="different evidence provenance",
        )


def test_correction_spec_rejects_mismatched_adjustments():
    result, confidence, decision = _approved_case()
    with pytest.raises(ValidationError, match="recommended adjustments"):
        CorrectionSpec(
            correction_spec_id="CS-1",
            batch_id="B-SYN-1",
            sku_id=result.sku_id,
            context_id=result.context_id,
            region_id=result.region_id,
            master_id="M-SYN-1",
            master_content_hash=HASH,
            recommended_adjustments=(
                result.recommended_adjustments[0].model_copy(update={"delta": 0.123}),
            ),
            current_adjustment=result.current_adjustment,
            optimizer_result=result,
            confidence_assessment=confidence,
            qc_decision=decision,
            data_origin=result.data_origin,
            provenance=confidence.provenance,
        )
