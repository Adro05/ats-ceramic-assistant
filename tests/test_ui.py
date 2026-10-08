from types import SimpleNamespace

import ats_ceramic.ui as ui
from ats_ceramic.correction import QCDecision
from ats_ceramic.synthetic import SyntheticScenario
from ats_ceramic.ui import (
    _target_batch,
    apply_human_qc,
    create_demo_bundle,
    create_demo_verification,
    create_instruction_sheet,
    record_verification_audit,
    run_pipeline,
)


def test_synthetic_demo_setup_preserves_synthetic_provenance_and_master_identity():
    bundle = create_demo_bundle(SyntheticScenario.CORRECTABLE)
    assert bundle.dataset.data_origin.value == "synthetic"
    assert bundle.master_record.master_kind.value == "canonical_master"
    assert bundle.master_record.content_hash
    assert bundle.registry.get(bundle.master_record.master_id, 1) == bundle.master_record
    assert bundle.registry.measurements_for(bundle.master_record.master_id, 1)


def test_correctable_pipeline_reaches_optimizer_and_confidence_without_ml_becoming_primary():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.CORRECTABLE))
    assert pipeline.quality is not None
    assert pipeline.comparison is not None
    assert pipeline.triage is not None
    assert pipeline.selected_model is not None
    assert pipeline.optimization is not None
    assert pipeline.confidence is not None
    assert pipeline.selected_model.data_origin.value == "synthetic"
    assert pipeline.residual_training is not None


def test_residual_call_uses_current_keyword_contract(monkeypatch):
    captured = {}

    class FakeResidualModel:
        result = SimpleNamespace(model_ready=True)

    def fake_fit(*args, **kwargs):
        return FakeResidualModel()

    def fake_apply(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(ui, "split_and_fit_residual_model", fake_fit)
    monkeypatch.setattr(ui, "apply_residual", fake_apply)

    pipeline = ui.run_pipeline(ui.create_demo_bundle(SyntheticScenario.CORRECTABLE))

    assert pipeline.residual_prediction is not None
    assert len(captured["args"]) == 2
    assert set(captured["kwargs"]) == {
        "sku_id",
        "context_id",
        "region_id",
        "channel_ids",
        "adjustments",
        "data_origin",
        "provenance",
        "operating_envelope",
    }


def test_repeated_human_qc_is_rejected_without_duplicate_audit_events():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.CORRECTABLE))
    assert pipeline.optimization is not None
    assert pipeline.confidence is not None

    first = apply_human_qc(pipeline, decision=QCDecision.REJECT, reviewer_id="TEST_REVIEWER")
    assert first.decision.decision is QCDecision.REJECT
    before = len(pipeline.bundle.audit_log.events())

    import pytest

    with pytest.raises(ValueError, match="human QC has already been recorded"):
        apply_human_qc(pipeline, decision=QCDecision.APPROVE, reviewer_id="TEST_REVIEWER")

    assert len(pipeline.bundle.audit_log.events()) == before


def test_explicit_pipeline_reruns_use_fresh_audit_lifecycles():
    first_bundle = create_demo_bundle(SyntheticScenario.CORRECTABLE)
    second_bundle = create_demo_bundle(SyntheticScenario.CORRECTABLE)

    first = run_pipeline(first_bundle)
    second = run_pipeline(second_bundle)

    first_ids = [event.event_id for event in first.bundle.audit_log.events()]
    second_ids = [event.event_id for event in second.bundle.audit_log.events()]
    assert first_ids == second_ids
    assert first.bundle is not second.bundle
    assert len(first_ids) == len(set(first_ids))
    assert len(second_ids) == len(set(second_ids))



def test_process_side_scenario_stops_before_optimizer():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE))
    assert pipeline.triage is not None
    assert pipeline.triage.outcome.value == "LIKELY_PROCESS_SIDE"
    assert pipeline.optimization is None
    assert pipeline.stopped_at == "triage"


def test_corrupted_measurement_stops_before_colour_and_optimizer():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT))
    assert pipeline.batch is None
    assert pipeline.comparison is None
    assert pipeline.optimization is None
    assert pipeline.stopped_at == "measurement_quality"


def test_unseen_sku_does_not_extrapolate_without_a_matching_response_model():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.UNSEEN_SKU_OOD))
    assert pipeline.optimization is None
    assert pipeline.stopped_at == "response_model"


def test_rejected_qc_does_not_create_correction_spec():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.CORRECTABLE))
    assert pipeline.optimization is not None
    assert pipeline.confidence is not None
    result = apply_human_qc(
        pipeline,
        decision=QCDecision.REJECT,
        reviewer_id="TEST_REVIEWER",
        reason_comment="synthetic rejection test",
    )
    assert result.correction_spec is None
    assert result.decision.decision is QCDecision.REJECT


def test_approved_qc_uses_existing_correction_boundary_and_preserves_master():
    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.CORRECTABLE))
    assert pipeline.optimization is not None
    assert pipeline.confidence is not None
    before_hash = pipeline.bundle.master_record.content_hash
    result = apply_human_qc(
        pipeline,
        decision=QCDecision.APPROVE,
        reviewer_id="TEST_REVIEWER",
        reason_comment="synthetic approval test",
    )
    if result.correction_spec is not None:
        assert result.correction_spec.master_content_hash == before_hash
        assert pipeline.bundle.master_record.content_hash == before_hash
        assert result.correction_spec.qc_decision.decision is QCDecision.APPROVE


def test_audit_events_are_from_existing_audit_log():
    bundle = create_demo_bundle(SyntheticScenario.CORRECTABLE)
    pipeline = run_pipeline(bundle)
    event_types = {event.event_type.value for event in pipeline.bundle.audit_log.events()}
    assert "master_registered" in event_types
    assert "measurement_accepted" in event_types
    assert "triage_completed" in event_types
    assert "response_model_fit" in event_types
    assert "optimization_completed" in event_types
    assert "confidence_assessed" in event_types


def test_approved_spec_uses_existing_instruction_adapter_and_verification_workflow():
    from ats_ceramic.schemas import AcceptanceMetric, DataOrigin
    from ats_ceramic.verification import VerificationAcceptanceCriterion

    pipeline = run_pipeline(create_demo_bundle(SyntheticScenario.CORRECTABLE))
    qc = apply_human_qc(pipeline, decision=QCDecision.APPROVE, reviewer_id="TEST_REVIEWER")
    assert qc.correction_spec is not None
    instruction = create_instruction_sheet(qc.correction_spec)
    assert "Golden Master status: UNCHANGED" in instruction.instruction_text

    baseline = _target_batch(pipeline.bundle.dataset)
    regions = tuple(
        region.model_copy(
            update={
                "L": region.L - 0.8 * (region.L - 50.0),
                "a": region.a - 0.8 * (region.a - 8.0),
                "b": region.b - 0.8 * (region.b - 4.0),
            }
        )
        for region in baseline.regions
    )
    verification_measurement = baseline.model_copy(
        update={
            "iteration": 1,
            "applied_correction_id": qc.correction_spec.correction_spec_id,
            "regions": regions,
        }
    )
    criterion = VerificationAcceptanceCriterion(
        acceptance_metric=AcceptanceMetric.MAX_DE00,
        delta_e00_threshold=1.0,
        gloss_abs_threshold=5.0,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=qc.correction_spec.provenance,
        is_placeholder=True,
    )
    verification = create_demo_verification(
        qc.correction_spec,
        pipeline.bundle.master_measurement,
        batch=verification_measurement,
        baseline_measurement=baseline,
        criterion=criterion,
        max_iterations=2,
    )
    assert verification.assessment.status.value == "verified"
    assert verification.learning_evidence is not None
    record_verification_audit(pipeline.bundle, verification)
    assert any(
        event.event_type.value == "verification_completed"
        for event in pipeline.bundle.audit_log.events()
    )
    assert any(
        event.event_type.value == "learning_evidence_created"
        for event in pipeline.bundle.audit_log.events()
    )
