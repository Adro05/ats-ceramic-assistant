"""Streamlit-facing orchestration helpers for the ceramic FDE prototype.

This module deliberately contains no UI framework code.  It assembles the existing
immutable domain modules into one deterministic synthetic demonstration so the
Streamlit entry point remains presentation-only.

The synthetic demo never changes a Golden Master and never creates an executable
CorrectionSpec without passing the existing optimizer, confidence, and human-QC
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from ats_ceramic.audit import AuditEventType, AuditLog, create_audit_event
from ats_ceramic.calibration_workflow import (
    CalibrationObservation,
    ControllableInput,
    ValidationCriterion,
    build_operating_envelopes,
    fit_calibration_models,
    split_held_out_by_run,
    validate_held_out,
)
from ats_ceramic.color import ComparisonResult, compare_measurements
from ats_ceramic.confidence import ConfidenceAssessment, ConfidenceSpec, post_solve_confidence
from ats_ceramic.config import load_tolerance_config
from ats_ceramic.correction import (
    CorrectionSpec,
    HumanQCDecision,
    QCDecision,
    create_correction_spec,
)
from ats_ceramic.correction_adapter import (
    CorrectionInstructionSheet,
    create_correction_instruction_sheet,
)
from ats_ceramic.master_registry import GoldenMasterRecord, GoldenMasterRegistry, MasterKind
from ats_ceramic.measurement_quality import (
    MeasurementQualityResult,
    QualityOutcome,
    assess_measurement_quality,
)
from ats_ceramic.ml_residual import (
    ResidualConfig,
    ResidualModel,
    ResidualPrediction,
    ResidualTrainingResult,
    apply_residual,
    split_and_fit_residual_model,
)
from ats_ceramic.optimizer import (
    OptimizationResult,
    OptimizationTarget,
    OptimizerObjectiveSpec,
    OptimizerStatus,
    optimize_region,
)
from ats_ceramic.response_model import (
    BayesianSensitivitySpec,
    ResponseModel,
    ResponseObservation,
)
from ats_ceramic.schemas import (
    BatchMeasurement,
    ChannelAdjustment,
    DataOrigin,
    MasterMeasurement,
)
from ats_ceramic.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    SyntheticScenario,
    generate,
)
from ats_ceramic.triage import TriageOutcome, TriageResult, triage_comparison
from ats_ceramic.verification import (
    LearningEvidence,
    VerificationAcceptanceCriterion,
    VerificationAssessment,
    VerificationMeasurement,
    assess_verification,
    create_learning_evidence,
    create_verification_measurement,
)

DEMO_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
DEMO_PROVENANCE = "synthetic_generator:v1:streamlit_demo"


@dataclass(frozen=True)
class DemoBundle:
    """Immutable references assembled for one UI demo run."""

    dataset: SyntheticDataset
    registry: GoldenMasterRegistry
    master_record: GoldenMasterRecord
    legacy_record: GoldenMasterRecord
    master_measurement: MasterMeasurement
    audit_log: AuditLog


@dataclass(frozen=True)
class PipelineResult:
    """Results produced by the orchestration layer; domain decisions stay in domain modules."""

    bundle: DemoBundle
    batch: BatchMeasurement | None
    quality: MeasurementQualityResult | None
    comparison: ComparisonResult | None
    triage: TriageResult | None
    calibration_models: tuple[ResponseModel, ...]
    selected_model: ResponseModel | None
    validation: Any | None
    operating_envelope: Any | None
    residual_training: ResidualTrainingResult | None
    residual_prediction: ResidualPrediction | None
    optimization_target: OptimizationTarget | None
    optimization: OptimizationResult | None
    confidence: ConfidenceAssessment | None
    stopped_at: str | None
    stop_reason: str | None


@dataclass(frozen=True)
class QCResult:
    """Outcome of calling the existing human-QC/correction domain functions."""

    decision: HumanQCDecision
    correction_spec: CorrectionSpec | None


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of the existing verification workflow plus optional learning evidence."""

    measurement: VerificationMeasurement
    assessment: VerificationAssessment
    learning_evidence: LearningEvidence | None


def _single_region_master(master: MasterMeasurement, region_id: str) -> MasterMeasurement:
    region = next(region for region in master.regions if region.region_id == region_id)
    return master.model_copy(update={"regions": (region,)})


def _calibration_observations(dataset: SyntheticDataset) -> tuple[CalibrationObservation, ...]:
    """Adapt synthetic observations into the existing calibration/response contracts."""
    observations: list[CalibrationObservation] = []
    for index, observed in enumerate(dataset.calibration_observations):
        master = _single_region_master(dataset.master_measurement, observed.region_id)
        batch = observed.as_batch_measurement()
        comparison = compare_measurements(master, batch)
        response = ResponseObservation(
            sku_id=observed.sku_id,
            context_id=observed.context_id,
            region_id=observed.region_id,
            adjustments=observed.adjustments,
            comparison=comparison,
            data_origin=DataOrigin.SYNTHETIC,
        )
        observations.append(
            CalibrationObservation(
                experiment_id=f"DEMO_EXP_{index:03d}",
                production_run_id=f"DEMO_RUN_{index // 2:03d}",
                measured_at=batch.metadata.measured_at,
                response_observation=response,
                provenance=observed.provenance,
                data_origin=DataOrigin.SYNTHETIC,
            )
        )
    return tuple(observations)


def _demo_channel_inputs(dataset: SyntheticDataset) -> tuple[ControllableInput, ...]:
    """Create the explicit control-plane contract used by the optimizer demo.

    The measurement/model data remain SYNTHETIC.  The optimizer's domain contract
    intentionally requires a client-confirmed controllability record, so this fixture
    represents that control-plane state separately and is never presented as client
    production measurement data.
    """
    return tuple(
        ControllableInput(
            channel_id=channel_id,
            controllable=True,
            lower_bound=dataset.config.channel_lower_bounds[index],
            upper_bound=dataset.config.channel_upper_bounds[index],
            nominal_value=0.0,
            unit="synthetic_relative_adjustment",
            source="synthetic demo control-contract fixture",
            client_confirmed=True,
            data_origin=DataOrigin.CLIENT,
        )
        for index, channel_id in enumerate(dataset.config.channel_ids)
    )


def _audit(
    log: AuditLog,
    *,
    event_id: str,
    event_type: AuditEventType,
    offset_seconds: int,
    dataset: SyntheticDataset,
    status: str | None = None,
    result: str | None = None,
    master_id: str | None = None,
    correction_spec_id: str | None = None,
    details: tuple[tuple[str, str | int | bool | float | None], ...] = (),
) -> None:
    from ats_ceramic.audit import AuditDetail

    log.append(
        create_audit_event(
            event_id=event_id,
            event_type=event_type,
            occurred_at=DEMO_TIME + timedelta(seconds=offset_seconds),
            batch_id=dataset.target_observation.batch_id,
            sku_id=dataset.target_observation.sku_id,
            context_id=dataset.target_observation.context_id,
            region_id=dataset.target_observation.region_id,
            master_id=master_id,
            correction_spec_id=correction_spec_id,
            status=status,
            result=result,
            data_origin=DataOrigin.SYNTHETIC,
            provenance=dataset.provenance,
            details=tuple(AuditDetail(key=key, value=value) for key, value in details),
        )
    )


def create_demo_bundle(scenario: SyntheticScenario) -> DemoBundle:
    """Create deterministic synthetic master, batch, registry, and audit state."""
    dataset = generate(SyntheticConfig(calibration_observations=12), scenario)
    registry = GoldenMasterRegistry()
    content = (
        f"synthetic-golden-master:{dataset.config.sku_id}:{dataset.config.context_id}"
    ).encode()
    master_id = f"MASTER_{dataset.config.sku_id}"
    record = GoldenMasterRecord(
        master_id=master_id,
        version=1,
        sku_id=dataset.config.sku_id,
        context_id=dataset.config.context_id,
        master_kind=MasterKind.CANONICAL_MASTER,
        content_bytes=content,
        content_hash=sha256(content).hexdigest(),
        registered_at=DEMO_TIME,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=dataset.provenance,
    )
    registry.register(record)
    registry.associate_measurement(master_id, 1, dataset.master_measurement)

    legacy_content = content + b":legacy-working-reference"
    legacy = GoldenMasterRecord(
        master_id=f"LEGACY_{dataset.config.sku_id}",
        version=2,
        sku_id=dataset.config.sku_id,
        context_id=dataset.config.context_id,
        master_kind=MasterKind.LEGACY_WORKING_REFERENCE,
        content_bytes=legacy_content,
        content_hash=sha256(legacy_content).hexdigest(),
        registered_at=DEMO_TIME,
        parent_version=1,
        parent_master_id=master_id,
        canonical_master_id=master_id,
        data_origin=DataOrigin.SYNTHETIC,
        provenance=dataset.provenance,
    )
    registry.register(legacy, parent=record)

    audit_log = AuditLog()
    _audit(
        audit_log,
        event_id="DEMO_AUDIT_MASTER",
        event_type=AuditEventType.MASTER_REGISTERED,
        offset_seconds=0,
        dataset=dataset,
        master_id=record.master_id,
        status="registered",
        result=record.master_kind.value,
    )
    return DemoBundle(dataset, registry, record, legacy, dataset.master_measurement, audit_log)


def _target_batch(
    dataset: SyntheticDataset,
    *,
    iteration: int = 0,
    applied_correction_id: str | None = None,
) -> BatchMeasurement:
    """Build a valid batch record while keeping the synthetic target region unchanged."""
    target = dataset.target_observation
    target_region = target.as_region_measurement()
    regions = tuple(
        target_region if master_region.region_id == target.region_id else master_region
        for master_region in dataset.master_measurement.regions
    )
    return BatchMeasurement(
        batch_id=target.batch_id,
        sku_id=target.sku_id,
        context_id=target.context_id,
        iteration=iteration,
        applied_correction_id=applied_correction_id,
        regions=regions,
        metadata=target.as_batch_measurement().metadata,
    )


def _quality_and_comparison(
    bundle: DemoBundle,
) -> tuple[
    BatchMeasurement | None,
    MeasurementQualityResult | None,
    ComparisonResult | None,
    str | None,
]:
    try:
        batch = _target_batch(bundle.dataset)
    except ValueError as exc:
        # The strict domain schema deliberately rejects corrupted observations before
        # downstream processing. The UI reports this as an escalation rather than
        # weakening the schema or fabricating a malformed BatchMeasurement.
        return None, None, None, str(exc)

    tolerance_config = load_tolerance_config()
    # The generator provides one reading per region. The shipped replicate-count
    # recommendation is a placeholder; for this UI-only demo we keep the actual
    # one-reading record and explicitly set that recommendation to 1. No domain
    # validation rule is reimplemented here.
    quality_config = tolerance_config.measurement_quality.model_copy(
        update={"min_replicates_recommended": 1}
    )
    quality = assess_measurement_quality(
        bundle.master_measurement,
        batch,
        quality_config,
    )
    if quality.outcome is not QualityOutcome.ACCEPT:
        return batch, quality, None, quality.explanation
    comparison = compare_measurements(bundle.master_measurement, batch)
    return batch, quality, comparison, None


def _fit_demo_calibration(dataset: SyntheticDataset) -> tuple[tuple[ResponseModel, ...], Any, Any]:
    observations = _calibration_observations(dataset)
    if len({item.production_run_id for item in observations}) < 2:
        raise ValueError("synthetic calibration does not contain enough production runs")
    validation_run = tuple(dict.fromkeys(item.production_run_id for item in observations))[-1]
    training, validation = split_held_out_by_run(observations, (validation_run,))
    model_spec = BayesianSensitivitySpec(
        prior_variance=10.0,
        observation_noise_std=(0.10, 0.10, 0.10),
        min_observations=2,
        is_placeholder=True,
        source="synthetic Streamlit demo calibration configuration",
    )
    models = fit_calibration_models(
        dataset.context,
        dataset.config.sku_id,
        training,
        dataset.config.channel_ids,
        model_spec,
    )
    criterion = ValidationCriterion(
        min_training_runs=1,
        min_validation_runs=1,
        min_validation_observations=1,
        max_mean_abs_error_L=2.0,
        max_mean_abs_error_a=2.0,
        max_mean_abs_error_b=2.0,
    )
    validation_result = validate_held_out(models, training, validation, criterion=criterion)
    envelopes = build_operating_envelopes(
        models,
        training,
        validation_result,
        dataset.master_measurement.metadata.measured_at,
    )
    return models, validation_result, envelopes


def run_pipeline(bundle: DemoBundle) -> PipelineResult:
    """Run the existing domain pipeline in order, stopping at failed safety gates."""
    dataset = bundle.dataset
    batch, quality, comparison, early_reason = _quality_and_comparison(bundle)
    _audit(
        bundle.audit_log,
        event_id="DEMO_AUDIT_MEASUREMENT",
        event_type=(
            AuditEventType.MEASUREMENT_ACCEPTED
            if quality is not None and quality.outcome is QualityOutcome.ACCEPT
            else AuditEventType.MEASUREMENT_REJECTED
        ),
        offset_seconds=1,
        dataset=dataset,
        master_id=bundle.master_record.master_id,
        status=(
            quality.outcome.value
            if quality is not None
            else "rejected_before_domain_measurement"
        ),
        result=early_reason,
    )
    if batch is None or quality is None or quality.outcome is not QualityOutcome.ACCEPT:
        return PipelineResult(
            bundle=bundle,
            batch=batch,
            quality=quality,
            comparison=comparison,
            triage=None,
            calibration_models=(),
            selected_model=None,
            validation=None,
            operating_envelope=None,
            residual_training=None,
            residual_prediction=None,
            optimization_target=None,
            optimization=None,
            confidence=None,
            stopped_at="measurement_quality",
            stop_reason=early_reason,
        )

    triage = triage_comparison(
        comparison,
        load_tolerance_config(),
        sku_id=dataset.target_observation.sku_id,
    )
    _audit(
        bundle.audit_log,
        event_id="DEMO_AUDIT_TRIAGE",
        event_type=AuditEventType.TRIAGE_COMPLETED,
        offset_seconds=2,
        dataset=dataset,
        master_id=bundle.master_record.master_id,
        status=triage.outcome.value,
        result=triage.reason_code.value,
    )
    if triage.outcome is not TriageOutcome.PRINTER_CORRECTABLE:
        return PipelineResult(
            bundle=bundle,
            batch=batch,
            quality=quality,
            comparison=comparison,
            triage=triage,
            calibration_models=(),
            selected_model=None,
            validation=None,
            operating_envelope=None,
            residual_training=None,
            residual_prediction=None,
            optimization_target=None,
            optimization=None,
            confidence=None,
            stopped_at="triage",
            stop_reason=triage.explanation,
        )

    try:
        models, validation, envelopes = _fit_demo_calibration(dataset)
    except ValueError as exc:
        return PipelineResult(
            bundle=bundle,
            batch=batch,
            quality=quality,
            comparison=comparison,
            triage=triage,
            calibration_models=(),
            selected_model=None,
            validation=None,
            operating_envelope=None,
            residual_training=None,
            residual_prediction=None,
            optimization_target=None,
            optimization=None,
            confidence=None,
            stopped_at="response_model",
            stop_reason=str(exc),
        )

    _audit(
        bundle.audit_log,
        event_id="DEMO_AUDIT_MODEL",
        event_type=AuditEventType.RESPONSE_MODEL_FIT,
        offset_seconds=3,
        dataset=dataset,
        master_id=bundle.master_record.master_id,
        status="fitted",
        result=validation.status.value,
    )
    selected_model = next(
        (model for model in models if model.region_id == dataset.target_observation.region_id),
        None,
    )
    envelope = next(
        (item for item in envelopes if item.region_id == dataset.target_observation.region_id),
        None,
    )
    if selected_model is None or envelope is None:
        return PipelineResult(
            bundle=bundle,
            batch=batch,
            quality=quality,
            comparison=comparison,
            triage=triage,
            calibration_models=models,
            selected_model=selected_model,
            validation=validation,
            operating_envelope=envelope,
            residual_training=None,
            residual_prediction=None,
            optimization_target=None,
            optimization=None,
            confidence=None,
            stopped_at="response_model",
            stop_reason="target region is not calibrated",
        )

    residual_model: ResidualModel | None = None
    residual_training: ResidualTrainingResult | None = None
    residual_prediction: ResidualPrediction | None = None
    residual_config = ResidualConfig(
        min_observations=8,
        min_relative_improvement=0.05,
        max_residual_abs=0.75,
        random_state=0,
        n_estimators=32,
        max_depth=3,
        min_samples_leaf=2,
        provenance=dataset.provenance,
    )
    calibration_observations = tuple(
        item
        for item in _calibration_observations(dataset)
        if item.response_observation.region_id == selected_model.region_id
    )
    try:
        residual_model = split_and_fit_residual_model(
            calibration_observations,
            (
                tuple(
                    dict.fromkeys(
                        item.production_run_id
                        for item in calibration_observations
                    )
                )[-1],
            ),
            selected_model,
            residual_config,
        )
        residual_training = residual_model.result
    except ValueError:
        residual_model = None
        residual_training = None

    target_region = next(
        region for region in batch.regions if region.region_id == selected_model.region_id
    )
    master_region = next(
        region
        for region in bundle.master_measurement.regions
        if region.region_id == selected_model.region_id
    )
    target = OptimizationTarget(
        master=master_region,
        batch=target_region,
        data_origin=DataOrigin.SYNTHETIC,
    )
    inputs = _demo_channel_inputs(dataset)
    objective = OptimizerObjectiveSpec(
        delta_e_weight=1.0,
        correction_size_weight=0.01,
        uncertainty_weight=0.05,
        minimum_objective_improvement=0.001,
        trust_region_radius=2.0,
        provenance="synthetic Streamlit demo optimizer configuration",
        is_placeholder=True,
    )
    pre_adjustments = tuple(
        ChannelAdjustment(channel_id=channel_id, delta=0.0)
        for channel_id in selected_model.channel_ids
    )

    if (
        residual_model is not None
        and residual_training is not None
        and residual_training.model_ready
    ):
        try:
            from ats_ceramic.response_model import predict_response

            empirical_prediction = predict_response(selected_model, pre_adjustments)
            residual_prediction = apply_residual(
                residual_model,
                empirical_prediction,
                sku_id=dataset.target_observation.sku_id,
                context_id=dataset.target_observation.context_id,
                region_id=dataset.target_observation.region_id,
                channel_ids=selected_model.channel_ids,
                adjustments=pre_adjustments,
                data_origin=target.data_origin,
                provenance=dataset.provenance,
                operating_envelope=envelope,
            )
        except ValueError:
            residual_prediction = None

    optimization = optimize_region(
        selected_model,
        target,
        inputs,
        objective,
        current_adjustment=pre_adjustments,
        channel_ids=selected_model.channel_ids,
        operating_envelope=envelope,
        provenance="synthetic Streamlit demo optimizer configuration",
    )
    _audit(
        bundle.audit_log,
        event_id="DEMO_AUDIT_OPTIMIZER",
        event_type=AuditEventType.OPTIMIZATION_COMPLETED,
        offset_seconds=4,
        dataset=dataset,
        master_id=bundle.master_record.master_id,
        status=optimization.status.value,
        result=optimization.reason.value,
    )
    if optimization.status is not OptimizerStatus.RECOMMEND:
        return PipelineResult(
            bundle=bundle,
            batch=batch,
            quality=quality,
            comparison=comparison,
            triage=triage,
            calibration_models=models,
            selected_model=selected_model,
            validation=validation,
            operating_envelope=envelope,
            residual_training=residual_training,
            residual_prediction=residual_prediction,
            optimization_target=target,
            optimization=optimization,
            confidence=None,
            stopped_at="optimizer",
            stop_reason=optimization.reason.value,
        )

    confidence_spec = ConfidenceSpec(
        near_boundary_fraction=0.10,
        uncertainty_warn_norm=0.50,
        uncertainty_red_norm=1.50,
        provenance="synthetic Streamlit demo confidence configuration",
        is_placeholder=True,
    )
    confidence = post_solve_confidence(
        selected_model,
        target,
        optimization,
        inputs,
        confidence_spec,
        operating_envelope=envelope,
    )
    _audit(
        bundle.audit_log,
        event_id="DEMO_AUDIT_CONFIDENCE",
        event_type=AuditEventType.CONFIDENCE_ASSESSED,
        offset_seconds=5,
        dataset=dataset,
        master_id=bundle.master_record.master_id,
        status=confidence.status.value,
        result=confidence.ood_status.value,
    )
    return PipelineResult(
        bundle=bundle,
        batch=batch,
        quality=quality,
        comparison=comparison,
        triage=triage,
        calibration_models=models,
        selected_model=selected_model,
        validation=validation,
        operating_envelope=envelope,
        residual_training=residual_training,
        residual_prediction=residual_prediction,
        optimization_target=target,
        optimization=optimization,
        confidence=confidence,
        stopped_at=None,
        stop_reason=None,
    )


def apply_human_qc(
    pipeline: PipelineResult,
    *,
    decision: QCDecision,
    reviewer_id: str,
    reason_comment: str | None = None,
) -> QCResult:
    """Call the existing QC/correction functions; never construct a spec directly."""
    if pipeline.optimization is None or pipeline.confidence is None:
        raise ValueError("a completed optimization and confidence assessment are required")
    if any(
        event.event_type is AuditEventType.QC_DECISION_RECORDED
        for event in pipeline.bundle.audit_log.events()
    ):
        raise ValueError("human QC has already been recorded for this pipeline run")
    spec_id = (
        "CORR_SPEC_"
        f"{pipeline.bundle.dataset.target_observation.batch_id}_"
        f"{pipeline.bundle.dataset.target_observation.region_id}"
    )
    qc = HumanQCDecision(
        decision_id=(
            f"QC_{decision.value.upper()}_"
            f"{pipeline.bundle.dataset.target_observation.batch_id}"
        ),
        correction_spec_id=spec_id,
        sku_id=pipeline.optimization.sku_id,
        context_id=pipeline.optimization.context_id,
        region_id=pipeline.optimization.region_id,
        decision=decision,
        reason_comment=reason_comment,
        reviewer_id=reviewer_id,
        decided_at=DEMO_TIME + timedelta(seconds=6),
        data_origin=DataOrigin.SYNTHETIC,
        provenance=pipeline.confidence.provenance,
    )
    correction_spec = create_correction_spec(
        pipeline.optimization,
        pipeline.confidence,
        qc,
        batch_id=pipeline.bundle.dataset.target_observation.batch_id,
        master_id=pipeline.bundle.master_record.master_id,
        master_content_hash=pipeline.bundle.master_record.content_hash,
    )
    _audit(
        pipeline.bundle.audit_log,
        event_id=f"DEMO_AUDIT_QC_{decision.value}",
        event_type=AuditEventType.QC_DECISION_RECORDED,
        offset_seconds=6,
        dataset=pipeline.bundle.dataset,
        master_id=pipeline.bundle.master_record.master_id,
        correction_spec_id=spec_id,
        status=decision.value,
        result="correction_spec_created" if correction_spec is not None else "rejected",
    )
    if correction_spec is not None:
        _audit(
            pipeline.bundle.audit_log,
            event_id="DEMO_AUDIT_CORRECTION_SPEC",
            event_type=AuditEventType.CORRECTION_SPEC_CREATED,
            offset_seconds=7,
            dataset=pipeline.bundle.dataset,
            master_id=pipeline.bundle.master_record.master_id,
            correction_spec_id=correction_spec.correction_spec_id,
            status="created",
            result="approved_human_qc",
        )
    return QCResult(decision=qc, correction_spec=correction_spec)


def create_instruction_sheet(correction_spec: CorrectionSpec) -> CorrectionInstructionSheet:
    """Use the existing CorrectionSpec adapter, without recalculating corrections."""
    return create_correction_instruction_sheet(correction_spec)


def create_demo_verification(
    correction_spec: CorrectionSpec,
    master: MasterMeasurement,
    *,
    batch: BatchMeasurement,
    baseline_measurement: BatchMeasurement | None = None,
    max_iterations: int = 2,
    criterion: VerificationAcceptanceCriterion | None = None,
) -> VerificationResult:
    """Run the existing verification and learning-evidence boundaries."""
    verification = create_verification_measurement(
        correction_spec,
        master,
        batch,
        master_id=correction_spec.master_id,
        region_id=correction_spec.region_id,
        provenance=correction_spec.provenance,
    )
    assessment = assess_verification(
        correction_spec,
        verification,
        criterion,
        max_iterations=max_iterations,
        baseline_measurement=baseline_measurement,
    )
    evidence = None
    if assessment.status.value == "verified" and baseline_measurement is not None:
        evidence = create_learning_evidence(correction_spec, assessment, baseline_measurement)
    return VerificationResult(verification, assessment, evidence)


def record_verification_audit(
    bundle: DemoBundle,
    result: VerificationResult,
) -> None:
    """Append verification/learning events to the existing immutable AuditLog."""
    dataset = bundle.dataset
    verification_event_id = f"DEMO_AUDIT_VERIFICATION_{result.assessment.iteration}"
    if not any(event.event_id == verification_event_id for event in bundle.audit_log.events()):
        bundle.audit_log.append(
            create_audit_event(
                event_id=verification_event_id,
                event_type=AuditEventType.VERIFICATION_COMPLETED,
                occurred_at=DEMO_TIME + timedelta(seconds=8),
                batch_id=result.assessment.batch_id,
                sku_id=result.assessment.sku_id,
                context_id=result.assessment.context_id,
                region_id=result.assessment.region_id,
                master_id=bundle.master_record.master_id,
                correction_spec_id=result.assessment.correction_spec_id,
                verification_id=(
                    f"VERIFICATION_{result.assessment.correction_spec_id}_"
                    f"{result.assessment.iteration}"
                ),
                status=result.assessment.status.value,
                result=result.assessment.reason.value,
                data_origin=DataOrigin.SYNTHETIC,
                provenance=dataset.provenance,
            )
        )
    if result.learning_evidence is not None and not any(
        event.event_type is AuditEventType.LEARNING_EVIDENCE_CREATED
        and event.correction_spec_id == result.learning_evidence.correction_spec_id
        for event in bundle.audit_log.events()
    ):
        bundle.audit_log.append(
            create_audit_event(
                event_id="DEMO_AUDIT_LEARNING_EVIDENCE",
                event_type=AuditEventType.LEARNING_EVIDENCE_CREATED,
                occurred_at=DEMO_TIME + timedelta(seconds=9),
                batch_id=result.learning_evidence.batch_id,
                sku_id=result.learning_evidence.sku_id,
                context_id=result.learning_evidence.context_id,
                region_id=result.learning_evidence.region_id,
                master_id=bundle.master_record.master_id,
                correction_spec_id=result.learning_evidence.correction_spec_id,
                status="created",
                result="evidence_only_no_retraining",
                data_origin=DataOrigin.SYNTHETIC,
                provenance=dataset.provenance,
            )
        )


def audit_rows(log: AuditLog) -> tuple[dict[str, Any], ...]:
    """Convert immutable audit events to presentation rows without creating another log."""
    return tuple(
        {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.occurred_at.isoformat(),
            "batch_id": event.batch_id,
            "sku": event.sku_id,
            "context": event.context_id,
            "region": event.region_id,
            "master_id": event.master_id,
            "correction_spec_id": event.correction_spec_id,
            "status": event.status,
            "result": event.result,
            "data_origin": event.data_origin.value,
            "provenance": event.provenance,
            "details": "; ".join(f"{detail.key}={detail.value}" for detail in event.details),
        }
        for event in log.events()
    )
