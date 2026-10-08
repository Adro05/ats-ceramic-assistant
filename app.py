"""Streamlit entry point for the Ceramic Tiles FDE synthetic demonstration."""

from __future__ import annotations

import streamlit as st

from ats_ceramic.correction import QCDecision
from ats_ceramic.schemas import DataOrigin
from ats_ceramic.synthetic import SyntheticScenario
from ats_ceramic.ui import (
    DEMO_PROVENANCE,
    apply_human_qc,
    audit_rows,
    create_demo_bundle,
    create_demo_verification,
    create_instruction_sheet,
    record_verification_audit,
    run_pipeline,
)

st.set_page_config(page_title="ATS Ceramic Assistant", page_icon="🧱", layout="wide")

SCENARIO_LABELS = {
    SyntheticScenario.CORRECTABLE: "CORRECTABLE",
    SyntheticScenario.NEAR_BOUNDARY: "NEAR_BOUNDARY",
    SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE: "PROCESS_SIDE_GLOSS_FAILURE",
    SyntheticScenario.UNREACHABLE: "UNREACHABLE",
    SyntheticScenario.UNSEEN_SKU_OOD: "UNSEEN_SKU_OOD",
    SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT: "NOISY_CORRUPTED_MEASUREMENT",
    SyntheticScenario.SPARSE_CALIBRATION: "SPARSE_CALIBRATION",
}


def _reset() -> None:
    for key in tuple(st.session_state):
        del st.session_state[key]


def _ensure_demo() -> None:
    scenario = st.session_state.get("scenario", SyntheticScenario.CORRECTABLE)
    if "bundle" not in st.session_state or st.session_state.get("bundle_scenario") is not scenario:
        st.session_state.bundle = create_demo_bundle(scenario)
        st.session_state.bundle_scenario = scenario
        st.session_state.pipeline = None
        st.session_state.qc = None
        st.session_state.instruction = None
        st.session_state.verification = None


def _status(label: str, value: str, *, success: bool | None = None) -> None:
    if success is True:
        st.success(f"{label}: {value}")
    elif success is False:
        st.error(f"{label}: {value}")
    else:
        st.info(f"{label}: {value}")


def _overview() -> None:
    st.title("ATS Ceramic Assistant — FDE Prototype")
    st.subheader("Batch shade calibration without Golden Master drift")
    st.write(
        "The current process relies on a 6–7 hour physical trial-and-error loop. "
        "The prototype separates immutable reference truth from a batch-specific correction layer."
    )
    st.info(
        "**Prototype principle:** The master is immutable. "
        "Batch-specific correction is a separate versioned layer."
    )
    st.warning(
        "SYNTHETIC DEMO ONLY — measurements, calibration, thresholds and "
        "placeholder control settings are not client production data."
    )

    stages = [
        "Golden Master",
        "Batch Measurement",
        "Quality Gate",
        "Colour Difference",
        "Triage",
        "Empirical Response Model",
        "Optional Gated ML Residual",
        "Constrained Optimizer",
        "Confidence / OOD",
        "Human QC",
        "CorrectionSpec",
        "Correction Instruction",
        "Verification",
        "Audit Trail",
    ]
    st.markdown(" → ".join(f"**{stage}**" for stage in stages))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Master", "Immutable")
    c2.metric("QC", "Human-gated")
    c3.metric("Release", "Never autonomous")
    c4.metric("Retraining", "Manual only")

    st.markdown("### Safety principles")
    st.markdown(
        "- Human QC remains the final approval boundary.\n"
        "- The Golden Master is never mutated or re-anchored.\n"
        "- No automatic production release.\n"
        "- No automatic retraining.\n"
        "- No invented client tolerances.\n"
        "- Out-of-domain or unsupported recommendations escalate rather than guess."
    )


def _master_page() -> None:
    bundle = st.session_state.bundle
    st.title("Golden Master")
    record = bundle.master_record
    st.success("CANONICAL_MASTER — read-only")
    st.write("The registry stores the canonical master as an immutable, versioned record.")
    st.dataframe(
        [
            {
                "master_id": record.master_id,
                "SKU": record.sku_id,
                "context": record.context_id,
                "version": record.version,
                "content_hash": record.content_hash,
                "kind": record.master_kind.value,
                "data_origin": record.data_origin.value,
                "provenance": record.provenance,
                "registered_at": record.registered_at.isoformat(),
            }
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("No UI action can mutate or re-anchor this record.")

    st.markdown("### Legacy working reference")
    legacy = bundle.legacy_record
    st.warning(
        f"LEGACY_WORKING_REFERENCE — separate artifact; it is not used as the canonical target. "
        f"Canonical master: `{legacy.canonical_master_id}`"
    )
    st.write(
        {
            "legacy_master_id": legacy.master_id,
            "version": legacy.version,
            "content_hash": legacy.content_hash,
            "canonical_master_id": legacy.canonical_master_id,
        }
    )

    st.markdown("### Associated measurements")
    measurements = bundle.registry.measurements_for(record.master_id, record.version)
    st.write(
        f"{len(measurements)} immutable measurement record(s) "
        "associated with this master version."
    )
    for measurement in measurements:
        st.dataframe(
            [
                {
                    "instrument": measurement.metadata.instrument_id,
                    "geometry": measurement.metadata.geometry,
                    "illuminant": measurement.metadata.illuminant,
                    "observer": measurement.metadata.observer.value,
                    "specular": measurement.metadata.specular_mode.value,
                    "regions": len(measurement.regions),
                    "data_origin": measurement.metadata.data_origin.value,
                }
            ],
            use_container_width=True,
            hide_index=True,
        )


def _batch_page() -> None:
    bundle = st.session_state.bundle
    st.title("Batch Measurement")
    target = bundle.dataset.target_observation
    st.write(
        {
            "batch_id": target.batch_id,
            "SKU": target.sku_id,
            "context": target.context_id,
            "region": target.region_id,
            "L*": target.observed_L,
            "a*": target.observed_a,
            "b*": target.observed_b,
            "gloss": target.observed_gloss,
            "data_origin": target.data_origin.value,
            "provenance": target.provenance,
            "corrupted": target.corrupted,
        }
    )
    if target.corrupted:
        st.error(
            "This scenario intentionally contains an invalid raw reading. "
            "The strict domain schema rejects it before downstream "
            "colour/optimizer stages; the UI does not clamp it."
        )

    if st.button("Run correction pipeline", type="primary"):
        # Each explicit run gets a fresh deterministic demo bundle so the append-only
        # AuditLog receives a new event lifecycle instead of duplicate event IDs.
        bundle = create_demo_bundle(st.session_state.bundle_scenario)
        st.session_state.bundle = bundle
        st.session_state.pipeline = run_pipeline(bundle)
        st.session_state.qc = None
        st.session_state.instruction = None
        st.session_state.verification = None


def _pipeline_page() -> None:
    st.title("Correction Pipeline")
    pipeline = st.session_state.pipeline
    if pipeline is None:
        st.info("Run the pipeline from the Batch Measurement page first.")
        return

    stages = [
        ("Measurement Quality Gate", pipeline.quality),
        ("Colour Difference", pipeline.comparison),
        ("Triage", pipeline.triage),
        ("Empirical Response Model", pipeline.selected_model),
        ("Optional Gated ML Residual", pipeline.residual_training),
        ("Constrained Optimizer", pipeline.optimization),
        ("Confidence / OOD", pipeline.confidence),
    ]
    for name, value in stages:
        with st.expander(name, expanded=True):
            if value is None:
                st.error(
                    f"Not reached. Pipeline stopped at `{pipeline.stopped_at}`: "
                    f"{pipeline.stop_reason}"
                )
            else:
                st.write(value)

    if pipeline.comparison is not None:
        st.markdown("### Per-region colour difference")
        st.dataframe(
            [
                {
                    "region": item.region_id,
                    "ΔL*": item.delta_L,
                    "Δa*": item.delta_a,
                    "Δb*": item.delta_b,
                    "ΔE00": item.delta_E00,
                    "Δgloss": item.delta_gloss,
                }
                for item in pipeline.comparison.regions
            ],
            use_container_width=True,
            hide_index=True,
        )

    if pipeline.residual_training is not None:
        st.caption(
            "The ML residual is optional and secondary. It can only be "
            "enabled by the existing training/ablation gates."
        )
        st.write(
            {
                "status": pipeline.residual_training.status.value,
                "model_ready": pipeline.residual_training.model_ready,
                "training_observations": pipeline.residual_training.training_observation_count,
                "held_out_observations": pipeline.residual_training.held_out_observation_count,
            }
        )
    else:
        st.info("Optional gated ML residual was not reached or could not be trained safely.")

    if pipeline.optimization is not None:
        st.markdown("### Recommendation")
        st.write(pipeline.optimization)
    if pipeline.confidence is not None:
        _status(
            "Confidence",
            f"{pipeline.confidence.status.value.upper()} / "
            f"OOD {pipeline.confidence.ood_status.value}",
            success=pipeline.confidence.status.value == "green",
        )


def _qc_page() -> None:
    st.title("QC / CorrectionSpec")
    pipeline = st.session_state.pipeline
    if pipeline is None or pipeline.optimization is None or pipeline.confidence is None:
        st.info(
            "A valid recommendation must reach this page through the "
            "existing pipeline gates."
        )
        return

    st.write(
        {
            "recommendation_status": pipeline.optimization.status.value,
            "confidence": pipeline.confidence.status.value,
            "OOD": pipeline.confidence.ood_status.value,
            "SKU": pipeline.optimization.sku_id,
            "context": pipeline.optimization.context_id,
            "region": pipeline.optimization.region_id,
            "master_id": pipeline.bundle.master_record.master_id,
            "master_hash": pipeline.bundle.master_record.content_hash,
            "data_origin": pipeline.optimization.data_origin.value,
            "provenance": pipeline.bundle.dataset.provenance,
        }
    )
    st.write("Proposed adjustments", pipeline.optimization.recommended_adjustments)
    st.write("Current adjustment", pipeline.optimization.current_adjustment)

    reviewer = st.text_input("Reviewer ID", value="DEMO_REVIEWER")
    comment = st.text_input("QC comment")
    c1, c2 = st.columns(2)
    qc_recorded = st.session_state.get("qc") is not None
    if c1.button("APPROVE", type="primary", disabled=qc_recorded):
        try:
            st.session_state.qc = apply_human_qc(
                pipeline,
                decision=QCDecision.APPROVE,
                reviewer_id=reviewer,
                reason_comment=comment or None,
            )
            st.session_state.instruction = None
        except ValueError as exc:
            st.error(str(exc))
    if c2.button("REJECT", disabled=qc_recorded):
        try:
            st.session_state.qc = apply_human_qc(
                pipeline,
                decision=QCDecision.REJECT,
                reviewer_id=reviewer,
                reason_comment=comment or None,
            )
            st.session_state.instruction = None
        except ValueError as exc:
            st.error(str(exc))

    qc = st.session_state.get("qc")
    if qc is not None:
        st.write(qc.decision)
        if qc.correction_spec is None:
            st.warning("Rejected: no executable CorrectionSpec was created.")
        else:
            st.success(f"CorrectionSpec created: {qc.correction_spec.correction_spec_id}")
            st.json(qc.correction_spec.model_dump(mode="json"))



def _instruction_page() -> None:
    st.title("Correction Instruction")
    qc = st.session_state.get("qc")
    if qc is None or qc.correction_spec is None:
        st.info(
            "An approved CorrectionSpec is required before an instruction "
            "sheet can be generated."
        )
        return
    if st.session_state.get("instruction") is None:
        st.session_state.instruction = create_instruction_sheet(qc.correction_spec)
    instruction = st.session_state.instruction
    st.success("Human-readable instruction sheet generated from the approved CorrectionSpec.")
    st.text_area("Instruction sheet", instruction.instruction_text, height=460)
    st.caption(
        "The adapter performs no new colour calculation or correction "
        "calculation. The Golden Master remains unchanged."
    )


def _verification_page() -> None:
    st.title("Verification")
    qc = st.session_state.get("qc")
    if qc is None or qc.correction_spec is None:
        st.info("Verification requires an approved CorrectionSpec.")
        return

    spec = qc.correction_spec
    bundle = st.session_state.bundle
    pipeline = st.session_state.pipeline
    target = bundle.dataset.target_observation
    st.write(
        "Verification uses the existing verification module. "
        "No client tolerance is invented here."
    )

    # A deterministic synthetic verification measurement is intentionally derived from the
    # observed synthetic target, not from the hidden synthetic oracle or target_required_adjustment.
    verified_regions = tuple(
        region.model_copy(
            update={
                "L": region.L - 0.8 * (region.L - 50.0),
                "a": region.a - 0.8 * (region.a - 8.0),
                "b": region.b - 0.8 * (region.b - 4.0),
                "gloss": region.gloss,
            }
        )
        for region in target.as_batch_measurement().regions
    ) if not target.corrupted else ()

    if not verified_regions:
        st.warning(
            "No valid synthetic verification measurement can be generated "
            "for the corrupted-measurement scenario."
        )
        return

    criterion = st.session_state.get("verification_criterion")
    if criterion is None:
        st.caption(
            "Enter explicit synthetic placeholder criteria to demonstrate "
            "the existing acceptance mechanism."
        )
        de_threshold = st.number_input("Synthetic ΔE00 criterion", min_value=0.01, value=1.0)
        gloss_threshold = st.number_input(
            "Synthetic absolute gloss criterion", min_value=0.0, value=5.0
        )
        if st.button("Set verification criterion"):
            from ats_ceramic.schemas import AcceptanceMetric
            from ats_ceramic.verification import VerificationAcceptanceCriterion

            st.session_state.verification_criterion = VerificationAcceptanceCriterion(
                acceptance_metric=AcceptanceMetric.MAX_DE00,
                delta_e00_threshold=de_threshold,
                gloss_abs_threshold=gloss_threshold,
                data_origin=DataOrigin.SYNTHETIC,
                provenance=DEMO_PROVENANCE,
                is_placeholder=True,
            )
            st.rerun()
        return

    measurement = target.as_batch_measurement().model_copy(
        update={
            "iteration": 1,
            "applied_correction_id": spec.correction_spec_id,
            "regions": verified_regions,
        }
    )
    if st.button("Run verification"):
        try:
            st.session_state.verification = create_demo_verification(
                spec,
                bundle.master_measurement,
                batch=measurement,
                baseline_measurement=pipeline.batch,
                max_iterations=2,
                criterion=criterion,
            )
            record_verification_audit(bundle, st.session_state.verification)
        except ValueError as exc:
            st.error(str(exc))

    verification = st.session_state.get("verification")
    if verification is not None:
        st.write(verification.assessment)
        if verification.assessment.status.value == "verified":
            st.success("Verification passed.")
        elif verification.assessment.status.value == "iteration_allowed":
            st.warning(
                "Verification did not meet the explicit criterion; "
                "another bounded iteration is allowed."
            )
        else:
            st.error("Verification escalates according to the existing domain logic.")


def _audit_page() -> None:
    st.title("Audit Trail")
    rows = audit_rows(st.session_state.bundle.audit_log)
    if not rows:
        st.info("No audit events yet.")
        return
    batch_filter = st.text_input("Filter by batch ID", value="")
    correction_filter = st.text_input("Filter by CorrectionSpec ID", value="")
    master_filter = st.text_input("Filter by master ID", value="")
    filtered = tuple(
        row
        for row in rows
        if (not batch_filter or row["batch_id"] == batch_filter)
        and (not correction_filter or row["correction_spec_id"] == correction_filter)
        and (not master_filter or row["master_id"] == master_filter)
    )
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.title("Demo controls")
    scenario = st.sidebar.selectbox(
        "Synthetic scenario",
        options=tuple(SyntheticScenario),
        format_func=lambda value: SCENARIO_LABELS[value],
    )
    st.session_state.scenario = scenario
    if st.sidebar.button("Reset Demo"):
        _reset()
        st.rerun()
    _ensure_demo()

    st.sidebar.warning("SYNTHETIC DATA ONLY")
    st.sidebar.caption(
        "No production release, master mutation, or automatic retraining "
        "is available."
    )
    page = st.sidebar.radio(
        "Page",
        [
            "Overview / Demo",
            "Golden Master",
            "Batch Measurement",
            "Correction Pipeline",
            "QC / CorrectionSpec",
            "Correction Instruction",
            "Verification",
            "Audit Trail",
        ],
    )

    if page == "Overview / Demo":
        _overview()
    elif page == "Golden Master":
        _master_page()
    elif page == "Batch Measurement":
        _batch_page()
    elif page == "Correction Pipeline":
        _pipeline_page()
    elif page == "QC / CorrectionSpec":
        _qc_page()
    elif page == "Correction Instruction":
        _instruction_page()
    elif page == "Verification":
        _verification_page()
    else:
        _audit_page()


if __name__ == "__main__":
    main()

