"""Focused tests for Chunk 11 CorrectionSpec Adapter A.

All values come from the existing explicitly synthetic CorrectionSpec fixtures.
They are not client printer settings, production tolerances, or production records.
"""

import re

import pytest
from pydantic import ValidationError
from test_correction import _approved_case

from ats_ceramic.correction import QCDecision, create_correction_spec
from ats_ceramic.correction_adapter import (
    CorrectionInstructionSheet,
    create_correction_instruction_sheet,
)
from ats_ceramic.schemas import DataOrigin

HASH = "b" * 64


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


def test_valid_approved_correction_spec_produces_instruction_sheet():
    sheet = create_correction_instruction_sheet(_spec())
    assert isinstance(sheet, CorrectionInstructionSheet)
    assert sheet.qc_decision is QCDecision.APPROVE


def test_instruction_sheet_contains_core_identifiers():
    sheet = create_correction_instruction_sheet(_spec())
    text = sheet.instruction_text

    for value in (
        sheet.correction_spec_id,
        sheet.batch_id,
        sheet.sku_id,
        sheet.context_id,
        sheet.region_id,
        sheet.master_id,
    ):
        assert value in text


def test_instruction_sheet_explicitly_keeps_golden_master_unchanged():
    text = create_correction_instruction_sheet(_spec()).instruction_text
    assert "Golden Master status: UNCHANGED" in text
    assert "The Golden Master remains unchanged." in text


def test_all_recommended_channel_adjustments_are_present_with_signed_values():
    sheet = create_correction_instruction_sheet(_spec())

    for adjustment in sheet.recommended_adjustments:
        expected = f"Channel {adjustment.channel_id}: {adjustment.delta}"
        assert expected in sheet.instruction_text
        assert adjustment.delta < 0.0


def test_current_adjustment_is_preserved():
    sheet = create_correction_instruction_sheet(_spec())
    for adjustment in sheet.current_adjustment:
        assert (
            f"Channel {adjustment.channel_id}: {adjustment.delta}"
            in sheet.instruction_text
        )


def test_confidence_ood_and_qc_approval_are_present():
    sheet = create_correction_instruction_sheet(_spec())
    assert f"Status: {sheet.confidence_status}" in sheet.instruction_text
    assert f"OOD: {sheet.ood_status}" in sheet.instruction_text
    assert "Decision: APPROVE" in sheet.instruction_text
    assert f"Reviewer: {sheet.reviewer_id}" in sheet.instruction_text


def test_data_origin_and_provenance_are_preserved():
    spec = _spec()
    sheet = create_correction_instruction_sheet(spec)

    assert sheet.data_origin is spec.data_origin is DataOrigin.SYNTHETIC
    assert sheet.provenance == spec.provenance
    assert f"Data origin: {spec.data_origin.value}" in sheet.instruction_text
    assert f"Provenance: {spec.provenance}" in sheet.instruction_text
    assert "SYNTHETIC" in sheet.instruction_text.upper()


def test_verification_follow_up_is_explicit():
    text = create_correction_instruction_sheet(_spec()).instruction_text
    assert "existing verification workflow" in text
    assert "resulting batch measurement" in text


def test_same_correction_spec_produces_identical_output():
    spec = _spec()
    first = create_correction_instruction_sheet(spec)
    second = create_correction_instruction_sheet(spec)
    assert first == second
    assert first.instruction_text == second.instruction_text


def test_rejected_qc_does_not_produce_a_correction_spec_or_instruction_sheet():
    result, confidence, decision = _approved_case()
    rejected = decision.model_copy(update={"decision": QCDecision.REJECT})

    assert create_correction_spec(
        result,
        confidence,
        rejected,
        batch_id="B-SYN-1",
        master_id="M-SYN-1",
        master_content_hash=HASH,
    ) is None

    with pytest.raises(ValueError, match="valid CorrectionSpec"):
        create_correction_instruction_sheet(None)  # type: ignore[arg-type]


def test_instruction_text_contains_no_timestamp_or_random_content():
    text = create_correction_instruction_sheet(_spec()).instruction_text
    assert "2026-" not in text
    assert re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text) is None
    assert "random" not in text.lower()
    assert "timestamp" not in text.lower()


def test_adapter_does_not_mutate_original_correction_spec():
    spec = _spec()
    before = spec.model_dump(mode="json")
    create_correction_instruction_sheet(spec)
    assert spec.model_dump(mode="json") == before


def test_instruction_sheet_is_immutable():
    sheet = create_correction_instruction_sheet(_spec())
    with pytest.raises((ValidationError, TypeError)):
        sheet.instruction_text = "changed"


def test_adapter_does_not_invent_printer_or_rip_instructions():
    text = create_correction_instruction_sheet(_spec()).instruction_text.lower()
    for forbidden in (
        "rip",
        "icc profile",
        "printer command",
        "submit to printer",
        "print job",
    ):
        assert forbidden not in text


def test_adapter_does_not_claim_guaranteed_success_or_invent_tolerances():
    text = create_correction_instruction_sheet(_spec()).instruction_text.lower()
    assert "guaranteed" not in text
    assert "tolerance" not in text
    assert "delta_e00 threshold" not in text
    assert "gloss threshold" not in text


def test_adapter_preserves_exact_spec_values_in_output_model():
    spec = _spec()
    sheet = create_correction_instruction_sheet(spec)

    assert sheet.correction_spec_id == spec.correction_spec_id
    assert sheet.batch_id == spec.batch_id
    assert sheet.master_id == spec.master_id
    assert sheet.sku_id == spec.sku_id
    assert sheet.context_id == spec.context_id
    assert sheet.region_id == spec.region_id
    assert sheet.recommended_adjustments == spec.recommended_adjustments
    assert sheet.current_adjustment == spec.current_adjustment
    assert sheet.confidence_status == spec.confidence_assessment.status.value
    assert sheet.ood_status == spec.confidence_assessment.ood_status.value
    assert sheet.reviewer_id == spec.qc_decision.reviewer_id


def test_instruction_sheet_channel_ids_follow_spec_recommendation_order():
    sheet = create_correction_instruction_sheet(_spec())
    assert sheet.channel_ids == tuple(
        item.channel_id for item in sheet.recommended_adjustments
    )