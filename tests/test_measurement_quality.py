"""Tests for the deterministic measurement-quality gate."""

import pytest
from pydantic import ValidationError

from ats_ceramic.config import load_tolerance_config
from ats_ceramic.measurement_quality import (
    MeasurementQualityResult,
    QualityOutcome,
    QualityReasonCode,
    QualitySeverity,
    assess_measurement_quality,
)
from ats_ceramic.schemas import MasterMeasurement, ReplicateSpread


@pytest.fixture
def quality_config():
    return load_tolerance_config().measurement_quality


@pytest.fixture
def make_master(make_region, make_metadata):
    def _make(**overrides):
        values = {
            "sku_id": "SKU_SYN_01",
            "regions": (make_region("R1"), make_region("R2", L=40.0)),
            "metadata": make_metadata(),
        }
        values.update(overrides)
        return MasterMeasurement(**values)

    return _make


def test_clean_measurement_is_accepted(make_master, make_batch, quality_config):
    batch = make_batch(
        regions=tuple(
            r.model_copy(
                update={
                    "n_replicates": 3,
                    "replicate_spread": ReplicateSpread(L=0.0, a=0.0, b=0.0, gloss=0.0),
                }
            )
            for r in make_batch().regions
        )
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert isinstance(result, MeasurementQualityResult)
    assert result.outcome is QualityOutcome.ACCEPT
    assert result.reason_codes == ()
    assert "passed" in result.explanation


def test_sku_mismatch_is_rejected(make_master, make_batch, quality_config):
    result = assess_measurement_quality(
        make_master(), make_batch(sku_id="SKU_OTHER"), quality_config
    )
    assert result.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.SKU_MISMATCH in result.reason_codes


def test_region_mismatch_is_rejected(make_master, make_batch, make_region, quality_config):
    batch = make_batch(regions=(make_region("R1"), make_region("R3")))
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.REGION_MISMATCH in result.reason_codes


def test_metadata_policy_controls_mismatch_severity(make_master, make_batch, quality_config):
    mismatch = make_batch(
        metadata=make_batch().metadata.model_copy(update={"instrument_id": "OTHER"})
    )
    rejected = assess_measurement_quality(make_master(), mismatch, quality_config)
    assert rejected.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.METADATA_INSTRUMENT_MISMATCH in rejected.reason_codes

    warn_config = quality_config.model_copy(
        update={
            "metadata_mismatch": quality_config.metadata_mismatch.model_copy(
                update={"instrument_id": "warn"}
            )
        }
    )
    reviewed = assess_measurement_quality(make_master(), mismatch, warn_config)
    assert reviewed.outcome is QualityOutcome.REVIEW
    assert reviewed.issues[0].severity is QualitySeverity.REVIEW


def test_ab_plausibility_is_rejected(make_master, make_batch, make_region, quality_config):
    batch = make_batch(
        regions=(make_region("R1", a=quality_config.ab_abs_max + 0.1), make_region("R2"))
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.AB_OUT_OF_PLAUSIBLE_RANGE in result.reason_codes


def test_gloss_plausibility_is_separate_from_colour_quality(
    make_master, make_batch, make_region, quality_config
):
    batch = make_batch(
        regions=(make_region("R1", gloss=quality_config.gloss_plausible_max + 1), make_region("R2"))
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.GLOSS_OUT_OF_PLAUSIBLE_RANGE in result.reason_codes


def test_low_replicate_count_is_review_not_reject(make_master, make_batch, quality_config):
    batch = make_batch(
        regions=tuple(r.model_copy(update={"n_replicates": 1}) for r in make_batch().regions)
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.REVIEW
    assert QualityReasonCode.REPLICATES_BELOW_RECOMMENDED in result.reason_codes


def test_missing_replicate_spread_is_review(make_master, make_batch, make_region, quality_config):
    batch = make_batch(regions=(make_region("R1", n_replicates=3), make_region("R2")))
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.REVIEW
    assert QualityReasonCode.REPLICATE_SPREAD_MISSING in result.reason_codes


def test_replicate_spread_warn_and_fail(make_master, make_batch, make_region, quality_config):
    warn_spread = make_region(
        "R1", n_replicates=3, replicate_spread=ReplicateSpread(L=0.5, a=0.0, b=0.0, gloss=0.0)
    )
    warn_batch = make_batch(regions=(warn_spread, make_region("R2")))
    warned = assess_measurement_quality(make_master(), warn_batch, quality_config)
    assert warned.outcome is QualityOutcome.REVIEW
    assert QualityReasonCode.REPLICATE_SPREAD_WARN in warned.reason_codes

    fail_spread = make_region(
        "R1", n_replicates=3, replicate_spread=ReplicateSpread(L=1.1, a=0.0, b=0.0, gloss=0.0)
    )
    fail_batch = make_batch(regions=(fail_spread, make_region("R2")))
    failed = assess_measurement_quality(make_master(), fail_batch, quality_config)
    assert failed.outcome is QualityOutcome.REJECT
    assert QualityReasonCode.REPLICATE_SPREAD_FAIL in failed.reason_codes


def test_gloss_angle_mismatch_is_review_only(make_master, make_batch, quality_config):
    master = make_master(
        metadata=make_batch().metadata.model_copy(update={"gloss_angle_deg": 60.0})
    )
    batch = make_batch(metadata=make_batch().metadata.model_copy(update={"gloss_angle_deg": 20.0}))
    result = assess_measurement_quality(master, batch, quality_config)
    assert result.outcome is QualityOutcome.REVIEW
    assert QualityReasonCode.GLOSS_CONTEXT_MISMATCH in result.reason_codes


def test_result_is_immutable(make_master, make_batch, quality_config):
    result = assess_measurement_quality(make_master(), make_batch(), quality_config)
    with pytest.raises(ValidationError):
        result.outcome = QualityOutcome.REJECT
    assert isinstance(result.issues, tuple)


def test_reason_codes_are_stable_and_deduplicated(
    make_master, make_batch, make_region, quality_config
):
    batch = make_batch(
        regions=(
            make_region(
                "R1",
                a=129.0,
                n_replicates=3,
                replicate_spread={"L": 0.0, "a": 0.0, "b": 0.0, "gloss": 0.0},
            ),
            make_region(
                "R2",
                a=129.0,
                n_replicates=3,
                replicate_spread={"L": 0.0, "a": 0.0, "b": 0.0, "gloss": 0.0},
            ),
        )
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.reason_codes == (QualityReasonCode.AB_OUT_OF_PLAUSIBLE_RANGE,)
    assert len(result.reason_codes) < len(result.issues)


def test_shipped_placeholder_config_does_not_change_gate_semantics(
    make_master, make_batch, quality_config
):
    assert quality_config.is_placeholder is True
    batch = make_batch(
        regions=tuple(
            r.model_copy(
                update={
                    "n_replicates": 3,
                    "replicate_spread": ReplicateSpread(L=0.0, a=0.0, b=0.0, gloss=0.0),
                }
            )
            for r in make_batch().regions
        )
    )
    result = assess_measurement_quality(make_master(), batch, quality_config)
    assert result.outcome is QualityOutcome.ACCEPT
