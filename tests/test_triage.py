"""Tests for deterministic colour/gloss triage."""

import pytest
from pydantic import ValidationError

from ats_ceramic.color import ComparisonResult, RegionDifference
from ats_ceramic.config import SkuToleranceOverride, load_tolerance_config
from ats_ceramic.schemas import AcceptanceMetric, ToleranceSpec
from ats_ceramic.triage import (
    TriageOutcome,
    TriageReasonCode,
    triage_comparison,
)


def _comparison(*, mean_de00: float, max_de00: float, max_abs_gloss: float) -> ComparisonResult:
    return ComparisonResult(
        regions=(
            RegionDifference(
                region_id="R1",
                delta_L=0.0,
                delta_a=0.0,
                delta_b=0.0,
                delta_E00=max_de00,
                delta_gloss=max_abs_gloss,
            ),
        ),
        mean_delta_E00=mean_de00,
        max_delta_E00=max_de00,
        worst_region_id="R1",
        mean_abs_delta_gloss=max_abs_gloss,
        max_abs_delta_gloss=max_abs_gloss,
    )


def _config(*, metric: AcceptanceMetric = AcceptanceMetric.MAX_DE00):
    config = load_tolerance_config()
    tolerance = config.acceptance.default.model_copy(update={"acceptance_metric": metric})
    acceptance = config.acceptance.model_copy(update={"default": tolerance})
    return config.model_copy(update={"acceptance": acceptance})


def test_color_outside_gloss_acceptable_is_printer_correctable():
    config = _config()
    threshold = config.acceptance.default.delta_e00_threshold
    comparison = _comparison(
        mean_de00=threshold + 0.1,
        max_de00=threshold + 0.1,
        max_abs_gloss=config.acceptance.default.gloss_abs_threshold,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.PRINTER_CORRECTABLE
    assert result.reason_code is TriageReasonCode.COLOR_OUTSIDE_TOLERANCE
    assert "printer-correction candidate" in result.explanation


def test_gloss_outside_color_acceptable_is_process_side():
    config = _config()
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold,
        max_de00=tolerance.delta_e00_threshold,
        max_abs_gloss=tolerance.gloss_abs_threshold + 0.1,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.LIKELY_PROCESS_SIDE
    assert result.reason_code is TriageReasonCode.GLOSS_OUTSIDE_TOLERANCE
    assert "blind printer correction" in result.explanation


def test_both_color_and_gloss_outside_are_process_side():
    config = _config()
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold + 0.1,
        max_de00=tolerance.delta_e00_threshold + 0.1,
        max_abs_gloss=tolerance.gloss_abs_threshold + 0.1,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.LIKELY_PROCESS_SIDE
    assert result.reason_code is TriageReasonCode.COLOR_AND_GLOSS_OUTSIDE_TOLERANCE


def test_both_within_tolerance_are_indeterminate():
    config = _config()
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold - 0.1,
        max_de00=tolerance.delta_e00_threshold - 0.1,
        max_abs_gloss=tolerance.gloss_abs_threshold - 0.1,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.INDETERMINATE
    assert result.reason_code is TriageReasonCode.COLOR_WITHIN_TOLERANCE_GLOSS_ACCEPTABLE


@pytest.mark.parametrize("metric", [AcceptanceMetric.MAX_DE00, AcceptanceMetric.MEAN_DE00])
def test_exact_color_tolerance_boundary_is_within(metric):
    config = _config(metric=metric)
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold,
        max_de00=tolerance.delta_e00_threshold,
        max_abs_gloss=tolerance.gloss_abs_threshold,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.INDETERMINATE


@pytest.mark.parametrize("metric", [AcceptanceMetric.MAX_DE00, AcceptanceMetric.MEAN_DE00])
def test_exact_gloss_tolerance_boundary_is_acceptable(metric):
    config = _config(metric=metric)
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold + 0.1,
        max_de00=tolerance.delta_e00_threshold + 0.1,
        max_abs_gloss=tolerance.gloss_abs_threshold,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.PRINTER_CORRECTABLE
    assert result.reason_code is TriageReasonCode.COLOR_OUTSIDE_TOLERANCE


def test_max_de00_metric_uses_max_not_mean():
    config = _config(metric=AcceptanceMetric.MAX_DE00)
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold - 0.1,
        max_de00=tolerance.delta_e00_threshold + 0.1,
        max_abs_gloss=tolerance.gloss_abs_threshold,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.PRINTER_CORRECTABLE
    assert result.reason_code is TriageReasonCode.COLOR_OUTSIDE_TOLERANCE


def test_mean_de00_metric_uses_mean_not_max():
    config = _config(metric=AcceptanceMetric.MEAN_DE00)
    tolerance = config.acceptance.default
    comparison = _comparison(
        mean_de00=tolerance.delta_e00_threshold - 0.1,
        max_de00=tolerance.delta_e00_threshold + 0.1,
        max_abs_gloss=tolerance.gloss_abs_threshold,
    )

    result = triage_comparison(comparison, config)

    assert result.outcome is TriageOutcome.INDETERMINATE
    assert result.reason_code is TriageReasonCode.COLOR_WITHIN_TOLERANCE_GLOSS_ACCEPTABLE


def test_sku_specific_tolerance_is_used():
    config = load_tolerance_config()
    override = ToleranceSpec(
        acceptance_metric=AcceptanceMetric.MAX_DE00,
        delta_e00_threshold=2.0,
        gloss_abs_threshold=config.acceptance.default.gloss_abs_threshold,
        is_placeholder=True,
        source="synthetic test override",
    )
    sku_override = SkuToleranceOverride(sku_id="SKU_TEST", tolerance=override)
    acceptance = config.acceptance.model_copy(update={"sku_overrides": (sku_override,)})
    config_with_override = config.model_copy(update={"acceptance": acceptance})

    comparison = _comparison(mean_de00=1.5, max_de00=1.5, max_abs_gloss=0.0)
    default_result = triage_comparison(comparison, config)
    override_result = triage_comparison(comparison, config_with_override, sku_id="SKU_TEST")

    assert default_result.outcome is TriageOutcome.PRINTER_CORRECTABLE
    assert override_result.outcome is TriageOutcome.INDETERMINATE


def test_result_is_immutable():
    config = _config()
    tolerance = config.acceptance.default
    result = triage_comparison(
        _comparison(
            mean_de00=tolerance.delta_e00_threshold,
            max_de00=tolerance.delta_e00_threshold,
            max_abs_gloss=tolerance.gloss_abs_threshold,
        ),
        config,
    )

    with pytest.raises(ValidationError):
        result.outcome = TriageOutcome.PRINTER_CORRECTABLE


def test_reason_codes_are_stable_and_single_valued():
    config = _config()
    tolerance = config.acceptance.default
    result = triage_comparison(
        _comparison(
            mean_de00=tolerance.delta_e00_threshold + 1.0,
            max_de00=tolerance.delta_e00_threshold + 1.0,
            max_abs_gloss=tolerance.gloss_abs_threshold + 1.0,
        ),
        config,
    )

    assert result.reason_code.value == "COLOR_AND_GLOSS_OUTSIDE_TOLERANCE"
    assert result.reason_codes == (TriageReasonCode.COLOR_AND_GLOSS_OUTSIDE_TOLERANCE,)
