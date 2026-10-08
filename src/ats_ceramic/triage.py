"""Deterministic triage of a validated colour comparison.

Pipeline position::

    measurement quality gate -> colour comparison -> triage

Triage is deliberately diagnostic rather than corrective. It keeps colour and gloss
as separate signals and does not calculate or recommend any printer adjustment.
"""

from enum import StrEnum

from pydantic import Field

from ats_ceramic.color import ComparisonResult
from ats_ceramic.config import ToleranceConfig
from ats_ceramic.schemas import AcceptanceMetric, FrozenModel


class TriageOutcome(StrEnum):
    """Exactly one deterministic diagnostic disposition."""

    PRINTER_CORRECTABLE = "PRINTER_CORRECTABLE"
    LIKELY_PROCESS_SIDE = "LIKELY_PROCESS_SIDE"
    INDETERMINATE = "INDETERMINATE"


class TriageReasonCode(StrEnum):
    """Stable machine-readable reasons for a triage outcome."""

    COLOR_OUTSIDE_TOLERANCE = "COLOR_OUTSIDE_TOLERANCE"
    GLOSS_OUTSIDE_TOLERANCE = "GLOSS_OUTSIDE_TOLERANCE"
    COLOR_AND_GLOSS_OUTSIDE_TOLERANCE = "COLOR_AND_GLOSS_OUTSIDE_TOLERANCE"
    COLOR_WITHIN_TOLERANCE_GLOSS_ACCEPTABLE = "COLOR_WITHIN_TOLERANCE_GLOSS_ACCEPTABLE"


class TriageResult(FrozenModel):
    """Immutable result of deterministic colour/gloss triage."""

    outcome: TriageOutcome
    reason_code: TriageReasonCode
    explanation: str = Field(min_length=1)

    @property
    def reason_codes(self) -> tuple[TriageReasonCode, ...]:
        """Return the stable reason code in tuple form for machine-readable consumers."""
        return (self.reason_code,)


def _color_value(comparison: ComparisonResult, metric: AcceptanceMetric) -> float:
    if metric is AcceptanceMetric.MAX_DE00:
        return comparison.max_delta_E00
    if metric is AcceptanceMetric.MEAN_DE00:
        return comparison.mean_delta_E00
    raise ValueError(f"unsupported acceptance metric: {metric!r}")


def triage_comparison(
    comparison: ComparisonResult,
    tolerance_config: ToleranceConfig,
    *,
    sku_id: str | None = None,
) -> TriageResult:
    """Classify a comparison using the configured colour and gloss tolerances.

    ``sku_id`` selects an existing SKU-specific tolerance when supplied. If omitted,
    the configured default tolerance is used. Gloss is evaluated independently from
    colour. Any gloss exceedance takes the conservative process-side path, including
    when colour is also outside tolerance.

    This function assumes the comparison was produced after the measurement-quality
    gate. It does not perform measurement validation, correction calculation,
    optimization, or causal diagnosis.
    """
    tolerance = (
        tolerance_config.for_sku(sku_id)
        if sku_id is not None
        else tolerance_config.acceptance.default
    )
    color_value = _color_value(comparison, tolerance.acceptance_metric)
    color_outside = color_value > tolerance.delta_e00_threshold
    gloss_outside = comparison.max_abs_delta_gloss > tolerance.gloss_abs_threshold

    if gloss_outside and color_outside:
        return TriageResult(
            outcome=TriageOutcome.LIKELY_PROCESS_SIDE,
            reason_code=TriageReasonCode.COLOR_AND_GLOSS_OUTSIDE_TOLERANCE,
            explanation=(
                "Both the configured colour metric and gloss deviation are outside "
                "tolerance; process-side evidence is present, so blind printer correction "
                "is not justified."
            ),
        )

    if gloss_outside:
        return TriageResult(
            outcome=TriageOutcome.LIKELY_PROCESS_SIDE,
            reason_code=TriageReasonCode.GLOSS_OUTSIDE_TOLERANCE,
            explanation=(
                "Gloss deviation is outside the configured tolerance; this is process-side "
                "evidence, so blind printer correction is not justified."
            ),
        )

    if color_outside:
        return TriageResult(
            outcome=TriageOutcome.PRINTER_CORRECTABLE,
            reason_code=TriageReasonCode.COLOR_OUTSIDE_TOLERANCE,
            explanation=(
                "The configured colour metric is outside tolerance while gloss is acceptable; "
                "the deviation is a printer-correction candidate for human QC review."
            ),
        )

    return TriageResult(
        outcome=TriageOutcome.INDETERMINATE,
        reason_code=TriageReasonCode.COLOR_WITHIN_TOLERANCE_GLOSS_ACCEPTABLE,
        explanation=(
            "Colour is within the configured tolerance and gloss is acceptable; no demonstrated "
            "correction need or safe causal determination is established."
        ),
    )
