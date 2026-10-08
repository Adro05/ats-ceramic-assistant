"""Deterministic measurement-quality gate for validated measurement objects.

Pipeline position::

    validated measurement objects -> measurement quality gate -> colour engine

The gate does not calculate colour differences or apply client acceptance tolerances.
It checks measurement usability and comparability using only the existing domain schemas
and configurable measurement-quality thresholds.
"""

from enum import StrEnum

from pydantic import Field

from ats_ceramic.config import MeasurementQualityConfig, MismatchPolicy
from ats_ceramic.schemas import BatchMeasurement, FrozenModel, MasterMeasurement


class QualityOutcome(StrEnum):
    """Overall disposition of a measurement-quality assessment."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class QualitySeverity(StrEnum):
    """Severity assigned to one deterministic quality finding."""

    REVIEW = "REVIEW"
    REJECT = "REJECT"


class QualityReasonCode(StrEnum):
    """Stable machine-readable reason codes emitted by the quality gate."""

    SKU_MISMATCH = "SKU_MISMATCH"
    REGION_MISMATCH = "REGION_MISMATCH"
    METADATA_INSTRUMENT_MISMATCH = "METADATA_INSTRUMENT_MISMATCH"
    METADATA_GEOMETRY_MISMATCH = "METADATA_GEOMETRY_MISMATCH"
    METADATA_ILLUMINANT_MISMATCH = "METADATA_ILLUMINANT_MISMATCH"
    METADATA_OBSERVER_MISMATCH = "METADATA_OBSERVER_MISMATCH"
    METADATA_SPECULAR_MODE_MISMATCH = "METADATA_SPECULAR_MODE_MISMATCH"
    METADATA_APERTURE_MISMATCH = "METADATA_APERTURE_MISMATCH"
    AB_OUT_OF_PLAUSIBLE_RANGE = "AB_OUT_OF_PLAUSIBLE_RANGE"
    GLOSS_OUT_OF_PLAUSIBLE_RANGE = "GLOSS_OUT_OF_PLAUSIBLE_RANGE"
    REPLICATES_BELOW_RECOMMENDED = "REPLICATES_BELOW_RECOMMENDED"
    REPLICATE_SPREAD_WARN = "REPLICATE_SPREAD_WARN"
    REPLICATE_SPREAD_FAIL = "REPLICATE_SPREAD_FAIL"
    REPLICATE_SPREAD_MISSING = "REPLICATE_SPREAD_MISSING"
    GLOSS_CONTEXT_MISMATCH = "GLOSS_CONTEXT_MISMATCH"


class QualityIssue(FrozenModel):
    """One quality finding, retained in deterministic evaluation order."""

    code: QualityReasonCode
    severity: QualitySeverity
    message: str = Field(min_length=1)
    region_id: str | None = None


class MeasurementQualityResult(FrozenModel):
    """Immutable outcome of a measurement-quality assessment."""

    outcome: QualityOutcome
    issues: tuple[QualityIssue, ...] = ()

    @property
    def reason_codes(self) -> tuple[QualityReasonCode, ...]:
        """Unique reason codes in first-seen deterministic order."""
        seen: set[QualityReasonCode] = set()
        codes: list[QualityReasonCode] = []
        for issue in self.issues:
            if issue.code not in seen:
                seen.add(issue.code)
                codes.append(issue.code)
        return tuple(codes)

    @property
    def explanation(self) -> str:
        """Human-readable deterministic explanation of the outcome."""
        if not self.issues:
            return (
                "Measurement quality checks passed; measurement is suitable for colour comparison."
            )
        return " ".join(issue.message for issue in self.issues)


def _issue(
    code: QualityReasonCode,
    severity: QualitySeverity,
    message: str,
    region_id: str | None = None,
) -> QualityIssue:
    return QualityIssue(code=code, severity=severity, message=message, region_id=region_id)


def _metadata_issues(
    master: MasterMeasurement, batch: BatchMeasurement, config: MeasurementQualityConfig
) -> list[QualityIssue]:
    checks = (
        ("instrument_id", QualityReasonCode.METADATA_INSTRUMENT_MISMATCH),
        ("geometry", QualityReasonCode.METADATA_GEOMETRY_MISMATCH),
        ("illuminant", QualityReasonCode.METADATA_ILLUMINANT_MISMATCH),
        ("observer", QualityReasonCode.METADATA_OBSERVER_MISMATCH),
        ("specular_mode", QualityReasonCode.METADATA_SPECULAR_MODE_MISMATCH),
        ("aperture_mm", QualityReasonCode.METADATA_APERTURE_MISMATCH),
    )
    issues: list[QualityIssue] = []
    for field_name, code in checks:
        policy = getattr(config.metadata_mismatch, field_name)
        if policy is MismatchPolicy.IGNORE:
            continue
        master_value = getattr(master.metadata, field_name)
        batch_value = getattr(batch.metadata, field_name)
        if master_value == batch_value:
            continue
        severity = (
            QualitySeverity.REJECT if policy is MismatchPolicy.FAIL else QualitySeverity.REVIEW
        )
        issues.append(
            _issue(
                code,
                severity,
                f"Metadata mismatch for {field_name}: master={master_value!r}, "
                f"batch={batch_value!r}.",
            )
        )
    return issues


def _structural_issues(master: MasterMeasurement, batch: BatchMeasurement) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if master.sku_id != batch.sku_id:
        issues.append(
            _issue(
                QualityReasonCode.SKU_MISMATCH,
                QualitySeverity.REJECT,
                f"SKU mismatch: master={master.sku_id!r}, batch={batch.sku_id!r}.",
            )
        )

    master_ids = tuple(region.region_id for region in master.regions)
    batch_ids = tuple(region.region_id for region in batch.regions)
    if set(master_ids) != set(batch_ids):
        missing = sorted(set(master_ids) - set(batch_ids))
        unexpected = sorted(set(batch_ids) - set(master_ids))
        issues.append(
            _issue(
                QualityReasonCode.REGION_MISMATCH,
                QualitySeverity.REJECT,
                f"Region mismatch: missing from batch={missing}, unexpected in batch={unexpected}.",
            )
        )
    return issues


def _measurement_issues(
    batch: BatchMeasurement, config: MeasurementQualityConfig
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for region in batch.regions:
        if abs(region.a) > config.ab_abs_max or abs(region.b) > config.ab_abs_max:
            issues.append(
                _issue(
                    QualityReasonCode.AB_OUT_OF_PLAUSIBLE_RANGE,
                    QualitySeverity.REJECT,
                    f"Region {region.region_id!r} has |a*| or |b*| above the configured "
                    f"plausibility limit {config.ab_abs_max:g}.",
                    region.region_id,
                )
            )
        if config.gloss_plausible_max is not None and region.gloss > config.gloss_plausible_max:
            issues.append(
                _issue(
                    QualityReasonCode.GLOSS_OUT_OF_PLAUSIBLE_RANGE,
                    QualitySeverity.REJECT,
                    f"Region {region.region_id!r} gloss {region.gloss:g} exceeds the configured "
                    f"plausibility limit {config.gloss_plausible_max:g}.",
                    region.region_id,
                )
            )

        if region.n_replicates < config.min_replicates_recommended:
            issues.append(
                _issue(
                    QualityReasonCode.REPLICATES_BELOW_RECOMMENDED,
                    QualitySeverity.REVIEW,
                    f"Region {region.region_id!r} has {region.n_replicates} replicate(s); "
                    f"{config.min_replicates_recommended} are recommended.",
                    region.region_id,
                )
            )

        if region.n_replicates > 1 and region.replicate_spread is None:
            issues.append(
                _issue(
                    QualityReasonCode.REPLICATE_SPREAD_MISSING,
                    QualitySeverity.REVIEW,
                    f"Region {region.region_id!r} reports {region.n_replicates} replicates "
                    "but no replicate spread.",
                    region.region_id,
                )
            )
            continue

        if region.replicate_spread is None:
            continue

        warn = config.replicate_std_warn
        fail = config.replicate_std_fail
        for quantity in ("L", "a", "b", "gloss"):
            spread = getattr(region.replicate_spread, quantity)
            if spread > getattr(fail, quantity):
                issues.append(
                    _issue(
                        QualityReasonCode.REPLICATE_SPREAD_FAIL,
                        QualitySeverity.REJECT,
                        f"Region {region.region_id!r} replicate spread for {quantity}={spread:g} "
                        f"exceeds fail limit {getattr(fail, quantity):g}.",
                        region.region_id,
                    )
                )
            elif spread > getattr(warn, quantity):
                issues.append(
                    _issue(
                        QualityReasonCode.REPLICATE_SPREAD_WARN,
                        QualitySeverity.REVIEW,
                        f"Region {region.region_id!r} replicate spread for {quantity}={spread:g} "
                        f"exceeds warning limit {getattr(warn, quantity):g}.",
                        region.region_id,
                    )
                )
    return issues


def _gloss_context_issues(master: MasterMeasurement, batch: BatchMeasurement) -> list[QualityIssue]:
    """Keep gloss comparability separate from colourimetric metadata policy."""
    master_angle = master.metadata.gloss_angle_deg
    batch_angle = batch.metadata.gloss_angle_deg
    if master_angle == batch_angle:
        return []
    return [
        _issue(
            QualityReasonCode.GLOSS_CONTEXT_MISMATCH,
            QualitySeverity.REVIEW,
            "Gloss measurement context differs: "
            f"master angle={master_angle!r}, batch angle={batch_angle!r}; "
            "gloss comparison requires matching measurement context.",
        )
    ]


def assess_measurement_quality(
    master: MasterMeasurement,
    batch: BatchMeasurement,
    config: MeasurementQualityConfig,
) -> MeasurementQualityResult:
    """Assess whether ``batch`` is fit for downstream colour comparison.

    Checks are deterministic and evaluated in structural, metadata, measurement,
    and gloss-context order. ``REJECT`` dominates ``REVIEW``, which dominates
    ``ACCEPT``. Client colour/gloss acceptance tolerances are intentionally not used.
    """
    issues = [
        *_structural_issues(master, batch),
        *_metadata_issues(master, batch, config),
        *_measurement_issues(batch, config),
        *_gloss_context_issues(master, batch),
    ]
    if any(issue.severity is QualitySeverity.REJECT for issue in issues):
        outcome = QualityOutcome.REJECT
    elif issues:
        outcome = QualityOutcome.REVIEW
    else:
        outcome = QualityOutcome.ACCEPT
    return MeasurementQualityResult(outcome=outcome, issues=tuple(issues))
