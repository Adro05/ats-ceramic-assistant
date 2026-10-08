"""Deterministic colour comparison of a batch measurement against its master.

Pipeline position::

    validated domain objects -> colour difference engine -> (triage, optimiser, ...)

This module assumes it receives VALIDATED domain objects. Measurement quality
(instrument calibration, physical ranges, metadata mismatch policy, ...) is the
quality gate's responsibility and is deliberately not repeated here. The only checks
made here are the structural ones comparison itself needs: master and batch must
describe exactly the same regions, each exactly once.

Sign convention, used everywhere::

    delta = Batch - Master

so a positive ``delta_L`` means the batch is lighter than the master.

Gloss is kept mathematically separate from colour: ``delta_E00`` uses only L*, a*, b*;
``delta_gloss`` is a plain difference in the instrument's native gloss scale. No
acceptance threshold is defined here; tolerances belong to a later stage.
"""

import math
from collections.abc import Mapping
from typing import Self

from pydantic import FiniteFloat, model_validator

from ats_ceramic.schemas import (
    BatchMeasurement,
    FrozenModel,
    Identifier,
    MasterMeasurement,
    NonNegativeFinite,
    RegionMeasurement,
    require_unique,
)

LabTriple = tuple[float, float, float]
"""CIELAB ``(L*, a*, b*)``."""

_POW25_7 = 25.0**7


# --------------------------------------------------------------------------- #
# CIEDE2000
# --------------------------------------------------------------------------- #


def _hue_angle_deg(b: float, a_prime: float) -> float:
    """Hue angle in degrees in [0, 360); defined as 0 when both components are 0."""
    if b == 0.0 and a_prime == 0.0:
        return 0.0
    return math.degrees(math.atan2(b, a_prime)) % 360.0


def delta_e_2000(lab1: LabTriple, lab2: LabTriple) -> float:
    """CIEDE2000 colour difference with ``kL = kC = kH = 1``.

    Implements the formulation of Sharma, Wu & Dalal (2005), "The CIEDE2000
    Color-Difference Formula: Implementation Notes, Supplementary Test Data, and
    Mathematical Observations", including the hue-angle wraparound rules. The result
    is symmetric in its arguments.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: chroma-dependent rescaling of a*.
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar_7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar_7 / (C_bar_7 + _POW25_7)))
    a1_p = (1.0 + G) * a1
    a2_p = (1.0 + G) * a2
    C1_p = math.hypot(a1_p, b1)
    C2_p = math.hypot(a2_p, b2)
    h1_p = _hue_angle_deg(b1, a1_p)
    h2_p = _hue_angle_deg(b2, a2_p)

    # Step 2: differences, with hue wraparound.
    delta_L_p = L2 - L1
    delta_C_p = C2_p - C1_p
    chroma_product = C1_p * C2_p
    if chroma_product == 0.0:
        delta_h_p = 0.0
    else:
        delta_h_p = h2_p - h1_p
        if delta_h_p > 180.0:
            delta_h_p -= 360.0
        elif delta_h_p < -180.0:
            delta_h_p += 360.0
    delta_H_p = 2.0 * math.sqrt(chroma_product) * math.sin(math.radians(delta_h_p / 2.0))

    # Step 3: means, with hue wraparound.
    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1_p + C2_p) / 2.0
    if chroma_product == 0.0:
        h_bar_p = h1_p + h2_p
    elif abs(h1_p - h2_p) <= 180.0:
        h_bar_p = (h1_p + h2_p) / 2.0
    elif h1_p + h2_p < 360.0:
        h_bar_p = (h1_p + h2_p + 360.0) / 2.0
    else:
        h_bar_p = (h1_p + h2_p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )
    delta_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    C_bar_p_7 = C_bar_p**7
    R_C = 2.0 * math.sqrt(C_bar_p_7 / (C_bar_p_7 + _POW25_7))
    S_L = 1.0 + 0.015 * (L_bar_p - 50.0) ** 2 / math.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * delta_theta)) * R_C

    # Step 4: combine (kL = kC = kH = 1).
    lightness_term = delta_L_p / S_L
    chroma_term = delta_C_p / S_C
    hue_term = delta_H_p / S_H
    total = lightness_term**2 + chroma_term**2 + hue_term**2 + R_T * chroma_term * hue_term
    return math.sqrt(max(total, 0.0))


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #


class RegionDifference(FrozenModel):
    """Batch - Master difference for one region."""

    region_id: Identifier
    delta_L: FiniteFloat
    delta_a: FiniteFloat
    delta_b: FiniteFloat
    delta_E00: NonNegativeFinite
    delta_gloss: FiniteFloat


class ComparisonResult(FrozenModel):
    """Per-region differences plus aggregates for one batch-vs-master comparison.

    ``regions`` follows the order of the master measurement's regions.
    ``mean_delta_E00`` is the plain mean unless region weights were supplied to
    :func:`compare_measurements`, in which case it is the weighted mean. The gloss
    aggregates are always unweighted.
    """

    regions: tuple[RegionDifference, ...]
    mean_delta_E00: NonNegativeFinite
    max_delta_E00: NonNegativeFinite
    worst_region_id: Identifier
    mean_abs_delta_gloss: NonNegativeFinite
    max_abs_delta_gloss: NonNegativeFinite

    @model_validator(mode="after")
    def _check_regions(self) -> Self:
        if not self.regions:
            raise ValueError("a comparison needs at least one region")
        region_ids = tuple(r.region_id for r in self.regions)
        require_unique(region_ids, "region_id")
        if self.worst_region_id not in region_ids:
            raise ValueError(f"worst_region_id {self.worst_region_id!r} is not a compared region")
        return self

    def region(self, region_id: str) -> RegionDifference:
        """Return the difference for ``region_id``; raise ``KeyError`` if absent."""
        for difference in self.regions:
            if difference.region_id == region_id:
                return difference
        raise KeyError(f"no region {region_id!r} in comparison result")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def _check_region_correspondence(master: MasterMeasurement, batch: BatchMeasurement) -> None:
    """Require the same region IDs in master and batch, each exactly once."""
    master_ids = tuple(r.region_id for r in master.regions)
    batch_ids = tuple(r.region_id for r in batch.regions)
    require_unique(master_ids, "master region_id")
    require_unique(batch_ids, "batch region_id")
    missing = sorted(set(master_ids) - set(batch_ids))
    unexpected = sorted(set(batch_ids) - set(master_ids))
    if missing or unexpected:
        raise ValueError(
            "batch regions do not match master regions: "
            f"missing from batch {missing}; unexpected in batch {unexpected}"
        )


def _check_weights(weights: Mapping[str, float], region_ids: tuple[str, ...]) -> None:
    """Require one finite, non-negative weight per region and a positive total."""
    missing = sorted(set(region_ids) - set(weights))
    unexpected = sorted(set(weights) - set(region_ids))
    if missing or unexpected:
        raise ValueError(
            "region weights do not match compared regions: "
            f"missing {missing}; unexpected {unexpected}"
        )
    for region_id in region_ids:
        weight = weights[region_id]
        if not math.isfinite(weight):
            raise ValueError(f"weight for region {region_id!r} must be finite, got {weight}")
        if weight < 0.0:
            raise ValueError(f"weight for region {region_id!r} must be >= 0, got {weight}")
    if not math.fsum(weights[region_id] for region_id in region_ids) > 0.0:
        raise ValueError("total region weight must be positive")


def _region_difference(master: RegionMeasurement, batch: RegionMeasurement) -> RegionDifference:
    return RegionDifference(
        region_id=master.region_id,
        delta_L=batch.L - master.L,
        delta_a=batch.a - master.a,
        delta_b=batch.b - master.b,
        delta_E00=delta_e_2000((master.L, master.a, master.b), (batch.L, batch.a, batch.b)),
        delta_gloss=batch.gloss - master.gloss,
    )


def compare_measurements(
    master: MasterMeasurement,
    batch: BatchMeasurement,
    *,
    region_weights: Mapping[str, float] | None = None,
) -> ComparisonResult:
    """Compare ``batch`` with ``master`` region by region (delta = Batch - Master).

    Raises ``ValueError`` if the two measurements do not contain exactly the same
    region IDs, or if ``region_weights`` is given but does not hold exactly one finite,
    non-negative weight per region with a positive total. Differences are returned in
    the order of ``master.regions``.
    """
    _check_region_correspondence(master, batch)
    batch_by_id = {r.region_id: r for r in batch.regions}
    differences = tuple(_region_difference(m, batch_by_id[m.region_id]) for m in master.regions)

    de00 = tuple(d.delta_E00 for d in differences)
    if region_weights is None:
        mean_de00 = math.fsum(de00) / len(de00)
    else:
        _check_weights(region_weights, tuple(d.region_id for d in differences))
        weighted = math.fsum(region_weights[d.region_id] * d.delta_E00 for d in differences)
        total_weight = math.fsum(region_weights[d.region_id] for d in differences)
        mean_de00 = weighted / total_weight

    worst = differences[0]
    for difference in differences[1:]:
        if difference.delta_E00 > worst.delta_E00:
            worst = difference

    abs_gloss = tuple(abs(d.delta_gloss) for d in differences)
    return ComparisonResult(
        regions=differences,
        mean_delta_E00=mean_de00,
        max_delta_E00=worst.delta_E00,
        worst_region_id=worst.region_id,
        mean_abs_delta_gloss=math.fsum(abs_gloss) / len(abs_gloss),
        max_abs_delta_gloss=max(abs_gloss),
    )
