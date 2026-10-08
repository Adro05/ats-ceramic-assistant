"""Shared test factories.

All values are arbitrary SYNTHETIC test values. In particular the instrument, geometry,
illuminant, observer, SCI/SCE mode and aperture below are NOT statements about the
client's instrument configuration, which is a discovery dependency.
"""

from datetime import UTC, datetime

import pytest

from ats_ceramic.schemas import BatchMeasurement, MeasurementMetadata, RegionMeasurement


@pytest.fixture
def make_metadata():
    """Factory for a valid MeasurementMetadata; keyword overrides replace defaults."""

    def _make(**overrides) -> MeasurementMetadata:
        values = {
            "instrument_id": "SPEC_SYN_1",
            "geometry": "d/8",
            "illuminant": "D65",
            "observer": "2deg",
            "specular_mode": "SCI",
            "aperture_mm": 8.0,
            "measured_at": datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            "data_origin": "synthetic",
        }
        values.update(overrides)
        return MeasurementMetadata(**values)

    return _make


@pytest.fixture
def make_region():
    """Factory for a valid RegionMeasurement; keyword overrides replace defaults."""

    def _make(region_id: str = "R1", **overrides) -> RegionMeasurement:
        values = {"region_id": region_id, "L": 55.0, "a": 12.0, "b": -4.0, "gloss": 60.0}
        values.update(overrides)
        return RegionMeasurement(**values)

    return _make


@pytest.fixture
def make_batch(make_metadata, make_region):
    """Factory for a valid BatchMeasurement; keyword overrides replace defaults."""

    def _make(**overrides) -> BatchMeasurement:
        values = {
            "batch_id": "B001",
            "sku_id": "SKU_SYN_01",
            "context_id": "ctx_syn_1",
            "iteration": 0,
            "applied_correction_id": None,
            "regions": (make_region("R1"), make_region("R2", L=40.0)),
            "metadata": make_metadata(),
        }
        values.update(overrides)
        return BatchMeasurement(**values)

    return _make