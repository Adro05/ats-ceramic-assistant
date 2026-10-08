"""Schema validation tests (synthetic values only)."""

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ats_ceramic.schemas import (
    BatchMeasurement,
    ChannelAdjustment,
    Correction,
    CorrectionSource,
    MasterMeasurement,
    ReplicateSpread,
    VerificationOutcome,
    VerificationResult,
)

HASH = "a" * 64
NOW = datetime(2026, 1, 2, 9, 0, tzinfo=UTC)


def make_correction(**overrides) -> Correction:
    values = {
        "correction_id": "C001",
        "batch_id": "B001",
        "sku_id": "SKU_SYN_01",
        "context_id": "ctx_syn_1",
        "master_id": "M001",
        "master_content_hash": HASH,
        "based_on_iteration": 0,
        "adjustments": (
            ChannelAdjustment(channel_id="ch_1", delta=0.02),
            ChannelAdjustment(channel_id="ch_2", delta=-0.01),
        ),
        "source": CorrectionSource.RECOMMENDED,
        "created_at": NOW,
        "data_origin": "synthetic",
    }
    values.update(overrides)
    return Correction(**values)


# ---- RegionMeasurement ------------------------------------------------------


@pytest.mark.parametrize("bad_L", [-0.01, 100.01, math.nan, math.inf, -math.inf])
def test_lightness_must_be_finite_and_within_0_100(make_region, bad_L):
    with pytest.raises(ValidationError):
        make_region(L=bad_L)


@pytest.mark.parametrize("L", [0.0, 100.0])
def test_lightness_bounds_are_inclusive(make_region, L):
    assert make_region(L=L).L == L


def test_gloss_cannot_be_negative(make_region):
    with pytest.raises(ValidationError):
        make_region(gloss=-0.1)


@pytest.mark.parametrize("field", ["a", "b", "gloss"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_values_rejected(make_region, field, bad):
    with pytest.raises(ValidationError):
        make_region(**{field: bad})


def test_unknown_field_rejected(make_region):
    with pytest.raises(ValidationError):
        make_region(L_star=50.0)


def test_region_is_immutable(make_region):
    region = make_region()
    with pytest.raises(ValidationError):
        region.L = 10.0


def test_replicate_spread_requires_multiple_replicates(make_region):
    spread = ReplicateSpread(L=0.1, a=0.1, b=0.1, gloss=0.3)
    with pytest.raises(ValidationError):
        make_region(n_replicates=1, replicate_spread=spread)
    assert make_region(n_replicates=3, replicate_spread=spread).replicate_spread == spread


def test_empty_region_id_rejected(make_region):
    with pytest.raises(ValidationError):
        make_region(region_id="   ")


# ---- MeasurementMetadata ----------------------------------------------------


def test_naive_datetime_rejected(make_metadata):
    with pytest.raises(ValidationError):
        make_metadata(measured_at=datetime(2026, 1, 1, 8, 0))


def test_illuminant_and_geometry_are_normalised(make_metadata):
    metadata = make_metadata(illuminant=" d65 ", geometry="D/8")
    assert metadata.illuminant == "D65"
    assert metadata.geometry == "d/8"


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_aperture_must_be_positive_and_finite(make_metadata, bad):
    with pytest.raises(ValidationError):
        make_metadata(aperture_mm=bad)


def test_invalid_specular_mode_rejected(make_metadata):
    with pytest.raises(ValidationError):
        make_metadata(specular_mode="XYZ")


def test_data_origin_is_required(make_metadata):
    with pytest.raises(ValidationError):
        make_metadata(data_origin=None)


# ---- Master / Batch ----------------------------------------------------------


def test_master_rejects_duplicate_regions(make_region, make_metadata):
    with pytest.raises(ValidationError, match="duplicate region_id"):
        MasterMeasurement(
            sku_id="SKU_SYN_01",
            regions=(make_region("R1"), make_region("R1")),
            metadata=make_metadata(),
        )


def test_master_requires_at_least_one_region(make_metadata):
    with pytest.raises(ValidationError):
        MasterMeasurement(sku_id="SKU_SYN_01", regions=(), metadata=make_metadata())


def test_batch_rejects_duplicate_regions(make_region, make_batch):
    with pytest.raises(ValidationError, match="duplicate region_id"):
        make_batch(regions=(make_region("R1"), make_region("R1")))


def test_iteration_zero_cannot_reference_correction(make_batch):
    with pytest.raises(ValidationError):
        make_batch(iteration=0, applied_correction_id="C001")


def test_later_iterations_require_correction_reference(make_batch):
    with pytest.raises(ValidationError):
        make_batch(iteration=1, applied_correction_id=None)
    assert make_batch(iteration=1, applied_correction_id="C001").iteration == 1


def test_batch_json_round_trip(make_batch):
    batch = make_batch()
    assert BatchMeasurement.model_validate_json(batch.model_dump_json()) == batch


# ---- Correction --------------------------------------------------------------


def test_correction_rejects_duplicate_channels():
    with pytest.raises(ValidationError, match="duplicate channel_id"):
        make_correction(
            adjustments=(
                ChannelAdjustment(channel_id="ch_1", delta=0.01),
                ChannelAdjustment(channel_id="ch_1", delta=0.02),
            )
        )


def test_correction_requires_at_least_one_adjustment():
    with pytest.raises(ValidationError):
        make_correction(adjustments=())


@pytest.mark.parametrize("bad_hash", ["abc", "G" * 64, "A" * 64])
def test_correction_requires_lowercase_sha256_hash(bad_hash):
    with pytest.raises(ValidationError):
        make_correction(master_content_hash=bad_hash)


def test_adjustment_delta_must_be_finite():
    with pytest.raises(ValidationError):
        ChannelAdjustment(channel_id="ch_1", delta=math.nan)


# ---- VerificationResult ------------------------------------------------------


def test_verification_result_consistency(make_batch):
    measurement = make_batch(iteration=1, applied_correction_id="C001")
    result = VerificationResult(
        batch_id="B001",
        iteration=1,
        correction_id="C001",
        measurement=measurement,
        outcome=VerificationOutcome.ACCEPTED,
        worst_delta_e00=0.7,
    )
    assert result.outcome is VerificationOutcome.ACCEPTED


@pytest.mark.parametrize(
    "overrides",
    [{"batch_id": "B999"}, {"iteration": 2}, {"correction_id": "C999"}],
)
def test_verification_result_detects_mismatched_measurement(make_batch, overrides):
    measurement = make_batch(iteration=1, applied_correction_id="C001")
    values = {
        "batch_id": "B001",
        "iteration": 1,
        "correction_id": "C001",
        "measurement": measurement,
        "outcome": VerificationOutcome.ITERATE,
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        VerificationResult(**values)