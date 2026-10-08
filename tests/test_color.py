"""Colour difference engine tests (synthetic values only).

Comparison tests use the real Chunk 1 domain models via the shared factories in
``conftest.py``; nothing here mocks them.
"""

import math

import pytest
from pydantic import ValidationError

from ats_ceramic.color import (
    ComparisonResult,
    RegionDifference,
    compare_measurements,
    delta_e_2000,
)
from ats_ceramic.schemas import BatchMeasurement, MasterMeasurement

# Published CIEDE2000 test data: Sharma, Wu & Dalal (2005), Table 1 (34 pairs).
# Each row: (L1, a1, b1), (L2, a2, b2), expected dE00 (4 decimals as published).
SHARMA_PAIRS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0010), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0011), 7.2195),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0012), 7.2195),
    ((50.0, -0.001, 2.49), (50.0, 0.0009, -2.49), 4.8045),
    ((50.0, -0.001, 2.49), (50.0, 0.0010, -2.49), 4.8045),
    ((50.0, -0.001, 2.49), (50.0, 0.0011, -2.49), 4.7461),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2972, 0.0), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 1.8634, 0.5757), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]

# Reference pairs reused to build regions with known colour differences.
PAIR_2_0425 = ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485))
PAIR_27_1492 = ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0))


# ---- Helpers ----------------------------------------------------------------


@pytest.fixture
def make_master(make_region, make_metadata):
    """Factory: MasterMeasurement from ``{region_id: (L, a, b, gloss)}``."""

    def _make(values: dict[str, tuple[float, float, float, float]]) -> MasterMeasurement:
        regions = tuple(
            make_region(region_id, L=L, a=a, b=b, gloss=gloss)
            for region_id, (L, a, b, gloss) in values.items()
        )
        return MasterMeasurement(sku_id="SKU_SYN_01", regions=regions, metadata=make_metadata())

    return _make


@pytest.fixture
def make_batch_from(make_batch, make_region):
    """Factory: BatchMeasurement from ``{region_id: (L, a, b, gloss)}``."""

    def _make(values: dict[str, tuple[float, float, float, float]]) -> BatchMeasurement:
        regions = tuple(
            make_region(region_id, L=L, a=a, b=b, gloss=gloss)
            for region_id, (L, a, b, gloss) in values.items()
        )
        return make_batch(regions=regions)

    return _make


def lab_gloss(lab: tuple[float, float, float], gloss: float = 60.0):
    return (*lab, gloss)


@pytest.fixture
def three_region_pair(make_master, make_batch_from):
    """R1 -> dE00 ~ 2.0425, R2 identical (0), R3 -> dE00 ~ 27.1492."""
    master = make_master(
        {
            "R1": lab_gloss(PAIR_2_0425[0]),
            "R2": lab_gloss((40.0, 10.0, 10.0)),
            "R3": lab_gloss(PAIR_27_1492[0]),
        }
    )
    batch = make_batch_from(
        {
            "R1": lab_gloss(PAIR_2_0425[1]),
            "R2": lab_gloss((40.0, 10.0, 10.0)),
            "R3": lab_gloss(PAIR_27_1492[1]),
        }
    )
    return master, batch


# ---- CIEDE2000 --------------------------------------------------------------


@pytest.mark.parametrize(("lab1", "lab2", "expected"), SHARMA_PAIRS)
def test_ciede2000_matches_published_reference_pairs(lab1, lab2, expected):
    assert delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=6e-5)


@pytest.mark.parametrize(("lab1", "lab2", "expected"), SHARMA_PAIRS)
def test_ciede2000_is_symmetric(lab1, lab2, expected):
    assert delta_e_2000(lab1, lab2) == pytest.approx(delta_e_2000(lab2, lab1), abs=1e-12)


def test_ciede2000_has_the_expected_number_of_reference_pairs():
    assert len(SHARMA_PAIRS) == 34


@pytest.mark.parametrize(
    "lab",
    [(50.0, 2.6772, -79.7751), (0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (72.5, -33.0, 41.0)],
)
def test_identical_lab_gives_zero(lab):
    assert delta_e_2000(lab, lab) == 0.0


# ---- Region differences / sign convention ----------------------------------


def test_delta_is_batch_minus_master(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 10.0, -5.0, 60.0)})
    batch = make_batch_from({"R1": (52.0, 8.0, -2.0, 55.0)})

    diff = compare_measurements(master, batch).region("R1")

    assert diff.delta_L == pytest.approx(2.0)
    assert diff.delta_a == pytest.approx(-2.0)
    assert diff.delta_b == pytest.approx(3.0)
    assert diff.delta_gloss == pytest.approx(-5.0)
    assert diff.delta_E00 == pytest.approx(
        delta_e_2000((50.0, 10.0, -5.0), (52.0, 8.0, -2.0)), abs=1e-12
    )


def test_swapping_roles_flips_signed_deltas_but_not_de00(make_master, make_batch_from):
    values_a = {"R1": (50.0, 10.0, -5.0, 60.0)}
    values_b = {"R1": (52.0, 8.0, -2.0, 55.0)}

    forward = compare_measurements(make_master(values_a), make_batch_from(values_b)).region("R1")
    reverse = compare_measurements(make_master(values_b), make_batch_from(values_a)).region("R1")

    assert reverse.delta_L == pytest.approx(-forward.delta_L)
    assert reverse.delta_a == pytest.approx(-forward.delta_a)
    assert reverse.delta_b == pytest.approx(-forward.delta_b)
    assert reverse.delta_gloss == pytest.approx(-forward.delta_gloss)
    assert reverse.delta_E00 == pytest.approx(forward.delta_E00, abs=1e-12)


def test_gloss_difference_is_separate_from_de00(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 10.0, -5.0, 60.0)})
    batch = make_batch_from({"R1": (50.0, 10.0, -5.0, 85.0)})  # same colour, glossier

    result = compare_measurements(master, batch)
    diff = result.region("R1")

    assert diff.delta_E00 == 0.0
    assert diff.delta_gloss == pytest.approx(25.0)
    assert result.max_delta_E00 == 0.0
    assert result.max_abs_delta_gloss == pytest.approx(25.0)


def test_colour_difference_does_not_depend_on_gloss(make_master, make_batch_from):
    colours = {"R1": (50.0, 10.0, -5.0), "R2": (40.0, 0.0, 0.0)}
    batch_colours = {"R1": (52.0, 8.0, -2.0), "R2": (41.0, 1.0, 1.0)}
    master_lo = make_master({k: (*v, 10.0) for k, v in colours.items()})
    master_hi = make_master({k: (*v, 90.0) for k, v in colours.items()})
    batch = make_batch_from({k: (*v, 50.0) for k, v in batch_colours.items()})

    lo = compare_measurements(master_lo, batch)
    hi = compare_measurements(master_hi, batch)

    assert [d.delta_E00 for d in lo.regions] == [d.delta_E00 for d in hi.regions]


# ---- Multi-region and aggregation ------------------------------------------


def test_multi_region_comparison_preserves_master_order(three_region_pair):
    master, batch = three_region_pair

    result = compare_measurements(master, batch)

    assert isinstance(result, ComparisonResult)
    assert [d.region_id for d in result.regions] == ["R1", "R2", "R3"]
    assert result.region("R1").delta_E00 == pytest.approx(2.0425, abs=6e-5)
    assert result.region("R2").delta_E00 == 0.0
    assert result.region("R3").delta_E00 == pytest.approx(27.1492, abs=6e-5)


def test_batch_region_order_does_not_change_result(make_master, make_batch_from):
    master_values = {
        "R1": (50.0, 10.0, -5.0, 60.0),
        "R2": (40.0, 0.0, 0.0, 50.0),
        "R3": (70.0, -10.0, 20.0, 40.0),
    }
    batch_values = {
        "R1": (52.0, 8.0, -2.0, 55.0),
        "R2": (41.0, 1.0, 1.0, 52.0),
        "R3": (68.0, -9.0, 22.0, 41.0),
    }
    master = make_master(master_values)

    in_order = compare_measurements(master, make_batch_from(batch_values))
    shuffled = compare_measurements(master, make_batch_from(dict(reversed(batch_values.items()))))

    assert shuffled == in_order
    assert [d.region_id for d in shuffled.regions] == ["R1", "R2", "R3"]


def test_comparison_is_deterministic(three_region_pair):
    master, batch = three_region_pair
    assert compare_measurements(master, batch) == compare_measurements(master, batch)


def test_mean_de00(three_region_pair):
    master, batch = three_region_pair

    result = compare_measurements(master, batch)

    assert result.mean_delta_E00 == pytest.approx((2.0425 + 0.0 + 27.1492) / 3, abs=1e-4)


def test_max_de00_and_worst_region(three_region_pair):
    master, batch = three_region_pair

    result = compare_measurements(master, batch)

    assert result.worst_region_id == "R3"
    assert result.max_delta_E00 == pytest.approx(27.1492, abs=6e-5)


def test_worst_region_tie_resolves_to_first_in_master_order(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 0.0, 0.0, 60.0), "R2": (50.0, 0.0, 0.0, 60.0)})
    batch = make_batch_from({"R1": (55.0, 0.0, 0.0, 60.0), "R2": (55.0, 0.0, 0.0, 60.0)})

    assert compare_measurements(master, batch).worst_region_id == "R1"


def test_gloss_aggregates(make_master, make_batch_from):
    master = make_master(
        {
            "R1": (50.0, 0.0, 0.0, 60.0),
            "R2": (50.0, 0.0, 0.0, 60.0),
            "R3": (50.0, 0.0, 0.0, 60.0),
        }
    )
    batch = make_batch_from(
        {
            "R1": (50.0, 0.0, 0.0, 66.0),  # +6
            "R2": (50.0, 0.0, 0.0, 58.0),  # -2
            "R3": (50.0, 0.0, 0.0, 60.0),  # 0
        }
    )

    result = compare_measurements(master, batch)

    assert result.mean_abs_delta_gloss == pytest.approx(8.0 / 3.0)
    assert result.max_abs_delta_gloss == pytest.approx(6.0)


def test_single_region_comparison(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 10.0, -5.0, 60.0)})
    batch = make_batch_from({"R1": (52.0, 8.0, -2.0, 55.0)})

    result = compare_measurements(master, batch)

    assert result.mean_delta_E00 == result.max_delta_E00 == result.regions[0].delta_E00
    assert result.worst_region_id == "R1"


# ---- Weighted mean ----------------------------------------------------------


def test_weighted_mean_de00(three_region_pair):
    master, batch = three_region_pair

    result = compare_measurements(master, batch, region_weights={"R1": 1.0, "R2": 1.0, "R3": 2.0})

    assert result.mean_delta_E00 == pytest.approx((2.0425 + 0.0 + 2 * 27.1492) / 4, abs=1e-4)


def test_equal_weights_match_unweighted_mean(three_region_pair):
    master, batch = three_region_pair

    weighted = compare_measurements(master, batch, region_weights={"R1": 3.0, "R2": 3.0, "R3": 3.0})
    plain = compare_measurements(master, batch)

    assert weighted.mean_delta_E00 == pytest.approx(plain.mean_delta_E00, abs=1e-12)


def test_zero_weight_region_is_ignored_in_mean_but_not_in_max(three_region_pair):
    master, batch = three_region_pair

    result = compare_measurements(master, batch, region_weights={"R1": 1.0, "R2": 1.0, "R3": 0.0})

    assert result.mean_delta_E00 == pytest.approx(2.0425 / 2, abs=1e-4)
    assert result.worst_region_id == "R3"


def test_weights_do_not_affect_max_or_gloss_aggregates(three_region_pair):
    master, batch = three_region_pair

    weighted = compare_measurements(master, batch, region_weights={"R1": 5.0, "R2": 1.0, "R3": 1.0})
    plain = compare_measurements(master, batch)

    assert weighted.max_delta_E00 == plain.max_delta_E00
    assert weighted.worst_region_id == plain.worst_region_id
    assert weighted.mean_abs_delta_gloss == plain.mean_abs_delta_gloss
    assert weighted.max_abs_delta_gloss == plain.max_abs_delta_gloss


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"R1": -1.0, "R2": 1.0, "R3": 1.0}, "must be >= 0"),
        ({"R1": math.nan, "R2": 1.0, "R3": 1.0}, "must be finite"),
        ({"R1": math.inf, "R2": 1.0, "R3": 1.0}, "must be finite"),
        ({"R1": -math.inf, "R2": 1.0, "R3": 1.0}, "must be finite"),
        ({"R1": 0.0, "R2": 0.0, "R3": 0.0}, "total region weight must be positive"),
        ({"R1": 1.0, "R2": 1.0}, "missing \\['R3'\\]"),
        ({"R1": 1.0, "R2": 1.0, "R3": 1.0, "R9": 1.0}, "unexpected \\['R9'\\]"),
    ],
)
def test_invalid_weights_are_rejected(three_region_pair, weights, message):
    master, batch = three_region_pair

    with pytest.raises(ValueError, match=message):
        compare_measurements(master, batch, region_weights=weights)


# ---- Structural compatibility ----------------------------------------------


def test_missing_region_in_batch_fails(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 0.0, 0.0, 60.0), "R2": (40.0, 0.0, 0.0, 60.0)})
    batch = make_batch_from({"R1": (50.0, 0.0, 0.0, 60.0)})

    with pytest.raises(ValueError, match="missing from batch \\['R2'\\]"):
        compare_measurements(master, batch)


def test_extra_region_in_batch_fails(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 0.0, 0.0, 60.0)})
    batch = make_batch_from({"R1": (50.0, 0.0, 0.0, 60.0), "R2": (40.0, 0.0, 0.0, 60.0)})

    with pytest.raises(ValueError, match="unexpected in batch \\['R2'\\]"):
        compare_measurements(master, batch)


def test_different_region_ids_report_both_missing_and_unexpected(make_master, make_batch_from):
    master = make_master({"R1": (50.0, 0.0, 0.0, 60.0)})
    batch = make_batch_from({"R9": (50.0, 0.0, 0.0, 60.0)})

    with pytest.raises(ValueError, match="missing from batch \\['R1'\\].*unexpected.*\\['R9'\\]"):
        compare_measurements(master, batch)


def test_schemas_already_reject_duplicate_regions_at_construction(make_region, make_metadata):
    with pytest.raises(ValidationError, match="duplicate region_id"):
        MasterMeasurement(
            sku_id="SKU_SYN_01",
            regions=(make_region("R1"), make_region("R1")),
            metadata=make_metadata(),
        )


def test_duplicate_batch_region_that_bypasses_schema_validation_fails(
    make_master, make_batch_from, make_region
):
    """Defence in depth: ``model_construct`` skips validation, so build a bad batch with it."""
    master = make_master({"R1": (50.0, 0.0, 0.0, 60.0), "R2": (40.0, 0.0, 0.0, 60.0)})
    good = make_batch_from({"R1": (50.0, 0.0, 0.0, 60.0), "R2": (40.0, 0.0, 0.0, 60.0)})
    bad = BatchMeasurement.model_construct(
        batch_id=good.batch_id,
        sku_id=good.sku_id,
        context_id=good.context_id,
        iteration=good.iteration,
        applied_correction_id=good.applied_correction_id,
        regions=(make_region("R1"), make_region("R1")),
        metadata=good.metadata,
    )

    with pytest.raises(ValueError, match="duplicate batch region_id"):
        compare_measurements(master, bad)


def test_duplicate_master_region_that_bypasses_schema_validation_fails(
    make_batch_from, make_region, make_metadata
):
    batch = make_batch_from({"R1": (50.0, 0.0, 0.0, 60.0)})
    bad = MasterMeasurement.model_construct(
        sku_id="SKU_SYN_01",
        regions=(make_region("R1"), make_region("R1")),
        metadata=make_metadata(),
    )

    with pytest.raises(ValueError, match="duplicate master region_id"):
        compare_measurements(bad, batch)


# ---- Immutability and lookup -----------------------------------------------


def test_region_difference_is_frozen(three_region_pair):
    master, batch = three_region_pair
    diff = compare_measurements(master, batch).region("R1")

    with pytest.raises(ValidationError):
        diff.delta_E00 = 0.0
    with pytest.raises(ValidationError):
        diff.region_id = "other"


def test_comparison_result_is_frozen(three_region_pair):
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    with pytest.raises(ValidationError):
        result.mean_delta_E00 = 0.0
    with pytest.raises(ValidationError):
        result.regions = ()


def test_comparison_result_holds_no_mutable_containers(three_region_pair):
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    assert type(result.regions) is tuple
    assert all(type(d) is RegionDifference for d in result.regions)
    for model in (result, *result.regions):
        assert type(model).model_config.get("frozen") is True
        for name in type(model).model_fields:
            assert not isinstance(getattr(model, name), (list, dict, set, bytearray))


def test_comparison_result_has_no_hidden_instance_state(three_region_pair):
    """No cached lookup mapping or other attribute may sit outside the declared fields."""
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    assert set(vars(result)) == set(ComparisonResult.model_fields)
    assert not result.model_extra


def test_comparison_result_rejects_unknown_fields(three_region_pair):
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    with pytest.raises(ValidationError):
        ComparisonResult(**result.model_dump(), unexpected=1)


def test_region_lookup(three_region_pair):
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    assert result.region("R3") is result.regions[2]
    assert result.region("R2").delta_E00 == 0.0


def test_region_lookup_unknown_id_raises_key_error(three_region_pair):
    master, batch = three_region_pair
    result = compare_measurements(master, batch)

    with pytest.raises(KeyError, match="R9"):
        result.region("R9")


def test_comparison_result_rejects_inconsistent_construction():
    diff = RegionDifference(
        region_id="R1", delta_L=0.0, delta_a=0.0, delta_b=0.0, delta_E00=0.0, delta_gloss=0.0
    )
    base = {
        "mean_delta_E00": 0.0,
        "max_delta_E00": 0.0,
        "mean_abs_delta_gloss": 0.0,
        "max_abs_delta_gloss": 0.0,
    }

    with pytest.raises(ValidationError, match="not a compared region"):
        ComparisonResult(regions=(diff,), worst_region_id="R2", **base)
    with pytest.raises(ValidationError, match="duplicate region_id"):
        ComparisonResult(regions=(diff, diff), worst_region_id="R1", **base)
    with pytest.raises(ValidationError, match="at least one region"):
        ComparisonResult(regions=(), worst_region_id="R1", **base)
