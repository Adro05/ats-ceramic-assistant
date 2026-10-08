"""Guards against mutable containers hiding inside frozen models.

A frozen pydantic model blocks field reassignment but NOT mutation of a ``dict`` or
``list`` stored in a field. These tests walk every field of the domain and config
models recursively and fail if any mutable container is found, so the "frozen"
guarantee cannot silently erode when new fields are added.
"""

from datetime import UTC, datetime
from enum import Enum

import pytest
from pydantic import BaseModel

from ats_ceramic.config import load_channel_config, load_synthetic_config, load_tolerance_config
from ats_ceramic.schemas import (
    ChannelAdjustment,
    Correction,
    CorrectionSource,
    FrozenModel,
    MasterMeasurement,
    VerificationOutcome,
    VerificationResult,
)

_MUTABLE_CONTAINERS = (list, dict, set, bytearray)
_IMMUTABLE_LEAVES = (str, int, float, bool, type(None), datetime, Enum)


def assert_deeply_immutable(obj: object, path: str = "root") -> None:
    """Fail if ``obj`` contains a mutable container or a non-frozen model.

    Allowed: frozen pydantic models, tuples, and simple scalar leaves.
    """
    if isinstance(obj, _MUTABLE_CONTAINERS):
        raise AssertionError(f"mutable {type(obj).__name__} at {path}")
    if isinstance(obj, BaseModel):
        if type(obj).model_config.get("frozen") is not True:
            raise AssertionError(f"{type(obj).__name__} at {path} is not frozen")
        for name in type(obj).model_fields:
            assert_deeply_immutable(getattr(obj, name), f"{path}.{name}")
    elif isinstance(obj, tuple):
        for index, item in enumerate(obj):
            assert_deeply_immutable(item, f"{path}[{index}]")
    elif not isinstance(obj, _IMMUTABLE_LEAVES):
        raise AssertionError(f"unexpected type {type(obj).__name__} at {path}")


# ---- The walker itself must not be vacuous -----------------------------------


def test_walker_detects_a_dict_field_in_a_frozen_model():
    class Leaky(FrozenModel):
        values: dict[str, int]

    with pytest.raises(AssertionError, match="mutable dict"):
        assert_deeply_immutable(Leaky(values={"a": 1}))


def test_walker_detects_a_list_field_in_a_frozen_model():
    class Leaky(FrozenModel):
        values: list[int]

    with pytest.raises(AssertionError, match="mutable list"):
        assert_deeply_immutable(Leaky(values=[1]))


def test_walker_detects_non_frozen_model():
    class Loose(BaseModel):
        x: int

    with pytest.raises(AssertionError, match="not frozen"):
        assert_deeply_immutable(Loose(x=1))


# ---- Config models -----------------------------------------------------------


@pytest.mark.parametrize(
    "loader", [load_tolerance_config, load_channel_config, load_synthetic_config]
)
def test_shipped_configs_are_deeply_immutable(loader):
    assert_deeply_immutable(loader())


# ---- Domain models -----------------------------------------------------------


def test_domain_objects_are_deeply_immutable(make_batch, make_region, make_metadata):
    batch = make_batch()
    master = MasterMeasurement(
        sku_id="SKU_SYN_01",
        regions=(make_region("R1"), make_region("R2")),
        metadata=make_metadata(),
    )
    correction = Correction(
        correction_id="C001",
        batch_id="B001",
        sku_id="SKU_SYN_01",
        context_id="ctx_syn_1",
        master_id="M001",
        master_content_hash="a" * 64,
        based_on_iteration=0,
        adjustments=(ChannelAdjustment(channel_id="ch_1", delta=0.01),),
        source=CorrectionSource.RECOMMENDED,
        created_at=datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
        data_origin="synthetic",
    )
    verification = VerificationResult(
        batch_id="B001",
        iteration=1,
        correction_id="C001",
        measurement=make_batch(iteration=1, applied_correction_id="C001"),
        outcome=VerificationOutcome.ACCEPTED,
    )
    for obj in (batch, master, correction, verification):
        assert_deeply_immutable(obj)


def test_collections_validated_from_lists_become_tuples(make_region, make_metadata):
    """Callers often pass lists (JSON, YAML); the stored value must still be a tuple."""
    master = MasterMeasurement(
        sku_id="SKU_SYN_01",
        regions=[make_region("R1")],  # list in, tuple out
        metadata=make_metadata(),
    )
    assert isinstance(master.regions, tuple)