"""Deterministic synthetic data generator for development and testing.



The generator is an explicitly synthetic data source. Its hidden oracle is a small,

nonlinear, saturating production-response function; the observed records are what the

pipeline would receive, with noise and controlled corruption where requested.



This module is not a production simulator and does not implement a learned model.

"""



from __future__ import annotations

from enum import StrEnum
from math import tanh
from random import Random
from typing import Self

from pydantic import Field, model_validator

from ats_ceramic.schemas import (
    BatchMeasurement,
    CalibrationContext,
    ChannelAdjustment,
    DataOrigin,
    FrozenModel,
    MasterMeasurement,
    MeasurementMetadata,
    RegionMeasurement,
)


class SyntheticScenario(StrEnum):

    """Controlled synthetic evaluation scenarios."""



    CORRECTABLE = "correctable"

    NEAR_BOUNDARY = "near_boundary"

    PROCESS_SIDE_GLOSS_FAILURE = "process_side_gloss_failure"

    UNREACHABLE = "unreachable"

    UNSEEN_SKU_OOD = "unseen_sku_ood"

    NOISY_CORRUPTED_MEASUREMENT = "noisy_corrupted_measurement"

    SPARSE_CALIBRATION = "sparse_calibration"





class SyntheticConfig(FrozenModel):

    """Small deterministic configuration for the synthetic generator.



    Bounds and noise are synthetic development assumptions, not client requirements.

    """



    seed: int = 1

    sku_id: str = "SKU_SYN_01"

    context_id: str = "CTX_SYN_01"

    region_ids: tuple[str, ...] = ("R1", "R2")

    channel_ids: tuple[str, ...] = ("channel_1", "channel_2")

    calibration_observations: int = Field(default=8, ge=2)

    noise_level: float = Field(default=0.10, ge=0.0)

    channel_lower_bounds: tuple[float, ...] = (-2.0, -2.0)

    channel_upper_bounds: tuple[float, ...] = (2.0, 2.0)

    provenance_label: str = "synthetic_generator:v1"



    @model_validator(mode="after")

    def _validate_config(self) -> Self:

        if len(self.region_ids) < 1:

            raise ValueError("at least one synthetic region is required")

        if len(self.channel_ids) < 1:

            raise ValueError("at least one synthetic channel is required")

        if len(set(self.region_ids)) != len(self.region_ids):

            raise ValueError("synthetic region_ids must be unique")

        if len(set(self.channel_ids)) != len(self.channel_ids):

            raise ValueError("synthetic channel_ids must be unique")

        if len(self.channel_lower_bounds) != len(self.channel_ids):

            raise ValueError("channel_lower_bounds must match channel_ids")

        if len(self.channel_upper_bounds) != len(self.channel_ids):

            raise ValueError("channel_upper_bounds must match channel_ids")

        if any(

            lo >= hi

            for lo, hi in zip(

                self.channel_lower_bounds,

                self.channel_upper_bounds,

                strict=True,

            )

        ):

            raise ValueError(

                "each synthetic channel lower bound must be below its upper bound"

            )

        return self





class SyntheticProcessState(FrozenModel):

    """Hidden/evaluation process variables used by the synthetic oracle."""



    batch_process_offset: tuple[float, float, float]

    gloss_process_variable: float

    oracle_adjustment: tuple[float, ...]





class SyntheticObservation(FrozenModel):

    """Observed synthetic record plus explicitly separated evaluation metadata."""



    observation_id: str

    batch_id: str

    sku_id: str

    context_id: str

    region_id: str

    adjustments: tuple[ChannelAdjustment, ...]

    observed_L: float

    observed_a: float

    observed_b: float

    observed_gloss: float

    process_state: SyntheticProcessState

    scenario: SyntheticScenario

    data_origin: DataOrigin

    provenance: str

    corrupted: bool = False

    expected_condition: str = Field(min_length=1)



    def as_batch_measurement(self) -> BatchMeasurement:

        """Convert a valid observed record into the existing batch domain object."""

        return BatchMeasurement(

            batch_id=self.batch_id,

            sku_id=self.sku_id,

            context_id=self.context_id,

            iteration=0,

            regions=(self.as_region_measurement(),),

            metadata=_metadata(DataOrigin.SYNTHETIC),

        )



    def as_region_measurement(self) -> RegionMeasurement:

        """Convert a valid observed record into the project's domain measurement."""

        if self.corrupted:

            raise ValueError(

                "corrupted synthetic observation cannot be converted to a valid measurement"

            )

        return RegionMeasurement(

            region_id=self.region_id,

            L=self.observed_L,

            a=self.observed_a,

            b=self.observed_b,

            gloss=self.observed_gloss,

        )





class SyntheticDataset(FrozenModel):

    """Immutable generated calibration observations and one evaluation target."""



    scenario: SyntheticScenario

    config: SyntheticConfig

    calibration_observations: tuple[SyntheticObservation, ...] = ()

    target_observation: SyntheticObservation

    target_required_adjustment: tuple[float, ...]

    master_measurement: MasterMeasurement

    context: CalibrationContext

    data_origin: DataOrigin

    provenance: str



    @model_validator(mode="after")

    def _synthetic_only(self) -> Self:

        if self.data_origin is not DataOrigin.SYNTHETIC:

            raise ValueError("synthetic dataset must have SYNTHETIC data origin")

        if not self.provenance.startswith("synthetic_generator:"):

            raise ValueError(

                "synthetic dataset provenance must identify the generator"

            )

        if any(

            item.data_origin is not DataOrigin.SYNTHETIC

            for item in self.calibration_observations

        ):

            raise ValueError("all calibration observations must be synthetic")

        if self.target_observation.data_origin is not DataOrigin.SYNTHETIC:

            raise ValueError("target observation must be synthetic")

        return self





# Interpretable hidden channel sensitivities. They are synthetic-only constants.

_L_EFFECT = (2.6, -0.7)

_A_EFFECT = (0.9, 2.0)

_B_EFFECT = (-0.5, 1.6)

_CHANNEL_SCALE = 1.5





def _required_adjustment(

    config: SyntheticConfig, scenario: SyntheticScenario

) -> tuple[float, ...]:

    if scenario is SyntheticScenario.NEAR_BOUNDARY:

        return tuple(0.95 * upper for upper in config.channel_upper_bounds)

    if scenario is SyntheticScenario.UNREACHABLE:

        return tuple(1.5 * upper for upper in config.channel_upper_bounds)

    if scenario is SyntheticScenario.CORRECTABLE:

        return tuple(0.5 * upper for upper in config.channel_upper_bounds)

    if scenario is SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE:

        return tuple(0.4 * upper for upper in config.channel_upper_bounds)

    if scenario is SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT:

        return tuple(0.5 * upper for upper in config.channel_upper_bounds)

    if scenario is SyntheticScenario.SPARSE_CALIBRATION:

        return tuple(0.5 * upper for upper in config.channel_upper_bounds)

    return tuple(0.5 * upper for upper in config.channel_upper_bounds)





def _scenario_offset(scenario: SyntheticScenario) -> tuple[float, float, float]:

    return {

        SyntheticScenario.CORRECTABLE: (1.4, -0.7, 0.5),

        SyntheticScenario.NEAR_BOUNDARY: (4.5, -0.8, 0.4),

        SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE: (1.2, -0.5, 0.3),

        SyntheticScenario.UNREACHABLE: (6.5, -2.0, 1.5),

        SyntheticScenario.UNSEEN_SKU_OOD: (1.8, 0.7, -0.8),

        SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT: (1.3, -0.6, 0.4),

        SyntheticScenario.SPARSE_CALIBRATION: (1.4, -0.7, 0.5),

    }[scenario]





def _expected_condition(scenario: SyntheticScenario) -> str:

    return {

        SyntheticScenario.CORRECTABLE: (

            "controllable colour deviation inside synthetic bounds"

        ),

        SyntheticScenario.NEAR_BOUNDARY: (

            "required correction is close to a synthetic operating bound"

        ),

        SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE: (

            "gloss deviation is driven by a process-side variable"

        ),

        SyntheticScenario.UNREACHABLE: (

            "required colour correction exceeds the synthetic controllable range"

        ),

        SyntheticScenario.UNSEEN_SKU_OOD: (

            "SKU/context is absent from calibration observations"

        ),

        SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT: (

            "measurement contains controlled noise or corruption"

        ),

        SyntheticScenario.SPARSE_CALIBRATION: (

            "calibration support is intentionally sparse"

        ),

    }[scenario]





def _provenance(

    config: SyntheticConfig, scenario: SyntheticScenario

) -> str:

    return f"{config.provenance_label}:{scenario.value}"





def _oracle_delta(

    adjustments: tuple[float, ...], process_offset: tuple[float, float, float]

) -> tuple[float, float, float]:

    """Hidden nonlinear/saturating response; never used as the learned model."""

    saturation = tanh(sum(value / _CHANNEL_SCALE for value in adjustments))

    channel_l = sum(

        effect * value

        for effect, value in zip(_L_EFFECT, adjustments, strict=True)

    )

    channel_a = sum(

        effect * value

        for effect, value in zip(_A_EFFECT, adjustments, strict=True)

    )

    channel_b = sum(

        effect * value 

        for effect, value in zip(_B_EFFECT, adjustments, strict=True)

    )

    # tanh bounds the channel contribution while preserving a locally interpretable response.

    return (

        1.8 * saturation + 0.7 * tanh(channel_l / 3.0) + process_offset[0],

        1.4 * tanh(channel_a / 3.0) + process_offset[1],

        1.4 * tanh(channel_b / 3.0) + process_offset[2],

    )





def _gloss_oracle(

    adjustments: tuple[float, ...], process_variable: float

) -> float:

    """Synthetic gloss response, deliberately separate from colour response."""

    channel_effect = 1.5 * tanh(sum(adjustments) / _CHANNEL_SCALE)

    return 60.0 + channel_effect + 8.0 * process_variable


def _base_master(config: SyntheticConfig) -> MasterMeasurement:

    return MasterMeasurement(

        sku_id=config.sku_id,

        regions=tuple(

            RegionMeasurement(

                region_id=region_id,

                L=50.0 + index * 2.0,

                a=8.0,

                b=4.0,

                gloss=60.0,

            )

            for index, region_id in enumerate(config.region_ids)

        ),

        metadata=_metadata(DataOrigin.SYNTHETIC),

    )





def _metadata(origin: DataOrigin) -> MeasurementMetadata:

    from datetime import UTC, datetime



    return MeasurementMetadata(

        instrument_id="SYNTHETIC_INSTRUMENT",

        geometry="synthetic_geometry",

        illuminant="SYNTHETIC",

        observer="2deg",

        specular_mode="SCI",

        aperture_mm=8.0,

        measured_at=datetime(2026, 1, 1, tzinfo=UTC),

        data_origin=origin,

    )





def _make_adjustments(

    config: SyntheticConfig, rng: Random, index: int

) -> tuple[ChannelAdjustment, ...]:

    values = []

    for lo, hi in zip(

        config.channel_lower_bounds,
        config.channel_upper_bounds,
        strict=True
    ):

        # Calibration observations remain inside the declared synthetic envelope.

        value = lo + (hi - lo) * rng.random()

        if index == 0:

            value = 0.0

        values.append(value)

    return tuple(

        ChannelAdjustment(channel_id=cid, delta=value)

        for cid, value in zip(config.channel_ids, values, strict=True)

    )





def _make_observation(

    config: SyntheticConfig,

    scenario: SyntheticScenario,

    rng: Random,

    *,

    observation_id: str,

    batch_id: str,

    region_id: str,

    adjustments: tuple[ChannelAdjustment, ...],

    process_offset: tuple[float, float, float],

    gloss_process_variable: float,

    corrupted: bool = False,

) -> SyntheticObservation:

    adjustment_values = tuple(item.delta for item in adjustments)

    delta_l, delta_a, delta_b = _oracle_delta(

        adjustment_values, process_offset

    )

    noise = config.noise_level

    observed_l = 50.0 + delta_l + rng.gauss(0.0, noise)

    observed_a = 8.0 + delta_a + rng.gauss(0.0, noise)

    observed_b = 4.0 + delta_b + rng.gauss(0.0, noise)

    observed_gloss = (

        _gloss_oracle(adjustment_values, gloss_process_variable)

        + rng.gauss(0.0, noise)

    )



    if corrupted:

        # Intentionally invalid for the quality gate; never clamp this value.

        observed_l = 105.0



    return SyntheticObservation(

        observation_id=observation_id,

        batch_id=batch_id,

        sku_id=config.sku_id,

        context_id=config.context_id,

        region_id=region_id,

        adjustments=adjustments,

        observed_L=observed_l,

        observed_a=observed_a,

        observed_b=observed_b,

        observed_gloss=observed_gloss,

        process_state=SyntheticProcessState(

            batch_process_offset=process_offset,

            gloss_process_variable=gloss_process_variable,

            oracle_adjustment=adjustment_values,

        ),

        scenario=scenario,

        data_origin=DataOrigin.SYNTHETIC,

        provenance=_provenance(config, scenario),

        corrupted=corrupted,

        expected_condition=_expected_condition(scenario),

    )





def generate(

    config: SyntheticConfig,

    scenario: SyntheticScenario = SyntheticScenario.CORRECTABLE,

) -> SyntheticDataset:

    """Generate one deterministic synthetic scenario using only a local seeded RNG."""

    rng = Random(config.seed)

    target_sku = config.sku_id

    target_context = config.context_id

    if scenario is SyntheticScenario.UNSEEN_SKU_OOD:

        target_sku = f"{config.sku_id}_UNSEEN"

        target_context = f"{config.context_id}_UNSEEN"



    calibration_config = config

    target_config = config.model_copy(

        update={"sku_id": target_sku, "context_id": target_context}

    )

    required_adjustment = _required_adjustment(target_config, scenario)

    inverse_values = tuple(-value for value in required_adjustment)

    process_offset = tuple(

        -value

        for value in _oracle_delta(

            inverse_values, (0.0, 0.0, 0.0)

        )

    )

    if scenario is SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE:

        process_offset = _scenario_offset(scenario)

    calibration_count = config.calibration_observations

    if scenario is SyntheticScenario.SPARSE_CALIBRATION:

        calibration_count = min(2, calibration_count)



    observations: list[SyntheticObservation] = []

    for index in range(calibration_count):

        adjustments = _make_adjustments(

            calibration_config, rng, index

        )

        for region_index, region_id in enumerate(

            calibration_config.region_ids

        ):

            observations.append(

                _make_observation(

                    calibration_config,

                    scenario,

                    rng,

                    observation_id=f"OBS_{index:03d}_{region_index:02d}",

                    batch_id=f"BATCH_CAL_{index:03d}",

                    region_id=region_id,

                    adjustments=adjustments,

                    process_offset=(0.0, 0.0, 0.0),

                    gloss_process_variable=0.0,

                )

            )



    target_adjustments = tuple(

        ChannelAdjustment(channel_id=cid, delta=0.0)

        for cid in target_config.channel_ids

    )

    if scenario is SyntheticScenario.NEAR_BOUNDARY:

        target_adjustments = tuple(

            ChannelAdjustment(channel_id=cid, delta=0.0)

            for cid in target_config.channel_ids

        )

    gloss_variable = (

        1.0

        if scenario is SyntheticScenario.PROCESS_SIDE_GLOSS_FAILURE

        else 0.0

    )

    corrupted = scenario is SyntheticScenario.NOISY_CORRUPTED_MEASUREMENT

    target = _make_observation(

        target_config,

        scenario,

        rng,

        observation_id="OBS_TARGET",

        batch_id="BATCH_TARGET",

        region_id=calibration_config.region_ids[0],

        adjustments=target_adjustments,

        process_offset=process_offset,

        gloss_process_variable=gloss_variable,

        corrupted=corrupted,

    )



    # Scenario ground truth is deliberately metadata, not an input to downstream decisions.

    return SyntheticDataset(

        scenario=scenario,

        config=target_config,

        calibration_observations=tuple(observations),

        target_observation=target,

        target_required_adjustment=required_adjustment,

        master_measurement=_base_master(target_config),

        context=CalibrationContext(

            context_id=target_config.context_id,

            printer_id="SYNTHETIC_PRINTER",

            glaze_context="SYNTHETIC_GLAZE",

            ink_set_id="SYNTHETIC_INK",

            data_origin=DataOrigin.SYNTHETIC,

        ),

        data_origin=DataOrigin.SYNTHETIC,

        provenance=_provenance(target_config, scenario),

    )
