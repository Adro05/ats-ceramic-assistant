"""Configuration models and loaders.

All tunable behaviour lives in YAML under ``configs/`` and is validated into
frozen pydantic models at load time. Loading fails loudly on:

* missing files,
* unknown keys (typos),
* duplicate YAML keys (PyYAML would otherwise silently keep the last one),
* out-of-range values,
* a synthetic-data file that is not labeled ``data_origin: synthetic``.

Config models contain no ``dict``/``list``/``set``: keyed collections are stored as
tuples of typed records and looked up through methods (``for_sku``, ``scenario``).
For readability the YAML may still write such a collection as a ``key: value``
mapping; it is converted to records at load time. This is enforced by
``tests/test_immutability.py``.

Every value in the shipped YAML files is a placeholder or synthetic modelling choice
unless a file says otherwise; none has been confirmed by the client.

The config directory is ``<repo>/configs`` by default and can be overridden with
the ``ATS_CERAMIC_CONFIG_DIR`` environment variable or an explicit ``path``.
(The default resolves relative to the source tree, so it assumes a source checkout
or editable install.)
"""

import hashlib
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

import yaml
from pydantic import (
    AfterValidator,
    Field,
    FiniteFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError

from ats_ceramic.schemas import (
    CalibrationContext,
    DataOrigin,
    FrozenModel,
    Identifier,
    NonEmptyText,
    NonNegativeFinite,
    PositiveFinite,
    ToleranceSpec,
    require_unique,
)

ENV_CONFIG_DIR = "ATS_CERAMIC_CONFIG_DIR"

T = TypeVar("T", bound=FrozenModel)


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed or invalid."""


# --------------------------------------------------------------------------- #
# Shared constrained types and helpers
# --------------------------------------------------------------------------- #

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


def _check_ordered(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError(f"range minimum {value[0]} exceeds maximum {value[1]}")
    return value


OrderedRange = Annotated[
    tuple[NonNegativeFinite, NonNegativeFinite], AfterValidator(_check_ordered)
]
"""Inclusive [min, max] range of non-negative finite floats."""

IntRange = Annotated[tuple[PositiveInt, PositiveInt], AfterValidator(_check_ordered)]
"""Inclusive [min, max] range of positive integers."""


def _records_from_mapping(value: Any, key_name: str, value_name: str) -> Any:
    """Turn ``{key: value}`` into a tuple of ``{key_name: key, value_name: value}`` records.

    Non-mapping input (e.g. an already record-shaped list) passes through unchanged.
    """
    if isinstance(value, Mapping):
        return tuple({key_name: k, value_name: v} for k, v in value.items())
    return value


class QuantityScales(FrozenModel):
    """One positive scale per measured quantity (noise sigma, replicate limit, ...)."""

    L: PositiveFinite
    a: PositiveFinite
    b: PositiveFinite
    gloss: PositiveFinite


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


class ChannelConfig(FrozenModel):
    """One controllable printer channel and its allowed correction range.

    The range is a change relative to the current state. Zero (no change) must always
    be allowed; one-sided ranges (e.g. a channel that can only be increased) are
    permitted. Real channels, units and ranges are discovery dependencies.
    """

    channel_id: Identifier
    label: NonEmptyText
    unit: Identifier
    min_delta: FiniteFloat
    max_delta: FiniteFloat

    @model_validator(mode="after")
    def _range_is_usable(self) -> Self:
        if not (self.min_delta <= 0.0 <= self.max_delta) or self.min_delta == self.max_delta:
            raise ValueError(
                f"channel {self.channel_id}: require min_delta <= 0 <= max_delta and "
                f"min_delta < max_delta (got {self.min_delta}, {self.max_delta})"
            )
        return self


class ChannelSetConfig(FrozenModel):
    """The set of controllable channels (abstract and synthetic in this prototype)."""

    schema_version: Literal[1]
    data_origin: DataOrigin
    is_placeholder: bool
    max_total_abs_delta: PositiveFinite | None = None
    channels: tuple[ChannelConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_channel_ids(self) -> Self:
        require_unique((c.channel_id for c in self.channels), "channel_id")
        return self

    @property
    def channel_ids(self) -> tuple[str, ...]:
        """Channel identifiers in configured order."""
        return tuple(c.channel_id for c in self.channels)


# --------------------------------------------------------------------------- #
# Tolerances and measurement quality
# --------------------------------------------------------------------------- #


class MismatchPolicy(StrEnum):
    """How to treat a metadata mismatch between batch and master measurements."""

    FAIL = "fail"
    WARN = "warn"
    IGNORE = "ignore"


class MetadataMismatchPolicy(FrozenModel):
    """Per-field policy for colorimetric metadata mismatches."""

    instrument_id: MismatchPolicy
    geometry: MismatchPolicy
    illuminant: MismatchPolicy
    observer: MismatchPolicy
    specular_mode: MismatchPolicy
    aperture_mm: MismatchPolicy


class MeasurementQualityConfig(FrozenModel):
    """Thresholds for the measurement quality gate (placeholders until a gauge study)."""

    is_placeholder: bool
    ab_abs_max: PositiveFinite
    gloss_plausible_max: PositiveFinite | None = None
    min_replicates_recommended: PositiveInt
    replicate_std_warn: QuantityScales
    replicate_std_fail: QuantityScales
    metadata_mismatch: MetadataMismatchPolicy

    @model_validator(mode="after")
    def _warn_not_above_fail(self) -> Self:
        for quantity in ("L", "a", "b", "gloss"):
            warn = getattr(self.replicate_std_warn, quantity)
            fail = getattr(self.replicate_std_fail, quantity)
            if warn > fail:
                raise ValueError(
                    f"replicate_std_warn.{quantity} ({warn}) exceeds "
                    f"replicate_std_fail.{quantity} ({fail})"
                )
        return self


class SkuToleranceOverride(FrozenModel):
    """Tolerance that replaces the default for one SKU."""

    sku_id: Identifier
    tolerance: ToleranceSpec


class AcceptanceConfig(FrozenModel):
    """Default acceptance tolerance plus optional per-SKU overrides (immutable tuple)."""

    default: ToleranceSpec
    sku_overrides: tuple[SkuToleranceOverride, ...] = ()

    @field_validator("sku_overrides", mode="before")
    @classmethod
    def _overrides_from_mapping(cls, value: Any) -> Any:
        return _records_from_mapping(value, "sku_id", "tolerance")

    @model_validator(mode="after")
    def _unique_override_skus(self) -> Self:
        require_unique((o.sku_id for o in self.sku_overrides), "sku_id")
        return self

    def for_sku(self, sku_id: str) -> ToleranceSpec:
        """Return the SKU-specific tolerance, falling back to the default.

        A linear scan is deliberate: override lists are tiny and this keeps the model
        free of any mutable lookup structure.
        """
        for override in self.sku_overrides:
            if override.sku_id == sku_id:
                return override.tolerance
        return self.default


class ToleranceConfig(FrozenModel):
    """Contents of ``tolerances.yaml``."""

    schema_version: Literal[1]
    acceptance: AcceptanceConfig
    measurement_quality: MeasurementQualityConfig

    def for_sku(self, sku_id: str) -> ToleranceSpec:
        """Return the SKU-specific tolerance, falling back to the default."""
        return self.acceptance.for_sku(sku_id)

    @property
    def has_placeholders(self) -> bool:
        """True if any tolerance or quality threshold is still a placeholder."""
        specs = [self.acceptance.default, *(o.tolerance for o in self.acceptance.sku_overrides)]
        return any(s.is_placeholder for s in specs) or self.measurement_quality.is_placeholder


# --------------------------------------------------------------------------- #
# Synthetic generator configuration
# --------------------------------------------------------------------------- #


class ScenarioId(StrEnum):
    """Intended synthetic scenarios; all must be configured."""

    CORRECTABLE_COLOR_SHIFT = "correctable_color_shift"
    NEAR_BOUNDARY = "near_boundary"
    PROCESS_SIDE_GLOSS_FAILURE = "process_side_gloss_failure"
    UNREACHABLE_COLOR_SHIFT = "unreachable_color_shift"
    UNSEEN_SKU = "unseen_sku"
    NOISY_MEASUREMENT = "noisy_measurement"
    INSUFFICIENT_CALIBRATION = "insufficient_calibration"


class ShiftReachability(StrEnum):
    """Whether the hidden process shift can be removed by a bounded correction."""

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"


class SkuCalibrationStatus(StrEnum):
    """Whether the scenario's SKUs have calibration data."""

    CALIBRATED = "calibrated"
    UNSEEN = "unseen"


class CalibrationDensity(StrEnum):
    """Amount of calibration data available for the scenario's context."""

    FULL = "full"
    SPARSE = "sparse"


class ExpectedBehavior(StrEnum):
    """Desired system behaviour; used only for evaluation, never as model input."""

    RECOMMEND = "recommend"
    RECOMMEND_WITH_CAUTION = "recommend_with_caution"
    ESCALATE = "escalate"


class ScenarioConfig(FrozenModel):
    """Parameters of one synthetic scenario (synthetic modelling choices)."""

    description: NonEmptyText
    n_batches: PositiveInt
    uncorrected_de00_range: OrderedRange
    gloss_shift_range: OrderedRange
    shift_reachability: ShiftReachability
    sku_calibration: SkuCalibrationStatus
    calibration_density: CalibrationDensity
    measurement_noise_multiplier: PositiveFinite = 1.0
    bad_measurement_probability: Probability
    expected_behavior: ExpectedBehavior


class ScenarioEntry(FrozenModel):
    """A scenario parameter set together with its identifier."""

    scenario_id: ScenarioId
    spec: ScenarioConfig


class CalibrationDesignConfig(FrozenModel):
    """Size of the synthetic calibration experiment."""

    n_kiln_runs: Annotated[int, Field(ge=2)]
    replicates: PositiveInt
    full_experiments_per_sku_context: PositiveInt
    sparse_experiments_per_sku_context: PositiveInt

    @model_validator(mode="after")
    def _sparse_smaller_than_full(self) -> Self:
        if self.sparse_experiments_per_sku_context >= self.full_experiments_per_sku_context:
            raise ValueError("sparse experiments must be fewer than full experiments")
        return self


class GeneratorConfig(FrozenModel):
    """Global parameters of the synthetic generator (synthetic modelling choices)."""

    seed: Annotated[int, Field(ge=0)]
    n_skus: PositiveInt
    n_uncalibrated_skus: PositiveInt
    regions_per_sku: IntRange
    contexts: tuple[CalibrationContext, ...] = Field(min_length=2)
    sparse_calibration_context_id: Identifier
    measurement_noise: QuantityScales
    kiln_run_effect: QuantityScales
    gloss_to_L_coupling: NonNegativeFinite
    calibration_design: CalibrationDesignConfig

    @model_validator(mode="after")
    def _check_contexts(self) -> Self:
        ids = [c.context_id for c in self.contexts]
        require_unique(ids, "context_id")
        if self.sparse_calibration_context_id not in ids:
            raise ValueError("sparse_calibration_context_id is not one of generator.contexts")
        if any(c.data_origin != DataOrigin.SYNTHETIC for c in self.contexts):
            raise ValueError("all synthetic generator contexts must have data_origin: synthetic")
        return self


class SyntheticConfig(FrozenModel):
    """Contents of ``synthetic_scenarios.yaml`` (scenarios stored as an immutable tuple)."""

    schema_version: Literal[1]
    data_origin: DataOrigin
    generator: GeneratorConfig
    scenarios: tuple[ScenarioEntry, ...]

    @field_validator("scenarios", mode="before")
    @classmethod
    def _scenarios_from_mapping(cls, value: Any) -> Any:
        return _records_from_mapping(value, "scenario_id", "spec")

    @model_validator(mode="after")
    def _check_labels_and_scenarios(self) -> Self:
        if self.data_origin != DataOrigin.SYNTHETIC:
            raise ValueError("synthetic scenario config must declare data_origin: synthetic")
        configured = [e.scenario_id.value for e in self.scenarios]
        require_unique(configured, "scenario_id")
        missing = {s.value for s in ScenarioId} - set(configured)
        if missing:
            raise ValueError(f"missing scenarios: {sorted(missing)}")
        return self

    @property
    def scenario_ids(self) -> tuple[ScenarioId, ...]:
        """Configured scenario identifiers in file order."""
        return tuple(e.scenario_id for e in self.scenarios)

    def scenario(self, scenario_id: ScenarioId) -> ScenarioConfig:
        """Return the parameters of one scenario."""
        for entry in self.scenarios:
            if entry.scenario_id == scenario_id:
                return entry.spec
        raise KeyError(scenario_id)  # unreachable for a validated config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in seen
        except TypeError:  # unhashable key; SafeLoader reports it with a better message
            continue
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:map", _construct_unique_mapping)


def default_config_dir() -> Path:
    """Return the configuration directory.

    Uses ``$ATS_CERAMIC_CONFIG_DIR`` if set, otherwise ``<repo root>/configs``.
    """
    override = os.environ.get(ENV_CONFIG_DIR)
    if override:
        directory = Path(override).expanduser()
    else:
        directory = Path(__file__).resolve().parents[2] / "configs"
    if not directory.is_dir():
        raise ConfigError(
            f"Configuration directory not found: {directory}. "
            f"Set {ENV_CONFIG_DIR} or pass an explicit path."
        )
    return directory


def config_fingerprint(path: Path) -> str:
    """SHA-256 of a config file with line endings normalised to LF.

    Intended for audit lineage ("which configuration produced this result").
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}:\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def _load(path: Path | None, default_name: str, model: type[T]) -> T:
    resolved = Path(path) if path is not None else default_config_dir() / default_name
    raw = _read_yaml(resolved)
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {resolved}:\n{exc}") from exc


def load_tolerance_config(path: Path | None = None) -> ToleranceConfig:
    """Load and validate ``tolerances.yaml``."""
    return _load(path, "tolerances.yaml", ToleranceConfig)


def load_channel_config(path: Path | None = None) -> ChannelSetConfig:
    """Load and validate ``channels.yaml``."""
    return _load(path, "channels.yaml", ChannelSetConfig)


def load_synthetic_config(path: Path | None = None) -> SyntheticConfig:
    """Load and validate ``synthetic_scenarios.yaml``."""
    return _load(path, "synthetic_scenarios.yaml", SyntheticConfig)