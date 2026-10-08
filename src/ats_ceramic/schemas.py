"""Validated domain schemas for the batch shade calibration assistant.

Two validation layers exist and must stay distinct::

    raw measurement -> quality gate -> validated domain object -> downstream processing

1. **Input validation / measurement quality gate** (``measurement.py``, planned).
   Works on RAW, untrusted records (plain mappings, e.g. parsed CSV rows). It runs
   deterministic checks and reports structured reason codes such as "L out of
   range" or "duplicate region". It must be able to *see* malformed data, because
   its job is to explain why data is rejected. Only acceptable data is turned into
   a domain object.
2. **Domain schemas** (this module). Deliberately strict: an invalid object cannot
   be constructed, so downstream code (colour engine, response model, optimiser)
   never re-checks basic validity. They are a backstop, not a diagnostic: a bare
   pydantic exception is not an acceptable message for a QC user.

Consequence: raw data such as ``L = 105`` goes through the gate, which returns FAIL
with a reason code. Constructing domain objects directly is for data already known
to be valid (tests, records this system itself wrote, clean synthetic data).

Design rules:

* Every model is frozen and rejects unknown fields, so records cannot be reassigned
  and misspelled keys fail loudly instead of being ignored.
* Models hold no ``list``/``dict``/``set``: collections are tuples. This is enforced
  by ``tests/test_immutability.py``. "Immutable" means in-process immutability of
  the objects; it is not tamper-proofing (that would belong to a storage layer).
* Numbers must be finite. CIELAB L* is bounded to [0, 100] by definition of the
  scale. Gloss is required to be non-negative: that is an ASSUMPTION about the
  gloss scale, to be confirmed.
* Every record that could come from either source carries ``data_origin`` so
  synthetic data can never be mistaken for client data.

What these schemas do NOT assert (all are discovery dependencies, represented as
opaque identifiers or optional fields): the instrument model and its configuration,
the gloss unit and angle, how many regions/measurement points a tile has and where
they are, what constitutes a calibration "context", the artwork file format, and
every tolerance value.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Shared constrained types
# --------------------------------------------------------------------------- #

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
"""Opaque, non-empty identifier (SKU, batch, printer, ...). No format is imposed."""

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Free text that must not be empty (labels, provenance notes)."""

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""Lower-case hexadecimal SHA-256 digest."""

PositiveFinite = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
LightnessL = Annotated[float, Field(ge=0.0, le=100.0, allow_inf_nan=False)]
"""CIELAB L*: bounded to [0, 100] by the definition of the scale."""


def require_unique(values: Iterable[str], label: str) -> None:
    """Raise ``ValueError`` if ``values`` contains duplicates.

    Shared by schema and config validators; ``label`` names the field in the message.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {label}: {sorted(duplicates)}")


class FrozenModel(BaseModel):
    """Base class: immutable, strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class DataOrigin(StrEnum):
    """Provenance label carried by every record."""

    SYNTHETIC = "synthetic"
    CLIENT = "client"


class SpecularMode(StrEnum):
    """Specular component handling of a colour measurement.

    * ``SCI`` = specular component **included**.
    * ``SCE`` = specular component **excluded**.

    Which mode(s) the client's spectrophotometer reports, and which one the
    laboratory uses for acceptance, is a DISCOVERY DEPENDENCY. This enum only lists
    the two standard options so a measurement can state which one it used. Nothing
    in the prototype assumes either.
    """

    SCI = "SCI"
    SCE = "SCE"


class Observer(StrEnum):
    """CIE standard observer as reported with a measurement.

    Which observer the client uses is a discovery dependency; nothing assumes one.
    """

    DEG_2 = "2deg"
    DEG_10 = "10deg"


class AcceptanceMetric(StrEnum):
    """Aggregate colour-difference statistic that governs acceptance (client decision)."""

    MAX_DE00 = "max_de00"
    MEAN_DE00 = "mean_de00"


class CorrectionSource(StrEnum):
    """Where a correction came from."""

    RECOMMENDED = "recommended"
    QC_MODIFIED = "qc_modified"
    MANUAL_BASELINE = "manual_baseline"  # operator's own edit, recorded in shadow mode


class VerificationOutcome(StrEnum):
    """Decision taken after a verification sample was measured."""

    ACCEPTED = "accepted"
    ITERATE = "iterate"
    ESCALATED = "escalated"


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


class ReplicateSpread(FrozenModel):
    """Sample standard deviation across replicate readings of one region."""

    L: NonNegativeFinite
    a: NonNegativeFinite
    b: NonNegativeFinite
    gloss: NonNegativeFinite


class RegionMeasurement(FrozenModel):
    """Lab colour and gloss of one named region of a tile.

    ``L``, ``a``, ``b`` are CIELAB L*, a*, b* (mean of replicate readings).

    ``gloss`` is the reading in the instrument's native scale. Its UNIT and
    measurement angle are discovery dependencies: the unit is not modelled at all,
    and the angle may be recorded in ``MeasurementMetadata.gloss_angle_deg``. Gloss
    values are therefore only comparable between measurements made the same way.

    Region definition (how many regions per tile, where, and how they map to the
    artwork) is also a discovery dependency; the prototype assumes the laboratory can
    report one L*, a*, b*, gloss value per named region.
    """

    region_id: Identifier
    L: LightnessL
    a: FiniteFloat
    b: FiniteFloat
    gloss: NonNegativeFinite
    n_replicates: int = Field(default=1, ge=1)
    replicate_spread: ReplicateSpread | None = None

    @model_validator(mode="after")
    def _spread_requires_replicates(self) -> Self:
        if self.replicate_spread is not None and self.n_replicates < 2:
            raise ValueError("replicate_spread requires n_replicates >= 2")
        return self


class MeasurementMetadata(FrozenModel):
    """Instrument and geometry context of a measurement, recorded as reported.

    No value here is assumed. The instrument model, geometry, illuminant, observer,
    SCI/SCE mode and aperture used by the client are all DISCOVERY DEPENDENCIES.

    Two measurements are only comparable if their colorimetric context matches. The
    quality gate compares ``instrument_id``, ``geometry``, ``illuminant``,
    ``observer``, ``specular_mode`` and ``aperture_mm`` between a batch and the
    master according to a configurable policy (``measurement_quality.
    metadata_mismatch`` in ``tolerances.yaml``).

    ``illuminant`` is normalised to upper case and ``geometry`` to lower case purely
    so equivalent spellings compare equal; this is a formatting convention, not a
    statement about what the client uses.
    """

    instrument_id: Identifier
    geometry: Identifier
    illuminant: Identifier
    observer: Observer
    specular_mode: SpecularMode
    aperture_mm: PositiveFinite
    gloss_angle_deg: PositiveFinite | None = None
    measured_at: AwareDatetime
    data_origin: DataOrigin

    @field_validator("illuminant")
    @classmethod
    def _normalise_illuminant(cls, value: str) -> str:
        return value.upper()

    @field_validator("geometry")
    @classmethod
    def _normalise_geometry(cls, value: str) -> str:
        return value.lower()


class MasterMeasurement(FrozenModel):
    """Approved master measurement of one SKU, per region."""

    sku_id: Identifier
    regions: tuple[RegionMeasurement, ...] = Field(min_length=1)
    metadata: MeasurementMetadata

    @model_validator(mode="after")
    def _unique_regions(self) -> Self:
        require_unique((r.region_id for r in self.regions), "region_id")
        return self


class BatchMeasurement(FrozenModel):
    """Measurement of a batch sample.

    ``iteration`` 0 is the first sample, printed before any system correction;
    iteration k >= 1 is the verification sample printed after correction k. (Whether
    the first sample uses an unmodified reference file is itself a discovery item:
    the client reports that historical reference files may have drifted.)
    """

    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    iteration: int = Field(default=0, ge=0)
    applied_correction_id: Identifier | None = None
    regions: tuple[RegionMeasurement, ...] = Field(min_length=1)
    metadata: MeasurementMetadata

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        require_unique((r.region_id for r in self.regions), "region_id")
        if self.iteration == 0 and self.applied_correction_id is not None:
            raise ValueError("iteration 0 cannot reference an applied correction")
        if self.iteration > 0 and self.applied_correction_id is None:
            raise ValueError("iteration > 0 requires applied_correction_id")
        return self


# --------------------------------------------------------------------------- #
# Context, tolerance, artwork
# --------------------------------------------------------------------------- #


class CalibrationContext(FrozenModel):
    """Conditions under which a printer response was calibrated.

    What actually constitutes a context (printer, glaze, ink lot, kiln, ...) is a
    discovery question; the fields are opaque identifiers until it is answered.
    """

    context_id: Identifier
    printer_id: Identifier
    glaze_context: Identifier
    ink_set_id: Identifier | None = None
    data_origin: DataOrigin


class ToleranceSpec(FrozenModel):
    """Acceptance tolerance for a SKU.

    ``is_placeholder`` has no default on purpose: whoever writes a tolerance must
    state whether it was confirmed by the client. The gloss unit/angle behind
    ``gloss_abs_threshold`` are discovery dependencies.
    """

    acceptance_metric: AcceptanceMetric
    delta_e00_threshold: PositiveFinite
    gloss_abs_threshold: NonNegativeFinite
    is_placeholder: bool
    source: NonEmptyText


class ArtworkReference(FrozenModel):
    """Pointer to the canonical artwork/reference file (not the file itself).

    The client's file formats are a discovery dependency, so ``file_format`` is an
    optional free identifier.
    """

    artwork_id: Identifier
    file_sha256: Sha256Hex | None = None
    file_format: Identifier | None = None


# --------------------------------------------------------------------------- #
# Corrections and verification
# --------------------------------------------------------------------------- #


class ChannelAdjustment(FrozenModel):
    """Change to one abstract printer channel, in that channel's configured unit."""

    channel_id: Identifier
    delta: FiniteFloat


class Correction(FrozenModel):
    """Batch-specific correction layer.

    The correction references the exact master version it was derived against
    (``master_id`` + ``master_content_hash``) but never modifies it. "No printer
    action" is represented by the absence of a Correction, not an empty one.
    Channels are abstract in this prototype; real controllable inputs are a
    discovery dependency.
    """

    correction_id: Identifier
    batch_id: Identifier
    sku_id: Identifier
    context_id: Identifier
    master_id: Identifier
    master_content_hash: Sha256Hex
    based_on_iteration: int = Field(ge=0)
    adjustments: tuple[ChannelAdjustment, ...] = Field(min_length=1)
    source: CorrectionSource
    created_at: AwareDatetime
    data_origin: DataOrigin

    @model_validator(mode="after")
    def _unique_channels(self) -> Self:
        require_unique((a.channel_id for a in self.adjustments), "channel_id")
        return self


class VerificationResult(FrozenModel):
    """Outcome of measuring a verification sample after a correction.

    ``worst_delta_e00`` and ``max_abs_gloss_delta`` are values recorded from the
    colour engine at decision time (for audit); they are not recomputed here.
    """

    batch_id: Identifier
    iteration: int = Field(ge=1)
    correction_id: Identifier
    measurement: BatchMeasurement
    outcome: VerificationOutcome
    worst_delta_e00: NonNegativeFinite | None = None
    max_abs_gloss_delta: NonNegativeFinite | None = None

    @model_validator(mode="after")
    def _measurement_matches(self) -> Self:
        m = self.measurement
        if m.batch_id != self.batch_id:
            raise ValueError("measurement.batch_id does not match batch_id")
        if m.iteration != self.iteration:
            raise ValueError("measurement.iteration does not match iteration")
        if m.applied_correction_id != self.correction_id:
            raise ValueError("measurement.applied_correction_id does not match correction_id")
        return self