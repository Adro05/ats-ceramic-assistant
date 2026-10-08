"""Immutable, versioned Golden Master registry boundary.

The registry is intentionally an in-memory prototype. Golden Master records are
immutable; registering a new version never edits an existing version. Instrument-
specific measurements are kept as immutable evidence associated with the exact
registered master version.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Self

from pydantic import Field, field_validator, model_validator

from ats_ceramic.schemas import (
    DataOrigin,
    FrozenModel,
    Identifier,
    MasterMeasurement,
    Sha256Hex,
    ToleranceSpec,
)


class MasterKind(StrEnum):
    """Role of a registered reference artifact."""

    CANONICAL_MASTER = "canonical_master"
    LEGACY_WORKING_REFERENCE = "legacy_working_reference"


class GoldenMasterRecord(FrozenModel):
    """Immutable identity and lineage record for one Golden Master version."""

    master_id: Identifier
    version: int = Field(ge=1)
    sku_id: Identifier
    context_id: Identifier
    master_kind: MasterKind
    content_bytes: bytes = Field(min_length=1)
    content_hash: Sha256Hex
    registered_at: datetime
    parent_version: int | None = Field(default=None, ge=1)
    parent_master_id: Identifier | None = None
    canonical_master_id: Identifier | None = None
    tolerance: ToleranceSpec | None = None
    data_origin: DataOrigin
    provenance: str = Field(min_length=1)

    @field_validator("registered_at")
    @classmethod
    def _require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        expected_hash = sha256(self.content_bytes).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash does not match content_bytes")

        if self.parent_version is None and self.parent_master_id is not None:
            raise ValueError("parent_master_id requires parent_version")

        if self.parent_version is not None and self.parent_master_id is None:
            raise ValueError("parent_version requires parent_master_id")

        if (
            self.master_kind is MasterKind.CANONICAL_MASTER
            and self.canonical_master_id is not None
        ):
            raise ValueError(
                "canonical master cannot point to a canonical_master_id"
            )

        if (
            self.master_kind is MasterKind.LEGACY_WORKING_REFERENCE
            and self.canonical_master_id is None
        ):
            raise ValueError(
                "legacy working reference requires canonical_master_id"
            )

        if (
            self.master_kind is MasterKind.LEGACY_WORKING_REFERENCE
            and self.canonical_master_id == self.master_id
        ):
            raise ValueError(
                "legacy canonical_master_id must identify a different master"
            )

        if self.data_origin is DataOrigin.SYNTHETIC and self.tolerance is not None:
            if not self.tolerance.is_placeholder:
                raise ValueError(
                    "synthetic tolerance must be explicitly marked as a placeholder"
                )

        if self.data_origin is DataOrigin.CLIENT and self.tolerance is not None:
            if self.tolerance.is_placeholder:
                raise ValueError("client tolerance cannot be a placeholder")

        return self


def content_sha256(content: bytes) -> str:
    """Return the deterministic lower-case SHA-256 digest of content bytes."""

    return sha256(content).hexdigest()


class GoldenMasterRegistry:
    """Write-once in-memory registry for Golden Master versions and measurements."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, int], GoldenMasterRecord] = {}
        self._measurements: dict[
            tuple[str, int], tuple[MasterMeasurement, ...]
        ] = {}

    def register(
        self,
        record: GoldenMasterRecord,
        *,
        parent: GoldenMasterRecord | None = None,
    ) -> GoldenMasterRecord:
        """Register one immutable version, rejecting conflicting lineage or identity."""

        key = (record.master_id, record.version)

        if key in self._records:
            raise ValueError("master version is already registered")

        if record.parent_version is None:
            if parent is not None:
                raise ValueError("parent supplied but record has no parent lineage")
        else:
            parent_key = (record.parent_master_id, record.parent_version)

            if parent_key not in self._records:
                raise ValueError("parent master version is not registered")

            registered_parent = self._records[parent_key]

            if parent is not None and parent != registered_parent:
                raise ValueError("supplied parent does not match registered lineage")

            if registered_parent.sku_id != record.sku_id:
                raise ValueError("parent SKU does not match child SKU")

            if registered_parent.context_id != record.context_id:
                raise ValueError("parent context does not match child context")

            if registered_parent.data_origin is not record.data_origin:
                raise ValueError(
                    "parent data origin does not match child data origin"
                )

            if record.version <= registered_parent.version:
                raise ValueError(
                    "child version must be greater than parent version"
                )

        if record.master_kind is MasterKind.LEGACY_WORKING_REFERENCE:
            canonical_id = record.canonical_master_id

            if not any(
                item.master_id == canonical_id
                and item.master_kind is MasterKind.CANONICAL_MASTER
                for item in self._records.values()
            ):
                raise ValueError(
                    "legacy reference must point to a registered canonical master"
                )

        self._records[key] = record
        self._measurements[key] = ()

        return record

    def get(self, master_id: str, version: int) -> GoldenMasterRecord:
        """Retrieve a registered version deterministically."""

        try:
            return self._records[(master_id, version)]
        except KeyError as exc:
            raise KeyError(
                f"master version not found: {master_id} v{version}"
            ) from exc

    def has_content_hash(self, content_hash: str) -> bool:
        """Return whether a SHA-256 content hash is already registered."""

        return any(
            record.content_hash == content_hash
            for record in self._records.values()
        )

    def list_versions(
        self,
        *,
        master_id: str | None = None,
        sku_id: str | None = None,
    ) -> tuple[GoldenMasterRecord, ...]:
        """List registered versions in deterministic master/version order."""

        records: Iterable[GoldenMasterRecord] = self._records.values()

        if master_id is not None:
            records = (
                record for record in records if record.master_id == master_id
            )

        if sku_id is not None:
            records = (
                record for record in records if record.sku_id == sku_id
            )

        return tuple(
            sorted(
                records,
                key=lambda record: (record.master_id, record.version),
            )
        )

    def associate_measurement(
        self,
        master_id: str,
        version: int,
        measurement: MasterMeasurement,
    ) -> None:
        """Attach immutable instrument-specific measurement evidence to a version."""

        record = self.get(master_id, version)

        if measurement.sku_id != record.sku_id:
            raise ValueError("measurement SKU does not match master SKU")

        if measurement.metadata.data_origin is not record.data_origin:
            raise ValueError(
                "measurement data origin does not match master"
            )

        if not measurement.metadata.instrument_id:
            raise ValueError("measurement instrument identity is required")

        key = (master_id, version)
        existing = self._measurements[key]

        if measurement in existing:
            raise ValueError(
                "measurement is already associated with this master version"
            )

        self._measurements[key] = existing + (measurement,)

    def measurements_for(
        self,
        master_id: str,
        version: int,
    ) -> tuple[MasterMeasurement, ...]:
        """Return associated measurements without exposing mutable registry state."""

        self.get(master_id, version)
        return self._measurements[(master_id, version)]