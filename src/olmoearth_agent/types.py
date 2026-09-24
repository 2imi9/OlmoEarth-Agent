# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Data classes shared across the tool, studio, and harness layers.

``ApiEnvelope`` wraps Studio responses, discovered from the live API on
2026-05-28 (``{"records": [...], "meta": {...}, "errors": ...}``, not bare
objects); ``ProjectRef`` and ``StudioContext`` carry ``load_context``;
``ProvenanceManifest`` is one provenance entry per tool call. The other
classes sketched in ``PLAN.md`` §2 were never used by the code and are not
defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ApiEnvelope(Generic[T]):
    """Studio API response envelope: ``{records, meta, errors}``.

    Verified live 2026-05-28 against ``GET /users/me`` and
    ``POST /projects/search``. Tools should unwrap :attr:`records` and
    read ``meta["total"]`` for pagination.
    """

    records: list[T] = field(default_factory=list)
    meta: dict[str, Any] | None = None
    errors: list[Any] | None = None

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> ApiEnvelope[dict[str, Any]]:
        """Build an envelope from a decoded JSON response body."""
        return ApiEnvelope(
            records=body.get("records") or [],
            meta=body.get("meta"),
            errors=body.get("errors"),
        )

    @property
    def one(self) -> T | None:
        """First record, or ``None`` when the envelope is empty."""
        return self.records[0] if self.records else None

    @property
    def total(self) -> int | None:
        """``meta.total`` when present (server-reported result count)."""
        if self.meta is None:
            return None
        value = self.meta.get("total")
        return int(value) if value is not None else None


@dataclass
class ProjectRef:
    """Reference to a Studio project."""

    id: str
    name: str


@dataclass
class StudioContext:
    """The user's active Studio context, returned by ``load_context``."""

    user_id: str | None = None
    user_name: str | None = None
    organization: str | None = None
    projects: list[ProjectRef] = field(default_factory=list)


@dataclass
class ProvenanceManifest:
    """One provenance entry per tool call (operational rule §3.13).

    ``response_summary`` holds only ids / status / counts: never raw
    geometry (rule §3.1). See
    :class:`olmoearth_agent.provenance.log.ProvenanceLog`.
    """

    run_id: str
    timestamp: str
    api_call: str
    request_hash: str
    response_summary: dict[str, Any]
    prediction_id: str | None = None
    model_id: str | None = None
    dataset_hashes: list[str] | None = None
