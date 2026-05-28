# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Harness data classes shared across the tool, studio, and harness layers.

Mirrors ``PLAN.md`` §2, plus the ``ApiEnvelope`` wrapper discovered from
the live API on 2026-05-28 (Studio responses are
``{"records": [...], "meta": {...}, "errors": ...}``, not bare objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")


@dataclass
class ToolExecutionError:
    """Structured error returned by a tool instead of raising."""

    message: str
    status_code: int | None = None


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
class TimeRange:
    """ISO-8601 date range, inclusive."""

    start: str
    end: str


@dataclass
class AOIWrapper:
    """Serialized geometry bundle. Mirrors Google Earth Agent's SGTWrapper."""

    error: ToolExecutionError | None = None
    count: int | None = None
    serialized_geojson: str | None = None
    crs: str | None = "EPSG:4326"


@dataclass
class ProjectRef:
    """Reference to a Studio project."""

    id: str
    name: str


@dataclass
class AreaRef:
    """Reference to a Studio area (the geographic anchor of a prediction)."""

    id: str
    aoi: AOIWrapper | None = None


@dataclass
class DatasetRef:
    """Reference to a Studio dataset (band/time configuration)."""

    id: str
    bands: list[str] = field(default_factory=list)
    time_range: TimeRange | None = None


@dataclass
class PredictionRef:
    """Reference to a Studio prediction.

    ``kind`` is a CLIENT-SIDE dispatch key (the API has no such field;
    every prediction takes a ``model_id``). See ``PLAN.md`` §4.
    """

    id: str
    kind: Literal["finetune", "embed", "reference"] = "reference"


@dataclass
class PredictionStatus:
    """Mirrors ``components.schemas.PredictionStatus`` (openapi v0.1.0)."""

    ref: PredictionRef
    state: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float | None = None
    eta_seconds: int | None = None


@dataclass
class ResultBundle:
    """Outputs of a completed prediction."""

    ref: PredictionRef
    tile_template: str | None = None
    vector_url: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class FilePath:
    """A written artifact path (used instead of returning raw geometry)."""

    error: ToolExecutionError | None = None
    path: str | None = None


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

    ``response_summary`` holds only ids / status / counts — never raw
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
