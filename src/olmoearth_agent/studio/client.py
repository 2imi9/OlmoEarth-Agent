# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Async client for the OlmoEarth Studio HTTP API.

Thin wrapper over ``httpx.AsyncClient``: Bearer auth, the
``{records, meta, errors}`` envelope, and typed helpers for the
endpoints the agent needs. Endpoint shapes verified against the live
API (``/users/me``, ``/projects/search``) on 2026-05-28 and the
openapi v0.1.0 spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from olmoearth_agent.types import ApiEnvelope, ProjectRef, StudioContext

DEFAULT_BASE_URL = "https://olmoearth.allenai.org/api/v1"


@dataclass
class StudioConfig:
    """Connection config for the Studio API."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> StudioConfig:
        """Read ``OLMOEARTH_API_KEY`` / ``OLMOEARTH_BASE_URL`` from env.

        Raises
        ------
        RuntimeError
            If ``OLMOEARTH_API_KEY`` is missing or still the placeholder.
        """
        key = os.environ.get("OLMOEARTH_API_KEY", "").strip()
        if not key or key == "replace-me":
            msg = (
                "OLMOEARTH_API_KEY is not set. Copy .env.example to .env "
                "and paste your Studio API key (Studio UI -> profile -> "
                "API Keys)."
            )
            raise RuntimeError(msg)
        return cls(
            api_key=key,
            base_url=os.environ.get("OLMOEARTH_BASE_URL", DEFAULT_BASE_URL),
        )


class StudioClient:
    """Async OlmoEarth Studio API client.

    Examples
    --------
    >>> import asyncio
    >>> async def main():
    ...     async with StudioClient.from_env() as studio:
    ...         ctx = await studio.load_context()
    ...         print(ctx.user_name, len(ctx.projects))
    >>> asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(self, config: StudioConfig | None = None) -> None:
        self.config = config or StudioConfig.from_env()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )

    @classmethod
    def from_env(cls) -> StudioClient:
        """Construct a client from environment variables (see :class:`StudioConfig`)."""
        return cls(StudioConfig.from_env())

    async def __aenter__(self) -> StudioClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # --- low-level ---

    async def get(self, path: str) -> ApiEnvelope[dict[str, Any]]:
        """GET ``path`` and unwrap the envelope."""
        resp = await self._client.get(path)
        resp.raise_for_status()
        return ApiEnvelope.from_response(resp.json())

    async def post(
        self, path: str, body: dict[str, Any]
    ) -> ApiEnvelope[dict[str, Any]]:
        """POST ``body`` to ``path`` and unwrap the envelope."""
        resp = await self._client.post(path, json=body)
        resp.raise_for_status()
        return ApiEnvelope.from_response(resp.json())

    # --- typed helpers ---

    async def users_me(self) -> dict[str, Any]:
        """Return the authenticated user's profile record."""
        env = await self.get("/users/me")
        return env.one or {}

    async def search_projects(
        self, *, limit: int = 50, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search projects (read-only). Returns the full envelope."""
        return await self.post(
            "/projects/search", {"limit": limit, "offset": offset}
        )

    async def create_project(
        self, *, name: str, description: str
    ) -> dict[str, Any]:
        """Create a project (``POST /projects``). Returns the new record."""
        env = await self.post(
            "/projects", {"name": name, "description": description}
        )
        return env.one or {}

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Fetch one prediction record (``GET /predictions/{id}``)."""
        env = await self.get(f"/predictions/{prediction_id}")
        return env.one or {}

    async def search_predictions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search predictions (read-only). Optionally scope to a project.

        Each record carries a ``model_id`` — this is how a client
        discovers a reusable model id for a new prediction (resolves the
        PLAN.md §4 ``model_id`` provenance gap for the reuse case).
        """
        body: dict[str, Any] = {"limit": limit, "offset": offset}
        if project_id is not None:
            body["project_id"] = project_id
        return await self.post("/predictions/search", body)

    async def submit_prediction(
        self,
        *,
        name: str,
        project_id: str,
        area_id: str,
        model_id: str,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        """Create a prediction (``POST /predictions``). Returns the record.

        All six fields are required by ``PredictionWrite`` (openapi
        v0.1.0). Note ``area_id`` (not dataset) is the geographic anchor.
        ``model_id`` is typically reused from a prior prediction found via
        :meth:`search_predictions`.
        """
        env = await self.post(
            "/predictions",
            {
                "name": name,
                "project_id": project_id,
                "area_id": area_id,
                "model_id": model_id,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        return env.one or {}

    async def load_context(self) -> StudioContext:
        """Assemble the user's active Studio context.

        Combines ``/users/me`` with a project search so the agent has the
        user identity, org, and available projects in one object.
        """
        me = await self.users_me()
        projects_env = await self.search_projects(limit=50)
        orgs = me.get("organizations") or []
        org_name = orgs[0].get("name") if orgs else None
        return StudioContext(
            user_id=me.get("id"),
            user_name=me.get("name"),
            organization=org_name,
            projects=[
                ProjectRef(id=p.get("id", ""), name=p.get("name", ""))
                for p in projects_env.records
            ],
        )
