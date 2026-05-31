# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Async client for the OlmoEarth Studio HTTP API.

Thin wrapper over ``httpx.AsyncClient``: Bearer auth, the
``{records, meta, errors}`` envelope, and typed helpers for the
endpoints the agent needs. Endpoint shapes verified against the live
API (``/users/me``, ``/projects/search``) on 2026-05-28 and the
openapi v0.1.0 spec.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, cast

import httpx

from olmoearth_agent.types import ApiEnvelope, ProjectRef, StudioContext

DEFAULT_BASE_URL = "https://olmoearth.allenai.org/api/v1"

#: Upstream statuses worth retrying (transient gateway / rate-limit).
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
#: Total attempts for a retryable (idempotent) request.
_MAX_ATTEMPTS = 3
#: Bounded client-side scan for endpoints with no server-side filter
#: (predictions by project, results by prediction): at most this many pages.
_MAX_FILTER_PAGES = 10


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

    async def _send(
        self, method: str, path: str, *, retry: bool = False, **kwargs: Any
    ) -> httpx.Response:
        """Issue a request and raise on non-2xx.

        When ``retry`` is set, transient failures (gateway 5xx, 429, and
        transport errors like timeouts) are retried with a short backoff.
        Use it only for idempotent reads, never for creates: a retried POST
        could duplicate a write.
        """
        attempts = _MAX_ATTEMPTS if retry else 1
        for attempt in range(attempts):
            try:
                resp = await self._client.request(method, path, **kwargs)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                if not (
                    exc.response.status_code in _RETRY_STATUSES
                    and attempt < attempts - 1
                ):
                    raise
            except httpx.TransportError:
                if attempt >= attempts - 1:
                    raise
            await asyncio.sleep(0.4 * (attempt + 1))
        raise RuntimeError("unreachable retry state")  # pragma: no cover

    @staticmethod
    def _unwrap(payload: Any) -> ApiEnvelope[dict[str, Any]]:
        """Build the envelope, surfacing a populated ``errors`` array.

        An HTTP 200 can still carry business errors in ``errors`` with empty
        ``records``; raising here stops the ``env.one or {}`` helpers from
        silently collapsing that to ``{}`` (which would read as success).
        """
        env = ApiEnvelope.from_response(payload)
        if env.errors and not env.records:
            raise RuntimeError(f"Studio API error: {env.errors}")
        return env

    async def get(self, path: str) -> ApiEnvelope[dict[str, Any]]:
        """GET ``path`` and unwrap the envelope (retries transient errors)."""
        resp = await self._send("GET", path, retry=True)
        return self._unwrap(resp.json())

    async def post(
        self, path: str, body: dict[str, Any], *, retry: bool = False
    ) -> ApiEnvelope[dict[str, Any]]:
        """POST ``body`` to ``path`` and unwrap the envelope.

        ``retry`` retries transient failures; set it only for idempotent
        search endpoints, never for creates (avoids duplicate writes).
        """
        resp = await self._send("POST", path, retry=retry, json=body)
        return self._unwrap(resp.json())

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
            "/projects/search", {"limit": limit, "offset": offset}, retry=True
        )

    async def create_project(self, *, name: str, description: str) -> dict[str, Any]:
        """Create a project (``POST /projects`` → 200). Returns the new record."""
        env = await self.post("/projects", {"name": name, "description": description})
        return env.one or {}

    async def get_project(self, project_id: str) -> dict[str, Any]:
        """Fetch one project (``GET /projects/{id}``). Raises on 404."""
        env = await self.get(f"/projects/{project_id}")
        return env.one or {}

    async def delete_project(self, project_id: str) -> dict[str, Any]:
        """Delete a project (``DELETE /projects/{id}`` → 202).

        The delete envelope wraps the deleted record as
        ``{"records": [{"record": {...}}]}`` (verified live 2026-05-28),
        so we unwrap the inner ``record``.
        """
        resp = await self._client.request("DELETE", f"/projects/{project_id}")
        resp.raise_for_status()
        env = ApiEnvelope.from_response(resp.json())
        record = env.one or {}
        return cast(dict[str, Any], record.get("record", record))

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

        Each record carries a ``model_id``: this is how a client
        discovers a reusable model id for a new prediction (resolves the
        PLAN.md §4 ``model_id`` provenance gap for the reuse case).

        Note: ``PredictionSearchRequest`` rejects a ``project_id`` field
        (openapi v0.1.0; sending one returns HTTP 422), so when
        ``project_id`` is given we filter client-side, scanning up to
        ``_MAX_FILTER_PAGES`` pages of ``limit`` to gather matches that fall
        past the first page.
        """
        env = await self.post(
            "/predictions/search", {"limit": limit, "offset": offset}, retry=True
        )
        if project_id is None:
            return env
        matches = [r for r in env.records if r.get("project_id") == project_id]
        page_len, pages = len(env.records), 1
        while page_len == limit and pages < _MAX_FILTER_PAGES:
            offset += limit
            page = await self.post(
                "/predictions/search", {"limit": limit, "offset": offset}, retry=True
            )
            page_len, pages = len(page.records), pages + 1
            matches.extend(r for r in page.records if r.get("project_id") == project_id)
        env.records = matches
        return env

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        """Fetch one prediction-result record (``GET /prediction-results/{id}``).

        Returns ``tile_urls`` (XYZ/MVT templates), ``property_names``,
        ``result_metadata``, ``file_format``, and ``download_token``.
        """
        env = await self.get(f"/prediction-results/{result_id}")
        return env.one or {}

    async def search_prediction_results(
        self,
        *,
        prediction_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search prediction-results.

        Note: the API's ``PredictionResultSearchRequest`` has no
        ``prediction_id`` filter (openapi v0.1.0), so when ``prediction_id``
        is given we filter client-side, scanning up to ``_MAX_FILTER_PAGES``
        pages of ``limit`` to gather results past the first page.
        """
        env = await self.post(
            "/prediction-results/search", {"limit": limit, "offset": offset}, retry=True
        )
        if prediction_id is None:
            return env
        matches = [r for r in env.records if r.get("prediction_id") == prediction_id]
        page_len, pages = len(env.records), 1
        while page_len == limit and pages < _MAX_FILTER_PAGES:
            offset += limit
            page = await self.post(
                "/prediction-results/search",
                {"limit": limit, "offset": offset},
                retry=True,
            )
            page_len, pages = len(page.records), pages + 1
            matches.extend(
                r for r in page.records if r.get("prediction_id") == prediction_id
            )
        env.records = matches
        return env

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

    # --- models ---

    async def search_models(
        self, *, limit: int = 50, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search models (read-only). Returns the full envelope.

        Each record carries ``model_type`` (e.g. ``"fine_tuned"`` vs an
        embeddings run), a human ``name``, and ``wizard_answers`` (encoder
        variant, imagery sources, spatial/temporal context). This is how a
        client tells a fine-tuned model apart from an embeddings one.

        Note: ``/models`` is absent from openapi v0.1.0, but the live API
        serves ``POST /models/search`` and ``GET /models/{id}`` (``GET
        /models`` with no id is 404). Verified live 2026-05-30.
        """
        return await self.post(
            "/models/search", {"limit": limit, "offset": offset}, retry=True
        )

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """Fetch one model record (``GET /models/{id}``).

        Returns ``model_type``, ``name``, ``status``, and ``wizard_answers``.
        Undocumented in openapi v0.1.0; verified live 2026-05-30.
        """
        env = await self.get(f"/models/{model_id}")
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
