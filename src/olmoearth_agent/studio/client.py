# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Async client for the OlmoEarth Studio HTTP API.

Thin wrapper over ``httpx.AsyncClient``: Bearer auth, the
``{records, meta, errors}`` envelope, and typed helpers for the
endpoints the agent needs. Endpoint shapes verified against the live
API (``/users/me``, ``/projects/search``) on 2026-05-28 and the
openapi v0.1.0 spec. On 2026-10-31 the predictions' ``project_id`` and
``deleted_time`` filters, the prediction-results' ``prediction_id`` filter
with ``include`` and the results' ``file_formats`` were checked against the
live API; the areas' ``project_id`` filter, the search limits and offsets and
the write bodies come from the spec alone (unverified live).

A 401 or 403 raises :class:`StudioAccessError` and is never retried: Studio
refused the key, and a retry cannot change that. The read-only methods for
Studio's annotations and datasets leave out the ids of the people behind a
record, and a result's download token never reaches an error text or httpx's
request log.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, cast

import httpx

from olmoearth_agent.security import egress
from olmoearth_agent.types import ApiEnvelope, ProjectRef, StudioContext

DEFAULT_BASE_URL = "https://olmoearth.allenai.org/api/v1"

#: Upstream statuses worth retrying (transient gateway / rate-limit).
_RETRY_STATUSES = frozenset({429, 502, 503, 504})
#: Studio refused the key: :class:`StudioAccessError`, never retried.
_ACCESS_STATUSES = frozenset({401, 403})
#: Total attempts for a retryable (idempotent) request.
_MAX_ATTEMPTS = 3
#: Bounded client-side scan, used only when a server answers a filtered search
#: with HTTP 422 (a server older than the filters): at most this many pages.
_MAX_FILTER_PAGES = 10
#: The largest ``limit`` each search accepts (the spec of 2026-10-31; a larger
#: one is a 422, e.g. 1001 on prediction-results, measured). The smallest one
#: sent is 1. ``/models/search`` is not in the spec: its 1000 is a cautious
#: cap, not a documented one.
_SEARCH_LIMIT_MAX: dict[str, int] = {
    "/projects/search": 1000,
    "/predictions/search": 10000,
    "/prediction-results/search": 1000,
    "/areas/search": 10000,
    "/models/search": 1000,
    "/annotations/search": 10000,
    "/labels/search": 10000,
    "/labelsets/search": 1000,
    "/datasets/search": 10000,
}
#: The largest ``offset`` the searches that bound it accept (the spec of
#: 2026-10-31); a larger one is refused before the request, since Studio
#: cannot page past it. The other searches take any offset of 0 or more.
_SEARCH_OFFSET_MAX: dict[str, int] = {
    "/predictions/search": 10000,
    "/areas/search": 10000,
    "/annotations/search": 10000,
    "/labels/search": 10000,
    "/datasets/search": 10000,
}
#: The default cap of :meth:`StudioClient.prediction_result_files`: 2 GiB.
DEFAULT_MAX_FILES_BYTES = 2 * 1024**3
#: Why a ``/models`` call is refused once the server has shown it has no route.
MODELS_UNAVAILABLE = (
    "Studio answered an earlier /models request on this client with HTTP 405, "
    "or with HTTP 404 as for a path it does not route, so no model record is "
    "read"
)
#: FastAPI's body for a path that no route matches (Starlette's default 404).
_UNROUTED_BODY = {"detail": "Not Found"}

#: A download token in a URL's query string, as httpx's request log writes it.
_TOKEN_IN_URL = re.compile(r"(download_token=)[^&\s\"']+")


class _RedactDownloadToken(logging.Filter):
    """Masks a download token in httpx's request log (``HTTP Request: GET ...``).

    httpx logs every request's URL at INFO, and the files download carries its
    token in the query string; a token is a credential and never reaches a log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite a record whose message holds a token; always keep the record."""
        try:
            message = record.getMessage()
        except Exception:  # a malformed record is not this filter's to break
            return True
        if "download_token=" in message:
            record.msg = _TOKEN_IN_URL.sub(r"\1<redacted>", message)
            record.args = ()
        return True


logging.getLogger("httpx").addFilter(_RedactDownloadToken())


class StudioAccessError(RuntimeError):
    """Studio refused a request with HTTP 401 or 403: the key may not do it.

    Raised by every request of :class:`StudioClient` and never retried: a
    key's access does not change on a second try. Studio's keys can be
    read-only, and ``/users/me`` does not show a key's access level (measured
    on 2026-10-31 with a read-only key: its ``permissions`` list the user's
    full set), so a refusal is the only signal; the spec declares a 403 on
    ``POST /predictions``. A read-only key never tried a write, so how Studio
    words that refusal is unverified. The text names the method, the path
    without its query string (a download token travels in one) and Studio's
    ``errors[0]`` code and message when the body has them.

    ``contract`` carries the sentence the model must state (the registry puts
    it on the failed envelope, beside the error), and :attr:`hint` replaces
    the registry's generic hint: neither a check of the ids nor a retry can
    help.

    Attributes
    ----------
    method, path
        The refused request; ``path`` has no query string.
    status_code
        401 or 403.
    code
        Studio's ``errors[0].code`` (e.g. ``permission_error``), or ``None``.
    contract
        ``{"must_state": [sentence]}``.
    """

    #: The failed envelope's hint in place of the registry's generic one.
    hint = (
        "Studio refused this request for the API key in use. Do not retry it, "
        "and do not try another call that writes to Studio in this run: tell "
        "the user that Studio refused it, that the key may be read-only or lack "
        "the permission, and that nothing was created or changed."
    )

    def __init__(
        self,
        method: str,
        path: str,
        status_code: int,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        self.method = method.upper()
        self.path = path.split("?", 1)[0]
        self.status_code = status_code
        self.code = code
        detail = f"HTTP {status_code}" + (f", {code}" if code else "")
        if message:
            detail += f": {message}"
        super().__init__(
            f"Studio refused {self.method} {self.path} ({detail}). The Studio API "
            "key may be read-only or lack the permission for this request; "
            "nothing was created or changed."
        )
        self.contract: dict[str, Any] = {
            "must_state": [
                f"Studio refused this request (HTTP {status_code}): the API key "
                "may be read-only or lack the permission, and nothing was "
                "created or changed."
            ]
        }


def _studio_error(
    resp: httpx.Response, *, secret: str | None = None
) -> tuple[str | None, str | None]:
    """Studio's ``errors[0]`` ``code`` and ``message`` from an error body, if it parses.

    The spec's error envelope is ``{"records", "meta", "errors": [{"code",
    "message"}]}``; anything else gives ``(None, None)``. ``secret`` (e.g. a
    download token), should Studio echo it, is masked before the message is
    cut to 300 characters, so no cut leaves a part of it behind.
    """
    try:
        body = resp.json()
    except ValueError:
        return None, None
    errors = body.get("errors") if isinstance(body, dict) else None
    first = errors[0] if isinstance(errors, list) and errors else None
    if not isinstance(first, dict):
        return None, None

    def text(value: Any) -> str | None:
        """``value`` as text with ``secret`` masked, or ``None`` when empty."""
        if not value:
            return None
        out = str(value)
        return out.replace(secret, "<redacted>") if secret else out

    message = text(first.get("message"))
    return text(first.get("code")), (message[:300] if message else None)


def _unrouted(resp: httpx.Response) -> bool:
    """Whether ``resp``'s body is FastAPI's for a path no route matches."""
    try:
        return bool(resp.json() == _UNROUTED_BODY)
    except ValueError:
        return False


def _page(path: str, limit: int, offset: int) -> dict[str, int]:
    """A search's ``limit`` and ``offset`` within what ``path`` accepts.

    ``limit`` is clamped to 1 .. the spec's maximum for ``path``
    (:data:`_SEARCH_LIMIT_MAX`) and ``offset`` to 0 or more, so a limit a model
    chose never comes back as an HTTP 422. An ``offset`` past the spec's
    maximum (:data:`_SEARCH_OFFSET_MAX`) is refused instead of clamped: a
    clamped offset would return another page than the one asked for.

    Raises
    ------
    ValueError
        If ``offset`` is past the most ``path`` accepts.
    """
    top = _SEARCH_LIMIT_MAX.get(path, 1000)
    start = max(0, int(offset))
    last = _SEARCH_OFFSET_MAX.get(path)
    if last is not None and start > last:
        raise ValueError(
            f"offset {start} is past the {last} that Studio's {path} accepts; "
            "Studio's search cannot page further, so narrow the search instead"
        )
    return {"limit": max(1, min(int(limit), top)), "offset": start}


def _without(value: Any, key: str) -> Any:
    """``value`` less ``key`` when it is a dict; anything else unchanged."""
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k != key}


def _annotation_without_people(record: dict[str, Any]) -> dict[str, Any]:
    """An annotation record less ``annotator_id`` and its images' ``user_id``."""
    out: dict[str, Any] = _without(record, "annotator_id")
    images = out.get("annotation_images")
    if isinstance(images, list):
        out["annotation_images"] = [_without(image, "user_id") for image in images]
    return out


def _dataset_without_people(record: dict[str, Any]) -> dict[str, Any]:
    """A dataset record less ``requester_id`` (the user who asked for it)."""
    out: dict[str, Any] = _without(record, "requester_id")
    return out


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
        # Guard the (env-overridable) base URL before the Bearer key is bound
        # to a client: a malicious OLMOEARTH_BASE_URL must not exfiltrate it.
        egress.validate_endpoint(self.config.base_url, "studio")
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )
        #: True once ``/models`` answered as a route this server does not serve
        #: (:meth:`_note_models_failure`); every later ``/models`` call is then
        #: refused without a request, for this client's lifetime.
        self.models_unavailable = False

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

    async def get_bytes(
        self, url: str, *, timeout: float | None = None
    ) -> httpx.Response:
        """Raw authenticated GET of an absolute URL (no envelope unwrap).

        For proxying binary Studio assets (raster tiles) over this client's
        keep-alive connection pool, so a burst of fetches reuses connections
        instead of a fresh TLS handshake per request. ``url`` may be absolute
        (it overrides ``base_url``); the Bearer header rides along. The caller
        is responsible for host allow-listing (the tile proxy SSRF-checks the
        URL before calling this).
        """
        # Override the client-level ``Accept: application/json`` so a binary
        # asset (e.g. an image tile) is returned as bytes, not a JSON error.
        return await self._client.get(
            url,
            headers={"Accept": "*/*"},
            timeout=timeout if timeout is not None else self.config.timeout_seconds,
        )

    # --- low-level ---

    async def _send(
        self, method: str, path: str, *, retry: bool = False, **kwargs: Any
    ) -> httpx.Response:
        """Issue a request and raise on non-2xx.

        When ``retry`` is set, transient failures (gateway 5xx, 429, and
        transport errors like timeouts) are retried with a short backoff.
        Use it only for idempotent reads, never for creates: a retried POST
        could duplicate a write. A 401 or 403 raises
        :class:`StudioAccessError` at once, never retried.
        """
        attempts = _MAX_ATTEMPTS if retry else 1
        for attempt in range(attempts):
            try:
                resp = await self._client.request(method, path, **kwargs)
                if resp.status_code in _ACCESS_STATUSES:
                    # Not retried: the key's access does not change on a retry.
                    raise StudioAccessError(
                        method, path, resp.status_code, *_studio_error(resp)
                    )
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

    async def _search(
        self,
        path: str,
        filters: dict[str, Any],
        *,
        limit: int,
        offset: int,
        match: tuple[str, str] | None = None,
    ) -> ApiEnvelope[dict[str, Any]]:
        """POST a search with server-side ``filters``; a 422 falls back to a scan.

        Studio's searches take filter objects (``{"project_id": {"eq": id}}``
        returned the same ids as a client-side filter, live on 2026-10-31). A
        server older than these filters answered such a field with HTTP 422, so
        on a 422 the page is asked for again without them and, when ``match``
        names a ``(field, value)``, filtered client-side over at most
        :data:`_MAX_FILTER_PAGES` pages of ``limit``, as this client did before
        the filters, and never past the spec's largest offset; ``meta.total``
        then counts the matches found. The fallback sends no other key of
        ``filters`` (an ``include`` among them), so its records are the bare
        ones an older server returns. A 422 also comes back for a value that
        is not a UUID: the scan then finds no match, the answer the client-side
        filter always gave.
        """
        page = _page(path, limit, offset)
        try:
            return await self.post(path, {**page, **filters}, retry=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 422:
                raise
        # Fallback for an older server: the request as it was before the filters.
        env = await self.post(path, page, retry=True)
        if match is None:
            return env
        key, value = match
        size, start = page["limit"], page["offset"]
        last = _SEARCH_OFFSET_MAX.get(path)
        matches = [r for r in env.records if r.get(key) == value]
        page_len, pages = len(env.records), 1
        while (
            page_len == size
            and pages < _MAX_FILTER_PAGES
            and (last is None or start + size <= last)
        ):
            start += size
            more = await self.post(path, {"limit": size, "offset": start}, retry=True)
            page_len, pages = len(more.records), pages + 1
            matches.extend(r for r in more.records if r.get(key) == value)
        env.records = matches
        # The server's total counts every record, not the matches.
        env.meta = {**(env.meta or {}), "total": len(matches)}
        return env

    async def _search_records(
        self, path: str, filters: dict[str, Any] | None, limit: int, offset: int
    ) -> ApiEnvelope[dict[str, Any]]:
        """POST ``path`` with ``filters`` passed through and a clamped page (a read)."""
        return await self.post(
            path, {**(filters or {}), **_page(path, limit, offset)}, retry=True
        )

    # --- typed helpers ---

    async def users_me(self) -> dict[str, Any]:
        """Return the authenticated user's profile record."""
        env = await self.get("/users/me")
        return env.one or {}

    async def search_projects(
        self, *, limit: int = 50, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search projects (read-only). Returns the full envelope."""
        path = "/projects/search"
        return await self.post(path, _page(path, limit, offset), retry=True)

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
        resp = await self._send("DELETE", f"/projects/{project_id}")
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
        """Search predictions (read-only), newest first. Optionally scope to a project.

        Each record carries a ``model_id``: this is how a client
        discovers a reusable model id for a new prediction (resolves the
        PLAN.md §4 ``model_id`` provenance gap for the reuse case).

        Deleted predictions are left out with ``{"deleted_time": {"exists":
        false}}`` (accepted live on 2026-10-31; none of the account's
        predictions was deleted, so its effect on one is unobserved). With
        ``project_id`` the server filters too (``{"project_id": {"eq": id}}``,
        the same ids as a client-side filter, live on 2026-10-31), and a page
        holds up to ``limit`` of the project's predictions; that ``meta.total``
        then counts the project's predictions, not all, is assumed (a filtered
        search's total was not checked live). A server that answers these
        filters with HTTP 422 gets the old request (:meth:`_search`).
        """
        filters: dict[str, Any] = {"deleted_time": {"exists": False}}
        match: tuple[str, str] | None = None
        if project_id is not None:
            filters["project_id"] = {"eq": project_id}
            match = ("project_id", project_id)
        return await self._search(
            "/predictions/search", filters, limit=limit, offset=offset, match=match
        )

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        """Fetch one prediction-result record (``GET /prediction-results/{id}``).

        Returns ``tile_urls`` (XYZ/MVT templates), ``property_names``,
        ``result_metadata`` (filled on a GET by id, measured on 2026-10-31),
        ``file_formats`` (a list, e.g. ``["png"]``), and ``download_token``,
        a credential that never goes to the model or a log
        (:meth:`prediction_result_files` takes it).
        """
        env = await self.get(f"/prediction-results/{result_id}")
        return env.one or {}

    async def pixel_value(
        self, result_id: str, lon: float, lat: float
    ) -> dict[str, Any]:
        """Pointwise model output at one coordinate.

        ``GET /prediction-results/{id}/pixel-value?lon=&lat=``. The record's
        ``bands`` carry per-property ``raw_value`` (regression) and
        ``classification`` (categorical). Used to sample a result on a grid
        for a quantitative two-result comparison (no ground truth needed).
        """
        env = await self.get(
            f"/prediction-results/{result_id}/pixel-value?lon={lon}&lat={lat}"
        )
        return env.one or {}

    async def search_prediction_results(
        self,
        *,
        prediction_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search prediction-results.

        With ``prediction_id`` the server filters (``{"prediction_id": {"eq":
        id}}``) and the records carry their ``result_metadata`` (``"include":
        ["result_metadata"]``; without it a search record's metadata is null),
        both verified live on 2026-10-31; a page holds up to ``limit`` of
        them, at most 1000. A server that answers these with HTTP 422 gets the
        old request and a bounded client-side scan (:meth:`_search`), without
        ``include``: its records then carry no ``result_metadata``, so the
        results' declared outputs and period are unknown.
        """
        path = "/prediction-results/search"
        if prediction_id is None:
            return await self.post(path, _page(path, limit, offset), retry=True)
        return await self._search(
            path,
            {"prediction_id": {"eq": prediction_id}, "include": ["result_metadata"]},
            limit=limit,
            offset=offset,
            match=("prediction_id", prediction_id),
        )

    async def prediction_result_files(
        self, download_token: str, *, max_bytes: int = DEFAULT_MAX_FILES_BYTES
    ) -> bytes:
        """A result's model output files: the zip of ``GET /prediction-results/files``.

        ``download_token`` is the result record's token, sent as the query
        parameter the spec names, through this client (its base URL passed the
        egress check). The zip holds one GeoTIFF per UTM zone and, for a top-k
        result, a ``README.md`` (measured on 2026-10-31: 230 KB for a kelp
        result, 209 MB for a [0, 1] score result). The token is a credential:
        no error raised here holds it, and httpx's request log masks it.
        Redirects are not followed, since a target host would be outside the
        egress check; whether Studio redirects this download is unverified
        (the live probe's client followed redirects without saying so).

        Parameters
        ----------
        download_token
            The result's ``download_token``.
        max_bytes
            The most this reads; a larger ``Content-Length`` is refused before
            the body is read, and a body that grows past it while read.

        Returns
        -------
        bytes
            The zip.

        Raises
        ------
        StudioAccessError
            On HTTP 401 or 403.
        RuntimeError
            On any other non-2xx answer, a body over ``max_bytes`` or a failed
            transfer.
        ValueError
            If ``download_token`` is empty.
        """
        if not download_token:
            raise ValueError("a result's download_token is required")
        path = "/prediction-results/files"
        # One growing buffer, which getvalue hands back (in CPython without a
        # copy): a list of chunks and their join would hold the zip twice.
        buf = io.BytesIO()
        failure: str | None = None
        try:
            async with self._client.stream(
                "GET",
                path,
                params={"download_token": download_token},
                headers={"Accept": "*/*"},
            ) as resp:
                if resp.status_code in _ACCESS_STATUSES:
                    await resp.aread()
                    code, message = _studio_error(resp, secret=download_token)
                    raise StudioAccessError(
                        "GET", path, resp.status_code, code, message
                    )
                if not resp.is_success:
                    raise RuntimeError(
                        f"Studio answered HTTP {resp.status_code} to GET {path}; "
                        "no file was read"
                    )
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise RuntimeError(
                        f"the result's files are {int(declared)} bytes, over this "
                        f"client's cap of {max_bytes}; nothing was read"
                    )
                read = 0
                async for chunk in resp.aiter_bytes():
                    read += len(chunk)
                    if read > max_bytes:
                        raise RuntimeError(
                            "the result's files passed this client's cap of "
                            f"{max_bytes} bytes while read; nothing was kept"
                        )
                    buf.write(chunk)
        except httpx.HTTPError as exc:
            # httpx's own texts may hold the URL, and with it the token: only
            # the exception's type is kept, and the error is raised below, out
            # of this block, so it carries no __context__ (which would still
            # hold httpx's exception and its URL under ``from None``).
            failure = type(exc).__name__
        if failure is not None:
            raise RuntimeError(f"GET {path} failed ({failure}); no file was read")
        return buf.getvalue()

    async def submit_prediction(
        self,
        *,
        name: str,
        project_id: str,
        area_id: str,
        model_id: str,
        start_time: str,
        end_time: str,
        probability_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Create a prediction (``POST /predictions``). Returns the record.

        ``PredictionWrite`` (the spec of 2026-10-31) takes its areas as
        ``target_area_ids`` and has no ``area_id``, so the area is sent as
        ``target_area_ids: [area_id]``. ``probability_top_k`` adds
        ``probability_output: {"top_k": k}``: Studio then also writes each
        pixel's top-k class scores (segmentation only, the spec says). Both
        follow the spec alone: a read-only key cannot try a write, so neither
        is verified live. ``model_id`` is typically reused from a prior
        prediction found via :meth:`search_predictions`.

        Raises
        ------
        ValueError
            If ``probability_top_k`` is given and is not an integer of 1 or more.
        """
        body: dict[str, Any] = {
            "name": name,
            "project_id": project_id,
            "target_area_ids": [area_id],
            "model_id": model_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        if probability_top_k is not None:
            top_k = probability_top_k
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError("probability_top_k must be an integer of 1 or more")
            body["probability_output"] = {"top_k": top_k}
        env = await self.post("/predictions", body)
        return env.one or {}

    # --- areas (AOIs) ---

    async def create_area(
        self, *, name: str, geom: dict[str, Any], project_id: str
    ) -> dict[str, Any]:
        """Create an area / AOI (``POST /areas`` -> 200). Returns the record.

        ``geom`` is a GeoJSON geometry object (``{"type": "Polygon"|
        "MultiPolygon", "coordinates": [...]}``); Studio requires a polygon
        or multipolygon, since an area is a spatial region for inference. All
        three fields are required by ``AreaWrite`` (openapi v0.1.0). The
        returned ``id`` is the ``area_id`` :meth:`submit_prediction` consumes.
        """
        env = await self.post(
            "/areas", {"name": name, "geom": geom, "project_id": project_id}
        )
        return env.one or {}

    async def search_areas(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search areas (read-only). Optionally scope to a project.

        With ``project_id`` the server filters (``{"project_id": {"eq": id}}``,
        from the spec's ``AreaSearchRequest`` alone, unverified live; the same
        filter on predictions was checked live); a server that answers it with
        HTTP 422 gets the old request and a bounded client-side scan
        (:meth:`_search`). Search records carry ``id``/``name``/``project_id``
        but omit ``geom`` (lightweight); fetch one with :meth:`get_area` for
        the geometry.
        """
        path = "/areas/search"
        if project_id is None:
            return await self.post(path, _page(path, limit, offset), retry=True)
        return await self._search(
            path,
            {"project_id": {"eq": project_id}},
            limit=limit,
            offset=offset,
            match=("project_id", project_id),
        )

    async def get_area(self, area_id: str) -> dict[str, Any]:
        """Fetch one area (``GET /areas/{id}``), including its ``geom``."""
        env = await self.get(f"/areas/{area_id}")
        return env.one or {}

    async def delete_area(self, area_id: str) -> dict[str, Any]:
        """Delete an area (``DELETE /areas/{id}``). Returns the deleted record.

        Used by the live write test for cleanup. The delete envelope may wrap
        the record as ``{"record": {...}}`` (as project delete does), so we
        unwrap the inner ``record`` defensively.
        """
        resp = await self._send("DELETE", f"/areas/{area_id}")
        env = ApiEnvelope.from_response(resp.json())
        record = env.one or {}
        return cast(dict[str, Any], record.get("record", record))

    # --- models ---

    def _check_models(self) -> None:
        """Refuse a ``/models`` call once this server has shown it has no route."""
        if self.models_unavailable:
            raise RuntimeError(MODELS_UNAVAILABLE)

    def _note_models_failure(
        self, exc: httpx.HTTPStatusError, *, one_record: bool
    ) -> None:
        """Remember a ``/models`` answer that says the route is not served.

        A 405, or a 404 from ``/models/search``, is a missing route. A 404
        from ``GET /models/{id}`` may instead be one missing model, and how
        Studio answers a missing model is unverified, so it counts as a missing
        route only when its body is FastAPI's for a path no route matches
        (``{"detail": "Not Found"}``). Any other 404 body (Studio's
        ``not_found_error`` envelope, a ``detail`` naming the model, none)
        leaves later lookups of other models running.
        """
        status = exc.response.status_code
        if status == 405 or (
            status == 404 and (not one_record or _unrouted(exc.response))
        ):
            self.models_unavailable = True

    async def search_models(
        self, *, limit: int = 50, offset: int = 0
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search models (read-only). Returns the full envelope.

        Each record carries ``model_type`` (e.g. ``"fine_tuned"`` vs an
        embeddings run), a human ``name``, and ``wizard_answers`` (encoder
        variant, imagery sources, spatial/temporal context). This is how a
        client tells a fine-tuned model apart from an embeddings one.

        Note: ``/models`` is absent from the spec (still on 2026-10-31), but
        the live API serves ``POST /models/search`` and ``GET /models/{id}``
        (both answered 200 on 2026-10-31; ``GET /models`` with no id was a 404
        on 2026-05-30).
        After a 405, or a 404 that says the route is gone
        (:meth:`_note_models_failure`), every later ``/models`` call of this
        client raises at once (:attr:`models_unavailable`); callers treat any
        failure as no model record.
        """
        self._check_models()
        path = "/models/search"
        try:
            return await self.post(path, _page(path, limit, offset), retry=True)
        except httpx.HTTPStatusError as exc:
            self._note_models_failure(exc, one_record=False)
            raise

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """Fetch one model record (``GET /models/{id}``).

        Returns ``model_type``, ``name``, ``status``, and ``wizard_answers``.
        Undocumented in the spec; verified live 2026-05-30 and 2026-10-31.
        Refused at once after the route was found missing (:meth:`search_models`).
        """
        self._check_models()
        try:
            env = await self.get(f"/models/{model_id}")
        except httpx.HTTPStatusError as exc:
            self._note_models_failure(exc, one_record=True)
            raise
        return env.one or {}

    # --- annotations, labels, datasets (read-only) ---

    async def search_annotations(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search Studio's annotations (``POST /annotations/search``; a read).

        ``filters`` are the spec's ``AnnotationSearchRequest`` filter objects,
        passed through (e.g. ``{"project_id": {"eq": id}}``, or ``{"id":
        {"inc": [...]}}`` and ``{"intersects_geometry": <GeoJSON>}``, both
        accepted live on 2026-10-31). ``limit`` is at most 10000. Personal
        data is left out: each record loses ``annotator_id``, and each of its
        ``annotation_images`` its ``user_id``, so who labelled or photographed
        a place never reaches the agent.
        """
        env = await self._search_records("/annotations/search", filters, limit, offset)
        env.records = [_annotation_without_people(r) for r in env.records]
        return env

    async def aggregate_annotations(
        self, body: dict[str, Any]
    ) -> ApiEnvelope[dict[str, Any]]:
        """Aggregate annotations (``POST /annotations/aggregate``; a read).

        ``body`` is the spec's ``AggregateAnnotationsRequest``, passed through,
        e.g. ``{"metric": {"type": "count"}, "group_by": [{"field": "status"}],
        "project_id": {"eq": id}}`` (accepted live on 2026-10-31). The records
        are buckets (``key``, ``value``).

        Raises
        ------
        ValueError
            If ``body`` groups by ``annotator_id``: the annotators' ids are
            personal data, never returned.
        """
        groups = body.get("group_by")
        listed = groups if isinstance(groups, list) else []
        if "annotator_id" in {g.get("field") for g in listed if isinstance(g, dict)}:
            raise ValueError(
                "annotations are not grouped by annotator_id: the annotators' "
                "ids are personal data"
            )
        return await self.post("/annotations/aggregate", body, retry=True)

    async def search_labels(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search labels, a project's classes (``POST /labels/search``; a read).

        ``filters`` are the spec's ``LabelSearchRequest`` filter objects (e.g.
        ``{"labelset_id": {"eq": id}}``), passed through; ``limit`` is at most
        10000. A label record holds a class's ``name``, ``display_name`` and
        ``color``.
        """
        return await self._search_records("/labels/search", filters, limit, offset)

    async def search_labelsets(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search labelsets (``POST /labelsets/search``; a read).

        ``filters`` are the spec's ``LabelsetSearchRequest`` filter objects,
        passed through; ``limit`` is at most 1000 (the spec's bound for this
        search). A labelset record lists its ``labels``.
        """
        return await self._search_records("/labelsets/search", filters, limit, offset)

    async def search_datasets(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ApiEnvelope[dict[str, Any]]:
        """Search datasets (``POST /datasets/search``; a read).

        ``filters`` are the spec's ``DatasetSearchRequest`` filter objects
        (e.g. ``{"project_id": {"eq": id}}``), passed through; ``limit`` is at
        most 10000. Each record loses ``requester_id``, the user who asked for
        the dataset (personal data).
        """
        env = await self._search_records("/datasets/search", filters, limit, offset)
        env.records = [_dataset_without_people(r) for r in env.records]
        return env

    async def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Fetch one dataset (``GET /datasets/{id}``), less its ``requester_id``."""
        env = await self.get(f"/datasets/{dataset_id}")
        return _dataset_without_people(env.one or {})

    async def load_context(self, *, limit: int = 50, offset: int = 0) -> StudioContext:
        """Assemble the user's active Studio context.

        Combines ``/users/me`` with a project search so the agent has the
        user identity, org, and available projects in one object. ``limit``
        and ``offset`` page the project list.
        """
        me = await self.users_me()
        projects_env = await self.search_projects(limit=limit, offset=offset)
        return studio_context(me, projects_env.records)


def studio_context(me: dict[str, Any], projects: list[dict[str, Any]]) -> StudioContext:
    """A :class:`StudioContext` from a ``/users/me`` record and project records."""
    orgs = me.get("organizations") or []
    org_name = orgs[0].get("name") if orgs else None
    return StudioContext(
        user_id=me.get("id"),
        user_name=me.get("name"),
        organization=org_name,
        projects=[
            ProjectRef(id=p.get("id", ""), name=p.get("name", "")) for p in projects
        ],
    )
