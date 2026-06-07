# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Literature search over arXiv (Atom API) and OpenAlex (``/works``).

Pure query-build / parse / cross-source dedup helpers, plus async fetchers that
take an *injected* async getter (``Fetcher``) so the logic is testable without
any network. Powers the ``olmoearth-litsearch`` tool bundle (skill #15).

Both sources are used **key-free**: arXiv needs no auth; OpenAlex is queried via
the documented "polite pool" by sending a ``mailto`` (read from
``OLMOEARTH_OPENALEX_MAILTO`` when set, else the anonymous pool). An optional
``OPENALEX_API_KEY`` is *not* required. All returned fields are bibliographic
(ids, titles, DOIs, URLs) — never geometry — so they are safe to log and cite.

Design notes
------------
* The arXiv result envelope is emitted **after** the parse loop (the upstream
  GDM ``search_arxiv.py`` has a real bug that reprints the cumulative list once
  per entry; we avoid it).
* Dedup is deterministic — lowercased DOI, else normalized arXiv id, else a
  normalized-title key — and complementary fields are merged across sources
  (an arXiv hit contributes ``arxiv_id``/``pdf_url``; its OpenAlex twin
  contributes ``doi``/``cited_by_count``). No semantic re-ranking; within a
  single source the API's own relevance order is preserved.
"""

from __future__ import annotations

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from olmoearth_agent.security import egress

#: arXiv Atom search endpoint (no auth, ~1 req / 3 s courtesy rate).
ARXIV_API = "http://export.arxiv.org/api/query"
#: OpenAlex works endpoint (public; polite pool via ``mailto``).
OPENALEX_WORKS = "https://api.openalex.org/works"

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_CAP = 25
_ABSTRACT_CAP = 1200
_AUTHOR_CAP = 12

#: Transient upstream statuses worth retrying (matches StudioClient).
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_TIMEOUT_SECONDS = 20.0
_USER_AGENT = "OlmoEarth-Agent-litsearch (+https://github.com/2imi9/OlmoEarth-Agent)"

#: An injected async getter: ``(url, params) -> (status_code, body_text)``.
#: Lets tests drive the parsers with canned payloads and no network.
Fetcher = Callable[[str, dict[str, Any]], Awaitable[tuple[int, str]]]

_ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
_DOI_RE = re.compile(r"(?:doi:|https?://doi\.org/)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def _truncate(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _record(**fields: Any) -> dict[str, Any]:
    """A curated paper record with every output field present (defaulting None)."""
    base: dict[str, Any] = {
        "id": None,
        "title": None,
        "authors": [],
        "year": None,
        "venue": None,
        "doi": None,
        "arxiv_id": None,
        "url": None,
        "abstract": None,
        "cited_by_count": None,
        "pdf_url": None,
        "source": None,
    }
    base.update(fields)
    return base


def normalize_arxiv_id(raw: str) -> str | None:
    """Extract the version-stripped arXiv id from a URL/id/``arxiv:`` form."""
    match = _ARXIV_ID_RE.search(raw.strip())
    return match.group(1) if match else None


def normalize_doi(raw: str) -> str | None:
    """Extract a bare lowercased DOI from a DOI / ``doi:`` / doi.org URL form."""
    match = _DOI_RE.search(raw.strip())
    return match.group(1).lower().rstrip(".") if match else None


def _title_key(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedup_key(paper: dict[str, Any]) -> str:
    """Deterministic identity: DOI, else arXiv id, else normalized title."""
    if paper.get("doi"):
        return f"doi:{str(paper['doi']).lower()}"
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    return f"title:{_title_key(paper.get('title'))}"


def dedup_merge(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup by :func:`dedup_key`, first-wins, merging complementary fields.

    Order is preserved (single-source API relevance ordering is respected).
    When the same paper appears in both sources, the kept record is filled with
    any non-null fields the duplicate carries that it was missing, and ``source``
    becomes ``"arxiv+openalex"``.
    """
    out: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = dedup_key(paper)
        if key not in by_key:
            by_key[key] = paper
            out.append(paper)
            continue
        kept = by_key[key]
        for field, value in paper.items():
            if value not in (None, [], "") and kept.get(field) in (None, [], ""):
                kept[field] = value
        if paper.get("source") and paper["source"] not in (kept.get("source") or ""):
            kept["source"] = "+".join(
                sorted({*(kept.get("source") or "").split("+"), paper["source"]} - {""})
            )
    return out


def parse_arxiv_atom(
    xml_text: str, *, include_abstract: bool = True
) -> list[dict[str, Any]]:
    """Parse an arXiv Atom feed into curated records (envelope emitted once)."""
    try:
        # noqa: S314 - feed is fetched by us from the official arXiv API over a
        # capped GET (not arbitrary user input); ET.fromstring resolves no
        # external entities, so XXE does not apply.
        root = ET.fromstring(xml_text)  # noqa: S314
    except ET.ParseError:
        return []
    results: list[dict[str, Any]] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
        arxiv_id = normalize_arxiv_id(raw_id)
        authors = [
            (a.findtext(f"{_ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{_ATOM_NS}author")
        ]
        published = (entry.findtext(f"{_ATOM_NS}published") or "").strip()
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf_url = None
        for link in entry.findall(f"{_ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        doi = entry.findtext(f"{_ARXIV_NS}doi")  # external DOI only; never arXiv's own
        journal = entry.findtext(f"{_ARXIV_NS}journal_ref")
        abstract = entry.findtext(f"{_ATOM_NS}summary") if include_abstract else None
        results.append(
            _record(
                id=f"arxiv:{arxiv_id}" if arxiv_id else raw_id,
                title=_truncate(entry.findtext(f"{_ATOM_NS}title"), 400),
                authors=authors[:_AUTHOR_CAP],
                year=year,
                venue=(journal.strip() if journal else "arXiv"),
                doi=normalize_doi(doi) if doi else None,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id,
                abstract=_truncate(abstract, _ABSTRACT_CAP),
                pdf_url=pdf_url,
                source="arxiv",
            )
        )
    return results


def _invert_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Reconstruct OpenAlex's ``abstract_inverted_index`` into plain text."""
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def parse_openalex_works(
    payload: dict[str, Any], *, include_abstract: bool = True
) -> list[dict[str, Any]]:
    """Parse an OpenAlex ``/works`` JSON payload into curated records."""
    results: list[dict[str, Any]] = []
    for work in payload.get("results", []) or []:
        doi = normalize_doi(work["doi"]) if work.get("doi") else None
        oa_id = str(work.get("id") or "").rsplit("/", 1)[-1] or None
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
        ]
        source_name = ((work.get("primary_location") or {}).get("source") or {}).get(
            "display_name"
        )
        landing = (work.get("primary_location") or {}).get("landing_page_url")
        url = (
            f"https://doi.org/{doi}"
            if doi
            else landing or (f"https://openalex.org/{oa_id}" if oa_id else None)
        )
        abstract = (
            _invert_abstract(work.get("abstract_inverted_index"))
            if include_abstract
            else None
        )
        results.append(
            _record(
                id=f"doi:{doi}" if doi else (f"openalex:{oa_id}" if oa_id else None),
                title=_truncate(work.get("display_name"), 400),
                authors=[a for a in authors if a][:_AUTHOR_CAP],
                year=work.get("publication_year"),
                venue=source_name,
                doi=doi,
                url=url,
                abstract=_truncate(abstract, _ABSTRACT_CAP),
                cited_by_count=work.get("cited_by_count"),
                source="openalex",
            )
        )
    return results


def build_arxiv_params(
    query: str, *, max_results: int, id_list: str | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    if id_list:
        params["id_list"] = id_list
    else:
        params["search_query"] = f"all:{query}"
    return params


def build_openalex_params(
    query: str,
    *,
    max_results: int,
    year_from: int | None = None,
    year_to: int | None = None,
    mailto: str | None = None,
    extra_filter: str | None = None,
) -> dict[str, Any]:
    filters = []
    if year_from is not None:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{int(year_to)}-12-31")
    if extra_filter:
        filters.append(extra_filter)
    params: dict[str, Any] = {"per-page": max_results}
    if query:
        params["search"] = query
    if filters:
        params["filter"] = ",".join(filters)
    if mailto:
        params["mailto"] = mailto
    return params


# --------------------------------------------------------------------------- #
# async I/O
# --------------------------------------------------------------------------- #
def _openalex_mailto() -> str | None:
    value = os.environ.get("OLMOEARTH_OPENALEX_MAILTO", "").strip()
    return value or None


def _httpx_fetcher(client: httpx.AsyncClient) -> Fetcher:
    """A :data:`Fetcher` over ``httpx`` with bounded retry on transient errors."""

    async def fetch(url: str, params: dict[str, Any]) -> tuple[int, str]:
        egress.validate_endpoint(url, "litsearch")
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError:
                if attempt >= _MAX_ATTEMPTS - 1:
                    raise
            else:
                if not (
                    resp.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1
                ):
                    return resp.status_code, resp.text
            await asyncio.sleep(0.4 * (attempt + 1))
        raise RuntimeError("unreachable retry state")  # pragma: no cover

    return fetch


async def search_arxiv(
    fetch: Fetcher,
    query: str,
    *,
    max_results: int,
    include_abstract: bool = True,
    id_list: str | None = None,
) -> list[dict[str, Any]]:
    status, body = await fetch(
        ARXIV_API, build_arxiv_params(query, max_results=max_results, id_list=id_list)
    )
    if status != 200:
        return []
    return parse_arxiv_atom(body, include_abstract=include_abstract)


async def search_openalex(
    fetch: Fetcher,
    query: str,
    *,
    max_results: int,
    year_from: int | None = None,
    year_to: int | None = None,
    include_abstract: bool = True,
    extra_filter: str | None = None,
) -> list[dict[str, Any]]:
    import json as _json

    params = build_openalex_params(
        query,
        max_results=max_results,
        year_from=year_from,
        year_to=year_to,
        mailto=_openalex_mailto(),
        extra_filter=extra_filter,
    )
    status, body = await fetch(OPENALEX_WORKS, params)
    if status != 200:
        return []
    try:
        payload = _json.loads(body)
    except ValueError:
        return []
    return parse_openalex_works(payload, include_abstract=include_abstract)


async def search_literature(
    query: str,
    *,
    source: str = "both",
    max_results: int = DEFAULT_MAX_RESULTS,
    year_from: int | None = None,
    year_to: int | None = None,
    include_abstract: bool = False,
    fetch: Fetcher | None = None,
) -> dict[str, Any]:
    """Search arXiv and/or OpenAlex, dedup across sources, return curated records.

    Returns ``{"count", "papers", "sources", "warnings"}``. An empty result set
    is reported as empty (never fabricated). ``source`` is ``"arxiv"``,
    ``"openalex"``, or ``"both"`` (default).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required and must be non-empty.")
    if source not in {"arxiv", "openalex", "both"}:
        raise ValueError("source must be 'arxiv', 'openalex', or 'both'.")
    capped = max(1, min(int(max_results), MAX_RESULTS_CAP))

    async def _run(owned_fetch: Fetcher) -> dict[str, Any]:
        tasks = []
        wanted = ["arxiv", "openalex"] if source == "both" else [source]
        if "arxiv" in wanted:
            tasks.append(
                search_arxiv(
                    owned_fetch,
                    query,
                    max_results=capped,
                    include_abstract=include_abstract,
                )
            )
        if "openalex" in wanted:
            tasks.append(
                search_openalex(
                    owned_fetch,
                    query,
                    max_results=capped,
                    year_from=year_from,
                    year_to=year_to,
                    include_abstract=include_abstract,
                )
            )
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        per_source: list[list[dict[str, Any]]] = []
        warnings: list[str] = []
        for name, res in zip(wanted, gathered):
            if isinstance(res, BaseException):
                warnings.append(f"{name}: {type(res).__name__}: {res}")
            else:
                per_source.append(res)
        # Round-robin interleave so a multi-source search blends results
        # (arxiv[0], openalex[0], arxiv[1], …) instead of front-loading one
        # source; dedup then preserves the first occurrence.
        depth = max((len(s) for s in per_source), default=0)
        interleaved: list[dict[str, Any]] = [
            s[i] for i in range(depth) for s in per_source if i < len(s)
        ]
        merged = dedup_merge(interleaved)[:capped]
        return {
            "count": len(merged),
            "papers": merged,
            "sources": wanted,
            "warnings": warnings,
        }

    if fetch is not None:
        return await _run(fetch)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        return await _run(_httpx_fetcher(client))


async def resolve_identifier(
    identifier: str,
    *,
    include_abstract: bool = True,
    fetch: Fetcher | None = None,
) -> dict[str, Any]:
    """Resolve a single DOI or arXiv id to one curated record (resolve-before-cite).

    Returns ``{"found": bool, "paper": {...} | None, "warnings": [...]}``.
    """
    identifier = (identifier or "").strip()
    if not identifier:
        raise ValueError("identifier is required (a DOI or an arXiv id).")
    arxiv_id = normalize_arxiv_id(identifier)
    doi = normalize_doi(identifier)

    async def _run(owned_fetch: Fetcher) -> dict[str, Any]:
        papers: list[dict[str, Any]] = []
        warnings: list[str] = []
        if doi:
            try:
                papers = await search_openalex(
                    owned_fetch,
                    "",
                    max_results=1,
                    include_abstract=include_abstract,
                    extra_filter=f"doi:{doi}",
                )
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                warnings.append(f"openalex: {type(exc).__name__}: {exc}")
        elif arxiv_id:
            try:
                papers = await search_arxiv(
                    owned_fetch,
                    "",
                    max_results=1,
                    include_abstract=include_abstract,
                    id_list=arxiv_id,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"arxiv: {type(exc).__name__}: {exc}")
        else:
            raise ValueError(
                f"could not parse {identifier!r} as a DOI (10.x/...) or arXiv id (YYMM.NNNNN)."
            )
        return {
            "found": bool(papers),
            "paper": papers[0] if papers else None,
            "warnings": warnings,
        }

    if fetch is not None:
        return await _run(fetch)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        return await _run(_httpx_fetcher(client))
