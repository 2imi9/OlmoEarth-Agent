# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the literature-search logic (parse / dedup / async fetch)."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.analysis.litsearch import (
    ARXIV_API,
    OPENALEX_WORKS,
    _guarded_get,
    dedup_merge,
    normalize_arxiv_id,
    normalize_doi,
    parse_arxiv_atom,
    parse_openalex_works,
    resolve_identifier,
    search_literature,
)
from olmoearth_agent.security.egress import EgressError

_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2511.13655v1</id>
    <title>OlmoEarth: an Earth-observation foundation model</title>
    <summary>We present OlmoEarth, a remote-sensing foundation model.</summary>
    <published>2025-11-17T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2511.13655v1"/>
    <arxiv:doi>10.1234/Example.2025</arxiv:doi>
    <arxiv:journal_ref>Remote Sensing of Environment 2025</arxiv:journal_ref>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>Spatial cross-validation for EO</title>
    <summary>A method paper.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Ann Other</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v2"/>
  </entry>
</feed>
"""

_OPENALEX = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "display_name": "OlmoEarth: an Earth-observation foundation model",
            "publication_year": 2025,
            "doi": "https://doi.org/10.1234/example.2025",
            "cited_by_count": 42,
            "authorships": [{"author": {"display_name": "Jane Doe"}}],
            "primary_location": {
                "source": {"display_name": "Remote Sensing of Environment"},
                "landing_page_url": "https://example.org/olmoearth",
            },
            "abstract_inverted_index": {"We": [0], "present": [1], "OlmoEarth.": [2]},
        }
    ]
}


def _fetch_map(mapping: dict[str, tuple[int, str]]):
    async def fetch(url: str, params: dict) -> tuple[int, str]:
        return mapping[url]

    return fetch


def test_parse_arxiv_atom_extracts_fields_once() -> None:
    papers = parse_arxiv_atom(_ARXIV_XML)
    assert len(papers) == 2  # one record per entry (no per-entry reprint bug)
    first = papers[0]
    assert first["arxiv_id"] == "2511.13655"
    assert first["id"] == "arxiv:2511.13655"
    assert first["url"] == "https://arxiv.org/abs/2511.13655"
    assert first["authors"] == ["Jane Doe", "John Roe"]
    assert first["year"] == 2025
    assert first["doi"] == "10.1234/example.2025"  # external DOI, lowercased
    assert first["venue"] == "Remote Sensing of Environment 2025"
    assert first["pdf_url"].endswith("2511.13655v1")
    assert papers[1]["venue"] == "arXiv"  # no journal_ref -> default


def test_parse_openalex_reconstructs_abstract_and_doi() -> None:
    papers = parse_openalex_works(_OPENALEX)
    assert len(papers) == 1
    p = papers[0]
    assert p["doi"] == "10.1234/example.2025"
    assert p["id"] == "doi:10.1234/example.2025"
    assert p["cited_by_count"] == 42
    assert p["url"] == "https://doi.org/10.1234/example.2025"
    assert p["abstract"] == "We present OlmoEarth."


def test_normalizers() -> None:
    assert normalize_arxiv_id("http://arxiv.org/abs/2511.13655v1") == "2511.13655"
    assert normalize_arxiv_id("arXiv:2401.00001") == "2401.00001"
    assert normalize_arxiv_id("not-an-id") is None
    assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert normalize_doi("doi:10.1000/xyz") == "10.1000/xyz"
    assert normalize_doi("nope") is None


def test_dedup_merges_shared_doi_across_sources() -> None:
    arxiv = parse_arxiv_atom(_ARXIV_XML)
    openalex = parse_openalex_works(_OPENALEX)
    merged = dedup_merge(arxiv + openalex)
    # 2 arxiv + 1 openalex, but the OlmoEarth paper shares a DOI -> 2 records
    assert len(merged) == 2
    olmo = next(p for p in merged if p["arxiv_id"] == "2511.13655")
    assert olmo["cited_by_count"] == 42  # filled from the OpenAlex twin
    assert "arxiv" in olmo["source"] and "openalex" in olmo["source"]


@pytest.mark.asyncio
async def test_search_literature_both_dedups_and_caps() -> None:
    fetch = _fetch_map(
        {ARXIV_API: (200, _ARXIV_XML), OPENALEX_WORKS: (200, json.dumps(_OPENALEX))}
    )
    result = await search_literature(
        "olmoearth", source="both", max_results=10, fetch=fetch
    )
    assert result["count"] == 2
    assert set(result["sources"]) == {"arxiv", "openalex"}
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_search_literature_non_200_is_empty_not_crash() -> None:
    fetch = _fetch_map({ARXIV_API: (503, ""), OPENALEX_WORKS: (500, "")})
    result = await search_literature("anything", source="both", fetch=fetch)
    assert result["count"] == 0
    assert result["papers"] == []


@pytest.mark.asyncio
async def test_search_literature_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query is required"):
        await search_literature("   ", fetch=_fetch_map({}))


@pytest.mark.asyncio
async def test_resolve_doi_routes_to_openalex() -> None:
    fetch = _fetch_map({OPENALEX_WORKS: (200, json.dumps(_OPENALEX))})
    out = await resolve_identifier("10.1234/example.2025", fetch=fetch)
    assert out["found"] is True
    assert out["paper"]["doi"] == "10.1234/example.2025"


@pytest.mark.asyncio
async def test_resolve_arxiv_routes_to_arxiv() -> None:
    fetch = _fetch_map({ARXIV_API: (200, _ARXIV_XML)})
    out = await resolve_identifier("2511.13655", fetch=fetch)
    assert out["found"] is True
    assert out["paper"]["arxiv_id"] == "2511.13655"


@pytest.mark.asyncio
async def test_resolve_unparseable_raises() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        await resolve_identifier("definitely-not-an-id", fetch=_fetch_map({}))


@pytest.mark.asyncio
async def test_guarded_get_blocks_redirect_to_internal_host_in_enforce(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An allowlisted host that 302-redirects to a cloud-metadata/internal IP is
    # re-validated and blocked in enforce mode (closes the redirect SSRF gap).
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    httpx_mock.add_response(
        url=ARXIV_API,
        status_code=302,
        headers={"Location": "http://169.254.169.254/latest/meta-data/"},
    )
    async with httpx.AsyncClient(follow_redirects=False) as client:
        with pytest.raises(EgressError):
            await _guarded_get(client, ARXIV_API, {}, "litsearch")


@pytest.mark.asyncio
async def test_guarded_get_follows_redirect_to_allowlisted_host(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A redirect to another allowlisted litsearch host passes re-validation and
    # is followed.
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    httpx_mock.add_response(
        url=ARXIV_API,
        status_code=302,
        headers={"Location": "https://api.openalex.org/works"},
    )
    httpx_mock.add_response(url="https://api.openalex.org/works", text="ok")
    async with httpx.AsyncClient(follow_redirects=False) as client:
        resp = await _guarded_get(client, ARXIV_API, {}, "litsearch")
    assert resp.status_code == 200
    assert resp.text == "ok"
