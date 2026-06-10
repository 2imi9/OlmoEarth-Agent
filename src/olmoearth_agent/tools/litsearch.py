# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-litsearch`` tool bundle (skill #15).

Literature search + identifier resolution over arXiv and OpenAlex so the agent
can GROUND EO/geospatial claims and citations in real papers instead of
world-knowledge or hallucinated links. Two tools:

* ``olmoearth_litsearch`` -- search arXiv and/or OpenAlex, deduped across
  sources, returning curated records (each with a real ``url`` to cite).
* ``olmoearth_litsearch_resolve`` -- resolve one DOI or arXiv id to a single
  record (the resolve-before-cite primitive).

Public + key-free (OpenAlex uses the documented polite-pool ``mailto`` when
``OLMOEARTH_OPENALEX_MAILTO`` is set). Returns bibliographic metadata only --
never full text/PDF bytes and never raw geometry -- so it is provenance-safe.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.litsearch import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    resolve_identifier,
    search_literature,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

# Behavioral rules echoed into both descriptions (the ToolSpec.description is the
# only routing + guardrail surface for an implemented tool -- no SKILL.md gate).
_RULES = (
    " Never invent DOIs, arXiv ids, titles, or authors; if nothing matches, "
    "report zero results. When you use a returned paper, cite its `url`. "
    "Results accumulate in your context: budget at most 2-3 searches per "
    "task, reuse records you already retrieved instead of re-querying, and "
    "synthesize from what you have rather than broadening the search."
)


async def _litsearch(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return await search_literature(
        query=str(args.get("query", "")),
        source=str(args.get("source", "both")),
        max_results=int(args.get("max_results", DEFAULT_MAX_RESULTS)),
        year_from=args.get("year_from"),
        year_to=args.get("year_to"),
        include_abstract=bool(args.get("include_abstract", False)),
    )


async def _litsearch_resolve(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return await resolve_identifier(
        identifier=str(args.get("identifier", "")),
        include_abstract=bool(args.get("include_abstract", True)),
    )


def build_litsearch_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-litsearch`` tool bundle (search + resolve)."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_litsearch",
                description=(
                    "Search the scholarly literature (arXiv + OpenAlex) for "
                    "papers: find research papers/preprints, look up a paper by "
                    "topic, list an author's recent work, get citation counts, or "
                    "back a claim with a real citation. Use when the user asks to "
                    "find/cite a paper, locate the source for a method, or check "
                    "the literature (e.g. spatial cross-validation, cloud masking, "
                    "OlmoEarth/AlphaEarth embeddings, WorldCereal). Returns curated "
                    "records (id, title, authors, year, venue, doi, arxiv_id, url, "
                    "cited_by_count; abstract optional) deduped across both sources. "
                    "Key-free; read-only." + _RULES
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text search (title/abstract/author terms).",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["arxiv", "openalex", "both"],
                            "default": "both",
                        },
                        "max_results": {
                            "type": "integer",
                            "default": DEFAULT_MAX_RESULTS,
                            "maximum": MAX_RESULTS_CAP,
                            "description": f"Cap on records (hard ceiling {MAX_RESULTS_CAP}).",
                        },
                        "year_from": {
                            "type": "integer",
                            "description": "Earliest publication year (OpenAlex).",
                        },
                        "year_to": {
                            "type": "integer",
                            "description": "Latest publication year (OpenAlex).",
                        },
                        "include_abstract": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include truncated abstracts (off by default to save context).",
                        },
                    },
                    "required": ["query"],
                },
            ),
            handler=_litsearch,
        ),
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_litsearch_resolve",
                description=(
                    "Resolve a single identifier -- a DOI (10.xxxx/...) or an "
                    "arXiv id (YYMM.NNNNN) -- to one curated paper record (title, "
                    "authors, year, venue, url, citation count). Use this to "
                    "verify a citation before relying on it, or to expand a bare "
                    "DOI/arXiv id the user gave you. Key-free; read-only." + _RULES
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "identifier": {
                            "type": "string",
                            "description": "A DOI (10.xxxx/...) or arXiv id (e.g. 2511.13655).",
                        },
                        "include_abstract": {"type": "boolean", "default": True},
                    },
                    "required": ["identifier"],
                },
            ),
            handler=_litsearch_resolve,
        ),
    ]
