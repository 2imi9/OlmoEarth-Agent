# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-litsearch`` tool bundle (skill #15).

Literature search + identifier resolution over arXiv and OpenAlex so the agent
can GROUND EO/geospatial claims and citations in real papers instead of
world-knowledge or hallucinated links. One tool, ``olmoearth_litsearch``:

* given a ``query``, it searches arXiv and/or OpenAlex, deduped across
  sources, returning curated records (each with a real ``url`` to cite);
* given an ``identifier`` (a DOI or an arXiv id), it resolves that one record
  instead (the resolve-before-cite primitive), from the same sources and in
  the same record shape.

Public + key-free (OpenAlex uses the documented polite-pool ``mailto`` when
``OLMOEARTH_OPENALEX_MAILTO`` is set). Returns bibliographic metadata only --
never full text/PDF bytes and never raw geometry -- so it is provenance-safe.

Optional upgrade: ``source="asta"`` routes to Ai2's Asta CLI (full-text ranked
search with relevance judgements + snippets) when the operator has installed
and authenticated it -- see :mod:`olmoearth_agent.analysis.asta`. Detection-
gated; when absent the tool answers with install guidance and the key-free
sources keep working.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis import asta
from olmoearth_agent.analysis.litsearch import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    resolve_identifier,
    search_literature,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext

# Behavioral rules echoed into the description (the ToolSpec.description is the
# only routing + guardrail surface for an implemented tool -- no SKILL.md gate).
_RULES = (
    " Never invent DOIs, arXiv ids, titles, or authors; if nothing matches, "
    "report zero results. When you use a returned paper, cite its `url`. "
    "Results accumulate in your context: budget at most 2-3 searches per "
    "task, reuse records you already retrieved instead of re-querying, and "
    "synthesize from what you have rather than broadening the search."
)


async def _litsearch(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Search by ``query``, or resolve one DOI / arXiv ``identifier``."""
    identifier = str(args.get("identifier") or "").strip()
    if identifier:
        return await resolve_identifier(
            identifier=identifier,
            include_abstract=bool(args.get("include_abstract", True)),
        )
    if not str(args.get("query") or "").strip():
        raise ValueError(
            "pass a non-empty query to search, or an identifier (a DOI or an "
            "arXiv id) to resolve one paper."
        )
    source = str(args.get("source", "both"))
    if source == "asta":
        # Optional full-text backend (Ai2's Asta CLI). Unavailability is a
        # structured answer with a working fallback, never a dead end.
        if not asta.asta_available():
            return {
                "available": False,
                "error": (
                    "source='asta' needs the Asta CLI, which is not installed "
                    "on this machine."
                ),
                "hint": (
                    "Retry with source='both' (arXiv + OpenAlex, always "
                    "available). To enable asta: install "
                    "github.com/allenai/asta-plugins and run 'asta auth login'."
                ),
            }
        return await asta.search_asta(
            query=str(args.get("query", "")),
            max_results=int(args.get("max_results", DEFAULT_MAX_RESULTS)),
            include_abstract=bool(args.get("include_abstract", False)),
        )
    return await search_literature(
        query=str(args.get("query", "")),
        source=source,
        max_results=int(args.get("max_results", DEFAULT_MAX_RESULTS)),
        year_from=args.get("year_from"),
        year_to=args.get("year_to"),
        include_abstract=bool(args.get("include_abstract", False)),
    )


def build_litsearch_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-litsearch`` tool bundle (search, or resolve an id)."""
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
                    "Key-free; read-only. source='asta' (only if installed) upgrades "
                    "to Ai2's full-text ranked search with relevance judgements and "
                    "supporting snippets — prefer it for grounding a specific claim "
                    "when available; if it reports itself unavailable, fall back to "
                    "'both'. Given an identifier (a DOI or an arXiv id) instead of "
                    "a query, it resolves that one paper ({found, paper}): use it "
                    "to verify a citation before relying on it." + _RULES
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text search (title/abstract/author terms).",
                        },
                        "identifier": {
                            "type": "string",
                            "description": "A DOI (10.xxxx/...) or arXiv id (e.g. "
                            "2511.13655) to resolve instead of searching.",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["arxiv", "openalex", "both", "asta"],
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
                            "description": "Include truncated abstracts (default: "
                            "off for a search, on for an identifier).",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_litsearch,
            capability=Capability(
                does="search arXiv and OpenAlex for papers, or resolve one DOI or "
                "arXiv id"
            ),
        ),
    ]
