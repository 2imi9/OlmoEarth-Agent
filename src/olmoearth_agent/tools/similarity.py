# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-similarity`` tool bundle (skill #9).

Exact top-K similarity search over embedding vectors the caller already
has, with an optional geographic-prior warning. Returns summary results
only (no raw coordinates), per operational rule §3.1.
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.analysis.similarity import (
    DEFAULT_K,
    DEFAULT_PRIOR_RADIUS_KM,
    geographic_prior_check,
    similarity_search,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

_VECTOR = {"type": "array", "items": {"type": "number"}}
_COORD = {"type": "array", "items": {"type": "number"}}


async def _similarity_search(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    query = [float(x) for x in args["query"]]
    corpus = [[float(x) for x in v] for v in args["corpus"]]
    ids = args.get("ids")
    result = similarity_search(
        query,
        corpus,
        ids=[str(i) for i in ids] if ids is not None else None,
        k=int(args.get("k", DEFAULT_K)),
        metric=str(args.get("metric", "cosine")),
    )

    query_coord = args.get("query_coord")
    coords = args.get("coords")
    if query_coord is not None and coords is not None:
        if len(coords) != len(corpus):
            raise ValueError("coords length must match corpus length.")
        result_coords = [coords[m["index"]] for m in result["matches"]]
        result["geographic_prior"] = geographic_prior_check(
            [float(query_coord[0]), float(query_coord[1])],
            [[float(c[0]), float(c[1])] for c in result_coords],
            radius_km=float(args.get("prior_radius_km", DEFAULT_PRIOR_RADIUS_KM)),
        )
    return result


def build_similarity_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-similarity`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_similarity_search",
                description=(
                    "Find the top-K embedding vectors most similar to a "
                    "query (exact kNN; cosine or euclidean). Pass the query "
                    "vector and a corpus of candidate vectors (optionally "
                    "with parallel ids). If you also pass query_coord + "
                    "coords (parallel [lon, lat] for the corpus), it adds a "
                    "GEOGRAPHIC-PRIOR WARNING when the matches cluster near "
                    "the query — i.e. the similarity may reflect location "
                    "(same region/biome), not feature resemblance. "
                    "Read-only; returns matches + a summary (no raw coords)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": _VECTOR,
                        "corpus": {"type": "array", "items": _VECTOR},
                        "ids": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer", "default": DEFAULT_K},
                        "metric": {
                            "type": "string",
                            "enum": ["cosine", "euclidean"],
                            "default": "cosine",
                        },
                        "query_coord": _COORD,
                        "coords": {"type": "array", "items": _COORD},
                        "prior_radius_km": {
                            "type": "number",
                            "default": DEFAULT_PRIOR_RADIUS_KM,
                        },
                    },
                    "required": ["query", "corpus"],
                },
            ),
            handler=_similarity_search,
        ),
    ]
