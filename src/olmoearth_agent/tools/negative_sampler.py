# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-negative-sampler`` tool bundle (skill #18).

Turns a *presence-only* label GeoJSON (every feature is a positive sighting of
one class) into a **trainable** dataset by generating the missing
negative/background class as buffered, spatially-thinned pseudo-absences, then
writing a combined GeoJSON that round-trips straight back into the vendored
``olmoearth-data-prep`` audit (whose ``check_negative_class`` hard-FAILs a
presence-only set).

Coordinates flow file -> file: the tool reads a positives path, writes a
combined path, and returns only counts / the output path / warnings -- never raw
geometry in chat (operational rule §3.1). The negative class label defaults to a
name the audit recognizes. When the input features (and an optional candidates
file) carry per-feature ``properties.embedding`` vectors, negatives are ranked
by environmental dissimilarity to the positive centroid (the inverse of skill
#9); otherwise selection is farthest-point spatial dispersion.
"""

from __future__ import annotations

import json
import os
from typing import Any

from olmoearth_agent.analysis.negative_sampler import (
    DEFAULT_EXCLUSION_KM,
    DEFAULT_OVERSAMPLE,
    sample_negatives,
)
from olmoearth_agent.evaluation.spatial_cv import Point
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

#: Negative-class names the data-prep audit's ``check_negative_class`` accepts.
#: Mirrors ``NEGATIVE_CLASS_NAMES`` in the vendored ``audit.py`` (kept in sync; a
#: round-trip test asserts the two agree). Generating any other label would not
#: clear the audit, defeating the skill's purpose.
NEGATIVE_CLASS_NAMES = frozenset(
    {"other", "background", "negative", "stable", "none", "no_event", "non_event"}
)
DEFAULT_NEGATIVE_LABEL = "background"


def _features(doc: Any) -> list[dict[str, Any]]:
    """Pull the feature list from a FeatureCollection or a bare feature list."""
    if isinstance(doc, dict) and isinstance(doc.get("features"), list):
        return [f for f in doc["features"] if isinstance(f, dict)]
    if isinstance(doc, list):
        return [f for f in doc if isinstance(f, dict)]
    raise ValueError("input is not a GeoJSON FeatureCollection or feature list.")


def _iter_coords(geometry: Any) -> list[tuple[float, float]]:
    """Flatten any GeoJSON geometry to its ``(lon, lat)`` coordinate pairs."""
    if not isinstance(geometry, dict):
        return []
    coords = geometry.get("coordinates")
    out: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and all(isinstance(v, (int, float)) for v in node[:2])
        ):
            out.append((float(node[0]), float(node[1])))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(coords)
    return out


def _centroid(geometry: Any) -> tuple[float, float] | None:
    """Representative ``(lon, lat)`` for a feature: the point, else the centroid."""
    pairs = _iter_coords(geometry)
    if not pairs:
        return None
    if len(pairs) == 1:
        return pairs[0]
    return (
        sum(p[0] for p in pairs) / len(pairs),
        sum(p[1] for p in pairs) / len(pairs),
    )


def _label_field(feature: dict[str, Any]) -> str | None:
    """Which schema field carries this feature's label (priority order), if any."""
    props = feature.get("properties") or {}
    if props.get("sample_category") is not None:
        return "sample_category"
    if props.get("es_label") is not None:
        return "es_label"
    oe = props.get("oe_labels")
    if isinstance(oe, dict) and oe.get("category") is not None:
        return "oe_labels.category"
    return None


def _embedding(feature: dict[str, Any]) -> list[float] | None:
    """A numeric ``properties.embedding`` vector, or ``None``."""
    vec = (feature.get("properties") or {}).get("embedding")
    if isinstance(vec, list) and vec and all(isinstance(v, (int, float)) for v in vec):
        return [float(v) for v in vec]
    return None


def _negative_feature(lon: float, lat: float, field: str, label: str) -> dict[str, Any]:
    """Build a negative Point feature under the detected schema ``field``."""
    if field == "es_label":
        props: dict[str, Any] = {"es_label": label}
    elif field == "oe_labels.category":
        props = {"oe_labels": {"category": label}}
    else:  # sample_category (default / Studio import)
        props = {"sample_category": label}
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def _collect(
    features: list[dict[str, Any]],
) -> tuple[list[Point], list[list[float]] | None, str, dict[str, int]]:
    """Extract positive coords (+ embeddings if present) and the label schema."""
    coords: list[Point] = []
    embeddings: list[list[float]] = []
    fields: dict[str, int] = {}
    all_have_emb = True
    for feature in features:
        centroid = _centroid(feature.get("geometry"))
        if centroid is None:
            continue
        field = _label_field(feature)
        if field is None:
            continue
        fields[field] = fields.get(field, 0) + 1
        coords.append(centroid)
        emb = _embedding(feature)
        if emb is None:
            all_have_emb = False
        else:
            embeddings.append(emb)
    if not coords:
        raise ValueError(
            "no positive features with both a geometry and a recognized label "
            "field (sample_category / es_label / oe_labels.category) were found."
        )
    schema = max(fields, key=lambda k: fields[k])
    emb_aligned = (
        embeddings if all_have_emb and len(embeddings) == len(coords) else None
    )
    return coords, emb_aligned, schema, fields


async def _negative_sampler(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    positives_path = str(args["positives_path"])
    with open(positives_path, encoding="utf-8") as handle:
        pos_doc = json.load(handle)
    pos_features = _features(pos_doc)
    positives, pos_emb, schema, fields = _collect(pos_features)

    negative_label = str(args.get("negative_label", DEFAULT_NEGATIVE_LABEL))
    if negative_label.lower() not in NEGATIVE_CLASS_NAMES:
        raise ValueError(
            f"negative_label {negative_label!r} is not a recognized negative class; "
            f"the data-prep audit accepts: {sorted(NEGATIVE_CLASS_NAMES)}."
        )

    # Optional explicit candidate background locations from a second file.
    candidates: list[Point] | None = None
    cand_emb: list[list[float]] | None = None
    candidates_path = args.get("candidates_path")
    if candidates_path:
        with open(str(candidates_path), encoding="utf-8") as handle:
            cand_doc = json.load(handle)
        candidates = []
        cand_embs_raw: list[list[float] | None] = []
        for feature in _features(cand_doc):
            point = _centroid(feature.get("geometry"))
            if point is None:
                continue
            candidates.append(point)
            cand_embs_raw.append(_embedding(feature))
        if not candidates:
            raise ValueError("candidates file has no usable point geometries.")
        if pos_emb is not None and all(e is not None for e in cand_embs_raw):
            cand_emb = [e for e in cand_embs_raw if e is not None]

    result = sample_negatives(
        positives,
        candidates=candidates,
        n_negatives=args.get("n_negatives"),
        exclusion_km=float(args.get("exclusion_km", DEFAULT_EXCLUSION_KM)),
        min_separation_km=args.get("min_separation_km"),
        positive_embeddings=pos_emb if cand_emb is not None else None,
        candidate_embeddings=cand_emb,
        oversample=int(args.get("oversample", DEFAULT_OVERSAMPLE)),
        contamination_threshold=args.get("contamination_threshold"),
    )

    # Build the combined FeatureCollection; keep coordinates out of the chat
    # result (rule §3.1) by writing them to disk and returning only the path.
    negatives = result.pop("negatives")
    new_features = [
        _negative_feature(lon, lat, schema, negative_label) for lon, lat in negatives
    ]
    combined = {"type": "FeatureCollection", "features": [*pos_features, *new_features]}

    out_path = str(args.get("out_path") or _default_out_path(positives_path))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=2)

    result["out_path"] = out_path
    result["negative_label"] = negative_label
    result["schema_field"] = schema
    result["total_features"] = len(combined["features"])
    result["note"] = (
        "Combined positives + generated negatives written to out_path. Re-run the "
        "data-prep audit on it to confirm the negative-class check now passes."
    )
    return result


def _default_out_path(positives_path: str) -> str:
    root, ext = os.path.splitext(positives_path)
    return f"{root}_with_negatives{ext or '.geojson'}"


def build_negative_sampler_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-negative-sampler`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_negative_sampler",
                description=(
                    "Generate the missing NEGATIVE / background class for a "
                    "PRESENCE-ONLY label GeoJSON so it becomes trainable and "
                    "passes the data-prep audit (which hard-FAILs a label set "
                    "with no negative class). Reads positives_path (a GeoJSON of "
                    "positive labels using sample_category / es_label / "
                    "oe_labels.category), samples buffered + spatially-thinned "
                    "pseudo-absences (default count = number of positives, a "
                    "balanced 1:1 set), and WRITES a combined GeoJSON to out_path. "
                    "If the positives (and an optional candidates_path) carry "
                    "properties.embedding vectors, negatives are ranked by "
                    "environmental dissimilarity to the positives (inverts skill "
                    "#9), and with contamination_threshold set it also drops "
                    "candidates that resemble a positive too closely (likely "
                    "unmapped positives); otherwise selection is by spatial "
                    "dispersion. Returns counts + the output path + a quality "
                    "report (buffer-km and similarity-to-positive stats so you can "
                    "judge the negatives) + warnings only (no raw coordinates). "
                    "Tune exclusion_km (buffer around positives) and "
                    "min_separation_km (spacing between negatives) for your "
                    "sensor resolution."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "positives_path": {
                            "type": "string",
                            "description": "Path to the presence-only labels GeoJSON.",
                        },
                        "out_path": {
                            "type": "string",
                            "description": (
                                "Where to write the combined GeoJSON (default: "
                                "alongside the input, suffixed _with_negatives)."
                            ),
                        },
                        "candidates_path": {
                            "type": "string",
                            "description": (
                                "Optional GeoJSON of candidate background locations "
                                "(else a grid is generated over the positives' extent)."
                            ),
                        },
                        "negative_label": {
                            "type": "string",
                            "default": DEFAULT_NEGATIVE_LABEL,
                            "description": (
                                "Class name for the negatives; must be one the "
                                f"audit recognizes ({sorted(NEGATIVE_CLASS_NAMES)})."
                            ),
                        },
                        "n_negatives": {
                            "type": "integer",
                            "description": "Target negative count (default: #positives).",
                        },
                        "exclusion_km": {
                            "type": "number",
                            "default": DEFAULT_EXCLUSION_KM,
                            "description": "Drop candidates within this of any positive.",
                        },
                        "min_separation_km": {
                            "type": "number",
                            "description": (
                                "Minimum spacing between negatives (default: "
                                "exclusion_km; 0 disables thinning)."
                            ),
                        },
                        "oversample": {
                            "type": "integer",
                            "default": DEFAULT_OVERSAMPLE,
                            "description": "Candidate-grid oversampling factor.",
                        },
                        "contamination_threshold": {
                            "type": "number",
                            "description": (
                                "Cosine in [-1,1]. With embeddings, drop candidates "
                                "whose similarity to any positive is >= this (likely "
                                "unmapped positives) before ranking. Omit to disable; "
                                "check the result's quality.similarity_to_positive to "
                                "pick a value."
                            ),
                        },
                    },
                    "required": ["positives_path"],
                },
            ),
            handler=_negative_sampler,
        ),
    ]
