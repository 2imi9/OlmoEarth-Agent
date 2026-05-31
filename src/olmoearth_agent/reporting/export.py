# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Group + serialize Studio data for easy export (skill #13, reframed).

Originally "external-data" (wire third-party MCPs); reframed to the more
useful + self-contained "export our own Studio data, grouped": projects
and predictions written to JSON files grouped by project or status. Pure
helpers here; the fetching/writing tool is in ``tools/export.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any


def slugify(name: str | None) -> str:
    """Filesystem-safe slug for a group/file name."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "untitled"


def group_items(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group a list of records by ``record[key]`` (stringified; missing -> 'unknown')."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(key, "unknown")), []).append(item)
    return grouped


def to_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, 2-space indent)."""
    return json.dumps(obj, indent=2, default=str, sort_keys=True)


def curate(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Keep only ``fields`` present in ``record`` (drops bulky/sensitive extras)."""
    return {k: record.get(k) for k in fields if k in record}
