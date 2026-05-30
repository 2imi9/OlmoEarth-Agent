# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Stakeholder narrative with a freshness gate (skill #15).

Assembles a Markdown report from prediction results + the run's
provenance, and **refuses to render stale tiles** past a configurable
window — so a disaster-response brief never shows imagery that has since
gone out of date. Pure functions; no I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_iso(timestamp: str) -> datetime:
    """Parse an ISO-8601 string, tolerating a trailing ``Z``."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def freshness_check(
    creation_time: str,
    window_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify one result as fresh or stale relative to ``window_hours``.

    Returns ``fresh`` (bool or ``None`` if the timestamp can't be parsed),
    ``age_hours``, and ``window_hours``.
    """
    now = now or datetime.now(timezone.utc)
    try:
        created = _parse_iso(creation_time)
    except (ValueError, AttributeError, TypeError):
        return {
            "creation_time": creation_time,
            "fresh": None,
            "age_hours": None,
            "window_hours": window_hours,
            "error": "unparseable timestamp",
        }
    age_hours = (now - created).total_seconds() / 3600.0
    return {
        "creation_time": creation_time,
        "age_hours": round(age_hours, 2),
        "fresh": age_hours <= window_hours,
        "window_hours": window_hours,
    }


def build_narrative(
    title: str,
    results: list[dict[str, Any]],
    *,
    provenance: Any = None,
    window_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a stakeholder Markdown report with a freshness gate.

    Parameters
    ----------
    title
        Report heading.
    results
        Each: ``result_id``, ``tile_urls``, ``property_names``,
        ``creation_time`` (ISO-8601), optional ``result_metadata``.
    provenance
        A ``ProvenanceLog`` (has ``.entries``) or a list of manifest
        dicts. Rendered as an audit section.
    window_hours
        Tiles older than this are gated (struck through, not rendered).

    Returns
    -------
    dict
        ``markdown`` (the report), ``result_count``, ``stale_count``,
        ``gated`` (bool), and per-result ``freshness``.
    """
    freshness = []
    stale_ids: list[str] = []
    for result in results:
        check = freshness_check(result.get("creation_time", ""), window_hours, now)
        entry = {"result_id": result.get("result_id"), **check}
        freshness.append(entry)
        if check.get("fresh") is False:
            stale_ids.append(str(result.get("result_id")))

    lines = [f"# {title}", ""]
    if stale_ids:
        lines.append(
            f"> **Freshness gate:** {len(stale_ids)} of {len(results)} "
            f"result(s) are older than {window_hours}h and were not "
            f"rendered. Re-run the prediction for current tiles."
        )
        lines.append("")

    lines.append("## Results")
    fresh_by_id = {f["result_id"]: f for f in freshness}
    for result in results:
        rid = result.get("result_id")
        check = fresh_by_id[rid]
        if check.get("fresh") is False:
            lines.append(
                f"- ~~{rid}~~ — stale ({check['age_hours']}h old), not rendered"
            )
            continue
        props = ", ".join(result.get("property_names") or []) or "(none)"
        lines.append(f"- **{rid}** — properties: {props}")
        lines.extend(f"  - tile: `{url}`" for url in result.get("tile_urls") or [])
    lines.append("")

    entries = getattr(provenance, "entries", provenance) or []
    if entries:
        lines.append("## Provenance")
        for entry in entries:
            call = getattr(entry, "api_call", None)
            req_hash = getattr(entry, "request_hash", None)
            if call is None and isinstance(entry, dict):
                call = entry.get("api_call")
                req_hash = entry.get("request_hash")
            lines.append(f"- `{call}` (args sha {str(req_hash)[:12]})")
        lines.append("")

    return {
        "markdown": "\n".join(lines).rstrip() + "\n",
        "result_count": len(results),
        "stale_count": len(stale_ids),
        "gated": bool(stale_ids),
        "freshness": freshness,
    }
