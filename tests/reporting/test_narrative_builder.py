# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the case-narrative builder and freshness gate."""

from __future__ import annotations

from datetime import datetime, timezone

from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.reporting.narrative import build_narrative, freshness_check

NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_freshness_fresh() -> None:
    r = freshness_check("2026-05-28T06:00:00Z", window_hours=24, now=NOW)
    assert r["fresh"] is True
    assert r["age_hours"] == 6.0


def test_freshness_stale() -> None:
    r = freshness_check("2026-05-26T12:00:00Z", window_hours=24, now=NOW)
    assert r["fresh"] is False
    assert r["age_hours"] == 48.0


def test_freshness_unparseable() -> None:
    r = freshness_check("not-a-date", now=NOW)
    assert r["fresh"] is None
    assert "error" in r


def test_build_narrative_renders_fresh_gates_stale() -> None:
    results = [
        {
            "result_id": "fresh1",
            "tile_urls": ["/t/fresh"],
            "property_names": ["score"],
            "creation_time": "2026-05-28T11:00:00Z",
        },
        {
            "result_id": "stale1",
            "tile_urls": ["/t/stale"],
            "property_names": ["score"],
            "creation_time": "2026-05-25T12:00:00Z",
        },
    ]
    out = build_narrative("Klamath alfalfa", results, window_hours=24, now=NOW)
    md = out["markdown"]
    assert out["result_count"] == 2
    assert out["stale_count"] == 1
    assert out["gated"] is True
    assert "# Klamath alfalfa" in md
    assert "/t/fresh" in md  # fresh tile rendered
    assert "/t/stale" not in md  # stale tile withheld
    assert "~~stale1~~" in md  # stale result struck through
    assert "Freshness gate" in md


def test_build_narrative_with_provenance_log() -> None:
    log = ProvenanceLog(run_id="r1")
    log.record_tool_call(
        "olmoearth_load_context", {}, {"ok": True, "result": {"project_count": 5}}
    )
    out = build_narrative("Run", [], provenance=log, now=NOW)
    assert "## Provenance" in out["markdown"]
    assert "olmoearth_load_context" in out["markdown"]


def test_build_narrative_no_stale_not_gated() -> None:
    results = [
        {
            "result_id": "r1",
            "tile_urls": ["/t"],
            "property_names": [],
            "creation_time": "2026-05-28T11:30:00Z",
        }
    ]
    out = build_narrative("T", results, window_hours=24, now=NOW)
    assert out["gated"] is False
    assert "Freshness gate" not in out["markdown"]
