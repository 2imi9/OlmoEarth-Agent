# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the olmoearth_negative_sampler tool, including a round-trip that
runs the emitted GeoJSON through the vendored data-prep audit to prove the
presence-only hard FAIL becomes a PASS."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.tools.negative_sampler import (
    NEGATIVE_CLASS_NAMES,
    build_negative_sampler_tools,
)
from olmoearth_agent.tools.registry import ToolContext

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_PATH = (
    _REPO_ROOT / "vendor/olmoearth-skills/skills/olmoearth-data-prep/scripts/audit.py"
)


def _load_audit() -> ModuleType:
    """Import the vendored stdlib-only audit.py by file path."""
    spec = importlib.util.spec_from_file_location("dataprep_audit", _AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _confine_io_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the path-traversal workspace root at this test's tmp_path so the
    tool's absolute tmp_path read/write paths resolve inside the workspace."""
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))


def _pt(lon: float, lat: float, field: str, label: str) -> dict[str, Any]:
    if field == "es_label":
        props: dict[str, Any] = {"es_label": label}
    elif field == "oe_labels.category":
        props = {"oe_labels": {"category": label}}
    else:
        props = {"sample_category": label}
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def _presence_only(
    n_cols: int = 10,
    n_rows: int = 6,
    field: str = "sample_category",
    label: str = "karst",
) -> dict[str, Any]:
    """A presence-only FeatureCollection: a spread grid, one positive class."""
    feats = []
    for r in range(n_rows):
        for c in range(n_cols):
            lon = 0.1 + 0.8 * c / (n_cols - 1)
            lat = 0.1 + 0.8 * r / (n_rows - 1)
            feats.append(_pt(lon, lat, field, label))
    return {"type": "FeatureCollection", "features": feats}


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc), encoding="utf-8")


def _run_full_audit(
    audit: ModuleType, features: list[dict[str, Any]]
) -> dict[str, str]:
    """Return {check_name: status} for every audit check (PASS/WARN/FAIL/ERROR)."""
    out = {}
    for name, fn in audit.CHECKS:
        try:
            status, _ = fn(features)
        except Exception as exc:  # noqa: BLE001
            status = f"ERROR:{exc}"
        out[name] = status
    return out


def test_negative_class_names_match_vendored_audit() -> None:
    """The skill must label negatives with names the audit recognizes."""
    audit = _load_audit()
    assert NEGATIVE_CLASS_NAMES == frozenset(audit.NEGATIVE_CLASS_NAMES)


@pytest.mark.asyncio
async def test_round_trip_presence_only_failure_becomes_pass(tmp_path: Path) -> None:
    audit = _load_audit()
    src = _presence_only()  # 60 positives, single class
    src_path = tmp_path / "labels.geojson"
    _write(src_path, src)

    # Before: the presence-only set hard-FAILs the negative-class check.
    before = _run_full_audit(audit, src["features"])
    assert before["Negative class"] == "FAIL"

    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())

    out_path = Path(result["out_path"])
    assert out_path.exists()
    combined = json.loads(out_path.read_text(encoding="utf-8"))["features"]
    assert len(combined) == 120  # 60 positives + 60 generated negatives
    assert result["n_negatives"] == 60
    assert result["schema_field"] == "sample_category"

    # After: the negative-class check passes, and the whole audit is FAIL-free.
    after = _run_full_audit(audit, combined)
    assert after["Negative class"] == "PASS"
    assert all(not s.startswith(("FAIL", "ERROR")) for s in after.values())


@pytest.mark.asyncio
async def test_result_has_no_raw_coordinates(tmp_path: Path) -> None:
    """Rule §3.1: the chat result returns counts + a path, never coordinates."""
    src_path = tmp_path / "labels.geojson"
    _write(src_path, _presence_only())
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())
    assert "negatives" not in result  # the coordinate list is written to disk only
    assert "out_path" in result and "n_negatives" in result


@pytest.mark.asyncio
async def test_default_out_path_is_suffixed(tmp_path: Path) -> None:
    src_path = tmp_path / "mylabels.geojson"
    _write(src_path, _presence_only())
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())
    assert result["out_path"].endswith("mylabels_with_negatives.geojson")


@pytest.mark.asyncio
async def test_es_label_schema_round_trips(tmp_path: Path) -> None:
    audit = _load_audit()
    src = _presence_only(field="es_label", label="flood")
    src_path = tmp_path / "labels.geojson"
    _write(src_path, src)
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler(
        {"positives_path": str(src_path), "negative_label": "no_event"}, _ctx()
    )
    assert result["schema_field"] == "es_label"
    combined = json.loads(Path(result["out_path"]).read_text(encoding="utf-8"))[
        "features"
    ]
    negs = [f for f in combined if (f["properties"].get("es_label")) == "no_event"]
    assert len(negs) == 60
    after = _run_full_audit(audit, combined)
    assert after["Negative class"] == "PASS"


@pytest.mark.asyncio
async def test_oe_labels_schema_round_trips(tmp_path: Path) -> None:
    audit = _load_audit()
    src = _presence_only(field="oe_labels.category", label="mine")
    src_path = tmp_path / "labels.geojson"
    _write(src_path, src)
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())
    assert result["schema_field"] == "oe_labels.category"
    combined = json.loads(Path(result["out_path"]).read_text(encoding="utf-8"))[
        "features"
    ]
    negs = [
        f
        for f in combined
        if isinstance(f["properties"].get("oe_labels"), dict)
        and f["properties"]["oe_labels"].get("category") == "background"
    ]
    assert len(negs) == 60
    after = _run_full_audit(audit, combined)
    assert after["Negative class"] == "PASS"


@pytest.mark.asyncio
async def test_unrecognized_negative_label_rejected(tmp_path: Path) -> None:
    src_path = tmp_path / "labels.geojson"
    _write(src_path, _presence_only())
    tool = build_negative_sampler_tools()[0]
    with pytest.raises(ValueError, match="not a recognized negative class"):
        await tool.handler(
            {"positives_path": str(src_path), "negative_label": "karst_absent"}, _ctx()
        )


@pytest.mark.asyncio
async def test_no_label_field_rejected(tmp_path: Path) -> None:
    src_path = tmp_path / "labels.geojson"
    _write(
        src_path,
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"name": "unlabeled"},
                }
            ],
        },
    )
    tool = build_negative_sampler_tools()[0]
    with pytest.raises(ValueError, match="recognized label field"):
        await tool.handler({"positives_path": str(src_path)}, _ctx())


@pytest.mark.asyncio
async def test_polygon_positives_use_centroid(tmp_path: Path) -> None:
    """Positives can be polygons; the buffer uses each polygon's centroid."""
    poly = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.0], [0.0, 0.0]]
            ],
        },
        "properties": {"sample_category": "karst"},
    }
    src_path = tmp_path / "polys.geojson"
    _write(src_path, {"type": "FeatureCollection", "features": [poly] * 12})
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())
    assert result["n_negatives"] == 12


@pytest.mark.asyncio
async def test_embedding_ranking_from_candidates_file(tmp_path: Path) -> None:
    """Positives + candidates with embeddings -> dissimilarity-ranked negatives."""
    # Positives clustered with embedding [1, 0].
    pos_feats = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "properties": {"sample_category": "karst", "embedding": [1.0, 0.0]},
        }
        for _ in range(3)
    ]
    # Two far-apart candidates: one similar ([1,0]), one dissimilar ([-1,0]).
    cand = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30.0, 30.0]},
                "properties": {"embedding": [1.0, 0.0]},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-30.0, -30.0]},
                "properties": {"embedding": [-1.0, 0.0]},
            },
        ],
    }
    pos_path = tmp_path / "pos.geojson"
    cand_path = tmp_path / "cand.geojson"
    _write(pos_path, {"type": "FeatureCollection", "features": pos_feats})
    _write(cand_path, cand)

    tool = build_negative_sampler_tools()[0]
    result = await tool.handler(
        {
            "positives_path": str(pos_path),
            "candidates_path": str(cand_path),
            "n_negatives": 1,
            "min_separation_km": 0.0,
        },
        _ctx(),
    )
    assert result["ranking"] == "embedding_dissimilarity"
    combined = json.loads(Path(result["out_path"]).read_text(encoding="utf-8"))[
        "features"
    ]
    negs = [
        f for f in combined if f["properties"].get("sample_category") == "background"
    ]
    assert len(negs) == 1
    # The dissimilar candidate ([-1,0] @ -30,-30) must be the one chosen.
    assert negs[0]["geometry"]["coordinates"] == [-30.0, -30.0]


@pytest.mark.asyncio
async def test_result_surfaces_quality_report(tmp_path: Path) -> None:
    src_path = tmp_path / "labels.geojson"
    _write(src_path, _presence_only())
    tool = build_negative_sampler_tools()[0]
    result = await tool.handler({"positives_path": str(src_path)}, _ctx())
    assert "n_contamination_excluded" in result
    quality = result["quality"]
    assert "min_buffer_km" in quality and "mean_buffer_km" in quality
    assert "note" in quality
    assert "negatives" not in result  # quality is summary-only, still no raw coords


@pytest.mark.asyncio
async def test_rejects_read_path_traversal(tmp_path: Path) -> None:
    # A model-controlled positives_path escaping the workspace root is refused
    # before any file is opened (closes the arbitrary-read half of the finding).
    from olmoearth_agent.security.paths import PathTraversalError

    tool = build_negative_sampler_tools()[0]
    with pytest.raises(PathTraversalError):
        await tool.handler({"positives_path": "../../etc/passwd"}, _ctx())


@pytest.mark.asyncio
async def test_rejects_out_path_traversal(tmp_path: Path) -> None:
    # A valid input but an escaping out_path is refused (arbitrary-write half).
    from olmoearth_agent.security.paths import PathTraversalError

    src_path = tmp_path / "labels.geojson"
    _write(src_path, _presence_only())
    tool = build_negative_sampler_tools()[0]
    with pytest.raises(PathTraversalError):
        await tool.handler(
            {"positives_path": str(src_path), "out_path": "../escaped.geojson"}, _ctx()
        )
