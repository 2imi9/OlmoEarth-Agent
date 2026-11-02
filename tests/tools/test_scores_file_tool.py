# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The scores provider, ``olmoearth_scores_from_file``, on stub model runs.

Each fixture is written the way the research record's
(``2imi9/olmoearth_inferenceX-record``) ``scripts/score_area.py`` writes a run
(and its ``tests/test_score_area.py`` builds one with a stub model): a
georeferenced ``(10, H, W)`` float32 raster, NaN where there is no prediction,
and a ``manifest.json`` naming the model, the classes (channel 9
``nodata_value_9``, the label fill value) and whether the values are logits.
Nothing here needs the cluster. The reference for every ranking is
``oe_inferencex.review.assess.assess_prediction`` on the same raster.

A run read by ``oe-inferencex from-olmoearth`` is made the way that command
makes it: an rslearn dataset written here with rasterio (each window's output
layer GeoTIFF and its ``completed`` marker, as ``rslearn model predict``
leaves them), read by the package's own reader
(``oe_inferencex.readers.olmoearth.read_output``) into ``scores_<EPSG>.tif``
and ``olmoearth_output.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import scores_file
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import build_review_set_tools
from olmoearth_agent.tools.scores_file import build_scores_file_tools, model_and_raster

try:
    import numpy as np
    import rasterio
    from oe_inferencex import cli
    from oe_inferencex.readers import olmoearth as oe_reader
    from oe_inferencex.review import assess
    from rasterio.transform import Affine

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

needs_extra = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

NAMES = [
    "woodland_forest",
    "open_water",
    "shrubland_savanna",
    "herbaceous_wetland",
    "grassland_barren",
    "agriculture_settlement",
    "montane_forest",
    "lava_forest",
    "urban_dense_development",
]
FILL = 9
LAT, LON = -2.55, 36.81
X0, Y0 = 256_330.0, 9_718_100.0  # UTM 37S, about (-2.55, 36.81)
_BANNED_KEYS = {
    "lon",
    "lat",
    "bbox",
    "bounds",
    "bounds_wgs84",
    "coordinates",
    "geometry",
    "centres_lon_lat",
    "transform",
    "crs",
    "area",
    "request",
}


def _ctx() -> ToolContext:
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_review_set_tools())
    registry.register_all(build_estimation_tools())
    return registry


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return await _registry().dispatch(
        ToolCall(id="1", name=name, arguments=args), _ctx()
    )


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    out = await _call(name, args)
    assert out["ok"] is True, out
    return out["result"]


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "scores_root"
    root.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logits(seed: int, h: int, w: int, fill_on_top: int = 0) -> Any:
    """Ten channels of stub logits whose fill channel is often the runner-up.

    Channel 9 sits just below the best real class at most pixels, so leaving
    it out of the margin changes the ranking; in the first ``fill_on_top``
    pixels of row 20 it is the arg-max, which the served map would show.
    """
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 1.5, (9, h, w))
    fill = real.max(0) - np.abs(rng.normal(0.0, 0.4, (h, w)))
    if fill_on_top:
        fill[20, :fill_on_top] = real.max(0)[20, :fill_on_top] + 0.5
    return np.concatenate([real, fill[None]]).astype(np.float32)


def _valid(h: int, w: int) -> Any:
    """No-data: an 8 x 8 corner (four whole windows) and a half-empty strip."""
    valid = np.ones((h, w), bool)
    valid[:8, :8] = False
    valid[8:12, 8:10] = False  # window (2, 2) keeps half its pixels: still valid
    valid[12:16, 8:11] = False  # window (3, 2) keeps a quarter: no-data
    return valid


def _write_run(
    root: Path,
    name: str,
    values: Any,
    valid: Any,
    *,
    kind: str = "logits",
    manifest_edit: dict[str, Any] | None = None,
) -> Path:
    """A run directory as score_area.py writes one: scores.tif + manifest.json."""
    run = root / name
    run.mkdir(parents=True)
    arr = np.where(valid[None], values, np.nan).astype(np.float32)
    c, h, w = arr.shape
    path = run / "scores.tif"
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "crs": "EPSG:32737",
        "transform": Affine(10.0, 0.0, X0, 0.0, -10.0, Y0),
        "nodata": float("nan"),
    }
    names = NAMES + [f"nodata_value_{FILL}"]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, n in enumerate(names[:c], 1):
            dst.set_band_description(i, n)
        dst.update_tags(values=kind)
    manifest: dict[str, Any] = {
        "what": "stub run",
        "scores": {
            "file": "scores.tif",
            "sha256": _sha256(path),
            "values": kind,
            "shape": [c, h, w],
            "dtype": "float32",
            "nodata": "NaN",
            "n_nodata_pixels": int((~valid).sum()),
        },
        "classes": {str(i): n for i, n in enumerate(names[:c])},
        "class_note": (
            "channel 9 is the label fill value (the task's nodata_value), never "
            "a training target; it is kept because the served argmax runs over "
            "every channel"
        ),
        "model": {
            "repo": "allenai/OlmoEarth-v1-FT-AWF-Base",
            "revision": "0" * 40,
            "revision_matches_record": True,
            "device": "stub",
        },
        "task_card": {"name": "awf", "task": {"nodata_value": FILL}},
        "area": {
            "request": {"lat": LAT, "lon": LON, "size_px": h},
            "crs": "EPSG:32737",
            "transform": [10.0, 0.0, X0, 0.0, -10.0, Y0],
            "bounds": [X0, Y0 - 10 * h, X0 + 10 * w, Y0],
            "bounds_wgs84": [LON - 0.003, LAT - 0.003, LON + 0.003, LAT + 0.003],
        },
        "date_window": {"start": "2023-01-01", "end": "2023-12-31"},
        "caveats": ["No labels were used."],
    }
    for key, value in (manifest_edit or {}).items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _read(run: Path) -> tuple[Any, Any]:
    """The raster and its valid mask, read independently of the tool."""
    with rasterio.open(run / "scores.tif") as src:
        a = src.read()
    return a, np.isfinite(a).all(axis=0)


def _package(scores: Any, valid: Any, is_logit: bool, budget: float = 0.05) -> Any:
    """``assess_prediction`` as ``oe-inferencex assess`` calls it."""
    return assess.assess_prediction(
        scores, is_logit=is_logit, patch=4, nodata_mask=~valid, budgets=(budget,)
    )


def _review_rowcols(result: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(r["row"]), int(r["col"])) for r in result["review"]]


def _assert_same_review(result: dict[str, Any], ref: Any, budget: float = 0.05):
    """The agent's review set is the package's: count, windows in order, margins."""
    rs = ref["review_sets"][budget]
    conf = ref["arrays"]["confidence"]
    expected = [tuple(int(v) for v in rc) for rc in rs["windows_rowcol"]]
    assert result["n_review"] == rs["n_windows"] == len(expected)
    margins = [conf[r, c] for r, c in expected]
    assert all(a < b for a, b in zip(margins, margins[1:])), "no ties at the cut"
    assert _review_rowcols(result) == expected
    for row in result["review"]:
        assert abs(row["margin"] - conf[row["row"], row["col"]]) <= 1e-6


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        out = set(obj)
        for v in obj.values():
            out |= _keys(v)
        return out
    if isinstance(obj, list):
        out = set()
        for v in obj:
            out |= _keys(v)
        return out
    return set()


# --------------------------------------------------------------------------- parity


@needs_extra
@pytest.mark.asyncio
async def test_logits_run_review_set_equals_the_package_without_the_fill_channel(
    _scores_root: Path,
) -> None:
    valid = _valid(64, 64)
    run = _write_run(_scores_root, "run_logits", _logits(0, 64, 64, 5), valid)
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run_logits"})

    a, read_valid = _read(run)
    assert (read_valid == valid).all()
    ref = _package(a[:FILL], read_valid, is_logit=True)
    assert got["grid"] == [16, 16] and got["patch_px"] == 4
    assert got["n_valid"] == ref["n_windows"] == 256 - 5
    assert got["n_nodata"] == 5 and got["n_windows"] == 256
    assert got["values"] == "logits"
    assert got["classes"] == {str(i): n for i, n in enumerate(NAMES)}
    assert got["fill_class"]["channel"] == FILL and got["fill_class"]["excluded"]
    assert got["fill_class"]["name"] == "nodata_value_9"
    assert "never a training target" in got["fill_class"]["note"]
    assert got["fill_class"]["valid_pixels_where_it_is_the_argmax"] == 5
    assert got["model"] == {
        "repo": "allenai/OlmoEarth-v1-FT-AWF-Base",
        "revision": "0" * 40,
        "revision_matches_record": True,
    }
    assert "not Studio point samples" in got["provenance"]
    assert got["raster_check"]["matches_manifest"] is True
    assert got["date_window"] == {"start": "2023-01-01", "end": "2023-12-31"}
    assert "top-1 minus top-2 logit margin" in got["signal"]

    review = await _ok(
        "olmoearth_review_set",
        {"scores_path": got["scores_path"], "budget": 0.05, "max_listed": 500},
    )
    _assert_same_review(review, ref)
    klass = ref["arrays"]["pooled_argmax"]
    for row in review["review"]:
        assert row["predicted_class"] == klass[row["row"], row["col"]]
        assert row["class_name"] == NAMES[row["predicted_class"]]
    assert review["score_type"] == "logit"
    assert review["signal"] == got["signal"]
    assert review["provenance"] == got["provenance"]

    # The whole population, not only the review set: every valid window's
    # margin and class are the package's.
    saved = json.loads(Path(got["scores_path"]).read_text(encoding="utf-8"))
    conf, valid_w = ref["arrays"]["confidence"], ref["arrays"]["valid"]
    assert saved["windows"] == np.flatnonzero(valid_w.ravel()).tolist()
    for w, row, k in zip(saved["windows"], saved["scores"], saved["map_class"]):
        r, c = divmod(w, 16)
        top = sorted(row, reverse=True)
        assert top[0] - top[1] == conf[r, c] and k == klass[r, c]


@needs_extra
@pytest.mark.asyncio
async def test_fill_channel_included_matches_the_served_map_and_changes_the_set(
    _scores_root: Path,
) -> None:
    valid = _valid(64, 64)
    run = _write_run(_scores_root, "run_fill", _logits(1, 64, 64, 5), valid)
    kept = await _ok(
        "olmoearth_scores_from_file",
        {"run_dir": "run_fill", "include_fill_class": True},
    )
    left = await _ok("olmoearth_scores_from_file", {"run_dir": "run_fill"})
    assert kept["scores_file"] != left["scores_file"]
    assert kept["fill_class"]["excluded"] is False
    assert len(kept["classes"]) == 10 and len(left["classes"]) == 9

    a, read_valid = _read(run)
    budget = 0.2
    both = [
        (kept, _package(a, read_valid, is_logit=True, budget=budget)),
        (left, _package(a[:FILL], read_valid, is_logit=True, budget=budget)),
    ]
    sets = []
    for got, ref in both:
        review = await _ok(
            "olmoearth_review_set",
            {"scores_path": got["scores_path"], "budget": budget, "max_listed": 500},
        )
        _assert_same_review(review, ref, budget)
        sets.append(set(_review_rowcols(review)))
    assert sets[0] != sets[1], "leaving the fill channel out changes the ranking"


@needs_extra
@pytest.mark.asyncio
async def test_probabilities_rank_by_the_packages_top1_probability(
    _scores_root: Path,
) -> None:
    logits = _logits(2, 48, 48).astype(np.float64)
    z = np.exp(logits - logits.max(0, keepdims=True))
    probs = (z / z.sum(0, keepdims=True)).astype(np.float32)
    valid = np.ones((48, 48), bool)
    valid[40:, 40:] = False
    run = _write_run(_scores_root, "run_probs", probs, valid, kind="probabilities")
    got = await _ok("olmoearth_scores_from_file", {"run_dir": str(run)})
    assert got["values"] == "probabilities"
    assert "top-1 probability" in got["signal"]
    assert got["n_nodata"] == 4

    a, read_valid = _read(run)
    ref = _package(a[:FILL], read_valid, is_logit=False)
    review = await _ok(
        "olmoearth_review_set", {"scores_path": got["scores_path"], "max_listed": 500}
    )
    _assert_same_review(review, ref)
    assert review["score_type"] == "probability"


@needs_extra
@pytest.mark.asyncio
async def test_the_label_plan_is_the_packages_sample_on_the_same_raster(
    _scores_root: Path, tmp_path: Path
) -> None:
    """With every channel kept, the agent's plan is `oe-inferencex sample`'s draw."""
    valid = _valid(64, 64)
    run = _write_run(_scores_root, "run_plan", _logits(3, 64, 64), valid)
    got = await _ok(
        "olmoearth_scores_from_file",
        {"run_dir": "run_plan", "include_fill_class": True},
    )
    plan = await _ok(
        "olmoearth_plan_label_sample",
        {"scores_path": got["scores_path"], "budget": 40, "seed": 3},
    )
    design = json.loads(Path(plan["design_path"]).read_text(encoding="utf-8"))

    sheet = tmp_path / "pkg" / "to_label.csv"
    assert (
        cli.main(
            [
                "sample",
                str(run / "scores.tif"),
                "--logits",
                "--budget",
                "40",
                "--seed",
                "3",
                "--out",
                str(sheet),
            ]
        )
        == 0
    )
    with open(sheet, encoding="utf-8", newline="") as fh:
        drawn = [int(r["index"]) for r in csv.DictReader(fh)]
    assert design["sample"]["indices"] == drawn
    assert [w["window_index"] for w in plan["windows"]] == drawn
    assert "majority class" in plan["population"]["what_the_rate_will_be_of"]
    # The reviewer's sheet carries each window's location; the result does not.
    with open(plan["labels_csv_path"], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["lon"] and r["lat"] for r in rows)
    assert not (_keys(plan) & _BANNED_KEYS)


@needs_extra
@pytest.mark.asyncio
async def test_a_zero_margin_window_keeps_its_class(_scores_root: Path) -> None:
    """A row cannot say which class won a tie; the file's map_class does."""
    logits = np.zeros((10, 8, 8), np.float32)
    logits[3] = 1.0
    logits[4] = 1.0  # classes 3 and 4 tie at every pixel: margin 0, class 3
    logits[:, 4:, :] = 0.0
    logits[5, 4:, :] = 2.0  # the lower half is class 5 by a margin of 2
    _write_run(_scores_root, "run_tie", logits, np.ones((8, 8), bool))
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run_tie"})
    review = await _ok(
        "olmoearth_review_set", {"scores_path": got["scores_path"], "budget": 1.0}
    )
    assert [r["margin"] for r in review["review"]] == [0.0, 0.0, 2.0, 2.0]
    assert [r["predicted_class"] for r in review["review"]] == [3, 3, 5, 5]
    assert review["review"][0]["class_name"] == "herbaceous_wetland"


@needs_extra
@pytest.mark.asyncio
async def test_two_runs_compare_on_identical_windows(_scores_root: Path) -> None:
    valid = _valid(32, 32)
    run_a = _write_run(_scores_root, "a", _logits(4, 32, 32), valid)
    run_b = _write_run(_scores_root, "b", _logits(5, 32, 32), valid)
    got_a = await _ok("olmoearth_scores_from_file", {"run_dir": "a"})
    got_b = await _ok("olmoearth_scores_from_file", {"run_dir": "b"})
    cmp = await _ok(
        "olmoearth_compare_review",
        {"scores_path_a": got_a["scores_path"], "scores_path_b": got_b["scores_path"]},
    )
    klass = []
    for run in (run_a, run_b):
        a, v = _read(run)
        ref = _package(a[:FILL], v, is_logit=True)
        klass.append(ref["arrays"]["pooled_argmax"][ref["arrays"]["valid"]])
    assert cmp["n_windows"] == got_a["n_valid"]
    assert cmp["n_differing"] == int((klass[0] != klass[1]).sum())
    assert cmp["which_side_is_right"] == "not resolvable without labels"
    # The files leave their no-data windows out: the spatial breakdown places
    # every differing window by its grid index, and the classes carry names.
    spatial = cmp["spatial"]
    assert spatial["grid"] == [8, 8]
    assert sum(b["n_differing"] for b in spatial["row_bands"]) == cmp["n_differing"]
    assert sum(b["n_windows"] for b in spatial["col_bands"]) == got_a["n_valid"]
    assert cmp["class_changes"][0]["class_name_a"] in NAMES
    saved = json.loads(Path(cmp["differing_path"]).read_text(encoding="utf-8"))
    assert len(saved["differing"]) == cmp["n_differing"]
    assert {f["id"] for f in cmp["facts"]} >= {"dominant_change", "concentration"}


@needs_extra
@pytest.mark.asyncio
async def test_the_model_revision_and_the_raster_hash_are_told_apart(
    _scores_root: Path,
) -> None:
    """exp86 round 7 gave the raster's sha256 prefix (also in the file's name)
    as the model's revision. One sentence keeps them apart."""
    valid = _valid(32, 32)
    run = _write_run(
        _scores_root,
        "run_id",
        _logits(8, 32, 32),
        valid,
        manifest_edit={
            "model": {
                "repo": "allenai/OlmoEarth-v1-FT-AWF-Base",
                "revision": "a347b1546ab8",
            }
        },
    )
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run_id"})
    digest = _sha256(run / "scores.tif")
    sentence = got["model_and_raster"]
    assert sentence.startswith(
        f"model allenai/OlmoEarth-v1-FT-AWF-Base at revision a347b15; raster sha256 {digest[:8]}"
    )
    assert f"the {digest[:8]} in the scores file's name is that hash" in sentence
    assert digest[:8] in got["scores_file"]
    (fact,) = got["facts"]
    assert fact["id"] == "model_and_raster" and fact["sentence"] == sentence
    assert (fact["revision"], fact["raster_sha256"]) == ("a347b1546ab8", digest)
    unrecorded = model_and_raster({"repo": "r/m", "revision": None}, "f" * 64, "x.json")
    assert unrecorded == (
        "model r/m at a revision the manifest does not record; raster sha256 "
        "ffffffff (a hash of the scores raster, not of the model)."
    )


@needs_extra
@pytest.mark.asyncio
async def test_the_providers_warnings_travel_with_its_file_to_the_ranking(
    _scores_root: Path,
) -> None:
    """exp86 round 6: the provider's multi-class warning reached no answer. The
    file carries it, and a ranking of the file states it."""
    valid = _valid(32, 32)
    _write_run(_scores_root, "run_warn", _logits(9, 32, 32), valid)
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run_warn"})
    saved = json.loads(Path(got["scores_path"]).read_text(encoding="utf-8"))
    assert saved["package_warnings"] == got["package_warnings"]
    assert got[
        "package_warnings"
    ], "a multi-class logit map draws the package's warning"
    review = await _ok("olmoearth_review_set", {"scores_path": got["scores_path"]})
    assert review["evidence_covers_this_case"] == "in part"
    assert review["scores_file_warnings"] == [
        "The scores provider (olmoearth-inferencex) warns: " + w
        for w in got["package_warnings"]
    ]
    # The package's own text ends in "pass form='top1'", an argument no agent
    # tool takes; must_state states the limit instead, once.
    assert any("form='top1'" in w for w in got["package_warnings"])
    assert review["must_state"] == [
        "This ranking uses the logit margin; on Ai2's suite one minus the top "
        "probability ranked errors better on 14 of 16 multi-class tasks."
    ]


# --------------------------------------------------------------------------- rule 1


@needs_extra
@pytest.mark.asyncio
async def test_no_coordinates_in_any_result(_scores_root: Path) -> None:
    valid = _valid(32, 32)
    _write_run(_scores_root, "run_geo", _logits(6, 32, 32), valid)
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run_geo"})
    review = await _ok("olmoearth_review_set", {"scores_path": got["scores_path"]})
    area = (LAT, LON, X0, Y0, LAT - 0.003, LON + 0.003, Y0 - 320.0, X0 + 320.0)
    for result in (got, review):
        assert not (_keys(result) & _BANNED_KEYS)
        text = json.dumps(result)
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", text)]
        for value in area:
            assert all(abs(n - value) > 1e-3 for n in numbers), value
        assert "EPSG" not in text
    # The file keeps each valid window's centre, for the labelling sheet only.
    saved = json.loads(Path(got["scores_path"]).read_text(encoding="utf-8"))
    centres = saved["centres_lon_lat"]
    assert len(centres) == got["n_valid"]
    assert all(abs(lon - LON) < 0.01 and abs(lat - LAT) < 0.01 for lon, lat in centres)


# --------------------------------------------------------------------------- guards


@needs_extra
@pytest.mark.asyncio
async def test_the_path_guard_confines_run_dir_to_the_scores_root(
    _scores_root: Path, tmp_path: Path
) -> None:
    outside = _write_run(
        tmp_path, "outside", _logits(7, 16, 16), np.ones((16, 16), bool)
    )
    (_scores_root / "link").symlink_to(outside, target_is_directory=True)
    for run_dir in ("../outside", str(outside), "link", "/"):
        out = await _call("olmoearth_scores_from_file", {"run_dir": run_dir})
        assert out["ok"] is False, run_dir
        assert "must be a directory under" in out["error"], run_dir
    missing = await _call("olmoearth_scores_from_file", {"run_dir": "nope"})
    assert missing["ok"] is False and "not a directory" in missing["error"]
    # A manifest cannot steer the read outside its directory.
    _write_run(
        _scores_root,
        "steer",
        _logits(8, 16, 16),
        np.ones((16, 16), bool),
        manifest_edit={"scores": {"file": "../outside/scores.tif", "values": "logits"}},
    )
    out = await _call("olmoearth_scores_from_file", {"run_dir": "steer"})
    assert out["ok"] is False and ".tif file name inside" in out["error"]
    # A path to the manifest or the raster names its directory.
    run = _write_run(_scores_root, "named", _logits(9, 16, 16), np.ones((16, 16), bool))
    for name in ("manifest.json", "scores.tif"):
        got = await _ok("olmoearth_scores_from_file", {"run_dir": str(run / name)})
        assert got["run_dir"] == "named"


@needs_extra
@pytest.mark.asyncio
async def test_a_manifest_that_does_not_describe_the_raster_is_refused(
    _scores_root: Path,
) -> None:
    ones = np.ones((16, 16), bool)
    _write_run(
        _scores_root,
        "wrong_hash",
        _logits(10, 16, 16),
        ones,
        manifest_edit={
            "scores": {"file": "scores.tif", "values": "logits", "sha256": "0" * 64}
        },
    )
    out = await _call("olmoearth_scores_from_file", {"run_dir": "wrong_hash"})
    assert out["ok"] is False and "sha256" in out["error"]
    _write_run(
        _scores_root,
        "no_kind",
        _logits(11, 16, 16),
        ones,
        manifest_edit={"scores": {"file": "scores.tif"}},
    )
    out = await _call("olmoearth_scores_from_file", {"run_dir": "no_kind"})
    assert out["ok"] is False and "logits or probabilities" in out["error"]
    _write_run(_scores_root, "too_many", _logits(12, 16, 16), ones)
    out = await _call(
        "olmoearth_scores_from_file", {"run_dir": "too_many", "patch": 32}
    )
    assert out["ok"] is False and "larger than" in out["error"]


def _bad_raster_run(root: Path, name: str, content: bytes | None) -> Path:
    """A run whose scores.tif is gone (``None``) or holds ``content``, not a raster.

    The manifest records no sha256, so the read gets as far as the raster.
    """
    run = _write_run(
        root,
        name,
        _logits(13, 16, 16),
        np.ones((16, 16), bool),
        manifest_edit={"scores": {"file": "scores.tif", "values": "logits"}},
    )
    path = run / "scores.tif"
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content)
    return run


@needs_extra
@pytest.mark.asyncio
async def test_a_missing_raster_is_refused(_scores_root: Path) -> None:
    _bad_raster_run(_scores_root, "gone", None)
    out = await _call("olmoearth_scores_from_file", {"run_dir": "gone"})
    assert out["ok"] is False
    assert "no raster 'scores.tif'" in out["error"]


@needs_extra
@pytest.mark.asyncio
@pytest.mark.parametrize("size_check", [True, False])
async def test_a_file_that_is_not_a_raster_is_refused_not_an_exit(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch, size_check: bool
) -> None:
    """The package's ``read_raster`` (``oe_inferencex.common.raster``) ends a file
    it cannot read with ``SystemExit``, which the registry does not catch: it
    would end the agent. The tool refuses instead, in the reader's words. With
    the size check off, the file reaches the package's reader itself."""
    _bad_raster_run(_scores_root, "text", b"not a GeoTIFF, just text\n")
    if not size_check:
        monkeypatch.setattr(scores_file, "_check_size", lambda *_args: None)
    out = await _call("olmoearth_scores_from_file", {"run_dir": "text"})
    assert out["ok"] is False
    assert out["error"].startswith("ValueError: ")
    assert "not a raster rasterio can read" in out["error"]
    if not size_check:
        assert "the scores raster was not read" in out["error"]


@needs_extra
def test_the_packages_reader_exit_becomes_a_refusal(tmp_path: Path) -> None:
    """``read_scores_raster`` turns the reader's ``SystemExit`` into the
    ``ValueError`` the registry answers with, for a missing file and a file
    that is not a raster alike, and passes a good raster through unchanged."""
    from oe_inferencex.common import raster as raster_io

    missing = tmp_path / "missing.tif"
    with pytest.raises(ValueError, match="no such file"):
        scores_file.read_scores_raster(raster_io, missing)
    text = tmp_path / "text.tif"
    text.write_bytes(b"not a GeoTIFF\n")
    with pytest.raises(ValueError, match="not a raster rasterio can read"):
        scores_file.read_scores_raster(raster_io, text)
    run = _write_run(tmp_path, "good", _logits(14, 16, 16), np.ones((16, 16), bool))
    array, valid, geo = scores_file.read_scores_raster(raster_io, run / "scores.tif")
    assert array.shape == (10, 16, 16) and bool(valid.all())
    assert geo is not None and str(geo["crs"]).endswith("32737")


@pytest.mark.asyncio
async def test_missing_extra_is_a_clear_answer_not_a_crash(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate an install without olmoearth-inferencex, then without rasterio."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "oe_inferencex.review.assess", None)
        out = await _ok("olmoearth_scores_from_file", {"run_dir": "anything"})
    assert out["available"] is False
    assert out["install"] == "pip install 'olmoearth-agent[inferencex]'"
    assert "olmoearth-inferencex" in out["missing"]
    assert "do not rank" in out["reason"]
    if not HAVE_EXTRA:
        return
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "rasterio", None)
        out = await _ok("olmoearth_scores_from_file", {"run_dir": "anything"})
    assert out["available"] is False and "rasterio" in out["missing"]


def test_fill_channels_follow_the_manifests_marks() -> None:
    from olmoearth_agent.tools.scores_file import fill_channels

    names = {i: n for i, n in enumerate(NAMES)} | {9: "nodata_value_9"}
    assert fill_channels({}, names) == [9]
    assert fill_channels({"task_card": {"task": {"nodata_value": 9}}}, names) == [9]
    # A task nodata_value that is a real class is not a fill channel.
    real = {0: "background", 1: "water"}
    assert fill_channels({"task_card": {"task": {"nodata_value": 0}}}, real) == []
    # A name that says another channel is not a mark for this one.
    assert fill_channels({}, {0: "a", 1: "nodata_value_7"}) == []


@pytest.mark.asyncio
async def test_a_scores_file_whose_classes_do_not_match_its_rows_is_refused(
    _scores_root: Path,
) -> None:
    path = _scores_root / "bad.json"
    path.write_text(
        json.dumps({"scores": [[0.0, 1.0], [2.0, 0.0]], "map_class": [1]}),
        encoding="utf-8",
    )
    out = await _call("olmoearth_review_set", {"scores_path": str(path)})
    assert out["ok"] is False and "map_class" in out["error"]


# --------------------------------------------------------------------------- a from-olmoearth run


def _softmax(seed: int, c: int, h: int, w: int) -> Any:
    """``c`` bands of float32 class probabilities, as output_probs writes them."""
    z = np.random.default_rng(seed).normal(0.0, 1.5, (c, h, w))
    e = np.exp(z - z.max(0, keepdims=True))
    return (e / e.sum(0, keepdims=True)).astype(np.float32)


def _write_dataset(
    root: Path, name: str, windows: list[tuple[Any, str, float, float]]
) -> Path:
    """An rslearn dataset after ``rslearn model predict``: an output per window.

    Each window is ``(values, crs, x0, y0)``: its ``(C, H, W)`` output in
    ``windows/default/w<k>/layers/output/probs/geotiff.tif`` at 10 m from
    ``(x0, y0)``, with the ``completed`` marker the writer leaves.
    """
    ds = root / name
    for k, (values, crs, x0, y0) in enumerate(windows):
        layer = ds / "windows" / "default" / f"w{k}" / "layers" / "output"
        (layer / "probs").mkdir(parents=True)
        c, h, w = values.shape
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": c,
            "dtype": str(values.dtype),
            "crs": crs,
            "transform": Affine(10.0, 0.0, x0, 0.0, -10.0, y0),
        }
        with rasterio.open(layer / "probs" / "geotiff.tif", "w", **profile) as dst:
            dst.write(values)
        (layer / "completed").write_text("", encoding="utf-8")
    return ds


@needs_extra
@pytest.mark.asyncio
async def test_a_from_olmoearth_run_is_read_as_the_package_reads_it(
    _scores_root: Path,
) -> None:
    """The directory, its summary or its raster names the run; the pooled
    windows are the package's assess on the raster from-olmoearth wrote."""
    first = _softmax(20, 3, 32, 32)
    first[:, :4, :4] = 0.0  # the fill rslearn leaves where no crop predicted
    windows = [
        (first, "EPSG:32737", X0, Y0),
        (_softmax(21, 3, 32, 32), "EPSG:32737", X0 + 320.0, Y0),
    ]
    ds = _write_dataset(_scores_root, "ds", windows)
    summary = oe_reader.read_output(str(ds), str(_scores_root / "run"))
    assert [g["label"] for g in summary["grids"]] == ["32737"]
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run"})
    files = got["run_files"]
    assert files["record"] == "olmoearth_output.json"
    assert files["written_by"] == "oe-inferencex from-olmoearth"
    assert files["raster"] == "scores_32737.tif" and files["windows_read"] == 2
    assert got["values"] == "probabilities" and "top-1 probability" in got["signal"]
    assert got["classes"] == {"0": "class_0", "1": "class_1", "2": "class_2"}
    assert got["grid"] == [8, 16] and got["n_windows"] == 128
    assert got["n_nodata"] == 1 and got["n_nodata_pixels"] == 16
    assert got["fill_class"] is None and got["date_window"] is None
    assert got["model_and_raster"].startswith(
        "model not named by the run's olmoearth_output.json, at a revision the "
        "run's olmoearth_output.json does not record; raster sha256"
    )

    raster = _scores_root / "run" / "scores_32737.tif"
    with rasterio.open(raster) as src:
        a = src.read()
    valid = np.isfinite(a).all(axis=0)
    ref = _package(a, valid, is_logit=False)
    conf, klass = ref["arrays"]["confidence"], ref["arrays"]["pooled_argmax"]
    saved = json.loads(Path(got["scores_path"]).read_text(encoding="utf-8"))
    assert saved["windows"] == np.flatnonzero(ref["arrays"]["valid"].ravel()).tolist()
    for w, row, k in zip(saved["windows"], saved["scores"], saved["map_class"]):
        r, c = divmod(w, 16)
        assert max(row) == conf[r, c] and k == klass[r, c]
    assert saved["raster"]["path"] == os.path.realpath(raster)
    for name in ("run/olmoearth_output.json", "run/scores_32737.tif"):
        again = await _ok("olmoearth_scores_from_file", {"run_dir": name})
        assert again["scores_file"] == got["scores_file"]
    assert not (_keys(got) & _BANNED_KEYS) and "EPSG" not in json.dumps(got)


@needs_extra
@pytest.mark.asyncio
async def test_a_run_that_wrote_class_ids_is_refused_with_how_to_get_probabilities(
    _scores_root: Path,
) -> None:
    """rslearn's SegmentationTask writes one band of argmax class ids by
    default. from-olmoearth refuses it, and so does the tool: given the
    dataset, or the class-id GeoTIFF itself, saying how to get probabilities
    (output_probs: true, a float32 layer, the fine-tuned checkpoint)."""
    ids = np.random.default_rng(22).integers(0, 4, (1, 16, 16)).astype(np.uint8)
    ds = _write_dataset(_scores_root, "ids", [(ids, "EPSG:32737", X0, Y0)])
    with pytest.raises(ValueError, match="class ids"):
        oe_reader.read_output(str(ds), str(_scores_root / "ids_run"))
    tif = ds / "windows" / "default" / "w0" / "layers" / "output" / "probs"
    for run_dir in ("ids", str(tif / "geotiff.tif")):
        out = await _call("olmoearth_scores_from_file", {"run_dir": run_dir})
        assert out["ok"] is False, run_dir
        error = out["error"]
        assert "one band of class ids (0, 1, 2, 3)" in error, error
        assert "output_probs: true" in error and "float32" in error
        assert "fine-tuned checkpoint" in error
        assert "oe-inferencex from-olmoearth DATASET --out RUN" in error
    # A dataset of probabilities is sent to from-olmoearth first.
    probs = [(_softmax(23, 3, 16, 16), "EPSG:32737", X0, Y0)]
    _write_dataset(_scores_root, "probs_ds", probs)
    out = await _call("olmoearth_scores_from_file", {"run_dir": "probs_ds"})
    assert out["ok"] is False
    assert "oe-inferencex from-olmoearth probs_ds --out RUN" in out["error"]
    (_scores_root / "empty").mkdir()
    out = await _call("olmoearth_scores_from_file", {"run_dir": "empty"})
    assert out["ok"] is False and "holds neither manifest.json" in out["error"]


@needs_extra
@pytest.mark.asyncio
async def test_a_run_of_several_crs_is_read_one_raster_at_a_time(
    _scores_root: Path,
) -> None:
    """from-olmoearth writes one scores raster per CRS: the tool asks which,
    and reads the one named; a per-window classification has no grid."""
    windows = [
        (_softmax(24, 3, 16, 16), "EPSG:32737", X0, Y0),
        (_softmax(25, 3, 16, 16), "EPSG:32736", 500_000.0, 9_000_000.0),
    ]
    ds = _write_dataset(_scores_root, "two_crs", windows)
    oe_reader.read_output(str(ds), str(_scores_root / "run2"))
    out = await _call("olmoearth_scores_from_file", {"run_dir": "run2"})
    assert out["ok"] is False and "one scores raster per CRS" in out["error"]
    assert "scores_32736.tif, scores_32737.tif" in out["error"]
    got = await _ok("olmoearth_scores_from_file", {"run_dir": "run2/scores_32736.tif"})
    assert got["run_files"]["raster"] == "scores_32736.tif"
    assert got["run_files"]["rasters"] == ["scores_32736.tif", "scores_32737.tif"]
    assert got["grid"] == [4, 4] and got["n_valid"] == 16
    per_window = _scores_root / "per_window"
    per_window.mkdir()
    (per_window / "olmoearth_output.json").write_text(
        json.dumps({"kind": "per-window probabilities", "bands": 3}),
        encoding="utf-8",
    )
    out = await _call("olmoearth_scores_from_file", {"run_dir": "per_window"})
    assert out["ok"] is False and "per-window classification" in out["error"]


def _class_id_tif(path: Path) -> None:
    """One band of class ids 41, 42 and 43, as rslearn's default output."""
    ids = np.array([[41, 42], [43, 41]], dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32737",
        "transform": Affine(10.0, 0.0, X0, 0.0, -10.0, Y0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(ids[None])


@needs_extra
@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
@pytest.mark.asyncio
async def test_a_raster_linked_from_outside_the_root_is_never_opened(
    _scores_root: Path, tmp_path: Path
) -> None:
    """A directory with no record: its class-id GeoTIFFs are read to say how
    to get probabilities, but only those inside the scores root. A link to a
    raster outside it is passed over, so none of its values reach the
    refusal; a class-id raster inside is still named."""
    outside = tmp_path / "outside" / "ids.tif"
    _class_id_tif(outside)
    linked = _scores_root / "linked"
    linked.mkdir()
    os.symlink(outside, linked / "ids.tif")
    out = await _call("olmoearth_scores_from_file", {"run_dir": "linked"})
    assert out["ok"] is False and "holds neither manifest.json" in out["error"]
    assert "41" not in out["error"] and "class ids" not in out["error"]
    layer = _scores_root / "ds_link" / "windows" / "default" / "w0" / "layers"
    (layer / "output" / "probs").mkdir(parents=True)
    os.symlink(outside, layer / "output" / "probs" / "geotiff.tif")
    out = await _call("olmoearth_scores_from_file", {"run_dir": "ds_link"})
    assert out["ok"] is False and "is an rslearn dataset" in out["error"]
    assert "41" not in out["error"] and "class ids (" not in out["error"]
    _class_id_tif(_scores_root / "inside" / "ids.tif")
    out = await _call("olmoearth_scores_from_file", {"run_dir": "inside"})
    assert out["ok"] is False and "class ids (41, 42, 43)" in out["error"]


@needs_extra
@pytest.mark.asyncio
async def test_a_run_record_that_is_not_json_is_refused_by_name(
    _scores_root: Path,
) -> None:
    """A truncated olmoearth_output.json or manifest.json is the tool's
    refusal naming the file and how to write it again, never a JSON parser's
    message; a summary whose grid label holds a path names no raster."""
    for name, record, again in (
        ("cut_summary", "olmoearth_output.json", "from-olmoearth DATASET --out RUN"),
        ("cut_manifest", "manifest.json", "score_area.py"),
    ):
        run = _scores_root / name
        run.mkdir()
        (run / record).write_text('{"grids": [', encoding="utf-8")
        out = await _call("olmoearth_scores_from_file", {"run_dir": name})
        assert out["ok"] is False, name
        assert f"the run's {record} is not valid JSON" in out["error"]
        assert again in out["error"] and "JSONDecodeError" not in out["error"]
    escape = _scores_root / "escape"
    escape.mkdir()
    (escape / "olmoearth_output.json").write_text(
        json.dumps({"grids": [{"label": "../x", "shape": [4, 4]}], "bands": 3}),
        encoding="utf-8",
    )
    out = await _call("olmoearth_scores_from_file", {"run_dir": "escape"})
    assert out["ok"] is False
    assert "names a raster outside its directory" in out["error"]


@needs_extra
@pytest.mark.asyncio
async def test_a_certificate_plan_needs_every_channel(_scores_root: Path) -> None:
    """A certificate is of the map as the package reads it, every channel: a
    scores file that left the label-fill channel out is refused for a plan
    drawn with seed_from_map, before anything is drawn."""
    _write_run(_scores_root, "run_cert", _logits(15, 32, 32), np.ones((32, 32), bool))
    left_out = await _ok("olmoearth_scores_from_file", {"run_dir": "run_cert"})
    assert left_out["fill_class"]["excluded"] is True
    out = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": left_out["scores_path"],
            "budget": 20,
            "design": "random",
            "seed_from_map": True,
            "alpha": 0.2,
        },
    )
    assert out["ok"] is False and "include_fill_class=true" in out["error"]
    assert not (_scores_root / "certificates").exists()


@needs_extra
@pytest.mark.asyncio
async def test_a_summary_raster_linked_out_of_the_root_is_refused(
    _scores_root: Path, tmp_path: Path
) -> None:
    """The raster a from-olmoearth summary lists is looked up inside the run
    directory and must resolve inside the scores root: a link to a file
    outside it is refused as no raster of the run."""
    windows = [(_softmax(26, 3, 16, 16), "EPSG:32737", X0, Y0)]
    ds = _write_dataset(_scores_root, "ds_out", windows)
    oe_reader.read_output(str(ds), str(_scores_root / "run_out"))
    raster = _scores_root / "run_out" / "scores_32737.tif"
    outside = tmp_path / "moved.tif"
    raster.rename(outside)
    os.symlink(outside, raster)
    out = await _call("olmoearth_scores_from_file", {"run_dir": "run_out"})
    assert out["ok"] is False
    assert "has no raster 'scores_32737.tif'" in out["error"]
