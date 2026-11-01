# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Every tool declares its capability, and every condition it states is the code.

The capability card in the system prompt (``harness/capabilities.py``) is
built from these declarations. exp86 round 9: 7 of the audit's 16 material
findings were offers of actions no tool can do or whose preconditions did not
hold. A card that says a tool refuses what it runs, or needs what it does not,
would teach the model the same errors, so each ``needs`` and ``cannot`` entry
has a probe below that calls the tool, through the registry, in that
condition and gets the refusal or the absence the entry states. A new entry
with no probe fails, and so does a probe whose entry is gone.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from olmoearth_agent.analysis.review_set import margins
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.skills import OPT_IN_ENV, build_default_registry
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.types import ApiEnvelope

_BOX = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}


def _registry() -> ToolRegistry:
    """The default registry with every opt-in tool enabled too.

    The tools trimmed from the default registry (``OLMOEARTH_OPT_IN_TOOLS``):
    an opt-in tool's card line is shown when it is on, so its declarations are
    probed all the same.
    """
    with mock.patch.dict(os.environ, {OPT_IN_ENV: "1"}):
        return build_default_registry()


def _stated() -> list[tuple[str, str]]:
    """Every ``(tool, entry)`` of every tool's ``needs`` and ``cannot``."""
    registry = _registry()
    out: list[tuple[str, str]] = []
    for name in registry.names():
        capability = registry.capability_of(name)
        if capability is not None:
            out += [(name, entry) for entry in (*capability.needs, *capability.cannot)]
    return out


# --------------------------------------------------------------------------- declarations


def test_every_tool_of_the_default_registry_declares_its_capability() -> None:
    registry = _registry()
    missing = [n for n in registry.names() if registry.capability_of(n) is None]
    assert missing == [], f"tools with no capability declared: {missing}"


def test_every_declaration_is_one_short_line() -> None:
    """The card's 30 default core tools fit 1,700 tokens; a line is a line."""
    registry = _registry()
    for name in registry.names():
        capability = registry.capability_of(name)
        assert capability is not None
        entries = (capability.does, *capability.needs, *capability.cannot)
        for entry in entries:
            assert entry.strip() == entry and entry, (name, entry)
            assert "\n" not in entry and not entry.endswith("."), (name, entry)
        assert len(capability.does) <= 160, name
        assert all(len(e) <= 100 for e in (*capability.needs, *capability.cannot))


# --------------------------------------------------------------------------- probes


class _Studio:
    """The Studio calls a refusal may make before any sampling; nothing else."""

    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        predictions: dict[str, dict[str, Any]] | None = None,
        models: dict[str, dict[str, Any]] | None = None,
        listed: list[dict[str, Any]] | None = None,
    ) -> None:
        self.results = results or {}
        self.predictions = predictions or {}
        self.models = models or {}
        self.listed = listed or []

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        return self.results[result_id]

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        return self.predictions[prediction_id]

    async def get_model(self, model_id: str) -> dict[str, Any]:
        return self.models[model_id]

    async def search_predictions(self, **_kw: Any) -> ApiEnvelope[dict[str, Any]]:
        return ApiEnvelope(records=self.listed, meta={"total": len(self.listed)})

    async def pixel_value(self, *_a: Any, **_kw: Any) -> dict[str, Any]:
        raise AssertionError("a refusal is made before any pixel is sampled")


def _result(
    rid: str, prop: str, meta: dict[str, Any], prediction_id: str | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": rid,
        "property_names": [prop],
        "result_metadata": {"geometry": _BOX, **meta},
    }
    if prediction_id:
        record["prediction_id"] = prediction_id
    return record


def _regression(prop: str, lo: float, hi: float) -> dict[str, Any]:
    return {
        "regression_fields": [{"property_name": prop, "min_value": lo, "max_value": hi}]
    }


_CLASSIFICATION = {
    "classification_fields": [
        {
            "property_name": "landcover",
            "allowed_values": [{"value": 1, "label": "water"}],
        }
    ]
}


async def _call(name: str, args: dict[str, Any], studio: Any = None) -> dict[str, Any]:
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await _registry().dispatch(ToolCall(id="p", name=name, arguments=args), ctx)


def _refused(envelope: dict[str, Any], phrase: str) -> None:
    """The call was refused (a failed envelope, or a result that declines), naming ``phrase``."""
    if envelope["ok"] is False:
        text = str(envelope["error"])
    else:
        result = envelope["result"]
        declined = ("ranked", "comparable", "certified", "available")
        assert any(result.get(k) is False for k in declined), result
        text = str(result.get("reason", ""))
    assert phrase in text, text


def _logits(n: int, seed: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(3)] for _ in range(n)]


Probe = Callable[[Path], Awaitable[None]]
PROBES: dict[tuple[str, str], Probe] = {}


def probe(tool: str, entry: str) -> Callable[[Probe], Probe]:
    """Register the probe of one declared entry."""

    def register(fn: Probe) -> Probe:
        PROBES[(tool, entry)] = fn
        return fn

    return register


@probe(
    "olmoearth_search_predictions",
    "say a model's metrics, or count its Studio annotations "
    "(olmoearth_studio_label_inventory does)",
)
async def _search_gives_no_training(_tmp: Path) -> None:
    """exp86 round 9 said "no ground-truth labels exist" of two fine-tuned models.

    The entry said "what a model's training data held" until the label
    inventory (31 October 2026) counted the annotations a model's training
    selector selects: the search still gives no metrics and counts no
    annotation (this fake Studio has no annotation search), and its
    ``labels_in_studio`` fact points at the inventory.
    """
    studio = _Studio(
        listed=[{"id": "p1", "name": "run", "status": "completed", "model_id": "m1"}],
        models={
            "m1": {
                "id": "m1",
                "name": "Karst",
                "model_type": "fine_tuned",
                "wizard_answers": {
                    "prediction_type": "per_pixel_regression",
                    "label_field": "karst_label_field",
                    "label_field_id": "karst_label_field",
                    "training_data": "karst_training_windows",
                },
                "training_metrics": {"val_loss": 0.123456},
            }
        },
    )
    out = await _call("olmoearth_search_predictions", {}, studio)
    assert out["ok"] is True
    model = out["result"]["models"]["m1"]
    # the label field it was fine-tuned on and its split, so a user can find
    # the labels in Studio (round 10); never the training data or metrics
    assert set(model) <= {
        "name",
        "model_type",
        "prediction_type",
        "nodata_value",
        "trained_on_labels",
        "label_field_id",
        "split_field_named",
        "split",
    }
    assert model.get("trained_on_labels") is True
    blob = json.dumps(out)
    for secret in ("karst_training_windows", "0.123456"):
        assert secret not in blob
    # no annotation is counted or named anywhere in the result
    assert "annotation" not in json.dumps(out["result"])
    (labels,) = out["result"]["facts"]
    assert "olmoearth_studio_label_inventory counts those" in labels["sentence"]


@probe("olmoearth_compare_results", "2 or more distinct results")
async def _compare_needs_two(_tmp: Path) -> None:
    out = await _call("olmoearth_compare_results", {"result_ids": ["r1", "r1"]})
    _refused(out, "need >= 2 DISTINCT result_ids")


@probe(
    "olmoearth_compare_results",
    "results of one property (allow_different_properties relaxes it for a pair or group)",
)
async def _compare_needs_one_property(_tmp: Path) -> None:
    studio = _Studio(
        results={
            "r1": _result("r1", "a", _regression("a", 0, 1)),
            "r2": _result("r2", "b", _regression("b", 0, 1)),
            "r3": _result("r3", "a", _regression("a", 0, 1)),
        }
    )
    pair = await _call(
        "olmoearth_compare_results", {"result_ids": ["r1", "r2"]}, studio
    )
    _refused(pair, "measure different properties")
    ensemble = await _call(
        "olmoearth_compare_results",
        {
            "result_ids": ["r1", "r2", "r3"],
            "mode": "ensemble",
            "allow_different_properties": True,
        },
        studio,
    )
    _refused(ensemble, "measure different properties")


@probe("olmoearth_compare_results", "for mode='series', 3 or more dates of one model")
async def _series_needs_three_dates_of_one_model(_tmp: Path) -> None:
    two = await _call(
        "olmoearth_compare_results", {"result_ids": ["r1", "r2"], "mode": "series"}
    )
    _refused(two, "a series needs >= 3 distinct result ids")
    ids = ["r1", "r2", "r3"]
    studio = _Studio(
        results={r: _result(r, "a", _regression("a", 0, 1), f"p{r}") for r in ids},
        predictions={
            f"p{r}": {"model_id": m, "start_time": d}
            for r, m, d in zip(
                ids, ("m1", "m1", "m2"), ("2021-06-01", "2022-06-01", "2023-06-01")
            )
        },
    )
    mixed = await _call(
        "olmoearth_compare_results", {"result_ids": ids, "mode": "series"}, studio
    )
    _refused(mixed, "span multiple models")
    shared = _Studio(
        results=studio.results,
        predictions={
            f"p{r}": {"model_id": "m1", "start_time": d}
            for r, d in zip(ids, ("2021-06-01", "2021-06-01", "2023-06-01"))
        },
    )
    same_dates = await _call(
        "olmoearth_compare_results", {"result_ids": ids, "mode": "series"}, shared
    )
    _refused(same_dates, "needs >= 3 distinct dates")


@probe("olmoearth_classification_metrics", "y_true and y_pred of the same length")
async def _metrics_need_aligned_labels(_tmp: Path) -> None:
    out = await _call(
        "olmoearth_classification_metrics", {"y_true": [1, 2, 1], "y_pred": [1, 2]}
    )
    _refused(out, "must have the same length")


@probe(
    "olmoearth_review_set",
    "2 or more class scores per window (a hard class has no margin)",
)
async def _review_set_needs_class_scores(_tmp: Path) -> None:
    out = await _call("olmoearth_review_set", {"scores": [[0.2], [0.9], [0.4]]})
    _refused(out, "needs >= 2 classes per window")


@probe(
    "olmoearth_review_set_from_result", "threshold, for a regression band not in [0, 1]"
)
async def _from_result_needs_threshold(_tmp: Path) -> None:
    """exp86 round 9 (B3/studio) offered a review set of KarstNumber, 0.2 to 1.2."""
    studio = _Studio(
        results={"k": _result("k", "karst", _regression("karst", 0.2, 1.2))}
    )
    out = await _call("olmoearth_review_set_from_result", {"result_id": "k"}, studio)
    _refused(out, "needs a decision threshold")


@probe(
    "olmoearth_review_set_from_result",
    "rank a class map with no top-k scores (no margin)",
)
async def _from_result_refuses_classes(_tmp: Path) -> None:
    """A class map without top-k scores (one with them is ranked: test_topk_scores)."""
    studio = _Studio(results={"lc": _result("lc", "landcover", _CLASSIFICATION)})
    out = await _call("olmoearth_review_set_from_result", {"result_id": "lc"}, studio)
    _refused(out, "A class map without top-k scores has no confidence to rank")
    _refused(out, "probability_output")


class _ClassMapStudio(_Studio):
    """A Studio class map: water west of longitude 2, forest east of it."""

    async def pixel_value(self, _result_id: str, lon: float, _lat: float) -> Any:
        label = "water" if lon < 2 else "forest"
        band = {
            "property_name": "landcover",
            "raw_value": 1 if label == "water" else 2,
            "classification": {"label": label},
        }
        return {"bands": [band]}


@probe(
    "olmoearth_plan_label_sample",
    "'random' or 'class' for a class map with no top-k scores",
)
async def _plan_takes_random_or_class_for_a_class_map(_tmp: Path) -> None:
    """The ranking designs read a confidence a class map does not have."""
    studio = _Studio(results={"lc": _result("lc", "landcover", _CLASSIFICATION)})
    for design in ("confidence", "proportional"):
        out = await _call(
            "olmoearth_plan_label_sample",
            {"result_id": "lc", "budget": 20, "design": design},
            studio,
        )
        _refused(out, "A class map without top-k scores has no confidence to rank")
        assert out["result"]["designs_for_a_class_map"] == ["random", "class"]


@probe("olmoearth_certify_zone", "certify a class map with no top-k scores")
async def _certify_refuses_a_class_map(_tmp: Path) -> None:
    """A random design of a Studio class map is planned, and refused a zone."""
    studio = _ClassMapStudio(
        results={"lc": _result("lc", "landcover", _CLASSIFICATION)}
    )
    plan = await _call(
        "olmoearth_plan_label_sample",
        {"result_id": "lc", "grid": 3, "budget": 4, "design": "random"},
        studio,
    )
    assert plan["ok"] is True, plan
    out = await _call(
        "olmoearth_certify_zone",
        {"design_path": plan["result"]["design_path"], "wrong": [0] * 4, "alpha": 0.1},
    )
    _refused(out, "A class map without top-k scores has no confidence to rank")


@probe("olmoearth_compare_review", "both inferences over the same windows")
async def _compare_review_needs_same_windows(_tmp: Path) -> None:
    out = await _call(
        "olmoearth_compare_review", {"scores_a": _logits(4), "scores_b": _logits(3)}
    )
    _refused(out, "cover different window counts")


@probe(
    "olmoearth_compare_review",
    "say which map is right, even with labels_date (it takes no labels)",
)
async def _compare_review_grades_no_map(_tmp: Path) -> None:
    """exp86 round 9 (B7/files) offered to "rerun with labels_date" to grade the maps."""
    spec = _registry().spec_of("olmoearth_compare_review")
    assert spec is not None
    taken = set(spec.parameters["properties"])
    assert not taken & {"labels", "labels_path", "wrong", "reference", "y_true"}
    dated = await _call(
        "olmoearth_compare_review",
        {
            "scores_a": _logits(6, 1),
            "scores_b": _logits(6, 2),
            "date_a": "2022-01-01/2022-12-31",
            "date_b": "2023-01-01/2023-12-31",
            "labels_date": "2022-01-01/2022-12-31",
        },
    )
    assert dated["ok"] is True
    assert dated["result"]["which_side_is_right"].startswith("not graded")
    undated = await _call(
        "olmoearth_compare_review",
        {"scores_a": _logits(6, 1), "scores_b": _logits(6, 2)},
    )
    ids = {c["id"] for c in undated["result"]["forbidden_claims"]}
    assert rules.WINNER_WITHOUT_LABELS in ids


@probe("olmoearth_scores_from_file", "run_dir under the scores root")
async def _scores_file_reads_only_the_root(tmp: Path) -> None:
    outside = tmp.parent / f"{tmp.name}_outside"
    outside.mkdir()
    out = await _call("olmoearth_scores_from_file", {"run_dir": str(outside)})
    _refused(out, "run_dir must be a directory under")


@probe("olmoearth_scores_from_studio", "read a class map with no top-k scores")
async def _studio_scores_refuse_a_class_map(_tmp: Path) -> None:
    """Refused from the result's metadata, before any download (_Studio has none).

    The id is a UUID, the API's format for it: the tool asks Studio for no
    other.
    """
    rid = "7b1d9e20-6c4f-4a8b-8e2d-3f5a9c0b1e44"
    studio = _Studio(results={rid: _result(rid, "landcover", _CLASSIFICATION)})
    out = await _call("olmoearth_scores_from_studio", {"result_id": rid}, studio)
    _refused(out, "class map without top-k scores")
    _refused(out, "probability_output")


@probe(
    "olmoearth_plan_label_sample",
    "threshold, for a Studio regression band not in [0, 1]",
)
async def _plan_needs_threshold(_tmp: Path) -> None:
    studio = _Studio(
        results={"k": _result("k", "karst", _regression("karst", 0.2, 1.2))}
    )
    out = await _call(
        "olmoearth_plan_label_sample", {"result_id": "k", "budget": 20}, studio
    )
    assert out["ok"] is True and out["result"]["ranked"] is False
    assert "needs a decision threshold" in out["result"]["reason"]


@probe(
    "olmoearth_plan_label_sample",
    "a budget of at most the valid windows (Studio grid 16 plans all)",
)
async def _plan_needs_a_budget_it_can_place(_tmp: Path) -> None:
    out = await _call(
        "olmoearth_plan_label_sample", {"scores": _logits(10), "budget": 20}
    )
    _refused(out, "more than the 10 valid windows")


@probe(
    "olmoearth_plan_label_sample",
    "seed_from_map: a scores file from a model run's GeoTIFF",
)
async def _plan_seeds_from_map_only_a_model_runs_scores_file(tmp: Path) -> None:
    """Agent trial 12 (N4/files run 6) offered seed_from_map on a scores file
    no model run's GeoTIFF wrote; inline scores and such a file are refused."""
    inline = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores": _logits(40),
            "budget": 10,
            "design": "random",
            "seed_from_map": True,
            "alpha": 0.1,
        },
    )
    _refused(inline, "olmoearth_scores_from_file wrote from a model run's output")
    path = tmp / "class_scores.json"
    path.write_text(json.dumps({"grid": [5, 8], "scores": _logits(40)}))
    other = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": str(path),
            "budget": 10,
            "design": "random",
            "seed_from_map": True,
            "alpha": 0.1,
        },
    )
    _refused(other, "is refused for any other scores file")


@probe(
    "olmoearth_plan_label_sample",
    "'disagreement': two maps of one time on one window grid",
)
async def _plan_two_maps_needs_one_time_and_one_grid(tmp: Path) -> None:
    """The dates are read before any pixel is sampled (_Studio would raise)."""
    studio = _Studio(
        results={r: _result(r, "landcover", _CLASSIFICATION, f"p{r}") for r in "ab"},
        predictions={
            "pa": {"model_id": "m1", "start_time": "2022-06-01T00:00:00Z"},
            "pb": {"model_id": "m2", "start_time": "2023-06-01T00:00:00Z"},
        },
    )
    apart = await _call(
        "olmoearth_plan_label_sample",
        {"result_id": "a", "result_id_b": "b", "budget": 20, "design": "disagreement"},
        studio,
    )
    _refused(apart, "the two maps are not of one time")
    for name, grid in (("a.json", [2, 2]), ("b.json", [1, 4])):
        (tmp / name).write_text(json.dumps({"grid": grid, "scores": _logits(4)}))
    grids = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": str(tmp / "a.json"),
            "scores_path_b": str(tmp / "b.json"),
            "budget": 2,
            "design": "disagreement",
        },
    )
    _refused(grids, "not of one window grid")


async def _plan(
    design: str = "confidence", n: int = 400, budget: int = 60
) -> dict[str, Any]:
    out = await _call(
        "olmoearth_plan_label_sample",
        {"scores": _logits(n), "budget": budget, "design": design},
    )
    assert out["ok"] is True, out
    return out["result"]


async def _two_map_plan(tmp: Path) -> dict[str, Any]:
    """A two-map plan of two 4 x 4 scores files that differ on 6 windows."""
    a = [[1.0, 0.0]] * 16
    b = [[0.0, 1.0] if i % 3 == 0 else [1.0, 0.0] for i in range(16)]
    for name, rows in (("map_a.json", a), ("map_b.json", b)):
        (tmp / name).write_text(json.dumps({"grid": [4, 4], "scores": rows}))
    out = await _call(
        "olmoearth_plan_label_sample",
        {
            "scores_path": str(tmp / "map_a.json"),
            "scores_path_b": str(tmp / "map_b.json"),
            "budget": 4,
            "design": "disagreement",
        },
    )
    assert out["ok"] is True, out
    return out["result"]


@probe(
    "olmoearth_estimate_map_error",
    "a label on every window of a planned design (a class for 'disagreement'), "
    "or on random windows",
)
async def _estimate_needs_every_label(tmp: Path) -> None:
    plan = await _plan()
    out = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": plan["design_path"], "wrong": [0] * 59},
    )
    _refused(out, "59 labels for the 60 windows of the design")
    two = await _two_map_plan(tmp)
    blank = await _call(
        "olmoearth_estimate_map_error",
        {"design_path": two["design_path"], "reference": ["0", "1", "?", ""]},
    )
    _refused(blank, "1 of 4 design windows have no reference_class yet")


@probe(
    "olmoearth_estimate_map_error",
    "use a review set, or windows chosen by margin, as the sample",
)
async def _estimate_refuses_a_review_set(_tmp: Path) -> None:
    """exp86 round 9 (B8/cluster): labels on flagged windows as "a proper error-rate estimate"."""
    scores = _logits(1200)
    marg = margins(scores)
    review = sorted(range(len(scores)), key=marg.__getitem__)[:100]
    out = await _call(
        "olmoearth_estimate_map_error",
        {"window_indices": review, "wrong": [1] * 50 + [0] * 50, "scores": scores},
    )
    _refused(out, "enriched set, not a sample")


@probe("olmoearth_certify_zone", "a design='random' plan and its labels")
async def _certify_needs_a_random_design(_tmp: Path) -> None:
    plan = await _plan("confidence")
    out = await _call(
        "olmoearth_certify_zone",
        {"design_path": plan["design_path"], "wrong": [0] * 60, "alpha": 0.1},
    )
    _refused(out, "needs a simple random sample of the map")


@probe("olmoearth_write_certificate", "a plan drawn with seed_from_map, and its labels")
async def _certificate_needs_a_map_seeded_plan(_tmp: Path) -> None:
    """A random plan whose seed was not taken from the map gets no certificate."""
    plan = await _plan("random")
    out = await _call(
        "olmoearth_write_certificate",
        {"design_path": plan["design_path"], "wrong": [0] * 60},
    )
    _refused(out, "was not drawn for a certificate")


@probe(
    "olmoearth_verify_certificate",
    "certificate_path and map_path under the scores root",
)
async def _verify_reads_only_the_scores_root(tmp: Path) -> None:
    outside = tmp.parent / f"{tmp.name}_bundle"
    outside.mkdir()
    for folder in (outside, tmp):
        (folder / "certificate.json").write_text("{}", encoding="utf-8")
    out = await _call(
        "olmoearth_verify_certificate", {"certificate_path": str(outside)}
    )
    _refused(out, "certificate_path must be under")
    out = await _call(
        "olmoearth_verify_certificate",
        {
            "certificate_path": str(tmp / "certificate.json"),
            "map_path": str(outside / "certificate.json"),
        },
    )
    _refused(out, "map_path must be under")


@probe("olmoearth_export_data", "out_dir inside the workspace")
async def _export_writes_only_the_workspace(tmp: Path) -> None:
    out = await _call(
        "olmoearth_export_data", {"out_dir": str(tmp.parent / "elsewhere")}
    )
    _refused(out, "resolves outside the workspace root")


_STUDIO_TILES = "/api/v1/prediction-results/abc/tiles/{z}/{x}/{y}.png?property_name=s"


@probe("olmoearth_qgis_bridge", "tile URLs on the Studio host")
async def _qgis_needs_studio_tiles(_tmp: Path) -> None:
    out = await _call(
        "olmoearth_qgis_bridge",
        {"tile_urls": ["https://tiles.example.org/{z}/{x}/{y}.png?property_name=s"]},
    )
    _refused(out, "refusing to build a QGIS auth pack for non-Studio host")


@probe(
    "olmoearth_qgis_bridge",
    "make a Cloud-Optimized GeoTIFF (it returns a command for the user to run)",
)
async def _qgis_returns_a_cog_command(tmp: Path) -> None:
    out = await _call("olmoearth_qgis_bridge", {"tile_urls": [_STUDIO_TILES]})
    assert out["ok"] is True
    assert "gdal_translate" in out["result"]["cog_recipe"]["command"]
    assert not list(tmp.rglob("*.tif"))


# --------------------------------------------------------------------------- the check


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "entry"), _stated())
async def test_every_stated_condition_is_what_the_tool_does(
    tool: str, entry: str, tmp_path: Path
) -> None:
    run = PROBES.get((tool, entry))
    assert (
        run is not None
    ), f"{tool} declares {entry!r}, and no probe calls it in that condition"
    await run(tmp_path)


def test_no_probe_outlives_its_entry() -> None:
    stale = set(PROBES) - set(_stated())
    assert stale == set(), f"probes of entries no tool declares: {sorted(stale)}"
