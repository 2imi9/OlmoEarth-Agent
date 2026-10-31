# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A Studio class map with top-k scores: ranked, sampled by margin, certified.

A segmentation prediction submitted with ``probability_output`` returns,
beside its class property ``P``, each rank's softmax score (``P_rank1_prob``,
``P_rank2_prob``) and runner-up class (``P_rank2_class``) as bands of their
own. Studio declares the scores as regression fields of range 0 to 254 and
stores them as uint8, read as ``(raw - 1) / 254``; the records below are
shaped like the kelp result measured on 31 October 2026 (one pixel-value
record holds every band with its raw value, and ``P``'s confidence band is
the band of ``P_rank1_prob``). Its margin, the rank-1 minus the rank-2 score,
gives it the routes a ``[0, 1]`` score has: the review set, the margin
designs and a certified zone from a random design. A class map without top-k
scores keeps its refusal, which now says where top-k scores come from.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.analysis.raster_compare import declared_fields, grid_windows
from olmoearth_agent.analysis.review_set import (
    MUST_STATE_MULTICLASS_LOGIT,
    MUST_STATE_TOPK_SCORE,
    TOPK_SCORE,
)
from olmoearth_agent.harness.checks import (
    RunEvidence,
    ToolRecord,
    check_forbidden_claims,
    check_must_state,
)
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.compare import build_compare_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.review_set import TOPK_ZONE_STEP, build_review_set_tools
from olmoearth_agent.tools.sampling import (
    NODATA,
    OK,
    ResultNodata,
    read_topk,
    select_band,
    topk_probability,
    topk_scores,
)

FROM_RESULT = "olmoearth_review_set_from_result"
PLAN = "olmoearth_plan_label_sample"
GRADE = "olmoearth_estimate_map_error"
CERTIFY = "olmoearth_certify_zone"

_BOX = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}
_BBOX = [0.0, 0.0, 4.0, 4.0]
_UINT8 = (0.0, 254.0)

#: The kelp result's classes, and a three-class legend.
KELP = [
    {"value": 0, "label": "absent", "color": "#2b83ba"},
    {"value": 1, "label": "present", "color": "#d7191c"},
]
COVER = [
    {"value": 0, "label": "water", "color": "#2b83ba"},
    {"value": 1, "label": "kelp", "color": "#1a9641"},
    {"value": 2, "label": "cloud", "color": "#ffffff"},
]

#: The result's properties, in the order the kelp result lists them.
PROPS = ["kelp", "kelp_rank2_class", "kelp_rank1_prob", "kelp_rank2_prob"]

#: A window's (rank-1 class index, rank-2 class index, raw rank-1 score, raw
#: rank-2 score); a rank-1 index of None is a no-data pixel.
Pixel = tuple[int | None, int, int, int]


def _meta(
    classes: list[dict[str, Any]], score_range: tuple[float, float] = _UINT8
) -> dict[str, Any]:
    """``result_metadata`` of a top-2 result, as Studio's GET by id returns it."""
    lo, hi = score_range
    return {
        "geometry": _BOX,
        "classification_fields": [
            {
                "property_name": "kelp",
                "confidence_property_name": None,
                "allowed_values": classes,
                "band_index": 1,
                "confidence_band_index": 2,
            },
            {
                "property_name": "kelp_rank2_class",
                "confidence_property_name": None,
                "allowed_values": classes,
                "band_index": 3,
                "confidence_band_index": 4,
            },
        ],
        "regression_fields": [
            {
                "property_name": "kelp_rank1_prob",
                "band_index": 2,
                "min_value": lo,
                "max_value": hi,
            },
            {
                "property_name": "kelp_rank2_prob",
                "band_index": 4,
                "min_value": lo,
                "max_value": hi,
            },
        ],
    }


def _result(
    rid: str,
    classes: list[dict[str, Any]],
    score_range: tuple[float, float] = _UINT8,
) -> dict[str, Any]:
    """A top-2 prediction result record (no prediction, so no model lookup)."""
    return {
        "id": rid,
        "property_names": list(PROPS),
        "result_metadata": _meta(classes, score_range),
    }


def _record(
    classes: list[dict[str, Any]],
    pixel: Pixel,
    score_range: tuple[float, float] = _UINT8,
    drop: str | None = None,
    first: str | None = None,
) -> dict[str, Any]:
    """One pixel-value record: every band with its raw value, as Studio returns it.

    At a no-data pixel the class bands hold 255 and no class, and so do the
    score bands. ``drop`` leaves one band out; ``first`` moves one band to the
    front (the band order is not a contract).
    """
    rank1, rank2, raw1, raw2 = pixel
    lo, hi = score_range

    def class_band(name: str, index: int, k: int | None) -> dict[str, Any]:
        if k is None:
            return {
                "band_index": index,
                "property_name": name,
                "raw_value": 255.0,
                "classification": None,
                "regression": None,
            }
        value = classes[k]
        return {
            "band_index": index,
            "property_name": name,
            "raw_value": float(value["value"]),
            "classification": {"label": value["label"], "color": value["color"]},
            "regression": None,
        }

    def score_band(name: str, index: int, raw: int) -> dict[str, Any]:
        return {
            "band_index": index,
            "property_name": name,
            "raw_value": float(raw),
            "classification": None,
            "regression": {"min_value": lo, "max_value": hi},
        }

    nodata = rank1 is None
    bands = [
        class_band("kelp", 1, rank1),
        class_band("kelp_rank2_class", 3, None if nodata else rank2),
        score_band("kelp_rank1_prob", 2, 255 if nodata else raw1),
        score_band("kelp_rank2_prob", 4, 255 if nodata else raw2),
    ]
    bands = [b for b in bands if b["property_name"] != drop]
    if first is not None:
        bands.sort(key=lambda b: b["property_name"] != first)
    return {"bands": bands}


def _two_class(lon: float, lat: float) -> Pixel:
    """The kelp map: absent west of longitude 2, present east of it.

    No-data north-east of (3, 3). The rank-1 score runs from raw 128 to 254
    and the rank-2 score is the rest of 254, so ``p1 + p2`` is 252/254 as on
    the measured result (raw 254 beside a rank-2 raw 0).
    """
    if lon > 3.0 and lat > 3.0:
        return None, 0, 0, 0
    raw1 = 128 + int((lon * 7.0 + lat * 13.0) * 10) % 127
    present = lon >= 2.0
    return (1 if present else 0), (0 if present else 1), raw1, 254 - raw1


def _three_class(lon: float, lat: float) -> Pixel:
    """Water, kelp and cloud by column, with ``p1 + p2`` below 1."""
    if lon > 3.0 and lat > 3.0:
        return None, 0, 0, 0
    raw1 = 90 + int((lon * 5.0 + lat * 11.0) * 10) % 80
    rank1 = int(lon) % 3
    return rank1, (rank1 + 1) % 3, raw1, raw1 // 2


def _prob(raw: int) -> float:
    """A uint8 top-k score as the tools read it."""
    return 0.0 if raw == 0 else (raw - 1.0) / 254.0


class _Studio:
    """Top-2 results over the box: every result id reads the same map."""

    def __init__(
        self,
        pixel: Callable[[float, float], Pixel] = _two_class,
        classes: list[dict[str, Any]] | None = None,
        *,
        score_range: tuple[float, float] = _UINT8,
        drop: Callable[[float, float], str | None] | None = None,
        first: str | None = None,
    ) -> None:
        self.pixel = pixel
        self.classes = classes if classes is not None else KELP
        self.score_range = score_range
        self.drop = drop
        self.first = first
        self.pixel_calls = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        return _result(result_id, self.classes, self.score_range)

    async def pixel_value(self, _result_id: str, lon: float, lat: float) -> Any:
        self.pixel_calls += 1
        return _record(
            self.classes,
            self.pixel(lon, lat),
            self.score_range,
            drop=self.drop(lon, lat) if self.drop else None,
            first=self.first,
        )


class _HardStudio:
    """A class map without top-k scores: one classification band."""

    def __init__(self) -> None:
        self.pixel_calls = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        return {
            "id": result_id,
            "property_names": ["landcover"],
            "result_metadata": {
                "geometry": _BOX,
                "classification_fields": [
                    {"property_name": "landcover", "allowed_values": COVER}
                ],
            },
        }

    async def pixel_value(self, *_a: Any) -> Any:
        self.pixel_calls += 1
        raise AssertionError("a class map without top-k scores is refused first")


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


async def _call(name: str, args: dict[str, Any], studio: Any = None) -> Any:
    registry = ToolRegistry()
    registry.register_all(build_review_set_tools())
    registry.register_all(build_estimation_tools())
    registry.register_all(build_compare_tools())
    ctx = ToolContext(studio=studio, state=ThreadState())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


def _expected(
    pixel: Callable[[float, float], Pixel], grid: int
) -> tuple[list[float], list[float], list[int]]:
    """Each window's margin and p1 (NaN at no-data) and rank-1 class (-1 there)."""
    margin: list[float] = []
    p1: list[float] = []
    classes: list[int] = []
    for _r, _c, lon, lat in grid_windows(_BBOX, grid):
        rank1, _rank2, raw1, raw2 = pixel(lon, lat)
        if rank1 is None:
            margin.append(math.nan)
            p1.append(math.nan)
            classes.append(-1)
        else:
            margin.append(_prob(raw1) - _prob(raw2))
            p1.append(_prob(raw1))
            classes.append(rank1)
    return margin, p1, classes


def _ids(result: dict[str, Any]) -> list[str]:
    return [c["id"] for c in result.get("forbidden_claims", [])]


# --------------------------------------------------------------------------- reading the bands


def test_a_score_is_read_by_the_range_it_declares() -> None:
    """Studio's uint8 scores: (raw - 1) / 254; raw 255 (the files' no-data
    value) is a score of 1.0, and raw 0 reads 0.0. Any other range is refused."""
    assert topk_probability(1.0, _UINT8, "kelp_rank1_prob") == 0.0
    assert topk_probability(255.0, _UINT8, "kelp_rank1_prob") == 1.0
    assert topk_probability(227.0, _UINT8, "kelp_rank1_prob") == pytest.approx(
        0.88976, abs=1e-5
    )
    assert topk_probability(0.0, _UINT8, "kelp_rank2_prob") == 0.0
    assert topk_probability(0.4, (0.0, 1.0), "kelp_rank1_prob") == 0.4
    with pytest.raises(ValueError, match=r"declares the range \[0, 100\]"):
        topk_probability(50.0, (0.0, 100.0), "kelp_rank1_prob")
    with pytest.raises(ValueError, match="declares no range"):
        topk_probability(50.0, None, "kelp_rank1_prob")
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        topk_probability(300.0, _UINT8, "kelp_rank1_prob")


def test_the_field_is_recognised_from_its_names_and_its_metadata() -> None:
    result = _result("r", KELP)
    topk = topk_scores(result)
    assert topk is not None
    assert (topk.base, topk.k) == ("kelp", 2)
    assert topk.score_properties == ("kelp_rank1_prob", "kelp_rank2_prob")
    assert topk.class_properties == ("kelp", "kelp_rank2_class")
    assert topk.labels == ["absent", "present"]
    # A rank's score or class names the same field, as a part of it.
    part = topk_scores(result, "kelp_rank1_prob")
    assert part == topk and part.reads_a_part("kelp_rank1_prob")
    assert not topk.reads_a_part(None) and not topk.reads_a_part("kelp")
    # Studio pairs P with the band of its rank-1 score; another pairing is
    # not taken for top-k scores.
    other = _result("r", KELP)
    other["result_metadata"]["classification_fields"][0]["confidence_band_index"] = 4
    assert topk_scores(other) is None
    # Without result_metadata the bands are taken by their names alone.
    bare = {"id": "r", "property_names": list(PROPS), "result_metadata": None}
    named = topk_scores(bare)
    assert named is not None and named.base == "kelp" and named.classes == ()
    # A class map without top-k scores, and a [0, 1] score, are not one.
    assert topk_scores({"property_names": ["landcover"]}) is None
    assert topk_scores({"property_names": ["sample_karst_score"]}) is None


def test_a_window_is_read_from_its_own_bands_and_its_class_band_decides() -> None:
    result = _result("r", KELP)
    topk = topk_scores(result)
    assert topk is not None
    nodata = ResultNodata(declared=declared_fields(result))
    # The pixel-value record measured on the kelp result: absent, 227 and 27.
    measured = _record(KELP, (0, 1, 227, 27))
    value, status = read_topk(measured, topk, nodata)
    assert status == OK and value is not None
    assert value[0] == "absent"
    assert value[1] == pytest.approx(226 / 254) and value[2] == pytest.approx(26 / 254)
    # A rank-1 score of raw 255 on a valid pixel is 1.0, not no-data.
    full, status = read_topk(_record(KELP, (1, 0, 255, 1)), topk, nodata)
    assert status == OK and full is not None and full[1] == 1.0
    # A no-data pixel: the class band says so, whatever the scores hold.
    assert read_topk(_record(KELP, (None, 0, 0, 0)), topk, nodata) == (None, NODATA)
    # A missing rank band is no-data, never read from another band: the
    # first-band fall back would have read the class band.
    missing = _record(KELP, (0, 1, 227, 27), drop="kelp_rank2_prob")
    assert select_band(missing, "kelp_rank2_prob")["property_name"] == "kelp"
    assert read_topk(missing, topk, nodata) == (None, NODATA)
    no_class = _record(KELP, (0, 1, 227, 27), drop="kelp")
    assert read_topk(no_class, topk, nodata) == (None, NODATA)
    assert read_topk(None, topk, nodata)[1] == "failed"


# --------------------------------------------------------------------------- the review set


@pytest.mark.asyncio
async def test_a_two_class_top2_result_gets_a_review_set_ordered_by_margin() -> None:
    studio = _Studio()
    out = await _call(
        FROM_RESULT, {"result_id": "kelp-1", "grid": 4, "budgets": [1.0]}, studio
    )
    assert out["ok"] is True, out
    result = out["result"]
    assert result["ranked"] is True and result["score_kind"] == TOPK_SCORE
    assert result["property_name"] == "kelp"
    sampling = result["sampling"]
    assert (sampling["n_windows_sampled"], sampling["n_valid"]) == (16, 15)
    assert sampling["n_nodata_dropped"] == 1
    assert studio.pixel_calls == 16
    margin, p1, classes = _expected(_two_class, 4)
    valid = [w for w in range(16) if math.isfinite(margin[w])]
    # Lowest margin first; ties as the package orders them (the later window).
    order = sorted(valid, key=lambda w: (margin[w], -w))
    review = result["review"]
    assert [r["window_index"] for r in review] == order
    for row in review:
        w = row["window_index"]
        assert row["margin"] == round(margin[w], 6)
        assert row["score"] == round(p1[w], 6)
        assert row["predicted_class"] == classes[w]
        assert row["class_name"] == KELP[classes[w]]["label"]
    assert result["classes"] == {"0": "absent", "1": "present"}
    assert "class_meaning" not in result
    assert result["top_k"]["k"] == 2 and result["top_k"]["quantized"] is True
    assert "uncalibrated" in result["assumption"] and "1/254" in result["assumption"]
    # Studio's uncalibrated, quantized softmax scores: stated, and no
    # recorded experiment grades them.
    assert result["evidence_covers_this_case"] == "no"
    assert result["must_state"] == [
        MUST_STATE_TOPK_SCORE,
        rules.topk_scores_must_state(True),
    ]
    ids = _ids(result)
    assert rules.CERTIFICATION_GUARANTEED in ids
    assert rules.ZONE_FOR_A_CLASS_MAP not in ids
    assert "design='random'" in result["next_step"]
    # Two classes whose scores sum to 252/254 are probabilities, never logits.
    assert "logit" not in json.dumps(result)

    saved = json.loads(Path(result["scores_path"]).read_text())
    assert saved["score_kind"] == TOPK_SCORE and saved["score_type"] == "probability"
    assert saved["windows"] == valid
    assert saved["map_class"] == [classes[w] for w in valid]
    assert saved["classes"] == {"0": "absent", "1": "present"}
    assert saved["p1"] == pytest.approx([p1[w] for w in valid])
    # The saved file ranks the same windows the same way, with the same contract.
    again = await _call(
        "olmoearth_review_set", {"scores_path": result["scores_path"], "budget": 1.0}
    )
    assert again["ok"] is True, again
    second = again["result"]
    assert [r["window_index"] for r in second["review"]] == order
    assert [r["class_name"] for r in second["review"]] == [
        KELP[classes[w]]["label"] for w in order
    ]
    assert second["must_state"] == result["must_state"]
    assert rules.CERTIFICATION_GUARANTEED in _ids(second)


@pytest.mark.asyncio
async def test_a_three_class_top2_result_is_ranked_and_not_called_a_logit() -> None:
    """With more than two classes p1 + p2 < 1: still probabilities."""
    studio = _Studio(_three_class, COVER)
    out = await _call(
        FROM_RESULT, {"result_id": "cover-1", "grid": 4, "budgets": [1.0]}, studio
    )
    assert out["ok"] is True, out
    result = out["result"]
    assert result["ranked"] is True and result["score_kind"] == TOPK_SCORE
    saved = json.loads(Path(result["scores_path"]).read_text())
    assert all(sum(row) < 1.0 for row in saved["scores"])
    margin, _p1, classes = _expected(_three_class, 4)
    review = result["review"]
    assert [r["margin"] for r in review] == sorted(r["margin"] for r in review)
    assert {r["predicted_class"] for r in review} == {0, 1, 2}
    for row in review:
        w = row["window_index"]
        assert row["class_name"] == COVER[classes[w]]["label"]
        assert row["margin"] == round(margin[w], 6)
    assert result["classes"] == {"0": "water", "1": "kelp", "2": "cloud"}
    assert MUST_STATE_MULTICLASS_LOGIT not in result["must_state"]
    assert "logit" not in json.dumps(result)


@pytest.mark.asyncio
async def test_a_missing_rank_band_is_no_data_in_the_grid() -> None:
    """The south-west window's rank-2 score is missing: dropped as no-data,
    never read from the class band that comes first."""

    def drop(lon: float, lat: float) -> str | None:
        return "kelp_rank2_prob" if lon < 1.0 and lat < 1.0 else None

    studio = _Studio(drop=drop)
    out = await _call(
        FROM_RESULT, {"result_id": "kelp-1", "grid": 4, "budgets": [1.0]}, studio
    )
    result = out["result"]
    assert result["ranked"] is True
    assert result["sampling"]["n_nodata_dropped"] == 2
    assert 12 not in {r["window_index"] for r in result["review"]}  # row 3, col 0


@pytest.mark.asyncio
async def test_a_score_range_other_than_uint8_or_unit_is_refused() -> None:
    studio = _Studio(score_range=(0.0, 100.0))
    out = await _call(FROM_RESULT, {"result_id": "kelp-1", "grid": 4}, studio)
    assert out["ok"] is True
    result = out["result"]
    assert result["ranked"] is False
    assert "declares the range [0, 100]" in result["reason"]
    assert result["top_k"]["class_property"] == "kelp"
    assert studio.pixel_calls == 0


@pytest.mark.asyncio
async def test_a_ranks_score_or_class_is_refused_as_the_map() -> None:
    studio = _Studio()
    for prop in ("kelp_rank1_prob", "kelp_rank2_class"):
        out = await _call(
            FROM_RESULT, {"result_id": "kelp-1", "property_name": prop}, studio
        )
        result = out["result"]
        assert result["ranked"] is False
        assert "pass property_name='kelp'" in result["reason"]
    assert studio.pixel_calls == 0


@pytest.mark.asyncio
async def test_a_class_map_without_top_k_scores_keeps_its_refusal() -> None:
    studio = _HardStudio()
    out = await _call(FROM_RESULT, {"result_id": "lc"}, studio)
    result = out["result"]
    assert result["ranked"] is False and result["class_map"] is True
    assert result["reason"] == rules.CLASS_MAP_REFUSAL
    assert result["reason"].startswith(rules.CLASS_MAP_NO_CONFIDENCE + " ")
    assert "probability_output (top_k of 2 or more)" in result["reason"]
    assert _ids(result) == [rules.ZONE_FOR_A_CLASS_MAP]
    assert studio.pixel_calls == 0


# --------------------------------------------------------------------------- the comparison


@pytest.mark.asyncio
async def test_a_comparison_reads_the_rank1_class_and_refuses_a_score() -> None:
    """The class band is read by its name, wherever it lies in the record."""
    studio = _Studio(first="kelp_rank1_prob")
    out = await _call(
        "olmoearth_compare_results",
        {"result_ids": ["kelp-1", "kelp-2"], "mode": "pair", "grid": 3},
        studio,
    )
    assert out["ok"] is True, out
    result = out["result"]
    assert result["value_type"] == "classification"
    assert result["property_a"]["property_name"] == "kelp"
    assert result["stats"]["agreement_fraction"] == 1.0
    calls = studio.pixel_calls
    refused = await _call(
        "olmoearth_compare_results",
        {
            "result_ids": ["kelp-1", "kelp-2"],
            "mode": "pair",
            "property_name": "kelp_rank1_prob",
        },
        studio,
    )
    assert refused["result"]["comparable"] is False
    assert "pass property_name='kelp'" in refused["result"]["reason"]
    assert studio.pixel_calls == calls


# --------------------------------------------------------------------------- designs and the zone


@pytest.mark.asyncio
async def test_the_confidence_design_draws_on_the_top_k_margin() -> None:
    estimate = pytest.importorskip("oe_inferencex.estimate")
    out = await _call(
        PLAN,
        {
            "result_id": "kelp-1",
            "grid": 8,
            "budget": 20,
            "design": "confidence",
            "seed": 3,
        },
        _Studio(),
    )
    assert out["ok"] is True, out
    plan = out["result"]
    design = json.loads(Path(plan["design_path"]).read_text())
    population = design["population"]
    assert population["score_kind"] == TOPK_SCORE
    assert population["classes"] == ["absent", "present"]
    margin, p1, classes = _expected(_two_class, 8)
    stored = [math.nan if m is None else m for m in population["margin"]]
    assert [m for m in stored if math.isfinite(m)] == [
        m for m in margin if math.isfinite(m)
    ]
    assert population["map_class"] == classes
    direct = estimate.sample_for_estimation(
        margin, 20, design="confidence", p1=p1, seed=3
    )
    assert design["sample"]["indices"] == direct["indices"].tolist()
    # Labelled with the class seen, as a class map is; no class-map refusal.
    assert "reference_class" in plan["how_to_label"]
    assert plan["map_classes"] == {"0": "absent", "1": "present"}
    for window in plan["windows"]:
        assert window["map_class"] == KELP[classes[window["window_index"]]]["label"]
    assert "class_map_note" not in plan
    assert rules.topk_scores_must_state(True) in plan["must_state"]
    ids = _ids(plan)
    assert rules.CERTIFICATION_GUARANTEED in ids
    assert rules.ZONE_FOR_A_CLASS_MAP not in ids


def _label_with_the_map(sheet: str, out: Path) -> str:
    """The plan's sheet, reference_class the map's class on every row, wrong blank."""
    with open(sheet, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = [dict(row, reference_class=row["map_class"]) for row in reader]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


@pytest.mark.asyncio
async def test_a_random_design_certifies_a_zone_of_the_top_k_map(
    tmp_path: Path,
) -> None:
    pytest.importorskip("oe_inferencex.estimate")
    pytest.importorskip("oe_inferencex.certify.zone")
    out = await _call(
        PLAN,
        {"result_id": "kelp-1", "grid": 16, "budget": 120, "design": "random"},
        _Studio(),
    )
    assert out["ok"] is True, out
    plan = out["result"]
    labels = _label_with_the_map(plan["labels_csv_path"], tmp_path / "labelled.csv")
    certified = await _call(
        CERTIFY,
        {"design_path": plan["design_path"], "labels_path": labels, "alpha": 0.1},
    )
    assert certified["ok"] is True, certified
    zone = certified["result"]
    assert zone["certified"] is True and "class_map" not in zone
    assert zone["verdict"].startswith("certified: the ")
    assert Path(zone["zone_path"]).is_file()
    assert any(s.startswith("Only the ") for s in zone["must_state"])
    assert rules.topk_scores_must_state(True) in zone["must_state"]
    ids = _ids(zone)
    assert rules.CERTIFICATION_GUARANTEED in ids
    assert rules.ZONE_FOR_A_CLASS_MAP not in ids
    # A certified zone of it, stated, is no claim about a class map.
    run = RunEvidence(
        tools=[ToolRecord(CERTIFY, {"design_path": plan["design_path"]}, certified)],
        user_messages=["Where can this kelp map be trusted?"],
    )
    sentence = (
        f"The {zone['n_zone']} most confident windows of the class map with top-k "
        "scores are certified."
    )
    assert check_forbidden_claims(sentence, run) == []
    # The same labels, graded: per-class accuracy by class name, no class-map
    # note, and the valid interval stated without "a class map has no
    # certified zone".
    graded = await _call(
        GRADE, {"design_path": plan["design_path"], "labels_path": labels}
    )
    assert graded["ok"] is True, graded
    result = graded["result"]
    assert result["classes"] == {"0": "absent", "1": "present"}
    assert "class_map_note" not in result
    assert result["estimate"] == 0.0
    names = {row.get("class") for row in result["per_class"]["per_class"].values()}
    assert names <= {"absent", "present"} and names
    assert rules.ZONE_FOR_A_CLASS_MAP not in _ids(result)
    assert not any("class map has no certified zone" in s for s in result["must_state"])


# --------------------------------------------------------------------------- the answer checks


def test_a_run_with_a_class_map_and_a_top_k_map_reads_each_zone_right() -> None:
    """A zone stated for the class map with top-k scores passes; one stated
    for the class map without them, by its kind or its result id, is flagged."""
    hard = ToolRecord(
        PLAN,
        {"result_id": "lc-1", "design": "random"},
        {
            "ok": True,
            "result": {
                "forbidden_claims": [{"id": rules.ZONE_FOR_A_CLASS_MAP, "why": "x"}]
            },
        },
    )
    topk = ToolRecord(
        CERTIFY,
        {"design_path": "kelp.json", "alpha": 0.1},
        {
            "ok": True,
            "result": {
                "certified": True,
                "design": "random",
                "forbidden_claims": [
                    {"id": rules.CERTIFICATION_GUARANTEED, "why": "x"}
                ],
            },
        },
    )
    run = RunEvidence(tools=[hard, topk], user_messages=[""])
    for fine in (
        "The class map with top-k scores is certified at coverage 0.4, alpha 0.1.",
        "The top-2 class map's 96 most confident windows are certified.",
        f"Its zone is only the most confident windows of the {rules.TOPK_MAP_KIND}, "
        "certified at alpha 0.1.",
    ):
        assert check_forbidden_claims(fine, run) == [], fine
    for flagged in (
        "The class map's top half is certified at alpha 0.05.",
        "The top half of lc-1 is certified at alpha 0.05.",
        "The land-cover classification result has a certified zone too.",
    ):
        violations = check_forbidden_claims(flagged, run)
        assert [v["text"] for v in violations] == [flagged], flagged
        assert violations[0]["detail"].startswith(rules.ZONE_FOR_A_CLASS_MAP)


@pytest.mark.asyncio
async def test_a_top_k_review_set_lets_its_zone_through_beside_a_class_map() -> None:
    """The top-k review set forbids certification_guaranteed, as a scores map's
    results do: beside a class map's plan, only the class map's zone is read."""
    out = await _call(FROM_RESULT, {"result_id": "kelp-1", "grid": 4}, _Studio())
    hard = ToolRecord(
        PLAN,
        {"result_id": "lc-1", "design": "random"},
        {
            "ok": True,
            "result": {
                "forbidden_claims": [{"id": rules.ZONE_FOR_A_CLASS_MAP, "why": "x"}]
            },
        },
    )
    run = RunEvidence(
        tools=[hard, ToolRecord(FROM_RESULT, {"result_id": "kelp-1"}, out)],
        user_messages=[""],
    )
    fine = "A random plan of the class map with top-k scores may certify a zone."
    assert check_forbidden_claims(fine, run) == []
    flagged = "The class map's top half is certified at alpha 0.05."
    assert [v["text"] for v in check_forbidden_claims(flagged, run)] == [flagged]


@pytest.mark.asyncio
async def test_a_top_k_review_sets_own_sentences_pass_its_checks() -> None:
    """An answer made of the result's sentences, its scores' statement among
    them, makes no claim the result forbids and conveys what it must state."""
    out = await _call(FROM_RESULT, {"result_id": "kelp-1", "grid": 4}, _Studio())
    result = out["result"]
    said = (
        [f["sentence"] for f in result["facts"]]
        + list(result["must_state"])
        + [result["evidence_scope"], result["assumption"], TOPK_ZONE_STEP]
    )
    run = RunEvidence(
        tools=[ToolRecord(FROM_RESULT, {"result_id": "kelp-1"}, out)],
        user_messages=["Which windows of this kelp map should I check first?"],
    )
    answer = "\n\n".join(said)
    assert check_forbidden_claims(answer, run) == []
    assert check_must_state(answer, run) == []
