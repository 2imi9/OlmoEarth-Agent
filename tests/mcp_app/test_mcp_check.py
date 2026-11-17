# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Check card's records: the key, the zones, the counts offered, the one draw.

``mcp_app/check.py`` keeps, per zone of a Studio result, the label counts
the package plans (the plan tool's own call and sentence), the draw a press
makes (its settings pinned at the first press, under a lock another process
honours) and one log line per zone pressed. These tests run without a
network; the ones that plan with the package run it (``uv sync
--all-extras``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.mcp_app import check, sheet
from olmoearth_agent.tools import estimation, inferencex
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

PLAN = "olmoearth_plan_label_sample"
FILES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")
OFFER = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty.", "rate_note": "r"},
        {"labels": 40, "width": 0.1, "sentence": "Forty.", "rate_note": "r"},
    ],
    "refused": None,
}
#: A plan's forbidden claims, as the plan tool writes them.
CLAIMS = [
    {"id": "subset_labelling_sufficient", "why": "Label every window drawn."},
    {"id": "error_rate_without_labels", "why": "No rate before the labels."},
]
#: A plan's facts, as the plan tool writes them (an id, a sentence, fields).
FACTS = [
    {"id": "prefix_is_random_sample", "sentence": "Label every row.", "n_rows": 20},
    {"id": "label_budget", "sentence": "Planned before any label.", "labels": 20},
]


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores and outputs), a folder of the test's own."""
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    return out.resolve()


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts offered without the package: 20 and 40, for every zone."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`OFFER`."""
        answer: dict[str, Any] = json.loads(json.dumps(OFFER))
        return answer

    monkeypatch.setattr(check, "offers", fake)


def _inner(
    root: Path,
    zones: tuple[tuple[str, int | None], ...] = ((FILES[0], 600), (FILES[1], 300)),
    digest: str = "a" * 63,
) -> dict[str, Any]:
    """A scores_from_studio result: one entry per ``(file, n_valid)``, a zone
    with ``n_valid`` None refused; each pooled zone's scores file exists."""
    listed: list[dict[str, Any]] = []
    for i, (file, n_valid) in enumerate(zones):
        if n_valid is None:
            listed.append(
                {"file": file, "scores_path": None, "refused": "smaller than a patch"}
            )
            continue
        path = root / f"scores_{i}.json"
        path.write_text("{}", encoding="utf-8")
        listed.append(
            {
                "file": file,
                "scores_path": str(path),
                "pooled": {
                    "n_valid": n_valid,
                    "n_windows": n_valid + 4,
                    "raster_check": {"sha256": f"{digest}{i}"},
                },
            }
        )
    facts = []
    if len(zones) > 1:
        share = round(1 / len(zones), 6)
        facts.append(
            {
                "id": "studio_zones",
                "sentence": "Several zones.",
                "zones": [{"file": f, "share": share} for f, _ in zones],
            }
        )
    return {
        "result_id": "res-1",
        "prediction_id": "pred-1",
        "zones": listed,
        "facts": facts,
    }


def _plan_answer(root: Path) -> dict[str, Any]:
    """A drawn plan's envelope, with the windows and figure a draw never keeps."""
    return {
        "ok": True,
        "result": {
            "available": True,
            "design": "random",
            "budget": 20,
            "seed": 0,
            "labels_csv_path": str(root / "plan_to_label.csv"),
            "design_path": str(root / "plan.json"),
            "n_population": 600,
            "windows": [{"window_index": 3, "lon": 1.5, "lat": 2.5, "map_class": 1}],
            "figure_path": str(root / "plan_figure.svg"),
            "facts": json.loads(json.dumps(FACTS)),
            "must_state": ["One sentence to state."],
            "forbidden_claims": json.loads(json.dumps(CLAIMS)),
        },
    }


def _draw_path(root: Path, key: str) -> Path:
    """Where a check's draw record is."""
    return root / "checks" / f"{key}.draw.json"


def _mode(path: Path) -> int:
    """A file's permission bits."""
    return path.stat().st_mode & 0o777


async def _call(name: str, args: dict[str, Any]) -> Any:
    """One call of the estimation tools through the registry, as the server makes it."""
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    ctx = ToolContext(studio=None, state=ThreadState())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


def _logits(n: int, seed: int = 4) -> list[list[float]]:
    """Three-class logits for ``n`` windows."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(3)] for _ in range(n)]


# --------------------------------------------------------------------------- the key


def test_the_key_is_stable_and_24_hex() -> None:
    """A check is one zone of one result: a SHA-256 of the two, cut to 24."""
    key = check.check_key("res-1", FILES[0])
    assert re.fullmatch(r"[0-9a-f]{24}", key)
    assert key == check.check_key(" res-1 ", FILES[0])
    assert key == hashlib.sha256(f"res-1\n{FILES[0]}".encode()).hexdigest()[:24]
    assert key != check.check_key("res-1", FILES[1])
    assert key != check.check_key("res-2", FILES[0])


@pytest.mark.parametrize(
    "bad", ["", "../../etc/passwd", "A" * 24, "g" * 24, "0" * 23, "0" * 25, 7, None]
)
def test_a_bad_key_is_refused_before_any_file(root: Path, bad: Any) -> None:
    """A key the page sends is checked before the records' folder is touched."""
    with pytest.raises(ValueError, match="not one of this card's checks"):
        check.read_offer(bad)
    with pytest.raises(ValueError, match="not one of this card's checks"):
        check.claim_draw(bad, 20)
    with pytest.raises(ValueError, match="not one of this card's checks"):
        check.finish_draw(bad, {"ok": True})
    with pytest.raises(ValueError, match="not one of this card's checks"):
        check.fail_draw(bad, "no", retry=True)
    assert not (root / "checks").exists()


# --------------------------------------------------------------------------- the zones


def test_the_zone_names() -> None:
    """A zone is named by its file's EPSG code; two files of one code by number."""
    assert check.zone_label("result_epsg32610_0_native.tif") == "UTM zone 10N"
    assert check.zone_label("result_epsg32733_1_native.tif") == "UTM zone 33S"
    assert check.zone_label("result_epsg3857_0_native.tif") == "EPSG 3857"
    assert check.zone_label("result_epsg32600_0_native.tif") == "EPSG 32600"
    assert check.zone_label("result_epsg32761_0_native.tif") == "EPSG 32761"
    assert check.zone_label("scores.tif") == "scores.tif"
    files = [
        "result_epsg32610_0_native.tif",
        "result_epsg32610_1_native.tif",
        "result_epsg32611_2_native.tif",
    ]
    assert [check.zone_label(f, files) for f in files] == [
        "UTM zone 10N (file 1)",
        "UTM zone 10N (file 2)",
        "UTM zone 11N",
    ]


def test_offer_zones_writes_one_offer_per_zone(root: Path, canned: None) -> None:
    """Each zone gets its offer record; a refused zone offers no count; the
    card's zones carry no scores path, raster digest or folder."""
    inner = _inner(
        root,
        ((FILES[0], 600), (FILES[1], 300), ("result_epsg3857_2_native.tif", None)),
    )
    zones = check.offer_zones(inner)
    assert [z["zone"] for z in zones] == ["UTM zone 10N", "UTM zone 33S", "EPSG 3857"]
    assert [z["share"] for z in zones] == ["33.3%"] * 3
    assert [z["n_valid"] for z in zones] == [600, 300, None]
    assert [z["state"] for z in zones] == ["to draw"] * 3
    assert zones[0]["offer"] == OFFER["offer"] and "refused" not in zones[0]
    assert "refused_at" not in zones[0]
    assert zones[2]["refused"] == "smaller than a patch" and "offer" not in zones[2]
    # Not read as a map, which the card says apart from no count planned.
    assert zones[2]["refused_at"] == check.REFUSED_READ
    folder = root / "checks"
    assert _mode(folder) == 0o700
    for zone, entry in zip(zones, inner["zones"]):
        assert zone["check_key"] == check.check_key("res-1", entry["file"])
        path = folder / f"{zone['check_key']}.json"
        assert _mode(path) == 0o600
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["format"] == check.CHECK_FORMAT
        assert (record["result_id"], record["prediction_id"]) == ("res-1", "pred-1")
        assert record["zone_file"] == entry["file"]
        assert record["scores_path"] == entry["scores_path"]
        assert record["offered_at"].endswith("Z")
    refused = check.read_offer(zones[2]["check_key"])
    assert refused["offer"] == [] and refused["refused"] == "smaller than a patch"
    assert check.read_offer(zones[0]["check_key"])["raster_sha256"] == "a" * 63 + "0"
    text = json.dumps(zones)
    for left_out in ("scores_path", "raster_sha256", "a" * 63, str(root)):
        assert left_out not in text


def test_a_result_of_one_zone_has_no_share(root: Path, canned: None) -> None:
    """One zone is the whole result: no share is stated."""
    (zone,) = check.offer_zones(_inner(root, ((FILES[0], 600),)))
    assert zone["share"] is None and zone["zone"] == "UTM zone 10N"


def test_offer_zones_needs_a_result_id(root: Path, canned: None) -> None:
    """A result without its id names no check."""
    with pytest.raises(ValueError, match="no result_id"):
        check.offer_zones({**_inner(root), "result_id": " "})


def test_a_missing_package_refuses_the_zone(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the package no count is planned: the zone says why."""

    def missing(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """The package's import failing."""
        raise inferencex.InferencexMissingError("no module named oe_inferencex")

    monkeypatch.setattr(check, "offers", missing)
    (zone,) = check.offer_zones(_inner(root, ((FILES[0], 600),)))
    assert "offer" not in zone
    assert zone["refused_at"] == check.REFUSED_PLAN
    assert "needs the optional package olmoearth-inferencex" in zone["refused"]


def test_a_root_that_cannot_be_written_is_refused(root: Path, canned: None) -> None:
    """No record can be written, so no zone is offered for a press."""
    if os.geteuid() == 0:
        pytest.skip("the superuser writes any folder")
    inner = _inner(root)
    root.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="cannot be written"):
            check.offer_zones(inner)
    finally:
        root.chmod(0o700)


# --------------------------------------------------------------------------- the counts


class _FakePlan:
    """``oe_inferencex.plan`` with set answers, counting its calls."""

    labels: dict[float, int | None] = {}
    calls: list[tuple[Any, ...]] = []

    @classmethod
    def plan_error_rate(cls, n: int, width: float, rate: Any) -> dict[str, Any]:
        """The set answer for ``width``; a population under one window is refused."""
        cls.calls.append((n, width, rate))
        if n < 1:
            raise ValueError("the number of windows must be at least 1")
        labels = cls.labels[width]
        out: dict[str, Any] = {
            "question": "error_rate",
            "n_population": n,
            "width": width,
            "error_rate": 0.5,
            "error_rate_stated": False,
            "labels": labels,
            "probability_at_labels": None if labels is None else 0.93,
            "probability_by_budget": {"10": 0.1},
            "rate_note": "planned at 50% wrong",
        }
        if labels is None:
            out["refusal"] = "no budget checked gives an interval this narrow"
        return out


@pytest.fixture
def fake_plan(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The package's planner replaced by :class:`_FakePlan`, the cache emptied."""
    _FakePlan.calls = []
    monkeypatch.setattr(inferencex, "load", lambda name: _FakePlan)
    check._planned.cache_clear()
    yield _FakePlan
    check._planned.cache_clear()


def test_the_counts_are_cached_ascending_and_once_each(fake_plan: Any) -> None:
    """Three widths planned once per population; a count two widths share is
    one chip, with the narrower width's sentence; the cache is not the
    caller's to change."""
    fake_plan.labels = {0.2: 50, 0.1: 120, 0.05: 120}
    first = check.offers(777)
    again = check.offers(777)
    assert len(fake_plan.calls) == 3
    assert all(rate is None for _, _, rate in fake_plan.calls)
    assert first == again and first["refused"] is None
    assert [o["labels"] for o in again["offer"]] == [50, 120]
    assert [o["width"] for o in again["offer"]] == [0.2, 0.05]
    # r4i F10: the width without "points" (Studio grid points only).
    assert "no wider than 5% from its low end" in again["offer"][1]["sentence"]
    assert " points" not in again["offer"][1]["sentence"]
    assert all(o["rate_note"] == "planned at 50% wrong" for o in again["offer"])
    first["offer"].clear()
    assert len(check.offers(777)["offer"]) == 2


def test_a_count_above_the_windows_is_dropped_and_none_left_is_refused(
    fake_plan: Any,
) -> None:
    """Counts above the windows are dropped; with none left the planner's own
    refusal sentence is the reason, as it is for a population it refuses."""
    fake_plan.labels = {0.2: None, 0.1: 9, 0.05: 9}
    planned = check.offers(5)
    assert planned["offer"] == []
    advice = {"n_population": 5, "width": 0.2, "labels": None}
    advice["refusal"] = "no budget checked gives an interval this narrow"
    expected = estimation._label_budget_fact(advice, "random", 0)["sentence"]
    assert planned["refused"] == expected
    assert planned["refused"].startswith("No label budget is advised")
    empty = check.offers(0)
    assert empty["offer"] == []
    assert "the number of windows must be at least 1" in empty["refused"]


def test_the_counts_are_the_packages() -> None:
    """Each count is plan.plan_error_rate's at its width, at most the windows."""
    plan_pkg = pytest.importorskip("oe_inferencex.plan")
    planned = check.offers(400)
    assert planned["refused"] is None
    counts = [o["labels"] for o in planned["offer"]]
    assert counts == sorted(set(counts)) and all(1 <= c <= 400 for c in counts)
    for width in check.CHECK_WIDTHS:
        direct = estimation._jsonable(plan_pkg.plan_error_rate(400, width, None))
        assert direct["labels"] in counts
    for offer in planned["offer"]:
        direct = estimation._jsonable(plan_pkg.plan_error_rate(400, offer["width"]))
        assert offer["labels"] == direct["labels"]
        assert offer["rate_note"] == direct["rate_note"]
        # The reader test's A/check-6 (the no-rate branch, estimation.py).
        assert offer["sentence"].endswith(f"This plan picks {offer['labels']} windows.")
        assert "the note below says more" in offer["sentence"]


@pytest.mark.asyncio
async def test_each_counts_sentence_is_the_plan_tools_own(root: Path) -> None:
    """The plan tool, asked for the chip's width and count, writes the chip's
    sentence (the class design adds its own note after it) and rate_note."""
    pytest.importorskip("oe_inferencex.estimate")
    scores = _logits(1000)
    planned = check.offers(len(scores))
    assert planned["offer"]
    for offer in planned["offer"]:
        out = await _call(
            PLAN,
            {
                "scores": scores,
                "design": "class",
                "min_per_class": 5,
                "width": offer["width"],
                "budget": offer["labels"],
            },
        )
        assert out["ok"] is True, out
        plan = out["result"]
        (fact,) = [f for f in plan["facts"] if f["id"] == "label_budget"]
        assert fact["sentence"] == offer["sentence"] + (
            estimation.CLASS_DESIGN_BUDGET_NOTE
        )
        assert plan["label_budget"]["rate_note"] == offer["rate_note"]


# --------------------------------------------------------------------------- the press


def test_a_first_press_pins_the_request(root: Path, canned: None) -> None:
    """The first press writes the draw, drawing, with its settings pinned."""
    zone, _ = check.offer_zones(_inner(root))
    claim = check.claim_draw(zone["check_key"], 20)
    assert claim["outcome"] == "first"
    draw = claim["draw"]
    assert draw["state"] == "drawing" and draw["pid"] == os.getpid()
    assert (draw["design"], draw["seed"], draw["budget"]) == ("random", 0, 20)
    assert draw["request"] == {
        "scores_path": str(root / "scores_0.json"),
        "design": "random",
        "budget": 20,
        "seed": 0,
    }
    assert draw["raster_sha256"] == "a" * 63 + "0"
    path = _draw_path(root, zone["check_key"])
    assert json.loads(path.read_text(encoding="utf-8")) == draw
    assert _mode(path) == 0o600


def test_a_count_not_offered_and_an_unknown_check_are_refused(
    root: Path, canned: None
) -> None:
    """Only a count the card offered draws, and only for a check written here."""
    zones = check.offer_zones(
        _inner(root, ((FILES[0], 600), ("result_epsg3857_1_native.tif", None)))
    )
    key = zones[0]["check_key"]
    for asked in (30, True, "20", 20.5, None):
        with pytest.raises(ValueError, match=re.escape(check.NOT_OFFERED)):
            check.claim_draw(key, asked)
    with pytest.raises(ValueError, match=re.escape(check.NOT_OFFERED)):
        check.claim_draw(zones[1]["check_key"], 20)
    assert not _draw_path(root, key).exists()
    with pytest.raises(ValueError, match="not known here"):
        check.claim_draw("0" * 24, 20)
    with pytest.raises(ValueError, match="not known here"):
        check.read_offer("0" * 24)
    with pytest.raises(ValueError, match="was not started here"):
        check.finish_draw(key, _plan_answer(root))


def test_a_drawn_zone_repeats_with_its_pinned_count(root: Path, canned: None) -> None:
    """A second press, with another count, gets the stored draw; the draw keeps
    the plan's facts, must_state and forbidden claims as the plan gave them,
    and nothing of the plan's windows or figure; the card's zone carries the
    same facts."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    draw = check.finish_draw(key, _plan_answer(root))
    assert draw["state"] == "drawn"
    assert draw["labels_csv_path"] == str(root / "plan_to_label.csv")
    assert draw["design_path"] == str(root / "plan.json")
    assert draw["must_state"] == ["One sentence to state."]
    assert draw["forbidden_claims"] == CLAIMS
    assert draw["facts"] == FACTS
    assert check.offer_zones(_inner(root))[0]["draw"]["facts"] == FACTS
    assert draw["n_population"] == 600 and draw["drawn_at"].endswith("Z")
    assert not {"windows", "figure_path", "map_class"} & set(draw)
    before = _draw_path(root, key).read_bytes()
    again = check.claim_draw(key, 40)
    assert again["outcome"] == "repeat" and again["draw"] == draw
    assert again["draw"]["budget"] == again["draw"]["request"]["budget"] == 20
    # Whatever count it sends: one the card does not offer gets the draw too.
    for asked in (30, None):
        assert check.claim_draw(key, asked) == {"outcome": "repeat", "draw": draw}
    assert _draw_path(root, key).read_bytes() == before
    # A drawn record is kept as it is, whatever comes after.
    assert check.fail_draw(key, "late", retry=True) == draw
    assert check.finish_draw(key, {"ok": False, "error": "RuntimeError: x"}) == draw
    assert _draw_path(root, key).read_bytes() == before


def test_a_press_while_drawing_is_busy_and_a_stale_draw_is_retried(
    root: Path, canned: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under STALE_SECONDS another press waits ("busy"); after it, the draw is
    taken as stopped and drawn again with the pinned request."""
    clock = [1_000_000.0]
    monkeypatch.setattr(check, "_now", lambda: clock[0])
    inner = _inner(root)
    zone, _ = check.offer_zones(inner)
    key = zone["check_key"]
    first = check.claim_draw(key, 20)["draw"]
    clock[0] += check.STALE_SECONDS - 1
    assert check.claim_draw(key, 40)["outcome"] == "busy"
    assert check.offer_zones(inner)[0]["state"] == "busy"
    clock[0] += 2
    stale = check.offer_zones(inner)[0]
    assert (stale["state"], stale["error"], stale["retry"]) == (
        "failed",
        check.STOPPED,
        True,
    )
    claim = check.claim_draw(key, 40)
    assert claim["outcome"] == "retry"
    draw = claim["draw"]
    assert draw["request"] == first["request"] and draw["budget"] == 20
    assert draw["attempts"] == 2 and draw["started_at"] == clock[0]
    assert draw["pressed_at"] == first["pressed_at"]


def test_a_failure_that_may_pass_is_retried_with_the_pinned_request(
    root: Path, canned: None
) -> None:
    """A plan that failed for a reason a second press may not meet is retried,
    with the first press's count, never the one sent now."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    error = "OSError: [Errno 28] No space left on device"
    draw = check.finish_draw(key, {"ok": False, "error": error})
    assert (draw["state"], draw["error"], draw["retry"]) == ("failed", error, True)
    shown = check.offer_zones(_inner(root))[0]
    assert (shown["state"], shown["error"], shown["retry"]) == ("failed", error, True)
    claim = check.claim_draw(key, 40)
    assert claim["outcome"] == "retry"
    assert claim["draw"]["request"]["budget"] == 20
    assert not {"error", "retry", "failed_at"} & set(claim["draw"])
    assert check.finish_draw(key, _plan_answer(root))["state"] == "drawn"


def test_a_retry_keeps_its_pinned_count_when_the_offer_is_written_again(
    root: Path, canned: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure that may pass, then the card read again with other counts
    planned (the same files): a press with the pinned count retries with the
    pinned request, one with neither that count nor an offered one is
    refused, and the drawn zone then repeats whatever count is sent. While
    the newest offer holds no count (the package missing), a retry waits."""
    inner = _inner(root)
    first, second = check.offer_zones(inner)
    for zone, count in ((first, 20), (second, 40)):
        check.claim_draw(zone["check_key"], count)
        check.finish_draw(zone["check_key"], {"ok": False, "error": "OSError: I/O"})
    other = {
        "offer": [
            {"labels": 50, "width": 0.2, "sentence": "Fifty.", "rate_note": "r"},
            {"labels": 90, "width": 0.1, "sentence": "Ninety.", "rate_note": "r"},
        ],
        "refused": None,
    }

    def rewritten(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """The counts another package plans: 50 and 90."""
        answer: dict[str, Any] = json.loads(json.dumps(other))
        return answer

    monkeypatch.setattr(check, "offers", rewritten)
    shown = check.offer_zones(inner)[0]
    assert [o["labels"] for o in shown["offer"]] == [50, 90]
    assert (shown["state"], shown["retry"], shown["budget"]) == ("failed", True, 20)
    assert "raster_changed" not in shown
    key = first["check_key"]
    before = _draw_path(root, key).read_bytes()
    for asked in (30, None):
        with pytest.raises(ValueError, match=re.escape(check.NOT_OFFERED)):
            check.claim_draw(key, asked)
    assert _draw_path(root, key).read_bytes() == before
    claim = check.claim_draw(key, 20)
    assert claim["outcome"] == "retry"
    assert claim["draw"]["request"]["budget"] == claim["draw"]["budget"] == 20
    drawn = check.finish_draw(key, _plan_answer(root))
    assert drawn["state"] == "drawn"
    assert check.claim_draw(key, 30) == {"outcome": "repeat", "draw": drawn}

    def missing(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """The package's import failing."""
        raise inferencex.InferencexMissingError("no module named oe_inferencex")

    monkeypatch.setattr(check, "offers", missing)
    # No count planned now: the retry is refused and the zone stays retryable.
    waiting = check.offer_zones(inner)[1]
    assert "offer" not in waiting and waiting["refused_at"] == check.REFUSED_PLAN
    assert (waiting["state"], waiting["retry"], waiting["budget"]) == (
        "failed",
        True,
        40,
    )
    held = _draw_path(root, second["check_key"]).read_bytes()
    with pytest.raises(ValueError, match=re.escape(check.NOT_OFFERED)):
        check.claim_draw(second["check_key"], 40)
    assert _draw_path(root, second["check_key"]).read_bytes() == held


def test_the_card_names_a_zones_pinned_count_and_its_plans_notes(
    root: Path, canned: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed zone, or one left drawing, carries the count and time its first
    press pinned (the card's retry names them); a drawn zone carries the
    plan's must_state, each line as the plan wrote it."""
    clock = [1_000_000.0]
    monkeypatch.setattr(check, "_now", lambda: clock[0])
    inner = _inner(root)
    zone, _ = check.offer_zones(inner)
    key = zone["check_key"]
    pressed = check.claim_draw(key, 20)["draw"]["pressed_at"]
    clock[0] += check.STALE_SECONDS
    stopped = check.offer_zones(inner)[0]
    assert (stopped["state"], stopped["retry"]) == ("failed", True)
    assert (stopped["budget"], stopped["pressed_at"]) == (20, pressed)
    check.claim_draw(key, 40)
    check.finish_draw(key, {"ok": False, "error": "OSError: [Errno 5] I/O"})
    failed = check.offer_zones(inner)[0]
    assert (failed["state"], failed["budget"], failed["pressed_at"]) == (
        "failed",
        20,
        pressed,
    )
    check.claim_draw(key, 40)
    check.finish_draw(key, _plan_answer(root))
    drawn = check.offer_zones(inner)[0]
    assert drawn["state"] == "drawn" and "budget" not in drawn
    assert drawn["draw"]["must_state"] == ["One sentence to state."]
    assert (drawn["draw"]["budget"], drawn["draw"]["pressed_at"]) == (20, pressed)


def test_a_first_press_records_the_press_time_it_is_given(
    root: Path, canned: None
) -> None:
    """A first press records the press's time the server took (pressed_at), a
    retry keeps the first press's, and a press given none is stamped now."""
    first, second = check.offer_zones(_inner(root))
    pressed = "2001-01-01T00:00:00Z"
    claim = check.claim_draw(first["check_key"], 20, pressed)
    assert (claim["outcome"], claim["draw"]["pressed_at"]) == ("first", pressed)
    check.finish_draw(first["check_key"], {"ok": False, "error": "OSError: [Errno 5]"})
    again = check.claim_draw(first["check_key"], 20, "2001-01-01T00:05:00Z")
    assert (again["outcome"], again["draw"]["pressed_at"]) == ("retry", pressed)
    stamped = check.claim_draw(second["check_key"], 20)["draw"]["pressed_at"]
    assert stamped != pressed and stamped.endswith("Z")


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (
            {
                "ok": False,
                "error": "ValueError: /x/p_to_label.csv holds labels, and its design "
                "file cannot be read: writing this plan would clear those labels, "
                "so neither file was written. Move /x/p_to_label.csv aside",
            },
            None,
        ),
        (
            {"ok": False, "error": "invalid arguments for olmoearth_plan: budget"},
            None,
        ),
        (
            {
                "ok": True,
                "result": {"available": False, "reason": "needs the optional package"},
            },
            "needs the optional package",
        ),
        ({"ok": True, "result": {"available": True}}, check.NO_SHEET),
    ],
)
def test_a_failure_the_same_settings_give_again_is_not_retried(
    root: Path, canned: None, answer: dict[str, Any], error: str | None
) -> None:
    """A refusal, a plan answering available false, or one with no sheet is a
    failed draw a press does not run again; its error is kept whole."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    draw = check.finish_draw(key, answer)
    assert draw["state"] == "failed" and draw["retry"] is False
    assert draw["error"] == (answer.get("error") if error is None else error)
    before = _draw_path(root, key).read_bytes()
    again = check.claim_draw(key, 20)
    assert again["outcome"] == "failed" and again["error"] == draw["error"]
    assert _draw_path(root, key).read_bytes() == before


def test_a_moved_scores_file_is_not_retried(root: Path, canned: None) -> None:
    """Once the pinned scores file is gone, no press can draw from it."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    request = check.claim_draw(key, 20)["draw"]["request"]
    Path(request["scores_path"]).unlink()
    draw = check.finish_draw(key, {"ok": False, "error": "OSError: gone"})
    assert draw["retry"] is False
    assert check.fail_draw(key, "RuntimeError: x", retry=False)["retry"] is False


def test_a_changed_raster_is_refused(root: Path, canned: None) -> None:
    """Studio's files changed under a drawn zone: the same key, the press refused."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    check.finish_draw(key, _plan_answer(root))
    changed, _ = check.offer_zones(_inner(root, digest="b" * 63))
    assert changed["check_key"] == key
    assert changed["state"] == "drawn" and changed["raster_changed"] is True
    claim = check.claim_draw(key, 20)
    assert claim["outcome"] == "changed" and claim["error"] == check.RASTER_CHANGED
    same, _ = check.offer_zones(_inner(root))
    assert "raster_changed" not in same
    assert check.claim_draw(key, 20)["outcome"] == "repeat"


@pytest.mark.skipif(fcntl is None, reason="no file locks on this platform")
def test_a_second_holder_of_the_lock_makes_the_press_wait(
    root: Path, canned: None
) -> None:
    """While another process holds a check's file lock, a press waits for it."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    holder = os.open(root / "checks" / f"{key}.lock", os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    out: dict[str, Any] = {}

    def press() -> None:
        """The press, in a thread of its own."""
        out.update(check.claim_draw(key, 20))

    worker = threading.Thread(target=press, daemon=True)
    try:
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not out
        assert not _draw_path(root, key).exists()
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    worker.join(10)
    assert not worker.is_alive()
    assert out["outcome"] == "first"


def test_one_log_line_per_zone_pressed(root: Path, canned: None) -> None:
    """Each zone pressed adds one line, with the count pinned and the count
    asked, at the press's time when it is given; a key that is not a check's
    is not written."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    draw = check.finish_draw(key, _plan_answer(root))
    pressed = "2026-11-12T08:00:00Z"
    check.append_log(
        check.log_entry(key, outcome="first", asked=20, draw=draw, pressed_at=pressed)
    )
    repeat = check.claim_draw(key, 40)
    check.append_log(
        check.log_entry(key, outcome="repeat", asked=40, draw=repeat["draw"])
    )
    offer = check.read_offer(key)
    check.append_log(check.log_entry(key, outcome="refused", asked=30, offer=offer))
    check.append_log({"check_key": "../x", "outcome": "refused", "asked": "lots"})
    log = root / "checks" / check.LOG_NAME
    assert _mode(log) == 0o600
    rows = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert len(rows) == 4
    assert set(rows[0]) == {
        "pressed_at",
        "check_key",
        "result_id",
        "zone_file",
        "budget",
        "asked",
        "design",
        "seed",
        "outcome",
        "state",
    }
    assert [r["outcome"] for r in rows] == ["first", "repeat", "refused", "refused"]
    assert [r["budget"] for r in rows] == [20, 20, 30, None]
    assert [r["asked"] for r in rows] == [20, 40, 30, None]
    assert [r["check_key"] for r in rows] == [key, key, key, None]
    assert (rows[0]["result_id"], rows[0]["zone_file"]) == ("res-1", FILES[0])
    assert (rows[0]["design"], rows[0]["seed"], rows[0]["state"]) == (
        "random",
        0,
        "drawn",
    )
    assert rows[0]["pressed_at"] == pressed
    assert all(r["pressed_at"].endswith("Z") for r in rows)


def test_the_page_gets_no_window_class_or_figure(root: Path, canned: None) -> None:
    """The page's copy of a draw: its settings, sheet and notes, nothing else."""
    zone, _ = check.offer_zones(_inner(root))
    key = zone["check_key"]
    check.claim_draw(key, 20)
    draw = check.finish_draw(key, _plan_answer(root))
    view = check.draw_view(draw, outcome="first")
    assert set(view) == {*check.VIEW_KEYS, "outcome"}
    assert not {"forbidden_claims", "facts", "request"} & set(view)
    assert (view["state"], view["outcome"], view["budget"]) == ("drawn", "first", 20)
    assert view["labels_csv_path"] == str(root / "plan_to_label.csv")
    assert view["must_state"] == ["One sentence to state."]
    text = json.dumps({k: v for k, v in view.items() if k != "labels_csv_path"})
    for left_out in ("window", "map_class", "figure", "scores", "raster", "lon"):
        assert left_out not in text
    refused = check.draw_view(draw, outcome="changed", error=check.RASTER_CHANGED)
    assert refused["error"] == check.RASTER_CHANGED


def test_a_sheet_csv_cannot_read_fails_its_zone_only(root: Path, canned: None) -> None:
    """A drawn zone's sheet re-saved with a cell past csv's field limit (a quote
    never closed): that zone says its sheet cannot be read, and the card still
    lists every zone with its state."""
    inner = _inner(root)
    zone, _ = check.offer_zones(inner)
    check.claim_draw(zone["check_key"], 20)
    check.finish_draw(zone["check_key"], _plan_answer(root))
    (root / "plan_to_label.csv").write_text('"' + "x" * 140_000, encoding="utf-8")
    drawn, other = check.offer_zones(inner)
    assert drawn["state"] == "drawn"
    words = drawn["draw"]
    assert (words["progress"], words["open_label"]) == (None, None)
    assert words["sheet_error"].startswith("plan_to_label.csv cannot be read as CSV")
    assert "save it again as CSV UTF-8" in words["sheet_error"]
    assert other["state"] == "to draw" and other["offer"] == OFFER["offer"]


# --------------------------------------------------------------------------- with the plan tool


@pytest.mark.asyncio
async def test_a_press_draws_through_the_plan_tool(root: Path) -> None:
    """The pinned request is the plan tool's own arguments: it draws a random
    sample at seed 0 of the count pressed, and the card then shows its sheet's
    progress; another count pressed later gets the same draw."""
    pytest.importorskip("oe_inferencex.estimate")
    scores = _logits(300)
    path = root / "scores_zone.json"
    path.write_text(json.dumps({"scores": scores}), encoding="utf-8")
    inner = {
        "result_id": "res-1",
        "prediction_id": None,
        "zones": [
            {
                "file": FILES[0],
                "scores_path": str(path),
                "pooled": {"n_valid": 300, "raster_check": {"sha256": "d" * 64}},
            }
        ],
    }
    (zone,) = check.offer_zones(inner)
    smallest = zone["offer"][0]["labels"]
    claim = check.claim_draw(zone["check_key"], smallest)
    assert claim["outcome"] == "first"
    out = await _call(PLAN, claim["draw"]["request"])
    draw = check.finish_draw(zone["check_key"], out)
    assert draw["state"] == "drawn", draw
    design = json.loads(Path(draw["design_path"]).read_text(encoding="utf-8"))
    assert (design["design"], design["seed"], design["budget"]) == (
        "random",
        0,
        smallest,
    )
    assert Path(draw["labels_csv_path"]).is_file()
    assert draw["must_state"] == out["result"].get("must_state", [])
    # The plan's own forbidden claims, as it wrote them, for the model.
    assert draw["forbidden_claims"] == out["result"]["forbidden_claims"]
    assert any(
        c.get("id") == "subset_labelling_sufficient" for c in draw["forbidden_claims"]
    )
    # And its facts: every row labelled for the design file's routes.
    assert draw["facts"] == out["result"]["facts"]
    assert any(f.get("id") == "prefix_is_random_sample" for f in draw["facts"])
    (again,) = check.offer_zones(inner)
    assert again["state"] == "drawn"
    assert again["draw"]["labels_csv_path"] == draw["labels_csv_path"]
    assert again["draw"]["progress"] == f"0 of {smallest} labelled"
    assert again["draw"]["open_label"] == sheet.OPEN_FIRST
    assert again["draw"]["must_state"] == draw["must_state"]
    assert again["draw"]["forbidden_claims"] == draw["forbidden_claims"]
    assert again["draw"]["facts"] == draw["facts"]
    repeat = check.claim_draw(zone["check_key"], zone["offer"][-1]["labels"])
    assert repeat["outcome"] == "repeat" and repeat["draw"]["budget"] == smallest
