# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The pictures' part of round 4, pass 2 (13 Nov 2026).

The reader test's words (the question asked once more, the rule's scope, the
test scene's opening), a review set's own test-scene caption and fetch
failure (no sheet, label or kelp map), a square's size for the Check card
and in front of a sheet's pictures caption, the sheet test scene's word, the
design structure the pictures and the page copy share, and the rule the user
typed before the draw in the pictures and their record. No test touches the
network: the search and the reads are replaced.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")
import httpx  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

from olmoearth_agent.mcp_app import imagery, s2_fetch, sheet  # noqa: E402

ONE_MAP = [
    "order",
    "window_index",
    "row",
    "col",
    "map_class",
    "lon",
    "lat",
    "stratum",
    "wrong",
    "reference_class",
]
P = 11
GRID = (6, 10)
MAP_EPSG = 32611
MAP_T = Affine(10, 0, 500000, 0, -10, 3700000)

#: The question as round 4's first pass asked it: the answer given to it no
#: longer counts, so everyone is asked once more.
OLD_CONSENT_ASK = (
    "Show Sentinel-2 pictures of each window? This computer will send the map's "
    f"extent, rounded, and its period to Earth Search's public catalogue "
    f"({s2_fetch.CATALOGUE_HOST}) once per sheet or review set, and requests for "
    f"parts of the image files to its public bucket ({s2_fetch.BUCKET_HOST}). No "
    "key, no account; both see this computer's IP address. Nothing is sent to "
    "Studio or to Claude."
)

TYPED = "Water: open water fills more than half of the outlined square."
TYPED_AT = "2026-11-13T10:15:00+00:00"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a test reached for the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    s2_fetch.clear_cache()
    imagery._maps.clear()
    imagery._test_cache.clear()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


@pytest.fixture
def no_scenes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search finds no scene in any month and the reads return none, so a
    picture call runs to its end (and its record) without a fetch."""
    monkeypatch.setattr(
        imagery,
        "_scenes",
        lambda _real, _extent, months, _client, _budget: ([[] for _ in months], []),
    )
    monkeypatch.setattr(
        s2_fetch,
        "dated",
        lambda _fetch, scenes, _box, _square: [(None, None, 0) for _ in scenes],
    )


def _raster(path: Path, *, epsg: int = MAP_EPSG, transform: Affine = MAP_T) -> Path:
    h, w = GRID[0] * P, GRID[1] * P
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=2,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
    return path


def _scores(root: Path, raster: Path | None, *, patch: Any = P) -> Path:
    payload: dict[str, Any] = {
        "scores": [[0.2, 0.8]] * (GRID[0] * GRID[1]),
        "grid": list(GRID),
        "patch_px": patch,
        "date_window": {
            "start": "2024-04-01T00:00:00Z",
            "end": "2024-06-30T23:59:59.999000Z",
        },
    }
    if raster is not None:
        payload["raster"] = {
            "file": raster.name,
            "path": str(raster),
            "shape": [2, GRID[0] * P, GRID[1] * P],
            "sha256": "abc",
        }
    scores = root / "scores_run_p11.json"
    scores.write_text(json.dumps(payload), encoding="utf-8")
    return scores


def _centre(index: int) -> tuple[float, float]:
    r, c = divmod(index, GRID[1])
    x, y = MAP_T * (c * P + P / 2, r * P + P / 2)
    (lon,), (lat,) = warp_transform(f"EPSG:{MAP_EPSG}", "EPSG:4326", [x], [y])
    return lon, lat


def _sheet(root: Path, names: dict[str, str]) -> Path:
    """A plan's sheet of two windows on a scores map with these class names,
    its design, its scores file and its map raster."""
    folder = root / "run"
    folder.mkdir(exist_ok=True)
    scores = _scores(root, _raster(folder / "scores.tif"))
    population = {
        "score_kind": "window_confidence",
        "class_names": names,
        "source": {"scores_path": str(scores)},
    }
    (root / "map.json").write_text(
        json.dumps({"design": "random", "population": population}), encoding="utf-8"
    )
    path = root / "map_to_label.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ONE_MAP)
        for order, index in enumerate((14, 25), start=1):
            lon, lat = _centre(index)
            r, c = divmod(index, GRID[1])
            writer.writerow(
                [order, index, r, c, "0", f"{lon:.7f}", f"{lat:.7f}", "0", "", ""]
            )
    return path


def _pictures(path: Path, index: int) -> dict[str, Any]:
    sheet_id, handles = sheet.window_handles(str(path))
    handle = next(h for h, i in handles.items() if i == index)
    return imagery.window_image(str(path), handle, sheet_id=sheet_id, on=True)


def _context(path: Path) -> dict[str, Any]:
    first = imagery.record_path(path).read_text(encoding="utf-8").splitlines()[0]
    context: dict[str, Any] = json.loads(first)["context"]
    return context


# ------------------------------------------------------------------ the words


def test_the_question_in_the_reader_tests_words_is_asked_once_more() -> None:
    """A/sheet-7 with D2: both hosts and what each gets, once per sheet or
    review set, no key or account, the IP address, nothing to Studio or
    Claude; its words changed, so its hash did and the old answer lapses.
    Since round r5-k K7 the face says the IP address and nothing to Studio or
    Claude, and the fold the hosts, what each gets (once for all the squares)
    and no key or account; the hash covers both (test_mcp_r5k_consent.py)."""
    ask = imagery.CONSENT_ASK
    full = imagery.CONSENT_FULL
    assert ask.startswith("Show satellite (Sentinel-2) pictures of each square?")
    for words in (
        f"Earth Search's catalogue is {imagery.CATALOGUE_HOST}: it gets the box and "
        "the months, once for all the squares.",
        f"Its file store is {imagery.BUCKET_HOST}: it gets the requests for parts "
        "of image files.",
        "No key and no account are used.",
    ):
        assert words in full, words
    for words in (
        "both see your IP address.",
        "Nothing is sent to Studio or to Claude.",
    ):
        assert words in ask, words
    old = hashlib.sha256(OLD_CONSENT_ASK.encode("utf-8")).hexdigest()[:16]
    assert imagery.consent_version() != f"{old}-fetch"
    asked = ask + "\n\n" + full
    assert imagery.consent_version().startswith(
        hashlib.sha256(asked.encode("utf-8")).hexdigest()[:16]
    )


def test_the_rule_scope_and_the_test_scenes_opening_are_the_reader_tests() -> None:
    # The last round's C5 (reader test 2's A/F12): shorter, the same facts;
    # kept for the kelp rule's records since round r5-l L1, whose general
    # scope names no map and no date.
    assert imagery.KELP_RULE_SCOPE == (
        "This rule came with the app (set 11 Nov 2026 for kelp canopy maps)."
    )
    assert imagery.RULE_SCOPE == (
        "This rule came with the app, for any map of present and absent."
    )
    assert "trial" not in imagery.RULE_SCOPE
    opening = (
        "Not sure yet? See how the pictures look on a public test place first. "
        "Nothing about your map is sent."
    )
    for ask in (imagery.TEST_ASK, imagery.REVIEW_TEST_ASK):
        assert ask.startswith(opening)
        # Since round r5-k K7 drawn after the hosts in the question's fold.
        assert "the same catalogue" in ask and "the same file store" in ask
    assert "unrelated to this sheet" in imagery.TEST_ASK
    assert "unrelated to this review set" in imagery.REVIEW_TEST_ASK


def test_a_review_sets_test_caption_and_fetch_failure_name_no_sheet() -> None:
    """Round 4's low findings 1, 6 and 9: no sheet, label or kelp map."""
    caption = imagery.REVIEW_TEST_CAPTION
    assert caption.endswith(imagery.CAPTION)
    assert caption.startswith("A public test scene, not a square of this review set")
    for word in ("sheet", "kelp", "label"):
        assert word not in caption.lower(), word
    failed = imagery.REVIEW_FETCH_FAILED
    assert "label" not in failed.lower() and failed.endswith("Try again.")
    assert imagery.FETCH_FAILED.startswith(failed.split(", so")[0])
    for kind in ("fetch", "budget"):
        assert imagery._REVIEW_FETCH_SENTENCE[kind] == failed
    for kind in ("catalogue", "deadline"):
        assert imagery._REVIEW_FETCH_SENTENCE[kind] == imagery._FETCH_SENTENCE[kind]


def test_the_test_scene_beside_a_review_set_says_so(
    root: Path, no_scenes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = imagery.public_test_scene(review=True)
    assert review["ok"] is True and review["test_scene"] is True, review
    assert review["caption"] == imagery.REVIEW_TEST_CAPTION
    imagery._test_cache.clear()
    plain = imagery.public_test_scene()
    assert plain["caption"] == imagery.TEST_CAPTION
    # A fetch that fails: the review set's sentence, the sheet's otherwise.

    def fail(*_a: Any, **_k: Any) -> Any:
        raise s2_fetch.FetchError("fetch")

    imagery._test_cache.clear()
    monkeypatch.setattr(imagery, "_scenes", fail)
    # Another try can get past either (again, r5-a L1).
    assert imagery.public_test_scene(review=True) == {
        "ok": False,
        "error": imagery.REVIEW_FETCH_FAILED,
        "again": True,
    }
    assert imagery.public_test_scene() == {
        "ok": False,
        "error": imagery.FETCH_FAILED,
        "again": True,
    }
    assert list(root.iterdir()) == []  # no record, no consent file


# ------------------------------------------------------------ a square's size


def test_a_squares_size_is_its_pixels_and_on_a_utm_grid_its_metres(
    root: Path,
) -> None:
    folder = root / "run"
    folder.mkdir()
    utm = _scores(root, _raster(folder / "utm.tif"))
    assert imagery.window_size(str(utm)) == {"pixels": P, "metres": 110.0}
    # A grid in degrees: no metres.
    degrees = _raster(
        folder / "lonlat.tif",
        epsg=4326,
        transform=Affine(0.0001, 0, -119.0, 0, -0.0001, 34.0),
    )
    assert imagery.window_size(str(_scores(root, degrees))) == {
        "pixels": P,
        "metres": None,
    }
    # No raster recorded, or one that is gone: the pixels alone.
    assert imagery.window_size(str(_scores(root, None))) == {
        "pixels": P,
        "metres": None,
    }
    assert imagery.window_size(str(_scores(root, folder / "gone.tif"))) == {
        "pixels": P,
        "metres": None,
    }
    # No whole window size, a file that cannot be read, or one outside the roots.
    for patch in (None, 0, "11", 2.5, True):
        assert imagery.window_size(str(_scores(root, None, patch=patch))) is None
    assert imagery.window_size(str(_scores(root, None, patch=11.0))) == {
        "pixels": 11,
        "metres": None,
    }
    assert imagery.window_size(str(root / "missing.json")) is None
    assert imagery.window_size(None) is None


def test_a_scores_file_outside_the_roots_gives_no_size(
    root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    scores = _scores(outside, None)
    assert imagery.window_size(str(scores)) is None


def test_the_square_size_sentence_is_said_as_the_check_card_says_it() -> None:
    """Round 4 pass 2, finding 8: the pixels, and the metres on a UTM grid,
    rounded to whole metres half up with thousands commas; no metres clause
    without metres above 0; nothing without a whole size of 1 or more."""
    said = imagery.square_size_text
    assert said({"pixels": 11, "metres": 110.0}) == (
        "A square is 11 x 11 map pixels, about 110 m a side. "
    )
    assert said({"pixels": 128, "metres": 1280.0}) == (
        "A square is 128 x 128 map pixels, about 1,280 m a side. "
    )
    assert said({"pixels": 11, "metres": 109.5}).endswith("about 110 m a side. ")
    assert said({"pixels": 11, "metres": 110.4}).endswith("about 110 m a side. ")
    plain = "A square is 11 x 11 map pixels. "
    for metres in (None, 0, -10.0, float("nan"), float("inf"), True, "110"):
        assert said({"pixels": 11, "metres": metres}) == plain, metres
    for size in (None, {}, {"pixels": 0}, {"pixels": True}, {"pixels": 2.5}, "11"):
        assert said(size) == "", size
    # The caption it leads stays word for word (works well, do not change).
    assert imagery.CAPTION == (
        "One date for each month of the map's period: the scene with the least "
        "scene cloud that has data here (dates in UTC). Near-infrared false "
        "colour shows vegetation red and water dark. Magenta: no data."
    )


def test_a_sheets_pictures_caption_says_how_big_a_square_is(
    root: Path, no_scenes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentence in front of the caption, from the scores file that placed
    the window: 11 pixels on a 10 m UTM grid; the pixels alone off UTM; the
    caption alone when the file gives no size."""
    path = _sheet(root, {"0": "absent", "1": "present"})
    out = _pictures(path, 14)
    assert out["ok"] is True, out
    assert out["caption"] == (
        "A square is 11 x 11 map pixels, about 110 m a side. " + imagery.CAPTION
    )
    asked: list[Any] = []

    def no_metres(scores_path: Any) -> dict[str, Any]:
        asked.append(scores_path)
        return {"pixels": 11, "metres": None}

    monkeypatch.setattr(imagery, "window_size", no_metres)
    off_utm = _pictures(path, 25)
    assert off_utm["caption"] == "A square is 11 x 11 map pixels. " + imagery.CAPTION
    assert asked == [str(root / "scores_run_p11.json")]
    monkeypatch.setattr(imagery, "window_size", lambda _path: None)
    assert _pictures(path, 14)["caption"] == imagery.CAPTION


def test_the_sheets_test_scene_says_square() -> None:
    """The lead's ruling W: the sheet face's word, in the test scene's caption
    (round r5-l L1: no "as on the kelp map")."""
    words = (
        "A public test scene, not a square of this sheet: nothing here is "
        "labelled. The outlined square is 11 pixels. "
    )
    assert imagery.TEST_CAPTION == words + imagery.CAPTION
    assert "window" not in imagery.TEST_CAPTION


def test_the_pictures_read_the_map_from_the_page_copys_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract C1: imagery._map_of loads the scores files sheet.picture_sources
    reads off the design (the first places the window), and refuses without
    opening anything where pictures_possible is false."""
    loaded: list[Any] = []

    def fake_load(path: Any) -> Any:
        loaded.append(path)
        return type("M", (), {"start": 1, "end": 2})()

    monkeypatch.setattr(imagery, "_load_map", fake_load)
    one = {"population": {"source": {"scores_path": "a.json", "result_id": "r"}}}
    two = {
        "population": {
            "source": {
                "map_a": {"scores_path": "a.json"},
                "map_b": {"scores_path": "b.json"},
            }
        }
    }
    for design, paths in ((one, ["a.json"]), (two, ["a.json", "b.json"])):
        loaded.clear()
        assert sheet.pictures_possible(design) is True
        assert sheet.picture_sources(design) == paths
        imagery._map_of(design)
        assert loaded == paths
    by_result = {"population": {"source": {"result_id": "r", "property_name": "p"}}}
    two_results = {
        "population": {
            "source": {"map_a": {"result_id": "r1"}, "map_b": {"result_id": "r2"}}
        }
    }
    half = {
        "population": {
            "source": {"map_a": {"scores_path": "a.json"}, "map_b": {"result_id": "r"}}
        }
    }
    inline = {"population": {"source": {"input": "inline scores"}}}
    odd = {"population": {"source": ["a.json"]}}
    for design in (by_result, two_results, half, inline, odd, {}, None):
        loaded.clear()
        assert sheet.pictures_possible(design) is False, design
        with pytest.raises(imagery._Refused) as caught:
            imagery._map_of(design)
        assert str(caught.value) == imagery.NO_FOOTPRINT
        assert loaded == []


# ---------------------------------------------------- the rule in the pictures


def test_a_typed_rule_reaches_the_pictures_and_their_record(
    root: Path, no_scenes: None
) -> None:
    """A map the owner's rule does not cover: the rule typed before the draw,
    its scope and its source, in the pane's answer and the record's context
    line."""
    path = _sheet(root, {"0": "water", "1": "trees"})
    sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    out = _pictures(path, 14)
    assert out["ok"] is True, out
    scope = "Typed by you before the squares were picked, 13 Nov 2026."
    assert (out["reading_rule"], out["rule_scope"], out["rule_source"]) == (
        TYPED,
        scope,
        "typed",
    )
    context = _context(path)
    kept = (context["reading_rule"], context["rule_scope"], context["rule_source"])
    assert kept == (TYPED, scope, "typed")
    assert imagery.rule_for(path, ["water", "trees"]) == (TYPED, scope, "typed")


def test_the_owners_rule_says_whose_it_is(root: Path, no_scenes: None) -> None:
    """A map of presence and absence with nothing typed in the box beside the
    app's rule (the box left empty): the app's rule (general since round r5-l
    L1), its scope and its source, in the pane's answer and the record's
    context line."""
    path = _sheet(root, {"0": "absent", "1": "present"})
    out = _pictures(path, 25)
    assert out["ok"] is True, out
    assert (out["reading_rule"], out["rule_scope"], out["rule_source"]) == (
        imagery.READING_RULE,
        imagery.RULE_SCOPE,
        "owner",
    )
    context = _context(path)
    assert context["reading_rule"] == imagery.READING_RULE
    assert context["rule_source"] == "owner"
    assert context["rule_scope"] == imagery.RULE_SCOPE


def test_a_rule_typed_in_the_box_replaces_the_owners_in_the_pictures(
    root: Path, no_scenes: None
) -> None:
    """The owner's option A (13 Nov 2026): on a map of presence and absence
    whose maker meant something else, the rule typed in the box beside the
    app's and
    pinned beside the sheet at the press replaces the owner's rule in the
    pane's answer and the record's context line, with the typed scope, never
    RULE_SCOPE."""
    path = _sheet(root, {"0": "absent", "1": "present"})
    sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    out = _pictures(path, 25)
    assert out["ok"] is True, out
    scope = "Typed by you before the squares were picked, 13 Nov 2026."
    typed = (TYPED, scope, "typed")
    assert (out["reading_rule"], out["rule_scope"], out["rule_source"]) == typed
    context = _context(path)
    kept = (context["reading_rule"], context["rule_scope"], context["rule_source"])
    assert kept == typed
    record = imagery.record_path(path).read_text(encoding="utf-8")
    assert imagery.READING_RULE not in record and imagery.RULE_SCOPE not in record
    assert imagery.rule_for(path, ["absent", "present"]) == typed


def test_the_pictures_refusals_on_a_face_say_square_not_window() -> None:
    """The owner's ruling 6 (13 Nov 2026): "windows" goes from the card faces.
    A sheet or review set that can show pictures is one of squares."""
    # The sheet's map button is "Street map" (reader test 2's A/F21).
    assert imagery.NO_FOOTPRINT == (
        "Pictures need a plan drawn from a scores file, whose map raster places "
        "the square; this sheet's plan has none. Street map still works."
    )
    assert imagery.MISMATCH == (
        "The square's place in the map raster does not match its row on the "
        "sheet, so no pictures are shown."
    )
    assert imagery.REVIEW_MISMATCH == (
        "The square's place in the map raster does not match its scores file, so "
        "no pictures are shown."
    )
    assert imagery.NOT_ON_SHEET == sheet.UNKNOWN_HANDLE
    for said in (
        imagery.NO_FOOTPRINT,
        imagery.MISMATCH,
        imagery.REVIEW_MISMATCH,
        imagery.NOT_ON_SHEET,
    ):
        assert "window" not in said.lower(), said


def test_no_rule_says_none(root: Path, no_scenes: None) -> None:
    path = _sheet(root, {"0": "water", "1": "trees"})
    out = _pictures(path, 14)
    assert out["ok"] is True, out
    assert (out["reading_rule"], out["rule_scope"], out["rule_source"]) == (
        None,
        None,
        None,
    )
    context = _context(path)
    assert context["reading_rule"] is None and context["rule_source"] is None
    assert imagery.rule_for(path, ["water", "trees"]) == (None, None, None)


# ------------------------- r4i F9, as round r5-l L1 leaves it: no rule_limit


@pytest.mark.parametrize(
    ("names", "typed", "source"),
    [
        ({"0": "absent", "1": "present"}, False, "owner"),
        ({"0": "absent", "1": "present"}, True, "typed"),
        ({"0": "water", "1": "trees"}, False, None),
    ],
)
def test_r4i_the_pictures_say_the_rule_and_its_scope_only(
    root: Path,
    no_scenes: None,
    names: dict[str, str],
    typed: bool,
    source: str | None,
) -> None:
    """r4i F9 gave the pane's answer rule_limit beside the owner's rule;
    round r5-l L1 took it away: beside the app's rule, a rule typed before
    the draw and no rule, the answer carries the rule, its scope and its
    source only, and the record's context line no rule_limit either."""
    path = _sheet(root, names)
    if typed:
        sheet.set_typed_rule(str(path), TYPED, typed_at=TYPED_AT)
    out = _pictures(path, 14)
    assert out["ok"] is True, out
    assert "rule_limit" not in out
    assert out["rule_source"] == source
    assert "rule_limit" not in _context(path)


def test_r4i_a_review_sets_pictures_and_the_test_scene_carry_no_rule_limit(
    root: Path, no_scenes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """r4i F9: nothing is labelled from a review set's pictures or from the
    public test scene, so their answers carry no rule (and, since round r5-l
    L1, no rule_limit key at all).
    The review's window is placed from its scores file's centre, as
    olmoearth_scores_from_file writes it; the review's own record is stood
    in for (review.review_row)."""
    folder = root / "run"
    folder.mkdir()
    scores = _scores(root, _raster(folder / "scores.tif"))
    payload = json.loads(scores.read_text(encoding="utf-8"))
    payload["centres_lon_lat"] = [list(_centre(i)) for i in range(GRID[0] * GRID[1])]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    sets = root / "review_sets"
    sets.mkdir()
    key = "0123456789abcdef01234567"
    placed = (sets / f"{key}.json", str(scores), 14)
    monkeypatch.setattr(imagery.review, "review_row", lambda _key, _rank: placed)
    out = imagery.review_image(key, 1, on=True)
    assert out["ok"] is True, out
    assert (out["reading_rule"], out["rule_scope"]) == (None, None)
    assert "rule_limit" not in out
    for beside_a_review in (True, False):
        imagery._test_cache.clear()
        scene = imagery.public_test_scene(review=beside_a_review)
        assert scene["ok"] is True, scene
        assert scene["reading_rule"] is None and "rule_limit" not in scene
