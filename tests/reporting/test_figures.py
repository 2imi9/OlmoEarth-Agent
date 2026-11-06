# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The determination figures: well-formed SVG drawn from the run's own data.

Each figure is the window grid with what the determination used marked on
it, one chart where the answer rests on one, and a caption written from the
same numbers; windows are placed by row and column, never by coordinates.
"""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import pytest

from olmoearth_agent.reporting import figures as fg

_NS = "{http://www.w3.org/2000/svg}"


def _png(svg: str) -> bytes:
    """The grid's embedded PNG."""
    found = re.search(r'href="data:image/png;base64,([^"]+)"', svg)
    assert found, "no embedded PNG"
    return base64.b64decode(found.group(1))


def _pixels(data: bytes) -> tuple[int, int, list[bytes]]:
    """``(width, height, rows)`` of an RGB PNG with filter 0 on every row."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    idat, i = b"", 8
    while i < len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        if data[i + 4 : i + 8] == b"IDAT":
            idat += data[i + 8 : i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = [raw[r * stride : (r + 1) * stride] for r in range(height)]
    assert all(row[0] == 0 for row in rows)
    return width, height, [row[1:] for row in rows]


def _rgb(rows: list[bytes], x: int, y: int) -> tuple[int, int, int]:
    return rows[y][3 * x], rows[y][3 * x + 1], rows[y][3 * x + 2]


def _rects(svg: str, **attrs: str) -> list[ET.Element]:
    """The SVG's rectangles carrying every attribute given."""
    root = ET.fromstring(svg)  # noqa: S314 - an SVG this module wrote
    return [
        r
        for r in root.iter(f"{_NS}rect")
        if all(r.get(k.replace("_", "-")) == v for k, v in attrs.items())
    ]


def _texts(svg: str) -> str:
    root = ET.fromstring(svg)  # noqa: S314 - an SVG this module wrote
    return " ".join(t.text or "" for t in root.iter(f"{_NS}text"))


def test_the_window_grid_is_the_runs_or_a_layout_by_index() -> None:
    assert fg.window_grid(12, (3, 4)) == (3, 4, True)
    assert fg.window_grid(12, [3, 4]) == (3, 4, True)
    # A grid that does not hold the windows is not the run's: index order.
    assert fg.window_grid(10, (3, 4)) == (3, 4, False)
    assert fg.window_grid(10, None) == (3, 4, False)
    with pytest.raises(ValueError, match="at least one window"):
        fg.window_grid(0, None)


def test_confidence_is_shaded_by_rank_and_no_data_stays_white() -> None:
    cells = fg.confidence_base([0.9, 0.1, None, 0.5, float("nan")], 5)
    assert cells[1] == fg.UNSURE  # least confident: darkest
    assert cells[0] == fg.SURE  # most confident: lightest
    assert cells[2] is None and cells[4] is None
    assert cells[3] == fg.mix(fg.UNSURE, fg.SURE, 0.5)
    faded = fg.confidence_base([0.9, 0.1], 2, faded=1.0)
    assert faded == [(255, 255, 255), (255, 255, 255)]


def test_the_png_holds_each_window_at_its_row_and_column() -> None:
    cells = [(10, 20, 30), None, (200, 0, 0), (0, 200, 0)]
    width, height, rows = _pixels(fg.png(cells, 2, 2, 3))
    assert (width, height) == (6, 6)
    assert _rgb(rows, 0, 0) == (10, 20, 30)
    assert _rgb(rows, 5, 0) == fg.NO_DATA  # row 0, col 1: no-data
    assert _rgb(rows, 1, 4) == (200, 0, 0)  # row 1, col 0
    assert _rgb(rows, 4, 5) == (0, 200, 0)  # row 1, col 1


def test_the_review_set_is_outlined_on_the_confidence_grid() -> None:
    margin: list[float | None] = [i / 16 for i in range(16)]
    margin[15] = None
    fig = fg.review_set_figure(margin, (4, 4), [0, 1], budget=0.125, n_valid=15)
    outlines = _rects(fig.svg, stroke=fg.REVIEW, fill="none")
    assert len(outlines) == 2 + 1  # the two windows, plus the legend's swatch
    texts = _texts(fig.svg)
    assert "Which windows to check first" in texts and "(a)" in texts
    assert "The 2 least confident of 15 windows (12.5% review budget)" in texts
    assert fig.caption.startswith("Which windows to check first. (a) The map's 15")
    assert "on its 4 x 4 window grid" in fig.caption
    assert "no coordinates are drawn" in fig.caption
    # The grid's PNG: window 0 (least confident) dark, the no-data window white.
    width, _, rows = _pixels(_png(fig.svg))
    scale = width // 4
    assert _rgb(rows, 0, 0) == fg.UNSURE
    assert _rgb(rows, 3 * scale, 3 * scale) == fg.NO_DATA


def test_the_estimate_marks_each_label_and_draws_the_interval() -> None:
    margin = [0.1 * i for i in range(9)]
    fig = fg.estimate_figure(
        margin,
        (3, 3),
        [0, 4, 8, 2],
        [0, 1, None, 0],
        intervals=[
            ("valid at every sample size", 0.148, 0.239, 0.19),
            ("design's 95% interval", 0.15, 0.235, 0.19),
        ],
        design="random",
        n_valid=9,
    )
    assert len(_rects(fig.svg, fill=fg.RIGHT)) == 2 + 1  # + legend swatch
    assert len(_rects(fig.svg, fill=fg.WRONG)) == 1 + 1
    assert len(_rects(fig.svg, fill=fg.UNJUDGED)) == 1 + 1
    texts = _texts(fig.svg)
    assert "(b) Error rate over all windows" in texts
    assert "19.0% (14.8% to 23.9%)" in texts
    assert "1 of 3 labelled windows are wrong" in texts
    # The estimate once, then each interval once: the headline is not repeated.
    assert (
        "(b) The map's error rate over all its windows, 19.0%; valid at every "
        "sample size: 14.8% to 23.9%; design's 95% interval: 15.0% to 23.5%."
        in fig.caption
    )
    assert fig.caption.count("14.8% to 23.9%") == 1
    assert "grey not judged (1)" in fig.caption
    assert "The intervals assume the labels are right." in fig.caption
    # One interval: the estimate with its interval and what it is.
    one = fg.estimate_figure(
        margin,
        (3, 3),
        [0, 4],
        [0, 1],
        intervals=[("valid at every sample size", 0.148, 0.239, 0.19)],
        design="random",
        n_valid=9,
    )
    assert (
        "(b) The map's error rate over all its windows, 19.0% (14.8% to 23.9%, "
        "valid at every sample size)." in one.caption
    )


def _ticks(svg: str) -> list[str]:
    """The interval chart's axis labels (the only 10 px text of its figure)."""
    root = ET.fromstring(svg)  # noqa: S314 - an SVG this module wrote
    return [t.text or "" for t in root.iter(f"{_NS}text") if t.get("font-size") == "10"]


def test_a_rate_axis_never_passes_100_percent() -> None:
    """An interval near 100% ends its axis at 100%, not at a round 200%; a
    level's bound of 100% fills its bar."""
    fig = fg.estimate_figure(
        [0.1 * i for i in range(9)],
        (3, 3),
        [0, 4],
        [1, 1],
        intervals=[("valid at every sample size", 0.6, 0.97, 0.9)],
        design="random",
        n_valid=9,
    )
    assert _ticks(fig.svg) == ["0%", "50%", "100%"]
    zone = fg.zone_figure(
        [i / 16 for i in range(16)],
        (4, 4),
        [],
        [1],
        [1],
        alpha=0.1,
        delta=0.1,
        rule="prefix",
        levels=[("top 100%", 1.0, False)],
        n_valid=16,
    )
    (bar,) = _rects(zone.svg, fill=fg.NEITHER)
    assert float(bar.get("width", "0")) == fg.SIDE - 150  # the bar's full width


def test_many_marks_are_painted_into_the_grid() -> None:
    n = 60 * 60
    margin = [i / n for i in range(n)]
    review = list(range(fg.MAX_MARKS + 1))
    fig = fg.review_set_figure(margin, (60, 60), review, budget=0.6, n_valid=n)
    assert len(_rects(fig.svg, stroke=fg.REVIEW, fill="none")) == 1  # legend only
    _, _, rows = _pixels(_png(fig.svg))
    marked = fg.mix(fg.UNSURE, fg.hex_rgb(fg.REVIEW), 0.8)
    assert _rgb(rows, 0, 0) == marked
    assert _rgb(rows, 0, 0) != fg.UNSURE


def test_the_zone_is_tinted_and_the_levels_drawn_against_alpha() -> None:
    margin = [i / 16 for i in range(16)]
    zone = list(range(8, 16))
    fig = fg.zone_figure(
        margin,
        (4, 4),
        zone,
        [1, 9, 12],
        [1, 0, 0],
        alpha=0.1,
        delta=0.05,
        rule="prefix",
        levels=[("top 50%", 0.08, True), ("top 75%", 0.14, False)],
        n_valid=16,
    )
    holds = "with a statement that fails on at most 5% of random draws (prefix rule)"
    certified = "8 of 16 windows (50.0%) certified at an error rate of at most 10%"
    assert f"{certified}, {holds}." in _texts(fig.svg)
    assert f"where the error rate is at most 10% {holds}." in fig.caption
    assert "stated confidence" not in fig.caption
    assert len(_rects(fig.svg, fill=fg.css(fg.ZONE))) == 1 + 1  # bar + legend
    assert len(_rects(fig.svg, fill=fg.NEITHER)) == 1  # the level above alpha
    assert "(b) The upper bound on the error rate at each confidence level" in (
        fig.caption
    )
    assert "blue marks the levels the prefix rule certified" in fig.caption
    none = fg.zone_figure(
        margin,
        (4, 4),
        [],
        [1],
        [1],
        alpha=0.1,
        delta=0.1,
        rule="bonferroni",
        levels=[],
        n_valid=16,
    )
    assert "No zone certified" in _texts(none.svg)
    assert "(bonferroni rule, delta 10%)" in _texts(none.svg)
    assert "The bonferroni rule certified no level with these labels" in none.caption
    assert "(b)" not in none.caption


def test_the_two_map_figures_show_where_they_differ_and_each_labels_side() -> None:
    differ: list[bool | None] = [True, False, True, None, True, False]
    plan = fg.two_map_plan_figure(differ, (2, 3), [0, 4])
    assert "They differ on 3 of the 5 windows compared" in _texts(plan.svg)
    assert "amber where the two maps' classes differ (3)" in plan.caption
    verdict = "These 3 labels name map a by the exact sign test <p = 0.25>."
    grade = fg.two_map_grade_figure(
        differ, (2, 3), [0, 2, 4], ["a", "a", "?"], verdict=verdict
    )
    assert len(_rects(grade.svg, fill=fg.MAP_A)) == 2 + 1 + 1  # cells, legend, bar
    assert verdict in _texts(grade.svg)  # escaped in the SVG, parsed back
    assert "&lt;p = 0.25&gt;" in grade.svg
    # Labelled counts the ?, as the verdict's n_labelled does; judged does not.
    assert "(a) The 2 judged windows of the 3 labelled where the maps differ" in (
        grade.caption
    )
    assert "blue where the label is map a's class (2)" in grade.caption
    assert "outlined not judged (1)" in grade.caption
    # A window either map leaves without a class is white, and the legend says so.
    for fig in (plan, grade):
        assert "not compared" in _texts(fig.svg)
        assert len(_rects(fig.svg, fill="#ffffff", stroke=fg.MUTED)) == 1  # swatch
    full = fg.two_map_plan_figure([True, False, True, False], (2, 2), [0])
    assert "not compared" not in _texts(full.svg)
    # A design file that does not record where the maps differ: only the drawn
    # windows are known to differ, so no window is drawn as agreeing.
    old = fg.two_map_grade_figure(
        [True, None, True, None, True, None],
        (2, 3),
        [0, 2, 4],
        ["a", "b", "neither"],
        verdict=verdict,
        differ_known=False,
    )
    note = (
        "This design file does not record where the maps differ elsewhere, so "
        "only the labelled windows are shown."
    )
    assert note in old.caption and note not in grade.caption
    assert "(a) The 3 labelled windows where the maps differ" in old.caption
    assert "the maps agree" not in _texts(old.svg)
    assert "the maps agree" in _texts(grade.svg)


def test_without_a_grid_the_caption_says_windows_are_in_index_order() -> None:
    fig = fg.plan_figure(
        [0.2, 0.4, 0.6, 0.8, 0.9], None, [1, 3], design="random", seed=0, n_valid=5
    )
    assert "laid out in index order" in fig.caption
    assert "window grid" not in fig.caption.split(".")[1]


def test_a_figure_is_written_beside_the_runs_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    fig = fg.plan_figure([0.5, 0.6], None, [0], design="random", seed=3, n_valid=2)
    fields = fg.figure_fields("design_scores_abc_random_1_s3", fig)
    path = Path(fields["figure_path"])
    assert path == tmp_path.resolve() / "design_scores_abc_random_1_s3_figure.svg"
    assert path.read_text(encoding="utf-8") == fig.svg
    assert fields["figure_caption"] == fig.caption
    ET.fromstring(path.read_text(encoding="utf-8"))  # noqa: S314 - well-formed


def test_tied_margins_share_a_shade_and_a_flat_map_is_drawn_flat() -> None:
    cells = fg.confidence_base([0.2, 0.5, 0.2, 0.9], 4)
    assert cells[0] == cells[2]  # tied: one shade, no order the scores lack
    assert cells[0] != cells[1] != cells[3]
    flat = fg.confidence_base([0.0, 0.0, None, 0.0], 4)
    assert flat[0] == flat[1] == flat[3] and flat[2] is None
    assert fg.flat_confidence([0.0, 0.0, None]) and not fg.flat_confidence([0, 1])
    fig = fg.plan_figure([0.0] * 9, (3, 3), [1, 5], design="random", seed=0, n_valid=9)
    assert "outlined on a flat grid, as the map gives one confidence" in fig.caption
    assert "one confidence everywhere (flat)" in _texts(fig.svg)
    assert "class map" not in fig.caption


def test_a_class_map_is_drawn_flat_as_having_no_confidence() -> None:
    """A class map without top-k scores has no confidence (its margin is 0
    wherever a window holds a class): each route's legend and caption say so,
    not merely that the grid is flat."""
    margin: list[float | None] = [0.0] * 8 + [None]
    flat = "on a grid drawn flat: a class map has no confidence to rank by"
    plan = fg.plan_figure(
        margin, (3, 3), [1, 5], design="random", seed=0, n_valid=8, no_confidence=True
    )
    assert f"outlined {flat}." in plan.caption
    estimate = fg.estimate_figure(
        margin,
        (3, 3),
        [1, 5],
        [0, 1],
        intervals=[("valid at every sample size", 0.1, 0.9, 0.5)],
        design="random",
        n_valid=8,
        no_confidence=True,
    )
    assert f"shown {flat}; green where" in estimate.caption
    zone = fg.zone_figure(
        margin,
        (3, 3),
        [],
        [1],
        [0],
        alpha=0.1,
        delta=0.1,
        rule="prefix",
        levels=[],
        n_valid=8,
        no_confidence=True,
    )
    assert f"shown {flat}; the random sample's labels" in zone.caption
    for fig in (plan, estimate, zone):
        texts = _texts(fig.svg)
        assert "no confidence (class map)" in texts
        assert "one confidence everywhere (flat)" not in texts


def test_a_zone_level_is_coloured_by_the_rules_decision_not_by_alpha() -> None:
    fig = fg.zone_figure(
        [i / 16 for i in range(16)],
        (4, 4),
        [],
        [1],
        [0],
        alpha=0.025,
        delta=0.1,
        rule="prefix",
        levels=[("top 40%", 0.02, False)],  # below alpha, refused by the rule
        n_valid=16,
    )
    assert len(_rects(fig.svg, fill=fg.NEITHER)) == 1
    assert len(_rects(fig.svg, fill=fg.css(fg.ZONE))) == 0
    assert "at most 2.5%" in _texts(fig.svg)  # alpha is not rounded to 2%
    assert "alpha 2.5%" in _texts(fig.svg)


def test_percentages_read_as_the_facts_give_them() -> None:
    assert fg.pct(0.19) == "19.0%"
    assert fg.pct(0.0012) == "0.12%"
    assert fg.pct(0.0) == "0.0%" and fg.pct(None) == "n/a"
    assert fg.level(0.025) == "2.5%" and fg.level(0.1) == "10%"


def test_a_grid_larger_than_the_box_is_drawn_in_blocks() -> None:
    side = fg.MAP_BOX + 30
    n = side * side
    fig = fg.review_set_figure(
        [i / n for i in range(n)], (side, side), [0, 1], budget=2 / n, n_valid=n
    )
    width, height, _ = _pixels(_png(fig.svg))
    assert (width, height) == (fg.math.ceil(side / 2), fg.math.ceil(side / 2))
    assert "a block of up to 2 x 2 windows" in fig.caption
    # Windows 0 and 1 share the first block: one outline, plus the legend's.
    assert len(_rects(fig.svg, stroke=fg.REVIEW, fill="none")) == 1 + 1
    assert int(ET.fromstring(fig.svg).get("width")) >= fg.MIN_WIDTH  # noqa: S314
