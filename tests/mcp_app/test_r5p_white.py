"""Round r5-p, after a coastal map's sheets were pictured live (20 Nov 2026):
Sen2Cor's cloud mask (SCL) classed squares that are white with thick cloud at
a coastline as land and water, so the picked date was a white square. The
picture itself is now checked: a scene white on at least
``s2_fetch.BRIGHT_SHARE`` of the outlined square (red, green and blue all at
least ``s2_fetch.BRIGHT``) counts as cloudy whatever its mask says and the
next scene is tried; when every scene tried is white there, the least cloudy
by both signals is picked, "covered", and the pane folds it with a line that
says why. After review: the bar is 0.4 (sand and concrete stay under it), the
fallback weighs the mask too, a resumed mask reading follows a white scene
only when its mask had called it clear, and no further read for whiteness
starts on a short band budget."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmoearth_agent.mcp_app import imagery, s2_fetch

SQUARE = (2, 2, 4, 4)


def _stack(refl: float, white_cells: int = 16, nodata: int = 0) -> Any:
    """A 4-band 8 x 8 stack (DN = reflectance * 10000 + 1000): the square's
    first ``white_cells`` pixels at reflectance 0.9 in red, green and blue,
    the rest at ``refl``; the first ``nodata`` pixels of the square no data."""
    out = np.full((4, 8, 8), int(refl * 10000 + 1000), dtype=np.uint16)
    cells = [(y, x) for y in range(2, 6) for x in range(2, 6)]
    for y, x in cells[:white_cells]:
        out[:3, y, x] = int(0.9 * 10000 + 1000)
    for y, x in cells[:nodata]:
        out[:, y, x] = 0
    return out


def test_the_white_check_constants() -> None:
    """White: red, green and blue all at least 0.4; a quarter of the square;
    a read for whiteness only under 70% of the band budget."""
    assert s2_fetch.BRIGHT == 0.4
    assert s2_fetch.BRIGHT_SHARE == 0.25
    assert s2_fetch.WHITE_ROOM == 0.7


def test_white_share_counts_the_white_pixels_with_data() -> None:
    """The share of the square's pixels with data that are white; clear
    water, sand, concrete and a little surf stay under the bar; no pixel with
    data gives 0."""
    assert s2_fetch.white_share(_stack(0.02, white_cells=0), SQUARE) == 0.0
    assert s2_fetch.white_share(_stack(0.02, white_cells=16), SQUARE) == 1.0
    assert s2_fetch.white_share(_stack(0.02, white_cells=2), SQUARE) == 2 / 16
    # Bright sand, and pale concrete at 0.35 in every band: not white.
    for rgb in ((0.35, 0.30, 0.18), (0.35, 0.35, 0.35)):
        ground = _stack(0.02, white_cells=0)
        for band, value in enumerate(rgb):
            ground[band, 2:6, 2:6] = int(value * 10000 + 1000)
        assert s2_fetch.white_share(ground, SQUARE) == 0.0, rgb
    # No-data pixels are left out of the share.
    assert s2_fetch.white_share(_stack(0.02, white_cells=8, nodata=8), SQUARE) == 0.0
    assert s2_fetch.white_share(np.zeros((4, 8, 8), dtype=np.uint16), SQUARE) == 0.0


class _Scene:
    """A stand-in candidate: an id and a mask link."""

    def __init__(self, ident: str) -> None:
        self.id = ident
        self.mask_href = "x"


def _run(
    monkeypatch: pytest.MonkeyPatch,
    stacks: dict[str, Any],
    scores: dict[int, tuple[float, float]],
    more: dict[int, tuple[float, float]] | None = None,
    budget: s2_fetch.Budget | None = None,
) -> tuple[Any, int, dict[str, Any], list[str], int]:
    """_month_bands over stand-in scenes with the given band stacks and mask
    scores; ``more``: the scores a resumed mask reading finds. Returns the
    scene, the band reads, the mask info, the scenes read and how many times
    mask reading resumed."""
    found = [_Scene(k) for k in stacks]
    read: list[str] = []
    resumed: list[int] = []

    def fake_read(fetch: Any, scene: Any, box: Any) -> Any:
        read.append(scene.id)
        return stacks[scene.id]

    def fake_more(fetch: Any, masks: Any, state: Any, box: Any, square: Any) -> None:
        resumed.append(1)
        state[1].update(more or {})

    monkeypatch.setattr(s2_fetch, "read_box", fake_read)
    monkeypatch.setattr(s2_fetch, "_score_more", fake_more)
    fetch = SimpleNamespace(budget=budget or s2_fetch.Budget())
    state = (found, dict(scores), set(scores))
    scene, _stack_, tried, info = s2_fetch._month_bands(
        fetch, None, state, None, SQUARE  # type: ignore[arg-type]
    )
    return scene, tried, info, read, len(resumed)


def test_a_white_scene_the_mask_calls_clear_loses_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live case: the mask says clear, the picture is white; mask reading
    resumes and finds a clear scene, which wins."""
    stacks = {
        "white": _stack(0.02),
        "cloudy": _stack(0.02, 0),
        "clear": _stack(0.02, 0),
    }
    scene, tried, info, read, resumed = _run(
        monkeypatch, stacks, {0: (0.0, 1.0), 1: (1.0, 1.0)}, more={2: (0.0, 1.0)}
    )
    assert scene.id == "clear" and read == ["white", "clear"] and tried == 2
    assert resumed == 1
    assert info["here"] == "clear" and info["white_share"] == 0.0


def test_when_every_scene_tried_is_white_the_least_white_is_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three white scenes with no mask against them: the least white is
    picked, "covered", and the pane folds it with the white line."""
    stacks = {
        "a": _stack(0.02, 16),
        "b": _stack(0.02, 6),
        "c": _stack(0.02, 10),
    }
    scene, tried, info, _, _ = _run(monkeypatch, stacks, {0: (0.0, 1.0)})
    assert scene.id == "b" and tried == 3
    assert info["here"] == "covered" and info["white_share"] == 6 / 16
    assert imagery._marked(info)
    assert imagery.mask_line(info) == (
        "picture: white on 38% of the square (cloud, fog, or bright ground such as "
        "snow)"
    )


def test_the_fallback_weighs_the_mask_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear mask and white all over (clean snow) against a mask that marks
    the whole square and a picture a little less white: the first is kept
    (the larger of the two signals ties at 1.0; the first tried wins). A
    white scene the mask had already called covered does not resume mask
    reading."""
    stacks = {"snow": _stack(0.02, 16), "cloud": _stack(0.02, 14)}
    scene, tried, info, read, resumed = _run(
        monkeypatch, stacks, {0: (0.0, 1.0), 1: (1.0, 1.0)}
    )
    assert scene.id == "snow" and read == ["snow", "cloud"] and tried == 2
    assert resumed == 1  # after "snow" (mask clear), not after "cloud"
    assert info["here"] == "covered" and info["covered_share"] == 0.0


def test_no_read_for_whiteness_on_a_short_band_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 80% of the band budget's requests spent, a white scene is kept
    and no further band read starts."""
    budget = s2_fetch.Budget()
    budget.requests = int(budget.max_requests * 0.8)
    stacks = {"white": _stack(0.02), "clear": _stack(0.02, 0)}
    scene, tried, info, read, _ = _run(
        monkeypatch, stacks, {0: (0.0, 1.0), 1: (0.0, 1.0)}, budget=budget
    )
    assert scene.id == "white" and read == ["white"] and tried == 1
    assert info["here"] == "covered"


def test_a_little_white_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surf or a white roof on under a quarter of the square: the scene
    stays clear by its mask."""
    stacks = {"surf": _stack(0.02, 3)}
    scene, tried, info, _, _ = _run(monkeypatch, stacks, {0: (0.0, 1.0)})
    assert scene.id == "surf" and tried == 1
    assert info["here"] == "clear" and info["white_share"] == 3 / 16
    assert not imagery._marked(info)
    assert imagery.mask_line(info) == imagery.MASK_CLEAR


def test_the_white_line_keeps_the_masks_share() -> None:
    """White and marked by the mask too: the Details line says both."""
    info = {
        "here": "covered",
        "white_share": 0.5,
        "covered_share": 0.4,
        "valid_share": 1.0,
    }
    assert imagery.mask_line(info) == (
        "picture: white on 50% of the square (cloud, fog, or bright ground such as "
        "snow); cloud mask: cloud or shadow on 40% of the square"
    )
    totals = imagery.cloud_totals([{"cloud_here": "covered"}, {"cloud_here": "clear"}])
    assert totals == {"cloudy_dates": 1, "all_cloudy": False}


def test_an_older_record_without_the_white_share_reads_as_before() -> None:
    """A record month written before the white check has no white_share."""
    old = {"here": "clear", "covered_share": 0.0, "valid_share": 1.0}
    assert imagery.mask_line(old) == imagery.MASK_CLEAR and not imagery._marked(old)
    assert not imagery._white({"white_share": True})


def test_the_pick_rule_and_the_fold_say_the_white_check() -> None:
    """The record's rule and the pane's fold line name the white check."""
    assert "counts as cloudy whatever its mask says" in imagery.PICK_RULE
    assert "at least 25% of the outlined square" in imagery.PICK_RULE
    assert "reflectance all at least 0.4" in imagery.PICK_RULE
    assert imagery.PICK == "scl-cover@2"
    view = (Path(imagery.__file__).parent / "views" / "sheet.js").read_text(
        encoding="utf-8"
    )
    assert "or it is white in the picture. Both are sometimes wrong" in view
