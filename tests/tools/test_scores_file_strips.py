# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A Studio raster over the size limit, pooled in strips (round r5-a, K1).

``olmoearth_scores_from_file`` reads a raster at or under
``scores_file.MAX_VALUES`` whole, as before. A raster this agent wrote from a
Studio result's files that holds more is pooled strip by strip
(``scores_file._pool_strips``), up to ``scores_file.MAX_STRIP_VALUES``. Each
test pools one small raster both ways, the strips forced by setting the
limits small, and compares: the two scores files byte for byte, the two tool
results, the window arrays and the package's warnings word for word. The
raster is the same file in both reads, so its sha256 in the scores file is
the same and the comparison is of every byte.

Each run directory is written the way ``olmoearth_scores_from_studio``
writes one: ``scores.tif`` (float32, NaN no-data, deflate) and a
``manifest.json`` whose ``source.kind`` is ``studio_files``. Nothing here
reaches Studio. Every test needs the inferencex extra (numpy, rasterio,
olmoearth-inferencex) and skips without it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import scores_file, studio_scores
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools

try:
    import numpy as np
    import rasterio
    from oe_inferencex.common import raster as raster_io
    from oe_inferencex.readers import olmoearth as oe_reader
    from oe_inferencex.review import assess
    from rasterio.transform import Affine

    HAVE_EXTRA = True
except ImportError:  # pragma: no cover - the core install has neither
    HAVE_EXTRA = False

needs_extra = pytest.mark.skipif(
    not HAVE_EXTRA, reason="needs the inferencex extra (numpy, rasterio)"
)

TOOL = "olmoearth_scores_from_file"
RID = "5f0c2a71-3b9e-4d8a-9c61-0a7e2b4f1d33"
X0, Y0 = 250_000.0, 3_800_000.0

#: The rasters pooled both ways (:func:`_case`).
CASES = (
    "divides",
    "ragged",
    "patch_20",
    "two_channels",
    "three_channels_zeros",
    "empty_strip",
    "half_valid",
    "tied_majority",
    "quantized",
    "one_window_row",
    "taller_than_raster",
)


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One tool call through the registry; no Studio client is needed."""
    registry = ToolRegistry()
    registry.register_all(build_scores_file_tools())
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    out = await _call(name, args)
    assert out["ok"] is True, out
    result: dict[str, Any] = out["result"]
    return result


@pytest.fixture(autouse=True)
def _scores_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "scores_root"
    root.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    return root


def _studio_run(
    root: Path,
    name: str,
    values: Any,
    *,
    classes: dict[str, str] | None = None,
    source: bool = True,
    profile_edit: dict[str, Any] | None = None,
) -> Path:
    """A run directory as ``olmoearth_scores_from_studio`` writes a zone's.

    ``values`` is ``(C, H, W)``, NaN where there is no prediction. Without
    ``source`` the manifest names no Studio files: a direct run's.
    ``profile_edit`` changes the raster's profile (its no-data value or
    type), for a raster a strip read does not take.
    """
    run = root / name
    run.mkdir(parents=True)
    c, h, w = values.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "crs": "EPSG:32616",
        "transform": Affine(10.0, 0.0, X0, 0.0, -10.0, Y0),
        "nodata": float("nan"),
        "compress": "deflate",
        **(profile_edit or {}),
    }
    path = run / "scores.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(profile["dtype"]))
    if classes is None and c == 1:
        classes = {"0": "not karst", "1": "karst"}
    elif classes is None:
        classes = {str(i): f"class {i}" for i in range(c)}
    manifest: dict[str, Any] = {
        "what": "a Studio result's native-resolution file, as probabilities",
        "model": {"id": "model-test", "revision": RID, "name": "Karst test"},
        "scores": {
            "file": "scores.tif",
            "values": "probabilities",
            "nodata": "NaN",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": [c, h, w],
        },
        "classes": classes,
        "studio": {
            "kind": "unit_score" if c == 1 else "top_k",
            "property_name": "karst",
            "assumption": "The score is read as a probability.",
        },
    }
    if source:
        manifest["source"] = {
            "kind": scores_file.STUDIO_FILES,
            "result_id": RID,
            "prediction_id": None,
            "file": "result_epsg32616_0_native.tif",
            "sha256": "0" * 64,
        }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _scores(seed: int, bands: int, h: int, w: int) -> Any:
    """``bands`` channels of scores in [0, 1), float32."""
    return np.random.default_rng(seed).random((bands, h, w)).astype(np.float32)


def _case(name: str) -> tuple[Any, int, int]:
    """A case's raster, its patch and the strip size, in values.

    Sides that divide by the patch and ragged right and bottom edges; patch
    4, 5 and 20; one band and more; a class written 0 on many pixels and a
    pixel with one band missing; two window rows with no prediction (a strip
    of them is all no-data); a window exactly half valid (valid) and one a
    pixel under (not); a window whose two classes tie in votes, which the
    package breaks by confidence; values quantized so a review set's cut
    falls inside a run of equal scores; strips of one window row, of
    several, and taller than the raster.
    """
    if name == "divides":
        return _scores(1, 1, 32, 24), 4, 24 * 8
    if name == "ragged":
        return _scores(2, 1, 33, 27), 4, 27 * 8
    if name == "patch_20":
        return _scores(3, 1, 64, 50), 20, 50 * 20
    if name == "two_channels":
        return _scores(4, 2, 32, 28), 4, 2 * 28 * 12
    if name == "three_channels_zeros":
        values = _scores(5, 3, 30, 26)
        values[2, :, ::2] = 0.0
        values[1, 3, 4] = np.nan
        return values, 5, 3 * 26 * 5
    if name == "empty_strip":
        values = _scores(6, 1, 32, 24)
        values[0, 8:16, :] = np.nan
        values[0, 30:, :] = np.nan
        return values, 4, 24 * 4
    if name == "half_valid":
        values = _scores(7, 1, 16, 16)
        values[0, 0:2, 0:4] = np.nan  # window (0, 0): 8 of 16 valid
        values[0, 4:6, 4:8] = np.nan  # window (1, 1): 7 of 16 valid
        values[0, 6, 4] = np.nan
        return values, 4, 16 * 4
    if name == "tied_majority":
        values = np.full((1, 16, 16), 0.9, dtype=np.float32)
        values[0, 0:2, 0:4] = 0.2  # window (0, 0): 8 pixels of each class
        values[0, 8:10, 4:8] = 0.45  # window (2, 1): the same, another margin
        return values, 4, 16 * 4
    if name == "quantized":
        rng = np.random.default_rng(8)
        values = (rng.integers(0, 5, (1, 40, 36)) / 4).astype(np.float32)
        return values, 4, 36 * 8
    if name == "one_window_row":
        return _scores(9, 1, 37, 29), 3, 1
    assert name == "taller_than_raster"
    return _scores(10, 1, 21, 19), 4, 10**9


async def _both(
    monkeypatch: pytest.MonkeyPatch, args: dict[str, Any], block_values: int
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    """The tool's result and its scores file for a whole read, then in strips.

    The second call runs with ``MAX_VALUES`` at 1, so the raster is over it,
    and ``BLOCK_VALUES`` at ``block_values``. It must pool in strips, once,
    and never through the whole read. Both calls write the scores file under
    one name (the raster and the patch are the same), so its bytes are read
    after each call.
    """
    whole = await _ok(TOOL, args)
    whole_file = Path(whole["scores_path"]).read_bytes()
    used: list[str] = []
    pool_strips, pool_whole = scores_file._pool_strips, scores_file._pool_whole

    def strips_spy(*given: Any) -> Any:
        """``_pool_strips``, noted."""
        used.append("strips")
        return pool_strips(*given)

    def whole_spy(*given: Any) -> Any:
        """``_pool_whole``, noted."""
        used.append("whole")
        return pool_whole(*given)

    with monkeypatch.context() as patched:
        patched.setattr(scores_file, "MAX_VALUES", 1)
        patched.setattr(scores_file, "BLOCK_VALUES", block_values)
        patched.setattr(scores_file, "_pool_strips", strips_spy)
        patched.setattr(scores_file, "_pool_whole", whole_spy)
        strips = await _ok(TOOL, args)
    assert used == ["strips"]
    strips_file = Path(strips["scores_path"]).read_bytes()
    return whole, whole_file, strips, strips_file


def _run_files(run: Path) -> Any:
    """The run's record and raster, as the handler reads them."""
    fix = str(oe_reader.FIX_PROBABILITIES)
    return scores_file.read_run_files(run, None, rasterio, np, fix)


# --------------------------------------------------------------------------- 1: the scores file


@needs_extra
@pytest.mark.asyncio
@pytest.mark.parametrize("name", CASES)
async def test_strips_give_the_whole_reads_scores_file_byte_for_byte(
    name: str, _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same raster pooled whole, then in strips: the two scores files are
    equal byte for byte and the two tool results are equal."""
    values, patch, block = _case(name)
    _studio_run(_scores_root, name, values)
    args = {"run_dir": name, "patch": patch}
    whole, whole_file, strips, strips_file = await _both(monkeypatch, args, block)
    assert strips_file == whole_file
    assert strips == whole
    assert whole["provenance"] == scores_file.STUDIO_PROVENANCE
    assert whole["n_valid"] > 0


@needs_extra
@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(20))
async def test_strips_give_the_whole_reads_scores_file_on_random_rasters(
    seed: int, _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Twenty seeded rasters: one to three bands, a patch of 1 to 5, a width
    of 9 to 40 pixels and a height up to 40 of at least 9 pixels and four
    rows of windows, a fifth of the pixels and one band of rows with no
    prediction, every other raster quantized to quarters, and a strip of one
    value (one window row), one window row, three, or the whole raster. The
    band of rows touches two rows of windows at most, so with four or more
    the map keeps valid windows and neither read refuses it."""
    rng = np.random.default_rng(100 + seed)
    bands = int(rng.integers(1, 4))
    patch = int(rng.choice([1, 2, 3, 4, 5]))
    h = int(rng.integers(max(9, 4 * patch), 41))
    w = int(rng.integers(9, 41))
    values = rng.random((bands, h, w))
    if seed % 2:
        values = np.round(values * 4) / 4
    values[:, rng.random((h, w)) < 0.2] = np.nan
    top = int(rng.integers(0, h - patch))
    values[:, top : top + patch, :] = np.nan
    row = bands * w * patch
    block = int(rng.choice([1, row, 3 * row, 10**9]))
    name = f"random{seed}"
    _studio_run(_scores_root, name, values.astype(np.float32))
    args = {"run_dir": name, "patch": patch}
    whole, whole_file, strips, strips_file = await _both(monkeypatch, args, block)
    assert strips_file == whole_file
    assert strips == whole
    assert whole["n_valid"] > 0


@needs_extra
@pytest.mark.asyncio
@pytest.mark.parametrize("include_fill", [False, True])
async def test_strips_keep_the_label_fill_block(
    include_fill: bool, _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A channel named as the label fill value is left out, or kept on
    request, and its count of valid arg-max pixels is summed over the strips
    (the rows under the last whole window among them)."""
    values = _scores(11, 3, 26, 20)
    values[2, 5, :] = 1.0  # the fill channel is the arg-max on a row
    values[2, 25, :3] = 1.0  # and under the last whole window of 4 px
    values[:, 9, 2:5] = np.nan
    classes = {"0": "water", "1": "land", "2": "nodata_value_2"}
    _studio_run(_scores_root, "fill", values, classes=classes)
    args = {"run_dir": "fill", "patch": 4, "include_fill_class": include_fill}
    whole, whole_file, strips, strips_file = await _both(monkeypatch, args, 3 * 20 * 8)
    assert strips_file == whole_file
    assert strips == whole
    fill = whole["fill_class"]
    assert fill["channel"] == 2 and fill["excluded"] is not include_fill
    assert fill["valid_pixels_where_it_is_the_argmax"] >= 23


@needs_extra
@pytest.mark.asyncio
async def test_a_score_of_exactly_half_is_not_the_named_class_either_way(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Check card's cut line (``server.CHECK_CUT_LINE``, r5-a K3) says a
    pixel counts as the named class when its score is "above 0.5". Through
    the tool, on a one-band float32 raster as the Studio files tool writes
    one, whole and in strips: a window whose every pixel is exactly 0.5 is
    the other class, one at the next float32 above 0.5 the named class, and
    one at 0.75 too. The package's rule alone is pinned in
    ``tests/mcp_app/test_mcp_r5a_check.py``."""
    values = np.empty((1, 4, 12), dtype=np.float32)
    values[0, :, 0:4] = 0.5
    values[0, :, 4:8] = np.nextafter(np.float32(0.5), np.float32(1.0))
    values[0, :, 8:12] = 0.75
    assert values[0, 0, 4] > 0.5 and values[0, 0, 0] == 0.5
    _studio_run(_scores_root, "cut", values)
    args = {"run_dir": "cut", "patch": 4}
    whole, whole_file, strips, strips_file = await _both(monkeypatch, args, 12 * 4)
    assert json.loads(whole_file)["map_class"] == [0, 1, 1]
    assert json.loads(strips_file)["map_class"] == [0, 1, 1]
    assert json.loads(whole_file)["classes"] == {"0": "not karst", "1": "karst"}


# --------------------------------------------------------------------------- 2: the window arrays


@needs_extra
@pytest.mark.parametrize("name", CASES)
def test_strips_give_the_whole_reads_window_arrays(
    name: str, _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The valid-window array whole; the confidence and the class at every
    valid window (a strip with no valid window keeps no number for its
    windows, which no result reads); the top-1 probability at every window,
    NaN equal to NaN; and what the result states of the raster. Shows the
    first number that differs when the scores files do."""
    values, patch, block = _case(name)
    run = _run_files(_studio_run(_scores_root, name, values))
    whole = scores_file._pool_whole(np, assess, raster_io, run, False, patch, False)
    monkeypatch.setattr(scores_file, "BLOCK_VALUES", block)
    strips = scores_file._pool_strips(np, rasterio, assess, run, False, patch, False)
    np.testing.assert_array_equal(strips.valid_w, whole.valid_w)
    valid = whole.valid_w
    np.testing.assert_array_equal(strips.conf[valid], whole.conf[valid])
    np.testing.assert_array_equal(strips.klass[valid], whole.klass[valid])
    np.testing.assert_array_equal(strips.p1_w, whole.p1_w)
    assert strips.shape == whole.shape
    assert strips.kept == whole.kept
    assert strips.labels == whole.labels
    assert strips.fill == whole.fill
    assert strips.n_nodata_pixels == whole.n_nodata_pixels
    assert strips.pixels_outside == whole.pixels_outside
    assert strips.warnings == whole.warnings
    assert strips.geo["transform"] == whole.geo["transform"]
    assert str(strips.geo["crs"]) == str(whole.geo["crs"])


# --------------------------------------------------------------------------- 3: the warnings


@needs_extra
def test_strips_rebuild_the_packages_warnings_word_for_word(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A map with a ragged edge whose windows all tie but one: the package's
    probability warning, its sentence on the pixels outside the window grid
    and its sentence on each review set cut inside the tie, in its order."""
    values = np.full((1, 42, 37), 0.9, dtype=np.float32)
    values[0, :4, :4] = 0.6  # one window less sure: alone in the 1% set
    run = _run_files(_studio_run(_scores_root, "ties", values))
    whole = scores_file._pool_whole(np, assess, raster_io, run, False, 4, False)
    monkeypatch.setattr(scores_file, "BLOCK_VALUES", 37 * 4)
    strips = scores_file._pool_strips(np, rasterio, assess, run, False, 4, False)
    assert strips.warnings == whole.warnings
    assert whole.warnings[0] == assess.PROBABILITY_WARNING
    edge = "on the right and bottom edge do not fill a 4 px window"
    assert sum(edge in warning for warning in whole.warnings) == 1
    cut = [w for w in whole.warnings if w.startswith("review set at ")]
    assert len(cut) == 2 and all("windows left outside" in w for w in cut)
    assert len(whole.warnings) == 4


# --------------------------------------------------------------------------- 4: the reads


class _SpyDataset:
    """An open raster that notes the window of each read."""

    def __init__(self, inner: Any, reads: list[Any]) -> None:
        self._inner = inner
        self._reads = reads

    def __enter__(self) -> _SpyDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.close()

    def read(self, *args: Any, **kwargs: Any) -> Any:
        """The read, its window noted (``None`` for a whole read)."""
        self._reads.append(kwargs.get("window"))
        return self._inner.read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SpyRasterio:
    """``rasterio`` whose ``open`` gives a :class:`_SpyDataset`."""

    def __init__(self, reads: list[Any]) -> None:
        self._reads = reads

    def open(self, *args: Any, **kwargs: Any) -> _SpyDataset:
        """The raster, opened by rasterio and wrapped."""
        return _SpyDataset(rasterio.open(*args, **kwargs), self._reads)

    def __getattr__(self, name: str) -> Any:
        return getattr(rasterio, name)


@needs_extra
@pytest.mark.asyncio
async def test_no_strip_read_is_taller_than_the_block(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every read of the strip path has a window, full width, no taller than
    the block (two window rows here; the last strip also takes the row under
    the last whole window), and together they cover the raster once. Through
    the tool, the package's whole reader is never called."""
    values = _scores(12, 1, 45, 31)
    run = _run_files(_studio_run(_scores_root, "reads", values))
    reads: list[Any] = []
    monkeypatch.setattr(scores_file, "BLOCK_VALUES", 31 * 8)
    spy = _SpyRasterio(reads)
    scores_file._pool_strips(np, spy, assess, run, False, 4, False)
    assert reads and all(window is not None for window in reads)
    assert all((int(w.col_off), int(w.width)) == (0, 31) for w in reads)
    assert [int(w.row_off) for w in reads] == [0, 8, 16, 24, 32, 40]
    assert [int(w.height) for w in reads] == [8, 8, 8, 8, 8, 5]

    def never(*_given: Any) -> Any:
        """The whole reader, which a strip read must not reach."""
        raise AssertionError("the whole reader was called")

    monkeypatch.setattr(scores_file, "MAX_VALUES", 100)
    monkeypatch.setattr(scores_file, "read_scores_raster", never)
    got = await _ok(TOOL, {"run_dir": "reads", "patch": 4})
    assert (got["grid"], got["n_valid"]) == ([11, 7], 77)
    assert got["pixels_outside_window_grid"] == 45 * 31 - 44 * 28


# --------------------------------------------------------------------------- 5: what stays refused


@needs_extra
@pytest.mark.asyncio
async def test_a_direct_runs_raster_over_the_limit_is_still_refused(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strips are for a raster this agent wrote from Studio's files. A direct
    run's raster over the limit keeps the refusal it had, word for word, and
    so does a Studio raster a strip would not read as the whole read does (a
    no-data value that is not NaN, a type that is not float32). A raster at
    the limit is read whole. Over the strip cap, a Studio raster is refused
    with that cap."""
    values = _scores(14, 1, 32, 24)
    _studio_run(_scores_root, "direct", values, source=False)
    _studio_run(_scores_root, "masked", values, profile_edit={"nodata": -1.0})
    _studio_run(_scores_root, "wide", values, profile_edit={"dtype": "float64"})
    _studio_run(_scores_root, "studio", values)

    def never(*_given: Any) -> Any:
        """The strip path, which none of these may reach."""
        raise AssertionError("the raster was read in strips")

    with monkeypatch.context() as patched:
        patched.setattr(scores_file, "_pool_strips", never)
        patched.setattr(scores_file, "MAX_VALUES", 100)
        for run in ("direct", "masked", "wide"):
            out = await _call(TOOL, {"run_dir": run})
            assert out["ok"] is False, run
            assert out["error"] == (
                "ValueError: the raster holds 1 x 32 x 24 values, more than 100 "
                "this tool reads; score a smaller area"
            ), run
        patched.setattr(scores_file, "MAX_VALUES", 32 * 24)
        at_limit = await _ok(TOOL, {"run_dir": "studio"})
        assert at_limit["n_valid"] == 48
    monkeypatch.setattr(scores_file, "MAX_VALUES", 100)
    monkeypatch.setattr(scores_file, "MAX_STRIP_VALUES", 500)
    out = await _call(TOOL, {"run_dir": "studio"})
    assert out["ok"] is False
    assert "1 x 32 x 24 values, more than 500 this tool reads" in out["error"]


@needs_extra
@pytest.mark.asyncio
async def test_a_map_with_no_valid_window_is_refused_in_the_packages_words_both_ways(
    _scores_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pixel of each window has a prediction, so no window is valid. A
    whole read is refused by the package (``NO_VALID_WINDOW``). In strips of
    one window row, each strip's refusal is taken as "no valid window here",
    and the map is refused once no strip had one, in the same words; no
    scores file is written either way."""
    values = np.full((1, 16, 16), np.nan, dtype=np.float32)
    values[0, 0::4, 0::4] = _scores(15, 1, 16, 16)[0, 0::4, 0::4]
    run = _studio_run(_scores_root, "sparse", values)
    args = {"run_dir": "sparse", "patch": 4}
    whole = await _call(TOOL, args)
    used: list[str] = []
    pool_strips, pool_whole = scores_file._pool_strips, scores_file._pool_whole

    def strips_spy(*given: Any) -> Any:
        """``_pool_strips``, noted."""
        used.append("strips")
        return pool_strips(*given)

    def whole_spy(*given: Any) -> Any:
        """``_pool_whole``, noted."""
        used.append("whole")
        return pool_whole(*given)

    with monkeypatch.context() as patched:
        patched.setattr(scores_file, "MAX_VALUES", 1)
        patched.setattr(scores_file, "BLOCK_VALUES", 16 * 4)
        patched.setattr(scores_file, "_pool_strips", strips_spy)
        patched.setattr(scores_file, "_pool_whole", whole_spy)
        strips = await _call(TOOL, args)
        assert scores_file._strip_rows(1, 16, 4) == 4
        with pytest.raises(ValueError) as refusal:
            pool_strips(np, rasterio, assess, _run_files(run), False, 4, False)
    assert used == ["strips"]
    assert whole["ok"] is False and strips["ok"] is False
    words = str(assess.NO_VALID_WINDOW)
    assert strips["error"] == whole["error"] == "ValueError: " + words
    assert str(refusal.value) == words
    assert not list(_scores_root.rglob("scores_run_*"))


def test_the_limits_and_the_strip_of_the_first_live_zone() -> None:
    """The whole-read limit is where it was; the strip cap and the strip size
    are the round's (r5-a K1). The cap on one zone's native file did not
    move: a one-band float32 score not compressed inside its GeoTIFF passes
    it up to about 128 million pixels."""
    assert scores_file.MAX_VALUES == 64_000_000
    assert scores_file.MAX_STRIP_VALUES == 256_000_000
    assert scores_file.BLOCK_VALUES == 2_000_000
    assert studio_scores.MAX_ENTRY_BYTES == 513_048_576
    assert studio_scores.MAX_ENTRY_BYTES // 4 == 128_262_144
    assert studio_scores.CERTIFICATE_MAX_VALUES == scores_file.MAX_VALUES
    # One strip of the zone the owner's first live run met: 7168 pixels wide,
    # pooled at 20 px (183,296 windows of 10240 x 7168 pixels).
    assert scores_file._strip_rows(1, 7168, 20) == 260
    assert scores_file.smallest_patch(10240, 7168) == 20
