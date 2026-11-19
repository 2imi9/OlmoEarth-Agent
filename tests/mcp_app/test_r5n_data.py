# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-n (19 Nov 2026), track DATA: the pre-release bug hunt's data path
(specs/bughunt-2026-11-19/confirmed.json), as specs/r5-n-round.md gives it.

- J01 (concurrency/F1 = files/F1): ``olmoearth_scores_from_file``'s read and
  pooling run in a worker thread, so each zone of a Studio read is pooled off
  the event loop.
- J02 (files/F3, files/F6, concurrency/F1): ``review_set.write_json_file``
  writes whole, mode 600, under a staged ``.part`` name moved over the
  target; a staged part a killed write left goes once an hour old.
- J03 (concurrency/F2): a Studio zip is kept in the call that decodes it, so
  a read stopped while it decodes still keeps it, and a read asked for again
  while that decode runs waits for it, then reads the copy.
- J04 (concurrency/F4 = files/F4): part files a killed write left an hour ago
  or more are removed, and a kept folder they alone kept goes too.
- J05 (files/F5): ``OLMOEARTH_STUDIO_KEEP_HOURS=0`` removes the copies kept
  before.
- J06 (files/F6): ``studio_results/`` is kept mode 700, at every read.
- J07 (files/F2): a direct run's raster that is not a GeoTIFF by its first
  bytes (a VRT named ``.tif``) is refused before any open.
- J08 (errors/B4): a sheet a spreadsheet saved is refused with the label
  sheet's save-again step, never with a raw KeyError, UnicodeDecodeError or
  int() message.
- J09 (errors/B6, packaging/PKG-3): the missing-extra answer says when the
  package is installed but does not import, or when Python is 3.14 or later,
  and when the package imports but rasterio does not.
- J10 (numbers/F2): a part not read as a map (no scores path, or no count of
  its squares) is no part left to check.
- J11 (numbers/F3): a part's share never rounds to "100.0%" beside a sliver,
  nor to "0.0%" with pixels in it.

No network. Tests that need the package or rasterio skip without them; the
rest need neither. The browser harness (scripts/mcp_views_e2e.py) draws J08,
J10 and J11 in 17_r5n_data_sheet_not_utf8, 17_r5n_data_part_cannot_be_checked
and 12_r5n_data_sliver_shares.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import threading
import time
import types as pytypes
import zipfile
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.mcp_app import check
from olmoearth_agent.tools import (
    estimation,
    inferencex,
    review_set,
    scores_file,
    studio_scores,
)
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools

#: A [0, 1] score result's id, its download token (a test value), its field
#: and its zone's file, as tests/mcp_app/test_mcp_r5h_server.py's.
RID = "5f0c2a71-3b9e-4d8a-9c61-0a7e2b4f1d33"
OTHER_RID = "6a1b2c3d-4e5f-4a6b-8c7d-8e9f0a1b2c3d"
DL_VALUE = "dl-r5n-test-only-3a9d"
SCORE_NAME = "sample_score"
SCORE_FILE = "result_epsg32616_0_native.tif"
ZIPPED = b"PK fake zip of a score result, r5-n " * 40

#: Two hours, in seconds: the age of a part file a killed write left.
OLD = 2 * 3600.0

#: A VRT that names another file: GDAL would read that file's bytes.
VRT = (
    b'<VRTDataset rasterXSize="2" rasterYSize="2"><VRTRasterBand '
    b'dataType="Float32" band="1"><SimpleSource><SourceFilename '
    b'relativeToVRT="1">other.tif</SourceFilename></SimpleSource>'
    b"</VRTRasterBand></VRTDataset>"
)

#: The label sheet's words a refused sheet's error carries (sheet._parse).
SAVE_AGAIN = "Save it again as CSV UTF-8, comma-separated"


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The scores root and the output root, a folder of the test's own; the
    kept copies at their default limit."""
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(root))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(root))
    monkeypatch.delenv(studio_scores.KEEP_HOURS_ENV, raising=False)
    return root.resolve()


def _ctx() -> ToolContext:
    """A tool context with no Studio client."""
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One call of the estimation or scores tools through the registry."""
    registry = ToolRegistry()
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_estimation_tools())
    return await registry.dispatch(ToolCall(id="1", name=name, arguments=args), _ctx())


def _mode(path: Path) -> int:
    """A file's or folder's permission bits."""
    return stat.S_IMODE(path.stat().st_mode)


def _aged(path: Path, data: bytes = b"part", age: float = OLD) -> Path:
    """Write ``path`` (its folder made) and set its mtime ``age`` seconds ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    then = time.time() - age
    os.utime(path, (then, then))
    return path


def _record(folder: Path, rid: str, age_hours: float) -> None:
    """A kept copy of ``rid`` in ``folder``, downloaded ``age_hours`` ago."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / studio_scores.KEPT_ZIP).write_bytes(b"old zip")
    stamp = time.strftime(
        studio_scores.KEPT_TIME, time.gmtime(time.time() - age_hours * 3600)
    )
    record = {
        "format": studio_scores.KEPT_FORMAT,
        "result_id": rid,
        "downloaded_at": stamp,
        "bytes": 7,
        "sha256": hashlib.sha256(b"old zip").hexdigest(),
    }
    (folder / studio_scores.KEPT_RECORD).write_text(json.dumps(record), "utf-8")


# --------------------------------------------------------------------------- J01


@pytest.mark.asyncio
async def test_j01_the_pooling_runs_in_a_worker_thread_and_the_loop_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler awaits the read and the pooling in a worker thread: the
    loop runs another coroutine while the pooling waits (on the loop, the
    pooling would wait out its 5 s alone), and the answer is the pooling's."""
    loop_thread = threading.get_ident()
    served = threading.Event()
    seen: dict[str, Any] = {}

    def pooled(args: dict[str, Any]) -> dict[str, Any]:
        """The read and pooling, stood in: waits until the loop served."""
        seen["thread"] = threading.get_ident()
        seen["args"] = dict(args)
        return {"available": True, "served": served.wait(5)}

    async def serve() -> None:
        """Another request, served by the loop meanwhile."""
        await asyncio.sleep(0.01)
        served.set()

    monkeypatch.setattr(scores_file, "_scores_from_file_sync", pooled)
    out, _ = await asyncio.gather(
        scores_file._scores_from_file({"run_dir": "zone", "patch": 4}, _ctx()), serve()
    )
    assert out == {"available": True, "served": True}
    assert seen["thread"] != loop_thread
    assert seen["args"] == {"run_dir": "zone", "patch": 4}


@pytest.mark.asyncio
async def test_j01_a_refusal_reaches_the_caller_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal raised in the worker thread reaches the caller (a Studio
    read's zone loop catches it as before) as the same ValueError."""

    def refused(args: dict[str, Any]) -> dict[str, Any]:
        """The pooling refuses the zone."""
        raise ValueError("patch 11 px is larger than the 6 x 9 map")

    monkeypatch.setattr(scores_file, "_scores_from_file_sync", refused)
    with pytest.raises(ValueError, match="patch 11 px is larger than the 6 x 9 map"):
        await scores_file._scores_from_file({"run_dir": "zone"}, _ctx())


# --------------------------------------------------------------------------- J02


def test_j02_a_json_file_is_written_whole_mode_600_with_no_staged_file(
    _roots: Path,
) -> None:
    """The file holds the payload, is mode 600 even over an older 644 file,
    and nothing else is left in its folder."""
    os.chmod(_roots, 0o755)  # noqa: S103 (a loose mode the code must tighten)
    path = Path(review_set.write_json_file("scores_run_m_abcd1234_p4.json", {"a": 1}))
    assert path == _roots / "scores_run_m_abcd1234_p4.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    assert _mode(path) == 0o600
    os.chmod(path, 0o644)
    review_set.write_json_file("scores_run_m_abcd1234_p4.json", {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert _mode(path) == 0o600
    assert [p.name for p in _roots.iterdir()] == [path.name]


def test_j02_a_write_that_fails_leaves_the_old_file_and_no_staged_one(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move that fails (a full disk, say) leaves the file a reader already
    had, whole, and removes the staged file."""
    path = Path(review_set.write_json_file("design_x.json", {"old": True}))

    def broken(src: Any, dst: Any) -> None:
        """The move fails."""
        raise OSError("no space left on device")

    monkeypatch.setattr(review_set.os, "replace", broken)
    with pytest.raises(OSError, match="no space left"):
        review_set.write_json_file("design_x.json", {"new": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert [p.name for p in _roots.iterdir()] == ["design_x.json"]


def test_j02_a_payload_that_cannot_be_written_touches_nothing(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload that is not JSON is refused and leaves no file (a guard: the
    old write also built the JSON first; the J02 change is shown by the tests
    around this one). The spy on the stdlib's mkstemp pins "JSON before any
    staged file", which also held of the old write."""
    calls: list[Any] = []
    real = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> Any:
        """mkstemp, noted."""
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    with pytest.raises(TypeError):
        review_set.write_json_file("scores_bad.json", {"x": object()})
    assert calls == []
    assert list(_roots.iterdir()) == []


def test_j02_a_staged_part_a_killed_write_left_goes_once_an_hour_old(
    _roots: Path,
) -> None:
    """A write killed part way runs no clean-up and leaves its staged
    ``.<name>.<random>.part``: the next write of that name removes it once
    it is an hour old, leaves a younger one (it may be a live write) and
    another name's, and writes the file whole. A name the user gave is
    matched as written, a ``[`` in it too; the staged name ends in .part."""
    name = "scores_run_m_abcd1234_p4.json"
    review_set.write_json_file(name, {"a": 1})
    old = _aged(_roots / f".{name}.deadbeef.part")
    young = _aged(_roots / f".{name}.cafecafe.part", age=60)
    other = _aged(_roots / ".design_x.json.0badf00d.part")
    review_set.write_json_file(name, {"a": 2})
    assert not old.exists()
    assert young.is_file() and other.is_file()
    assert json.loads((_roots / name).read_text(encoding="utf-8")) == {"a": 2}
    marked = "plan[1]_zone_a0.05.json"
    review_set.write_json_file(marked, {"b": 1})
    left = _aged(_roots / f".{marked}.0ddba11.part")
    review_set.write_json_file(marked, {"b": 2})
    assert not left.exists()
    assert json.loads((_roots / marked).read_text(encoding="utf-8")) == {"b": 2}
    staged: list[str] = []
    real = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> Any:
        """mkstemp, its staged name noted."""
        fd, path = real(*args, **kwargs)
        staged.append(Path(path).name)
        return fd, path

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(tempfile, "mkstemp", spy)
        review_set.write_json_file(name, {"a": 3})
    (made,) = staged
    assert made.startswith(f".{name}.") and made.endswith(".part")


# --------------------------------------------------------------------------- J03


def _score_record() -> dict[str, Any]:
    """A [0, 1] score result's record, its download token included."""
    return {
        "id": RID,
        "prediction_id": "99999999-8888-4777-8666-555555555555",
        "prediction_name": "Score test",
        "property_names": [SCORE_NAME],
        "download_token": DL_VALUE,
        "result_metadata": {
            "start_datetime": "2023-01-01T00:00:00Z",
            "end_datetime": "2023-12-31T00:00:00Z",
            "classification_fields": [],
            "regression_fields": [
                {
                    "property_name": SCORE_NAME,
                    "band_index": 1,
                    "min_value": 0.0,
                    "max_value": 1.0,
                },
            ],
        },
    }


class FakeStudio:
    """Studio for one [0, 1] score result: its record and its files, each
    download counted; no prediction or model is served."""

    def __init__(self) -> None:
        """No download yet."""
        self.downloads = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        """The result's record."""
        assert result_id == RID
        return _score_record()

    async def prediction_result_files(
        self, download_token: str, *, max_bytes: int = 0
    ) -> bytes:
        """The files zip, counted."""
        assert download_token == DL_VALUE
        self.downloads += 1
        return ZIPPED

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Not served here."""
        raise RuntimeError("not served by this fake")

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """Not served here."""
        raise RuntimeError("not served by this fake")


def _stand_ins(
    root: Path, monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> threading.Event:
    """The decode (which waits for ``release``), the pooling and the extra's
    checks stood in; returns the event the decode sets once it started."""
    entered = threading.Event()

    def read_zones(
        data: bytes, plan: Any, info: Any, clip: bool, np: Any, rasterio: Any
    ) -> tuple[list[Any], int, bool]:
        """One zone, once the test lets the decode end."""
        entered.set()
        release.wait(10)
        run = root / studio_scores.STUDIO_DIR / "Score-test" / "epsg32616_0"
        counts = {
            "valid_pixels": 240,
            "nodata_pixels": 16,
            "class_counts": {f"not {SCORE_NAME}": 120, SCORE_NAME: 120},
            "clipped": False,
        }
        zone = studio_scores.Zone(
            file=SCORE_FILE,
            run_dir=run,
            sha256="a" * 64,
            reused=False,
            counts=counts,
            height=16,
            width=16,
        )
        return [zone], 0, False

    async def pool(args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        """A pooled zone."""
        return {"available": True, "scores_path": str(root / "s.json"), "n_valid": 15}

    monkeypatch.setattr(studio_scores, "read_zones", read_zones)
    monkeypatch.setattr(studio_scores.scores_file, "_scores_from_file", pool)
    monkeypatch.setattr(studio_scores.inferencex, "load", lambda _name: None)
    modules = pytypes.SimpleNamespace(import_module=lambda _name: object())
    monkeypatch.setattr(studio_scores, "importlib", modules)
    return entered


async def _read(studio: FakeStudio) -> dict[str, Any]:
    """olmoearth_scores_from_studio's handler on ``studio``, at patch 4."""
    ctx = ToolContext(studio=studio, state=ThreadState())  # type: ignore[arg-type]
    out: dict[str, Any] = await studio_scores._scores_from_studio(
        {"result_id": RID, "patch": 4}, ctx
    )
    return out


@pytest.mark.asyncio
async def test_j03_a_read_stopped_while_it_decodes_still_keeps_the_zip(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read is stopped (as the Check card's preparation is past its
    time) while the decode runs: nothing is kept yet. Once the decode ends
    in its thread, the zip is kept, and the read asked for again reads that
    copy and downloads nothing."""
    release = threading.Event()
    entered = _stand_ins(_roots, monkeypatch, release)
    studio = FakeStudio()
    task = asyncio.ensure_future(_read(studio))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert studio_scores.read_kept(RID, len(ZIPPED)) is None
    release.set()
    kept = None
    for _ in range(250):
        kept = studio_scores.read_kept(RID, len(ZIPPED))
        if kept is not None:
            break
        await asyncio.sleep(0.02)
    assert kept is not None and kept[0] == ZIPPED
    again = await _read(studio)
    assert studio.downloads == 1
    assert "read_from_copy" in again


@pytest.mark.asyncio
async def test_j03_a_read_asked_again_while_the_stopped_decode_runs_waits_for_it(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read asked for again at once, while the stopped read's decode
    still runs (the Check card opened again as its failed face says): it
    neither downloads nor decodes beside that thread, but waits for it to end
    and keep the zip, then reads that copy. One download in all."""
    release = threading.Event()
    entered = _stand_ins(_roots, monkeypatch, release)
    studio = FakeStudio()
    first = asyncio.ensure_future(_read(studio))
    assert await asyncio.to_thread(entered.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.ensure_future(_read(studio))
    await asyncio.sleep(0.2)
    assert studio.downloads == 1
    assert not second.done()
    release.set()
    again = await asyncio.wait_for(second, 10)
    assert studio.downloads == 1
    assert "read_from_copy" in again
    key = (id(asyncio.get_running_loop()), RID)
    assert key not in studio_scores._DECODES


def test_j03_the_copy_read_is_not_kept_again_and_a_refusal_is_never_kept(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_read_then_keep keeps a zip downloaded now once it read; a zip read
    from the kept copy is not kept again, and a zip the decode refuses is
    never kept."""
    kept: list[tuple[str, bytes, str | None]] = []
    refuse = {"now": False}

    def read_zones(*args: Any) -> tuple[list[Any], int, bool]:
        """No zone, or a refusal."""
        if refuse["now"]:
            raise ValueError("refused by the stand-in")
        return [], 0, False

    monkeypatch.setattr(studio_scores, "read_zones", read_zones)
    monkeypatch.setattr(
        studio_scores, "_keep_quietly", lambda r, d, g: kept.append((r, d, g))
    )
    info = studio_scores.ResultInfo(RID, None, "x", None, None, None)
    plan: Any = None
    got = studio_scores._read_then_keep(
        ZIPPED, plan, info, False, None, None, "d1", None
    )
    assert got == ([], 0, False)
    assert kept == [(RID, ZIPPED, "d1")]
    studio_scores._read_then_keep(
        ZIPPED, plan, info, False, None, None, None, "2026-11-19T10:00:00Z"
    )
    assert len(kept) == 1
    refuse["now"] = True
    with pytest.raises(ValueError, match="refused by the stand-in"):
        studio_scores._read_then_keep(ZIPPED, plan, info, False, None, None, "d2", None)
    assert len(kept) == 1


# --------------------------------------------------------------------------- J04


def test_j04_keeping_removes_the_stale_parts_and_the_folders_they_kept(
    _roots: Path,
) -> None:
    """A keep removes the parts left an hour ago or more: beside another
    result's expired copy (whose folder then goes), in a folder with no
    record that held only such a part (it goes too) and in this result's
    own folder. A part younger than an hour stays, and so does an empty
    folder with no record (a keep may have just made it)."""
    base = _roots / studio_scores.KEPT_DIR
    expired = base / OTHER_RID
    _record(expired, OTHER_RID, 25)
    _aged(expired / ".files.zip.deadbeef.part")
    broken = base / "7b2c3d4e-5f6a-4b7c-9d8e-9f0a1b2c3d4e"
    _aged(broken / ".files.zip.0badf00d.part")
    live = base / "8c3d4e5f-6a7b-4c8d-8e9f-0a1b2c3d4e5f"
    fresh_part = _aged(live / ".files.zip.12345678.part", age=60)
    empty = base / "9d4e5f6a-7b8c-4d9e-8f0a-1b2c3d4e5f6a"
    empty.mkdir(parents=True)
    own = base / RID
    old_own = _aged(own / ".files.zip.abcdef01.part")
    young_own = _aged(own / ".kept.json.abcdef02.part", age=60)
    studio_scores.keep(RID, ZIPPED)
    assert not expired.exists()
    assert not broken.exists()
    assert fresh_part.is_file()
    assert empty.is_dir()
    assert not old_own.exists() and young_own.is_file()
    assert (own / studio_scores.KEPT_ZIP).read_bytes() == ZIPPED


def test_j04_forgetting_a_copy_removes_its_stale_parts_and_folder(
    _roots: Path,
) -> None:
    """A kept copy removed (a changed download) takes its stale parts with it,
    so its folder goes; a young part keeps the folder."""
    folder = _roots / studio_scores.KEPT_DIR / RID
    _record(folder, RID, 1)
    _aged(folder / ".files.zip.deadbeef.part")
    studio_scores.forget_kept(RID)
    assert not folder.exists()
    _record(folder, RID, 1)
    young = _aged(folder / ".files.zip.cafecafe.part", age=60)
    studio_scores.forget_kept(RID)
    assert young.is_file() and not (folder / studio_scores.KEPT_ZIP).exists()


class _Written:
    """A raster opened for writing by the fake rasterio: it writes a file of
    TIFF bytes at its path when it is closed."""

    def __init__(self, path: Path) -> None:
        """The part file to write."""
        self.path = Path(path)

    def __enter__(self) -> _Written:
        """Open it."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close it: its bytes are written."""
        self.path.write_bytes(b"II*\x00 a zone raster")

    def write(self, array: Any) -> None:
        """The values, taken."""

    def set_band_description(self, band: int, label: str) -> None:
        """A band's name, taken."""


class _FakeRasterio:
    """rasterio, as far as studio_scores._write_zone uses it."""

    def open(self, path: Any, mode: str = "r", **profile: Any) -> _Written:
        """A raster to write."""
        assert mode == "w" and profile["driver"] == "GTiff"
        return _Written(path)


def _write_zone(root: Path) -> Path:
    """studio_scores._write_zone of a one-band 2 x 2 zone under ``root``, with
    the fake rasterio; its folder."""
    zone_dir = root / studio_scores.STUDIO_DIR / "Score-test-5f0c2a71" / "epsg32616_0"
    src = pytypes.SimpleNamespace(crs=None, transform=None)
    array = pytypes.SimpleNamespace(shape=(1, 2, 2))
    manifest = {"scores": {"file": studio_scores.RASTER, "values": "probabilities"}}
    studio_scores._write_zone(
        _FakeRasterio(), src, array, zone_dir, manifest, (SCORE_NAME,)
    )
    return zone_dir


def test_j04_a_zone_write_removes_its_folders_stale_parts(_roots: Path) -> None:
    """The raster's and the manifest's parts left an hour ago or more go when
    the zone is written; a young part stays; the zone's files are written."""
    zone_dir = _roots / studio_scores.STUDIO_DIR / "Score-test-5f0c2a71" / "epsg32616_0"
    old_raster = _aged(zone_dir / ".scores.tif.aaaaaaaa.part")
    old_manifest = _aged(zone_dir / ".manifest.json.bbbbbbbb.part")
    young = _aged(zone_dir / ".scores.tif.cccccccc.part", age=60)
    other = _aged(zone_dir / ".notes.txt.dddddddd.part")
    _write_zone(_roots)
    assert not old_raster.exists() and not old_manifest.exists()
    assert young.is_file() and other.is_file()
    assert (zone_dir / studio_scores.RASTER).is_file()
    assert (zone_dir / scores_file.MANIFEST).is_file()


# --------------------------------------------------------------------------- J05


def test_j05_keeping_off_removes_the_copies_kept_before(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy kept at the default limit is removed by the next download once
    OLMOEARTH_STUDIO_KEEP_HOURS is 0, and that download keeps nothing."""
    studio_scores.keep(OTHER_RID, ZIPPED)
    first = _roots / studio_scores.KEPT_DIR / OTHER_RID / studio_scores.KEPT_ZIP
    assert first.read_bytes() == ZIPPED
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "0")
    studio_scores.keep(RID, ZIPPED)
    assert not first.exists()
    assert not (_roots / studio_scores.KEPT_DIR / OTHER_RID).exists()
    assert not (_roots / studio_scores.KEPT_DIR / RID).exists()
    assert studio_scores.read_kept(OTHER_RID, len(ZIPPED)) is None


# --------------------------------------------------------------------------- J06


def test_j06_the_zones_folder_is_mode_700_under_an_open_root(_roots: Path) -> None:
    """Under a root of mode 755 whose studio_results already existed open,
    a zone write makes studio_results 700; a zone's scores file is 600."""
    os.chmod(_roots, 0o755)  # noqa: S103 (a loose mode the code must tighten)
    top = _roots / studio_scores.STUDIO_DIR
    top.mkdir()
    os.chmod(top, 0o755)  # noqa: S103 (a loose mode the code must tighten)
    _write_zone(_roots)
    assert _mode(top) == 0o700
    path = Path(review_set.write_json_file("scores_run_m_12345678_p4.json", {"s": []}))
    assert _mode(path) == 0o600


def _score_zip(tmp_path: Path, rasterio: Any, np: Any) -> bytes:
    """A [0, 1] score result's files zip: one 4 x 4 zone's GeoTIFF, written by
    rasterio on a grid at the origin (no place)."""
    from rasterio.transform import Affine

    band = np.full((1, 4, 4), 0.25, dtype=np.float32)
    band[0, 0, :] = 0.75
    native = tmp_path / SCORE_FILE
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32616",
        "transform": Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        "nodata": -1.0,
    }
    with rasterio.open(native, "w", **profile) as dst:
        dst.write(band)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"Score test/{SCORE_FILE}", native.read_bytes())
    return buf.getvalue()


def test_j06_a_read_whose_zones_are_all_reused_closes_studio_results_too(
    _roots: Path, tmp_path: Path
) -> None:
    """A result read before (its zone on disk) and read again after its
    studio_results was left open (mode 755, as a release before this one
    left it): the zone comes back reused, and studio_results is 700."""
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    data = _score_zip(tmp_path, rasterio, np)
    plan = studio_scores.files_plan(_score_record())
    info = studio_scores.ResultInfo(RID, None, "Score test", None, None, None)
    first, _, _ = studio_scores.read_zones(data, plan, info, False, np, rasterio)
    assert [zone.reused for zone in first] == [False]
    top = _roots / studio_scores.STUDIO_DIR
    os.chmod(top, 0o755)  # noqa: S103 (a loose mode the code must tighten)
    again, _, _ = studio_scores.read_zones(data, plan, info, False, np, rasterio)
    assert [zone.reused for zone in again] == [True]
    assert _mode(top) == 0o700


# --------------------------------------------------------------------------- J07


def _manifest_run(root: Path, name: str, raster: bytes) -> Path:
    """A direct run's folder: a manifest naming scores.tif, which holds
    ``raster``."""
    run = root / name
    run.mkdir()
    manifest = {"scores": {"file": "scores.tif", "values": "probabilities"}}
    (run / scores_file.MANIFEST).write_text(json.dumps(manifest), "utf-8")
    (run / "scores.tif").write_bytes(raster)
    return run


def _summary_run(root: Path, name: str, raster: bytes) -> Path:
    """A from-olmoearth run's folder: its summary listing one grid, whose
    scores_32610.tif holds ``raster``."""
    run = root / name
    run.mkdir()
    summary = {"grids": [{"label": "32610", "shape": [2, 2]}], "bands": 2}
    (run / scores_file.SUMMARY).write_text(json.dumps(summary), "utf-8")
    (run / "scores_32610.tif").write_bytes(raster)
    return run


def test_j07_a_tiff_is_told_by_its_first_bytes(_roots: Path) -> None:
    """Classic and big TIFF, either byte order, pass; a VRT, text, a folder
    or a missing file do not. Studio's files are checked against the same."""
    for magic in scores_file.TIFF_MAGIC:
        tif = _roots / f"t{magic.hex()}.tif"
        tif.write_bytes(magic + b"rest")
        assert scores_file._plain_tiff(tif) is True
    vrt = _roots / "v.tif"
    vrt.write_bytes(VRT)
    assert scores_file._plain_tiff(vrt) is False
    assert scores_file._plain_tiff(_roots) is False
    assert scores_file._plain_tiff(_roots / "missing.tif") is False
    assert studio_scores.TIFF_MAGIC is scores_file.TIFF_MAGIC


def test_j07_a_vrt_named_tif_is_refused_before_any_open(_roots: Path) -> None:
    """From the manifest and from a from-olmoearth summary alike, a raster
    holding VRT XML is refused in the tool's words, before rasterio is used
    (here there is none to use); a folder of VRTs is refused without opening
    one, and none of the class-id checks reads it."""
    words = (
        " is not a plain GeoTIFF by its first bytes (a VRT or another format); "
        "write the scores as a GeoTIFF"
    )
    run = _manifest_run(_roots, "vrt_manifest", VRT)
    with pytest.raises(ValueError) as caught:
        scores_file.read_run_files(run, None, None, None, "FIX")
    assert str(caught.value) == "scores.tif" + words
    run = _summary_run(_roots, "vrt_summary", VRT)
    with pytest.raises(ValueError) as caught:
        scores_file.read_run_files(run, None, None, None, "FIX")
    assert str(caught.value) == "scores_32610.tif" + words
    loose = _roots / "vrt_loose"
    loose.mkdir()
    (loose / "a.tif").write_bytes(VRT)
    assert scores_file._class_ids(None, None, loose / "a.tif") is None
    refusal = scores_file._no_record(loose, None, None, "FIX")
    assert "holds neither manifest.json" in str(refusal)
    assert "a.tif" not in str(refusal)
    named = str(scores_file._not_plain_tiff("x.tif"))
    assert named == "x.tif" + words


@pytest.mark.asyncio
async def test_j07_the_tool_refuses_a_vrt_from_both_records(_roots: Path) -> None:
    """Through olmoearth_scores_from_file, with the package: the manifest's
    raster and the summary's, each holding VRT XML, are refused, and the
    other file the VRT names is never read."""
    pytest.importorskip("oe_inferencex.review.assess")
    pytest.importorskip("rasterio")
    _manifest_run(_roots, "vrt_manifest", VRT)
    _summary_run(_roots, "vrt_summary", VRT)
    for run in ("vrt_manifest", "vrt_summary"):
        out = await _call("olmoearth_scores_from_file", {"run_dir": run})
        assert out["ok"] is False, out
        assert "is not a plain GeoTIFF by its first bytes" in out["error"]


# --------------------------------------------------------------------------- J08


def _sheet_rows() -> list[dict[str, str]]:
    """A filled sheet's rows: three windows, each labelled."""
    return [
        {"order": "1", "window_index": "4", "wrong": "0", "reference_class": "0"},
        {"order": "2", "window_index": "9", "wrong": "1", "reference_class": "1"},
        {"order": "3", "window_index": "2", "wrong": "0", "reference_class": "1"},
    ]


def _write_sheet(
    path: Path,
    rows: list[dict[str, str]],
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    blank_row: bool = False,
) -> str:
    """``rows`` written as a sheet, as a spreadsheet might save it; its path."""
    fields = list(rows[0])
    with path.open("w", encoding=encoding, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
        if blank_row:
            fh.write(delimiter * (len(fields) - 1) + "\r\n")
    return str(path)


def test_j08_a_sheet_saved_with_semicolons_is_refused_in_the_sheets_words(
    _roots: Path,
) -> None:
    """A German or French spreadsheet's ';' sheet: the label sheet's words,
    never a KeyError, through the estimate's reading of labels_path too."""
    path = _write_sheet(_roots / "plan_to_label.csv", _sheet_rows(), delimiter=";")
    words = (
        "plan_to_label.csv has no window_index column; a spreadsheet may have "
        "saved it with ';' or a tab. " + SAVE_AGAIN
    )
    with pytest.raises(ValueError) as caught:
        estimation._read_labels_csv(path)
    assert str(caught.value) == words
    with pytest.raises(ValueError, match="has no window_index column"):
        estimation._labels_unjudged({"labels_path": path}, [4, 9, 2])


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16"])
def test_j08_a_sheet_not_in_utf8_is_refused_in_the_sheets_words(
    _roots: Path, encoding: str
) -> None:
    """A plain CSV of a spreadsheet (cp1252 with an accent in a note) or a
    UTF-16 one: the save-again step, never a UnicodeDecodeError."""
    rows = [{**row, "note": "forêt"} for row in _sheet_rows()]
    path = _write_sheet(_roots / "plan_to_label.csv", rows, encoding=encoding)
    with pytest.raises(ValueError) as caught:
        estimation._read_labels_csv(path)
    assert str(caught.value) == (
        "plan_to_label.csv is not UTF-8; a spreadsheet saved it in another "
        "encoding. " + SAVE_AGAIN
    )
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_j08_a_blank_row_is_skipped_and_a_byte_order_mark_read_past(
    _roots: Path,
) -> None:
    """A row of commas a spreadsheet left is skipped, and a UTF-8 sheet with a
    byte-order mark (a spreadsheet's "CSV UTF-8") reads as the plain one."""
    want = {4: ("0", "0"), 9: ("1", "1"), 2: ("0", "1")}
    path = _write_sheet(_roots / "a_to_label.csv", _sheet_rows(), blank_row=True)
    assert estimation._read_labels_csv(path) == want
    marked = _write_sheet(
        _roots / "b_to_label.csv", _sheet_rows(), encoding="utf-8-sig"
    )
    assert Path(marked).read_bytes().startswith(b"\xef\xbb\xbf")
    assert estimation._read_labels_csv(marked) == want


def test_j08_a_window_index_that_is_not_a_whole_number_is_named_by_its_row(
    _roots: Path,
) -> None:
    """A row whose window_index is not a whole number (blank beside a label,
    or text) is refused naming its line, never with int()'s message."""
    rows = _sheet_rows()
    rows[1]["window_index"] = "x"
    path = _write_sheet(_roots / "plan_to_label.csv", rows)
    with pytest.raises(ValueError) as caught:
        estimation._read_labels_csv(path)
    assert str(caught.value) == (
        "row 3 of plan_to_label.csv: window_index is 'x', not a whole number; keep "
        "the plan's window_index column as it was written"
    )
    rows[1]["window_index"] = ""
    path = _write_sheet(_roots / "plan_to_label.csv", rows)
    with pytest.raises(
        ValueError, match="row 3 of plan_to_label.csv: window_index is ''"
    ):
        estimation._read_labels_csv(path)


@pytest.mark.asyncio
async def test_j08_the_estimate_of_a_spreadsheets_sheet_gives_the_save_again_step(
    _roots: Path,
) -> None:
    """Through olmoearth_estimate_map_error with the plan's design and a
    labels_path a spreadsheet saved with ';', then in cp1252: each a refusal
    naming the save-again step, with no KeyError or UnicodeDecodeError."""
    pytest.importorskip("oe_inferencex.estimate")
    scores = [[0.9 - 0.002 * i, 0.1 + 0.002 * i] for i in range(400)]
    planned = await _call(
        "olmoearth_plan_label_sample",
        {"scores": scores, "grid": 20, "budget": 20, "design": "random"},
    )
    assert planned["ok"] is True, planned
    plan = planned["result"]
    with Path(plan["labels_csv_path"]).open(encoding="utf-8", newline="") as fh:
        rows = [{**row, "wrong": "0"} for row in csv.DictReader(fh)]
    semi = _write_sheet(_roots / "semi_to_label.csv", rows, delimiter=";")
    noted = [{**row, "note": "forêt"} for row in rows]
    latin = _write_sheet(_roots / "latin_to_label.csv", noted, encoding="cp1252")
    for path in (semi, latin):
        out = await _call(
            "olmoearth_estimate_map_error",
            {"design_path": plan["design_path"], "labels_path": path},
        )
        assert out["ok"] is False, out
        assert out["error"].startswith("ValueError: ")
        assert SAVE_AGAIN in out["error"]
        assert "KeyError" not in out["error"] and "codec" not in out["error"]


# --------------------------------------------------------------------------- J09


def test_j09_installed_but_not_importable_is_said_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current release whose import fails (a numpy or rasterio build that
    does not match) is never "not installed": the reinstall line."""
    monkeypatch.setattr(inferencex, "version", lambda: "1.20.3")
    monkeypatch.setattr(inferencex, "python_version", lambda: (3, 12))
    out = inferencex.missing_extra("Estimating a map's error rate", "numpy broke")
    said = "which is installed here but cannot be imported. Nothing was computed;"
    assert said in out["reason"]
    assert "not installed" not in out["reason"]
    assert out["install"] == inferencex.REINSTALL_HINT
    hint = "pip install --force-reinstall 'olmoearth-agent[inferencex]'"
    assert inferencex.REINSTALL_HINT == hint
    assert out["missing"] == "numpy broke"


@pytest.mark.parametrize(("running", "said"), [((3, 14), "3.14"), ((3, 15), "3.15")])
def test_j09_python_314_or_later_names_the_python_and_runs_the_agent_on_312(
    monkeypatch: pytest.MonkeyPatch, running: tuple[int, int], said: str
) -> None:
    """With no release installed on Python 3.14 or later, where the extra
    installs nothing, the clause names that Python and the line runs the
    agent on Python 3.12, never a pip line."""
    monkeypatch.setattr(inferencex, "version", lambda: None)
    monkeypatch.setattr(inferencex, "python_version", lambda: running)
    status, install = inferencex.unavailable()
    assert status == (
        f"which does not install on Python {said} (it supports Python 3.11 to 3.13)"
    )
    assert install == inferencex.PYTHON_HINT
    assert "--python 3.12" in install and "olmoearth-agent-mcp" in install
    assert "uv sync --python 3.12 --all-extras" in install
    assert "pip install" not in install
    out = inferencex.missing_extra("Planning a labelled sample")
    assert f"Python {said}" in out["reason"] and "missing" not in out


def test_j09_absent_on_312_and_older_releases_say_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no release on Python 3.11 to 3.13 the answer is still "not
    installed" with the pip line; an older release still gets the upgrade
    line, whatever the Python. python_version is the running Python's."""
    assert inferencex.python_version() == (sys.version_info[0], sys.version_info[1])
    monkeypatch.setattr(inferencex, "python_version", lambda: (3, 12))
    monkeypatch.setattr(inferencex, "version", lambda: None)
    assert inferencex.unavailable() == (
        "which is not installed in this environment",
        inferencex.INSTALL_HINT,
    )
    monkeypatch.setattr(inferencex, "python_version", lambda: (3, 14))
    monkeypatch.setattr(inferencex, "version", lambda: "1.20.2")
    assert inferencex.unavailable() == (
        "and this environment has version 1.20.2, which is older",
        inferencex.UPGRADE_HINT,
    )


@pytest.mark.asyncio
async def test_j09_the_tools_answers_name_the_python_and_keep_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The estimate and the scores tool, their package unimportable on
    Python 3.14: each names the Python, gives the line that runs the agent
    on 3.12 and keeps the import's own error as missing."""
    monkeypatch.setitem(sys.modules, "oe_inferencex.estimate", None)
    monkeypatch.setitem(sys.modules, "oe_inferencex.review.assess", None)
    monkeypatch.setattr(inferencex, "version", lambda: None)
    monkeypatch.setattr(inferencex, "python_version", lambda: (3, 14))
    for name, args in (
        ("olmoearth_estimate_map_error", {"design_path": "x.json", "wrong": [1]}),
        ("olmoearth_scores_from_file", {"run_dir": "anything"}),
    ):
        out = await _call(name, args)
        assert out["ok"] is True, out
        result = out["result"]
        assert result["available"] is False
        assert "which does not install on Python 3.14" in result["reason"]
        assert result["install"] == inferencex.PYTHON_HINT
        assert "oe_inferencex" in result["missing"]


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", ["1.20.3", None])
async def test_j09_the_package_imports_and_rasterio_does_not_says_so(
    monkeypatch: pytest.MonkeyPatch, installed: str | None
) -> None:
    """olmoearth-inferencex's modules load and rasterio does not: the scores
    answer says so, never "cannot be imported" or "not installed" of the
    package (its version known, or none for an import from a source path),
    nor the Python 3.14 clause, and gives the reinstall line, which installs
    the geo extra too."""
    monkeypatch.setattr(
        scores_file.inferencex, "load", lambda _name: pytypes.ModuleType("x")
    )
    monkeypatch.setitem(sys.modules, "rasterio", None)
    monkeypatch.setattr(scores_file.inferencex, "version", lambda: installed)
    monkeypatch.setattr(scores_file.inferencex, "python_version", lambda: (3, 14))
    out = await _call("olmoearth_scores_from_file", {"run_dir": "anything"})
    assert out["ok"] is True, out
    result = out["result"]
    assert result["available"] is False
    assert "rasterio" in result["missing"]
    for untrue in ("cannot be imported", "not installed", "does not install"):
        assert untrue not in result["reason"]
    assert result["reason"] == (
        "Reading a model run's scores raster needs the optional package "
        f"olmoearth-inferencex (>= {inferencex.MIN_VERSION}) with rasterio (its "
        "geo extra); olmoearth-inferencex imports here, but rasterio does not. "
        "Nothing was read; do not rank the map's windows by hand."
    )
    assert result["install"] == inferencex.REINSTALL_HINT


# --------------------------------------------------------------------------- J10

#: Counts offered without the package, for every zone check plans.
OFFER = {
    "offer": [{"labels": 20, "width": 0.2, "sentence": "Twenty.", "rate_note": "r"}],
    "refused": None,
}


def _canned(monkeypatch: pytest.MonkeyPatch) -> None:
    """check.offers stood in: 20 squares for every zone, no package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`OFFER`."""
        return copy.deepcopy(OFFER)

    monkeypatch.setattr(check, "offers", fake)


def _studio_inner(
    root: Path,
    readable: tuple[bool, ...],
    pixels: tuple[int, ...],
    rid: str = "res-1",
) -> dict[str, Any]:
    """olmoearth_scores_from_studio's result for a map in parts: each zone
    read (with a scores file) or refused at read (no scores path), and the
    studio_zones fact written from the zones' valid pixels as the tool
    writes it."""
    files = [f"result_epsg{32610 + i}_{i}_native.tif" for i in range(len(readable))]
    zones: list[dict[str, Any]] = []
    for i, (file, ok) in enumerate(zip(files, readable)):
        if ok:
            path = root / f"{rid}_scores_{i}.json"
            path.write_text("{}", encoding="utf-8")
            zones.append(
                {
                    "file": file,
                    "scores_path": str(path),
                    "pooled": {"n_valid": 600, "raster_check": {"sha256": str(i) * 64}},
                }
            )
        else:
            zones.append(
                {
                    "file": file,
                    "scores_path": None,
                    "refused": "patch 11 px is larger than the 6 x 9 map",
                }
            )
    made = [
        studio_scores.Zone(
            file=file,
            run_dir=root,
            sha256="0" * 64,
            reused=False,
            counts={"valid_pixels": n},
        )
        for file, n in zip(files, pixels)
    ]
    return {
        "result_id": rid,
        "prediction_id": "pred-1",
        "zones": zones,
        "facts": [studio_scores._zones_fact(made)],
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
    }


def _drawn(root: Path, key: str, name: str) -> Path:
    """Finish a claimed draw as drawn, with a sheet file of its own."""
    sheet_path = root / f"{name}_to_label.csv"
    sheet_path.write_text("order,window_index,wrong\n", encoding="utf-8")
    answer = {
        "ok": True,
        "result": {
            "available": True,
            "labels_csv_path": str(sheet_path),
            "design_path": str(root / f"{name}.json"),
            "n_population": 600,
            "must_state": ["Stated."],
        },
    }
    assert check.finish_draw(key, answer)["state"] == "drawn"
    return sheet_path


def test_j10_the_sheet_context_says_which_part_was_not_read(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part refused at read (its offer record holds no scores path) is
    readable false in the drawn part's sheet context; the drawn part, and a
    part read as a map, are readable."""
    _canned(monkeypatch)
    west, east = check.offer_zones(
        _studio_inner(_roots, (True, False), (999_588, 412)), map_name="Kelp Q2"
    )
    assert east["refused_at"] == check.REFUSED_READ
    check.claim_draw(west["check_key"], 20)
    sheet_path = _drawn(_roots, west["check_key"], "west")
    got = check.sheet_context(str(sheet_path))
    assert got is not None
    assert [(p["part"], p["readable"]) for p in got["parts"]] == [
        ("West part", True),
        ("East part", False),
    ]
    both = check.offer_zones(
        _studio_inner(_roots, (True, True), (600, 400), rid="res-2"), map_name="Q3"
    )
    check.claim_draw(both[0]["check_key"], 20)
    other = _drawn(_roots, both[0]["check_key"], "west_both")
    again = check.sheet_context(str(other))
    assert again is not None
    assert [p["readable"] for p in again["parts"]] == [True, True]


def test_j10_a_part_whose_scores_file_gives_no_count_is_not_readable(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other read refusal: a part with a scores path whose scores file
    gives no count of its squares (its Check card says "This part cannot be
    checked" too) is readable false; a part refused only at plan (every field
    there, no count offered) is still readable."""
    _canned(monkeypatch)
    inner = _studio_inner(_roots, (True, True), (999_588, 412), rid="res-3")
    inner["zones"][1]["pooled"] = {"raster_check": {"sha256": "1" * 64}}
    west, east = check.offer_zones(inner, map_name="Kelp Q2")
    assert east["refused_at"] == check.REFUSED_READ
    check.claim_draw(west["check_key"], 20)
    sheet_path = _drawn(_roots, west["check_key"], "west_no_count")
    got = check.sheet_context(str(sheet_path))
    assert got is not None
    assert [(p["part"], p["readable"]) for p in got["parts"]] == [
        ("West part", True),
        ("East part", False),
    ]

    def offered(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """Counts for a part of 600 squares; none for the part of 7."""
        if n_valid == 7:
            return {"offer": [], "refused": "Too few squares to check."}
        return copy.deepcopy(OFFER)

    monkeypatch.setattr(check, "offers", offered)
    planned = _studio_inner(_roots, (True, True), (600, 400), rid="res-4")
    planned["zones"][1]["pooled"]["n_valid"] = 7
    first, second = check.offer_zones(planned, map_name="Q4")
    assert second["refused_at"] == check.REFUSED_PLAN
    check.claim_draw(first["check_key"], 20)
    drawn = _drawn(_roots, first["check_key"], "west_planned")
    seen = check.sheet_context(str(drawn))
    assert seen is not None
    assert [p["readable"] for p in seen["parts"]] == [True, True]


def _estimate_inner() -> dict[str, Any]:
    """An ok estimate's result: 80 labels on a random design, 12.5% wrong."""
    return {
        "available": True,
        "plan_design": "random",
        "n_labelled": 80,
        "n_unjudged": 0,
        "nominal_coverage": 0.95,
        "valid_interval": {
            "estimate": 0.125,
            "low": 0.061,
            "high": 0.217,
            "method": "exact hypergeometric interval at 95%",
        },
        "must_state": ["The error rate is estimated at 12.5%."],
    }


def test_j10_a_part_not_read_as_a_map_is_no_part_left_to_check(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The West part's result of a map whose East part was not read: the
    East part is "East part: cannot be checked", state unreadable, never
    "not checked yet", and Claude is told to report side by side, not to
    check the other part. The same part read as a map is as before."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server, sheet

    west = {
        "part": "West part",
        "share_value": 0.999588,
        "labels_csv_path": str(_roots / "west_to_label.csv"),
        "this": True,
        "readable": True,
    }
    east = {
        "part": "East part",
        "share_value": 0.000412,
        "labels_csv_path": None,
        "this": False,
        "readable": False,
    }
    ctx = {
        "result_id": "r1",
        "zone_file": "result_epsg32610_0_native.tif",
        "part": "West part",
        "n_parts": 2,
        "parts_known": True,
        "map_name": "Kelp Q2",
        "period": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "parts": [west, east],
    }
    monkeypatch.setattr(server.check, "sheet_context", lambda p: copy.deepcopy(ctx))
    monkeypatch.setattr(server.imagery, "served_any", lambda p: False)
    monkeypatch.setattr(sheet, "labelling_rule", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_state", lambda p: None, raising=False)
    monkeypatch.setattr(sheet, "graded_squares", lambda p, v: None, raising=False)
    monkeypatch.setattr(sheet, "class_counts", lambda p, v: None, raising=False)
    monkeypatch.setattr(
        sheet,
        "sheet_progress",
        lambda p: {"finish_label": "Estimate the error rate"},
        raising=False,
    )
    mark = {"first_at": "2026-11-13T10:04:00Z", "changed": False, "gradings": 1}
    calm = ("v" + "b" * 16, mark)
    path = str(_roots / "west_to_label.csv")
    got = server._estimate_result(_estimate_inner(), path, calm)
    assert got is not None
    assert server.PART_UNREADABLE == "{part}: cannot be checked"
    assert got["view"]["parts"][1] == {
        "name": "East part",
        "share": 0.000412,
        "state": "unreadable",
        "label": "East part: cannot be checked",
    }
    said = " ".join(got["say_first"])
    assert server.SAY_OTHER not in said
    assert server.SAY_SIDE_BY_SIDE in said
    assert "not checked yet" not in json.dumps(got["view"])
    east["readable"] = True
    before = server._estimate_result(_estimate_inner(), path, calm)
    assert before is not None
    assert before["view"]["parts"][1]["state"] == "not"
    assert before["view"]["parts"][1]["label"] == "East part: not checked yet"
    assert server.SAY_OTHER in " ".join(before["say_first"])


# --------------------------------------------------------------------------- J11


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.99962, ">99.9%"),
        (0.99951, ">99.9%"),
        (0.9994, "99.9%"),
        (1.0, "100.0%"),
        (0.467, "46.7%"),
        (0.00038, "0.038%"),
        (4e-7, "<0.0001%"),
        (1e-6, "0.0001%"),
        (0.0, "0.0%"),
        (None, "n/a"),
    ],
)
def test_j11_a_share_never_rounds_to_an_untrue_figure(
    value: float | None, text: str
) -> None:
    """share_percent writes a share as percent does, but a part under the
    whole map never reads "100.0%", and one above 0 never "0.0%" or in
    scientific notation."""
    assert rules.share_percent(value) == text


def test_j11_the_result_sentence_and_the_card_say_a_sliver_truly(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A map of 9,996,200 pixels in one part and 3,800 in another (shares
    0.99962 and 0.00038), and one of a single pixel beside 10 million: the
    studio_zones sentence and the Check card's shares say ">99.9%", "0.038%"
    and "<0.0001%" (the single pixel's 6-decimal share is 0.0), never
    "100.0%" or "0.0%"; a fact without counts reads its share as before."""
    big, small = (
        studio_scores.Zone(
            file=f"result_epsg{code}_{i}_native.tif",
            run_dir=_roots,
            sha256="0" * 64,
            reused=False,
            counts={"valid_pixels": n},
        )
        for i, (code, n) in enumerate(((32610, 9_996_200), (32611, 3_800)))
    )
    fact = studio_scores._zones_fact([big, small])
    assert ">99.9%" in fact["sentence"] and "0.038%" in fact["sentence"]
    assert "100.0%" not in fact["sentence"]
    assert [z["share"] for z in fact["zones"]] == [0.99962, 0.00038]
    assert check._shares({"facts": [fact]}) == {
        big.file: ">99.9%",
        small.file: "0.038%",
    }
    one = studio_scores.Zone(
        file="result_epsg32611_1_native.tif",
        run_dir=_roots,
        sha256="0" * 64,
        reused=False,
        counts={"valid_pixels": 1},
    )
    huge = studio_scores.Zone(
        file="result_epsg32610_0_native.tif",
        run_dir=_roots,
        sha256="0" * 64,
        reused=False,
        counts={"valid_pixels": 10_000_000},
    )
    tiny = studio_scores._zones_fact([huge, one])
    assert tiny["zones"][1]["share"] == 0.0
    assert check._shares({"facts": [tiny]}) == {
        huge.file: ">99.9%",
        one.file: "<0.0001%",
    }
    plain = {"id": "studio_zones", "zones": [{"file": "a.tif", "share": 0.467}]}
    assert check._shares({"facts": [plain]}) == {"a.tif": "46.7%"}
    _canned(monkeypatch)
    west, east = check.offer_zones(
        _studio_inner(_roots, (True, True), (9_996_200, 3_800)), map_name="Kelp Q2"
    )
    assert (west["share"], east["share"]) == (">99.9%", "0.038%")
    assert (west["share_value"], east["share_value"]) == (0.99962, 0.00038)
