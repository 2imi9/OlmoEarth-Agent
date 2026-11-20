# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-p (19 Nov 2026), track CLOUD, as specs/r5-p-round.md gives it.

The owner, labelling live: "right now labeling contains a lot of can't-decide
cloud pictures". Each month's date is now the scene with the least cloud and
shadow on the outlined square by the scene's own Scene Classification Layer
(``s2_fetch.MASK``, classes ``s2_fetch.COVERED_CLASSES``), not the least
whole-scene cloud; the dates the mask marks are folded away in the pane.

- ``s2_fetch``: the mask's link on a scene (``scene_of``, ``as_record``,
  ``from_record``, derived for a record made before the round), ``read_mask``,
  ``cover``, and the rule in ``month_stack`` and ``dated`` (a 4-tuple, the
  masks' own fetch and budget, a mask that fails never fails the pictures).
- ``imagery``: ``PICK_RULE``, ``R5_PICK_RULE``, ``PICK``, ``CAPTION``, each
  date's ``cloud_here`` and ``mask_line``, the answers' ``cloudy_dates`` and
  ``all_cloudy``; the consent question and its hash do not change.
- ``views/sheet.js`` (read as text; no test here runs the script): the fold
  of cloudy dates in ``OE.datesBlock`` and the mask lines in ``cloudFold``.
  The browser harness (scripts/mcp_views_e2e.py, 10p_kelp_cloudy_dates) draws
  them.

No test touches the network: a fake catalogue and bucket serve small GeoTIFFs
written here. The fake's masks are written on the bands' 10 m grid (the real
ones are 20 m): the code reads any grid onto the picture's.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import time
import types
from pathlib import Path
from typing import Any

import pytest

rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")
import httpx  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from olmoearth_agent.mcp_app import (  # noqa: E402
    imagery,
    review,
    s2_fetch,
    sheet,
    views,
)

ROOT = Path(__file__).resolve().parents[2]
BUCKET = f"https://{s2_fetch.BUCKET_HOST}"
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
ASSETS = {"red": "B04", "green": "B03", "blue": "B02", "nir": "B08"}
#: The fake scenes' grid: 10 m pixels from this corner, 350 x 300 pixels.
SKY_ORIGIN = (499000, 3701000)
#: The picture of window 14 of the map (row 1, column 4): seven squares a
#: side, the square outlined in its middle, as ``imagery._window_box`` makes it.
BOX = s2_fetch.Box(
    f"EPSG:{MAP_EPSG}", (10.0, 0.0, 500110.0, 0.0, -10.0, 3700220.0), 7 * P, 7 * P
)
SQUARE = (3 * P, 3 * P, P, P)
#: Window 14's square in a fake scene's own pixels (rows, columns).
ROWS = slice(111, 122)
COLS = slice(144, 155)
CLEAR, CLOUD = 4, 9  # SCL: vegetation; cloud of high probability

#: The caption, word for word (the spec's).
CAPTION = (
    "One date for each month of the map's period: the date with the least cloud "
    "and shadow on the square by Sentinel-2's own cloud mask and how white the "
    "square is (dates in UTC); dates marked cloudy are folded away. Near-infrared "
    "false colour shows vegetation red and water dark. Magenta: no data."
)
#: The rule before the round, word for word (kept as ``R5_PICK_RULE``).
OLD_PICK_RULE = (
    "For each calendar month of the map's period, the scene with the least scene "
    "cloud whose data footprint covers the picture and whose data covers at least "
    "90% of the outlined square, up to 3 scenes tried; the same rule for every "
    "window, blind to its map class, stratum and order."
)
#: The consent question before the round, word for word: its face's three
#: lines and its fold. The hash of these is the version a yes was given to.
ASK_BEFORE = "\n".join(
    (
        "Show satellite (Sentinel-2) pictures of each square? To fetch them, this "
        "computer sends the box around the map, rounded out to 0.1 degree, and the "
        "map's months.",
        "They go to Earth Search, a public catalogue, and its file store; both see "
        "your IP address.",
        "Nothing is sent to Studio or to Claude.",
    )
)
FULL_BEFORE = (
    "Earth Search's catalogue is earth-search.aws.element84.com: it gets the box "
    "and the months, once for all the squares. Its file store is "
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com: it gets the "
    "requests for parts of image files. No key and no account are used."
)
#: The view's words (sheet.js), word for word.
CLOUDY_WHY = (
    "Sentinel-2\\'s own cloud mask marks cloud or shadow on the ' + "
    "(unit === 'point' ? 'point' : 'square')\n"
    "  + ' on these dates, or it is white in the picture. Both are sometimes wrong "
    "(bright roofs, sand, surf, snow).';"
)
ALL_CLOUDY = (
    "Sentinel-2\\'s own cloud mask marks cloud or shadow on the ' + "
    "(unit === 'point' ? 'point' : 'square')\n"
    "  + ' on every date, or it is white in the picture.';"
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reaches the network; the caches start empty."""

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a test reached for the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    s2_fetch.clear_cache()
    imagery._maps.clear()
    imagery._test_cache.clear()


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root of a test, empty."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    return out


class Sky:
    """A fake Earth Search: a catalogue answering POST searches and a bucket
    answering Range requests, from GeoTIFFs written in ``folder``. A scene may
    have a cloud mask (its ``scl`` asset): one class everywhere, or an array."""

    def __init__(
        self,
        folder: Path,
        epsg: int = MAP_EPSG,
        origin: tuple[float, float] = SKY_ORIGIN,
    ) -> None:
        self.folder = folder
        self.epsg = epsg
        self.t = Affine(10, 0, origin[0], 0, -10, origin[1])
        self.size = (350, 300)
        self.files: dict[str, bytes] = {}
        self.items: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str, str]] = []

    def _tif(self, key: str, arr: Any, dtype: str) -> str:
        """Write one file of the bucket; its link."""
        w, h = self.size
        path = self.folder / key.strip("/").replace("/", "_")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype=dtype,
            crs=f"EPSG:{self.epsg}",
            transform=self.t,
            nodata=0,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            compress="deflate",
        ) as dst:
            dst.write(arr, 1)
        self.files[key] = path.read_bytes()
        return f"{BUCKET}{key}"

    def mask_of(self, value: int = CLEAR) -> Any:
        """A mask of one class everywhere, to change in places."""
        w, h = self.size
        return np.full((h, w), value, dtype=np.uint8)

    def add(
        self,
        ident: str,
        when: str,
        cloud: float,
        *,
        mask: Any = None,
        mask_link: str | None = None,
        hole: bool = False,
    ) -> None:
        """A scene: four bands (no data everywhere when ``hole``), and a cloud
        mask when ``mask`` is a class or an array; ``mask_link`` names a mask
        asset of its own (a file that is not there, or another host)."""
        w, h = self.size
        assets: dict[str, Any] = {}
        for (band, asset), value in zip(ASSETS.items(), (1500, 1400, 1300, 3000)):
            arr = np.full((h, w), 0 if hole else value, dtype=np.uint16)
            key = f"/sentinel-2-c1-l2a/{ident}/{asset}.tif"
            assets[band] = {"href": self._tif(key, arr, "uint16")}
        if mask is not None:
            arr = self.mask_of(mask) if isinstance(mask, int) else mask
            key = f"/sentinel-2-c1-l2a/{ident}/SCL.tif"
            assets["scl"] = {"href": self._tif(key, arr, "uint8")}
        if mask_link is not None:
            assets["scl"] = {"href": mask_link}
        x0, y0 = self.t * (0, 0)
        x1, y1 = self.t * (w, h)
        west, south, east, north = transform_bounds(
            f"EPSG:{self.epsg}", "EPSG:4326", x0, y1, x1, y0
        )
        ring = [
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]
        self.items.append(
            {
                "type": "Feature",
                "id": ident,
                "bbox": [west, south, east, north],
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "datetime": when,
                    "eo:cloud_cover": cloud,
                    "s2:datatake_id": "DT_" + ident,
                    "grid:code": "MGRS-11SLT",
                    "proj:epsg": self.epsg,
                    "s2:processing_baseline": "05.10",
                },
                "assets": assets,
            }
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        """The catalogue's and the bucket's answers."""
        host, path = request.url.host, request.url.path
        self.requests.append((request.method, host, path))
        if host == s2_fetch.CATALOGUE_HOST:
            body = json.loads(request.content)
            lo, hi = body["datetime"].split("/")
            found = sorted(
                (i for i in self.items if lo <= i["properties"]["datetime"] <= hi),
                key=lambda i: (i["properties"]["eo:cloud_cover"], i["id"]),
            )
            return httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": found,
                    "numberMatched": len(found),
                    "links": [],
                },
            )
        if host == s2_fetch.BUCKET_HOST:
            data = self.files.get(path)
            if data is None:
                return httpx.Response(404)
            a, z = (
                int(v)
                for v in request.headers["range"].removeprefix("bytes=").split("-")
            )
            z = min(z, len(data) - 1)
            return httpx.Response(
                206,
                content=data[a : z + 1],
                headers={"content-range": f"bytes {a}-{z}/{len(data)}"},
            )
        raise AssertionError(f"request to another host: {host}")

    def client(self) -> httpx.Client:
        """A client whose every request this fake answers."""
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), follow_redirects=False
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the app's picture calls use this fake."""
        monkeypatch.setattr(s2_fetch, "client_factory", self.client)

    def scenes(self) -> list[s2_fetch.Scene]:
        """Every scene added, as the search would give them."""
        found = [s2_fetch.scene_of(item) for item in self.items]
        return [s for s in found if s is not None]

    def searches(self) -> int:
        """How many searches the catalogue heard."""
        return sum(1 for _, host, _ in self.requests if host == s2_fetch.CATALOGUE_HOST)

    def mask_reads(self) -> list[str]:
        """The scenes whose mask file was asked for, in order, each once."""
        seen: list[str] = []
        for _, _, path in self.requests:
            if path.endswith("/SCL.tif") and path.split("/")[-2] not in seen:
                seen.append(path.split("/")[-2])
        return seen


@pytest.fixture
def sky(tmp_path: Path) -> Sky:
    """A fake Earth Search with no scene yet."""
    folder = tmp_path / "sky"
    folder.mkdir()
    return Sky(folder)


def _feature(**over: Any) -> dict[str, Any]:
    """A search feature with the four band links, and what ``over`` changes."""
    feature: dict[str, Any] = {
        "id": "S2A_X",
        "bbox": [10.0, 50.0, 11.0, 51.0],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[10, 50], [11, 50], [11, 51], [10, 51], [10, 50]]],
        },
        "properties": {
            "datetime": "2024-04-10T10:00:00Z",
            "eo:cloud_cover": 3.0,
            "s2:datatake_id": "D",
            "grid:code": "MGRS-32UNA",
            "proj:epsg": 32632,
        },
        "assets": {
            band: {"href": f"{BUCKET}/tiles/x/{asset}.tif"}
            for band, asset in ASSETS.items()
        },
    }
    feature.update(over)
    return feature


def _with_mask(href: Any) -> dict[str, Any]:
    """:func:`_feature` with a ``scl`` asset whose link is ``href``."""
    assets = {**_feature()["assets"], "scl": {"href": href}}
    return _feature(assets=assets)


def _stack(
    sky: Sky, scenes: list[s2_fetch.Scene], **kwargs: Any
) -> tuple[Any, s2_fetch.Fetch, s2_fetch.Fetch]:
    """``month_stack`` of ``scenes`` on window 14's picture: its result, the
    band fetch and the masks' fetch."""
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        masks = kwargs.pop("masks", None) or s2_fetch.mask_fetch(fetch)
        out = s2_fetch.month_stack(fetch, scenes, BOX, SQUARE, masks=masks, **kwargs)
    return out, fetch, masks


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _sheet(root: Path) -> Path:
    """A plan's sheet of two squares (windows 14 and 25) on a scores map of
    presence and absence, with its design, its scores file and its map raster;
    the map's period is April to June 2024. Each square's place is worked out
    from the grid, never written here."""
    folder = root / "run"
    folder.mkdir(exist_ok=True)
    raster = folder / "scores.tif"
    h, w = GRID[0] * P, GRID[1] * P
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=2,
        dtype="float32",
        crs=f"EPSG:{MAP_EPSG}",
        transform=MAP_T,
    ) as dst:
        dst.write(np.zeros((2, h, w), dtype=np.float32))
    scores = root / "scores_run_p11.json"
    scores.write_text(
        json.dumps(
            {
                "scores": [[0.2, 0.8]] * (GRID[0] * GRID[1]),
                "grid": list(GRID),
                "patch_px": P,
                "raster": {
                    "file": raster.name,
                    "path": str(raster),
                    "shape": [2, h, w],
                    "sha256": "abc",
                },
                "date_window": {
                    "start": "2024-04-01T00:00:00Z",
                    "end": "2024-06-30T23:59:59.999000Z",
                },
            }
        ),
        encoding="utf-8",
    )
    population = {
        "score_kind": "window_confidence",
        "class_names": {"0": "absent", "1": "present"},
        "source": {"scores_path": str(scores)},
    }
    (root / "kelp.json").write_text(
        json.dumps({"design": "random", "population": population}), encoding="utf-8"
    )
    rows = []
    for order, index in enumerate((14, 25), 1):
        r, c = divmod(index, GRID[1])
        x, y = MAP_T * (c * P + P / 2, r * P + P / 2)
        (lon,), (lat,) = warp_transform(f"EPSG:{MAP_EPSG}", "EPSG:4326", [x], [y])
        rows.append(
            [
                str(order),
                str(index),
                str(r),
                str(c),
                "0",
                f"{lon:.7f}",
                f"{lat:.7f}",
                "0",
                "",
                "",
            ]
        )
    path = root / "kelp_to_label.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ONE_MAP)
        writer.writerows(rows)
    return path


def _pictures(path: Path, index: int) -> dict[str, Any]:
    """window_image of the window ``index``, asked by its handle as the page asks."""
    sheet_id, handles = sheet.window_handles(str(path))
    handle = next(h for h, i in handles.items() if i == index)
    return imagery.window_image(str(path), handle, sheet_id=sheet_id, on=True)


def _record(path: Path) -> list[dict[str, Any]]:
    """The lines of the pictures' record beside a sheet."""
    text = imagery.record_path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


# ------------------------------------------------------------ the mask's link


def test_the_constants_of_the_rule() -> None:
    """The mask is the scene's ``scl`` asset; cloud shadow, cloud of medium
    and of high probability and thin cirrus are covered (rslearn's default),
    snow, dark features and unclassified are not; six masks a month at most;
    a mask counts with data on half of the square."""
    assert s2_fetch.MASK == "scl"
    assert s2_fetch.COVERED_CLASSES == (3, 8, 9, 10)
    assert s2_fetch.MASK_TRIES == 6 and s2_fetch.MASK_VALID == 0.5
    assert s2_fetch.TRIES_PER_MONTH == 3 and s2_fetch.BANDS == tuple(ASSETS)
    assert "four band files and of the\nscene's own cloud mask" in (
        s2_fetch.__doc__ or ""
    )


def test_a_scene_takes_its_mask_link_from_the_scl_asset() -> None:
    """scene_of: with ``scl`` on the bucket the scene has the link; without
    it, on another host, with a file that is no .tif or a link that is no
    text, the link is empty and the scene is kept."""
    link = f"{BUCKET}/tiles/x/SCL.tif"
    scene = s2_fetch.scene_of(_with_mask(link))
    assert scene is not None and scene.mask_href == link
    plain = s2_fetch.scene_of(_feature())
    assert plain is not None and plain.mask_href == ""
    for other in (
        "https://evil.example/tiles/x/SCL.tif",
        f"http://{s2_fetch.BUCKET_HOST}/tiles/x/SCL.tif",
        f"{BUCKET}/tiles/x/SCL.tif?x=1",
        f"{BUCKET}/tiles/x/SCL.jp2",
        None,
        7,
    ):
        kept = s2_fetch.scene_of(_with_mask(other))
        assert kept is not None and kept.mask_href == "", other
        assert kept.hrefs == plain.hrefs
    # An scl asset that is no object leaves the scene usable too.
    odd = s2_fetch.scene_of(_feature(assets={**_feature()["assets"], "scl": "x"}))
    assert odd is not None and odd.mask_href == ""


def test_the_record_keeps_the_mask_link_and_an_old_record_derives_it() -> None:
    """as_record writes ``mask``; from_record reads it back through the same
    checks. A record without ``mask`` (made before the round) derives the
    link from the red band's when that is ``B04.tif`` (the same folder's
    ``SCL.tif``); any other path leaves it empty."""
    link = f"{BUCKET}/tiles/x/SCL.tif"
    scene = s2_fetch.scene_of(_with_mask(link))
    assert scene is not None
    kept = scene.as_record()
    assert kept["mask"] == link
    assert s2_fetch.Scene.from_record(kept) == scene
    # New, with no mask: the empty link stays empty (nothing is derived).
    plain = s2_fetch.scene_of(_feature())
    assert plain is not None and plain.as_record()["mask"] == ""
    back = s2_fetch.Scene.from_record(plain.as_record())
    assert back is not None and back.mask_href == ""
    # New, with a link that is not on the bucket: empty, the scene kept.
    for bad in ("https://evil.example/SCL.tif", f"{BUCKET}/tiles/x/SCL.txt", 3, None):
        got = s2_fetch.Scene.from_record({**kept, "mask": bad})
        assert got is not None and got.mask_href == "", bad
    # Old: no "mask" key; the red band is B04.tif.
    old = {k: v for k, v in kept.items() if k != "mask"}
    derived = s2_fetch.Scene.from_record(old)
    assert derived is not None and derived.mask_href == link
    assert derived.hrefs == scene.hrefs
    # Old, with a red band of another name: no link.
    hrefs = {**old["hrefs"], "red": f"{BUCKET}/tiles/x/red.tif"}
    other = s2_fetch.Scene.from_record({**old, "hrefs": hrefs})
    assert other is not None and other.mask_href == ""


# ---------------------------------------------------------------------- cover


def test_cover_is_the_marked_share_of_the_pixels_with_data() -> None:
    """cover over the outlined square: all clear, all covered, half with no
    data (valid share 0.5, never read as clear), no data at all, and the
    4 x 4 square of a 40 m window."""
    square = (3, 2, 11, 11)
    mask = np.full((20, 20), CLEAR, dtype=np.uint8)
    assert s2_fetch.cover(mask, square) == (0.0, 1.0)
    # Cloud outside the square does not count.
    mask[:2, :] = CLOUD
    mask[:, :3] = CLOUD
    assert s2_fetch.cover(mask, square) == (0.0, 1.0)
    mask[:] = CLOUD
    assert s2_fetch.cover(mask, square) == (1.0, 1.0)
    # Each covered class counts; snow, dark features and unclassified do not.
    for value in (3, 8, 9, 10):
        assert s2_fetch.cover(np.full((20, 20), value, np.uint8), square)[0] == 1.0
    for value in (1, 2, 4, 5, 6, 7, 11):
        assert s2_fetch.cover(np.full((20, 20), value, np.uint8), square)[0] == 0.0
    # Half of the square has no data: the share is of the pixels with data.
    half = np.full((10, 10), CLEAR, dtype=np.uint8)
    half[:, :5] = 0
    assert s2_fetch.cover(half, (0, 0, 10, 10)) == (0.0, 0.5)
    half[:, 5:] = CLOUD
    assert s2_fetch.cover(half, (0, 0, 10, 10)) == (1.0, 0.5)
    half[:5, 5:] = CLEAR
    assert s2_fetch.cover(half, (0, 0, 10, 10)) == (0.5, 0.5)
    # No pixel with data: covered, and nothing valid.
    assert s2_fetch.cover(np.zeros((10, 10), np.uint8), (0, 0, 10, 10)) == (1.0, 0.0)
    assert s2_fetch.cover(np.zeros((10, 10), np.uint8), (0, 0, 0, 0)) == (1.0, 0.0)
    # A 40 m window: 4 x 4 pixels in the middle of a picture of 28.
    small = np.full((28, 28), CLEAR, dtype=np.uint8)
    assert s2_fetch.cover(small, (12, 12, 4, 4)) == (0.0, 1.0)
    small[12, 12] = 3  # one pixel of cloud shadow
    assert s2_fetch.cover(small, (12, 12, 4, 4)) == (1 / 16, 1.0)
    small[11, :] = CLOUD  # just outside it
    assert s2_fetch.cover(small, (12, 12, 4, 4)) == (1 / 16, 1.0)


def test_read_mask_puts_the_mask_on_the_pictures_grid(sky: Sky) -> None:
    """read_mask: a uint8 array of the picture's size, the mask's classes
    where the mask has them, 0 where it has no data; a scene with no mask
    link raises. read_box reads the four bands as before."""
    arr = sky.mask_of(CLEAR)
    arr[ROWS, 144:150] = CLOUD  # six of the square's eleven columns
    arr[:60, :] = 0  # no data far above the picture
    sky.add("A", "2024-04-10T18:40:00Z", 2.0, mask=arr)
    sky.add("B", "2024-04-11T18:40:00Z", 3.0)
    with_mask, without = sky.scenes()
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        mask = s2_fetch.read_mask(fetch, with_mask, BOX)
        assert mask.dtype == np.uint8 and mask.shape == (BOX.height, BOX.width)
        x, y, w, h = SQUARE
        part = mask[y : y + h, x : x + w]
        assert (part[:, :6] == CLOUD).all() and (part[:, 6:] == CLEAR).all()
        covered, valid = s2_fetch.cover(mask, SQUARE)
        assert covered == pytest.approx(6 / 11) and valid == 1.0
        with pytest.raises(s2_fetch.FetchError):
            s2_fetch.read_mask(fetch, without, BOX)
        stack = s2_fetch.read_box(fetch, without, BOX)
    assert stack.dtype == np.uint16 and stack.shape == (4, BOX.height, BOX.width)
    assert [int(band[0, 0]) for band in stack] == [1500, 1400, 1300, 3000]
    assert s2_fetch.has_data(stack, SQUARE)


# ---------------------------------------------------------------- month_stack


def test_a_scene_clear_on_the_square_beats_a_lower_scene_cloud(sky: Sky) -> None:
    """A scene of 2% scene cloud whose mask covers the square loses to one
    of 8% whose mask marks nothing on it."""
    sky.add("LOW", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("HIGH", "2024-04-20T18:40:00Z", 8.0, mask=CLEAR)
    (scene, stack, tried, info), fetch, masks = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "HIGH" and tried == 1
    assert s2_fetch.has_data(stack, SQUARE)
    assert info == {
        "here": "clear",
        "white_share": 0.0,
        "covered_share": 0.0,
        "valid_share": 1.0,
        "scored": [
            {"id": "LOW", "covered": 1.0, "valid": 1.0},
            {"id": "HIGH", "covered": 0.0, "valid": 1.0},
        ],
    }
    assert sky.mask_reads() == ["LOW", "HIGH"]
    assert fetch.failed is None and masks.failed is None
    # The band fetch asked for no mask file, the masks' fetch for no band.
    assert not any(href.endswith("/SCL.tif") for href, _, _ in fetch.log)
    assert masks.log and all(href.endswith("/SCL.tif") for href, _, _ in masks.log)


def test_when_every_scene_is_covered_the_least_covered_wins(sky: Sky) -> None:
    """Every candidate's mask marks the square: the least covered is picked,
    and the date is said to be covered."""
    part = sky.mask_of(CLEAR)
    part[ROWS, 144:147] = 8  # three of the square's eleven columns
    sky.add("FULL", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("PART", "2024-04-20T18:40:00Z", 8.0, mask=part)
    sky.add("SHADOW", "2024-04-25T18:40:00Z", 9.0, mask=3)
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "PART" and tried == 1
    assert info["here"] == "covered"
    assert info["covered_share"] == pytest.approx(3 / 11)
    assert info["valid_share"] == 1.0
    assert [s["id"] for s in info["scored"]] == ["FULL", "PART", "SHADOW"]
    assert sky.mask_reads() == ["FULL", "PART", "SHADOW"]


def test_the_masks_stop_at_the_first_clear_scene(sky: Sky) -> None:
    """The first scene whose mask marks nothing on the square wins: no
    further mask is read."""
    sky.add("A", "2024-04-10T18:40:00Z", 1.0, mask=CLEAR)
    sky.add("B", "2024-04-15T18:40:00Z", 5.0, mask=CLEAR)
    sky.add("C", "2024-04-20T18:40:00Z", 9.0, mask=CLOUD)
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "A" and tried == 1
    assert sky.mask_reads() == ["A"]
    assert info["scored"] == [{"id": "A", "covered": 0.0, "valid": 1.0}]
    assert info["here"] == "clear"


def test_at_most_six_masks_are_read_in_a_month(sky: Sky) -> None:
    """Of eight covered scenes the six of least scene cloud have their masks
    read; the bands of three scenes at most are read."""
    for k in range(8):
        sky.add(f"S{k}", f"2024-04-{10 + k:02d}T18:40:00Z", float(k), mask=CLOUD)
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert sky.mask_reads() == [f"S{k}" for k in range(s2_fetch.MASK_TRIES)]
    # All equally covered: the candidates' order decides, and the first has data.
    assert scene is not None and scene.id == "S0" and tried == 1
    assert info["here"] == "covered" and info["covered_share"] == 1.0


def test_a_mask_that_cannot_be_read_leaves_the_scene_unscored(sky: Sky) -> None:
    """A mask read that fails (the file is not there): the scene is ranked as
    one with no mask, in today's order; the band reads are untouched and the
    band fetch has not failed."""
    sky.add(
        "LOW",
        "2024-04-10T18:40:00Z",
        2.0,
        mask_link=f"{BUCKET}/sentinel-2-c1-l2a/LOW/SCL.tif",
    )
    sky.add("HIGH", "2024-04-20T18:40:00Z", 8.0)
    (scene, stack, tried, info), fetch, masks = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "LOW" and tried == 1
    assert s2_fetch.has_data(stack, SQUARE)
    assert info == {
        "here": "unknown",
        "white_share": 0.0,
        "covered_share": None,
        "valid_share": None,
        "scored": [],
    }
    assert sky.mask_reads() == ["LOW"]
    assert isinstance(masks.failed, s2_fetch.FetchError)
    assert fetch.failed is None


def test_one_failed_mask_does_not_stop_the_next(sky: Sky) -> None:
    """Each mask is read on its own: after one that fails, the next scene's
    mask is still read and scored."""
    sky.add(
        "LOW",
        "2024-04-10T18:40:00Z",
        2.0,
        mask_link=f"{BUCKET}/sentinel-2-c1-l2a/LOW/SCL.tif",
    )
    sky.add("MID", "2024-04-15T18:40:00Z", 5.0, mask=CLOUD)
    sky.add("HIGH", "2024-04-20T18:40:00Z", 8.0, mask=CLEAR)
    (scene, _, tried, info), fetch, masks = _stack(sky, sky.scenes())
    # Clear first, then the unscored, then the covered.
    assert scene is not None and scene.id == "HIGH" and tried == 1
    assert [s["id"] for s in info["scored"]] == ["MID", "HIGH"]
    assert info["here"] == "clear"
    assert masks.failed is not None and fetch.failed is None


def test_an_unscored_scene_ranks_before_a_covered_one(sky: Sky) -> None:
    """Clear scenes first, then the scenes with no mask read in today's
    order, then the covered ones: a covered scene of 2% loses to one of 8%
    with no mask. The masks stop at the first clear scene; when that scene
    has no data in the square, mask reading resumes before the next band
    read, so the covered scene after it is scored and the scene with no
    mask still ranks before it."""
    sky.add("COVERED", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("NOMASK", "2024-04-20T18:40:00Z", 8.0)
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "NOMASK" and tried == 1
    assert info["here"] == "unknown" and info["covered_share"] is None
    assert info["scored"] == [{"id": "COVERED", "covered": 1.0, "valid": 1.0}]
    # A clear mask over bands with no data: the masks go on, then the rank.
    sky.add("EMPTY", "2024-04-05T18:40:00Z", 1.0, mask=CLEAR, hole=True)
    s2_fetch.clear_cache()
    sky.requests.clear()
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "NOMASK" and tried == 2
    assert info["here"] == "unknown" and sky.mask_reads() == ["EMPTY", "COVERED"]
    assert info["scored"] == [
        {"id": "EMPTY", "covered": 0.0, "valid": 1.0},
        {"id": "COVERED", "covered": 1.0, "valid": 1.0},
    ]


def test_a_scene_read_after_the_clear_one_is_never_left_unscored(sky: Sky) -> None:
    """The clear scene has no data in the square and the only other scene is
    covered: its mask is read before its bands, so the date says "covered"
    (and is folded), never "unknown"."""
    sky.add("EMPTY", "2024-04-05T18:40:00Z", 1.0, mask=CLEAR, hole=True)
    sky.add("COVERED", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    (scene, _, tried, info), fetch, masks = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "COVERED" and tried == 2
    assert info["here"] == "covered" and info["covered_share"] == 1.0
    assert sky.mask_reads() == ["EMPTY", "COVERED"]
    assert fetch.failed is None and masks.failed is None
    # The masks of a month are still six at most, however the bands go.
    for k in range(8):
        sky.add(f"S{k}", f"2024-04-{12 + k:02d}T18:40:00Z", 3.0 + k, mask=CLOUD)
    s2_fetch.clear_cache()
    sky.requests.clear()
    (scene, _, tried, info), _, _ = _stack(sky, sky.scenes())
    assert len(sky.mask_reads()) == s2_fetch.MASK_TRIES
    assert scene is not None and scene.id == "COVERED" and tried == 2


def test_a_mask_with_little_data_on_the_square_does_not_count_as_clear(
    sky: Sky,
) -> None:
    """A mask with data on under half of the square and nothing marked is
    not clear: a scene with no mask ranks before it, and picked alone the
    record says it is covered (its line: no data on most of the square). The
    pane does not call its date cloudy, since the mask marks nothing: the
    date is "unknown" there, shown and never folded, and counts in no total.
    The same for a mask with no data on the square at all."""
    thin = sky.mask_of(0)
    thin[ROWS, 144:148] = CLEAR  # four of the square's eleven columns
    sky.add("THIN", "2024-04-10T18:40:00Z", 2.0, mask=thin)
    (scene, stack, _, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "THIN"
    assert info["here"] == "covered" and info["covered_share"] == 0.0
    assert info["valid_share"] == pytest.approx(4 / 11)
    hole = "cloud mask: no data on most of the square"
    assert imagery.mask_line(info) == hole
    months = s2_fetch.months(
        s2_fetch.parse_moment("2024-04-01", end=False),
        s2_fetch.parse_moment("2024-04-30", end=True),
    )
    none = {**info, "covered_share": 1.0, "valid_share": 0.0}
    marked = {**info, "covered_share": 0.5, "valid_share": 0.2}
    for mask, here in ((info, "unknown"), (none, "unknown"), (marked, "covered")):
        dates, served = imagery._dates(months, [(scene, stack, 1, mask)], SQUARE)
        assert dates[0]["cloud_here"] == here and dates[0]["mask_line"] == hole
        assert dates[0]["label"] == "10 Apr 2024, scene cloud 2%"
        assert len(dates[0]["pictures"]) == 4
        # The record keeps what the masks said, unchanged.
        assert served[0]["mask"] == mask and served[0]["mask"]["here"] == "covered"
        cloudy = here == "covered"
        assert imagery.cloud_totals(dates) == {
            "cloudy_dates": int(cloudy),
            "all_cloudy": cloudy,
        }
    assert imagery._marked(marked) and not imagery._marked(info)
    assert not imagery._marked(none) and not imagery._marked({"here": "clear"})
    sky.add("NOMASK", "2024-04-20T18:40:00Z", 8.0)
    s2_fetch.clear_cache()
    (scene, _, _, info), _, _ = _stack(sky, sky.scenes())
    assert scene is not None and scene.id == "NOMASK" and info["here"] == "unknown"


def test_with_no_mask_anywhere_the_pick_is_as_before(sky: Sky) -> None:
    """Scenes with no mask (the fixtures of before the round): the same scene
    and the same count of scenes tried as the least-scene-cloud rule gives,
    and no mask file is asked for."""
    sky.add("HOLE", "2024-04-05T18:40:00Z", 1.0, hole=True)
    sky.add("LOW", "2024-04-10T18:40:00Z", 2.0)
    sky.add("HIGH", "2024-04-20T18:40:00Z", 40.0)
    scenes = sky.scenes()
    # The rule before the round, written out.
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        want, count = None, 0
        for scene in s2_fetch.candidates(scenes, BOX.lonlat(), BOX.points())[:3]:
            count += 1
            if s2_fetch.has_data(s2_fetch.read_box(fetch, scene, BOX), SQUARE):
                want = scene
                break
    assert want is not None and (want.id, count) == ("LOW", 2)
    s2_fetch.clear_cache()
    (scene, _, tried, info), fetch, masks = _stack(sky, scenes)
    assert (scene, tried) == (want, count)
    assert info == {
        "here": "unknown",
        "white_share": 0.0,
        "covered_share": None,
        "valid_share": None,
        "scored": [],
    }
    assert sky.mask_reads() == [] and masks.log == [] and masks.failed is None
    # No scene with data in the square: none, after three tries at most.
    only = [s for s in scenes if s.id == "HOLE"]
    (scene, stack, tried, info), _, _ = _stack(sky, only)
    assert (scene, stack, tried) == (None, None, 1) and info["here"] == "unknown"
    (scene, stack, tried, info), _, _ = _stack(sky, [])
    assert (scene, stack, tried, info["scored"]) == (None, None, 0, [])


def test_a_spent_mask_budget_reads_no_mask_and_fails_nothing(sky: Sky) -> None:
    """A mask read that would start after the masks' budget is spent is not
    made: the scenes are unscored, today's order holds, and the masks'
    fetch never sets the band fetch's failure."""
    sky.add("LOW", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("HIGH", "2024-04-20T18:40:00Z", 8.0, mask=CLEAR)
    with sky.client() as client:
        spent = s2_fetch.Fetch(client, s2_fetch.Budget(max_requests=0))
        (scene, _, tried, info), fetch, masks = _stack(sky, sky.scenes(), masks=spent)
    assert masks is spent and sky.mask_reads() == []
    assert scene is not None and scene.id == "LOW" and tried == 1
    assert info["here"] == "unknown" and info["scored"] == []
    assert fetch.failed is None
    # A budget that runs out during a read: that mask fails, the bands do not.
    s2_fetch.clear_cache()
    with sky.client() as client:
        tight = s2_fetch.Fetch(client, s2_fetch.Budget(max_bytes=1))
        (scene, stack, _, info), fetch, masks = _stack(sky, sky.scenes(), masks=tight)
    assert scene is not None and scene.id == "LOW" and info["scored"] == []
    assert isinstance(masks.failed, s2_fetch.FetchError) and fetch.failed is None
    assert s2_fetch.has_data(stack, SQUARE)


def test_the_masks_have_a_fetch_and_a_budget_of_their_own(
    sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mask_fetch: the band fetch's client, its own budget (15 seconds, 32
    MiB, 250 requests: 12 months of 6 masks, up to 3 requests each). dated
    makes one for all its months and gives each month's result as a 4-tuple,
    in month order."""
    assert s2_fetch.MASK_SECONDS == 15.0 and s2_fetch.MASK_WORKERS == 8
    assert imagery.MAX_MONTHS * s2_fetch.MASK_TRIES * 3 <= 250
    sky.add("APR", "2024-04-10T18:40:00Z", 2.0, mask=CLEAR)
    sky.add("MAY", "2024-05-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("JUN", "2024-06-10T18:40:00Z", 2.0)
    made: list[s2_fetch.Fetch] = []
    real = s2_fetch.mask_fetch

    def counted(fetch: s2_fetch.Fetch) -> s2_fetch.Fetch:
        made.append(real(fetch))
        return made[-1]

    monkeypatch.setattr(s2_fetch, "mask_fetch", counted)
    months = [[scene] for scene in sky.scenes()]
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        results = s2_fetch.dated(fetch, months, BOX, SQUARE)
        assert s2_fetch.dated(fetch, [], BOX, SQUARE) == []
    assert len(made) == 1
    masks = made[0]
    assert masks is not fetch and masks.client is fetch.client
    assert masks.budget is not fetch.budget
    assert (masks.budget.max_bytes, masks.budget.max_requests) == (32 << 20, 250)
    assert 0 < masks.budget.deadline - time.monotonic() <= 15.0
    assert [len(result) for result in results] == [4, 4, 4]
    assert [r[0].id for r in results] == ["APR", "MAY", "JUN"]
    assert [r[3]["here"] for r in results] == ["clear", "covered", "unknown"]
    # A masks' fetch that is given is the one used.
    s2_fetch.clear_cache()
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        given = s2_fetch.Fetch(client, s2_fetch.Budget(max_requests=0))
        results = s2_fetch.dated(fetch, months, BOX, SQUARE, masks=given)
    assert len(made) == 1
    assert [r[3]["here"] for r in results] == ["unknown"] * 3


def test_every_months_masks_are_read_before_any_band(sky: Sky) -> None:
    """dated reads the masks of all the months first, then the bands: no
    month loses its masks because its bands start late. With more months
    than band workers, every mask request comes before the first band's."""
    months = []
    for k in range(1, 9):
        sky.add(f"M{k}A", f"2024-{k:02d}-10T18:40:00Z", 2.0, mask=CLOUD)
        sky.add(f"M{k}B", f"2024-{k:02d}-20T18:40:00Z", 8.0, mask=CLEAR)
        months.append([s for s in sky.scenes() if s.id.startswith(f"M{k}")])
    for workers in (1, 4):
        s2_fetch.clear_cache()
        sky.requests.clear()
        with sky.client() as client:
            fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
            results = s2_fetch.dated(fetch, months, BOX, SQUARE, workers=workers)
        assert [r[0].id for r in results] == [f"M{k}B" for k in range(1, 9)]
        assert [r[3]["here"] for r in results] == ["clear"] * 8
        assert [len(r[3]["scored"]) for r in results] == [2] * 8
        is_mask = [path.endswith("/SCL.tif") for _, _, path in sky.requests]
        assert is_mask.count(True) and is_mask.count(False)
        assert is_mask.index(False) > max(i for i, m in enumerate(is_mask) if m)


def test_the_time_the_masks_take_is_given_back_to_the_bands(
    sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow masks never make the bands late: the time the masks' pass took
    (here 14 seconds of a clock the test moves) is added to the bands'
    deadline, 15 seconds at most, so band reads that then take 30 seconds
    still end within their own 40; without that they would be too slow. A
    spent masks' budget takes no time and adds none."""
    sky.add("APR", "2024-04-10T18:40:00Z", 2.0, mask=CLEAR)
    clock = [1000.0]
    monkeypatch.setattr(
        s2_fetch, "time", types.SimpleNamespace(monotonic=lambda: clock[0])
    )
    seen: set[bool] = set()

    def slow(request: httpx.Request) -> httpx.Response:
        """The fake's answer; the first mask request takes 14 seconds of the
        clock and the first band request 30."""
        mask = request.url.path.endswith("/SCL.tif")
        if request.url.host == s2_fetch.BUCKET_HOST and mask not in seen:
            seen.add(mask)
            clock[0] += 14.0 if mask else 30.0
        return sky.handler(request)

    def run(masks_budget: s2_fetch.Budget | None = None) -> s2_fetch.Fetch:
        """dated on the slow fake, the clock set back: the band fetch."""
        s2_fetch.clear_cache()
        seen.clear()
        clock[0] = 1000.0
        transport = httpx.MockTransport(slow)
        with httpx.Client(transport=transport, follow_redirects=False) as client:
            fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
            given = masks_budget
            masks = None if given is None else s2_fetch.Fetch(client, given)
            results = s2_fetch.dated(fetch, [sky.scenes()], BOX, SQUARE, masks=masks)
            assert results[0][0] is not None and results[0][0].id == "APR"
        return fetch

    fetch = run()
    assert fetch.failed is None
    assert fetch.budget.deadline == pytest.approx(1000.0 + 40.0 + 14.0)
    # No mask read: nothing is added, and 30 seconds of bands are in time.
    fetch = run(s2_fetch.Budget(max_requests=0))
    assert fetch.budget.deadline == pytest.approx(1040.0)
    # More than the masks' own time is never added.
    monkeypatch.setattr(s2_fetch, "MASK_SECONDS", 3.0)
    with pytest.raises(s2_fetch.FetchError) as too_slow:
        run(s2_fetch.Budget(seconds=60.0))
    assert too_slow.value.kind == "deadline"
    monkeypatch.setattr(s2_fetch, "MASK_SECONDS", 15.0)
    # Without the time given back, the same reads would be too slow.
    monkeypatch.setattr(s2_fetch.Budget, "extend", lambda self, seconds: None)
    with pytest.raises(s2_fetch.FetchError) as late:
        run()
    assert late.value.kind == "deadline"


def test_a_failed_band_read_stops_the_mask_reads_too(sky: Sky) -> None:
    """When the band reads fail the call fails with the bands' error, and
    the masks' budget is ended, so no mask read still running holds the call:
    none can start a further request."""
    sky.add("APR", "2024-04-10T18:40:00Z", 2.0, mask=CLEAR)
    sky.add("MAY", "2024-05-10T18:40:00Z", 2.0, mask=CLEAR)
    del sky.files["/sentinel-2-c1-l2a/MAY/B04.tif"]
    months = [[scene] for scene in sky.scenes()]
    with sky.client() as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        masks = s2_fetch.mask_fetch(fetch)
        assert s2_fetch._room(masks.budget)
        with pytest.raises(s2_fetch.FetchError) as failed:
            s2_fetch.dated(fetch, months, BOX, SQUARE, masks=masks)
    assert failed.value.kind == "fetch"
    assert isinstance(fetch.failed, s2_fetch.FetchError)
    assert fetch.failed.kind == "fetch"
    assert not s2_fetch._room(masks.budget)
    with pytest.raises(s2_fetch.FetchError):
        masks.budget.timeout()
    # A budget is only ever moved later by extend, and stop never moves it later.
    budget = s2_fetch.Budget(seconds=40.0)
    before = budget.deadline
    budget.extend(-3.0)
    assert budget.deadline == before
    budget.extend(2.5)
    assert budget.deadline == pytest.approx(before + 2.5)
    budget.stop()
    assert budget.deadline <= time.monotonic()


def test_the_pick_is_blind_to_class_stratum_answer_and_order() -> None:
    """The functions that pick a date take the fetch, the scenes, the picture
    and the square only: no parameter names a class, a stratum, an answer or
    an order, and their words say so."""
    names: dict[Any, list[str]] = {
        s2_fetch.month_stack: ["fetch", "scenes", "box", "square", "tries", "masks"],
        s2_fetch.dated: [
            "fetch",
            "scenes_by_month",
            "box",
            "square",
            "workers",
            "masks",
        ],
        s2_fetch.read_mask: ["fetch", "scene", "box"],
        s2_fetch.cover: ["mask", "square"],
        s2_fetch.candidates: ["scenes", "box", "points"],
    }
    for function, want in names.items():
        got = list(inspect.signature(function).parameters)
        assert got == want, function.__name__
        for name in got:
            for word in ("class", "stratum", "strata", "answer", "order", "label"):
                assert word not in name, (function.__name__, name)
    assert "Blind to the window's\n    class, stratum and order." in (
        s2_fetch.month_stack.__doc__ or ""
    )
    assert "blind to its map class, stratum and order." in imagery.PICK_RULE


# ------------------------------------------------------- the pane and record


def test_mask_line_says_what_the_mask_marks_on_the_square() -> None:
    """The Details fold's words for a date: none on the square, cloud or
    shadow on a rounded share of it ("under 1%" below one), no data on most
    of the square, none for the date."""

    def line(here: str, covered: float | None, valid: float | None = 1.0) -> str:
        return imagery.mask_line(
            {"here": here, "covered_share": covered, "valid_share": valid}
        )

    assert line("clear", 0.0) == "cloud mask: none on the square"
    assert line("covered", 1.0) == "cloud mask: cloud or shadow on 100% of the square"
    assert line("covered", 0.336) == "cloud mask: cloud or shadow on 34% of the square"
    assert line("covered", 0.01) == "cloud mask: cloud or shadow on 1% of the square"
    under = "cloud mask: cloud or shadow on under 1% of the square"
    assert line("covered", 0.004) == under
    hole = "cloud mask: no data on most of the square"
    assert line("covered", 0.0, 0.3) == hole
    # No data at all on the square, or on under half of it: no share is said
    # (it would be a share of the pixels with data, not of the square).
    assert line("covered", 1.0, 0.0) == hole
    assert line("covered", 0.5, 0.2) == hole
    half = "cloud mask: cloud or shadow on 50% of the square"
    assert line("covered", 0.5, 0.5) == half
    # A record with no valid share reads as before.
    whole = "cloud mask: cloud or shadow on 100% of the square"
    assert line("covered", 1.0, None) == whole
    assert line("covered", 0.0, None) == hole
    assert line("unknown", None, None) == "cloud mask: none for this date"
    assert imagery.mask_line({}) == "cloud mask: none for this date"
    for name in ("MASK_CLEAR", "MASK_COVERED", "MASK_HOLE", "MASK_NONE"):
        assert f"# WORDING: r5p\n{name} = " in inspect.getsource(imagery), name


def test_cloud_totals_count_the_covered_dates() -> None:
    """cloudy_dates: the dates the mask marks; all_cloudy: every date with a
    scene is marked, and there is one. A month with no scene has no
    cloud_here and counts in neither."""
    covered = {"cloud_here": "covered"}
    clear = {"cloud_here": "clear"}
    unknown = {"cloud_here": "unknown"}
    empty = {"month": "Apr 2024", "label": "Apr 2024: no scene with data here"}
    totals = imagery.cloud_totals
    assert totals([]) == {"cloudy_dates": 0, "all_cloudy": False}
    assert totals([empty, empty]) == {"cloudy_dates": 0, "all_cloudy": False}
    assert totals([covered, clear, unknown]) == {
        "cloudy_dates": 1,
        "all_cloudy": False,
    }
    assert totals([covered, covered]) == {"cloudy_dates": 2, "all_cloudy": True}
    assert totals([covered, empty]) == {"cloudy_dates": 1, "all_cloudy": True}
    assert totals([covered, unknown]) == {"cloudy_dates": 1, "all_cloudy": False}


def test_a_month_with_no_scene_has_no_cloud_fields() -> None:
    """_dates: the "no scene with data here" date has neither cloud_here nor
    mask_line; the record's month keeps what the masks said."""
    months = s2_fetch.months(
        s2_fetch.parse_moment("2024-04-01", end=False),
        s2_fetch.parse_moment("2024-04-30", end=True),
    )
    info = {"here": "unknown", "covered_share": None, "valid_share": None, "scored": []}
    dates, served = imagery._dates(months, [(None, None, 2, info)], SQUARE)
    assert dates == [
        {
            "month": "Apr 2024",
            "label": "Apr 2024: no scene with data here",
            "pictures": [],
        }
    ]
    assert served == [{"month": "Apr 2024", "scene": None, "tried": 2, "mask": info}]
    assert imagery._no_image([(None, None, 2, info)]) is True
    assert imagery._no_image([(None, None, 2, info), (object(), 1, 1, info)]) is False


def test_a_sheets_pictures_say_which_dates_the_mask_marks(
    root: Path, sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A square's pictures: each date says what the mask marks on the square
    (cloud_here, mask_line), its label is word for word as before, the answer
    counts the cloudy dates, and the record keeps the pick's name, the mask's
    link and what the masks said. A second square reads the search from the
    record and gets the masks again, with no second search."""
    sky.add("APR_LOW", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("APR_HIGH", "2024-04-20T18:40:00Z", 8.0, mask=CLEAR)
    sky.add("MAY", "2024-05-12T18:40:00Z", 5.0, mask=CLOUD)
    sky.add("JUN", "2024-06-03T18:40:00Z", 0.5)
    sky.install(monkeypatch)
    path = _sheet(root)
    out = _pictures(path, 14)
    assert out["ok"] is True and "no_image" not in out, out
    assert out["caption"] == (
        "A square is 11 x 11 map pixels, about 110 m a side. " + CAPTION
    )
    dates = out["dates"]
    assert [d["month"] for d in dates] == ["Apr 2024", "May 2024", "Jun 2024"]
    # The label string is unchanged: its day, and its scene cloud.
    assert [d["label"] for d in dates] == [
        "20 Apr 2024, scene cloud 8%",
        "12 May 2024, scene cloud 5%",
        "3 Jun 2024, scene cloud under 1%",
    ]
    assert [d["cloud_here"] for d in dates] == ["clear", "covered", "unknown"]
    assert [d["mask_line"] for d in dates] == [
        "cloud mask: none on the square",
        "cloud mask: cloud or shadow on 100% of the square",
        "cloud mask: none for this date",
    ]
    assert all(len(d["pictures"]) == 4 for d in dates)
    assert (out["cloudy_dates"], out["all_cloudy"]) == (1, False)
    lines = _record(path)
    assert [next(iter(line)) for line in lines] == ["context", "search", "served"]
    context = lines[0]["context"]
    assert context["pick"] == "scl-cover@2" == imagery.PICK
    assert context["pick_rule"] == imagery.PICK_RULE
    assert context["consent_version"] == imagery.consent_version()
    april = lines[1]["search"]["months"][0]["scenes"]
    assert {s["id"]: s["mask"] for s in april} == {
        "APR_LOW": f"{BUCKET}/sentinel-2-c1-l2a/APR_LOW/SCL.tif",
        "APR_HIGH": f"{BUCKET}/sentinel-2-c1-l2a/APR_HIGH/SCL.tif",
    }
    served = lines[2]["served"]["months"]
    assert [m["id"] for m in served] == ["APR_HIGH", "MAY", "JUN"]
    assert [m["mask"]["here"] for m in served] == ["clear", "covered", "unknown"]
    assert served[0]["mask"]["scored"] == [
        {"id": "APR_LOW", "covered": 1.0, "valid": 1.0},
        {"id": "APR_HIGH", "covered": 0.0, "valid": 1.0},
    ]
    assert served[0]["mask_link"] == f"{BUCKET}/sentinel-2-c1-l2a/APR_HIGH/SCL.tif"
    assert served[2]["mask_link"] == "" and served[2]["mask"]["scored"] == []
    assert "src" not in imagery.record_path(path).read_text()  # no picture kept
    # The second square: the scenes come back from the record with their masks.
    assert sky.searches() == 3
    again = _pictures(path, 25)
    assert again["ok"] is True and sky.searches() == 3
    assert [d["cloud_here"] for d in again["dates"]] == ["clear", "covered", "unknown"]
    assert [d["label"] for d in again["dates"]] == [d["label"] for d in dates]


def test_a_search_recorded_before_the_round_still_gets_the_masks(
    root: Path, sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet in progress: its record's search has no mask link (made before
    the round). The next square derives each mask's link from the red band's
    and is filtered, with no second search."""
    sky.add("APR_LOW", "2024-04-10T18:40:00Z", 2.0, mask=CLOUD)
    sky.add("APR_HIGH", "2024-04-20T18:40:00Z", 8.0, mask=CLEAR)
    sky.add("MAY", "2024-05-12T18:40:00Z", 5.0, mask=CLEAR)
    sky.add("JUN", "2024-06-03T18:40:00Z", 0.5, mask=CLEAR)
    sky.install(monkeypatch)
    path = _sheet(root)
    assert _pictures(path, 14)["ok"] is True
    # The record as a build before the round wrote it: no "mask" in a scene.
    record = imagery.record_path(path)
    lines = _record(path)
    for month in lines[1]["search"]["months"]:
        for scene in month["scenes"]:
            del scene["mask"]
    record.write_text(
        "".join(json.dumps(line) + "\n" for line in lines[:2]), encoding="utf-8"
    )
    s2_fetch.clear_cache()
    sky.requests.clear()
    out = _pictures(path, 25)
    assert out["ok"] is True and sky.searches() == 0
    assert out["dates"][0]["label"] == "20 Apr 2024, scene cloud 8%"
    assert [d["cloud_here"] for d in out["dates"]] == ["clear", "clear", "clear"]
    assert (out["cloudy_dates"], out["all_cloudy"]) == (0, False)


def test_the_test_scene_and_the_review_set_carry_the_same_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The public test scene's answer has the dates' cloud fields and the two
    totals; with every date marked, all_cloudy is true and no date is left
    out. The review set's answer is built the same way."""
    (x,), (y,) = warp_transform(
        "EPSG:4326",
        f"EPSG:{imagery.TEST_EPSG}",
        [imagery.TEST_PLACE[0]],
        [imagery.TEST_PLACE[1]],
    )
    folder = tmp_path / "cape"
    folder.mkdir()
    fake = Sky(
        folder, epsg=imagery.TEST_EPSG, origin=(round(x) - 1500, round(y) + 1500)
    )
    for month in ("01", "02", "03"):
        fake.add(f"S2A_{month}", f"2024-{month}-15T08:49:00Z", 1.0, mask=CLOUD)
    fake.install(monkeypatch)
    out = imagery.public_test_scene()
    assert out["ok"] is True and out["test_scene"] is True, out
    assert out["caption"] == imagery.TEST_CAPTION
    assert [d["cloud_here"] for d in out["dates"]] == ["covered"] * 3
    assert all(len(d["pictures"]) == 4 for d in out["dates"])
    assert (out["cloudy_dates"], out["all_cloudy"]) == (3, True)
    review = imagery.public_test_scene(review=True)
    assert review["caption"] == imagery.REVIEW_TEST_CAPTION
    assert (review["cloudy_dates"], review["all_cloudy"]) == (3, True)
    for function in (
        imagery._window_image,
        imagery._review_image,
        imagery._public_test_scene,
    ):
        source = inspect.getsource(function)
        assert '        "dates": dates,\n        **cloud_totals(dates),\n' in source
        assert "s2_fetch.dated(" in source


def _review_key(root: Path) -> str:
    """A review set ranked from the sheet's scores file (one centre per
    window, worked out from the grid), with window 14 listed first: its key."""
    _sheet(root)
    scores = root / "scores_run_p11.json"
    payload = json.loads(scores.read_text(encoding="utf-8"))
    xs, ys = [], []
    for index in range(GRID[0] * GRID[1]):
        r, c = divmod(index, GRID[1])
        x, y = MAP_T * (c * P + P / 2, r * P + P / 2)
        xs.append(x)
        ys.append(y)
    lons, lats = warp_transform(f"EPSG:{MAP_EPSG}", "EPSG:4326", xs, ys)
    payload["centres_lon_lat"] = [list(pair) for pair in zip(lons, lats)]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    listed = {
        "rank": 1,
        "window_index": 14,
        "row": 1,
        "col": 4,
        "margin": 0.1,
        "predicted_class": 1,
        "class_name": "present",
    }
    block = review.review_view(
        "olmoearth_review_set",
        {"n_review": 1, "review": [listed]},
        {"scores_path": str(scores)},
    )
    assert block is not None and block["pictures"] is True
    return str(block["key"])


def test_a_review_sets_pictures_say_which_dates_the_mask_marks(
    root: Path, sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review set's window, read through the real path: each date says
    what the mask marks on the square, the answer counts the cloudy dates as
    a sheet's does, the labels are word for word as before, and the review's
    record keeps what the masks said for each month."""
    sky.add("APR", "2024-04-10T18:40:00Z", 2.0, mask=CLEAR)
    sky.add("MAY", "2024-05-12T18:40:00Z", 5.0, mask=CLOUD)
    sky.add("JUN", "2024-06-03T18:40:00Z", 0.5)
    sky.install(monkeypatch)
    key = _review_key(root)
    out = imagery.review_image(key, 1, on=True)
    assert out["ok"] is True and "no_image" not in out, out
    dates = out["dates"]
    assert [d["label"] for d in dates] == [
        "10 Apr 2024, scene cloud 2%",
        "12 May 2024, scene cloud 5%",
        "3 Jun 2024, scene cloud under 1%",
    ]
    assert [d["cloud_here"] for d in dates] == ["clear", "covered", "unknown"]
    assert [d["mask_line"] for d in dates] == [
        "cloud mask: none on the square",
        "cloud mask: cloud or shadow on 100% of the square",
        "cloud mask: none for this date",
    ]
    assert all(len(d["pictures"]) == 4 for d in dates)
    assert (out["cloudy_dates"], out["all_cloudy"]) == (1, False)
    real, _, index = review.review_row(key, 1)
    assert index == 14
    text = imagery.record_path(real).read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    served = [line["served"] for line in lines if "served" in line]
    assert len(served) == 1 and served[0]["rank"] == 1
    months = served[0]["months"]
    assert [m["id"] for m in months] == ["APR", "MAY", "JUN"]
    assert [m["mask"]["here"] for m in months] == ["clear", "covered", "unknown"]
    assert months[1]["mask"]["covered_share"] == 1.0
    assert "src" not in text  # no picture kept


# ------------------------------------------------------------------ the words


def test_the_caption_and_the_pick_rule_say_the_new_rule() -> None:
    """CAPTION word for word (the test scenes' captions end with it);
    PICK_RULE names the mask, its classes and the fold; the old rule is kept
    word for word for old records; a record's context names the pick."""
    assert imagery.CAPTION == CAPTION
    assert imagery.TEST_CAPTION == (
        "A public test scene, not a square of this sheet: nothing here is labelled. "
        "The outlined square is 11 pixels. " + CAPTION
    )
    assert imagery.REVIEW_TEST_CAPTION.endswith("differ in size. " + CAPTION)
    rule = imagery.PICK_RULE
    for text in (
        "the scene with the least cloud and shadow on the outlined square",
        "Scene Classification Layer",
        "classes 3/8/9/10",
        "up to 6 masks read and up to 3 scenes' bands read, least scene cloud first; "
        "when a scene its mask shows clear has too little data on the square, mask "
        "reading resumes.",
        "at least 90% of the outlined square",
        "A date the mask marks cloudy (any marked pixel on the square) is folded "
        "away in the pane, not dropped.",
        "A mask with data on less than 50% of the square does not count as clear.",
        "A date without a mask is ranked by scene cloud as before.",
        "blind to its map class, stratum and order.",
    ):
        assert text in rule, text
    assert imagery.R5_PICK_RULE == OLD_PICK_RULE
    assert imagery.PICK == "scl-cover@2"
    source = inspect.getsource(imagery)
    assert "# WORDING: r5p\nPICK_RULE = (" in source
    assert "# WORDING: r5p\nCAPTION = (" in source
    assert source.count('            "pick": PICK,\n') == 2
    for name in ("s2_fetch.py", "imagery.py", "server.py"):
        text = (Path(imagery.__file__).parent / name).read_text(encoding="utf-8")
        assert "the least scene cloud that has data here" not in text, name


def test_the_dates_label_is_word_for_word_as_before() -> None:
    """The label string "D Mon YYYY, scene cloud N%" does not change (the
    pane's heading splits it): the new facts travel in new fields."""
    scene = s2_fetch.scene_of(_feature())
    assert scene is not None
    assert imagery._date_label(scene) == "10 Apr 2024, scene cloud 3%"
    assert '}, scene cloud {cloud}"' in inspect.getsource(imagery._date_label)
    assert "return /^(.+?), (scene cloud .+)$/.exec(label);" in _read("sheet.js")


def test_the_consent_question_and_its_version_do_not_change() -> None:
    """The question's words are those of before the round, so the version a
    yes was given to is the same and nobody is asked again: the hash of the
    words as they stood (kept here word for word) is the version."""
    assert imagery.CONSENT_ASK == ASK_BEFORE
    assert imagery.CONSENT_FULL == FULL_BEFORE
    asked = ASK_BEFORE + "\n\n" + FULL_BEFORE
    digest = hashlib.sha256(asked.encode("utf-8")).hexdigest()[:16]
    assert imagery.consent_version() == f"{digest}-fetch"


# ------------------------------------------------------------------- the view


def test_the_dates_block_folds_the_dates_the_mask_marks() -> None:
    """OE.datesBlock (the face, the list, the test scene and a review set's
    pane all draw their dates with it): the covered dates go into one closed
    fold after the shown dates, its summary their count, its first line whose
    mark it is; when every date with a scene is covered nothing is folded and
    one line says so; a date with no mask is never folded."""
    script = _read("sheet.js")
    for text in (
        "// WORDING: r5p\nOE.CLOUDY_DATES = (n) => "
        "(n === 1 ? '1 cloudy date' : n + ' cloudy dates');\n",
        "// WORDING: r5p\nOE.CLOUDY_WHY = (unit) => '" + CLOUDY_WHY + "\n",
        "// WORDING: r5p\nOE.ALL_CLOUDY = (unit) => '" + ALL_CLOUDY + "\n",
        "OE.cloudyOpen = new Set();\n",
    ):
        assert text in script, text
    dates = _between(script, "OE.datesBlock = (", "\n};")
    for text in (
        "const cloudy = all.filter((d) => d.cloud_here === 'covered');",
        "const masked = all.filter((d) => typeof d.cloud_here === 'string');",
        "const every = cloudy.length > 0 && cloudy.length === masked.length;",
        "const folds = cloudy.length > 0 && !every;",
        "const open = folds ? all.filter((d) => d.cloud_here !== 'covered') : all;",
        "fold = OE.el('details', { class: 'oe-cloudy-dates', "
        "open: OE.cloudyOpen.has(key) },",
        "OE.el('summary', { class: 'small' }, OE.CLOUDY_DATES(cloudy.length)),",
        "OE.CLOUDY_WHY(unit)),",
        "OE.el('div', strip ? { class: 'face-dates' } : { class: 'stack' }, "
        "cloudy.map(one)));",
        "every ? OE.el('div', { class: 'small oe-all-cloudy' }, "
        "OE.ALL_CLOUDY(unit)) : null,",
        "if (typeof d.cloud_here === 'string') "
        "block.setAttribute('data-cloud', d.cloud_here);",
        "if (fold.open) OE.cloudyOpen.add(key); else OE.cloudyOpen.delete(key);",
    ):
        assert text in dates, text
    # The line of every date, then the shown dates, then the fold.
    order = [
        dates.index("every ? OE.el('div', { class: 'small oe-all-cloudy' }"),
        dates.index(": { class: 'stack' }, open.map(one)),"),
        dates.index("    fold);"),
    ]
    assert order == sorted(order)
    # The sheet's pane, its test scene and a review set's pane draw it.
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    assert "return OE.datesBlock(dates, what, {" in editor
    assert "strip: state.view === 'face'," in editor
    assert "datesBlock(dates, 'the public test scene')," in editor
    assert _read("result.html").count("OE.datesBlock(") >= 2


def test_the_details_fold_adds_each_dates_mask_line() -> None:
    """cloudFold ("Scene cloud of each date"): every line as before, and
    after each date's label its mask line as the server words it."""
    script = _read("sheet.js")
    editor = _between(script, "OE.sheetEditor = (app, data,", "\n};")
    fold = _between(editor, "  function cloudFold(dates) {", "\n  }\n")
    for text in (
        "const said = dates.filter((d) => d && "
        "sceneCloud(String(d.label || '')) !== null);",
        "if (!said.length) return null;",
        "OE.el('details', { class: 'oe-clouds small' }, "
        "OE.el('summary', {}, W.cloudFold),",
        "OE.el('div', {}, W.cloudWhole), ...said.map((d) => "
        "[OE.el('div', {}, String(d.label)),",
        "typeof d.mask_line === 'string' && d.mask_line",
        "? OE.el('div', { class: 'oe-mask-line', style: 'padding-left:12px' }, "
        "d.mask_line) : null]));",
    ):
        assert text in fold, text
    assert "cloudFold: 'Scene cloud of each date'," in script
    assert (
        "cloudLine: 'If cloud covers the ' + one + ' on every date, "
        "press Cannot tell.'," in script
    )


# ------------------------------------------------------------------- the docs


def test_the_changelog_the_guide_and_the_notes_say_the_cloud_rule() -> None:
    """The CHANGELOG's Unreleased part has the cloud filter's line; the
    panel guide says it where the pictures are described; the design notes
    give the rule with its classes, thresholds and sources."""
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = _between(log, "## [Unreleased]", "\n## [1.5.1]")
    assert "### Fixes\n" in unreleased
    assert (
        "- Label sheet pictures: each month now shows the date with the least cloud "
        "and shadow on the square itself, by Sentinel-2's own cloud mask, and the "
        "dates the mask marks cloudy are folded away under the others"
    ) in unreleased
    guide = (ROOT / "docs" / "mcp-app.md").read_text(encoding="utf-8")
    assert (
        "Each month shows the date with the least cloud and\nshadow on the square, "
        "by Sentinel-2's own cloud mask." in guide
    )
    assert "Both are sometimes wrong (bright roofs, sand, surf, snow)." in guide
    notes = (ROOT / "spec" / "design" / "mcp-app-notes.md").read_text(encoding="utf-8")
    pictures = _between(notes, "## Satellite pictures", "\n## ")
    for text in (
        "- The cloud rule (round r5-p).",
        "Scene Classification Layer",
        "3 (cloud shadow), 8 and\n  9 (cloud of medium and of high probability), "
        "10 (thin cirrus)",
        "`Sentinel2SCLToMask`",
        "Up to 6 masks are read",
        "at least half of the square",
        "15 seconds, 32 MiB, 250 requests",
        "The masks of all the months are read first",
        "ESA SentiWiki",
    ):
        assert text in pictures, text
