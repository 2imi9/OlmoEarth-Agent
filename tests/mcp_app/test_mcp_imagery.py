# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Window pictures for the label sheet: off until the user says yes, the window
found from the files on disk and checked against its row, one search per
sheet, the least-cloud scene per month, a record beside the sheet, fixed
error sentences, nothing for the model. No test touches the network: a fake
catalogue and bucket serve small GeoTIFFs written here."""

from __future__ import annotations

import csv
import functools
import json
import re
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
rasterio = pytest.importorskip("rasterio")
np = pytest.importorskip("numpy")
import httpx  # noqa: E402
from affine import Affine  # noqa: E402
from mcp import types  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from olmoearth_agent.mcp_app import (  # noqa: E402
    auth,
    imagery,
    review,
    s2_fetch,
    server,
    sheet,
)

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
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out


class Sky:
    """A fake Earth Search: a catalogue answering POST searches and a bucket
    answering Range requests, from GeoTIFFs written in ``folder``."""

    def __init__(
        self,
        folder: Path,
        epsg: int = MAP_EPSG,
        origin: tuple[float, float] = (499000, 3701000),
    ) -> None:
        self.folder = folder
        self.epsg = epsg
        self.t = Affine(10, 0, origin[0], 0, -10, origin[1])
        self.size = (350, 300)
        self.files: dict[str, bytes] = {}
        self.items: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str, str]] = []
        self.bodies: list[dict[str, Any]] = []  # what the catalogue received
        self.bucket_status: int | None = None  # force an answer (302, 200...)
        self.catalogue_status = 200

    def add(
        self,
        ident: str,
        when: str,
        cloud: float,
        values: tuple[int, ...] = (1500, 1400, 1300, 3000),
        hole: bool = False,
    ) -> None:
        w, h = self.size
        hrefs = {}
        for (band, asset), value in zip(ASSETS.items(), values):
            arr = np.full((h, w), value, dtype=np.uint16)
            if hole:
                arr[:, :] = 0
            path = self.folder / f"{ident}_{asset}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=w,
                height=h,
                count=1,
                dtype="uint16",
                crs=f"EPSG:{self.epsg}",
                transform=self.t,
                nodata=0,
                tiled=True,
                blockxsize=128,
                blockysize=128,
                compress="deflate",
            ) as dst:
                dst.write(arr, 1)
            key = f"/sentinel-2-c1-l2a/{ident}/{asset}.tif"
            self.files[key] = path.read_bytes()
            hrefs[band] = {"href": f"https://{s2_fetch.BUCKET_HOST}{key}"}
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
                "assets": hrefs,
            }
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        self.requests.append((request.method, host, path))
        assert (
            "authorization" not in request.headers and "cookie" not in request.headers
        )
        if host == s2_fetch.CATALOGUE_HOST:
            assert request.method == "POST" and str(request.url) == s2_fetch.SEARCH_URL
            if self.catalogue_status != 200:
                return httpx.Response(self.catalogue_status, json={"code": "error"})
            body = json.loads(request.content)
            self.bodies.append(body)
            lo, hi = body["datetime"].split("/")
            found = sorted(
                (i for i in self.items if lo <= i["properties"]["datetime"] <= hi),
                key=lambda i: (i["properties"]["eo:cloud_cover"], i["id"]),
            )
            # Earth Search sends a next link even on the last page.
            nxt = {
                "rel": "next",
                "method": "POST",
                "href": s2_fetch.SEARCH_URL,
                "merge": False,
                "body": {**body, "next": "x"},
            }
            return httpx.Response(
                200,
                json={
                    "type": "FeatureCollection",
                    "features": found,
                    "numberMatched": len(found),
                    "links": [nxt],
                },
            )
        if host == s2_fetch.BUCKET_HOST:
            if self.bucket_status == 302:
                return httpx.Response(
                    302, headers={"location": "https://elsewhere.example/x.tif"}
                )
            data = self.files.get(path)
            if data is None:
                return httpx.Response(404)
            if self.bucket_status == 200:
                return httpx.Response(200, content=data)
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

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            s2_fetch,
            "client_factory",
            lambda: httpx.Client(
                transport=httpx.MockTransport(self.handler), follow_redirects=False
            ),
        )

    def searches(self) -> int:
        return sum(1 for _, host, _ in self.requests if host == s2_fetch.CATALOGUE_HOST)

    def reads(self) -> int:
        return sum(1 for _, host, _ in self.requests if host == s2_fetch.BUCKET_HOST)


#: A handle and a sheet id of no sheet here.
OTHER_HANDLE = "w" + "a" * 16
OTHER_SHEET = "s" + "a" * 16


def _handle(path: Path, index: int) -> tuple[str, str]:
    """The sheet's id and the handle of its window ``index`` (the server's own
    map from handle to window, never sent to a page)."""
    sheet_id, handles = sheet.window_handles(str(path))
    return sheet_id, next(h for h, i in handles.items() if i == index)


def _pictures(path: Path, index: int, **kwargs: Any) -> dict[str, Any]:
    """window_image of the window ``index``, asked by its handle as the page asks."""
    sheet_id, handle = _handle(path, index)
    return imagery.window_image(str(path), handle, sheet_id=sheet_id, **kwargs)


def _centre(index: int) -> tuple[float, float]:
    r, c = divmod(index, GRID[1])
    x, y = MAP_T * (c * P + P / 2, r * P + P / 2)
    (lon,), (lat,) = warp_transform(f"EPSG:{MAP_EPSG}", "EPSG:4326", [x], [y])
    return lon, lat


def _map(
    root: Path,
    *,
    names: dict[str, str] | None = None,
    window: dict[str, str] | None = None,
    raster_dir: Path | None = None,
    source: bool = True,
    rows: list[list[str]] | None = None,
    bands: int = 2,
) -> Path:
    """A plan's sheet, its design, its scores file and its map raster."""
    folder = raster_dir or root / "run"
    folder.mkdir(parents=True, exist_ok=True)
    raster = folder / "scores.tif"
    h, w = GRID[0] * P, GRID[1] * P
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=bands,
        dtype="float32",
        crs=f"EPSG:{MAP_EPSG}",
        transform=MAP_T,
    ) as dst:
        dst.write(np.zeros((bands, h, w), dtype=np.float32))
    # The scores file records the array as read: one band is squeezed to [h, w].
    shape = [h, w] if bands == 1 else [bands, h, w]
    scores = root / "scores_run_p11.json"
    scores.write_text(
        json.dumps(
            {
                "scores": [[0.2, 0.8]] * (GRID[0] * GRID[1]),
                "grid": list(GRID),
                "patch_px": P,
                "raster": {
                    "file": "scores.tif",
                    "path": str(raster),
                    "shape": shape,
                    "sha256": "abc",
                },
                "date_window": (
                    window
                    if window is not None
                    else {
                        "start": "2024-04-01T00:00:00Z",
                        "end": "2024-06-30T23:59:59.999000Z",
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    population: dict[str, Any] = {
        "score_kind": "window_confidence",
        "class_names": names if names is not None else {"0": "absent", "1": "present"},
    }
    if source:
        population["source"] = {"scores_path": str(scores)}
    (root / "kelp.json").write_text(
        json.dumps({"design": "confidence", "population": population}), encoding="utf-8"
    )
    path = root / "kelp_to_label.csv"
    if rows is None:
        rows = []
        for order, (index, cls, stratum) in enumerate(
            [(14, "1", "0"), (25, "0", "1")], start=1
        ):
            lon, lat = _centre(index)
            r, c = divmod(index, GRID[1])
            rows.append(
                [
                    str(order),
                    str(index),
                    str(r),
                    str(c),
                    cls,
                    f"{lon:.7f}",
                    f"{lat:.7f}",
                    stratum,
                    "",
                    "",
                ]
            )
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ONE_MAP)
        writer.writerows(rows)
    return path


@pytest.fixture
def sky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sky:
    folder = tmp_path / "sky"
    folder.mkdir()
    fake = Sky(folder)
    fake.add("S2A_APR_LOW", "2024-04-10T18:40:00Z", 2.0)
    fake.add(
        "S2B_APR_HIGH", "2024-04-20T18:40:00Z", 40.0, values=(9000, 9000, 9000, 9000)
    )
    fake.add("S2A_MAY", "2024-05-12T18:40:00Z", 5.0)
    fake.add("S2B_JUN", "2024-06-03T18:40:00Z", 0.5)
    fake.install(monkeypatch)
    return fake


def test_pictures_are_off_until_the_user_says_yes(root: Path) -> None:
    path = _map(root)
    assert imagery.is_on() is False
    out = _pictures(path, 14, on=imagery.is_on())
    assert out == {
        "ok": True,
        "needs_consent": True,
        "ask": imagery.CONSENT_ASK,
        "test_ask": imagery.TEST_ASK,
    }
    # The question names both hosts and what each receives; so does the test's.
    for text in (imagery.CONSENT_ASK, imagery.TEST_ASK):
        assert imagery.CATALOGUE_HOST in text and imagery.BUCKET_HOST in text
    # The reader test's A/sheet-7: "internet address (IP)".
    assert "Studio" in imagery.CONSENT_ASK
    assert "internet address (IP)" in imagery.CONSENT_ASK
    assert "Nothing of this sheet is sent" in imagery.TEST_ASK
    assert imagery.set_on(True)["on"] is True
    assert imagery.consent_path().is_file() and imagery.consent_path().parent == root
    assert imagery.set_on(False)["on"] is False
    assert not imagery.consent_path().exists()


def test_an_answer_to_another_question_or_to_the_stub_does_not_count(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imagery.set_on(True)
    assert imagery.is_on() is True
    with monkeypatch.context() as patch:
        patch.setattr(imagery, "CONSENT_ASK", imagery.CONSENT_ASK + " Changed.")
        assert imagery.is_on() is False
    assert imagery.is_on() is True
    # This build fetches: an answer given to the stub does not count.
    assert imagery.FETCHES is True and imagery.consent_version().endswith("-fetch")
    stub = imagery.consent_version().replace("-fetch", "-stub")
    imagery.consent_path().write_text(
        json.dumps({"on": True, "version": stub}), encoding="utf-8"
    )
    assert imagery.is_on() is False
    imagery.consent_path().write_text("not json", encoding="utf-8")
    assert imagery.is_on() is False


def test_a_window_gets_one_least_cloud_date_per_month_and_a_record(
    root: Path, sky: Sky
) -> None:
    path = _map(root)
    out = _pictures(path, 14, on=True)
    assert out["ok"] is True and out["synthetic"] is False, out
    assert out["handle"] == _handle(path, 14)[1] and "window_index" not in out
    assert (
        out["reading_rule"] == imagery.READING_RULE
        and out["rule_scope"] == imagery.RULE_SCOPE
    )
    assert out["label_is"] == imagery.LABEL_IS and "not ground truth" in out["label_is"]
    # The caption, word for word, after how big the outlined square is: the
    # scores file's 11-pixel window on a 10 m UTM grid.
    assert out["caption"] == (
        "A square is 11 x 11 map pixels, about 110 m a side. " + imagery.CAPTION
    )
    assert [d["month"] for d in out["dates"]] == ["Apr 2024", "May 2024", "Jun 2024"]
    assert out["dates"][0]["label"] == "10 Apr 2024, scene cloud 2%"
    assert out["dates"][2]["label"] == "3 Jun 2024, scene cloud under 1%"
    pictures = out["dates"][0]["pictures"]
    assert [(p["kind"], p["view"]) for p in pictures] == [
        ("nir", "context"),
        ("nir", "detail"),
        ("true", "context"),
        ("true", "detail"),
    ]
    context, detail = pictures[0], pictures[1]
    assert context["native"] == 7 * P and detail["native"] == 3 * P
    assert context["box"] == {"x": 3 * P, "y": 3 * P, "w": P, "h": P}
    assert detail["box"] == {"x": P, "y": P, "w": P, "h": P}
    assert context["src"].startswith("data:image/png;base64,")
    assert context["width"] == context["native"] * context["scale"]
    # The catalogue heard once per month; the record holds the search and the serving.
    assert sky.searches() == 3
    lines = [json.loads(x) for x in imagery.record_path(path).read_text().splitlines()]
    assert [next(iter(x)) for x in lines] == ["context", "search", "served"]
    assert lines[0]["context"]["reading_rule"] == imagery.READING_RULE
    assert lines[0]["context"]["stretch"] == imagery.STRETCH
    served = lines[2]["served"]
    assert served["window_index"] == 14
    assert [m["id"] for m in served["months"]] == ["S2A_APR_LOW", "S2A_MAY", "S2B_JUN"]
    assert max(abs(v) for v in served["centre_offset_px"]) <= 0.5
    assert "src" not in imagery.record_path(path).read_text()  # no picture kept
    assert imagery.served_any(str(path)) is True
    # A second window: the search comes from the record; the tiles from the cache.
    reads = sky.reads()
    again = _pictures(path, 25, on=True)
    assert again["ok"] is True and sky.searches() == 3 and sky.reads() == reads


def test_the_pictures_are_blind_to_class_stratum_and_order(
    root: Path, sky: Sky
) -> None:
    path = _map(root)
    first = _pictures(path, 14, on=True)
    lon, lat = _centre(14)
    _map(
        root, rows=[["9", "14", "1", "4", "0", f"{lon:.7f}", f"{lat:.7f}", "4", "", ""]]
    )
    second = _pictures(path, 14, on=True)
    assert first["dates"] == second["dates"]


def test_a_month_without_data_here_says_so(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "sky2"
    folder.mkdir()
    fake = Sky(folder)
    fake.add("S2A_APR_HOLE", "2024-04-10T18:40:00Z", 1.0, hole=True)
    fake.add("S2A_MAY", "2024-05-12T18:40:00Z", 5.0)
    fake.install(monkeypatch)
    out = _pictures(_map(root), 14, on=True)
    assert out["ok"] is True
    assert out["dates"][0] == {
        "month": "Apr 2024",
        "label": "Apr 2024: no scene with data here",
        "pictures": [],
    }
    assert out["dates"][2]["pictures"] == []  # June: no scene at all


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda root, tmp: _map(root, source=False), imagery.NO_FOOTPRINT),
        (lambda root, tmp: _map(root, window={}), imagery.NO_PERIOD),
        (
            lambda root, tmp: _map(
                root, window={"start": "2023-01-01", "end": "2024-02-01"}
            ),
            imagery.LONG_PERIOD,
        ),
        (
            lambda root, tmp: _map(root, raster_dir=tmp / "elsewhere"),
            imagery.NO_FOOTPRINT,
        ),
        (
            lambda root, tmp: _map(
                root, rows=[["1", "14", "1", "4", "1", "10.0", "50.0", "0", "", ""]]
            ),
            imagery.MISMATCH,
        ),
        (
            lambda root, tmp: _map(
                root, rows=[["1", "14", "2", "4", "1", "", "", "0", "", ""]]
            ),
            imagery.MISMATCH,
        ),
    ],
)
def test_a_window_that_cannot_be_placed_is_refused_before_any_fetch(
    root: Path, tmp_path: Path, sky: Sky, make: Any, error: str
) -> None:
    out = _pictures(make(root, tmp_path), 14, on=True)
    assert out == {"ok": False, "error": error}
    assert sky.requests == []


def test_fetch_failures_are_fixed_sentences(
    root: Path, sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _map(root)
    assert imagery.window_image(
        str(path), OTHER_HANDLE, sheet_id=sheet.window_handles(str(path))[0], on=True
    ) == {
        "ok": False,
        "error": imagery.NOT_ON_SHEET,
    }
    assert imagery.window_image(
        "/etc/passwd", OTHER_HANDLE, sheet_id=OTHER_SHEET, on=True
    ) == {
        "ok": False,
        "error": imagery.NOT_READABLE,
    }
    # A failure another try can get past says so (again, r5-a L1); the two
    # refusals above would be the same on a second try, and carry no again.
    sky.catalogue_status = 500
    assert _pictures(path, 14, on=True) == {
        "ok": False,
        "error": imagery.CATALOGUE_FAILED,
        "again": True,
    }
    sky.catalogue_status = 200
    # A redirect is never followed; a bucket that ignores the range is not read.
    for status in (302, 200):
        sky.bucket_status = status
        s2_fetch.clear_cache()
        before = len(sky.requests)
        assert _pictures(path, 14, on=True) == {
            "ok": False,
            "error": imagery.FETCH_FAILED,
            "again": True,
        }
        assert all(host != "elsewhere.example" for _, host, _ in sky.requests[before:])
    sky.bucket_status = None
    s2_fetch.clear_cache()
    monkeypatch.setattr(
        s2_fetch, "Budget", functools.partial(s2_fetch.Budget, seconds=0.0)
    )
    imagery.record_path(path).unlink()
    assert _pictures(path, 14, on=True) == {
        "ok": False,
        "error": imagery.TOO_SLOW,
        "again": True,
    }


def test_the_public_test_scene_sends_nothing_of_a_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
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
        fake.add(f"S2A_{month}", f"2024-{month}-15T08:49:00Z", 1.0)
    fake.install(monkeypatch)
    out = imagery.public_test_scene()
    assert out["ok"] is True and out["test_scene"] is True, out
    assert [d["month"] for d in out["dates"]] == ["Jan 2024", "Feb 2024", "Mar 2024"]
    assert all(len(d["pictures"]) == 4 for d in out["dates"])
    assert (
        out["reading_rule"] is None and "not a square of this sheet" in out["caption"]
    )
    # The test scene's caption is its own: no sheet's square size in front.
    assert out["caption"] == imagery.TEST_CAPTION
    assert list(root.iterdir()) == []  # no record, no consent file


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    return (await srv.request_handlers[types.CallToolRequest](request)).root


@pytest.mark.asyncio
async def test_the_view_tools_tell_the_model_nothing(root: Path, sky: Sky) -> None:
    path = _map(root)
    app = server.AppServer(None, keys=auth.KeyStore(), base_url="http://mock/api/v1")
    srv = server.build_server(app)
    sheet_id, handle = _handle(path, 14)
    by_handle = {"labels_csv_path": str(path), "sheet_id": sheet_id, "handle": handle}
    asked = await _call(srv, "olmoearth_app_window_image", by_handle)
    assert asked.structuredContent["needs_consent"] is True
    assert [c.text for c in asked.content] == ['{"ok": true}']
    switched = await _call(srv, "olmoearth_app_imagery", {"on": True})
    assert switched.structuredContent["on"] is True
    shown = await _call(srv, "olmoearth_app_window_image", by_handle)
    assert shown.structuredContent["dates"][0]["pictures"]
    assert shown.structuredContent["handle"] == handle
    assert "window_index" not in shown.structuredContent
    assert [c.text for c in shown.content] == ['{"ok": true}']
    dumped = shown.model_dump_json()
    assert (
        str(root) not in dumped
        and s2_fetch.BUCKET_HOST not in dumped
        and "S2A_APR_LOW" not in dumped
    )
    missing = await _call(
        srv, "olmoearth_app_window_image", {**by_handle, "handle": OTHER_HANDLE}
    )
    assert missing.structuredContent == {"ok": False, "error": imagery.NOT_ON_SHEET}
    # A panel of an older server names the window by its index: refused.
    older = await _call(
        srv,
        "olmoearth_app_window_image",
        {"labels_csv_path": str(path), "window_index": 14},
    )
    assert older.structuredContent == {"ok": False, "error": imagery.OLDER_PANEL}
    assert [c.text for c in older.content] == ['{"ok": true}']
    refused = await _call(srv, "olmoearth_app_imagery", {"on": "yes"})
    assert refused.isError is True
    assert [c.text for c in refused.content] == ['{"ok": true}']
    assert "on must be true or false" in refused.structuredContent["error"]
    await app.aclose()


@pytest.mark.asyncio
async def test_an_estimate_on_labels_read_off_pictures_says_what_they_are(
    root: Path, sky: Sky, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _map(root)
    app = server.AppServer(None, keys=auth.KeyStore(), base_url="http://mock/api/v1")

    async def dispatch(call: Any, ctx: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "result": {"error_rate": 0.1, "must_state": ["The rate is 10%."]},
        }

    monkeypatch.setattr(app.registry, "dispatch", dispatch)
    args = {"design_path": str(root / "kelp.json"), "labels_path": str(path)}
    before = await app.call_agent_tool("olmoearth_estimate_map_error", args)
    assert imagery.LABEL_MUST_STATE not in before.content[0].text
    assert _pictures(path, 14, on=True)["ok"] is True
    after = await app.call_agent_tool("olmoearth_estimate_map_error", args)
    assert imagery.LABEL_MUST_STATE in after.content[0].text
    assert "The rate is 10%." in after.content[0].text
    await app.aclose()


def test_the_catalogue_hears_the_maps_rounded_extent_once(root: Path, sky: Sky) -> None:
    path = _map(root)
    assert _pictures(path, 14, on=True)["ok"] is True
    assert _pictures(path, 25, on=True)["ok"] is True
    with rasterio.open(root / "run" / "scores.tif") as src:
        extent = s2_fetch.rounded(
            transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        )
    assert [b["bbox"] for b in sky.bodies] == [list(extent)] * 3
    assert [b["datetime"][:7] for b in sky.bodies] == ["2024-04", "2024-05", "2024-06"]


def test_another_map_gets_no_rule_and_one_band_maps_are_placed(
    root: Path, sky: Sky
) -> None:
    path = _map(root, names={"0": "water", "1": "trees"}, bands=1)
    out = _pictures(path, 14, on=True)
    assert out["ok"] is True, out
    assert out["reading_rule"] is None and out["rule_scope"] is None
    context = json.loads(imagery.record_path(path).read_text().splitlines()[0])[
        "context"
    ]
    assert context["reading_rule"] is None
    assert imagery.reading_rule(["Absent", "Present"]) == imagery.READING_RULE
    assert imagery.reading_rule(["water", "trees"]) is None


def test_pictures_switched_off_while_waiting_send_nothing(root: Path, sky: Sky) -> None:
    path = _map(root)
    out = _pictures(path, 14, on=True, still_on=lambda: False)
    assert out == {
        "ok": True,
        "needs_consent": True,
        "ask": imagery.CONSENT_ASK,
        "test_ask": imagery.TEST_ASK,
    }
    assert sky.requests == []


def test_a_window_is_named_by_its_handle_and_an_old_page_is_refused(
    root: Path, sky: Sky
) -> None:
    """The pictures take a window's handle and the sheet's id: an id the sheet
    no longer has, a handle not on it, or a window index is refused before
    any request leaves the computer."""
    path = _map(root)
    sheet_id, handle = _handle(path, 14)
    assert imagery.window_image(str(path), handle, sheet_id=OTHER_SHEET, on=True) == {
        "ok": False,
        "error": imagery.CHANGED,
    }
    for wrong in (OTHER_HANDLE, 14, None):
        assert imagery.window_image(str(path), wrong, sheet_id=sheet_id, on=True) == {
            "ok": False,
            "error": imagery.NOT_ON_SHEET,
        }
    # Drawn again with other windows: the old id is refused.
    lon, lat = _centre(14)
    _map(
        root, rows=[["1", "15", "1", "5", "1", f"{lon:.7f}", f"{lat:.7f}", "0", "", ""]]
    )
    assert imagery.window_image(str(path), handle, sheet_id=sheet_id, on=True) == {
        "ok": False,
        "error": imagery.CHANGED,
    }
    assert sky.requests == []
    assert not imagery.record_path(path).exists()


def _words(text: str) -> str:
    return " ".join(text.split())


def _review(
    root: Path, *, skip: tuple[int, ...] = (), centres: bool = True
) -> tuple[Any, Path]:
    """A review set ranked from the kelp map's scores file, as
    olmoearth_scores_from_file writes it (raster, window size, period and one
    centre per row; ``skip``: no-data windows left out, so the file carries
    ``windows``): its key and the scores file. Windows 14 and 25 are listed."""
    _map(root)
    scores = root / "scores_run_p11.json"
    payload = json.loads(scores.read_text(encoding="utf-8"))
    kept = [i for i in range(GRID[0] * GRID[1]) if i not in skip]
    payload["scores"] = [[0.2, 0.8]] * len(kept)
    if skip:
        payload["windows"] = kept
    if centres:
        payload["centres_lon_lat"] = [list(_centre(i)) for i in kept]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    inner = {
        "n_review": 2,
        "review": [
            {
                "rank": 1,
                "window_index": 14,
                "row": 1,
                "col": 4,
                "margin": 0.1,
                "predicted_class": 1,
                "class_name": "present",
            },
            {
                "rank": 2,
                "window_index": 25,
                "row": 2,
                "col": 5,
                "margin": 0.2,
                "predicted_class": 0,
                "class_name": "absent",
            },
        ],
    }
    block = review.review_view(
        "olmoearth_review_set", inner, {"scores_path": str(scores)}
    )
    assert block is not None
    return block["key"], scores


def test_the_sheet_record_path_is_unchanged() -> None:
    """A sheet's record keeps its name; a review set's lies beside its record."""
    assert imagery.record_path(Path("/o/a.csv")) == Path("/o/a.imagery.jsonl")
    assert imagery.record_path(Path("/o/a.b_to_label.csv")) == Path(
        "/o/a.b_to_label.imagery.jsonl"
    )
    key = "0123456789abcdef01234567"
    assert imagery.record_path(Path(f"/o/review_sets/{key}.json")) == Path(
        f"/o/review_sets/{key}.imagery.jsonl"
    )


def test_review_pictures_are_off_until_the_user_says_yes(root: Path, sky: Sky) -> None:
    """While pictures are off a review set's window answers the question, with
    the review set's own test-scene question; nothing reaches Earth Search.
    The question now names sheets and review sets, so it is asked once more."""
    key, _ = _review(root)
    assert re.fullmatch(r"[0-9a-f]{24}", key)
    out = imagery.review_image(key, 1, on=imagery.is_on())
    assert out == {
        "ok": True,
        "needs_consent": True,
        "ask": imagery.CONSENT_ASK,
        "test_ask": imagery.REVIEW_TEST_ASK,
    }
    assert "once per sheet or review set" in _words(imagery.CONSENT_ASK)
    assert "Nothing of this review set is sent" in _words(imagery.REVIEW_TEST_ASK)
    for host in (imagery.CATALOGUE_HOST, imagery.BUCKET_HOST):
        assert host in imagery.REVIEW_TEST_ASK
    assert sky.requests == []


@pytest.mark.parametrize("skip", [(), (3, 40)])
def test_a_review_window_of_a_scores_file_is_placed_from_its_centre(
    root: Path, sky: Sky, skip: tuple[int, ...]
) -> None:
    """The map path, as the label sheet's: the window placed from the scores
    file's centre for it (its place in ``windows`` when no-data windows were
    left out), one least-cloud date per month, the window outlined whole, a
    record beside the review's; no sheet is needed and nothing is labelled."""
    key, _ = _review(root, skip=skip)
    out = imagery.review_image(key, 1, on=True)
    assert out["ok"] is True and out["rank"] == 1, out
    assert out["reading_rule"] is None and out["label_is"] is None
    assert [d["month"] for d in out["dates"]] == ["Apr 2024", "May 2024", "Jun 2024"]
    context = out["dates"][0]["pictures"][0]
    assert (context["kind"], context["view"]) == ("nir", "context")
    assert context["box"] == {"x": 3 * P, "y": 3 * P, "w": P, "h": P}
    assert "window_index" not in out and "handle" not in out
    record = root / "review_sets" / f"{key}.imagery.jsonl"
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    assert [next(iter(x)) for x in lines] == ["context", "search", "served"]
    assert lines[0]["context"]["record"] == "olmoearth-agent review-set pictures"
    assert lines[0]["context"]["review"] == key
    served = lines[2]["served"]
    assert (served["review"], served["rank"], served["window_index"]) == (key, 1, 14)
    assert max(abs(v) for v in served["centre_offset_px"]) <= 0.5
    assert not list(root.glob("*.imagery.jsonl"))  # no sheet's record
    # The second window: the search from the record, the tiles from the cache.
    reads = sky.reads()
    again = imagery.review_image(key, 2, on=True)
    assert again["ok"] is True and sky.searches() == 3 and sky.reads() == reads


def test_a_review_window_that_cannot_be_placed_is_refused_before_any_fetch(
    root: Path, sky: Sky
) -> None:
    """A rank not listed, a key of no review, a scores file changed since the
    ranking, or a file whose centre disagrees with the raster is refused with
    its sentence before anything leaves the computer."""
    key, scores = _review(root)
    assert imagery.review_image(key, 3, on=True) == {
        "ok": False,
        "error": review.NOT_LISTED,
    }
    assert imagery.review_image("f" * 24, 1, on=True) == {
        "ok": False,
        "error": review.REVIEW_UNKNOWN,
    }
    payload = json.loads(scores.read_text(encoding="utf-8"))
    payload["centres_lon_lat"][14] = [10.0, 50.0]
    scores.write_text(json.dumps(payload), encoding="utf-8")
    assert imagery.review_image(key, 1, on=True) == {
        "ok": False,
        "error": review.REVIEW_CHANGED,
    }
    # Ranked again on the changed file: the centre now disagrees with the raster.
    moved = review.review_view(
        "olmoearth_review_set",
        {"review": [{"rank": 1, "window_index": 14, "row": 1, "col": 4}]},
        {"scores_path": str(scores)},
    )
    assert moved is not None and moved["pictures"] is True
    assert imagery.review_image(moved["key"], 1, on=True) == {
        "ok": False,
        "error": imagery.REVIEW_MISMATCH,
    }
    assert sky.requests == []


def test_a_scores_file_without_centres_gets_no_cards_that_picture(root: Path) -> None:
    """A file that does not record each window's centre is not a map path."""
    key, _ = _review(root, centres=False)
    assert key is None
