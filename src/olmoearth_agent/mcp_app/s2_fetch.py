# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Dated Sentinel-2 pictures of a small box, from Earth Search's public
catalogue and bucket (the label sheet's pictures, :mod:`imagery`).

What leaves this computer: one search per month of the period to
:data:`CATALOGUE_HOST` (an extent rounded outward to 0.1 degree, and the
month), and byte-range requests for parts of four band files and of the
scene's own cloud mask (its Scene Classification Layer, :data:`MASK`) to
:data:`BUCKET_HOST`. Nothing else, and no credential.

How that is held:

- every URL is https on exactly one of the two hosts, with no port, user,
  query or fragment (:func:`allowed`), checked before each request;
- one httpx client per call: redirects are never followed, no proxy or
  .netrc is taken from the environment, no key is sent; it is closed at the
  end of the call;
- GDAL opens no socket: rasterio reads each band through a Python opener
  (rasterio 1.4), and this module fetches every byte;
- one deadline, a byte budget and a request cap per call (:class:`Budget`);
  the cloud masks have their own, smaller one (:func:`mask_fetch`), and a
  mask that cannot be read never fails the pictures.

numpy and rasterio are imported lazily (the ``inferencex`` extra), so the
core install never needs them.
"""

from __future__ import annotations

import importlib
import itertools
import json
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

CATALOGUE_HOST = "earth-search.aws.element84.com"
BUCKET_HOST = "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
SEARCH_URL = f"https://{CATALOGUE_HOST}/v1/search"
COLLECTION = "sentinel-2-c1-l2a"
#: The four bands read, by their Earth Search asset keys (B04, B03, B02, B08).
BANDS = ("red", "green", "blue", "nir")
#: Search page size; a month over a small extent holds far fewer scenes.
SEARCH_LIMIT = 100
#: Scenes tried per month before the month shows no data here.
TRIES_PER_MONTH = 3
#: The scene's own cloud mask, by its Earth Search asset key: the Scene
#: Classification Layer (SCL) that ESA's Sen2Cor makes with every L2A product,
#: a 20 m uint8 file beside the bands (0 is no data).
MASK = "scl"
#: The SCL classes that hide the ground: cloud shadow (3), cloud of medium (8)
#: and of high (9) probability, thin cirrus (10); rslearn's Sentinel2SCLToMask
#: excludes the same four by default. Snow (11), dark features (2),
#: unclassified (7), water, vegetation and bare ground are not covered: they
#: are often the real ground.
COVERED_CLASSES = (3, 8, 9, 10)
#: Scenes whose mask is read per month at most.
MASK_TRIES = 6
#: Share of the outlined square the mask must have data on (SCL != 0) for its
#: score to count.
MASK_VALID = 0.5
#: The cloud masks' own time for one call (:func:`mask_fetch`), and the most
#: that the masks' pass adds back to the bands' deadline (:func:`dated`).
MASK_SECONDS = 15.0
#: Months whose masks are read at a time (one small read each).
MASK_WORKERS = 8

_JSON_CAP = 20 << 20  # a search answer
#: Share of the outlined square that must have data in all four bands: L2A
#: water holds scattered no-data pixels (seen live, 12 Nov 2026), and a
#: strict "every pixel" would turn water windows away.
DATA_SHARE = 0.9
#: A pixel the picture itself shows white: red, green and blue reflectance
#: all at least this. The thick cloud Sen2Cor's SCL classed as land and water
#: at a coastline (seen live on 20 Nov 2026 on a coastal map) was 0.8
#: to 1.0; clear water, plants, sand, concrete and most beaches stay below 0.4
#: (snow, salt flats and white roofs may not: the fold says the check can be
#: wrong).
BRIGHT = 0.4
#: The share of the square's pixels with data that are white from which a
#: picture counts as covered, whatever its mask says.
BRIGHT_SHARE = 0.25
#: Another band read for a white scene starts only while the band budget has
#: spent under this share of its requests and bytes.
WHITE_ROOM = 0.7
#: Search pages followed at most (the next link is a POST with a token).
SEARCH_PAGES = 5

_BLOCK = 1 << 16  # bytes fetched per cache block (64 KiB; a COG header fits)
_CACHE_CAP = 96 << 20  # blocks kept across calls, for neighbouring windows
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_USER_AGENT = "olmoearth-agent-pictures/1"


class FetchError(Exception):
    """A fetch that did not complete; ``kind`` picks the fixed sentence."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind  # catalogue | fetch | deadline | budget


def allowed(url: str, host: str) -> bool:
    """Whether ``url`` is https on exactly ``host``: no user, port, query or
    fragment, and a path that starts at the root."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and parts.netloc == host
        and parts.path.startswith("/")
        and not parts.query
        and not parts.fragment
    )


class Budget:
    """One call's deadline, bytes and requests (thread-safe)."""

    def __init__(
        self, seconds: float = 40.0, max_bytes: int = 128 << 20, max_requests: int = 400
    ) -> None:
        self.deadline = time.monotonic() + seconds
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self.bytes = 0
        self.requests = 0
        self._lock = threading.Lock()

    def timeout(self) -> httpx.Timeout:
        """A per-request timeout that never passes the deadline."""
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise FetchError("deadline")
        t = min(10.0, left)
        return httpx.Timeout(t, connect=min(5.0, t))

    def take_request(self) -> None:
        with self._lock:
            self.requests += 1
            if self.requests > self.max_requests:
                raise FetchError("budget")

    def take_bytes(self, n: int) -> None:
        with self._lock:
            self.bytes += n
            if self.bytes > self.max_bytes:
                raise FetchError("budget")
        if time.monotonic() > self.deadline:
            raise FetchError("deadline")

    def extend(self, seconds: float) -> None:
        """Move the deadline ``seconds`` later (never earlier)."""
        with self._lock:
            self.deadline += max(0.0, seconds)

    def stop(self) -> None:
        """End the time now: no further request starts on this budget."""
        with self._lock:
            self.deadline = min(self.deadline, time.monotonic())


def _default_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
    )


#: Builds the call's client; tests replace it with a mock transport.
client_factory: Callable[[], httpx.Client] = _default_client


# ----------------------------------------------------------------- catalogue


@dataclass(frozen=True)
class Scene:
    """One Sentinel-2 item, as much of it as the pictures need."""

    id: str
    datetime: str
    cloud: float
    datatake: str
    grid: str
    epsg: int
    bbox: tuple[float, float, float, float]
    hrefs: dict[str, str] = field(hash=False)
    baseline: str = ""
    #: Outer rings of the item's data footprint (lon, lat); empty: unknown.
    footprint: tuple[tuple[tuple[float, float], ...], ...] = ()
    #: The scene's cloud mask file (:data:`MASK`); empty: none known.
    mask_href: str = ""

    def as_record(self) -> dict[str, Any]:
        """What the record keeps of the scene."""
        return {
            "id": self.id,
            "datetime": self.datetime,
            "cloud": self.cloud,
            "datatake": self.datatake,
            "grid": self.grid,
            "epsg": self.epsg,
            "bbox": list(self.bbox),
            "hrefs": dict(self.hrefs),
            "baseline": self.baseline,
            "footprint": [[list(p) for p in ring] for ring in self.footprint],
            "mask": self.mask_href,
        }

    def covers(self, points: Iterable[tuple[float, float]]) -> bool:
        """Whether every point (lon, lat) lies in the data footprint (True
        when the footprint is unknown: the read then decides)."""
        if not self.footprint:
            return True
        return all(any(_inside(p, ring) for ring in self.footprint) for p in points)

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> Scene | None:
        """A scene back from the record, checked as a search answer is. A
        record written before the cloud mask (no ``"mask"`` key) derives the
        mask's link from the red band's (:func:`_derived_mask`)."""
        try:
            hrefs = {b: str(data["hrefs"][b]) for b in BANDS}
            bbox = tuple(float(v) for v in data["bbox"])
            if len(bbox) != 4 or not all(
                allowed(h, BUCKET_HOST) for h in hrefs.values()
            ):
                return None
            return cls(
                id=str(data["id"]),
                datetime=str(data["datetime"]),
                cloud=float(data["cloud"]),
                datatake=str(data["datatake"]),
                grid=str(data["grid"]),
                epsg=int(data["epsg"]),
                bbox=bbox,
                hrefs=hrefs,
                baseline=str(data.get("baseline", "")),
                footprint=_rings(data.get("footprint") or []),
                mask_href=(
                    _mask_link(data["mask"])
                    if "mask" in data
                    else _derived_mask(hrefs["red"])
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _mask_link(href: Any) -> str:
    """A cloud mask's link when it is a .tif on the bucket, else ``""`` (the
    scene is then used without a mask)."""
    if not isinstance(href, str) or not allowed(href, BUCKET_HOST):
        return ""
    return href if urlsplit(href).path.endswith(".tif") else ""


def _derived_mask(red: str) -> str:
    """The cloud mask's link of a scene recorded before masks were kept: the
    file ``SCL.tif`` in the red band's folder, when that band is ``B04.tif``;
    else ``""``. A wrong guess only fails that mask's read."""
    tail = "/B04.tif"
    if not urlsplit(red).path.endswith(tail) or not red.endswith(tail):
        return ""
    return _mask_link(red[: -len(tail)] + "/SCL.tif")


def _inside(point: tuple[float, float], ring: tuple[tuple[float, float], ...]) -> bool:
    """Ray casting: whether (lon, lat) lies inside the closed ring."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _rings(raw: Any, cap: int = 4000) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Rings of (lon, lat) from a list of rings; () when malformed."""
    try:
        rings = tuple(
            tuple((float(p[0]), float(p[1])) for p in ring) for ring in raw[:16]
        )
    except (TypeError, ValueError, IndexError):
        return ()
    if any(len(r) < 4 or len(r) > cap for r in rings):
        return ()
    return rings


def _footprint(geometry: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    """The outer rings of a GeoJSON Polygon or MultiPolygon; () otherwise."""
    if not isinstance(geometry, dict):
        return ()
    coords = geometry.get("coordinates")
    if geometry.get("type") == "Polygon" and isinstance(coords, list) and coords:
        return _rings([coords[0]])
    if geometry.get("type") == "MultiPolygon" and isinstance(coords, list):
        return _rings([poly[0] for poly in coords if isinstance(poly, list) and poly])
    return ()


def _epsg(props: dict[str, Any]) -> int | None:
    value = props.get("proj:epsg")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    code = props.get("proj:code")
    if isinstance(code, str) and code.upper().startswith("EPSG:"):
        try:
            return int(code[5:])
        except ValueError:
            return None
    return None


def scene_of(feature: Any) -> Scene | None:
    """A search feature as a :class:`Scene`, or ``None`` when it lacks a field
    the pictures need or a band link is not on the bucket. A cloud mask link
    (:data:`MASK`) that is missing or not on the bucket leaves the scene
    usable, without a mask."""
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties")
    assets = feature.get("assets")
    bbox = feature.get("bbox")
    if not isinstance(props, dict) or not isinstance(assets, dict):
        return None
    try:
        box = tuple(float(v) for v in bbox)  # type: ignore[union-attr]
        cloud = float(props["eo:cloud_cover"])
    except (KeyError, TypeError, ValueError):
        return None
    epsg = _epsg(props)
    hrefs: dict[str, str] = {}
    for band in BANDS:
        asset = assets.get(band)
        href = asset.get("href") if isinstance(asset, dict) else None
        if not isinstance(href, str) or not allowed(href, BUCKET_HOST):
            return None
        if not urlsplit(href).path.endswith(".tif"):
            return None
        hrefs[band] = href
    mask = assets.get(MASK)
    ident, when = feature.get("id"), props.get("datetime")
    datatake, grid = props.get("s2:datatake_id"), props.get("grid:code")
    if not all(isinstance(v, str) and v for v in (ident, when, datatake, grid)):
        return None
    if epsg is None or len(box) != 4 or not math.isfinite(cloud):
        return None
    return Scene(
        id=str(ident),
        datetime=str(when),
        cloud=cloud,
        datatake=str(datatake),
        grid=str(grid),
        epsg=epsg,
        bbox=box,
        hrefs=hrefs,
        baseline=str(props.get("s2:processing_baseline", "")),
        footprint=_footprint(feature.get("geometry")),
        mask_href=_mask_link(mask.get("href") if isinstance(mask, dict) else None),
    )


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def search_body(
    bbox: tuple[float, float, float, float], start: datetime, end: datetime
) -> dict[str, Any]:
    """The POST body of one month's search (least scene cloud first)."""
    return {
        "collections": [COLLECTION],
        "bbox": [round(v, 1) for v in bbox],
        "datetime": f"{iso(start)}/{iso(end)}",
        "limit": SEARCH_LIMIT,
        "sortby": [
            {"field": "properties.eo:cloud_cover", "direction": "asc"},
            {"field": "id", "direction": "asc"},
        ],
    }


def _post_json(
    client: httpx.Client, budget: Budget, url: str, body: dict[str, Any]
) -> dict[str, Any]:
    if not allowed(url, CATALOGUE_HOST):
        raise FetchError("catalogue")
    budget.take_request()
    try:
        with client.stream("POST", url, json=body, timeout=budget.timeout()) as resp:
            if resp.status_code != 200:
                raise FetchError("catalogue")
            chunks, got = [], 0
            for chunk in resp.iter_bytes():
                got += len(chunk)
                if got > _JSON_CAP:
                    raise FetchError("catalogue")
                budget.take_bytes(len(chunk))
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise FetchError("catalogue") from exc
    try:
        data = json.loads(b"".join(chunks))
    except ValueError as exc:
        raise FetchError("catalogue") from exc
    if not isinstance(data, dict):
        raise FetchError("catalogue")
    return data


def search_month(
    client: httpx.Client,
    budget: Budget,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
) -> tuple[list[Scene], dict[str, Any]]:
    """The month's scenes over ``bbox`` (one page, least cloud first) and
    what was asked, for the record."""
    first = body = search_body(bbox, start, end)
    features: list[Any] = []
    matched: Any = None
    pages = 0
    while True:
        data = _post_json(client, budget, SEARCH_URL, body)
        page = data.get("features")
        if not isinstance(page, list):
            raise FetchError("catalogue")
        pages += 1
        features.extend(page)
        matched = data.get("numberMatched")
        done = len(page) < SEARCH_LIMIT or (
            isinstance(matched, int) and len(features) >= matched
        )
        nxt = next(
            (
                link
                for link in data.get("links") or []
                if isinstance(link, dict) and link.get("rel") == "next"
            ),
            None,
        )
        # Earth Search sends a next link even on the last page (seen live):
        # follow it only while the answer is not complete, and only as the
        # same POST to the same URL with the body it gives.
        if done or nxt is None or pages >= SEARCH_PAGES:
            break
        if nxt.get("href") != SEARCH_URL or nxt.get("method") != "POST":
            raise FetchError("catalogue")
        token = nxt.get("body")
        if not isinstance(token, dict):
            raise FetchError("catalogue")
        body = {**body, **token} if nxt.get("merge") else token
    scenes = [s for s in (scene_of(f) for f in features) if s is not None]
    asked = {
        "body": first,
        "pages": pages,
        "returned": len(features),
        "kept": len(scenes),
        "matched": matched,
        "complete": isinstance(matched, int) and len(features) >= matched,
    }
    return scenes, asked


# -------------------------------------------------------------------- period


def parse_moment(text: Any, *, end: bool) -> datetime:
    """A date_window bound: '2024-04-01', or Studio's '2024-06-30T23:59:59.999000Z'.
    A date-only end is inclusive to the end of that day (UTC)."""
    raw = str(text).strip()
    moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    if end and len(raw) == 10:
        moment = moment + timedelta(days=1) - timedelta(microseconds=1)
    return moment.astimezone(timezone.utc)


def months(start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    """The calendar months of the period, each clipped to it: (label, start, end)."""
    if end < start:
        raise ValueError("period ends before it starts")
    out = []
    year, month = start.year, start.month
    while True:
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        nxt = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=timezone.utc)
        if first > end:
            break
        lo, hi = max(first, start), min(nxt - timedelta(microseconds=1), end)
        out.append((first.strftime("%b %Y"), lo, hi))
        year, month = nxt.year, nxt.month
    # A period that ends on the first day of a month (a half-open end, such as
    # 2025-01-01 for the year 2024) does not add that month.
    if len(out) > 1 and end.day == 1:
        out.pop()
    return out


def rounded(
    bbox: Iterable[float], step: float = 0.1
) -> tuple[float, float, float, float]:
    """An extent rounded outward to ``step`` degrees, clamped to the globe."""
    west, south, east, north = (float(v) for v in bbox)
    if not west <= east:
        raise ValueError("an extent across the antimeridian")
    f = 1 / step
    return (
        max(-180.0, math.floor(west * f) / f),
        max(-90.0, math.floor(south * f) / f),
        min(180.0, math.ceil(east * f) / f),
        min(90.0, math.ceil(north * f) / f),
    )


def candidates(
    scenes: Iterable[Scene],
    box: tuple[float, float, float, float],
    points: Iterable[tuple[float, float]] = (),
) -> list[Scene]:
    """Scenes whose extent holds ``box`` (lon/lat) and whose data footprint
    holds ``points`` (the picture's corners and centre), least scene cloud
    first; one per datatake and grid square. Blind to anything but cloud and
    time."""
    points = list(points)
    best: dict[tuple[str, str], Scene] = {}
    for s in scenes:
        w, so, e, n = s.bbox
        if not (w <= box[0] and so <= box[1] and e >= box[2] and n >= box[3]):
            continue
        if not s.covers(points):
            continue
        key = (s.datatake, s.grid)
        old = best.get(key)
        if old is None or (s.baseline, s.id) > (old.baseline, old.id):
            best[key] = s
    return sorted(best.values(), key=lambda s: (s.cloud, s.datetime, s.id))


# --------------------------------------------------------------------- reads

_blocks: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_sizes: dict[str, int] = {}
_blocks_bytes = 0
_cache_lock = threading.Lock()


def _cache_get(href: str, n: int) -> bytes | None:
    with _cache_lock:
        data = _blocks.get((href, n))
        if data is not None:
            _blocks.move_to_end((href, n))
        return data


def _cache_put(href: str, n: int, data: bytes) -> None:
    global _blocks_bytes
    with _cache_lock:
        old = _blocks.pop((href, n), None)
        if old is not None:
            _blocks_bytes -= len(old)
        _blocks[(href, n)] = data
        _blocks_bytes += len(data)
        while _blocks_bytes > _CACHE_CAP and _blocks:
            _, dropped = _blocks.popitem(last=False)
            _blocks_bytes -= len(dropped)


def clear_cache() -> None:
    """Forget every cached block (tests)."""
    global _blocks_bytes
    with _cache_lock:
        _blocks.clear()
        _sizes.clear()
        _blocks_bytes = 0


class Fetch:
    """One call's byte fetching: the client, the budget, the first failure."""

    def __init__(self, client: httpx.Client, budget: Budget) -> None:
        self.client = client
        self.budget = budget
        self.failed: BaseException | None = None
        self.log: list[tuple[str, int, int]] = []  # (href, start, end) per request

    def fail(self, exc: BaseException) -> None:
        if self.failed is None:
            self.failed = exc

    def size(self, href: str) -> int:
        with _cache_lock:
            known = _sizes.get(href)
        if known is None:
            self._blocks(href, 0, 1)
            with _cache_lock:
                known = _sizes[href]
        return known

    def read(self, href: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of ``href``; ``end`` within the file."""
        first, last = start // _BLOCK, (end - 1) // _BLOCK
        joined = b"".join(self._blocks(href, first, last + 1))
        off = start - first * _BLOCK
        return joined[off : off + (end - start)]

    def _blocks(self, href: str, first: int, stop: int) -> list[bytes]:
        have: dict[int, bytes] = {}
        missing: list[int] = []
        for n in range(first, stop):
            data = _cache_get(href, n)
            if data is None:
                missing.append(n)
            else:
                have[n] = data
        runs = [
            [n for _, n in group]
            for _, group in itertools.groupby(enumerate(missing), lambda p: p[1] - p[0])
        ]
        for run in runs:
            a = run[0] * _BLOCK
            z = (run[-1] + 1) * _BLOCK - 1
            with _cache_lock:
                total = _sizes.get(href)
            if total is not None:
                z = min(z, total - 1)
            data = self._range(href, a, z)
            for n in run:
                piece = data[(n - run[0]) * _BLOCK : (n - run[0] + 1) * _BLOCK]
                _cache_put(href, n, piece)
                have[n] = piece
        return [have[n] for n in range(first, stop)]

    def _range(self, href: str, a: int, z: int) -> bytes:
        if not allowed(href, BUCKET_HOST) or z < a:
            raise FetchError("fetch")
        self.budget.take_request()
        self.log.append((href, a, z))
        try:
            with self.client.stream(
                "GET",
                href,
                headers={"Range": f"bytes={a}-{z}"},
                timeout=self.budget.timeout(),
            ) as resp:
                # 200 means the range was ignored: never read a whole file.
                if resp.status_code != 206:
                    raise FetchError("fetch")
                m = _CONTENT_RANGE.fullmatch(resp.headers.get("content-range", ""))
                if m is None:
                    raise FetchError("fetch")
                ra, rz, total = (int(v) for v in m.groups())
                if ra != a or rz > z or rz < ra or total <= rz:
                    raise FetchError("fetch")
                want = rz - ra + 1
                chunks, got = [], 0
                for chunk in resp.iter_bytes():
                    got += len(chunk)
                    if got > want:
                        raise FetchError("fetch")
                    self.budget.take_bytes(len(chunk))
                    chunks.append(chunk)
                if got != want:
                    raise FetchError("fetch")
        except httpx.HTTPError as exc:
            raise FetchError("fetch") from exc
        with _cache_lock:
            _sizes[href] = total
        return b"".join(chunks)


class _Remote:
    """A file-like view of one band file for GDAL (through rasterio's opener).
    GDAL's read callback cannot take a Python exception, so a failure is
    recorded on the call and read returns no bytes."""

    def __init__(self, fetch: Fetch, href: str) -> None:
        self._fetch = fetch
        self._href = href
        self._pos = 0
        self.closed = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        try:
            if whence == 0:
                self._pos = offset
            elif whence == 1:
                self._pos += offset
            elif whence == 2:
                self._pos = self._fetch.size(self._href) + offset
        except Exception as exc:  # noqa: BLE001 - recorded, see the class
            self._fetch.fail(exc)
        self._pos = max(0, self._pos)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._fetch.failed is not None:
            return b""
        try:
            total = self._fetch.size(self._href)
            end = total if n is None or n < 0 else min(self._pos + n, total)
            if end <= self._pos:
                return b""
            data = self._fetch.read(self._href, self._pos, end)
        except Exception as exc:  # noqa: BLE001 - recorded, see the class
            self._fetch.fail(exc)
            return b""
        self._pos += len(data)
        return data

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _Remote:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _container(fetch: Fetch, files: dict[str, str]) -> Any:
    """A rasterio FileContainer that serves only ``files`` (name -> href)."""
    abc = importlib.import_module("rasterio.abc")

    class _Bucket(abc.FileContainer):  # type: ignore[name-defined]
        def _href(self, path: str) -> str | None:
            return files.get(str(path).lstrip("/"))

        def open(self, path: str, mode: str = "r", **kwds: Any) -> _Remote:
            href = self._href(path)
            if href is None or any(c in mode for c in "wa+"):
                raise FileNotFoundError(path)
            return _Remote(fetch, href)

        def isfile(self, path: str) -> bool:
            return self._href(path) is not None

        def isdir(self, path: str) -> bool:
            return False

        def ls(self, path: str) -> list[str]:
            return []

        def mtime(self, path: str) -> int:
            return 0

        def size(self, path: str) -> int:
            href = self._href(path)
            if href is None:
                raise FileNotFoundError(path)
            try:
                return fetch.size(href)
            except Exception as exc:  # GDAL turns it into its own error: keep ours
                fetch.fail(exc)
                raise

        def rm(self, path: str) -> None:
            raise PermissionError(path)

    return _Bucket()


_names = itertools.count()
_names_lock = threading.Lock()


def _name() -> str:
    with _names_lock:
        return f"s2band{next(_names)}.tif"


@dataclass(frozen=True)
class Box:
    """A picture on the map's own grid: CRS (WKT or 'EPSG:n'), the affine of
    its top-left pixel (a, b, c, d, e, f), and its size in pixels."""

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    def lonlat(self) -> tuple[float, float, float, float]:
        warp = importlib.import_module("rasterio.warp")
        affine = importlib.import_module("affine")
        t = affine.Affine(*self.transform)
        xs = [(t * (0, 0))[0], (t * (self.width, self.height))[0]]
        ys = [(t * (0, 0))[1], (t * (self.width, self.height))[1]]
        return tuple(
            warp.transform_bounds(
                self.crs,
                "EPSG:4326",
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                densify_pts=21,
            )
        )

    def points(self) -> list[tuple[float, float]]:
        """The picture's four corners and centre, in lon/lat."""
        warp = importlib.import_module("rasterio.warp")
        affine = importlib.import_module("affine")
        t = affine.Affine(*self.transform)
        w, h = self.width, self.height
        pix = [(0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)]
        xy = [t * p for p in pix]
        lons, lats = warp.transform(
            self.crs, "EPSG:4326", [p[0] for p in xy], [p[1] for p in xy]
        )
        return list(zip(lons, lats))


def _read_band(fetch: Fetch, href: str, scene: Scene, box: Box, out: Any) -> None:
    """One file of ``scene`` (a band, or its cloud mask) onto the box's grid,
    written into ``out`` (height, width; 0 where there is no data), by nearest
    neighbour. Raises :class:`FetchError`."""
    rasterio = importlib.import_module("rasterio")
    warp = importlib.import_module("rasterio.warp")
    windows = importlib.import_module("rasterio.windows")
    session_mod = importlib.import_module("rasterio.session")
    affine = importlib.import_module("affine")
    dst = affine.Affine(*box.transform)
    corners = [dst * (0, 0), dst * (box.width, box.height)]
    left, right = sorted(p[0] for p in corners)
    bottom, top = sorted(p[1] for p in corners)
    name = _name()
    opener = _container(fetch, {name: href})
    try:
        # Only the GeoTIFF driver (a VRT or WMS body could make GDAL fetch on
        # its own), no .netrc, no signing, no session from the environment.
        with rasterio.Env(
            session=session_mod.DummySession(),
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_NUM_THREADS="1",
            GDAL_HTTP_NETRC="NO",
            AWS_NO_SIGN_REQUEST="YES",
        ):
            with rasterio.open(name, driver="GTiff", opener=opener) as src:
                if src.crs is None or src.crs.to_epsg() != scene.epsg:
                    raise FetchError("fetch")
                b = warp.transform_bounds(
                    box.crs, src.crs, left, bottom, right, top, densify_pts=21
                )
                w = windows.from_bounds(*b, transform=src.transform)
                col0 = max(0, math.floor(w.col_off) - 2)
                row0 = max(0, math.floor(w.row_off) - 2)
                col1 = min(src.width, math.ceil(w.col_off + w.width) + 3)
                row1 = min(src.height, math.ceil(w.row_off + w.height) + 3)
                if col1 <= col0 or row1 <= row0:
                    return  # the file does not reach the picture: no data
                win = windows.Window(col0, row0, col1 - col0, row1 - row0)
                arr = src.read(1, window=win)
                if fetch.failed is not None:
                    raise FetchError("fetch")
                warp.reproject(
                    source=arr,
                    destination=out,
                    src_transform=src.window_transform(win),
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=dst,
                    dst_crs=box.crs,
                    dst_nodata=0,
                    resampling=warp.Resampling.nearest,
                )
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001 - rasterio's error after a failed fetch
        if isinstance(fetch.failed, FetchError):
            raise fetch.failed from exc
        raise FetchError("fetch") from exc
    if fetch.failed is not None:
        raise FetchError("fetch") from fetch.failed


def read_box(fetch: Fetch, scene: Scene, box: Box) -> Any:
    """The four bands of ``scene`` on the box's grid (uint16, 0 where there is
    no data), by nearest neighbour. Raises :class:`FetchError`."""
    np = importlib.import_module("numpy")
    out = np.zeros((len(BANDS), box.height, box.width), dtype=np.uint16)
    for k, band in enumerate(BANDS):
        _read_band(fetch, scene.hrefs[band], scene, box, out[k])
    return out


def read_mask(fetch: Fetch, scene: Scene, box: Box) -> Any:
    """The cloud mask of ``scene`` (:data:`MASK`, its Scene Classification
    Layer) on the box's grid: a uint8 array (box.height, box.width) of SCL
    classes, 0 where there is no data, by nearest neighbour, read as one band
    of :func:`read_box` is. Raises :class:`FetchError` (also for a scene with
    no mask link)."""
    np = importlib.import_module("numpy")
    if not scene.mask_href:
        raise FetchError("fetch")
    out = np.zeros((box.height, box.width), dtype=np.uint8)
    _read_band(fetch, scene.mask_href, scene, box, out)
    return out


def cover(mask: Any, square: tuple[int, int, int, int]) -> tuple[float, float]:
    """How much of the outlined square (x, y, w, h) a cloud mask marks:
    (covered share, valid share). Covered: the pixels of
    :data:`COVERED_CLASSES` over the pixels with data (SCL != 0); valid: the
    pixels with data over all the square's. No pixel with data: (1.0, 0.0),
    so a hole in the mask never reads as clear."""
    np = importlib.import_module("numpy")
    x, y, w, h = square
    part = mask[y : y + h, x : x + w]
    valid = int((part != 0).sum()) if part.size else 0
    if not valid:
        return 1.0, 0.0
    covered = int(np.isin(part, COVERED_CLASSES).sum())
    return covered / valid, valid / int(part.size)


def has_data(stack: Any, square: tuple[int, int, int, int]) -> bool:
    """Whether at least :data:`DATA_SHARE` of the outlined square (x, y, w, h)
    has data in all four bands (DN 0 in any band is no data)."""
    x, y, w, h = square
    part = stack[:, y : y + h, x : x + w]
    if not part.size:
        return False
    return float((part != 0).all(axis=0).mean()) >= DATA_SHARE


def white_share(stack: Any, square: tuple[int, int, int, int]) -> float:
    """The share of the outlined square's pixels with data (all four bands)
    that the picture itself shows white: red, green and blue reflectance all
    at least :data:`BRIGHT` (thick cloud or fog; clear water, plants and most
    bare ground stay well below). 0.0 when no pixel has data."""
    np = importlib.import_module("numpy")
    x, y, w, h = square
    part = stack[:, y : y + h, x : x + w]
    data = (part != 0).all(axis=0)
    n = int(data.sum()) if part.size else 0
    if not n:
        return 0.0
    idx = [BANDS.index(b) for b in ("red", "green", "blue")]
    refl = (part[idx].astype(np.float32) - 1000.0) / 10000.0
    white = (refl >= BRIGHT).all(axis=0) & data
    return int(white.sum()) / n


def mask_fetch(fetch: Fetch) -> Fetch:
    """The cloud masks' own fetching for one call: the band fetch's client,
    and a budget of its own (15 seconds, 32 MiB, 250 requests: up to 72 masks
    a call, up to 3 requests each), so a mask read never spends the bands'
    bytes or requests and never fails their fetch. The time the masks' pass
    takes is added back to the bands' deadline (:func:`dated`)."""
    budget = Budget(seconds=MASK_SECONDS, max_bytes=32 << 20, max_requests=250)
    return Fetch(fetch.client, budget)


def _room(budget: Budget) -> bool:
    """Whether a read may still start on ``budget`` (time, bytes, requests)."""
    return (
        time.monotonic() < budget.deadline
        and budget.requests < budget.max_requests
        and budget.bytes < budget.max_bytes
    )


def _score(
    masks: Fetch, scene: Scene, box: Box, square: tuple[int, int, int, int]
) -> tuple[float, float] | None:
    """:func:`cover` of one scene's cloud mask on the square, or ``None`` when
    the mask could not be read (the scene is then ranked as one without a
    mask). Each mask is read through a fetch of its own on the masks' client
    and budget, so one mask that fails stops no other read; the masks' fetch
    keeps the first failure (``masks.failed``) and every request
    (``masks.log``)."""
    one = Fetch(masks.client, masks.budget)
    one.log = masks.log
    try:
        return cover(read_mask(one, scene, box), square)
    except Exception as exc:  # noqa: BLE001 - a mask never fails the pictures
        masks.fail(exc)
        return None


def _mask_info(
    found: list[Scene],
    scores: dict[int, tuple[float, float]],
    winner: int | None,
    white: float | None = None,
) -> dict[str, Any]:
    """What the masks said, for the record and the pane: ``here`` for the
    scene picked (``"clear"``: the mask marks nothing on the square;
    ``"covered"``: it marks cloud or shadow there, or has data on less than
    :data:`MASK_VALID` of it, or the picture is white on at least
    :data:`BRIGHT_SHARE` of the square (``white_share``, :func:`white_share`)
    whatever the mask says; ``"unknown"``: its mask was not read and the
    picture is not white, or no scene was picked), its shares, and the score
    of every mask read."""
    score = scores.get(winner) if winner is not None else None
    here = "unknown"
    if score is not None:
        here = "covered" if score[0] > 0 or score[1] < MASK_VALID else "clear"
    if white is not None and white >= BRIGHT_SHARE:
        here = "covered"
    return {
        "here": here,
        "white_share": white,
        "covered_share": None if score is None else score[0],
        "valid_share": None if score is None else score[1],
        "scored": [
            {"id": found[i].id, "covered": scores[i][0], "valid": scores[i][1]}
            for i in sorted(scores)
        ],
    }


#: A month between its two halves: its candidates, the score of each mask
#: read (by the candidate's index), and the indices whose mask read was made.
_Scored = tuple[list[Scene], dict[int, tuple[float, float]], set[int]]


def _clear(score: tuple[float, float] | None) -> bool:
    """Whether a mask's score says it marks nothing on the square and has
    data on at least :data:`MASK_VALID` of it."""
    return score is not None and score[0] == 0 and score[1] >= MASK_VALID


def _score_more(
    fetch: Fetch,
    masks: Fetch,
    state: _Scored,
    box: Box,
    square: tuple[int, int, int, int],
) -> None:
    """Read the masks of the month's candidates not read yet, in their order
    (the first :data:`MASK_TRIES` candidates only), and stop at the first that
    marks nothing on the square. A candidate with no mask link is passed
    over; no read starts once the band fetch has failed or the masks' budget
    has no room; a read that fails leaves its scene unscored and is not made
    again."""
    found, scores, asked = state
    for i, scene in enumerate(found[:MASK_TRIES]):
        if i in asked or not scene.mask_href:
            continue
        if fetch.failed is not None or not _room(masks.budget):
            break
        asked.add(i)
        got = _score(masks, scene, box, square)
        if got is None:
            continue
        scores[i] = got
        if _clear(got):
            break  # nothing marked on the square: it wins


def _month_masks(
    fetch: Fetch,
    masks: Fetch,
    scenes: list[Scene],
    box: Box,
    square: tuple[int, int, int, int],
    tries: int = TRIES_PER_MONTH,
) -> _Scored:
    """The first half of :func:`month_stack`: the month's candidates and
    their masks' scores, read until one marks nothing on the square
    (:func:`_score_more`). No band is read."""
    found = candidates(scenes, box.lonlat(), box.points())[: max(MASK_TRIES, tries)]
    state: _Scored = (found, {}, set())
    _score_more(fetch, masks, state, box, square)
    return state


def _month_bands(
    fetch: Fetch,
    masks: Fetch,
    state: _Scored,
    box: Box,
    square: tuple[int, int, int, int],
    tries: int = TRIES_PER_MONTH,
) -> tuple[Scene | None, Any, int, dict[str, Any]]:
    """The second half of :func:`month_stack`: the bands, read in the masks'
    rank, at most ``tries`` scenes; the first with data in the square whose
    picture is not white there (:func:`white_share` under
    :data:`BRIGHT_SHARE`) wins. When a scene its mask shows clear has too
    little data in the square, or a scene's picture is white on the square
    (Sen2Cor's mask can miss thick cloud right at a coastline), the masks'
    reading resumes (:func:`_score_more`) and the scenes left are ranked
    again (after a white scene, only when its mask had called it clear), so
    no scene is picked unscored while its mask can be read. No further read
    for whiteness starts once the band budget is short (:func:`_band_room`).
    When every scene read with data is white on the square, the least cloudy
    of them by both signals (the larger of its white share and its mask's
    covered share; the first on a tie) is picked, ``"covered"``."""
    found, scores, _ = state
    # The fallback so far: (its cloud by both signals, index, stack, white share).
    whitest: tuple[float, int, Any, float] | None = None

    def rank(i: int) -> tuple[int, float, int]:
        """Clear scenes, then the unscored in their order, then the rest."""
        if i not in scores:
            return (1, 0.0, i)
        if _clear(scores[i]):
            return (0, 0.0, i)
        return (2, scores[i][0], i)

    left = set(range(len(found)))
    tried = 0
    while left and tried < tries:
        i = min(left, key=rank)
        left.discard(i)
        tried += 1
        stack = read_box(fetch, found[i], box)
        if has_data(stack, square):
            white = white_share(stack, square)
            if white < BRIGHT_SHARE:
                return found[i], stack, tried, _mask_info(found, scores, i, white)
            # The fallback: the least cloud by both signals, the first on a tie
            # (the least white alone would favour a darker, shadowed scene).
            worst = max(white, scores[i][0] if i in scores else 0.0)
            if whitest is None or worst < whitest[0]:
                whitest = (worst, i, stack, white)
            if not _band_room(fetch.budget):
                break  # no further read for whiteness on a short budget
            if _clear(scores.get(i)):
                _score_more(fetch, masks, state, box, square)
            continue
        if _clear(scores.get(i)):
            _score_more(fetch, masks, state, box, square)
    if whitest is not None:
        _, i, stack, white = whitest
        return found[i], stack, tried, _mask_info(found, scores, i, white)
    return None, None, tried, _mask_info(found, scores, None)


def _band_room(budget: Budget) -> bool:
    """Whether another band read for a white scene may start: time left, and
    under :data:`WHITE_ROOM` of the band budget's requests and bytes spent (a
    spent band budget fails the whole call)."""
    return (
        time.monotonic() < budget.deadline
        and budget.requests < budget.max_requests * WHITE_ROOM
        and budget.bytes < budget.max_bytes * WHITE_ROOM
    )


def month_stack(
    fetch: Fetch,
    scenes: list[Scene],
    box: Box,
    square: tuple[int, int, int, int],
    *,
    tries: int = TRIES_PER_MONTH,
    masks: Fetch | None = None,
) -> tuple[Scene | None, Any, int, dict[str, Any]]:
    """The scene of the month with the least cloud and shadow on the outlined
    square by its own cloud mask, with data in the square: (scene, stack,
    scenes whose bands were tried, what the masks said, :func:`_mask_info`).

    The month's candidates (least scene cloud first) have their masks read,
    at most :data:`MASK_TRIES`, until one marks nothing on the square. They
    are then ranked: a scene whose mask marks nothing first; then the scenes
    with no mask read (no link, no room left on the masks' budget, a read
    that failed), in the candidates' order; then the others, least covered
    first. Bands are read in that order, at most ``tries`` scenes, and the
    first with data in the square whose picture is not white there wins
    (:func:`white_share`); when the scene its mask shows clear
    has too little data there, mask reading resumes before the next band
    read. With no mask anywhere this is the least-scene-cloud rule. The time
    the masks take (at most :data:`MASK_SECONDS`) is added to the bands'
    deadline. ``masks``: the masks' own fetch (:func:`mask_fetch`, made here
    when not given). Blind to the window's
    class, stratum and order."""
    masks = masks if masks is not None else mask_fetch(fetch)
    began = time.monotonic()
    state = _month_masks(fetch, masks, scenes, box, square, tries)
    fetch.budget.extend(min(time.monotonic() - began, MASK_SECONDS))
    return _month_bands(fetch, masks, state, box, square, tries)


def dated(
    fetch: Fetch,
    scenes_by_month: list[list[Scene]],
    box: Box,
    square: tuple[int, int, int, int],
    *,
    workers: int = 4,
    masks: Fetch | None = None,
) -> list[tuple[Scene | None, Any, int, dict[str, Any]]]:
    """:func:`month_stack` for every month, in month order, in two passes:
    first every month's masks (up to :data:`MASK_WORKERS` months at a time),
    so which months are scored does not depend on when a month's bands start;
    then the bands, ``workers`` months at a time. The time the first pass
    took (at most :data:`MASK_SECONDS`) is added to the bands' deadline: the
    masks never take the bands' time. Any failure of the band reads fails the
    whole call (no label on a partial set) and stops the mask reads too; a
    cloud mask that cannot be read fails nothing. ``masks``: the masks' own
    fetch, one for all the months (:func:`mask_fetch`, made here when not
    given)."""
    if not scenes_by_month:
        return []
    masks = masks if masks is not None else mask_fetch(fetch)
    futures_mod = importlib.import_module("concurrent.futures")
    count = len(scenes_by_month)
    scoring = futures_mod.ThreadPoolExecutor(max_workers=min(count, MASK_WORKERS))
    pool = futures_mod.ThreadPoolExecutor(max_workers=max(1, min(workers, count)))
    try:
        began = time.monotonic()
        jobs = [
            scoring.submit(_month_masks, fetch, masks, s, box, square)
            for s in scenes_by_month
        ]
        states = [job.result() for job in jobs]
        fetch.budget.extend(min(time.monotonic() - began, MASK_SECONDS))
        jobs = [
            pool.submit(_month_bands, fetch, masks, state, box, square)
            for state in states
        ]
        return [job.result() for job in jobs]
    except BaseException as exc:
        # The other months' band reads stop at their next read, and the mask
        # reads at their next request or chunk.
        fetch.fail(exc)
        masks.budget.stop()
        raise
    finally:
        scoring.shutdown(wait=True, cancel_futures=True)
        pool.shutdown(wait=True, cancel_futures=True)


# ------------------------------------------------------------------ rendering

#: Reflectance of the pinned collection (sentinel-2-c1-l2a): a constant, never
#: the item's own scale field. DN 0 is no data.
REFLECTANCE = "(DN - 1000) / 10000; DN 0 in any band is no data"


def to_rgb(stack: Any, bands: tuple[str, str, str], top: float) -> Any:
    """An (h, w, 3) uint8 picture: three bands, reflectance 0..top, linear.
    No-data pixels are magenta (not white, which reads as cloud)."""
    np = importlib.import_module("numpy")
    idx = [BANDS.index(b) for b in bands]
    refl = (stack[idx].astype(np.float32) - 1000.0) / 10000.0
    rgb = np.clip(refl / top, 0.0, 1.0) * 255.0 + 0.5
    rgb = rgb.astype(np.uint8).transpose(1, 2, 0)
    nodata = (stack == 0).any(axis=0)
    rgb[nodata] = (255, 0, 255)
    return rgb
