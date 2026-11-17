# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The Sentinel-2 fetch's rules, each on its own: exact hosts, no redirects,
ranges read exactly, search paging, the period's months, the data share and
the fixed stretch. No test touches the network (a mock transport)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from olmoearth_agent.mcp_app import s2_fetch

np = pytest.importorskip("numpy")

BUCKET = f"https://{s2_fetch.BUCKET_HOST}"
HREF = f"{BUCKET}/sentinel-2-c1-l2a/a/B04.tif"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a test reached for the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    s2_fetch.clear_cache()


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (HREF, True),
        (HREF.replace("https", "http"), False),
        (HREF.replace(".com/", ".com:443/"), False),
        (HREF.replace("https://", "https://user@"), False),
        (HREF.replace(".com/", ".com.evil.example/"), False),
        (HREF + "?x=1", False),
        (HREF + "#f", False),
        ("https://evil.example/" + s2_fetch.BUCKET_HOST + "/a.tif", False),
    ],
)
def test_only_https_on_exactly_the_host(url: str, ok: bool) -> None:
    assert s2_fetch.allowed(url, s2_fetch.BUCKET_HOST) is ok


def _feature(**over: Any) -> dict[str, Any]:
    feature = {
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
        "assets": {b: {"href": f"{BUCKET}/x/{b}.tif"} for b in s2_fetch.BANDS},
    }
    feature.update(over)
    return feature


def test_a_scene_needs_every_field_and_bucket_links() -> None:
    scene = s2_fetch.scene_of(_feature())
    assert scene is not None and scene.epsg == 32632 and scene.footprint
    assert s2_fetch.Scene.from_record(scene.as_record()) == scene
    other = _feature(
        assets={b: {"href": f"https://evil.example/{b}.tif"} for b in s2_fetch.BANDS}
    )
    assert s2_fetch.scene_of(other) is None
    assert (
        s2_fetch.scene_of(_feature(properties={"datetime": "2024-04-10T10:00:00Z"}))
        is None
    )
    code = _feature()
    del code["properties"]["proj:epsg"]
    code["properties"]["proj:code"] = "EPSG:32632"
    assert s2_fetch.scene_of(code).epsg == 32632  # type: ignore[union-attr]
    # The footprint decides, not only the bbox.
    assert scene.covers([(10.5, 50.5)]) and not scene.covers([(10.5, 51.5)])


def test_candidates_least_cloud_one_per_datatake_and_grid() -> None:
    a = s2_fetch.scene_of(
        _feature(id="A", properties={**_feature()["properties"], "eo:cloud_cover": 9.0})
    )
    b = s2_fetch.scene_of(
        _feature(
            id="B",
            properties={
                **_feature()["properties"],
                "eo:cloud_cover": 1.0,
                "s2:datatake_id": "E",
            },
        )
    )
    dup = s2_fetch.scene_of(
        _feature(
            id="A2",
            properties={
                **_feature()["properties"],
                "eo:cloud_cover": 0.1,
                "s2:processing_baseline": "05.11",
            },
        )
    )
    out = s2_fetch.candidates([a, b, dup], (10.2, 50.2, 10.4, 50.4), [(10.3, 50.3)])  # type: ignore[list-item]
    assert [s.id for s in out] == [
        "A2",
        "B",
    ]  # A and A2 share datatake and grid: the newer baseline
    assert s2_fetch.candidates([a], (10.2, 50.2, 11.4, 50.4)) == []  # type: ignore[list-item]


def test_the_period_and_its_months() -> None:
    start = s2_fetch.parse_moment("2024-04-01T00:00:00Z", end=False)
    end = s2_fetch.parse_moment("2024-06-30T23:59:59.999000Z", end=True)
    assert [m[0] for m in s2_fetch.months(start, end)] == [
        "Apr 2024",
        "May 2024",
        "Jun 2024",
    ]
    day_end = s2_fetch.parse_moment("2023-12-31", end=True)
    assert day_end == datetime(2023, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
    months = s2_fetch.months(s2_fetch.parse_moment("2023-11-15", end=False), day_end)
    assert [m[0] for m in months] == ["Nov 2023", "Dec 2023"]
    assert months[0][1].day == 15  # clipped to the period
    assert s2_fetch.rounded((10.03, 50.07, 10.11, 50.12)) == (10.0, 50.0, 10.2, 50.2)
    with pytest.raises(ValueError):
        s2_fetch.rounded((179.9, 0, -179.9, 1))


def test_search_pages_only_while_the_answer_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s2_fetch, "SEARCH_LIMIT", 1)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        page = 2 if body.get("next") else 1
        nxt = {
            "rel": "next",
            "method": "POST",
            "href": s2_fetch.SEARCH_URL,
            "merge": False,
            "body": {"next": "t2"},
        }
        feats = [_feature(id=f"P{page}")]
        return httpx.Response(
            200, json={"features": feats, "numberMatched": 2, "links": [nxt]}
        )

    t = datetime(2024, 4, 1, tzinfo=timezone.utc)
    with _client(handler) as client:
        scenes, asked = s2_fetch.search_month(
            client, s2_fetch.Budget(), (10, 50, 11, 51), t, t
        )
    assert (
        [s.id for s in scenes] == ["P1", "P2"]
        and asked["pages"] == 2
        and asked["complete"]
    )
    assert bodies[1] == {"next": "t2"}  # merge false: the link's body replaces ours

    def elsewhere(request: httpx.Request) -> httpx.Response:
        nxt = {
            "rel": "next",
            "method": "POST",
            "href": "https://evil.example/search",
            "body": {},
        }
        return httpx.Response(
            200, json={"features": [_feature()], "numberMatched": 5, "links": [nxt]}
        )

    with _client(elsewhere) as client, pytest.raises(s2_fetch.FetchError) as caught:
        s2_fetch.search_month(client, s2_fetch.Budget(), (10, 50, 11, 51), t, t)
    assert caught.value.kind == "catalogue"


@pytest.mark.parametrize(
    "answer",
    [
        lambda a, z: httpx.Response(200, content=b"x" * 10),  # range ignored
        lambda a, z: httpx.Response(
            206,
            content=b"x" * (z - a + 1),
            headers={"content-range": f"bytes {a + 1}-{z}/100"},
        ),
        lambda a, z: httpx.Response(
            206,
            content=b"x" * (z - a + 2),
            headers={"content-range": f"bytes {a}-{z}/100000"},
        ),
        lambda a, z: httpx.Response(
            206,
            content=b"x" * (z - a),
            headers={"content-range": f"bytes {a}-{z}/100000"},
        ),
        lambda a, z: httpx.Response(
            302, headers={"location": "https://evil.example/x"}
        ),
    ],
)
def test_a_range_is_read_exactly_or_not_at_all(answer: Any) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        a, z = (
            int(v) for v in request.headers["range"].removeprefix("bytes=").split("-")
        )
        return answer(a, z)

    with _client(handler) as client:
        fetch = s2_fetch.Fetch(client, s2_fetch.Budget())
        with pytest.raises(s2_fetch.FetchError):
            fetch.read(HREF, 0, 10)
    assert seen == [HREF]  # one request, never a redirect followed


def test_reads_are_cached_and_budgeted() -> None:
    data = bytes(range(256)) * 1024  # 256 KiB

    def handler(request: httpx.Request) -> httpx.Response:
        a, z = (
            int(v) for v in request.headers["range"].removeprefix("bytes=").split("-")
        )
        z = min(z, len(data) - 1)
        return httpx.Response(
            206,
            content=data[a : z + 1],
            headers={"content-range": f"bytes {a}-{z}/{len(data)}"},
        )

    with _client(handler) as client:
        budget = s2_fetch.Budget()
        fetch = s2_fetch.Fetch(client, budget)
        assert fetch.read(HREF, 70000, 70010) == data[70000:70010]
        assert fetch.size(HREF) == len(data)
        n = budget.requests
        assert (
            fetch.read(HREF, 70002, 70008) == data[70002:70008] and budget.requests == n
        )
        tight = s2_fetch.Fetch(client, s2_fetch.Budget(max_requests=0))
        with pytest.raises(s2_fetch.FetchError) as caught:
            tight.read(f"{BUCKET}/other.tif", 0, 4)
        assert caught.value.kind == "budget"


def test_the_data_share_and_the_stretch() -> None:
    stack = np.full((4, 11, 11), 3000, dtype=np.uint16)
    assert s2_fetch.has_data(stack, (0, 0, 11, 11))
    stack[0, 0, :11] = 0  # 11 of 121 pixels: 91% left
    assert s2_fetch.has_data(stack, (0, 0, 11, 11))
    stack[0, 1, :2] = 0  # 13 of 121: under 90%
    assert not s2_fetch.has_data(stack, (0, 0, 11, 11))
    one = np.array([1000, 2000, 3000, 0], dtype=np.uint16).reshape(4, 1, 1)
    # DN 1000 is reflectance 0; DN 3000 is 0.2, the top of the true-colour range.
    assert s2_fetch.to_rgb(
        np.full((4, 1, 1), 1000, dtype=np.uint16), ("red", "green", "blue"), 0.2
    )[0, 0].tolist() == [0, 0, 0]
    assert s2_fetch.to_rgb(
        np.full((4, 1, 1), 3000, dtype=np.uint16), ("red", "green", "blue"), 0.2
    )[0, 0].tolist() == [255, 255, 255]
    assert s2_fetch.to_rgb(one, ("nir", "red", "green"), 0.3)[0, 0].tolist() == [
        255,
        0,
        255,
    ]  # no data: magenta


def test_the_real_client_follows_no_redirect_and_ignores_the_environment() -> None:
    client = s2_fetch._default_client()
    try:
        assert client.follow_redirects is False and client.trust_env is False
        assert "authorization" not in client.headers and not client.cookies
    finally:
        client.close()


def test_a_period_ending_on_the_first_of_a_month_does_not_add_it() -> None:
    year = s2_fetch.months(
        s2_fetch.parse_moment("2024-01-01", end=False),
        s2_fetch.parse_moment("2025-01-01", end=True),
    )
    assert len(year) == 12 and year[-1][0] == "Dec 2024"
    studio = s2_fetch.months(
        s2_fetch.parse_moment("2024-04-01T00:00:00Z", end=False),
        s2_fetch.parse_moment("2024-07-01T00:00:00Z", end=True),
    )
    assert [m[0] for m in studio] == ["Apr 2024", "May 2024", "Jun 2024"]
