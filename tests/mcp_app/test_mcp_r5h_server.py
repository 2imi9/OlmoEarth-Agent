# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-h, track SERVER: a large map's Check card prepares in the
background (H1, the server's side) and a read Studio result is kept on this
computer (H2).

H1 runs the server's background path with a fake slow preparation: the
Studio files tool (``app.run_agent_tool``) answers only once an event is
set, so a call's wait (``OLMOEARTH_CHECK_WAIT_SECONDS``, set small here)
ends first. H2 runs ``olmoearth_scores_from_studio``'s handler on a fake
Studio client that counts its downloads; the zip's decode and the pooling
are stand-ins too, so these tests need neither the package nor a raster.
Nothing here reaches Studio.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import random
import stat
import threading
import time
import types as pytypes
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.harness.state import ThreadState  # noqa: E402
from olmoearth_agent.mcp_app import auth, check, server, views  # noqa: E402
from olmoearth_agent.tools import studio_scores  # noqa: E402
from olmoearth_agent.tools.registry import ToolContext  # noqa: E402

BASE = "http://mock-studio/api/v1"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
FILES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")
MUST = [
    "Each UTM zone is its own map: one zone's estimate is not the result's.",
    "These are Studio's native-resolution files.",
]
CLAIMS = [{"claim": "the result is certified", "why": "nothing certifies it"}]
#: A map of presence and absence: the owner's labelling rule is set.
CLASSES = [
    {"channel": 0, "value": 0, "label": "absent"},
    {"channel": 1, "value": 1, "label": "present"},
]
CANNED = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty labels.", "rate_note": "r50"},
        {"labels": 40, "width": 0.1, "sentence": "Forty labels.", "rate_note": "r50"},
    ],
    "refused": None,
}
#: A download that timed out, as the registry writes the files tool's error.
READ_FAILED = (
    "RuntimeError: GET /prediction-results/files failed (ReadTimeout); "
    "no file was read"
)
#: The time a kept copy was downloaded, and the Check card's line for it.
COPIED = "2026-11-18T10:04:00Z"
COPY_LINE = "Read from this computer's copy of 18 Nov 2026, 10:04 UTC."


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The output root (and so the scores root) of this test's own; no key."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    monkeypatch.delenv(server.CHECK_WAIT_ENV, raising=False)
    monkeypatch.delenv(studio_scores.KEEP_HOURS_ENV, raising=False)
    return out.resolve()


@pytest.fixture
def canned_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts of 20 and 40 offered for every zone, without the package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`CANNED`."""
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", fake)


def _app() -> server.AppServer:
    """A server with the default registry, signed out, on a mock base URL."""
    return server.AppServer(keys=auth.KeyStore(), base_url=BASE)


async def _list(srv: Any) -> list[types.Tool]:
    """The tools the server lists."""
    result = await srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    tools: list[types.Tool] = result.root.tools
    return tools


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> Any:
    """One tool call through the server's handler."""
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await srv.request_handlers[types.CallToolRequest](request)
    return result.root


def _text(result: Any) -> str:
    """Every text block of a tool's answer, joined."""
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _view(result: Any) -> dict[str, Any]:
    """A tool answer's view data."""
    view: dict[str, Any] = (result.meta or {})[server.VIEW_DATA_KEY]
    return view


def _model(result: Any) -> dict[str, Any]:
    """The model's copy of a Check card's answer: its last text block."""
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    body: dict[str, Any] = json.loads(texts[-1])
    return body


def _logits(n: int, seed: int) -> list[list[float]]:
    """Two-class logits for ``n`` windows of a zone."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.5) for _ in range(2)] for _ in range(n)]


def _studio_answer(root: Path, copied: str | None = None) -> dict[str, Any]:
    """olmoearth_scores_from_studio's envelope for a result of two UTM zones,
    each zone's scores file written under the root, as the tool writes it;
    ``copied``: read from the copy this computer kept, downloaded then."""
    zones = []
    for i, (file, n) in enumerate(zip(FILES, (300, 200))):
        path = root / f"zone{i}_scores.json"
        path.write_text(json.dumps({"scores": _logits(n, 4 + i)}), "utf-8")
        zones.append(
            {
                "file": file,
                "scores_path": str(path),
                "patch": 4,
                "valid_pixels": 16 * n,
                "pooled": {
                    "n_valid": n,
                    "n_windows": n,
                    "raster_check": {"sha256": str(i) * 64},
                },
            }
        )
    shares = [{"file": f, "share": s} for f, s in zip(FILES, (0.6, 0.4))]
    inner: dict[str, Any] = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": "top_k",
        "property_name": "landcover",
        "classes": copy.deepcopy(CLASSES),
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "n_zones": len(zones),
        "zones": zones,
        "facts": [{"id": "studio_zones", "sentence": "Two.", "zones": shares}],
        "must_state": list(MUST),
        "forbidden_claims": copy.deepcopy(CLAIMS),
    }
    if copied is not None:
        inner["read_from_copy"] = {
            "downloaded_at": copied,
            "note": studio_scores.KEPT_NOTE.format(when=copied),
        }
    return {"ok": True, "result": inner}


class SlowFiles:
    """``app.run_agent_tool`` for the Studio files tool: each call reports
    ``steps`` (from a worker thread, as the decode does) and sets
    ``reported``, then answers once ``release`` is set, with ``answers`` in
    turn (the last for every later call); every call's arguments noted, and
    the Studio client its preparation reads with (``clients``)."""

    def __init__(
        self,
        app: server.AppServer,
        monkeypatch: pytest.MonkeyPatch,
        answers: list[dict[str, Any]],
        *,
        released: bool = False,
    ) -> None:
        """Put the fake in place of ``app.run_agent_tool``."""
        self.calls: list[dict[str, Any]] = []
        self.clients: list[Any] = []
        self.answers = answers
        self.steps: list[tuple[str, int, int]] = []
        self.reported = asyncio.Event()
        self.release = asyncio.Event()
        if released:
            self.release.set()
        monkeypatch.setattr(app, "run_agent_tool", self.run)

    async def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The files tool's envelope, once released; no other tool is called."""
        assert name == server.SCORES_TOOL, name
        self.calls.append(dict(arguments))
        held = server._PREPARATION_READ.get()
        self.clients.append(held[0] if held is not None else None)
        for step in self.steps:
            await asyncio.to_thread(studio_scores.report_progress, *step)
        self.reported.set()
        await self.release.wait()
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        return copy.deepcopy(answer)


async def _ended(app: server.AppServer, rid: str = R) -> None:
    """Wait, at most 5 s, until the preparation of ``rid`` has ended."""
    task = app._preparations[rid].task
    await asyncio.wait({task}, timeout=5)
    assert task.done()


# --------------------------------------------------------------------------- H1


def test_h1_the_wait_is_20_seconds_unless_the_environment_says_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CHECK_WAIT_SECONDS, 20; the environment variable when it holds a number
    of 0 or more; anything else leaves the 20."""
    assert server.CHECK_WAIT_SECONDS == 20.0
    assert server._check_wait_seconds() == 20.0
    for raw, seconds in (("5", 5.0), ("0", 0.0), ("0.5", 0.5)):
        monkeypatch.setenv(server.CHECK_WAIT_ENV, raw)
        assert server._check_wait_seconds() == seconds
    for raw in ("-1", "soon", "nan", "inf", " "):
        monkeypatch.setenv(server.CHECK_WAIT_ENV, raw)
        assert server._check_wait_seconds() == 20.0


@pytest.mark.asyncio
async def test_h1_a_preparation_that_ends_in_time_answers_as_before(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Ended within the wait: the card's view and Claude's text as before
    round r5-h (no preparing key, the card's note, each zone with its counts,
    no copy line for files downloaded now), and the next call prepares the
    map again, so each call reads the files once, as before."""
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)], released=True)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is False
    view = _view(result)
    assert (view["tool"], view["ok"]) == (server.CHECK_TOOL, True)
    assert view["result_id"] == R
    assert not {"preparing", "stage", "read_from_copy"} & set(view)
    counts = [[o["labels"] for o in z["offer"]] for z in view["zones"]]
    assert counts == [[20, 40], [20, 40]]
    model = _model(result)
    assert model["note"].startswith(server.CHECK_CARD_NOTE)
    assert not {"preparing", "stage", "read_from_copy"} & set(model)
    assert model["must_state"] == MUST
    assert files.calls == [{"result_id": R, "clip_out_of_range": True}]
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert _view(again) == view and _model(again) == model
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_slow_preparation_answers_preparing_then_the_status_gives_the_card(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Not ended within the wait: a card that is preparing (its stage, the id
    as given, no zone, no path) and Claude's short note with the rules that
    hold for any map; the status says preparing with the same stage, then
    ready with the view data a call that ends in time gets. A ready status
    is not answered from: Claude's next call answers from the finished read
    at once (no read, no wait), as a call in time answers (the card's note,
    must_state), its offers read again, so a part drawn since shows as
    drawn; the call after that prepares the map again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    files.steps = [(studio_scores.STEP_DOWNLOAD, 0, 0)]
    first = await app.check_map_tool({"result_id": f"  {R} "}, views=True)
    assert first.isError is False
    # The download's step may or may not have been told yet.
    stage = _view(first)["stage"]
    assert stage in (server.CHECK_STAGE_START, server.CHECK_STAGE_DOWNLOAD)
    assert _view(first) == {
        "tool": server.CHECK_TOOL,
        "ok": True,
        "needs_sign_in": False,
        "preparing": True,
        "result_id": R,
        "stage": stage,
    }
    assert _model(first) == {
        "ok": True,
        "preparing": True,
        "result_id": R,
        "stage": stage,
        "note": server.CHECK_PREPARING_NOTE + " " + server.CHECK_PREPARING_RULES,
    }
    note = " ".join(_model(first)["note"].split())
    # Round r5-i B10: no call while the card reads; one more only where the
    # card did not hand Claude the contract (test_mcp_r5i_server).
    assert "Do not call olmoearth_check_map again for this result" in note
    assert "call olmoearth_check_map once" not in note
    assert "fills in by itself" in note
    # The blind rule and the never-combine line hold before the card fills in.
    assert "Do not tell the user any window's map class" in note
    assert "never give one rate for all parts" in note
    assert note.endswith(server.CONTEXT_RULE)
    text = _text(first)
    for left_out in ("scores_path", str(_roots), "check_key", "zone0"):
        assert left_out not in text
    await asyncio.wait_for(files.reported.wait(), timeout=5)
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status == {"state": "preparing", "stage": server.CHECK_STAGE_DOWNLOAD}
    joined = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(joined)["stage"] == server.CHECK_STAGE_DOWNLOAD
    files.release.set()
    await _ended(app)
    ready = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    # Round r5-i B10: and the text a call in time gives Claude.
    assert ready["state"] == "ready"
    assert set(ready) == {"state", "view", "model_text"}
    assert "preparing" not in ready["view"]
    assert len(files.calls) == 1
    shown = ready["view"]["zones"]
    assert [z["state"] for z in shown] == ["to draw", "to draw"]
    # The user draws the first part on the card before Claude's next call.
    key = shown[0]["check_key"]
    drawn = {
        "state": "drawn",
        "draw": {
            "pressed_at": "2026-11-18T10:30:00Z",
            "budget": 20,
            "design": check.DESIGN,
            "seed": check.SEED,
            "labels_csv_path": str(_roots / "zone0_sheet.csv"),
            "facts": [],
            "must_state": ["A drawn part's plan line."],
            "forbidden_claims": [],
            "progress": "0 of 20 squares labelled",
        },
    }
    real = check._zone_state

    def zone_state(folder: Path, k: str, digest: Any) -> dict[str, Any]:
        """The first part drawn; any other as on disk."""
        return copy.deepcopy(drawn) if k == key else real(folder, k, digest)

    monkeypatch.setattr(check, "_zone_state", zone_state)
    # Still a wait of 0.05 s: the call answers from the finished read.
    later = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 1
    card = _view(later)
    assert "preparing" not in card
    assert [z["state"] for z in card["zones"]] == ["drawn", "to draw"]
    assert card["zones"][1] == shown[1]
    model = _model(later)
    assert model["note"].startswith(server.CHECK_CARD_NOTE)
    assert model["zones"][0]["must_state"] == ["A drawn part's plan line."]
    # The same answer as a call that reads the map in time.
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    other = _app()
    SlowFiles(other, monkeypatch, [_studio_answer(_roots)], released=True)
    in_time = await other.check_map_tool({"result_id": R}, views=True)
    assert model["must_state"] == _model(in_time)["must_state"] == MUST
    assert _model(in_time) == model and _view(in_time) == card
    assert _text(in_time) == _text(later)
    # Answered from now: the next call prepares the map again.
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert _view(again) == card
    await other.aclose()
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_second_call_joins_and_a_later_one_answers_from_the_ended_read(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A second call while the preparation runs joins it, fresh or not: one
    read. A call after it ended, with nothing answered from it yet, answers
    from it with no read; the call after that prepares the map again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    first, second = await asyncio.gather(
        app.check_map_tool({"result_id": R}, views=True),
        app.check_map_tool({"result_id": R, "fresh": True}, views=True),
    )
    third = await app.check_map_tool({"result_id": R}, views=True)
    assert all(_view(r)["preparing"] is True for r in (first, second, third))
    assert files.calls == [{"result_id": R, "clip_out_of_range": True}]
    files.release.set()
    await _ended(app)
    later = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 1
    card = _view(later)
    assert "preparing" not in card
    assert [z["zone"] for z in card["zones"]] == ["UTM zone 10N", "UTM zone 33S"]
    assert _model(later)["note"].startswith(server.CHECK_CARD_NOTE)
    # A new read, which must end in time.
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert _view(again) == card
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_without_panels_the_next_call_answers_from_the_finished_read(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Without panels the text says the map is being read and to call again
    in about a minute; that call answers from the finished preparation,
    worded for the chat (each zone's counts, the say-first lines with no card
    open)."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=False)
    told = server.CHECK_PREPARING_TEXT_NOTE + " " + server.NO_GRADE_RULE
    assert _model(first)["note"] == told
    text_note = " ".join(server.CHECK_PREPARING_TEXT_NOTE.split())
    wanted = "call olmoearth_check_map again for this result in about a minute"
    assert wanted in text_note
    files.release.set()
    await _ended(app)
    later = await app.check_map_tool({"result_id": R}, views=False)
    assert len(files.calls) == 1
    model = _model(later)
    assert model["note"].startswith(server.CHECK_TEXT_NOTE)
    assert [z["counts"] for z in model["zones"]] == [[20, 40], [20, 40]]
    texts = [c.text for c in later.content if isinstance(c, types.TextContent)]
    assert json.loads(texts[0])["say_first"][0] == server.SAY_CHECK_CHAT
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_failed_preparation_gives_the_failed_card_and_starts_again(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A preparation that ends failed: the status gives the failed card's view
    data in plain words (r5-a K4: no exception's name, no hint written for
    the model); the next call starts it again, which then ends in time."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    failed = {
        "ok": False,
        "error": READ_FAILED,
        "tool": server.SCORES_TOOL,
        "hint": "Do not retry the same call.",
    }
    files = SlowFiles(app, monkeypatch, [failed, _studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    files.release.set()
    await _ended(app)
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status["state"] == "failed"
    view = status["view"]
    assert (view["tool"], view["ok"], view["needs_sign_in"]) == (
        server.CHECK_TOOL,
        False,
        False,
    )
    assert view["error"] == server._plain_error(READ_FAILED)
    assert not view["error"].startswith("RuntimeError") and view["hint"] is None
    # The status answered the failure: a new read, which must end in time.
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert again.isError is False and len(_view(again)["zones"]) == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_without_panels_a_failed_preparation_is_answered_once(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Without panels nothing asks the status: the call after a preparation
    that ended failed answers that failure (Claude hears why, with no read);
    the call after that starts it again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    failed = {"ok": False, "error": READ_FAILED, "tool": server.SCORES_TOOL}
    files = SlowFiles(app, monkeypatch, [failed, _studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=False)
    assert _model(first)["preparing"] is True
    files.release.set()
    await _ended(app)
    later = await app.check_map_tool({"result_id": R}, views=False)
    assert len(files.calls) == 1
    assert later.isError is True
    model = _model(later)
    assert model["ok"] is False and model["error"] == READ_FAILED
    assert "preparing" not in model and "preparing" not in _view(later)
    assert _view(later)["error"] == server._plain_error(READ_FAILED)
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    again = await app.check_map_tool({"result_id": R}, views=False)
    assert len(files.calls) == 2
    assert again.isError is False
    assert _model(again)["note"].startswith(server.CHECK_TEXT_NOTE)
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_preparation_that_raised_after_the_wait_is_raised_once(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preparation that raised after its call's wait: the next call raises
    it, with no read; the call after that reads again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    calls: list[str] = []
    release = asyncio.Event()

    async def broken(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """A files tool that raises once released."""
        calls.append(name)
        await release.wait()
        raise RuntimeError("words-r5h-raised-late")

    monkeypatch.setattr(app, "run_agent_tool", broken)
    first = await app.check_map_tool({"result_id": R}, views=False)
    assert _model(first)["preparing"] is True
    release.set()
    await _ended(app)
    with pytest.raises(RuntimeError, match="words-r5h-raised-late"):
        await app.check_map_tool({"result_id": R}, views=False)
    assert len(calls) == 1
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    with pytest.raises(RuntimeError, match="words-r5h-raised-late"):
        await app.check_map_tool({"result_id": R}, views=False)
    assert len(calls) == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_stopped_preparation_is_not_answered_from(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A preparation stopped (cancelled) before it ended is never answered
    from: the next call reads again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    task = app._preparations[R].task
    task.cancel()
    await asyncio.wait({task}, timeout=5)
    assert task.cancelled()
    files.release.set()
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert again.isError is False and len(_view(again)["zones"]) == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_fresh_answers_from_a_fresh_read_only(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """Without panels, a call repeated with fresh true, as the note asks,
    answers from the fresh read that ended (no download again); the call
    after that reads again. A fresh call never answers from a plain read
    that ended, even one nothing answered from: it reads again."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    asked = {"result_id": R, "fresh": True}
    first = await app.check_map_tool(asked, views=False)
    assert _model(first)["preparing"] is True
    files.release.set()
    await _ended(app)
    second = await app.check_map_tool(asked, views=False)
    assert _model(second)["note"].startswith(server.CHECK_TEXT_NOTE)
    assert len(files.calls) == 1
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    third = await app.check_map_tool(asked, views=False)
    assert _model(third)["note"].startswith(server.CHECK_TEXT_NOTE)
    assert len(files.calls) == 2
    # A plain read that ended, answered by nothing, then a fresh call.
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    files.release.clear()
    plain_first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(plain_first)["preparing"] is True
    files.release.set()
    await _ended(app)
    await app.check_map_tool(asked, views=True)
    await _ended(app)
    plain = {"result_id": R, "clip_out_of_range": True}
    fresh = {**plain, "fresh": True}
    assert files.calls == [fresh, fresh, plain, fresh]
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_the_status_says_ready_for_a_result_whose_files_cannot_be_read(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result whose files cannot be read (the extra missing) is a card with
    no zones: the status says ready with the view data a call in time gets."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    unread = {
        "ok": True,
        "result": {"available": False, "result_id": R, "reason": "Needs the extra."},
    }
    files = SlowFiles(app, monkeypatch, [unread])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    files.release.set()
    await _ended(app)
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    # Round r5-i B10: no model_text here (the card's words carry the reason).
    assert status == {
        "state": "ready",
        "view": {
            "tool": server.CHECK_TOOL,
            "ok": True,
            "needs_sign_in": False,
            "reason": "Needs the extra.",
            "zones": [],
            "must_state": [],
        },
    }
    monkeypatch.delenv(server.CHECK_WAIT_ENV)
    other = _app()
    SlowFiles(other, monkeypatch, [unread], released=True)
    in_time = await other.check_map_tool({"result_id": R}, views=True)
    assert _view(in_time) == status["view"]
    await other.aclose()
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_preparation_belongs_to_the_key_it_started_with(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A preparation reads with the Studio client of the key in effect when
    it started. After a sign-out a call does not join it: the read is
    stopped and one with no key starts (its card asks to sign in). An ended
    read nothing answered from is not answered from under another key, and
    the status after a sign-out does not give its card. (The map's name is
    not read here: nothing reaches Studio.)"""

    async def no_name(_prediction_id: Any) -> None:
        """No name read from Studio."""
        return None

    one, two = "sk_r5h_test_key_one", "sk_r5h_test_key_two"
    ana, bo = auth.Account("Ana", "Lab", 1), auth.Account("Bo", "Lab", 1)
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    monkeypatch.setattr(app, "_map_name", no_name)
    app.keys.sign_in(one, ana, remember=False)
    no_key = {
        "ok": False,
        "error": auth.NOT_SIGNED_IN,
        "needs_sign_in": True,
        "hint": auth.NOT_SIGNED_IN,
        "tool": server.SCORES_TOOL,
    }
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots), no_key])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    old = app._preparations[R].task
    signed_in = await app.studio()
    app.keys.sign_out()
    second = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(second)["preparing"] is True
    await asyncio.wait({old}, timeout=5)
    assert old.cancelled()
    assert len(files.calls) == 2
    assert files.clients[0] is signed_in
    assert files.clients[1] is await app.studio()
    assert files.clients[1] is not signed_in
    files.release.set()
    await _ended(app)
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status["state"] == "failed" and status["view"]["needs_sign_in"] is True
    assert "zones" not in status["view"]
    await app.aclose()
    # An ended read nothing answered from, then a sign-out: the status gives
    # no card of it; and another key's call reads again.
    app = _app()
    monkeypatch.setattr(app, "_map_name", no_name)
    app.keys.sign_in(one, ana, remember=False)
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    files.release.set()
    await _ended(app)
    app.keys.sign_out()
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status["state"] == "failed"
    assert status["view"]["error"] == server.CHECK_NOT_PREPARING
    app.keys.sign_in(two, bo, remember=False)
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    other_key = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert files.clients[1] is await app.studio()
    assert files.clients[1] is not files.clients[0]
    assert len(_view(other_key)["zones"]) == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_preparation_that_takes_too_long_ends_failed(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A preparation past CHECK_PREPARE_SECONDS (540 s, under the card's 10
    minutes) is stopped and ends with a failed card in plain words, on its
    face; the status says failed, and the next call reads again rather than
    join the stalled read."""
    assert server.CHECK_PREPARE_SECONDS == 540.0
    assert server.CHECK_PREPARE_SECONDS < 600
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    monkeypatch.setattr(server, "CHECK_PREPARE_SECONDS", 0.2)
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    first = await app.check_map_tool({"result_id": R}, views=True)
    assert _view(first)["preparing"] is True
    stalled = app._preparations[R].task
    await _ended(app)
    assert not stalled.cancelled()
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status["state"] == "failed"
    view = status["view"]
    assert view["error"] == server.CHECK_PREPARE_TOO_LONG
    assert view["plain"] is True and view["ok"] is False
    assert server.CHECK_PREPARE_TOO_LONG == (
        "Preparing this map took too long. Ask Claude to open the Check card again."
    )
    monkeypatch.setattr(server, "CHECK_PREPARE_SECONDS", 540.0)
    files.release.set()
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "10")
    again = await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    assert app._preparations[R].task is not stalled
    assert again.isError is False and len(_view(again)["zones"]) == 2
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_the_stage_follows_the_read_in_plain_words(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read's steps, reported in the preparation's own task (from a worker
    thread too), set the stage; a step reported in another task does not.
    While the counts are planned the stage says so and the status and a
    joined call carry the map's line. No stage says window, zone, UTM or a
    file's name."""
    words = {
        (studio_scores.STEP_DOWNLOAD, 0, 0): "Downloading the map from Studio",
        (studio_scores.STEP_COPY, 0, 0): "Reading this computer's copy of the map",
        (studio_scores.STEP_READ, 1, 2): "Reading part 1 of 2",
        (studio_scores.STEP_READ, 1, 1): "Reading the map",
        (studio_scores.STEP_POOL, 2, 2): "Cutting part 2 of 2 into squares",
        (studio_scores.STEP_POOL, 1, 1): "Cutting the map into squares",
    }
    for step, said in words.items():
        assert server._stage_words(*step) == said
    assert server._stage_words("unpacking", 0, 0) is None
    stages = [server.CHECK_STAGE_START, server.CHECK_STAGE_PLAN, *words.values()]
    assert server.CHECK_STAGE_PLAN == "Planning how many squares to offer"
    for stage in stages:
        low = stage.lower()
        for word in ("window", "zone", "utm", ".tif", "epsg"):
            assert word not in low, stage

    planning = threading.Event()

    def slow_offers(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """CANNED, once the test lets the planning go on."""
        planning.wait(5)
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", slow_offers)
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    files.steps.append((studio_scores.STEP_DOWNLOAD, 0, 0))
    files.steps.append((studio_scores.STEP_READ, 1, 2))
    try:
        first = await app.check_map_tool({"result_id": R}, views=True)
        assert _view(first)["preparing"] is True
        await asyncio.wait_for(files.reported.wait(), timeout=5)
        # A step this test's own task reports reaches no preparation.
        studio_scores.report_progress(studio_scores.STEP_POOL, 1, 1)
        status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
        assert status == {"state": "preparing", "stage": "Reading part 1 of 2"}
        files.release.set()
        for _ in range(250):
            status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
            if "map" in status:
                break
            await asyncio.sleep(0.02)
        line = {"name": None, "period_text": "1 Apr 2024 to 30 Jun 2024"}
        assert status == {
            "state": "preparing",
            "stage": server.CHECK_STAGE_PLAN,
            "map": line,
        }
        joined = _view(await app.check_map_tool({"result_id": R}, views=True))
        assert joined["preparing"] is True and joined["map"] == line
        assert joined["stage"] == server.CHECK_STAGE_PLAN
    finally:
        planning.set()
    await _ended(app)
    assert len(files.calls) == 1
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_the_status_tool_is_an_app_tool_that_tells_the_model_nothing(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listed app-only, its one argument the result's id; quiet (the model
    reads {"ok": true}, its failure too). A result this server prepares no
    map for (it started again since the card opened, say) gets the failed
    card, its words in Details."""
    assert server.CHECK_STATUS_TOOL == "olmoearth_app_check_status"
    _text_of_tool, schema = server.APP_TOOLS[server.CHECK_STATUS_TOOL]
    assert schema["required"] == ["result_id"]
    assert set(schema["properties"]) == {"result_id"}
    assert server.CHECK_STATUS_TOOL in server.QUIET_APP_TOOLS
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    tools = {t.name: t for t in await _list(srv)}
    assert tools[server.CHECK_STATUS_TOOL].meta["ui"]["visibility"] == ["app"]
    out = await _call(srv, server.CHECK_STATUS_TOOL, {"result_id": R})
    assert out.isError is False
    assert json.loads(_text(out)) == {"ok": True}
    body = out.structuredContent
    assert body["state"] == "failed"
    assert body["view"] == {
        "tool": server.CHECK_TOOL,
        "ok": False,
        "needs_sign_in": False,
        "error": server.CHECK_NOT_PREPARING,
        "hint": None,
    }
    missing = await _call(srv, server.CHECK_STATUS_TOOL, {})
    assert missing.isError is True
    assert json.loads(_text(missing)) == {"ok": True}


@pytest.mark.asyncio
async def test_h1_fresh_reaches_the_files_tool_only_when_asked(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """``fresh: true`` (a boolean only) is passed on to the files tool; the
    card's tool lists it, still asking only for result_id."""
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)], released=True)
    await app.check_map_tool({"result_id": R}, views=True)
    await app.check_map_tool({"result_id": R, "fresh": True}, views=True)
    await app.check_map_tool({"result_id": R, "fresh": "yes"}, views=True)
    plain = {"result_id": R, "clip_out_of_range": True}
    assert files.calls == [plain, {**plain, "fresh": True}, plain]
    schema = server.OWN_TOOLS[server.CHECK_TOOL][2]
    assert schema["required"] == ["result_id"]
    assert schema["properties"]["fresh"]["type"] == "boolean"
    described = " ".join(server.OWN_TOOLS[server.CHECK_TOOL][1].split())
    assert "do not call this again for that result" in described
    assert "pass fresh true only when the user asks" in described
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_a_preparation_that_raised_is_raised_in_time_and_logged_by_kind(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An exception within the wait reaches the call as before; the log names
    its kind alone, never its words; the status then gives a failed card."""
    app = _app()

    async def broken(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """A files tool that raises, with words no log may hold."""
        raise RuntimeError("words-r5h-not-logged")

    monkeypatch.setattr(app, "run_agent_tool", broken)
    caplog.set_level(logging.WARNING, logger=server.__name__)
    with pytest.raises(RuntimeError, match="words-r5h-not-logged"):
        await app.check_map_tool({"result_id": R}, views=True)
    assert "RuntimeError" in caplog.text
    assert "words-r5h-not-logged" not in caplog.text
    status = await app.app_tool(server.CHECK_STATUS_TOOL, {"result_id": R})
    assert status["state"] == "failed" and status["view"]["ok"] is False
    await app.aclose()


@pytest.mark.asyncio
async def test_h1_an_old_preparation_is_dropped_and_closing_stops_one_running(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """A preparation ended more than CHECK_KEEP_SECONDS ago is not answered
    from (the next call prepares the map again); one still running when the
    server closes is stopped."""
    monkeypatch.setenv(server.CHECK_WAIT_ENV, "0.05")
    monkeypatch.setattr(server, "CHECK_KEEP_SECONDS", 0.0)
    app = _app()
    files = SlowFiles(app, monkeypatch, [_studio_answer(_roots)])
    await app.check_map_tool({"result_id": R}, views=True)
    files.release.set()
    await _ended(app)
    await asyncio.sleep(0.01)
    await app.check_map_tool({"result_id": R}, views=True)
    assert len(files.calls) == 2
    files.release.clear()
    other = "22222222-3333-4444-8555-666666666666"
    preparing = await app.check_map_tool({"result_id": other}, views=True)
    assert _view(preparing)["preparing"] is True
    task = app._preparations[other].task
    await app.aclose()
    assert task.cancelled()


# --------------------------------------------------------------------------- H2 (the card)


@pytest.mark.asyncio
async def test_h2_a_card_read_from_this_computers_copy_says_so_once(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """The files tool read the kept copy: the card's view and Claude's text
    each carry one line with when it was downloaded; the card shows it in
    Details (check.html reads read_from_copy)."""
    app = _app()
    SlowFiles(app, monkeypatch, [_studio_answer(_roots, COPIED)], released=True)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert server.CHECK_COPY_LINE.format(when="18 Nov 2026, 10:04 UTC") == COPY_LINE
    view = _view(result)
    assert view["read_from_copy"] == COPY_LINE
    assert json.dumps(view).count("this computer's copy") == 1
    assert _model(result)["read_from_copy"] == COPY_LINE
    odd = server._copy_line({"downloaded_at": "not a time"})
    assert odd == "Read from this computer's copy of not a time."
    assert server._copy_line(None) is None and server._copy_line({}) is None
    page = (views.VIEW_DIR / "check.html").read_text(encoding="utf-8")
    assert "data.read_from_copy" in page and "'oe-check-copy'" in page
    await app.aclose()


# --------------------------------------------------------------------------- H2 (the kept zip)

#: A [0, 1] score result's id, its download token (a test value: no file may
#: hold it), its field and its zone's file.
SCORE_RID = "5f0c2a71-3b9e-4d8a-9c61-0a7e2b4f1d33"
DL_VALUE = "dl-r5h-test-only-7c2e"
SCORE_NAME = "sample_score"
SCORE_FILE = "result_epsg32616_0_native.tif"
#: The zone's counts, as read_zones gives a [0, 1] score's.
COUNTS = {
    "valid_pixels": 240,
    "nodata_pixels": 16,
    "class_counts": {f"not {SCORE_NAME}": 120, SCORE_NAME: 120},
    "clipped": False,
}


def _score_record() -> dict[str, Any]:
    """A [0, 1] score result's record, its download token included."""
    return {
        "id": SCORE_RID,
        "prediction_id": P,
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
    download counted (after ``delay`` seconds); no prediction or model is
    served, so the model stays unknown."""

    def __init__(self, zipped: bytes, delay: float = 0.0) -> None:
        """The zip every download gives."""
        self.zipped = zipped
        self.delay = delay
        self.downloads = 0

    async def get_prediction_result(self, result_id: str) -> dict[str, Any]:
        """The result's record, as ``GET /prediction-results/{id}`` gives it."""
        assert result_id == SCORE_RID
        return _score_record()

    async def prediction_result_files(
        self, download_token: str, *, max_bytes: int = 0
    ) -> bytes:
        """The files zip, counted."""
        assert download_token == DL_VALUE
        self.downloads += 1
        await asyncio.sleep(self.delay)
        return self.zipped

    async def get_prediction(self, prediction_id: str) -> dict[str, Any]:
        """Not served here."""
        raise RuntimeError("not served by this fake")

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """Not served here."""
        raise RuntimeError("not served by this fake")


class Reads:
    """Stand-ins for the zip's decode (``read_zones``, which notes the bytes
    it was given) and the pooling, and the extra's checks passed."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Put the stand-ins in place."""
        self.seen: list[bytes] = []
        self.root = root
        self.refuse = False
        monkeypatch.setattr(studio_scores, "read_zones", self.read_zones)
        monkeypatch.setattr(studio_scores.scores_file, "_scores_from_file", self.pool)
        monkeypatch.setattr(studio_scores.inferencex, "load", lambda _name: None)
        modules = pytypes.SimpleNamespace(import_module=lambda _name: object())
        monkeypatch.setattr(studio_scores, "importlib", modules)

    def read_zones(
        self, data: bytes, plan: Any, info: Any, clip: bool, np: Any, rasterio: Any
    ) -> tuple[list[Any], int, bool]:
        """One zone, as read_zones returns it; a refusal when ``refuse``."""
        self.seen.append(data)
        if self.refuse:
            raise ValueError("this zip is refused by the stand-in")
        run = self.root / studio_scores.STUDIO_DIR / "Score-test" / "epsg32616_0"
        zone = studio_scores.Zone(
            file=SCORE_FILE,
            run_dir=run,
            sha256="a" * 64,
            reused=len(self.seen) > 1,
            counts=dict(COUNTS),
            height=16,
            width=16,
        )
        return [zone], 0, False

    async def pool(self, args: dict[str, Any], _ctx: Any) -> dict[str, Any]:
        """A pooled zone, as olmoearth_scores_from_file gives it."""
        return {
            "available": True,
            "scores_path": str(self.root / "score_scores.json"),
            "n_valid": 15,
            "patch_px": args["patch"],
        }


async def _read(studio: FakeStudio, **args: Any) -> dict[str, Any]:
    """olmoearth_scores_from_studio's handler on ``studio``, at patch 4."""
    ctx = ToolContext(studio=studio, state=ThreadState())
    out: dict[str, Any] = await studio_scores._scores_from_studio(
        {"result_id": SCORE_RID, "patch": 4, **args}, ctx
    )
    return out


def _kept(root: Path) -> Path:
    """The folder that keeps the score result's zip."""
    return root / studio_scores.KEPT_DIR / SCORE_RID


ZIPPED = b"PK fake zip of a score result, r5-h " * 40


@pytest.mark.asyncio
async def test_h2_a_second_read_downloads_nothing_and_says_so(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first read downloads and keeps the zip (mode 600) with its record;
    the second, within a day, reads the copy, decodes the same bytes, and its
    result says when the copy was downloaded. No file holds the token."""
    reads = Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    first = await _read(studio)
    assert studio.downloads == 1 and "read_from_copy" not in first
    folder = _kept(_roots)
    zipped = folder / studio_scores.KEPT_ZIP
    record_path = folder / studio_scores.KEPT_RECORD
    assert zipped.read_bytes() == ZIPPED
    for path in (zipped, record_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["result_id"] == SCORE_RID and record["bytes"] == len(ZIPPED)
    assert record["sha256"] == hashlib.sha256(ZIPPED).hexdigest()
    assert time.strptime(record["downloaded_at"], studio_scores.KEPT_TIME)
    second = await _read(studio)
    assert studio.downloads == 1
    assert reads.seen == [ZIPPED, ZIPPED]
    kept = second["read_from_copy"]
    assert kept["downloaded_at"] == record["downloaded_at"]
    assert kept["note"] == studio_scores.KEPT_NOTE.format(when=record["downloaded_at"])
    # The rest of the result is the first read's.
    assert second["zones"][0]["scores_path"] == first["zones"][0]["scores_path"]
    assert second["must_state"] == first["must_state"]
    for path in _roots.rglob("*"):
        if path.is_file():
            assert DL_VALUE.encode() not in path.read_bytes(), path
    assert not list(_roots.rglob("*.part"))


@pytest.mark.asyncio
async def test_h2_an_old_copy_is_downloaded_again_and_fresh_downloads(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy kept 24 hours ago or longer is downloaded again, its record's
    time now the new download's; fresh downloads whatever is kept; a download
    that differs from the kept zip replaces it, and is what a later read
    reads."""
    reads = Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    now = time.time()
    day_later = now + 24 * 3600
    monkeypatch.setattr(studio_scores, "_now", lambda: now)
    await _read(studio)
    monkeypatch.setattr(studio_scores, "_now", lambda: day_later)
    old = await _read(studio)
    assert studio.downloads == 2 and "read_from_copy" not in old
    record_path = _kept(_roots) / studio_scores.KEPT_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stamp = time.strftime(studio_scores.KEPT_TIME, time.gmtime(day_later))
    assert record["downloaded_at"] == stamp
    fresh = await _read(studio, fresh=True)
    assert studio.downloads == 3 and "read_from_copy" not in fresh
    studio.zipped = ZIPPED + b"changed in Studio"
    await _read(studio, fresh=True)
    assert (_kept(_roots) / studio_scores.KEPT_ZIP).read_bytes() == studio.zipped
    later = await _read(studio)
    assert studio.downloads == 4 and "read_from_copy" in later
    assert reads.seen[-1] == ZIPPED + b"changed in Studio"


@pytest.mark.asyncio
async def test_h2_a_copy_that_is_not_its_records_zip_is_downloaded_again(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kept zip whose bytes are not the ones its record names, or one with
    no record, is not read: the zip is downloaded again and kept anew."""
    Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    await _read(studio)
    zipped = _kept(_roots) / studio_scores.KEPT_ZIP
    zipped.write_bytes(b"X" * len(ZIPPED))
    again = await _read(studio)
    assert studio.downloads == 2 and "read_from_copy" not in again
    assert zipped.read_bytes() == ZIPPED
    (_kept(_roots) / studio_scores.KEPT_RECORD).unlink()
    await _read(studio)
    assert studio.downloads == 3


@pytest.mark.asyncio
async def test_h2_keeping_off_or_a_refused_zip_keeps_nothing(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With OLMOEARTH_STUDIO_KEEP_HOURS 0 nothing is kept and every read
    downloads; a zip the decode refuses is never kept."""
    assert studio_scores.STUDIO_KEEP_HOURS == 24.0
    assert studio_scores.keep_hours() == 24.0
    for raw, hours in (("2", 2.0), ("0", 0.0), ("-3", 0.0), ("nan", 0.0)):
        monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, raw)
        assert studio_scores.keep_hours() == hours
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "soon")
    assert studio_scores.keep_hours() == 24.0
    reads = Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "0")
    await _read(studio)
    await _read(studio)
    assert studio.downloads == 2
    assert not (_roots / studio_scores.KEPT_DIR).exists()
    monkeypatch.delenv(studio_scores.KEEP_HOURS_ENV)
    reads.refuse = True
    with pytest.raises(ValueError, match="refused by the stand-in"):
        await _read(studio)
    assert studio.downloads == 3
    assert not (_roots / studio_scores.KEPT_DIR).exists()


@pytest.mark.asyncio
async def test_h2_a_changed_download_the_decode_refuses_removes_the_copy(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that differs from the kept zip removes the kept copy before
    its decode: when the decode refuses it, no older copy is left for a later
    read to take for Studio's files, so that read downloads again. A refused
    download of the same zip keeps the copy (the refusal is the call's)."""
    reads = Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    await _read(studio)
    folder = _kept(_roots)
    assert (folder / studio_scores.KEPT_ZIP).read_bytes() == ZIPPED
    studio.zipped = ZIPPED + b"changed in Studio"
    reads.refuse = True
    with pytest.raises(ValueError, match="refused by the stand-in"):
        await _read(studio, fresh=True)
    assert studio.downloads == 2
    assert not (folder / studio_scores.KEPT_ZIP).exists()
    assert not (folder / studio_scores.KEPT_RECORD).exists()
    reads.refuse = False
    later = await _read(studio)
    assert studio.downloads == 3 and "read_from_copy" not in later
    assert (folder / studio_scores.KEPT_ZIP).read_bytes() == studio.zipped
    # The same zip refused: the copy stays, and is read next.
    reads.refuse = True
    with pytest.raises(ValueError, match="refused by the stand-in"):
        await _read(studio, fresh=True)
    assert studio.downloads == 4
    assert (folder / studio_scores.KEPT_ZIP).read_bytes() == studio.zipped
    assert (folder / studio_scores.KEPT_RECORD).exists()
    reads.refuse = False
    kept = await _read(studio)
    assert studio.downloads == 4 and "read_from_copy" in kept
    # Removing a copy never fails a read, and leaves no folder behind.
    studio_scores.forget_kept(SCORE_RID)
    assert not folder.exists()
    studio_scores.forget_kept(SCORE_RID)


@pytest.mark.asyncio
async def test_h2_two_reads_at_once_download_once(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reads of one result at the same time: the second waits for the
    first and reads the copy it kept."""
    Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED, delay=0.05)
    first, second = await asyncio.gather(_read(studio), _read(studio))
    assert studio.downloads == 1
    assert ["read_from_copy" in out for out in (first, second)] == [False, True]


@pytest.mark.asyncio
async def test_h2_a_read_reports_its_steps_to_the_callers_hook(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download (or the copy read instead), then each part pooled, are
    told to the hook set in the caller's context; nothing else is."""
    Reads(_roots, monkeypatch)
    studio = FakeStudio(ZIPPED)
    told: list[tuple[str, int, int]] = []

    def hook(step: str, number: int, parts: int) -> None:
        """Note a step."""
        told.append((step, number, parts))

    marker = studio_scores.PROGRESS.set(hook)
    try:
        await _read(studio)
        await _read(studio)
    finally:
        studio_scores.PROGRESS.reset(marker)
    assert told == [
        (studio_scores.STEP_DOWNLOAD, 0, 0),
        (studio_scores.STEP_POOL, 1, 1),
        (studio_scores.STEP_COPY, 0, 0),
        (studio_scores.STEP_POOL, 1, 1),
    ]


def test_h2_keeping_removes_the_copies_older_than_the_limit(_roots: Path) -> None:
    """Keeping a zip removes another result's copy kept 24 hours ago or
    longer (its zip, its record, its folder), never a newer one."""
    stale = "6a1b2c3d-4e5f-4a6b-8c7d-8e9f0a1b2c3d"
    recent = "7b2c3d4e-5f6a-4b7c-9d8e-9f0a1b2c3d4e"
    now = time.time()
    for rid, age in ((stale, 25 * 3600), (recent, 3600)):
        folder = _roots / studio_scores.KEPT_DIR / rid
        folder.mkdir(parents=True)
        (folder / studio_scores.KEPT_ZIP).write_bytes(b"old zip")
        stamp = time.strftime(studio_scores.KEPT_TIME, time.gmtime(now - age))
        record = {
            "format": studio_scores.KEPT_FORMAT,
            "result_id": rid,
            "downloaded_at": stamp,
            "bytes": 7,
            "sha256": hashlib.sha256(b"old zip").hexdigest(),
        }
        (folder / studio_scores.KEPT_RECORD).write_text(json.dumps(record), "utf-8")
    studio_scores.keep(SCORE_RID, ZIPPED)
    assert not (_roots / studio_scores.KEPT_DIR / stale).exists()
    assert (_roots / studio_scores.KEPT_DIR / recent / studio_scores.KEPT_ZIP).exists()
    assert (_kept(_roots) / studio_scores.KEPT_ZIP).read_bytes() == ZIPPED
    kept = studio_scores.read_kept(SCORE_RID, len(ZIPPED))
    assert kept is not None and kept[0] == ZIPPED
    assert studio_scores.read_kept(SCORE_RID, len(ZIPPED) - 1) is None


def test_h2_the_files_tool_takes_fresh() -> None:
    """olmoearth_scores_from_studio lists fresh, a boolean, off by default."""
    (tool,) = studio_scores.build_studio_scores_tools()
    fresh = tool.spec.parameters["properties"]["fresh"]
    assert fresh["type"] == "boolean" and fresh["default"] is False
    assert tool.spec.parameters["required"] == ["result_id"]
