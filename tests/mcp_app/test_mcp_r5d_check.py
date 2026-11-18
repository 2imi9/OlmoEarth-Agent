# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-d (17 Nov 2026), track CHECK, on the MCP App: the Check card's
items of the last audit of the whole app (specs/r5-final-audit/build_list.md).

- R01: a check that cannot start is a card with its title, one plain line in
  red (what happened, that earlier work is kept, and to ask Claude, with no
  promise that opening the card again helps) and the server's own words in
  a closed Details fold; a refusal the server wrote for the card's face (a
  map that is not a score from 0 to 1) is the red card's first line, with
  no fold; the model's text is unchanged;
- R05: when a part's sheet opens again, the toast that said its answers were
  not saved goes with the part's own line, while it still says those words;
- R11: a card's error leaves out an OSError's number at the start of the
  error, after the exception's name if there is one, when words follow it
  ("[Errno 28] ", or on Windows "[WinError 32] "); the model's text and the
  stored draw record keep the whole error.

The views are read as text (no test here runs the scripts); a JS
concatenation of single-quoted strings is read back into its text
(:func:`_concat`). The server's words are read through its own functions.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import auth, check, server, views  # noqa: E402
from olmoearth_agent.tools import studio_scores  # noqa: E402

BASE = "http://mock-studio/api/v1"
R = "11111111-2222-4333-8444-555555555555"
P = "99999999-8888-4777-8666-555555555555"
ZONES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")

#: One token of a JS concatenation: a single-quoted string or a plus.
_TOKEN = re.compile(r"\s*(?:'((?:[^'\\]|\\.)*)'|(\+))")

#: Words no face holds (the owner's ruling 6 and r4-i's rule).
_OFF_FACE = re.compile(r"\b(?:windows?|budget|stratum|design|review set)\b", re.I)

#: The registry's hint on a failed tool, written for the model.
MODEL_HINT = (
    "The tool ran but failed. Check that referenced ids exist (discover them "
    "with search/list tools) and that argument values are valid; do not retry "
    "with identical arguments."
)

#: R01: the red card's one line when the check cannot start.
CHECK_FAILED = (
    "The check could not start. Anything picked or saved before is kept. Ask "
    "Claude what to do next: it has the reason."
)
#: R01: the error of the audit's card (15c), as the server sends it to a card.
READ_FAILED = "GET /prediction-results/files failed (ReadTimeout); no file was read"

#: R11: an OSError's error, as the registry writes it, and as a card shows it.
NO_SPACE = "OSError: [Errno 28] No space left on device"
NO_SPACE_PLAIN = "No space left on device"

CANNED = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty labels.", "rate_note": "r50"},
        {"labels": 40, "width": 0.1, "sentence": "Forty labels.", "rate_note": "r50"},
    ],
    "refused": None,
}

Plan = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores, outputs and the check records): the test's own."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    return out


@pytest.fixture
def canned_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts of 20 and 40 offered for every zone, without the package."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`CANNED`."""
        return copy.deepcopy(CANNED)

    monkeypatch.setattr(check, "offers", fake)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def _after(text: str, start: str, end: str) -> str:
    """The part of text after start up to the first end after it."""
    at = text.index(start) + len(start)
    return text[at : text.index(end, at)]


def _concat(expr: str) -> str:
    """The text a JS concatenation of single-quoted strings makes; anything
    else fails."""
    out: list[str] = []
    text = expr.strip()
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        assert match is not None, text[at:]
        literal, _plus = match.groups()
        if literal is not None:
            out.append(re.sub(r"\\(.)", r"\1", literal))
        at = match.end()
    return "".join(out)


def _code(page: str) -> str:
    """The page with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _app() -> server.AppServer:
    """The app's server on the mocked Studio's address."""
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


def _view(result: types.CallToolResult) -> dict[str, Any]:
    """The view data of a tool's result."""
    view: dict[str, Any] = (result.meta or {})[server.VIEW_DATA_KEY]
    return view


def _model(result: types.CallToolResult) -> dict[str, Any]:
    """The model's copy of a Check card's result: its last text block."""
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    body: dict[str, Any] = json.loads(texts[-1])
    return body


def _zone_entry(root: Path, index: int, file: str) -> dict[str, Any]:
    """One zone of ``olmoearth_scores_from_studio``'s result, its scores file
    written under the root."""
    path = root.resolve() / f"zone{index}_scores.json"
    path.write_text(json.dumps({"scores": [[0.0, 0.4]] * 300}), "utf-8")
    return {
        "file": file,
        "scores_path": str(path),
        "patch": 4,
        "valid_pixels": 4800,
        "pooled": {
            "n_valid": 300,
            "n_windows": 300,
            "raster_check": {"sha256": str(index) * 64},
        },
    }


def _topk_answer(root: Path) -> dict[str, Any]:
    """The envelope for a class map with top-k scores in two zones, of presence
    and absence: the owner's labelling rule is set, so a press needs none."""
    zones = [_zone_entry(root, i, file) for i, file in enumerate(ZONES)]
    shares = [{"file": f, "share": s} for f, s in zip(ZONES, (0.6, 0.4))]
    inner = {
        "available": True,
        "result_id": R,
        "prediction_id": P,
        "kind": studio_scores.TOPK,
        "property_name": "kelp",
        "classes": [
            {"channel": 0, "value": 0, "label": "absent"},
            {"channel": 1, "value": 1, "label": "present"},
        ],
        "date_window": {"start": "2024-04-01T00:00:00Z", "end": "2024-06-30T00:00:00Z"},
        "n_zones": 2,
        "zones": zones,
        "facts": [{"id": "studio_zones", "sentence": "Two.", "zones": shares}],
        "must_state": [
            studio_scores.TOPK_SCORES_SENTENCE,
            studio_scores.ZONES_SENTENCE,
        ],
        "forbidden_claims": [{"id": "certified", "why": "Nothing certifies it."}],
    }
    return {"ok": True, "result": inner}


def _fake_tools(
    app: server.AppServer,
    monkeypatch: pytest.MonkeyPatch,
    studio: dict[str, Any],
    plan: Plan | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """``app.run_agent_tool`` answering ``studio`` for the Studio files and
    ``plan`` for a plan; the calls made, in order."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's envelope, as run_agent_tool gives it."""
        calls.append((name, dict(arguments)))
        if name == server.SCORES_TOOL:
            return copy.deepcopy(studio)
        assert plan is not None, name
        return await plan(dict(arguments))

    monkeypatch.setattr(app, "run_agent_tool", run)
    return calls


def _failing(error: str) -> Plan:
    """A plan that always fails with ``error``, as the registry writes it."""

    async def plan(args: dict[str, Any]) -> dict[str, Any]:
        """The failed envelope."""
        return {"ok": False, "error": error, "hint": MODEL_HINT}

    return plan


def _press(*draws: tuple[str, int]) -> dict[str, Any]:
    """The press's arguments: ``(check_key, budget)`` per zone."""
    return {"draws": [{"check_key": k, "budget": b} for k, b in draws]}


# --------------------------------------------------------------------------- R01


def test_r01_the_failed_card_has_a_title_a_plain_line_and_its_words_folded() -> None:
    """R01: a check that cannot start (the last failure branch of show) draws
    the card's title, a red card with one plain line that says what happened
    and what to do, and the server's words (its plain error, a hint when the
    view holds one) in a closed Details fold, each a small line; the error
    is no longer the red card's text, and no fallback sentence is left. A
    view the server marks ``plain`` (a refusal it wrote for this card's
    face) gets its error as the red card's first line, over the plain line,
    and no fold: two lines on the face."""
    page = _read("check.html")
    tags = "// WORDING: r5d R01\nconst CHECK_FAILED = "
    assert tags in page
    assert _concat(_after(page, "const CHECK_FAILED = ", ";\n")) == CHECK_FAILED
    show = _between(page, "function show(app, params) {", "\n}\n")
    assert _in_order(
        show,
        "if (data.needs_sign_in) {",
        "if (!data.ok && reason) {\n    classMapCard(app, data, reason);",
        "if (!data.ok && next && typeof data.error === 'string' && data.error) {\n"
        "    numberMapCard(data, next);",
        "  if (!data.ok) {\n    // The check could not start (r5-d R01)",
        "    root.append(OE.el('h2', {}, TITLE));\n"
        "    const own = data.plain === true && typeof data.error === 'string'"
        " && data.error ? data.error : null;\n"
        "    root.append(OE.el('div', { class: own ? 'card error stack' : "
        "'card error' },\n"
        "      own ? OE.el('div', {}, own) : null, "
        "OE.el('div', {}, CHECK_FAILED)));\n",
        "    const said = own ? [] : [data.error, data.hint]"
        ".filter((s) => typeof s === 'string' && s);\n",
        "    if (said.length) root.append(OE.fold(DETAILS, "
        "...said.map((s) => OE.el('div', { class: 'small' }, s))));\n",
        "    return;\n  }\n  const zones = ",
    )
    failed = _between(
        show, "  if (!data.ok) {\n    // The check could not", "const zones"
    )
    # The error and the hint are only in the fold, or, for a plain refusal,
    # the error alone over the plain line: never the red card's only text.
    assert "data.error ||" not in failed and "data.hint ?" not in failed
    assert "'The check failed.'" not in _code(page)
    assert "OE.fold(" not in _between(
        page, "function numberMapCard(data, next) {", "\n}\n"
    )
    # Defined once, used once.
    assert _code(page).count("CHECK_FAILED") == 2
    # The line is plain: no transport word, no exception's name, nothing for
    # the model, and it says the next step.
    assert _OFF_FACE.search(CHECK_FAILED) is None
    for word in ("GET", "Error", "Errno", "olmoearth_", "Studio"):
        assert word not in CHECK_FAILED, word
    assert CHECK_FAILED.endswith("Ask Claude what to do next: it has the reason.")
    # True on every route: no promise that opening the card again helps (a
    # refusal comes back the same on every open, and Claude's text says not
    # to retry with the same arguments), and no claim that nothing was saved
    # (a card opened again keeps its earlier picks and answers).
    for false_here in ("open the Check card again", "nothing was", "Details"):
        assert false_here not in CHECK_FAILED, false_here
    assert "Anything picked or saved before is kept." in CHECK_FAILED
    # The fold's label is the card's own (every card's fold says "Details").
    assert "// WORDING: r4p2\nconst DETAILS = 'Details';\n" in page


@pytest.mark.asyncio
async def test_r01_the_server_still_sends_the_plain_error_and_no_hint(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R01: the view of a check whose files could not be read holds the
    error without its exception's name (the fold's line), no hint and no
    ``plain`` (the card folds the tool's words); the model's text keeps the
    error and the hint."""
    failed = {
        "ok": False,
        "error": "RuntimeError: " + READ_FAILED,
        "tool": server.SCORES_TOOL,
        "hint": MODEL_HINT,
    }
    app = _app()
    _fake_tools(app, monkeypatch, failed)
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    view = _view(result)
    assert view == {
        "tool": server.CHECK_TOOL,
        "ok": False,
        "needs_sign_in": False,
        "error": READ_FAILED,
        "hint": None,
    }
    model = _model(result)
    assert (model["error"], model["hint"]) == (failed["error"], MODEL_HINT)
    await app.aclose()


@pytest.mark.asyncio
async def test_r01_a_map_that_is_not_a_score_keeps_its_sentence_on_the_face(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R01: the refusal of a map that is not a score from 0 to 1 is the
    server's own plain sentence for this card's face (``CHECK_NOT_A_SCORE``,
    r5-a K2), so its view is marked ``plain`` and the card shows it over the
    plain line; a refusal in the tool's words (an output folder that cannot
    be written, ``check.NOT_WRITABLE``) is not marked, so the card folds it.
    The model's text is as before, with no ``plain``."""
    inner = {
        "available": True,
        "result_id": R,
        "kind": studio_scores.UNIT,
        "zones": [{"values_range": [0.0, 100.0]}],
        "must_state": [],
    }
    app = _app()
    _fake_tools(app, monkeypatch, {"ok": True, "result": inner})
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    words = server.CHECK_NOT_A_SCORE.format(low="0", high="100")
    assert _view(result) == {
        "tool": server.CHECK_TOOL,
        "ok": False,
        "needs_sign_in": False,
        "error": words,
        "hint": None,
        "plain": True,
    }
    assert _model(result) == {"ok": False, "error": words}

    unwritable = check.NOT_WRITABLE.format(root="/srv/outputs")

    def refuse(*_given: Any, **_named: Any) -> Any:
        """``check.offer_zones`` on an output folder that cannot be written."""
        raise ValueError(unwritable)

    monkeypatch.setattr(check, "offer_zones", refuse)
    _fake_tools(app, monkeypatch, _topk_answer(_roots))
    result = await app.check_map_tool({"result_id": R}, views=True)
    assert result.isError is True
    view = _view(result)
    assert (view["error"], view["hint"], view["needs_sign_in"]) == (
        unwritable,
        None,
        False,
    )
    assert "plain" not in view and "zones" not in view
    assert _model(result) == {"ok": False, "error": unwritable}
    await app.aclose()


# --------------------------------------------------------------------------- R05


def test_r05_the_toast_on_answers_not_saved_goes_when_the_parts_sheet_opens() -> None:
    """R05: once the part's sheet is read again, its own line goes, and the
    toast goes with it while the toast still says the line's words (OE.say's
    test: any other toast stays); it is cleared before the editor swap writes
    a toast of its own for the part swapped away, so that one stays."""
    page = _read("check.html")
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    clear = (
        "    m.sheetError = null;\n"
        "    // Its sheet opens again: the line on answers not saved goes "
        "(r5-b J10),\n"
        "    // and so does the toast that said it in the same words (r5-d R05; the\n"
        "    // same test as OE.say's clear: any other toast stays).\n"
        "    const bar = document.getElementById('oe-toast');\n"
        "    if (m.line && bar && bar.textContent === m.line) OE.toast('');\n"
        "    m.line = null;\n"
    )
    assert clear in swap
    assert _in_order(
        swap,
        "sheet = await OE.call(app, 'olmoearth_app_sheet', "
        "{ labels_csv_path: m.draw.labels_csv_path });",
        "    } catch (err) {\n",
        "      if (had) OE.focusIfHere(m.openButton);\n      return;\n    }\n",
        clear,
        "const editor = OE.sheetEditor(app, sheet, {",
        "      before.line = NOT_SAVED_PART(dropped, label(before));\n"
        "      fillZone(before);\n"
        "      OE.toast(before.line, true);\n",
    )
    # A sheet that cannot be read keeps both: the clear is after the catch.
    caught = _between(swap, "    } catch (err) {\n", "      return;\n    }\n")
    assert "OE.toast(" not in caught and "m.line" not in caught
    # The test holds because the toast's text is the line's, set whole, in
    # the one region of that id.
    common = _read("common.js")
    toast = _between(common, "OE.toast = (text, isError = false) => {", "\n};\n")
    assert "  bar.textContent = text;" in toast
    assert "let bar = document.getElementById('oe-toast');" in common
    # No new words.
    assert "WORDING: r5d R05" not in page


# --------------------------------------------------------------------------- R11


def test_r11_a_cards_error_leaves_out_the_oserrors_number() -> None:
    """R11: an OSError's number at the start of the error, after the
    exception's name if there is one, when words follow it, goes (a name
    lookup's is negative; on Windows a Win32 error's is "[WinError n] "); a
    number with no words after it, or anywhere but the start, stays; a
    ``KeyError`` is put in words as before."""
    plain = server._plain_error
    assert plain(NO_SPACE) == NO_SPACE_PLAIN
    assert plain("OSError: [Errno 5] Input/output error") == "Input/output error"
    lost = "FileNotFoundError: [Errno 2] No such file or directory: 'x.tif'"
    assert plain(lost) == "No such file or directory: 'x.tif'"
    assert plain("ConnectionError: [Errno -2] Name or service not known") == (
        "Name or service not known"
    )
    assert plain("[Errno 28] No space left on device") == NO_SPACE_PLAIN
    busy = (
        "The process cannot access the file because it is being used by another "
        "process: 'C:\\x.csv'"
    )
    assert plain("PermissionError: [WinError 32] " + busy) == busy
    for kept in (
        "OSError: [Errno 5]",
        "OSError: [Errno 5] ",
        "OSError: [WinError 5]",
        "the write failed: [Errno 28] No space left on device",
        "OSError: [Errno x] not a number",
    ):
        assert plain(kept) == server._ERROR_NAME.sub("", kept, count=1), kept
    assert plain("KeyError: 'labels'") == server.MISSING_ENTRY.format(key="'labels'")
    assert server._ERRNO.pattern == r"^\[(?:Errno|WinError) -?\d+\] (?=\S)"
    assert server._card_error(NO_SPACE) == NO_SPACE_PLAIN
    assert server._refusal_words(NO_SPACE) == NO_SPACE_PLAIN
    # The card's copy of a failed part and of a press's reply.
    zone = {"state": "failed", "error": NO_SPACE, "retry": True}
    assert server._check_zone_for_view(zone)["error"] == NO_SPACE_PLAIN
    reply = {"check_key": "k", "error": NO_SPACE}
    assert server._draw_for_view(reply)["error"] == NO_SPACE_PLAIN
    assert (zone["error"], reply["error"]) == (NO_SPACE, NO_SPACE)
    marked = "# WORDING: r5d R11 (what a card's error leaves out)\n_ERRNO = "
    assert marked in Path(server.__file__).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_r11_a_failed_pick_reads_plain_and_its_record_keeps_the_number(
    _roots: Path, monkeypatch: pytest.MonkeyPatch, canned_offers: None
) -> None:
    """R11: a pick that failed with an ``OSError`` (one a press may retry) and
    one with a ``FileNotFoundError`` (one it may not): the press's reply and
    the card opened again show the words alone; the draw record and the
    model's zone keep the name and the number, so whether a press retries
    is as before."""
    for error, shown, retry in (
        (NO_SPACE, NO_SPACE_PLAIN, True),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: 'p.csv'",
            "No such file or directory: 'p.csv'",
            False,
        ),
    ):
        app = _app()
        _fake_tools(app, monkeypatch, _topk_answer(_roots), _failing(error))
        card = _view(await app.check_map_tool({"result_id": R}, views=True))
        key = card["zones"][0]["check_key"]
        pressed = await app.app_tool("olmoearth_app_check_draw", _press((key, 20)))
        (reply,) = pressed["draws"]
        assert (reply["state"], reply["error"], reply["retry"]) == (
            "failed",
            shown,
            retry,
        )
        path = _roots / "checks" / f"{key}.draw.json"
        record = json.loads(path.read_text("utf-8"))
        assert (record["error"], record["retry"]) == (error, retry)
        later = await app.check_map_tool({"result_id": R}, views=True)
        assert _view(later)["zones"][0]["error"] == shown
        told = _model(later)["zones"][0]
        assert (told["error"], told["failed_for_good"]) == (error, not retry)
        assert "Errno" not in json.dumps(_view(later))
        path.unlink()
        await app.aclose()


def test_r11_the_page_shows_the_servers_words_as_they_come() -> None:
    """R11 is the server's: the page writes "The pick failed: " before the
    error the view gives and strips nothing of its own."""
    page = _read("check.html")
    assert "// WORDING: squares\nconst PICK_FAILED = 'The pick failed: ';\n" in page
    assert "PICK_FAILED + bare(m.error) + '.'" in page
    assert "Errno" not in _code(page)
