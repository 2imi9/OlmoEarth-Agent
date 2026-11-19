# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-n (19 Nov 2026), track SETTINGS, as specs/r5-n-settings.md gives it
(the owner's design of 19 Nov 2026: "let's make a setting button at bottom left
that handles some status like on and off of asta"):

- S1: every card ends with one quiet Settings text button at its bottom left
  (views/common.js OE.settingsBlock, put after #root and the page's status line
  by OE.start once the card is connected), a gear and the word; its box opens
  in place under it, a
  fold closed by the button or by Escape, the focus then back on the button;
  it reads its rows from the server at each opening.
- S2: the box's three rows (mcp_app/settings.py): Asta research skills (a
  switch, off by default, kept in <output root>/settings.json, mode 600,
  written by an atomic rename, holding the switch alone), Satellite pictures
  (Forget my yes removes imagery's stored yes; the consent version is
  unchanged) and Maps kept on this computer (Remove kept copies:
  studio_scores.forget_kept for each); the app-only, quiet tools
  olmoearth_app_settings and olmoearth_app_setting.
- S3: the start card's Asta tile, only while the switch is on (start_tool's
  view data carries asta only then); its press is the only thing in the app
  that asks for Asta; no server instruction, tool description or text for the
  model says Asta, and the views say it only in the box and the tile.
- S4: the docs lines (mcp-app-setup.md Options, mcp-app.md Settings, Usage.md).

No test here runs a view's script or touches the network; the views are read
as text. The browser harness (scripts/mcp_views_e2e.py) draws the box on the
start card (11g, 11h), the Check card (12x) and the label sheet (3x) at 780 and
380 px, by the keyboard, and the Asta tile coming and going; 11i and 11j show a
start card following the switch as the server holds it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")
from mcp import types  # noqa: E402

from olmoearth_agent.mcp_app import auth, imagery, server, settings, views  # noqa: E402
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV  # noqa: E402
from olmoearth_agent.skills import build_default_registry  # noqa: E402
from olmoearth_agent.tools import studio_scores  # noqa: E402

BASE = "http://mock-studio/api/v1"

#: The docs folder, beside the tests.
DOCS = Path(__file__).resolve().parents[2] / "docs"

#: S3: the setup page's lines on Ai2's Asta skills, which the box's "setup"
#: link opens and the tile's message names.
SETUP_URL = (
    "https://github.com/2imi9/OlmoEarth-Agent/blob/main/docs/mcp-app-setup.md#options"
)

#: S2: the switch's line, with its "setup" link in the brackets, word for word.
ASTA_LINE = (
    "On: the start card shows Search papers with Asta. Asta runs only when you "
    "press it. Needs Ai2's Asta skills in Claude Code (setup)."
)

#: S3: the tile's label and line, word for word.
TILE_LABEL = "Search papers with Asta"
TILE_LINE = (
    "Ai2's research skills find and summarise papers. Claude asks what to search "
    "first."
)

#: S3, after review: signed out with the tile shown, the card's line and the
#: line over the tiles (the tile needs no sign-in).
START_NEXT_STUDIO = (
    "Sign in first for the Studio jobs. Press a job or ask Claude in the chat."
)
SIGN_IN_STUDIO_JOB = "Sign in to start a Studio job."

#: S3: the tile's one message, before the setup page's address.
TILE_MESSAGE = (
    "Search the scientific literature with Ai2's Asta skills (asta-tools). First "
    "ask me what to search for. Use the quick literature search. Before any long "
    "or multi-step run (Asta's workflows, its long-running assistant, a summary of "
    "many papers), tell me what it will do and roughly how long it takes, and wait "
    "for my yes. If the Asta skills are not available here, say so and point me to "
    "the setup page: "
)

#: Three Studio result ids, each with a kept copy in some tests.
RID_A = "6a1b2c3d-4e5f-4a6b-8c7d-8e9f0a1b2c3d"
RID_B = "7b2c3d4e-5f6a-4b7c-9d8e-9f0a1b2c3d4e"
RID_OLD = "8c3d4e5f-6a7b-4c8d-ae9f-0a1b2c3d4e5f"

#: Asta as a word, in any case.
ASTA = re.compile(r"\basta\b", re.I)

#: The browser harness's stand-in of olmoearth_app_settings' answer
#: (scripts/mcp_views_e2e.py SETTINGS_STATE; a copy, as the harness is outside
#: this tree), and of the answer once the yes is forgotten and the copies are
#: removed (SETTINGS_REMOVED): r5-b J30, a state a real run reaches.
HARNESS_STATE = {
    "asta": {"on": False},
    "pictures": {"allowed": True, "said": "You allowed them on 18 Nov 2026."},
    "kept": {
        "count": 2,
        "said": "2 maps kept for 24 h, so they open without downloading again.",
    },
}
HARNESS_REMOVED = {
    "asta": {"on": False},
    "pictures": {"allowed": False, "said": "Not allowed yet."},
    "kept": {"count": 0, "said": "None kept."},
}


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An output root (and so the scores root) of each test's own; no key; the
    default hours a copy is kept."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))
    monkeypatch.delenv(studio_scores.KEEP_HOURS_ENV, raising=False)
    return out.resolve()


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after start."""
    at = text.index(start)
    return text[at : text.index(end, at + len(start))]


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


def _code(script: str) -> str:
    """script without its comments: each block comment, and each line that is
    a line comment alone (an address's two slashes inside a string stay)."""
    script = re.sub(r"/\*.*?\*/", "", script, flags=re.S)
    return re.sub(r"(?m)^[ \t]*//[^\n]*$", "", script)


def _js_strings(code: str) -> list[str]:
    """The single-quoted JS strings in code, in order, their escaped quotes
    undone."""
    found = re.findall(r"'((?:[^'\\]|\\.)*)'", code)
    return [s.replace("\\'", "'") for s in found]


def _block() -> str:
    """common.js's OE.settingsBlock, whole."""
    return _between(_read("common.js"), "OE.settingsBlock = (app) => {", "\n};\n")


def _keep(root: Path, result_id: str, age_hours: float) -> None:
    """A kept copy of result_id downloaded age_hours ago, as studio_scores.keep
    writes it: its zip, then its record."""
    folder = root / studio_scores.KEPT_DIR / result_id
    folder.mkdir(parents=True)
    data = b"zip of " + result_id.encode("ascii")
    (folder / studio_scores.KEPT_ZIP).write_bytes(data)
    stamp = time.strftime(
        studio_scores.KEPT_TIME, time.gmtime(time.time() - age_hours * 3600)
    )
    record = {
        "format": studio_scores.KEPT_FORMAT,
        "result_id": result_id,
        "downloaded_at": stamp,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (folder / studio_scores.KEPT_RECORD).write_text(json.dumps(record), "utf-8")


def _app() -> server.AppServer:
    """An app server with no key, against a Studio that is never called."""
    return server.AppServer(None, keys=auth.KeyStore(), base_url=BASE)


async def _list(srv: Any) -> list[types.Tool]:
    """The tools the server lists."""
    result = await srv.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    return list(result.root.tools)


async def _call(srv: Any, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """One tool call through the server's handler."""
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await srv.request_handlers[types.CallToolRequest](request)
    return result.root


def _text(result: types.CallToolResult) -> str:
    """What the model reads of a result: its text."""
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _view(result: types.CallToolResult) -> dict[str, Any]:
    """What a result's view gets (its _meta)."""
    data: dict[str, Any] = (result.meta or {})[server.VIEW_DATA_KEY]
    return data


# --------------------------------------------------------------------------- S1


def test_s1_every_card_ends_with_the_settings_button() -> None:
    """S1: OE.start puts the Settings block in place once the card is
    connected (after its host report, right before it returns); every view
    inlines common.js; the block goes after #root and the page's status line
    (so after the card's Details fold and, on the label sheet, after its
    sticky footer, which is inside #root), once per page."""
    common = _read("common.js")
    start = _between(
        common,
        "OE.start = async (name, { onResult, onInput, onContext } = {}) => {",
        "\n};\n",
    )
    assert _in_order(
        start, "await app.connect();", "report('start');", "OE.settingsBlock(app);"
    )
    assert "\n  OE.settingsBlock(app);\n  return app;" in start
    for name, spec in views.VIEWS.items():
        assert spec["scripts"][0] == "common.js", name
        assert "OE.settingsBlock = (app) => {" in views.view_html(name), name
    block = _block()
    assert "if (had) return had;" in block
    # After review (r5-k K5): the page's status line goes back between #root
    # and the block, so a status never sits under Settings or its box; a line
    # the Check card holds inside #root stays there.
    assert block.rstrip().endswith(
        "  if (root) {\n"
        "    root.after(block);\n"
        "    const bar = document.getElementById('oe-toast');\n"
        "    if (bar && bar.parentElement === document.body) root.after(bar);\n"
        "  } else document.body.append(block);\n"
        "  return block;"
    )
    assert _in_order(block, "root.after(block);", "root.after(bar);")
    assert "sheet-footer" not in _code(block)
    show = _between(_read("check.html"), "function show(app, params) {", "\n}\n")
    assert "  if (root.contains(bar)) root.after(bar);\n" in show
    assert "document.body.append(bar)" not in show


def test_s1_the_button_is_a_quiet_text_button_with_an_outline_gear() -> None:
    """S1: the button is a text button (link), never the card's contained one,
    a fold over the box it names (aria-expanded, aria-controls), holding the
    gear and the word; the gear is drawn in code (createElementNS, no markup
    string), an outline in the text's colour, hidden from a screen reader."""
    common = _read("common.js")
    block = _block()
    for text in (
        "id: 'oe-settings-button', class: 'link oe-settings-button',",
        "'aria-expanded': 'false', 'aria-controls': 'oe-settings-box' }, "
        "OE.gear(), OE.el('span', {}, OE.SETTINGS));",
        "id: 'oe-settings-box', class: 'oe-settings-box', role: 'group', "
        "'aria-label': OE.SETTINGS,",
    ):
        assert text in block, text
    assert "primary" not in _code(block)
    gear = _between(common, "OE.gear = () => {", "\n};\n")
    for text in (
        "document.createElementNS(NS, tag)",
        "fill: 'none', stroke: 'currentColor',",
        "'aria-hidden': 'true', focusable: 'false'",
    ):
        assert text in gear, text
    assert "innerHTML" not in _code(block + gear)
    assert "// WORDING: r5n\nOE.SETTINGS = 'Settings';" in common


def test_s1_the_style_puts_it_at_the_bottom_left_of_one_size() -> None:
    """S1: the block stands 16 px under the card, the button an inline row of
    the gear and the word, its gear at the cards' left edge (the text button's
    5 px padding taken back), a 16 px gear; the box a white card under it,
    hidden while closed; under 480 px a row's control goes under its words."""
    style = _between(_read("common.js"), "OE.STYLE = `", "\n`;\n")
    for rule in (
        ".oe-settings { margin-top: 16px; }",
        ".oe-settings > button.oe-settings-button { display: inline-flex; "
        "align-items: center; gap: 6px; margin-left: -5px; }",
        ".oe-settings-button svg { width: 16px; height: 16px; flex: none; "
        "display: block; }",
        ".oe-settings-box[hidden] { display: none; }",
        ".oe-set-control { grid-column: 2; grid-row: 1 / span 2; "
        "justify-self: end; }",
        "  .oe-set-control { grid-column: 1; grid-row: auto; justify-self: start; "
        "margin-top: 4px; }",
    ):
        assert rule in style, rule
    assert _in_order(
        style,
        "@media (max-width: 480px) {\n  .oe-set-row {",
        "  .oe-set-control { grid-column: 1;",
    )


def test_s1_the_box_is_a_fold_read_at_each_opening() -> None:
    """S1: a press on the button opens the box or closes it; Escape on the
    button or in the open box closes it; closing empties the box (no word of
    it stays), and gives the focus back to the button; each opening reads the
    rows from the server, then the first control, the switch, takes the focus
    only while it is still on the button. Nothing in the box tells Claude
    anything."""
    block = _block()
    for text in (
        "button.addEventListener('click', () => { if (open) close(); else "
        "openBox(); });",
        "if (e.key !== 'Escape' || !open || e.defaultPrevented) return;",
        "got = await OE.call(app, 'olmoearth_app_settings');",
        "if (now === button || now === document.body || !now) "
        "OE.focusIfHere(toggle);",
    ):
        assert text in block, text
    closing = _between(block, "  const close = () => {", "\n  };\n")
    assert _in_order(
        closing,
        "turn += 1;",
        "button.setAttribute('aria-expanded', 'false');",
        "box.hidden = true;",
        "box.replaceChildren();",
        "OE.focusIfHere(button);",
    )
    code = _code(block)
    for gone in ("updateModelContext", "OE.tellClaude", "OE.say(", "sendMessage"):
        assert gone not in code, gone


# ---------------------------------------------------------------- S2: the box


def test_s2_the_box_holds_three_rows_each_with_one_control() -> None:
    """S2: the rows in order, each a short label, its state in words and one
    control: the switch (role switch, its state On or Off, named by its label
    and described by its line); Forget my yes only while pictures are
    allowed; Remove kept copies only while some maps are kept. Each press
    sends one of the server's changes, its words as the box's own."""
    common = _read("common.js")
    for name, text in (
        ("SET_ASTA", "Asta research skills"),
        ("SET_ON", "On"),
        ("SET_OFF", "Off"),
        ("SET_PICTURES", "Satellite pictures"),
        ("SET_FORGET", "Forget my yes"),
        ("SET_KEPT", "Maps kept on this computer"),
        ("SET_REMOVE", "Remove kept copies"),
        ("SET_SETUP", "setup"),
    ):
        assert f"// WORDING: r5n\nOE.{name} = '{text}';" in common, name
    pieces = _js_strings(_between(common, "OE.SET_ASTA_LINE = [", "];"))
    assert pieces[0] + "setup" + pieces[1] == ASTA_LINE
    assert f"OE.SETUP_URL = '{SETUP_URL}';" in common
    block = _block()
    assert _in_order(
        block,
        "row('asta', OE.SET_ASTA, on ? OE.SET_ON : OE.SET_OFF, toggle, astaLine),",
        "row('pictures', OE.SET_PICTURES, text(pictures.said), forget),",
        "row('kept', OE.SET_KEPT, text(kept.said), remove),",
    )
    for text in (
        "role: 'switch', class: 'oe-switch', 'aria-checked': String(on),",
        "'aria-labelledby': 'oe-set-asta', 'aria-describedby': 'oe-set-asta-line',",
        "if (pictures.allowed === true) {",
        "if (Number.isInteger(kept.count) && kept.count > 0) {",
        "change(toggle, { name: 'asta', on: !on })",
        "change(forget, { name: 'forget_pictures' })",
        "change(remove, { name: 'remove_kept' })",
        "const got = await OE.call(app, 'olmoearth_app_setting', args);",
        "say(now, OE.SETTING_NOT_CHANGED + err.message, true);",
    ):
        assert text in block, text
    named = set(re.findall(r"\{ name: '([a-z_]+)'", block))
    assert named == set(settings.NAMES)


def test_s2_the_setup_link_opens_through_the_host() -> None:
    """S2: the switch line's "setup" link is a text button the host opens
    (OE.open, quiet); when it does not, its slot shows the address to copy,
    as the docs link's does. Its anchor is the setup page's Options."""
    link = _between(_read("common.js"), "OE.setupLink = (app) => {", "\n};\n")
    for text in (
        "OE.el('button', { type: 'button', class: 'link' }, OE.SET_SETUP);",
        "await OE.open(app, OE.SETUP_URL, { quiet: true });",
        "OE.el('code', {}, OE.SETUP_URL)",
    ):
        assert text in link, text
    setup = (DOCS / "mcp-app-setup.md").read_text(encoding="utf-8")
    assert SETUP_URL.endswith("#options") and "\n## Options\n" in setup


# ------------------------------------------------------------- S2: the server


def test_s2_a_missing_or_broken_file_reads_as_the_defaults(_roots: Path) -> None:
    """S2: the settings are kept in <output root>/settings.json; a missing,
    broken or unreadable file, or a value of another kind, reads as the
    defaults: the switch off."""
    assert settings.settings_path() == _roots / "settings.json"
    assert settings.read() == {"asta": False}
    assert settings.asta_on() is False
    for broken in ("not json", "[true]", '{"asta": "yes"}', '{"asta": 1}', ""):
        settings.settings_path().write_text(broken, encoding="utf-8")
        assert settings.read() == {"asta": False}, broken
    settings.settings_path().write_bytes(b"\xff\xfe\x00")
    assert settings.asta_on() is False


def test_s2_the_switch_is_kept_mode_600_by_an_atomic_rename(_roots: Path) -> None:
    """S2: the switch is written by a rename (no part file left), mode 600,
    and the file holds its format and the switch alone."""
    settings.set_asta(True)
    path = settings.settings_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {"format": 1, "asta": True}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert settings.asta_on() is True
    settings.set_asta(False)
    assert settings.asta_on() is False
    assert sorted(p.name for p in _roots.iterdir()) == ["settings.json"]


def test_s2_a_failed_write_keeps_the_old_file_and_leaves_no_part(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: a write that fails at its rename raises, leaves the file as it was
    and no part file."""
    settings.set_asta(True)

    def refuse(*_args: Any) -> None:
        """A rename that fails, as on a full disk."""
        raise OSError("no room")

    monkeypatch.setattr(settings.os, "replace", refuse)
    with pytest.raises(OSError):
        settings.set_asta(False)
    assert settings.asta_on() is True
    assert sorted(p.name for p in _roots.iterdir()) == ["settings.json"]


def test_s2_the_pictures_row_says_the_day_of_the_yes(_roots: Path) -> None:
    """S2: not allowed (no answer, or a yes to older words): "Not allowed
    yet."; allowed: "You allowed them on <day>.", the day the answer was kept
    (its file's time, this computer's date)."""
    assert settings.pictures() == {"allowed": False, "said": "Not allowed yet."}
    imagery.set_on(True)
    moment = time.mktime((2026, 11, 18, 12, 0, 0, 0, 0, -1))
    os.utime(imagery.consent_path(), (moment, moment))
    assert settings.pictures() == {
        "allowed": True,
        "said": "You allowed them on 18 Nov 2026.",
    }
    older = json.dumps({"on": True, "version": "0123456789abcdef-fetch"})
    imagery.consent_path().write_text(older, encoding="utf-8")
    assert settings.pictures() == {"allowed": False, "said": "Not allowed yet."}


def test_s2_forget_my_yes_removes_the_stored_yes_alone(_roots: Path) -> None:
    """S2: Forget my yes removes the stored yes of imagery.is_on (so the next
    square asks again); the consent version is unchanged and nothing else
    changes; a second press does no harm."""
    imagery.set_on(True)
    version = imagery.consent_version()
    settings.set_asta(True)
    got = settings.change({"name": "forget_pictures"})
    assert imagery.is_on() is False
    assert not imagery.consent_path().exists()
    assert imagery.consent_version() == version
    assert got["pictures"] == {"allowed": False, "said": settings.PICTURES_NOT_ALLOWED}
    assert got["asta"] == {"on": True}
    again = settings.change({"name": "forget_pictures"})
    assert again["pictures"]["allowed"] is False


def test_s2_the_kept_row_counts_the_copies_still_read(_roots: Path) -> None:
    """S2: "None kept."; "1 map kept for 24 h, so it opens without
    downloading again."; "<n> maps kept for 24 h, so they open without
    downloading again.": only copies downloaded less than the keeping hours
    ago, whose record names their folder."""
    assert settings.kept() == {"count": 0, "said": "None kept."}
    _keep(_roots, RID_A, 1)
    assert settings.kept() == {
        "count": 1,
        "said": "1 map kept for 24 h, so it opens without downloading again.",
    }
    _keep(_roots, RID_B, 2)
    _keep(_roots, RID_OLD, 30)
    (_roots / studio_scores.KEPT_DIR / "not-a-copy").mkdir()
    assert settings.kept() == {
        "count": 2,
        "said": "2 maps kept for 24 h, so they open without downloading again.",
    }


def test_s2_the_kept_row_says_the_hours_kept(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: the row's hours are the keeping hours in effect (1.5 h); a copy
    older than them is not counted, nor any while no copy is kept (0)."""
    _keep(_roots, RID_A, 1)
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "1.5")
    assert settings.kept()["said"] == (
        "1 map kept for 1.5 h, so it opens without downloading again."
    )
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "0.5")
    assert settings.kept() == {"count": 0, "said": "None kept."}
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "0")
    assert settings.kept()["count"] == 0


def test_s2_reading_the_kept_row_removes_the_expired_copies(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2, after review: "None kept." never stands over copies still on
    this computer. Reading the row removes the copies kept the keeping
    hours ago or longer, as the next download would (studio_scores
    ._prune_kept), and every readable copy while the hours are 0; a copy
    still read stays, and a folder whose record cannot be read is left."""
    _keep(_roots, RID_A, 1)
    _keep(_roots, RID_OLD, 30)
    base = _roots / studio_scores.KEPT_DIR
    stray = base / "not-a-copy"
    stray.mkdir()
    (stray / "note.txt").write_text("kept", encoding="utf-8")
    assert settings.kept()["count"] == 1
    assert not (base / RID_OLD).exists()
    assert (base / RID_A / studio_scores.KEPT_ZIP).exists()
    assert (stray / "note.txt").exists()
    monkeypatch.setenv(studio_scores.KEEP_HOURS_ENV, "0")
    assert settings.kept() == {"count": 0, "said": settings.KEPT_NONE}
    assert not (base / RID_A / studio_scores.KEPT_ZIP).exists()
    assert (stray / "note.txt").exists()


def test_s2_a_failed_prune_still_reads_the_row(
    _roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2, after review: a prune that fails (a folder that cannot be read)
    does not fail the row: it is counted as before."""
    _keep(_roots, RID_A, 1)

    def refuse(*_args: Any) -> None:
        """A prune that cannot read the kept folder."""
        raise OSError("no access")

    monkeypatch.setattr(studio_scores, "_prune_kept", refuse)
    assert settings.kept()["count"] == 1


def test_s2_remove_kept_copies_removes_each_copy(_roots: Path) -> None:
    """S2: Remove kept copies removes every kept copy whose record names its
    folder, an expired one too (studio_scores.forget_kept for each), so none
    is read again; a folder whose record cannot be read is left as it is."""
    for rid, age in ((RID_A, 1), (RID_B, 2), (RID_OLD, 30)):
        _keep(_roots, rid, age)
    stray = _roots / studio_scores.KEPT_DIR / "not-a-copy"
    stray.mkdir()
    (stray / "note.txt").write_text("kept", encoding="utf-8")
    got = settings.change({"name": "remove_kept"})
    assert got["kept"] == {"count": 0, "said": settings.KEPT_NONE}
    for rid in (RID_A, RID_B, RID_OLD):
        assert not (_roots / studio_scores.KEPT_DIR / rid).exists(), rid
        assert studio_scores.read_kept(rid, 10**6) is None, rid
    assert (stray / "note.txt").exists()


def test_s2_a_change_the_box_does_not_know_is_refused(_roots: Path) -> None:
    """S2: a change of another name, or a switch given no true or false, is
    refused in plain words and writes nothing."""
    for args in ({}, {"name": "nope"}, {"name": 1}):
        with pytest.raises(ValueError, match="There is no such setting."):
            settings.change(args)
    for args in (
        {"name": "asta"},
        {"name": "asta", "on": "yes"},
        {"name": "asta", "on": 1},
    ):
        with pytest.raises(ValueError, match="on must be true or false"):
            settings.change(args)
    assert not settings.settings_path().exists()


def test_s2_the_rows_hold_no_key_id_path_or_place(_roots: Path) -> None:
    """S2: the three rows hold states and words only: no result id, no path,
    no file name; the settings file holds the switch alone."""
    imagery.set_on(True)
    _keep(_roots, RID_A, 1)
    settings.set_asta(True)
    got = settings.state()
    assert set(got) == {"asta", "pictures", "kept"}
    assert got["asta"] == {"on": True}
    assert set(got["pictures"]) == {"allowed", "said"}
    assert set(got["kept"]) == {"count", "said"}
    text = json.dumps(got)
    for gone in (
        RID_A,
        str(_roots),
        "settings.json",
        ".imagery_on",
        "studio_downloads",
    ):
        assert gone not in text, gone
    kept = json.loads(settings.settings_path().read_text(encoding="utf-8"))
    assert kept == {"format": 1, "asta": True}


def test_s2_the_harness_stand_ins_are_what_the_server_sends(_roots: Path) -> None:
    """r5-b J30: the browser harness's answers of the settings tools are the
    server's own: a yes kept on 18 Nov 2026 and two copies kept within the
    day, the switch off; then, after Forget my yes and Remove kept copies,
    each row's empty state; with the switch on, the same rows."""
    imagery.set_on(True)
    moment = time.mktime((2026, 11, 18, 12, 0, 0, 0, 0, -1))
    os.utime(imagery.consent_path(), (moment, moment))
    _keep(_roots, RID_A, 1)
    _keep(_roots, RID_B, 3)
    assert settings.state() == HARNESS_STATE
    on = settings.change({"name": "asta", "on": True})
    assert on == {**HARNESS_STATE, "asta": {"on": True}}
    settings.change({"name": "asta", "on": False})
    settings.change({"name": "forget_pictures"})
    assert settings.change({"name": "remove_kept"}) == HARNESS_REMOVED


def test_s2_the_tools_are_app_only_and_quiet() -> None:
    """S2: olmoearth_app_settings (no argument) and olmoearth_app_setting
    (name, and on for the switch) are views' tools (APP_TOOLS, each with its
    handler), quiet (a failure, too, tells the model nothing) and not tools
    of the app's own for the model."""
    assert server.SETTINGS_TOOLS == {"olmoearth_app_settings", "olmoearth_app_setting"}
    for name in server.SETTINGS_TOOLS:
        assert name in server.APP_TOOLS and name in server.QUIET_APP_TOOLS
        assert name not in server.OWN_TOOLS
        handler = "_app_" + name.removeprefix("olmoearth_app_")
        assert callable(getattr(server.AppServer, handler, None)), handler
    assert server.APP_TOOLS["olmoearth_app_settings"][1] == server._NO_ARGS
    schema = server.APP_TOOLS["olmoearth_app_setting"][1]
    assert schema["required"] == ["name"]
    assert schema["properties"]["name"]["enum"] == [
        "asta",
        "forget_pictures",
        "remove_kept",
    ]
    assert schema["properties"]["on"] == {"type": "boolean"}
    assert server.HOST_REPORT_TOOL in server.QUIET_APP_TOOLS
    assert server.IMAGERY_TOOLS < server.QUIET_APP_TOOLS


@pytest.mark.asyncio
async def test_s2_the_tools_are_listed_to_a_panel_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: a client without panels is not listed the two tools; one with
    panels gets them app-only (visibility app)."""
    srv = server.build_server(_app())
    plain = {t.name for t in await _list(srv)}
    assert not plain & server.SETTINGS_TOOLS
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    tools = {t.name: t for t in await _list(srv)}
    for name in server.SETTINGS_TOOLS:
        assert tools[name].meta is not None
        assert tools[name].meta["ui"]["visibility"] == ["app"], name


@pytest.mark.asyncio
async def test_s2_a_panels_calls_read_and_change_and_tell_the_model_nothing(
    _roots: Path,
) -> None:
    """S2: a read answers the three rows to the panel; a change answers them
    after the change; the model reads {"ok": true} from each, a refusal too
    (quiet), whose words go to the panel alone and which changes nothing."""
    srv = server.build_server(_app())
    read = await _call(srv, "olmoearth_app_settings", {})
    assert read.isError is False
    assert json.loads(_text(read)) == {"ok": True}
    assert read.structuredContent == settings.state()
    changed = await _call(srv, "olmoearth_app_setting", {"name": "asta", "on": True})
    assert changed.structuredContent is not None
    assert changed.structuredContent["asta"] == {"on": True}
    assert json.loads(_text(changed)) == {"ok": True}
    assert settings.asta_on() is True
    refused = await _call(srv, "olmoearth_app_setting", {"name": "nope"})
    assert refused.isError is True
    assert json.loads(_text(refused)) == {"ok": True}
    assert refused.structuredContent is not None
    assert refused.structuredContent["error"] == settings.UNKNOWN_SETTING
    assert settings.asta_on() is True


# --------------------------------------------------------------------------- S3


@pytest.mark.asyncio
async def test_s3_the_start_card_gets_asta_only_while_the_switch_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: the start card's view data carries asta (true) only while the
    switch is on; its jobs are the same either way, and the model's text never
    says Asta."""
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    off = await _call(srv, "olmoearth_start", {})
    assert "asta" not in _view(off)
    settings.set_asta(True)
    on = await _call(srv, "olmoearth_start", {})
    assert _view(on)["asta"] is True
    keys = [j["key"] for j in server.start_jobs()]
    assert [j["key"] for j in _view(on)["jobs"]] == keys
    assert [j["key"] for j in _view(off)["jobs"]] == keys
    for told in (off, on):
        assert not ASTA.search(_text(told))


@pytest.mark.asyncio
async def test_s3_no_instruction_tool_description_or_prompt_says_asta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: no server instruction, no listed tool's description and no job of
    the start card's menu says Asta; the only "asta" in what the server lists
    is the switch's name in olmoearth_app_setting's schema."""
    assert not ASTA.search(server.server_instructions(build_default_registry()))
    monkeypatch.setattr(server, "client_renders_views", lambda _srv: True)
    srv = server.build_server(_app())
    for tool in await _list(srv):
        assert not ASTA.search(tool.description or ""), tool.name
        schema = json.dumps(tool.inputSchema)
        if tool.name == "olmoearth_app_setting":
            assert len(ASTA.findall(schema)) == 1
        else:
            assert not ASTA.search(schema), tool.name
    for job in server.START_JOBS:
        assert not ASTA.search(json.dumps(job)), job["key"]
    for note in (
        server.START_CARD_NOTE,
        server.START_TEXT_NOTE,
        server.START_SIGNED_OUT_CARD,
        server.START_SIGNED_OUT_TEXT,
    ):
        assert not ASTA.search(note), note


def test_s3_the_tile_says_what_it_does_and_sends_one_message() -> None:
    """S3: the tile's label and line (85 characters at most, as every tile's)
    and its one message: Ai2's Asta skills, ask what to search first, the
    quick search, say what a long run does and how long and wait for a yes,
    and the setup page when the skills are missing."""
    page = _read("start.html")
    job = _between(page, "const ASTA_JOB = {", "\n};\n")
    assert "  key: 'asta',\n" in job
    assert _js_strings(_between(job, "label: ", ",\n")) == [TILE_LABEL]
    assert _js_strings(_between(job, "line: ", ",\n")) == [TILE_LINE]
    assert len(TILE_LINE) <= 85
    message = _between(job, "message: ", "+ OE.SETUP_URL")
    assert "".join(_js_strings(message)) == TILE_MESSAGE
    assert job.rstrip().endswith("+ OE.SETUP_URL,")
    assert "main" not in job
    assert "// WORDING: r5n\nconst ASTA_JOB = {" in page


def test_s3_the_tile_comes_only_with_the_switch_and_needs_no_studio() -> None:
    """S3: the tile is the jobs list's last row, under "Other jobs" (the
    heading drawn for it when no other job is), a job row that is on signed in
    or signed out; the switch's state comes from the tool's view data and from
    the card's Settings box. After review: only the tile (and the heading
    when no other job is under it) is added or removed in place, so every
    other row keeps its button, its press under way and its slot."""
    page = _read("start.html")
    rows = _between(page, "function jobRows(app, jobs, on) {", "\n}\n")
    assert "others.length || astaOn ? otherJobsHead() : null," in rows
    assert (
        "// WORDING: held for the reader test\nconst otherJobsHead = () => "
        "OE.el('li', { class: 'jobs-head' }, OE.el('h2', {}, 'Other jobs'));" in page
    )
    assert _in_order(
        rows,
        "...others.map((job) => jobRow(app, job, on)),",
        "astaOn ? astaRow(app) : null,",
    )
    tile = _between(page, "function astaRow(app) {", "\n}\n")
    assert "const row = jobRow(app, ASTA_JOB, true, async (slot) => {" in tile
    assert "row.classList.add('oe-asta');" in tile
    assert "let astaOn = false;" in page
    result = _between(page, "  onResult: (params, app) => {", "\n  },\n")
    assert _in_order(result, "astaOn = data.asta === true;", "render(app, data);")
    hook = _between(page, "OE.onSettings = (state) => {", "\n};\n")
    assert _in_order(
        hook,
        "settingsTold += 1;",
        "const want = Boolean(state && state.asta && state.asta.on === true);",
        "if (want === astaOn) return;",
        "if (!list || !jobsDrawn) return;",
        "const alone = jobsDrawn.jobs.length < 2;",
        "const row = list.querySelector(':scope > li.oe-asta');",
        "if (alone && !list.querySelector(':scope > li.jobs-head')) "
        "list.append(otherJobsHead());",
        "list.append(astaRow(jobsDrawn.app));",
        "row.remove();",
        "if (head) head.remove();",
        "offWords();",
    )
    assert "replaceChildren" not in hook and "jobRows(" not in hook
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert _in_order(
        render, "jobsDrawn = null;", "jobsDrawn = { app, jobs, on };", "offWords();"
    )


def test_s3_signed_out_the_words_over_the_tiles_leave_the_tile_out() -> None:
    """S3, after review: signed out with the tile shown (it needs no
    sign-in), the card's line and the line over the tiles say the sign-in is
    for the Studio jobs; with the tile gone they are render's own words again
    (the r5-e lines, which the older tests pin in render). Neither new line
    says Asta, so no face says it while the switch is off; signed in, or with
    no jobs, nothing is changed."""
    page = _read("start.html")
    for name, text in (
        ("START_NEXT_STUDIO", START_NEXT_STUDIO),
        ("SIGN_IN_STUDIO_JOB", SIGN_IN_STUDIO_JOB),
    ):
        assert f"// WORDING: r5n\nconst {name} = '{text}';" in page, name
        assert not ASTA.search(text), name
    words = _between(page, "function offWords() {", "\n}\n")
    assert _in_order(
        words,
        "if (!jobsDrawn || jobsDrawn.on) return;",
        "root.querySelector(':scope > .oe-lead p.next');",
        "next.textContent = astaOn ? START_NEXT_STUDIO : START_LEAD.next;",
        "const line = list ? list.previousElementSibling : null;",
        "line.textContent = astaOn ? SIGN_IN_STUDIO_JOB : 'Sign in to start a job.';",
    )
    render = _between(page, "function render(app, data) {", "\n}\n")
    assert "on ? null : OE.el('div', {}, 'Sign in to start a job.')," in render


def test_s3_every_start_card_follows_the_switch() -> None:
    """S3, after review: a start card reads the switch itself
    (olmoearth_app_settings) once when it opens, beside the status read, and
    again each time its panel is shown or takes the focus (a card reopened
    from the history, or one still open while the switch turned in another
    card's box); an answer after a newer state draws nothing and a failed read
    changes nothing. The listeners go when Claude closes the card. The tile's
    press reads the switch first: off, the tile goes, nothing is sent and the
    focus goes to the card's one contained button; a failed read sends as
    before."""
    page = _read("start.html")
    read = _between(page, "async function readSettings(app) {", "\n}\n")
    assert _in_order(
        read,
        "const seen = settingsTold;",
        "try { state = await OE.call(app, 'olmoearth_app_settings'); } "
        "catch (_) { return; }",
        "if (seen === settingsTold && state && typeof state === 'object') "
        "OE.onSettings(state);",
    )
    result = _between(page, "  onResult: (params, app) => {", "\n  },\n")
    assert _in_order(
        result,
        "astaOn = data.asta === true;",
        "settingsTold += 1;",
        "settingsApp = app;",
        "render(app, data);",
        "recheck(app, data);",
        "readSettings(app);",
    )
    assert "olmoearth_app_settings" not in _between(
        page, "async function recheck(app, data) {", "\n}\n"
    )
    for text in (
        "if (settingsApp && document.visibilityState === 'visible') "
        "readSettings(settingsApp);",
        "document.addEventListener('visibilitychange', onShown);",
        "window.addEventListener('focus', onShown);",
        "  document.removeEventListener('visibilitychange', onShown);\n"
        "  window.removeEventListener('focus', onShown);\n",
    ):
        assert text in page, text
    assert "OE.cleanups.add(() => {\n  document.removeEventListener(" in page
    tile = _between(page, "function astaRow(app) {", "\n}\n")
    assert _in_order(
        tile,
        "try { state = await OE.call(app, 'olmoearth_app_settings'); } "
        "catch (_) { state = null; }",
        "!(state.asta && state.asta.on === true)) {",
        "OE.onSettings(state);",
        "OE.focusIfHere(root.querySelector('button:not(:disabled):not(.link)'));",
        "return false;",
        "return OE.sendJob(app, ASTA_JOB.message, slot);",
    )
    row = _between(page, "function jobRow(app, job, on, press = null) {", "\n}\n")
    assert "if (busy || button.disabled || e.detail > 1) return;" in row
    assert (
        "try { await (press ? press(slot) : OE.sendJob(app, job.message, slot)); }"
        in row
    )


def test_s3_the_views_say_asta_only_in_the_box_and_the_tile() -> None:
    """S3: once comments are left out, the views' pages and scripts say Asta
    only in the box's words (common.js, from OE.SET_ASTA to its line) and in
    the start card's tile (start.html ASTA_JOB); no other view says it."""
    for path in sorted(views.VIEW_DIR.iterdir()):
        if path.suffix not in (".html", ".js"):
            continue
        code = _code(path.read_text(encoding="utf-8"))
        found = [m.start() for m in re.finditer("Asta", code)]
        if path.name == "start.html":
            low = code.index("const ASTA_JOB = {")
            high = code.index("\n};\n", low)
        elif path.name == "common.js":
            low = code.index("OE.SET_ASTA = ")
            high = code.index("OE.SET_SETUP = ")
        else:
            assert not found, path.name
            continue
        assert found, path.name
        assert all(low <= at < high for at in found), path.name


def test_s3_the_tile_is_not_a_prompt_of_the_menu() -> None:
    """S3: the tile is the card's alone: the start jobs (the menu's prompts
    and the model's list) are the same four as before."""
    assert [j["key"] for j in server.START_JOBS] == ["check", "first", "which", "run"]
    assert all(j["key"] != "asta" for j in server.start_jobs())


# --------------------------------------------------------------------------- S4


def test_s4_the_setup_page_says_how_to_get_the_skills() -> None:
    """S4: the setup page's Options say how to install Ai2's Asta skills in
    Claude Code, that they are for Claude Code only, that some commands need
    an Ai2 login, to turn Asta on in Settings, and that the switch only shows
    or hides the app's tile."""
    setup = (DOCS / "mcp-app-setup.md").read_text(encoding="utf-8")
    options = _between(setup, "\n## Options\n", "\n## ")
    for text in (
        "`/plugin marketplace add allenai/asta-plugins`",
        "`/plugin install asta-tools`",
        "Claude Code only (not the chat in Claude Desktop)",
        "Ai2 login",
        "turn Asta on in **Settings**",
        "the switch only shows or hides the app's tile",
    ):
        assert text in options, text


def test_s4_the_panels_page_has_a_settings_section_and_names_the_tile() -> None:
    """S4: mcp-app.md has a Settings section in the shape of the panels'
    (what you see, what to press, what it means, when it fails), and its
    start card section names the tile."""
    page = (DOCS / "mcp-app.md").read_text(encoding="utf-8")
    section = _between(page, "\n## Settings\n", "\n## ")
    for text in (
        "**What you see.**",
        "**What to press.**",
        "**What it means.**",
        "**When it fails.**",
        "Asta research skills",
        "Forget my yes",
        "Remove kept copies",
    ):
        assert text in section, text
    start = _between(page, "\n## Start card\n", "\n## ")
    assert TILE_LABEL in start


def test_s4_usage_names_the_tools_and_the_file() -> None:
    """S4: the developer reference names the two tools and the settings file."""
    usage = (DOCS.parent / "spec" / "Usage.md").read_text(encoding="utf-8")
    section = _between(usage, "\n### Settings\n", "\n### ")
    for text in (
        "`olmoearth_app_settings`",
        "`olmoearth_app_setting`",
        "settings.json",
    ):
        assert text in section, text
    assert "| `settings.json` in the output root |" in usage


def test_the_changelog_says_each_item() -> None:
    """The round's marker in spec/changes/1.5.0.md is replaced by one plain line
    per item, each naming
    its item of the settings spec."""
    log = (DOCS.parent / "spec" / "changes" / "1.5.0.md").read_text(encoding="utf-8")
    assert "@@R5N-SETTINGS@@" not in log
    for number, item in (("J46", "S1"), ("J47", "S2"), ("J48", "S3"), ("J49", "S4")):
        line = _between(log, f"\n    - {number}: ", "\n")
        assert line.endswith(f"(Settings {item})."), item
