# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r4-l (17 Nov 2026), each string as specs/r4-l-round.md gives it; the
views are read as text, as views.view_html() serves them (no test here runs
the scripts), and the Check card's zone states from check.py's records:

- L1: the figure's "Save the picture" button is drawn only where the host
  saves files (OE.canDownload; Claude offers no downloadFile), with no line
  in its place, and every download a view asks for is gated the same way;
- L2: past draws_max failed picks, a press retries those that have waited
  longest: check._zone_state gives a failed zone its failure's time
  (failed_at, as _failed writes it; a draw left drawing past STALE_SECONDS,
  the time it was claimed), and so does a press's answer (draw_view); the
  page sorts by it alone, earliest first, and never shows it or tells it to
  Claude; the line under the press says "waited longest";
- L3: an editor that stopped saving (its sheet changed under the panel),
  swapped away with no change left unsaved, clears its last toast about
  that change (reopenToast); a toast about an unsaved change stays.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import check, views

#: The figure's save button (result.html, the reader test's B/R13).
SAVE_PICTURE = "Save the picture (SVG file, no places in it)"

#: Two of the editor's toasts about a sheet that changed under the page
#: (sheet.js), word for word; a swap with nothing unsaved clears the last such
#: toast (r4-l L3).
REOPEN = "This sheet changed since this panel opened it. Reopen it to go on."
NOT_SAVED_REOPEN = (
    "Not saved: this sheet changed since the panel opened it. Reopen the sheet to"
    " go on."
)

#: A plan error that may pass on a second press (check.FINAL_ERRORS has no OSError).
NO_SPACE = "OSError: [Errno 28] No space left on device"

OFFER = {
    "offer": [
        {"labels": 20, "width": 0.2, "sentence": "Twenty.", "rate_note": "r"},
        {"labels": 40, "width": 0.1, "sentence": "Forty.", "rate_note": "r"},
    ],
    "refused": None,
}
FILES = ("result_epsg32610_0_native.tif", "result_epsg32733_1_native.tif")


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


def _code(page: str) -> str:
    """The text with its comments taken out."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", page, flags=re.S)


def _in_order(text: str, *pieces: str) -> bool:
    """Whether each piece is in text, each after the one before it."""
    at = [text.index(p) for p in pieces]
    return at == sorted(at)


@pytest.fixture(autouse=True)
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The write root (scores and outputs), a folder of the test's own."""
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(out))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    return out.resolve()


@pytest.fixture
def canned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counts offered without the package: 20 and 40, for every zone."""

    def fake(n_valid: int, widths: Any = check.CHECK_WIDTHS) -> dict[str, Any]:
        """A fresh copy of :data:`OFFER`."""
        answer: dict[str, Any] = json.loads(json.dumps(OFFER))
        return answer

    monkeypatch.setattr(check, "offers", fake)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """check's time now, which a test moves on."""
    now = [1_000_000.0]
    monkeypatch.setattr(check, "_now", lambda: now[0])
    return now


def _inner(root: Path) -> dict[str, Any]:
    """A scores_from_studio result of two zones, each with a scores file."""
    zones = []
    for i, file in enumerate(FILES):
        path = root / f"scores_{i}.json"
        path.write_text("{}", encoding="utf-8")
        zones.append(
            {
                "file": file,
                "scores_path": str(path),
                "pooled": {
                    "n_valid": 600,
                    "n_windows": 604,
                    "raster_check": {"sha256": f"{'a' * 63}{i}"},
                },
            }
        )
    return {"result_id": "res-1", "prediction_id": "pred-1", "zones": zones}


def _plan_answer(root: Path) -> dict[str, Any]:
    """A drawn plan's envelope."""
    return {
        "ok": True,
        "result": {
            "available": True,
            "design": "random",
            "budget": 20,
            "seed": 0,
            "labels_csv_path": str(root / "plan_to_label.csv"),
            "design_path": str(root / "plan.json"),
            "n_population": 600,
            "facts": [],
            "must_state": ["One sentence to state."],
            "forbidden_claims": [],
        },
    }


def _draw_path(root: Path, key: str) -> Path:
    """Where a check's draw record is."""
    return root / "checks" / f"{key}.draw.json"


def _rewrite(root: Path, key: str, **fields: Any) -> None:
    """A check's draw record with fields set (a value None: the field taken out)."""
    path = _draw_path(root, key)
    record = json.loads(path.read_text(encoding="utf-8"))
    for name, value in fields.items():
        if value is None:
            record.pop(name, None)
        else:
            record[name] = value
    path.write_text(json.dumps(record), encoding="utf-8")


# --------------------------------------------------------------------------- L1


def test_l1_the_figure_saves_only_where_the_host_saves_files() -> None:
    """r4-l L1: figureBlock draws its save button only when OE.canDownload
    (the K8 helper) is true; otherwise no button and nothing in its place:
    the figure's row holds the Full screen button, as before, and the gated
    button only. The button's words are kept, marked, and used once."""
    page = views.view_html("result")
    figure = _between(page, "function figureBlock(app, data,", "\n}\n")
    gate = (
        "  const canSave = typeof OE.canDownload === 'function'"
        " && OE.canDownload(app);\n"
    )
    row = (
        "    OE.el('div', { class: 'row', style: 'margin-top:12px' },\n"
        "      OE.fullscreenButton(app),\n"
        "      canSave && OE.el('button', {\n"
    )
    assert gate in figure and row in figure
    assert _in_order(
        figure,
        gate,
        "OE.el('img', { src, alt: data.figure.caption || 'Determination figure' }),",
        row,
        "await app.downloadFile({ contents: [{ type: 'resource', resource:"
        " { uri: 'file:///' + name, mimeType: 'image/svg+xml', text: svg } }] });",
        "}, SAVE_PICTURE)));",
    )
    # The row ends with the button: no line or other node in its place.
    assert figure.rstrip().endswith("}, SAVE_PICTURE)));")
    assert figure.count("OE.el(") == 5
    assert f"// WORDING: r4p2\nconst SAVE_PICTURE = '{SAVE_PICTURE}';" in page
    assert len(re.findall(r"\bSAVE_PICTURE\b", _code(_read("result.html")))) == 2
    # OE.canDownload as K8 wrote it: the client's method and the host's flag.
    can = _between(page, "OE.canDownload = (app) => {", "\n};\n")
    assert "if (!app || typeof app.downloadFile !== 'function') return false;" in can
    assert (
        "return Boolean(caps && typeof caps === 'object' && caps.downloadFile);" in can
    )


def test_l1_every_download_a_view_asks_for_is_gated() -> None:
    """r4-l L1: the views ask the host for a download in two places only, the
    figure's save button and the look-first CSV (K8), each after
    OE.canDownload(app) in the same block; no other page calls it."""
    calls = {
        name: len(re.findall(r"\bapp\.downloadFile\(", views.view_html(name)))
        for name in views.VIEWS
    }
    assert calls == {name: 2 if name == "result" else 0 for name in views.VIEWS}
    page = views.view_html("result")
    figure = _between(page, "function figureBlock(app, data,", "\n}\n")
    csv = _between(page, "  function csvBlock() {", "\n  }\n")
    for block in (figure, csv):
        assert block.count("app.downloadFile(") == 1
        assert block.index("OE.canDownload(app)") < block.index("app.downloadFile(")


# --------------------------------------------------------------------------- L2


def test_l2_a_failed_zone_says_when_it_failed(
    root: Path, canned: None, clock: list[float]
) -> None:
    """r4-l L2: a zone to draw has no failed_at; a failed one has its
    record's failed_at, as _failed wrote it; each failure writes a new one."""
    inner = _inner(root)
    first, second = check.offer_zones(inner)
    assert "failed_at" not in first and "failed_at" not in second
    key = first["check_key"]
    check.claim_draw(key, 20)
    clock[0] += 30
    record = check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    assert record["failed_at"] == check._stamp(clock[0])
    zone = check.offer_zones(inner)[0]
    assert (zone["state"], zone["retry"], zone["failed_at"]) == (
        "failed",
        True,
        record["failed_at"],
    )
    # _zone_state itself, as offer_zones calls it (the offer's raster digest).
    folder = check.checks_dir()
    direct = check._zone_state(folder, key, "a" * 63 + "0")
    assert direct["failed_at"] == record["failed_at"] and "raster_changed" not in direct
    assert check._zone_state(folder, second["check_key"], "a" * 63 + "1") == {
        "state": "to draw"
    }
    # Retried, it fails again later: the newer time.
    clock[0] += 120
    assert check.claim_draw(key, 20)["outcome"] == "retry"
    clock[0] += 30
    check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    again = check.offer_zones(inner)[0]
    assert again["failed_at"] == check._stamp(clock[0]) > zone["failed_at"]
    assert again["pressed_at"] == zone["pressed_at"]


def test_l2_a_failure_for_good_has_its_time_too(root: Path, canned: None) -> None:
    """r4-l L2: the failed state carries failed_at whether or not a press
    retries it (the card orders only the retries by it)."""
    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    check.claim_draw(key, 20)
    record = check.finish_draw(
        key, {"ok": False, "error": "ValueError: the scores file is gone"}
    )
    zone = check.offer_zones(inner)[0]
    assert (zone["state"], zone["retry"]) == ("failed", False)
    assert zone["failed_at"] == record["failed_at"] and zone["failed_at"].endswith("Z")


def test_l2_a_stopped_draw_failed_when_it_was_claimed(
    root: Path, canned: None, clock: list[float]
) -> None:
    """r4-l L2: busy under STALE_SECONDS (no failed_at); past it, failed with
    the time its press claimed it. A retry's claim moves that time on while
    its pressed_at keeps the first press's, so a draw that stops again waits
    behind the others."""
    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    pressed = check.claim_draw(key, 20)["draw"]["pressed_at"]
    assert pressed == check._stamp(clock[0])
    clock[0] += check.STALE_SECONDS - 1
    busy = check.offer_zones(inner)[0]
    assert busy["state"] == "busy" and "failed_at" not in busy
    clock[0] += 1
    stale = check.offer_zones(inner)[0]
    assert (stale["state"], stale["error"], stale["retry"]) == (
        "failed",
        check.STOPPED,
        True,
    )
    assert stale["failed_at"] == pressed
    clock[0] += 120
    assert check.claim_draw(key, 20)["outcome"] == "retry"
    claimed = clock[0]
    clock[0] += check.STALE_SECONDS
    again = check.offer_zones(inner)[0]
    assert again["error"] == check.STOPPED
    assert again["failed_at"] == check._stamp(claimed) != pressed
    assert again["pressed_at"] == pressed


@pytest.mark.parametrize("started", ["noon", float("nan"), 1e20, None])
def test_l2_a_stopped_draw_with_no_claim_time_gives_its_press_time(
    root: Path, canned: None, clock: list[float], started: Any
) -> None:
    """r4-l L2: a draw left drawing whose claim time cannot be read (text,
    NaN, a time gmtime refuses, none) is stopped, with its pressed_at as its
    time; with no pressed_at either, none (the page sorts it first)."""
    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    pressed = check.claim_draw(key, 20)["draw"]["pressed_at"]
    _rewrite(root, key, started_at=started)  # None: taken out
    stale = check.offer_zones(inner)[0]
    assert (stale["state"], stale["error"], stale["failed_at"]) == (
        "failed",
        check.STOPPED,
        pressed,
    )
    _rewrite(root, key, pressed_at=None)
    assert check.offer_zones(inner)[0]["failed_at"] is None


def test_l2_a_failed_record_with_no_time_gives_none(root: Path, canned: None) -> None:
    """r4-l L2: a failed record without a failed_at text (one written before
    it was kept, or not a text) gives None; the page sorts it first."""
    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    check.claim_draw(key, 20)
    check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    _rewrite(root, key, failed_at=None)
    assert check.offer_zones(inner)[0]["failed_at"] is None
    _rewrite(root, key, failed_at=5)
    assert check.offer_zones(inner)[0]["failed_at"] is None


def test_l2_a_drawn_zone_has_no_failure_time(root: Path, canned: None) -> None:
    """r4-l L2: a draw that failed and then was drawn keeps no failed_at, on
    the zone or its draw."""
    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    check.claim_draw(key, 20)
    check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    check.claim_draw(key, 20)
    check.finish_draw(key, _plan_answer(root))
    zone = check.offer_zones(inner)[0]
    assert zone["state"] == "drawn"
    assert "failed_at" not in zone and "failed_at" not in zone["draw"]


def test_l2_the_press_answer_carries_a_failures_time(root: Path, canned: None) -> None:
    """r4-l L2: draw_view gives the page a failed draw's failed_at (so a
    retry that fails again waits behind the others on the same card), None
    for a drawn one."""
    assert "failed_at" in check.VIEW_KEYS
    key = check.offer_zones(_inner(root))[0]["check_key"]
    check.claim_draw(key, 20)
    failed = check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    view = check.draw_view(failed, outcome="first")
    assert view["failed_at"] == failed["failed_at"] and view["failed_at"].endswith("Z")
    check.claim_draw(key, 20)
    drawn = check.finish_draw(key, _plan_answer(root))
    assert check.draw_view(drawn, outcome="retry")["failed_at"] is None


def test_l2_the_model_never_reads_a_failures_time(root: Path, canned: None) -> None:
    """r4-l L2: the time reaches the card's view data only; the zone the
    model reads (with panels or without) has no failed_at."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    inner = _inner(root)
    key = check.offer_zones(inner)[0]["check_key"]
    check.claim_draw(key, 20)
    check.finish_draw(key, {"ok": False, "error": NO_SPACE})
    zone = check.offer_zones(inner)[0]
    assert zone["failed_at"]
    for panels in (True, False):
        assert "failed_at" not in server._check_zone_for_model(
            zone, inner, views=panels
        )
    assert server._check_zone_for_view(zone)["failed_at"] == zone["failed_at"]


def test_l2_the_servers_times_sort_as_the_page_reads_them() -> None:
    """r4-l L2: the page takes a time only in check._stamp's one form, and
    such times sort as text in time order: the page compares, computes
    nothing."""
    page = _read("check.html")
    pattern = _after(
        page, "const failedAtOf = (stamp) => (typeof stamp === 'string'\n  && /", "/"
    )
    assert pattern == r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    moments = [0.0, 59.0, 3600.0, 86_399.0, 1_000_000.0, 1_763_300_000.0]
    stamps = [check._stamp(t) for t in moments]
    assert all(re.fullmatch(pattern[1:-1], s) for s in stamps)
    assert sorted(stamps) == stamps


def test_l2_the_press_retries_those_that_have_waited_longest() -> None:
    """r4-l L2: each zone's failedAt from the view data and from each press's
    answer; the retries sorted by it alone, earliest first, one with no time
    first, ties in the card's order, and at most draws_max of them sent, in
    the card's order; nothing else reads it."""
    page = views.view_html("check")
    model = _between(page, "function zoneModel(z, index) {", "\n}\n")
    assert "error: null, budget: null, failedAt: null," in model
    failed = _between(
        model, "  } else if (z.state === 'failed') {", "  } else if (offer.length"
    )
    assert "    m.failedAt = failedAtOf(z.failed_at);\n" in failed
    answer = _between(page, "  function takeAnswer(m, a) {", "\n  }\n")
    assert _in_order(
        answer,
        "if (a.state === 'failed') {",
        "m.failedAt = failedAtOf(a.failed_at);",
        "m.kind = a.retry && m.budget !== null ? 'retry' : 'failed';",
    )
    plan = _between(page, "  function sendPlan() {", "\n  }\n")
    order = (
        "    const waitedLonger = (a, b) => (a.failedAt === b.failedAt"
        " ? a.index - b.index\n"
        "      : a.failedAt === null ? -1 : b.failedAt === null ? 1"
        " : a.failedAt < b.failedAt ? -1 : 1);\n"
    )
    assert _in_order(
        plan,
        order,
        "const retries = models.filter((m) => m.kind === 'retry').sort(waitedLonger);",
        "const retrying = retries.slice(0, drawsMax);",
        "const chosen = new Set([...retrying, ...picks.slice(0, room)]);",
        "send: models.filter((m) => chosen.has(m)), retrying: retrying.length,",
    )
    # Read in those three places only: never shown, never told to Claude.
    card = _code(_read("check.html"))
    assert len(re.findall(r"\bfailed_at\b", card)) == 2
    # The model's field (null), its two assignments (zoneModel, takeAnswer)
    # and the six reads in sendPlan's comparator: nowhere else in the code.
    assert len(re.findall(r"\bfailedAt\b", card)) == 9
    for start in (
        "  function tellCard() {",
        "  function limitWords(",
        "  function updatePress() {",
        "  function fillZone(m) {",
        "  function fillParts() {",
        "  function fillFold() {",
        "  async function drawPicked() {",
    ):
        block = _between(page, start, "\n  }\n")
        assert "failedAt" not in block and "failed_at" not in block, start


def test_l2_the_line_says_those_that_have_waited_longest() -> None:
    """r4-l L2: limitWords, marked for squares, says the press retries the
    failed picks that have waited longest, never "the first". The sentence
    it renders for 18 failed picks and draws_max 16 (the spec's example) is
    checked in the browser (scripts/mcp_views_e2e.py, 14h)."""
    page = views.view_html("check")
    assert "  // WORDING: squares\n  function limitWords(" in page
    words = _between(page, "  function limitWords(", "\n  }\n")
    assert "first" not in words
    pieces = (
        "    const most = 'At most ' + drawsMax + ' parts in one press';\n",
        "    const retried = waiting\n"
        "      ? 'the ' + OE.counted(retrying, 'failed pick') + (retrying === 1"
        " ? ' that has' : ' that have') + ' waited longest'\n"
        "      : OE.counted(retrying, 'failed pick');\n",
        "    const rest = waiting ? ' ' + OE.counted(waiting, 'more failed pick')"
        " + (waiting === 1 ? ' waits' : ' wait')\n"
        "      + ' for the next press, or for the Check card opened again.' : '';\n",
        "    if (over <= 0) return waiting ? most + ': this press retries ' + retried"
        " + '.' + rest : '';\n",
    )
    assert _in_order(words, *pieces)
    assert "OE.counted = (n, noun) => n + ' ' + noun + (n === 1 ? '' : 's');" in page


# --------------------------------------------------------------------------- L3


def test_l3_a_stale_editor_swapped_away_clears_its_toast_to_reopen() -> None:
    """r4-l L3: every toast the editor shows about its sheet having changed
    under the page goes through reopenToast, which keeps the last one:
    readFresh's and adopt's, the map link's and the clear's (only when the
    server said the sheet changed), the clear's whose sheet cannot be read
    again, the blind switch's, and save's on a press of Save then. oeLeave
    clears the toast only while the editor stopped saving with no change
    left unsaved and that toast is still the one shown. Save's toast about
    changes not kept is not one (an unsaved change: it stays)."""
    editor = _between(_read("sheet.js"), "OE.sheetEditor = (app, data,", "\n};")
    assert "    reopenSaid: '',\n" in editor
    helper = _between(editor, "  function reopenToast(text) {", "\n  }\n")
    assert helper == (
        "  function reopenToast(text) {\n"
        "    state.reopenSaid = text;\n"
        "    OE.toast(text, true);"
    )
    code = _code(editor)
    assert code.count("state.reopenSaid = ") == 1
    assert code.count(f"reopenToast('{REOPEN}');") == 2
    assert f"if (!quiet) reopenToast('{NOT_SAVED_REOPEN}');" in code
    either = "{ markStale(); reopenToast(text); } else OE.toast(text, true);"
    assert code.count("if (said.startsWith(OE.CHANGED_UNDER_PAGE)) " + either) == 2
    for line in (
        "      const text = 'The map was not opened: ' + said + '.';\n",
        "          const text = 'Not cleared: ' + said + '.';\n",
        "          reopenToast('The ' + answersOf(column) + ' were removed, but the"
        " sheet could not be read again: '\n",
        "      if (state.stale && err.message === OE.CHANGED_UNDER_PAGE)"
        " reopenToast(text); else OE.toast(text, true);\n",
    ):
        assert line in code, line
    assert len(re.findall(r"\breopenToast\(", code)) == 8  # the helper, seven calls
    # After each markStale, the next toast is a reopenToast (r5-b J15: the
    # removal whose sheet could not be read again passes 'read').
    for after in re.split(r"markStale\((?:'read')?\);", code)[1:]:
        assert after.index("reopenToast(") < after.index("OE.toast("), after[:80]
    assert code.count("markStale();") == 5
    assert code.count("markStale('read');") == 1
    # Save's toast that the changes not saved are not kept stays a plain one.
    assert (
        "        if (!quiet) {\n          OE.toast('Not saved: this sheet changed"
        " since the panel opened it. '\n" in code
    )
    leave = _between(editor, "  wrap.oeLeave = () => {", "\n  };\n")
    assert leave == (
        "  wrap.oeLeave = () => {\n"
        "    if (!state.stale || pending()) return;\n"
        "    const bar = document.getElementById('oe-toast');\n"
        "    if (bar && state.reopenSaid && bar.textContent === state.reopenSaid)"
        " OE.toast('');"
    )
    # oeSettle and oeUnsaved as they were.
    assert "  wrap.oeUnsaved = () => (state.stale ? pending() : 0);\n" in editor


def test_l3_the_check_card_calls_it_once_the_swap_is_done() -> None:
    """r4-l L3: openZone, the one place an editor is swapped for another,
    calls the old editor's oeLeave after the new one is in place, and only
    when no change was dropped (else its own toast counts them, as before;
    since r5-b J10 that toast is the part's own line, by the part's name)."""
    page = _read("check.html")
    swap = _between(
        page, "  async function openZone(m, { hadFocus = null } = {}) {", "\n  }\n"
    )
    assert _in_order(
        swap,
        "!(await current.oeSettle())",
        "const dropped = current && typeof current.oeUnsaved === 'function'"
        " ? current.oeUnsaved() : 0;",
        "action.replaceChildren(editor);",
        "    if (dropped > 0 && before && before.draw) {\n",
        "      OE.toast(before.line, true);\n"
        "    } else if (current && typeof current.oeLeave === 'function') {\n",
        "      current.oeLeave();\n    }\n",
    )
    assert swap.count("oeLeave") == 2
    assert page.count("oeLeave") == swap.count("oeLeave") + 1  # and its comment
    # No other page swaps an editor for another (the other editors only go).
    for name in ("result.html", "sheet.html", "common.js"):
        assert "oeSettle" not in _read(name), name
