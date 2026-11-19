# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-k (18 Nov 2026), track CONSENT, as specs/r5-k-round.md gives it:
K7, the owner's choice N1 (specs/r5-i-audit/owner_decisions.md and
specs/r5-g-audit/owner_decisions.md), the question before pictures.

- K7: its face is three short lines (``imagery.CONSENT_LINES``: what is
  sent, to whom, and that nothing goes to Studio or Claude; the first two
  rows of the panel at 780 px, the others one row each), sent as one text
  with line breaks (``imagery.CONSENT_ASK``) and drawn one line each. The
  two hosts by name, what each gets and that no key or account is used
  (``imagery.CONSENT_FULL``, the answer's new ``ask_full``) and the test
  scene's question, which calls them the same catalogue and the same file
  store, are in one closed fold, "What is sent, in full", between "Turn on
  pictures" and "Try the public test scene", on the label sheet's pane
  (sheet.js) and on a look-first list's pane (result.html) alike. Each
  sentence is true of imagery.py and s2_fetch.py. The words changed, so the
  consent version (a hash of the face and the fold) did: a yes given to the
  old words lapses once, and a yes given to these words counts.

No test here touches the network or runs a view's script; the views are
read as text. The browser harness (scripts/mcp_views_e2e.py) draws the
question on 3q, 10, 6b and 6h (ask_face_check).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

from olmoearth_agent.mcp_app import imagery, s2_fetch, views
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV

#: K7: the face's three lines, word for word: what is sent, to whom, and
#: that nothing goes to Studio or Claude.
WHAT = (
    "Show satellite (Sentinel-2) pictures of each square? To fetch them, this "
    "computer sends the box around the map, rounded out to 0.1 degree, and the "
    "map's months."
)
TO = (
    "They go to Earth Search, a public catalogue, and its file store; both see "
    "your IP address."
)
NOT_SENT = "Nothing is sent to Studio or to Claude."
#: K7 (the round's review D1): the longest each face line may be. The first,
#: the question and what is sent, fills two rows of the card at 780 px (about
#: 96 characters a row); the other two fill one row each.
LONGEST = (165, 90, 90)
#: K7: the fold's hosts and what each gets, word for word.
FULL = (
    f"Earth Search's catalogue is {s2_fetch.CATALOGUE_HOST}: it gets the box and "
    "the months, once for all the squares. Its file store is "
    f"{s2_fetch.BUCKET_HOST}: it gets the requests for parts of image files. No "
    "key and no account are used."
)
#: K7: the fold's words in the panel.
FOLD = "What is sent, in full"
#: The question as round r5-j asked it, one paragraph with both hosts: a yes
#: given to it lapses once.
OLD_ASK = (
    "Show satellite (Sentinel-2) pictures of each square? They come from Earth "
    "Search, a public image archive. To fetch them, this computer sends two "
    f"things. To the archive's catalogue ({s2_fetch.CATALOGUE_HOST}): the map's "
    "rough outline and its months, once per sheet or review set. To its file "
    f"store ({s2_fetch.BUCKET_HOST}): requests for parts of image files. No key "
    "and no account are used. Both can see this computer's internet address "
    "(IP). Nothing is sent to Studio or to Claude."
)

#: Words no face holds (round r5-e's rule R7, and the owner's ruling 6).
_OFF_FACE = re.compile(
    r"\b(?:windows?|UTM|zones?|intervals?|strat(?:um|a)|design|budget"
    r"|recommended)\b",
    re.I,
)


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The part of text from start up to the first end after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An output root of its own, where the answer is kept."""
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(out))
    return out


# --------------------------------------------------------------------------- K7


def test_k7_the_face_is_three_short_lines() -> None:
    """K7: the face says what is sent, to whom, and that nothing goes to
    Studio or Claude, in three lines sent with line breaks between them; the
    first opens with the question the harness's stand-ins quote (r5-f F17).
    Each is short (the round's review D1): the first two rows of the card at
    780 px, the others one row each (LONGEST; the harness checks the rows).
    No line names a host, the test place, a key or an account, or holds a
    word no face says (R7), and the face as a whole is shorter than the old
    paragraph was without its test question."""
    assert imagery.CONSENT_LINES == (WHAT, TO, NOT_SENT)
    assert imagery.CONSENT_ASK == "\n".join((WHAT, TO, NOT_SENT))
    assert imagery.CONSENT_ASK.split("\n") == [WHAT, TO, NOT_SENT]
    assert WHAT.startswith("Show satellite (Sentinel-2) pictures of each square? ")
    for line, longest in zip(imagery.CONSENT_LINES, LONGEST, strict=True):
        assert len(line) <= longest, (len(line), line)
        assert not _OFF_FACE.search(line), line
        for word in (
            s2_fetch.CATALOGUE_HOST,
            s2_fetch.BUCKET_HOST,
            "Cape Peninsula",
            "key",
            "account",
        ):
            assert word not in line, (word, line)
    assert len(imagery.CONSENT_ASK) < len(OLD_ASK)


def test_k7_the_fold_names_the_hosts_and_what_each_gets() -> None:
    """K7: the fold says each host by name and what it gets, once for all the
    squares, and that no key or account is used; neither host is on the face
    any more. The test scene's questions, drawn in the same fold right after
    it, name neither host again but call them by the fold's own names, the
    same catalogue and the same file store (the round's review D3): no text
    of the fold says "bucket", "extent" or the IP sentence the face has."""
    assert imagery.CONSENT_FULL == FULL
    assert (imagery.CATALOGUE_HOST, imagery.BUCKET_HOST) == (
        s2_fetch.CATALOGUE_HOST,
        s2_fetch.BUCKET_HOST,
    )
    tests = (imagery.TEST_ASK, imagery.REVIEW_TEST_ASK)
    for host in (s2_fetch.CATALOGUE_HOST, s2_fetch.BUCKET_HOST):
        assert host in imagery.CONSENT_FULL
        assert host not in imagery.CONSENT_ASK
        assert all(host not in ask for ask in tests), host
    assert "once for all the squares" in imagery.CONSENT_FULL
    assert f"Its file store is {s2_fetch.BUCKET_HOST}:" in imagery.CONSENT_FULL
    assert imagery.CONSENT_FULL.endswith("No key and no account are used.")
    assert not _OFF_FACE.search(imagery.CONSENT_FULL)
    for ask in tests:
        assert ask.startswith(imagery.TEST_WHY + " ")
        assert "the same catalogue" in ask and "the same file store" in ask
        assert not _OFF_FACE.search(ask), ask
    for text in (imagery.CONSENT_FULL, *tests):
        for word in ("bucket", "extent", "IP"):
            assert word not in text, (word, text)


def test_k7_each_sentence_is_true_of_the_fetch() -> None:
    """K7: what the lines say is what imagery.py and s2_fetch.py do. The
    catalogue gets the map's extent, rounded outward to 0.1 degree (the
    default step, which the map's call does not change), and the months (one
    search per month), from the record once a sheet or review set was
    searched; the test place's box goes the same way, rounded; the file
    store gets byte-range requests for parts of image files; every URL is
    checked to be on exactly one of the two hosts; the client carries no
    key, account or cookie, takes no proxy or credentials from the
    environment (so both hosts see this computer's address) and follows no
    redirect. What goes over the wire is checked by test_mcp_imagery.py's
    fake Earth Search, whose handler refuses any authorization or cookie
    header."""
    assert inspect.signature(s2_fetch.rounded).parameters["step"].default == 0.1
    assert re.search(
        r"extent = s2_fetch\.rounded\(\s*warp\.transform_bounds\([^()]*\)\s*\)",
        inspect.getsource(imagery._load_map),
    )
    test_scene = inspect.getsource(imagery._public_test_scene)
    assert "s2_fetch.rounded(box.lonlat())" in test_scene
    body = inspect.getsource(s2_fetch.search_body)
    assert '"bbox": [round(v, 1) for v in bbox],' in body
    assert '"datetime": f"{iso(start)}/{iso(end)}",' in body
    scenes = inspect.getsource(imagery._scenes)
    assert "for label, lo, hi in months:" in scenes
    assert 'search.get("key") != key' in scenes
    fetch = inspect.getsource(s2_fetch)
    assert "if not allowed(url, CATALOGUE_HOST):" in fetch
    assert "if not allowed(href, BUCKET_HOST) or z < a:" in fetch
    assert 'headers={"Range": f"bytes={a}-{z}"},' in fetch
    client_source = inspect.getsource(s2_fetch._default_client)
    assert "follow_redirects=False," in client_source
    assert "trust_env=False," in client_source
    client = s2_fetch._default_client()
    try:
        assert client.auth is None and not client.cookies
        assert "authorization" not in client.headers
        assert "cookie" not in client.headers
    finally:
        client.close()
    assert s2_fetch.allowed(
        f"https://{s2_fetch.CATALOGUE_HOST}/v1/search", s2_fetch.CATALOGUE_HOST
    )
    assert not s2_fetch.allowed(
        f"https://{s2_fetch.BUCKET_HOST}/v1/search", s2_fetch.CATALOGUE_HOST
    )


def test_k7_the_version_is_raised_once_and_a_new_yes_counts(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K7: the version is a hash of the face and the fold (and "-fetch"), not
    the old paragraph's: a yes kept with the old version reads as off, so the
    question is asked once more; the yes given then is kept with the new
    version and counts on every later read. A change of the fold alone asks
    again too, and the yes counts once the words are back."""
    old = hashlib.sha256(OLD_ASK.encode("utf-8")).hexdigest()[:16] + "-fetch"
    new = imagery.consent_version()
    asked = imagery.CONSENT_ASK + "\n\n" + imagery.CONSENT_FULL
    assert new == hashlib.sha256(asked.encode("utf-8")).hexdigest()[:16] + "-fetch"
    assert new != old and imagery.consent_version() == new
    imagery.consent_path().write_text(
        json.dumps({"on": True, "version": old}), encoding="utf-8"
    )
    assert imagery.is_on() is False
    assert imagery.status() == {
        "on": False,
        "ask": imagery.CONSENT_ASK,
        "version": new,
    }
    assert imagery.set_on(True)["on"] is True
    kept = json.loads(imagery.consent_path().read_text(encoding="utf-8"))
    assert kept == {"on": True, "version": new}
    assert imagery.is_on() is True
    with monkeypatch.context() as patch:
        patch.setattr(imagery, "CONSENT_FULL", imagery.CONSENT_FULL + " Changed.")
        assert imagery.is_on() is False
    assert imagery.is_on() is True
    assert imagery.consent_path().parent == root


def test_k7_the_answer_carries_the_face_the_fold_and_the_test_question() -> None:
    """K7: while pictures are off the pane is answered the face, the fold and
    the test scene's question (a sheet's, or a review set's own); a sheet's
    window with pictures off reads no file and gets the same answer."""
    want = {
        "ok": True,
        "needs_consent": True,
        "ask": imagery.CONSENT_ASK,
        "ask_full": imagery.CONSENT_FULL,
        "test_ask": imagery.TEST_ASK,
    }
    assert imagery._ask_answer() == want
    assert imagery._ask_answer(imagery.REVIEW_TEST_ASK) == {
        **want,
        "test_ask": imagery.REVIEW_TEST_ASK,
    }
    assert imagery.window_image("no.csv", "h", sheet_id="s", on=False) == want


def test_k7_the_label_sheets_pane_draws_three_lines_then_the_fold() -> None:
    """K7 on the label sheet: the question's card draws the server's lines one
    each (split at the line breaks), then "Turn on pictures" (its row the one
    r5-f F01 keeps in view), then the closed fold "What is sent, in full"
    with the hosts and the test question at the small size in the text's
    colour (no longer grey on the face), where that grey paragraph stood,
    then last "Try the public test scene", so its press comes after the fold
    that says what it sends (the round's review D2). The shared parts are
    sheet.js's, which result.html takes too."""
    script = _read("sheet.js")
    assert f"// WORDING: r5k K7\nOE.ASK_FOLD = '{FOLD}';\n" in script
    lines = _between(script, "OE.askLines = (answer) =>", "\nOE.askFold")
    assert "String((answer && answer.ask) || '').split('\\n')" in lines
    assert "class: 'oe-ask-line'" in lines
    fold = _between(script, "OE.askFold = (answer) => {", "\n};\n")
    assert "[answer && answer.ask_full, answer && answer.test_ask]" in fold
    assert (
        "OE.el('details', { class: 'oe-ask-full' }, OE.el('summary', {}, "
        "OE.ASK_FOLD)," in fold
    )
    assert "OE.el('div', { class: 'small', style: 'margin-top:8px' }, s)" in fold
    assert "faint" not in fold and "open" not in fold
    card = _between(script, "    if (answer.needs_consent) {", "\n      return;\n    }")
    assert "OE.el('div', { class: 'card summary stack oe-ask' }," in card
    order = [
        card.index("OE.askLines(answer),"),
        card.index("'Turn on pictures'"),
        card.index("OE.askFold(answer),"),
        card.index("'Try the public test scene')) : null));"),
    ]
    assert order == sorted(order)
    assert "answer.ask)" not in card and "faint" not in card
    assert (
        "        OE.el('div', { class: 'row', style: 'justify-content:flex-end', "
        "'data-ask-yes': '' },\n"
        "          OE.el('button', { onclick: () => setPictures(true) }, "
        "'Turn on pictures')),\n" in card
    )


def test_k7_the_look_first_pane_draws_the_same() -> None:
    """K7 on a look-first list's pane (result.html): the same lines, buttons
    and fold in the same order (the fold between the two buttons), its
    buttons with their own words and focus keys."""
    page = _read("result.html")
    card = _between(
        page, "      else if (answer.needs_consent) {", "\n      } else if (!answer.ok)"
    )
    assert "OE.el('div', { class: 'card summary stack oe-ask' }," in card
    order = [
        card.index("OE.askLines(answer),"),
        card.index("REVIEW_WORDS.turnOn"),
        card.index("OE.askFold(answer),"),
        card.index("REVIEW_WORDS.test)) : null));"),
    ]
    assert order == sorted(order)
    assert "answer.ask)" not in card and "faint" not in card
    assert "turnOn: 'Turn on pictures'," in page
    assert "test: 'Try the public test scene'," in page


def test_k7_a_press_on_the_folds_heading_keeps_the_focus() -> None:
    """K7: on the sheet's face a click (or Enter) on a fold's heading leaves
    the focus there; the face's own click handler took it to the face, so a
    keyboard user who opened "What is sent, in full" lost the place."""
    script = _read("sheet.js")
    handler = _between(script, "  faceEl.addEventListener('click', (e) => {", "\n  });")
    assert "!t.closest('button, select, input, textarea, a, label, summary')" in handler
