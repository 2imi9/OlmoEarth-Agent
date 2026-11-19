# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-j (18 Nov 2026), track HARNESS, as specs/r5-j-round.md gives it:
item B10, the end shots of 1_sign_in and of the projects scenarios are of a
state a user reaches. No change to the app.

The harness (scripts/mcp_views_e2e.py) reads no file under tests, and no
test runs it. So each test here pins the app's side of what the harness now
does before its end shot, as the views are on disk:

- (a) sign_in_steps presses "Use another key" a second time once it has
  checked that the sign-in it opens is outlined: the button toggles that
  sign-in, so the second press closes it, and "Continue in the chat" is on
  the account card's own row, not in the closed sign-in. 1_sign_in then shows
  the account card alone (it showed a second "Sign in to OlmoEarth Studio"
  card under it).
- (b) studio_steps (4_studio_projects) and studio_narrow_steps (4d) close the
  open project's "Make a new map" fold again with a press on its summary once
  its models are measured: the fold is a details closed by default, "Run on
  an area" is drawn only inside it, and a project's summary is drawn only by
  its own opener, so opening a second project leaves the first one's fold as
  it was (the harness finds one fold open and closes it).
"""

from __future__ import annotations

from olmoearth_agent.mcp_app import views


def _read(name: str) -> str:
    """A file of views/, as it is on disk."""
    return (views.VIEW_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- (a)


def test_b10a_a_second_press_closes_the_other_keys_sign_in() -> None:
    """B10 (a): "Use another key" starts closed and toggles the sign-in under
    the account card: a press with the sign-in open empties its slot and
    tells it closed (the harness's second press, before the end shot of
    1_sign_in); opened after a fresh sign-in, that sign-in is outlined (the
    harness's check before that press)."""
    page = _read("sign_in.html")
    starts_closed = (
        "const another = OE.el('button', { class: 'link', 'aria-expanded': "
        "'false' }, 'Use another key');"
    )
    assert starts_closed in page
    closes = (
        "if (slot.firstChild) { slot.replaceChildren(); "
        "another.setAttribute('aria-expanded', 'false'); return; }"
    )
    assert closes in page
    assert "OE.signInPanel(app, { primary: !signedInNow, onDone:" in page
    # What the harness reads to see it closed: the panel's heading and button.
    common = _read("common.js")
    assert "OE.el('h2', {}, 'Sign in to OlmoEarth Studio')," in common
    assert "OE.SIGN_IN_BUTTON = 'Open the sign-in page';" in common


def test_b10a_continue_in_the_chat_is_on_the_account_card() -> None:
    """B10 (a): "Continue in the chat", which the harness presses after the
    sign-in is closed again, is on the account card's own row (actions),
    beside "Use another key"; the slot that the second press empties comes
    after the card and holds only the other key's sign-in."""
    page = _read("sign_in.html")
    start = page.index("function accountCard(")
    card = page[start : page.index("function render(", start)]
    row = card.index("actions.append(OE.el('button', {")
    assert card.index("}, 'Continue in the chat'));") > row
    assert card.index("return [") > row
    tail = "OE.el('div', { class: 'spread' }, another, actions)),\n    slot,\n  ];"
    assert tail in card


# --------------------------------------------------------------------------- (b)


def test_b10b_the_models_fold_is_closed_until_pressed() -> None:
    """B10 (b): an open project's Models box is in the fold "Make a new map",
    a details of class oe-more with no open attribute (closed by default; a
    press on its summary opens it and a second one closes it), and "Run on an
    area" is drawn only in that box, so with the fold closed again no "Run on
    an area" shows (the end shots of 4_studio_projects and 4d)."""
    studio = _read("studio.html")
    assert "const MODELS_FOLD = 'Make a new map';" in studio
    assert "modelsBox ? OE.fold(MODELS_FOLD, modelsBox) : null);" in studio
    fold = (
        "OE.fold = (label, ...nodes) => OE.el('details', { class: 'oe-more' },\n"
        "  OE.el('summary', {}, label), ...nodes);"
    )
    assert fold in _read("common.js")
    assert studio.count("'Run on an area'") == 1
    models = studio.index("function modelsCard(")
    assert models < studio.index("'Run on an area'")
    assert studio.index("'Run on an area'") < studio.index("function projectSummary(")
    # Nothing in the projects card opens a fold by itself.
    for opens in (".open = ", "setAttribute('open'", "toggleAttribute('open'"):
        assert opens not in studio


def test_b10b_a_second_project_leaves_the_first_ones_fold() -> None:
    """B10 (b): a project's summary, with its fold, is drawn only by its own
    opener, into its own card's slot: opening a second project (the end of
    4_studio_projects) does not draw the first one's again, so the fold the
    harness opened there is still the one fold open, which it closes."""
    studio = _read("studio.html")
    assert studio.count("projectSummary(") == 2
    assert "slot.replaceChildren(projectSummary(app, project, out));" in studio
    assert "toggle = opener(app, p, title, slot, (open) => {" in studio
