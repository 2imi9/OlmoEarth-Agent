# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-i, track HARNESS (B9): the browser harness's stand-ins are what
the server sends. No change to the app.

The harness (scripts/mcp_views_e2e.py) reads no file under tests, and no
test runs it. So each test here pins the app's side of a stand-in the
harness now copies word for word, as r5-f F17's test does:

- (a) the Check card's labelling rule as ``server._check_rule`` gives it for
  a map of presence and absence. That is the owner's rule with the box until
  a part is drawn or being drawn, then settled; a map that needs a rule gets
  one that is not set. ``check_card`` now sends it by default; it sent none,
  so 13, 12b, 12c, 14 and 14b drew no rule over the press. Every press reply
  carries it too (``server._press_rule_view``), and the harness's now do.
- (b) the pictures question whole, as ``imagery._ask_answer`` gives it to a
  label sheet and to a review set's pane. The look-first scenarios (6b, 6h,
  6j) now send it; they sent the first sentence alone.
- (c) 17h's result is of design 'confidence', a stratified pick: the
  scenario is now 17h_estimate_stratified (it was 17h_estimate_not_random).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

mcp = pytest.importorskip("mcp")

from olmoearth_agent.mcp_app import check, imagery, review, server, views  # noqa: E402

#: The harness's owner's rule and its scope (OWNER_RULE, OWNER_SCOPE): the
#: app's general rule since round r5-l (L1), worded by round r5-m B1 (what the
#: map is meant to find).
OWNER_RULE = (
    "Present: on at least one clear date shown, what the map is meant to find "
    "covers more than half of the outlined square. Absent: on every clear date "
    "shown, it covers half or less."
)
OWNER_SCOPE = "This rule came with the app, for any map of present and absent."
#: check_card's rule by default with no zone drawn (RULE_OWNER_VIEW).
RULE_OWNER_VIEW = {
    "set": True,
    "text": OWNER_RULE,
    "source": "owner",
    "scope": OWNER_SCOPE,
    "replaceable": True,
}
#: The rule not set, where the gate is under test (RULE_NONE_VIEW, 12e).
RULE_NONE_VIEW = {
    "set": False,
    "text": None,
    "source": None,
    "scope": None,
    "replaceable": False,
}

#: The harness's pictures question: its first sentence (ASK, still used where
#: the question is not shot) and the whole (ASK_WHOLE). Since round r5-k K7
#: the whole is its three lines joined by line breaks (ASK_LINES in the
#: harness), and the hosts are in its fold (ASK_FULL, the answer's ask_full).
ASK_FIRST = "Show satellite (Sentinel-2) pictures of each square?"
ASK_WHAT = (
    "Show satellite (Sentinel-2) pictures of each square? To fetch them, this "
    "computer sends the box around the map, rounded out to 0.1 degree, and the "
    "map's months."
)
ASK_TO = (
    "They go to Earth Search, a public catalogue, and its file store; both see "
    "your IP address."
)
ASK_NOT = "Nothing is sent to Studio or to Claude."
ASK_WHOLE = "\n".join((ASK_WHAT, ASK_TO, ASK_NOT))
ASK_FULL = (
    "Earth Search's catalogue is earth-search.aws.element84.com: it gets the box "
    "and the months, once for all the squares. Its file store is "
    "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com: it gets the "
    "requests for parts of image files. No key and no account are used."
)
#: A sheet's test question whole (TEST_ASK_WHOLE). Since round r5-k K7 it
#: follows ASK_FULL in the fold and names neither host again: "the same
#: catalogue", "the same file store".
SHEET_TEST_ASK_WHOLE = (
    "Not sure yet? See how the pictures look on a public test place first. "
    "Nothing about your map is sent. It is a public place unrelated to this "
    "sheet (the Cape Peninsula's Atlantic shore, January to March 2024): that "
    "place's box, rounded, goes to the same catalogue, and requests for parts "
    "of image files to the same file store. Pictures stay off for the sheet."
)
#: A review set's test question whole (REVIEW_TEST_ASK_WHOLE, new in B9).
REVIEW_TEST_ASK_WHOLE = (
    "Not sure yet? See how the pictures look on a public test place first. "
    "Nothing about your map is sent. It is a public place unrelated to this "
    "review set (the Cape Peninsula's Atlantic shore, January to March 2024): "
    "that place's box, rounded, goes to the same catalogue, and requests for "
    "parts of image files to the same file store. Pictures stay off for it."
)
#: What a sheet's pane and a review set's pane are answered while pictures
#: are off (CONSENT_WHOLE, REVIEW_CONSENT_WHOLE).
CONSENT_WHOLE = {
    "ok": True,
    "needs_consent": True,
    "ask": ASK_WHOLE,
    "ask_full": ASK_FULL,
    "test_ask": SHEET_TEST_ASK_WHOLE,
}
REVIEW_CONSENT_WHOLE = {
    "ok": True,
    "needs_consent": True,
    "ask": ASK_WHOLE,
    "ask_full": ASK_FULL,
    "test_ask": REVIEW_TEST_ASK_WHOLE,
}


def _source(*parts: str) -> str:
    """A source file of the package (olmoearth_agent/<parts>), as text."""
    return views.VIEW_DIR.parent.parent.joinpath(*parts).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- (a)


def test_b9a_the_rule_the_server_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    """B9 (a): with no rule pinned, a map of presence and absence gets the
    owner's rule, with the box (replaceable) while the rule is not settled
    and read only once it is; a map that needs a rule gets a rule not set
    (the gate). check_card sends the first two by default, the third where a
    scenario passes RULE_NONE_VIEW (12e)."""
    settled = {"now": False}
    monkeypatch.setattr(check, "map_rule", lambda result_id: None)
    # r5-l L1: no part drawn before the round (no draw record names a rule),
    # and no sheet of the map labelled under the earlier rule.
    monkeypatch.setattr(check, "map_app_rule", lambda result_id: None)
    monkeypatch.setattr(check, "result_app_rule", lambda result_id: None)
    monkeypatch.setattr(server, "_rule_settled", lambda result_id: settled["now"])
    assert server._check_rule(False, "r1") == RULE_OWNER_VIEW
    settled["now"] = True
    assert server._check_rule(False, "r1") == {**RULE_OWNER_VIEW, "replaceable": False}
    assert server._check_rule(True, "r1") == RULE_NONE_VIEW
    assert (imagery.READING_RULE, imagery.RULE_SCOPE) == (OWNER_RULE, OWNER_SCOPE)


def test_b9a_a_zone_drawn_or_busy_settles_the_rule() -> None:
    """B9 (a): check_card settles its rule when a zone's state is "drawn" or
    "busy", as the server does: check.result_settled counts a draw record
    that is drawn, or drawing and not stale, and the card's zone state shows
    a drawing record that is not stale as busy."""
    source = _source("mcp_app", "check.py")
    drawn = 'state == "drawn" or (state == "drawing" and not _stale(draw, now))'
    assert drawn in source
    busy = 'elif state == "drawing" and not _stale(draw, _now()):'
    after = source[source.index(busy) :].splitlines()[1]
    assert after.strip() == 'out = {"state": "busy"}'


def test_b9a_the_card_draws_a_rule_only_when_sent_one() -> None:
    """B9 (a): the Check card reads a missing rule as set with no text, so it
    draws no rule at all (the fixtures' artefact); the box for the map's own
    rule shows only beside the owner's rule while a part can still be
    picked, so a card whose parts are all drawn shows the rule read only
    (with_rule's last check in 12b and 14b)."""
    page = (views.VIEW_DIR / "check.html").read_text(encoding="utf-8")
    assert "set: r.set !== false," in page
    assert "replaceable: source === 'owner' && r.replaceable === true," in page
    assert ": owner && rule.replaceable && drawable().length ? 'replace'" in page


def test_b9a_every_press_reply_carries_the_rule() -> None:
    """B9 (a): every reply of olmoearth_app_check_draw carries the map's
    rule, a reply that refuses the typed rule too, and the Check card draws
    its gate again from it. The harness's press replies now send it: read
    only once a part of the map is drawn, with the box while none is."""
    source = _source("mcp_app", "server.py")
    assert 'return {"draws": answers, "rule": rule}' in source
    assert '"refused": check.RULE_SETTLED, "rule": rule}' in source
    page = (views.VIEW_DIR / "check.html").read_text(encoding="utf-8")
    assert "rule = ruleModel(got.rule);" in page


# --------------------------------------------------------------------------- (b)


def test_b9b_the_questions_whole_as_the_server_sends_them() -> None:
    """B9 (b): with pictures off, a sheet's pane is asked the whole question
    and the sheet's test question, a review set's pane the whole question and
    the review set's own test question; the harness's ASK is the whole
    question's first sentence only (6h's shot asked with no word of what is
    sent to whom)."""
    assert imagery._ask_answer() == CONSENT_WHOLE
    assert imagery._ask_answer(imagery.REVIEW_TEST_ASK) == REVIEW_CONSENT_WHOLE
    assert imagery.CONSENT_ASK == ASK_WHOLE
    assert ASK_WHOLE.startswith(ASK_FIRST + " ") and ASK_WHOLE != ASK_FIRST


def test_b9b_a_review_sets_pane_is_asked_the_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B9 (b): olmoearth_app_review_image with pictures off answers the
    review set's whole question (the fixture 6b, 6h and 6j now give it), once
    the review's key and rank are found; no picture is fetched."""
    asked: list[tuple[Any, Any]] = []

    def found(key: Any, rank: Any) -> tuple[Path, str, int]:
        """The review's row, found (a stand-in: no file is read)."""
        asked.append((key, rank))
        return Path("/out/r.jsonl"), "/out/r.tif", 0

    monkeypatch.setattr(review, "review_row", found)
    assert imagery.review_image("k" * 16, 1, on=False) == REVIEW_CONSENT_WHOLE
    assert asked == [("k" * 16, 1)]


# --------------------------------------------------------------------------- (c)


def test_b9c_17h_is_a_stratified_pick() -> None:
    """B9 (c): 17h's result is of design 'confidence' (its fold's Design
    row). The tools say that design stratifies the map by its confidence
    margin, and the server draws the squares and the random design's range
    row only for a random design: the card 17h shows is a stratified pick's,
    so the scenario is named 17h_estimate_stratified."""
    stratifies = "design='{design}' stratifies the map by its confidence margin"
    assert stratifies in _source("tools", "estimation.py")
    random_only = 'picked_at_random = inner.get("plan_design") == "random"'
    assert random_only in _source("mcp_app", "server.py")
