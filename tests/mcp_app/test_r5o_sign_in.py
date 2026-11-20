# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-o: when the host does not open the sign-in page, the panel says so once and offers a Copy button.

The page is a local http address (127.0.0.1), which a host may decline to open; the panel runs in a sandbox and
cannot open a browser window itself. The owner, 19 Nov 2026, on the address shown as plain text: "why the open the
sign-in page is not auto jump website but given a non linked website".
"""

from __future__ import annotations

from olmoearth_agent.mcp_app import views


def _common() -> str:
    """The panels' shared script."""
    return (views.VIEW_DIR / "common.js").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    """The text from ``start`` to the next ``end`` after it."""
    at = text.index(start)
    return text[at : text.index(end, at)]


def test_the_line_says_claude_did_not_open_the_page() -> None:
    """The words over the address name what happened and what to do, in one line."""
    common = _common()
    assert (
        "OE.SIGN_IN_NOT_OPENED = 'Claude did not open the page. "
        "Copy this address into a browser on this computer:';"
    ) in common
    assert "'Open this address in a browser on this computer to sign in:'" not in common


def test_the_sign_in_press_opens_quietly_and_adds_a_copy_button_when_declined() -> None:
    """The press opens the page with no toast (the line says it once) and, when declined, puts a Copy button
    beside the address; when the page opened, no button."""
    press = _between(
        _common(),
        "const page = await OE.call(app, 'olmoearth_app_begin_sign_in');",
        "stop();",
    )
    assert "await OE.open(app, page.url, { quiet: true });" in press
    assert ": OE.SIGN_IN_NOT_OPENED;" in press
    assert (
        "link.replaceChildren(address, opened ? null : OE.copyButton(page.url, address));"
        in press
    )


def test_the_copy_button_copies_or_selects_and_says_which() -> None:
    """The button writes the clipboard where it may, else selects the address and asks for the keyboard's copy."""
    copy = _between(
        _common(),
        "OE.copyButton = (text, shown, label = 'Copy address') => {",
        "\n};",
    )
    assert "OE.el('button', { type: 'button', class: 'link' }, label);" in copy
    assert "navigator.clipboard.writeText(text)" in copy
    assert "range.selectNodeContents(shown);" in copy
    assert "button.textContent = copied ? OE.COPIED : OE.COPY_BY_HAND;" in copy
    common = _common()
    assert "OE.COPIED = 'Copied';" in common
    assert (
        "OE.COPY_BY_HAND = 'Selected: press Cmd+C (Ctrl+C on Windows) to copy';"
        in common
    )
