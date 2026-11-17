# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Signing in to OlmoEarth Studio for the MCP App.

The app's tools call Studio with the user's own API key. The key comes from,
in this order:

1. a sign-in in this process, through the sign-in page (:class:`BrowserSignIn`);
2. ``OLMOEARTH_API_KEY`` in the server's environment (for example the ``env``
   block of ``claude_desktop_config.json``);
3. the key file, ``~/.config/olmoearth-agent/studio.env`` (or
   ``OLMOEARTH_KEY_FILE``), whose ``OLMOEARTH_API_KEY=`` line a sign-in with
   "remember" writes, mode 600.

The key is typed into a page this server serves on ``127.0.0.1``, opened in
the user's own browser, and posted straight to this server. It never travels
in an MCP message, so it is in no tool call the Claude app relays or logs.
A key typed there is checked with Studio (``/users/me`` and one page of
projects) before it is used or saved; a key Studio does not accept is
neither. A key from the environment or the key file is used as given; a
refusal by Studio shows the sign-in again. The key never appears in a tool
result, an error text or a log line: results carry the account's name and
organisation only.

Without a key, the tools that call Studio get a :class:`SignedOutStudio`,
whose every request raises :class:`NotSignedIn` before anything is sent, so
the tool's error says to sign in rather than reporting an HTTP 401.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import logging
import os
import re
import secrets
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx

from olmoearth_agent.studio.client import (
    DEFAULT_BASE_URL,
    StudioAccessError,
    StudioClient,
    StudioConfig,
)

logger = logging.getLogger(__name__)

#: Environment variable naming the key file (default :data:`DEFAULT_KEY_FILE`).
KEY_FILE_ENV = "OLMOEARTH_KEY_FILE"

#: Where a remembered key is saved, as ``OLMOEARTH_API_KEY=<key>``.
DEFAULT_KEY_FILE = Path("~/.config/olmoearth-agent/studio.env")

#: The environment variable the agent reads its Studio key from.
KEY_ENV = "OLMOEARTH_API_KEY"

#: The longest key accepted; Studio's keys are far shorter.
MAX_KEY_CHARS = 4096

#: A key is one token of printable characters: no space, no control character.
_KEY_RE = re.compile(r"^[\x21-\x7e]+$")

#: The placeholder ``.env.example`` ships, never a key.
_PLACEHOLDER = "replace-me"

#: The first words of :data:`NOT_SIGNED_IN`, by which a tool's own text that
#: quotes it (a tool that catches the error) is still recognised.
NOT_SIGNED_IN_MARK = "Not signed in to OlmoEarth Studio"

#: The error a Studio tool returns while nobody is signed in.
NOT_SIGNED_IN = (
    f"{NOT_SIGNED_IN_MARK}. Call olmoearth_sign_in: it opens the sign-in, a "
    "page in the user's browser on this computer where they enter their Studio "
    "API key. Never ask the user to paste a key into the chat. Then call this "
    "tool again."
)

#: How long a sign-in page stays open (seconds).
SIGN_IN_SECONDS = 600

#: The largest request the sign-in page reads (headers, then a form body).
_MAX_HEAD_BYTES = 16 * 1024
_MAX_BODY_BYTES = 8 * 1024


class NotSignedIn(RuntimeError):
    """A Studio request was made with no key signed in; nothing was sent."""

    def __init__(self) -> None:
        super().__init__(NOT_SIGNED_IN)


class _SignedOutTransport(httpx.AsyncBaseTransport):
    """An httpx transport that refuses every request: there is no key."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Refuse the request before anything leaves the machine."""
        raise NotSignedIn()


class SignedOutStudio(StudioClient):
    """A Studio client for a signed-out session: every request raises :class:`NotSignedIn`.

    The tools take a :class:`StudioClient`; this one has the same methods,
    and each of them fails at the transport, before a request is sent, with
    the text that says to sign in (:data:`NOT_SIGNED_IN`).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        super().__init__(StudioConfig(api_key="signed-out", base_url=base_url))
        self._client = httpx.AsyncClient(
            base_url=base_url, transport=_SignedOutTransport()
        )


def clean_key(raw: Any) -> str:
    """The key as typed, stripped; ``ValueError`` when it cannot be a key.

    Examples
    --------
    >>> clean_key("  abc123  ")
    'abc123'
    >>> clean_key("two words")
    Traceback (most recent call last):
    ...
    ValueError: A Studio API key is one line of letters, digits and symbols, with no spaces.
    """
    key = str(raw or "").strip()
    if not key or key == _PLACEHOLDER:
        raise ValueError("Enter your Studio API key (Studio: profile, API Keys).")
    if len(key) > MAX_KEY_CHARS or not _KEY_RE.match(key):
        raise ValueError(
            "A Studio API key is one line of letters, digits and symbols, with no spaces."
        )
    return key


def key_file(environ: Mapping[str, str] | None = None) -> Path:
    """The key file in effect: ``OLMOEARTH_KEY_FILE``, else :data:`DEFAULT_KEY_FILE`."""
    env = os.environ if environ is None else environ
    raw = env.get(KEY_FILE_ENV, "").strip()
    return Path(raw or DEFAULT_KEY_FILE).expanduser()


def _is_key_line(line: str) -> bool:
    """Whether ``line`` assigns :data:`KEY_ENV` (``OLMOEARTH_API_KEY=...``)."""
    name, sep, _value = line.strip().partition("=")
    name = name.strip()
    if name.startswith("export "):
        name = name[len("export ") :].strip()
    return bool(sep) and name == KEY_ENV


def read_key_file(path: Path) -> str | None:
    """The key a key file holds (its ``OLMOEARTH_API_KEY=`` line), or ``None``."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        if _is_key_line(line):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            try:
                return clean_key(value)
            except ValueError:
                return None
    return None


def _replace_file(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, mode 600, through a file beside it moved over it."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=".studio-key-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _other_lines(path: Path) -> list[str]:
    """The key file's lines other than the key's (a shared dotenv keeps them)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [line for line in text.splitlines() if not _is_key_line(line)]


def write_key_file(path: Path, key: str) -> None:
    """Save ``key`` as the file's ``OLMOEARTH_API_KEY=`` line, mode 600, atomically.

    Every other line of the file is kept as it was. The folder is made mode
    700 when it does not exist. The new file is written beside the target,
    made 600 before the key goes in, then moved over it, so no reader finds it
    half written or readable by others.
    """
    lines = [*_other_lines(path), f"{KEY_ENV}={key}"]
    _replace_file(path, "\n".join(lines) + "\n")


def remove_key_from_file(path: Path) -> bool:
    """Remove the file's ``OLMOEARTH_API_KEY=`` line; ``True`` when it had one.

    The file is deleted only when nothing else is left in it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    had = any(_is_key_line(line) for line in text.splitlines())
    if not had:
        return False
    rest = [line for line in text.splitlines() if not _is_key_line(line)]
    if any(line.strip() for line in rest):
        _replace_file(path, "\n".join(rest) + "\n")
    else:
        path.unlink(missing_ok=True)
    return True


@dataclass
class Account:
    """Who a key signs in as: what the views show, never the key."""

    user_name: str | None
    organization: str | None
    n_projects: int

    def as_dict(self) -> dict[str, Any]:
        """The account as JSON for a view."""
        return {
            "user_name": self.user_name,
            "organization": self.organization,
            "n_projects": self.n_projects,
        }


async def check_key(key: str, base_url: str) -> Account:
    """Ask Studio who ``key`` signs in as; ``ValueError`` when Studio refuses it.

    Reads ``/users/me`` and one page of projects; writes nothing. A transport
    failure (no network) is raised as it is, so the page can say Studio could
    not be reached rather than that the key is wrong.
    """
    async with StudioClient(StudioConfig(api_key=key, base_url=base_url)) as studio:
        try:
            context = await studio.load_context(limit=50)
        except StudioAccessError as exc:
            raise ValueError(
                "Studio did not accept this key (it may be mistyped, expired "
                "or revoked); nothing was saved."
            ) from exc
    return Account(
        user_name=context.user_name,
        organization=context.organization,
        n_projects=len(context.projects),
    )


class KeyStore:
    """The Studio key in effect for this server, and where it came from.

    ``signed_in`` (this process) comes first, then the environment, then the
    key file (module docstring). :meth:`sign_out` forgets the process's key
    and removes the key file's key line; a key in the environment stays, and
    :meth:`status` says so.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ
        self._key: str | None = None
        self.account: Account | None = None

    @property
    def path(self) -> Path:
        """The key file this store reads and writes."""
        return key_file(self._environ)

    def current(self) -> tuple[str, str] | None:
        """``(key, source)`` for the key in effect, or ``None`` when there is none."""
        if self._key:
            return self._key, "signed-in"
        raw = self._environ.get(KEY_ENV, "")
        if raw.strip():
            try:
                return clean_key(raw), "environment"
            except ValueError:
                pass
        saved = read_key_file(self.path)
        if saved:
            return saved, "key-file"
        return None

    def sign_in(self, key: str, account: Account, *, remember: bool) -> str | None:
        """Use ``key`` (already checked) from now on; save it when ``remember``.

        Returns why the key could not be remembered (the key file could not
        be read or written), or ``None``; the key is in effect either way.
        """
        problem = None
        if remember:
            try:
                write_key_file(self.path, key)
            except (OSError, UnicodeError) as exc:
                problem = (
                    f"The key could not be saved to {self.path} "
                    f"({type(exc).__name__}); it is used for this session only."
                )
        self._key = key
        self.account = account
        return problem

    def sign_out(self) -> dict[str, Any]:
        """Forget the process's key and the key file's key line; say what remains."""
        self._key = None
        self.account = None
        removed = remove_key_from_file(self.path)
        still = bool(self._environ.get(KEY_ENV, "").strip())
        return {"key_file_removed": removed, "environment_key_remains": still}

    def status(self) -> dict[str, Any]:
        """Whether a key is in effect, its source and account; never the key."""
        held = self.current()
        return {
            "signed_in": held is not None,
            "source": held[1] if held else None,
            "account": self.account.as_dict() if self.account else None,
            "key_file": str(self.path),
        }


#: The sign-in page, in OlmoEarth Studio's look (``views/common.js``): the
#: dark-teal top bar with the wordmark and the AGENT tag, the cream canvas, and
#: a Studio dialog without its backdrop (cream, 1px border, 4px corners, a
#: 600 20px title, one 56px outlined field, the contained dark-teal button at
#: the right). Nothing is loaded from elsewhere (system fonts).
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to OlmoEarth Studio</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; min-height: 100vh; background: #faf2e9; color: #0a3235;
  font: 14px/21px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
.bar {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: #032629;
  box-shadow: 0 2px 4px -1px rgba(0,0,0,.2), 0 4px 5px 0 rgba(0,0,0,.14), 0 1px 10px 0 rgba(0,0,0,.12); }}
.bar img {{ height: 18px; display: block; }}
main {{ width: min(480px, calc(100vw - 32px)); margin: 56px auto; background: #faf2e9;
  border: 1px solid rgba(0,0,0,.12); border-radius: 4px; padding: 0 24px 20px; }}
h1 {{ font-size: 20px; line-height: 28px; font-weight: 600; margin: 0 -24px 4px; padding: 16px 24px; }}
p {{ margin: 0 0 12px; color: #105257; }}
input[type=password] {{ width: 100%; box-sizing: border-box; height: 56px; font: inherit; font-size: 16px;
  padding: 16.5px 14px; border-radius: 4px; border: 1px solid rgba(0,0,0,.23); background: #fff;
  color: #0a3235; }}
input[type=password]:hover {{ border-color: #0a3235; }}
input[type=password]:focus {{ outline: none; border-color: #0a3235; box-shadow: inset 0 0 0 1px #0a3235; }}
label {{ display: flex; gap: 8px; align-items: center; margin: 12px 0 16px; color: #105257; }}
input[type=checkbox] {{ accent-color: #105257; width: 16px; height: 16px; }}
.actions {{ display: flex; justify-content: flex-end; }}
button {{ font: inherit; font-size: 14px; line-height: 24.5px; font-weight: 500; padding: 6px 16px;
  border-radius: 4px; border: 0; background: #0a3235; color: #faf2e9; cursor: pointer;
  box-shadow: 0 3px 1px -2px rgba(0,0,0,.2), 0 2px 2px 0 rgba(0,0,0,.14), 0 1px 5px 0 rgba(0,0,0,.12); }}
button:hover {{ background: rgb(7,35,37); }}
button:focus-visible {{ outline: 2px solid #0a3235; outline-offset: 2px; }}
.error {{ color: #d02f2e; }} .ok {{ color: #0a3235; }}
</style></head>
<body>
<div class="bar">{brand}</div>
<main>
{body}
</main></body></html>
"""

_FORM = """<h1>Sign in to OlmoEarth Studio</h1>
<p>Your Studio API key (Studio: profile, API Keys). This page is served by the
OlmoEarth Agent on this computer; the key goes from this page to it, not
through Claude, and is checked with Studio before it is used.</p>
<p>A read-only key is enough to check maps; to save an area or run a model,
use a key that can write.</p>
{error}
<form method="post" autocomplete="off">
<input type="password" name="api_key" placeholder="Studio API key" aria-label="Studio API key"
  aria-describedby="key-error" required autofocus spellcheck="false">
<label><input type="checkbox" name="remember" value="1" checked> Remember on this computer
(a file only you can read)</label>
<div class="actions"><button type="submit">Sign in</button></div>
</form>"""

_DONE = """<h1 class="ok">Signed in</h1>
<p>{who} You can close this tab and go back to Claude.</p>
{note}"""

_GONE = """<h1>This sign-in link has expired</h1>
<p>Open the sign-in again from Claude.</p>"""


class BrowserSignIn:
    """A one-time sign-in page on ``127.0.0.1``, opened in the user's browser.

    :meth:`start` listens on a free port of the loopback address and returns
    the page's URL, whose path holds a random token. The page takes the key
    and "remember", checks the key with Studio (:func:`check_key`) and signs
    the :class:`KeyStore` in; the key is never in an MCP message. A request
    is answered only when its ``Host`` is the page's own (a DNS-rebound name
    is refused) and its path holds the token. A form post's ``Origin`` must
    be the page's own or ``null`` (what a browser sends for a page whose
    referrer policy withholds it, or from an opaque origin); a site elsewhere
    cannot know the token in any case. The page closes after a sign-in, or
    after :data:`SIGN_IN_SECONDS`. :attr:`sign_ins` counts the sign-ins made
    on it, so a panel can tell a new one from a key already in effect.
    """

    def __init__(
        self,
        keys: KeyStore,
        base_url: str,
        *,
        on_signed_in: Callable[[], Awaitable[None]] | None = None,
        brand_html: str = "",
    ) -> None:
        self.keys = keys
        self.base_url = base_url
        self.on_signed_in = on_signed_in
        self.brand_html = brand_html
        self._server: asyncio.Server | None = None
        self._token = ""
        self._port = 0
        self._deadline = 0.0
        self._closer: asyncio.TimerHandle | None = None
        #: Sign-ins made on this server's pages since it started.
        self.sign_ins = 0

    @property
    def url(self) -> str | None:
        """The open page's URL, or ``None`` when no page is open."""
        if self._server is None or time.monotonic() >= self._deadline:
            return None
        return f"http://127.0.0.1:{self._port}/sign-in/{self._token}"

    def seconds_left(self) -> int:
        """Seconds until the open page closes (0 when none is open)."""
        return max(0, int(self._deadline - time.monotonic())) if self.url else 0

    async def start(self) -> str:
        """Open the page (or keep the open one) and return its URL."""
        existing = self.url
        if existing:
            return existing
        await self.close()
        self._token = secrets.token_urlsafe(32)
        self._server = await asyncio.start_server(
            self._handle, host="127.0.0.1", port=0, limit=_MAX_HEAD_BYTES
        )
        self._port = int(self._server.sockets[0].getsockname()[1])
        self._deadline = time.monotonic() + SIGN_IN_SECONDS
        loop = asyncio.get_running_loop()
        self._closer = loop.call_later(
            SIGN_IN_SECONDS, lambda: loop.create_task(self.close())
        )
        return f"http://127.0.0.1:{self._port}/sign-in/{self._token}"

    async def close(self) -> None:
        """Stop listening; the URL stops working."""
        if self._closer is not None:
            self._closer.cancel()
            self._closer = None
        server, self._server = self._server, None
        self._deadline = 0.0
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()

    def _page(self, body: str) -> bytes:
        return _PAGE.format(brand=self.brand_html, body=body).encode("utf-8")

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Answer one request (one per connection)."""
        try:
            status, body = await asyncio.wait_for(self._answer(reader), timeout=60)
        except (asyncio.TimeoutError, asyncio.LimitOverrunError, ValueError):
            status, body = 400, self._page("<h1>Bad request</h1>")
        except Exception:  # noqa: BLE001 - never echo internals, and never the key
            logger.warning("sign-in page request failed", exc_info=False)
            status, body = 500, self._page("<h1>Something went wrong</h1>")
        reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found"}
        head = (
            f"HTTP/1.1 {status} {reason.get(status, 'Error')}\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            # same-origin: the page's own form post carries its Origin (under
            # no-referrer a browser sends "Origin: null"), and no Referer with
            # the token in it leaves the page's origin.
            "Referrer-Policy: same-origin\r\n"
            "X-Content-Type-Options: nosniff\r\n"
            "X-Frame-Options: DENY\r\n"
            "Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; "
            "img-src data:; form-action 'self'; frame-ancestors 'none'; base-uri 'none'\r\n"
            "Connection: close\r\n\r\n"
        )
        with contextlib.suppress(Exception):
            writer.write(head.encode("ascii") + body)
            await writer.drain()
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()

    async def _answer(self, reader: asyncio.StreamReader) -> tuple[int, bytes]:
        """The status and page for one request."""
        raw = await reader.readuntil(b"\r\n\r\n")
        lines = raw.decode("latin-1").split("\r\n")
        method, path, _version = (lines[0].split(" ") + ["", "", ""])[:3]
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, sep, value = line.partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        own = f"127.0.0.1:{self._port}"
        if headers.get("host") != own:
            return 403, self._page("<h1>Forbidden</h1>")
        want = f"/sign-in/{self._token}"
        if not self.url or not hmac.compare_digest(path.split("?")[0], want):
            return 404, self._page(_GONE)
        if method == "GET":
            return 200, self._page(_FORM.format(error=""))
        if method != "POST":
            return 400, self._page("<h1>Bad request</h1>")
        origin = headers.get("origin")
        if origin not in (None, "null", f"http://{own}"):
            return 403, self._page("<h1>Forbidden</h1>")
        length = int(headers.get("content-length", "0") or 0)
        if length < 0 or length > _MAX_BODY_BYTES:
            return 400, self._page("<h1>Bad request</h1>")
        form = parse_qs((await reader.readexactly(length)).decode("utf-8"))
        remember = bool(form.get("remember"))
        try:
            key = clean_key((form.get("api_key") or [""])[0])
            account = await check_key(key, self.base_url)
        except ValueError as exc:
            # An alert tied to the key field (aria-describedby="key-error").
            error = (
                '<p class="error" id="key-error" role="alert">'
                f"{html.escape(str(exc))}</p>"
            )
            return 200, self._page(_FORM.format(error=error))
        except Exception as exc:  # noqa: BLE001 - offline or Studio down, not the key
            error = (
                '<p class="error" id="key-error" role="alert">'
                "Studio could not be reached "
                f"({html.escape(type(exc).__name__)}); nothing was saved.</p>"
            )
            return 200, self._page(_FORM.format(error=error))
        problem = await asyncio.to_thread(
            self.keys.sign_in, key, account, remember=remember
        )
        self.sign_ins += 1
        if self.on_signed_in is not None:
            await self.on_signed_in()
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.get_running_loop().create_task(self.close())
        )
        who = account.user_name or "your account"
        org = f" ({account.organization})" if account.organization else ""
        text = html.escape(f"Signed in as {who}{org}.")
        note = f'<p class="error">{html.escape(problem)}</p>' if problem else ""
        return 200, self._page(_DONE.format(who=text, note=note))
