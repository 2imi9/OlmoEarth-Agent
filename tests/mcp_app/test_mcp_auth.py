# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The MCP App's Studio sign-in: key checks, the key file, the signed-out client."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent.mcp_app import auth

BASE = "http://mock-studio/api/v1"
KEY = "sk_test_mcp_key_123"


def _studio_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me",
        json={
            "records": [{"id": "u1", "name": "Ana", "organizations": [{"name": "Lab"}]}]
        },
    )
    httpx_mock.add_response(
        url=f"{BASE}/projects/search",
        method="POST",
        json={
            "records": [{"id": "p1", "name": "A"}, {"id": "p2", "name": "B"}],
            "meta": {"total": 2},
        },
    )


@pytest.mark.parametrize(
    "raw", ["", "   ", "replace-me", "two words", "tab\tkey", "x" * 5000, None]
)
def test_clean_key_refuses_what_cannot_be_a_key(raw: object) -> None:
    with pytest.raises(ValueError):
        auth.clean_key(raw)


def test_clean_key_strips() -> None:
    assert auth.clean_key(f"  {KEY}\n") == KEY


def test_key_file_is_written_600_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "studio.env"
    auth.write_key_file(path, KEY)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.read_text(encoding="utf-8") == f"OLMOEARTH_API_KEY={KEY}\n"
    assert auth.read_key_file(path) == KEY
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".studio-key-")]


def test_read_key_file_ignores_other_lines_and_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "studio.env"
    path.write_text("# comment\nOTHER=1\nOLMOEARTH_API_KEY='abc'\n", encoding="utf-8")
    assert auth.read_key_file(path) == "abc"
    path.write_text("OLMOEARTH_API_KEY=replace-me\n", encoding="utf-8")
    assert auth.read_key_file(path) is None
    assert auth.read_key_file(tmp_path / "missing.env") is None


def test_keystore_prefers_sign_in_then_environment_then_file(tmp_path: Path) -> None:
    path = tmp_path / "studio.env"
    auth.write_key_file(path, "from-file")
    env = {auth.KEY_FILE_ENV: str(path)}
    store = auth.KeyStore(env)
    assert store.current() == ("from-file", "key-file")
    env[auth.KEY_ENV] = "from-env"
    assert store.current() == ("from-env", "environment")
    store.sign_in("typed", auth.Account("Ana", "Lab", 2), remember=False)
    assert store.current() == ("typed", "signed-in")
    assert (
        auth.read_key_file(path) == "from-file"
    )  # not remembered: the file is unchanged


def test_sign_in_with_remember_saves_and_sign_out_removes(tmp_path: Path) -> None:
    path = tmp_path / "studio.env"
    store = auth.KeyStore({auth.KEY_FILE_ENV: str(path)})
    store.sign_in(KEY, auth.Account("Ana", "Lab", 2), remember=True)
    assert auth.read_key_file(path) == KEY
    out = store.sign_out()
    assert out == {"key_file_removed": True, "environment_key_remains": False}
    assert not path.exists()
    assert store.current() is None


def test_status_never_holds_the_key(tmp_path: Path) -> None:
    store = auth.KeyStore(
        {auth.KEY_FILE_ENV: str(tmp_path / "s.env"), auth.KEY_ENV: KEY}
    )
    status = store.status()
    assert status["signed_in"] is True and status["source"] == "environment"
    assert KEY not in repr(status)


@pytest.mark.asyncio
async def test_signed_out_client_refuses_before_sending(httpx_mock: HTTPXMock) -> None:
    studio = auth.SignedOutStudio(BASE)
    with pytest.raises(auth.NotSignedIn) as raised:
        await studio.users_me()
    assert "olmoearth_sign_in" in str(raised.value)
    assert not httpx_mock.get_requests()
    await studio.aclose()


@pytest.mark.asyncio
async def test_check_key_reads_the_account(httpx_mock: HTTPXMock) -> None:
    _studio_ok(httpx_mock)
    account = await auth.check_key(KEY, BASE)
    assert account.as_dict() == {
        "user_name": "Ana",
        "organization": "Lab",
        "n_projects": 2,
    }
    sent = httpx_mock.get_requests()
    assert all(r.headers["Authorization"] == f"Bearer {KEY}" for r in sent)


@pytest.mark.asyncio
async def test_check_key_refused_says_nothing_was_saved(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me", status_code=401, json={"errors": []}
    )
    with pytest.raises(ValueError, match="nothing was saved") as raised:
        await auth.check_key(KEY, BASE)
    assert KEY not in str(raised.value)


def test_the_key_file_keeps_its_other_lines(tmp_path: Path) -> None:
    """A shared dotenv: remember writes only the key line, sign-out removes only it."""
    path = tmp_path / "shared.env"
    path.write_text("OTHER=1\nOLMOEARTH_API_KEY=old\n# note\n", encoding="utf-8")
    auth.write_key_file(path, KEY)
    assert (
        path.read_text(encoding="utf-8")
        == f"OTHER=1\n# note\nOLMOEARTH_API_KEY={KEY}\n"
    )
    assert auth.remove_key_from_file(path) is True
    assert path.read_text(encoding="utf-8") == "OTHER=1\n# note\n"
    assert auth.remove_key_from_file(path) is False
    only = tmp_path / "only.env"
    auth.write_key_file(only, KEY)
    assert auth.remove_key_from_file(only) is True and not only.exists()


async def _http(
    port: int,
    method: str,
    path: str,
    *,
    host: str | None = None,
    origin: str | None = None,
    form: str = "",
) -> tuple[int, str]:
    """One request to the sign-in page over a raw socket (httpx is mocked here)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host or f'127.0.0.1:{port}'}"]
    if origin:
        lines.append(f"Origin: {origin}")
    body = form.encode()
    if method == "POST":
        lines += [
            "Content-Type: application/x-www-form-urlencoded",
            f"Content-Length: {len(body)}",
        ]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode() + body)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    head, _sep, page = raw.decode().partition("\r\n\r\n")
    return int(head.split(" ")[1]), page


@pytest.mark.asyncio
async def test_the_sign_in_page_takes_the_key_in_the_browser(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _studio_ok(httpx_mock)
    store = auth.KeyStore({auth.KEY_FILE_ENV: str(tmp_path / "s.env")})
    page = auth.BrowserSignIn(store, BASE)
    url = await page.start()
    assert url.startswith("http://127.0.0.1:") and "/sign-in/" in url
    assert await page.start() == url  # one page at a time
    port = int(url.split(":")[2].split("/")[0])
    path = url.split(str(port), 1)[1]
    status, html = await _http(port, "GET", path)
    assert status == 200 and 'name="api_key"' in html
    # Another name for the loopback address (DNS rebinding), a wrong token, a foreign origin.
    assert (await _http(port, "GET", path, host=f"evil.test:{port}"))[0] == 403
    assert (await _http(port, "GET", "/sign-in/not-the-token"))[0] == 404
    assert (
        await _http(
            port, "POST", path, origin="https://evil.test", form=f"api_key={KEY}"
        )
    )[0] == 403
    assert store.current() is None and not httpx_mock.get_requests()
    status, html = await _http(
        port,
        "POST",
        path,
        origin=f"http://127.0.0.1:{port}",
        form=f"api_key={KEY}&remember=1",
    )
    assert status == 200 and "Signed in as Ana (Lab)" in html and KEY not in html
    assert store.current() == (KEY, "signed-in")
    assert auth.read_key_file(tmp_path / "s.env") == KEY
    await asyncio.sleep(0.05)
    assert page.url is None  # one sign-in per page
    with pytest.raises(OSError):
        await _http(port, "GET", path)


@pytest.mark.asyncio
async def test_a_refused_key_on_the_page_is_neither_used_nor_saved(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    httpx_mock.add_response(
        url=f"{BASE}/users/me", status_code=401, json={"errors": []}
    )
    store = auth.KeyStore({auth.KEY_FILE_ENV: str(tmp_path / "s.env")})
    page = auth.BrowserSignIn(store, BASE)
    url = await page.start()
    port = int(url.split(":")[2].split("/")[0])
    status, html = await _http(
        port, "POST", url.split(str(port), 1)[1], form=f"api_key={KEY}&remember=1"
    )
    assert status == 200 and "did not accept this key" in html and KEY not in html
    assert store.current() is None and not (tmp_path / "s.env").exists()
    assert page.url == url  # the page stays open for another try
    await page.close()
    assert page.url is None


@pytest.mark.asyncio
async def test_a_browsers_form_post_is_taken_and_counted(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    """A browser posts the page's form with ``Origin: null`` when the referrer
    policy withholds the origin (review wf_e14d1b98-18f, P1); the page sends
    same-origin, and takes null or its own origin. Each sign-in is counted."""
    _studio_ok(httpx_mock)
    store = auth.KeyStore({auth.KEY_FILE_ENV: str(tmp_path / "s.env")})
    page = auth.BrowserSignIn(store, BASE)
    url = await page.start()
    port = int(url.split(":")[2].split("/")[0])
    path = url.split(str(port), 1)[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
    await writer.drain()
    head = (await reader.read()).decode().split("\r\n\r\n", 1)[0]
    writer.close()
    assert "Referrer-Policy: same-origin" in head
    assert page.sign_ins == 0
    status, html = await _http(
        port, "POST", path, origin="null", form=f"api_key={KEY}&remember=1"
    )
    assert status == 200 and "Signed in as Ana (Lab)" in html
    assert page.sign_ins == 1 and store.current() == (KEY, "signed-in")


@pytest.mark.asyncio
async def test_a_key_file_that_cannot_be_written_signs_in_for_the_session(
    httpx_mock: HTTPXMock, tmp_path: Path
) -> None:
    _studio_ok(httpx_mock)
    blocked = tmp_path / "s.env"
    blocked.mkdir()  # a folder where the key file should be: it cannot be written
    store = auth.KeyStore({auth.KEY_FILE_ENV: str(blocked)})
    page = auth.BrowserSignIn(store, BASE)
    url = await page.start()
    port = int(url.split(":")[2].split("/")[0])
    status, html = await _http(
        port, "POST", url.split(str(port), 1)[1], form=f"api_key={KEY}&remember=1"
    )
    assert status == 200 and "Signed in as Ana (Lab)" in html
    assert "this session only" in html and KEY not in html
    assert store.current() == (KEY, "signed-in")
