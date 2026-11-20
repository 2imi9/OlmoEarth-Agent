# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-q (20 Nov 2026): ``olmoearth-agent-mcp install`` and ``uninstall``.

The owner: "let Claude read a section to auto update the config for other
users". The setup page made the user paste a JSON entry into Claude Desktop's
config by hand, because the file may hold a Studio key and an agent must not
open it. ``mcp_app/install.py`` edits the file itself and never shows it.

No test here touches the network, a real config, a real ``claude`` or a real
uvx: every config is a file under ``tmp_path``, ``shutil.which`` answers from
a table, and ``subprocess.run`` is a recorder. The keys and tokens are made
up for the tests.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.mcp_app import install

ROOT = Path(__file__).resolve().parents[2]
UVX = os.path.abspath("/opt/fake/bin/uvx")
CLAUDE = os.path.abspath("/opt/fake/bin/claude")
SPEC = "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent"
ARGS = ["--python", "3.12", "--from", SPEC, "olmoearth-agent-mcp"]
ENTRY = {"command": UVX, "args": ARGS}
#: A config that holds the app's entry alone.
ENTRY_ONLY = {"mcpServers": {"olmoearth-agent": ENTRY}}
#: The one line the README and the setup page give.
INSTALL_LINE = f'uvx --python 3.12 --from "{SPEC}" olmoearth-agent-mcp install'
UNINSTALL_LINE = f'uvx --python 3.12 --from "{SPEC}" olmoearth-agent-mcp uninstall'
ADD = [CLAUDE, "mcp", "add", "--scope", "user", "--transport", "stdio"]
ADD += ["olmoearth-agent", "--", UVX, *ARGS]
REMOVE = [CLAUDE, "mcp", "remove", "--scope", "user", "olmoearth-agent"]
#: Made up: an env value of the app's old entry, and another server's token.
FAKE_KEY = "sk_test_r5q_fake_key_42"
OTHER_VALUE = "tok_r5q_other_server_17"
NAME = "claude_desktop_config.json"
POSIX = os.name == "posix"

Answer = Callable[[list[str]], tuple[int, str]]


class Tools:
    """What the commands find on this computer, and what they ran.

    ``found`` is ``shutil.which``'s table; ``commands`` lists every command
    given to ``subprocess.run``; ``answer`` gives each one's exit code and
    output (0 and a usage line unless a test sets it).
    """

    def __init__(self) -> None:
        """uvx is found, ``claude`` is not, and every command works."""
        self.found: dict[str, str] = {"uvx": UVX}
        self.commands: list[list[str]] = []
        self.answer: Answer = lambda command: (0, "usage: olmoearth-agent-mcp\n")

    def which(self, name: str, *args: Any, **kwargs: Any) -> str | None:
        """``shutil.which``, from the table."""
        return self.found.get(name)

    def run(self, command: list[str], **kwargs: Any) -> Any:
        """``subprocess.run``: record the command, never start it."""
        assert kwargs.get("shell") in (None, False)
        self.commands.append(list(command))
        code, output = self.answer(list(command))
        return subprocess.CompletedProcess(command, code, output, None)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Tools:
    """No real uvx, ``claude`` or subprocess; the home folder is ``tmp_path``."""
    fake = Tools()
    monkeypatch.setattr(install.shutil, "which", fake.which)
    monkeypatch.setattr(install.subprocess, "run", fake.run)
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(install.Path, "home", classmethod(lambda cls: tmp_path))
    return fake


def _run(*argv: str) -> int:
    """Parse a command line as the program does and run its subcommand."""
    parser = argparse.ArgumentParser(prog="olmoearth-agent-mcp")
    install.add_subcommands(parser)
    return install.run(parser.parse_args(list(argv)))


def _config(tmp_path: Path, data: Any = None, raw: bytes | None = None) -> Path:
    """A config under ``tmp_path``: absent, ``raw`` bytes, or ``data`` as JSON."""
    path = tmp_path / "Claude" / NAME
    if data is not None or raw is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            raw if raw is not None else json.dumps(data, indent=4).encode()
        )
    return path


def _read(path: Path) -> Any:
    """A config's JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def _backups(path: Path) -> list[Path]:
    """The backups beside a config."""
    return sorted(path.parent.glob(path.name + ".bak-*"))


def _mode(path: Path) -> int:
    """A file's permission bits."""
    return path.stat().st_mode & 0o777


def _others() -> dict[str, Any]:
    """A config with two other servers and other top-level keys, in an order
    that sorting would change."""
    return {
        "globalShortcut": "Alt+Space",
        "mcpServers": {
            "zeta": {"command": "node", "args": ["server.js"]},
            "alpha": {"command": "npx", "args": ["-y", "x"], "env": {"T": OTHER_VALUE}},
        },
        "preferences": {"nested": [1, 2, {"a": None}], "ratio": 1.5, "on": True},
    }


# ------------------------------------------------------------------- install


def test_a_new_config_is_made_with_the_entry_and_mode_600(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No file and no folder: both are made; the file holds the entry alone,
    indent 2 with a final newline, mode 600, and no backup is made."""
    config = _config(tmp_path)
    assert not config.parent.exists()
    assert _run("install", "--config", str(config)) == 0
    expected = {"mcpServers": {"olmoearth-agent": ENTRY}}
    assert config.read_text(encoding="utf-8") == json.dumps(expected, indent=2) + "\n"
    if POSIX:
        assert _mode(config) == 0o600
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    out = capsys.readouterr().out
    assert f"Added olmoearth-agent to Claude Desktop's config: {config}" in out
    assert install.NEXT_STEP in out and "Cmd+Q" in out
    assert "Then ask Claude: Show what the OlmoEarth Agent can do." in out
    assert "run this again with --claude-code." in out
    assert "A backup" not in out


def test_an_empty_config_starts_from_an_empty_object(
    tools: Tools, tmp_path: Path
) -> None:
    """An empty file (as Edit Config can leave) takes the whole block."""
    config = _config(tmp_path, raw=b" \n")
    assert _run("install", "--config", str(config)) == 0
    assert _read(config) == {"mcpServers": {"olmoearth-agent": ENTRY}}
    assert [p.read_bytes() for p in _backups(config)] == [b" \n"]


def test_other_servers_and_top_level_keys_are_kept_in_order(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every other server and every other top-level key: the same values, in
    the same order; the entry is added last. No other server's value is
    printed."""
    old = _others()
    config = _config(tmp_path, old)
    assert _run("install", "--config", str(config)) == 0
    new = _read(config)
    assert list(new) == ["globalShortcut", "mcpServers", "preferences"]
    assert list(new["mcpServers"]) == ["zeta", "alpha", "olmoearth-agent"]
    assert new["mcpServers"].pop("olmoearth-agent") == ENTRY
    assert new == old
    assert list(new["preferences"]) == ["nested", "ratio", "on"]
    said = capsys.readouterr()
    assert OTHER_VALUE not in said.out + said.err
    assert "zeta" not in said.out + said.err


def test_a_config_without_servers_gets_them_after_its_own_keys(
    tools: Tools, tmp_path: Path
) -> None:
    """``mcpServers`` is made when missing, after the keys already there."""
    config = _config(tmp_path, {"theme": "dark"})
    assert _run("install", "--config", str(config)) == 0
    new = _read(config)
    assert list(new) == ["theme", "mcpServers"]
    assert new == {"theme": "dark", "mcpServers": {"olmoearth-agent": ENTRY}}


def test_an_old_entry_keeps_its_env_and_the_key_is_never_printed(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ``olmoearth-agent`` entry already there keeps its ``env`` and its
    other keys, in their order; only ``command`` and ``args`` are replaced.
    The env's value is in no line the command prints."""
    env = {"OLMOEARTH_API_KEY": FAKE_KEY, "OLMOEARTH_READ_ONLY": "1"}
    old = _others()
    old["mcpServers"] = {
        "olmoearth-agent": {
            "command": "/old/place/uvx",
            "env": dict(env),
            "args": ["--from", "an-old-package", "olmoearth-agent-mcp"],
            "disabled": False,
        },
        **old["mcpServers"],
    }
    config = _config(tmp_path, old)
    assert _run("install", "--config", str(config)) == 0
    new = _read(config)
    assert list(new["mcpServers"]) == ["olmoearth-agent", "zeta", "alpha"]
    entry = new["mcpServers"]["olmoearth-agent"]
    assert list(entry) == ["command", "env", "args", "disabled"]
    assert entry == {"command": UVX, "env": env, "args": ARGS, "disabled": False}
    said = capsys.readouterr()
    assert f"Updated olmoearth-agent in Claude Desktop's config: {config}" in said.out
    for secret in (FAKE_KEY, OTHER_VALUE, "OLMOEARTH_API_KEY", "an-old-package"):
        assert secret not in said.out + said.err, secret


def test_an_identical_entry_writes_nothing(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Already set up: no write, no backup, and it says so."""
    data = _others()
    data["mcpServers"]["olmoearth-agent"] = {**ENTRY, "env": {"K": FAKE_KEY}}
    config = _config(tmp_path, data)
    before = config.read_bytes()
    stamp = config.stat().st_mtime_ns

    def refuse(*args: Any, **kwargs: Any) -> None:
        """No file may be replaced in this test."""
        raise AssertionError("the config was written")

    monkeypatch.setattr(install.os, "replace", refuse)
    assert _run("install", "--config", str(config)) == 0
    assert config.read_bytes() == before and config.stat().st_mtime_ns == stamp
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    said = capsys.readouterr()
    assert f"Already set up in Claude Desktop's config: {config}" in said.out
    assert "A backup" not in said.out and FAKE_KEY not in said.out + said.err


@pytest.mark.parametrize(
    "raw",
    [
        b'{"mcpServers": {"r5q_marker": 1}',
        b'["r5q_marker"]',
        b'"r5q_marker"',
        b'{"mcpServers": "r5q_marker"}',
        b'{"mcpServers": ["r5q_marker"]}',
        b'{"mcpServers": null, "x": "r5q_marker"}',
        b"\xff\xfe r5q_marker",
        b'{"r5q_marker": 1e400}',
        b'{"r5q_marker": -1e400}',
        b'{"r5q_marker": NaN}',
        b'{"r5q_marker": Infinity}',
        b'{"r5q_marker": -Infinity}',
    ],
    ids=[
        "broken",
        "list",
        "string",
        "servers-string",
        "servers-list",
        "null",
        "bytes",
        "too-large",
        "too-large-negative",
        "nan",
        "infinity",
        "minus-infinity",
    ],
)
@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_a_config_that_is_not_a_json_object_is_left_as_it_is(
    tools: Tools,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    raw: bytes,
    command: str,
) -> None:
    """Not valid JSON, not an object, or ``mcpServers`` not an object: exit 1,
    the file unchanged byte for byte, no backup, no part file, and none of it
    printed. The same for ``NaN``, ``Infinity`` or a number too large to keep
    (``1e400``): Python reads them, but would write back what is not JSON."""
    config = _config(tmp_path, raw=raw)
    assert _run(command, "--config", str(config)) == 1
    assert config.read_bytes() == raw
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    said = capsys.readouterr()
    assert (
        f"Claude Desktop's config at {config} is not valid JSON; it was not changed. "
        "Fix it in Settings, Developer, Edit Config, or move it aside, then run "
        "this again."
    ) in said.err
    assert "r5q_marker" not in said.out + said.err
    assert "Added" not in said.out and "Removed" not in said.out


def test_the_backup_holds_the_old_bytes_and_the_write_is_atomic(
    tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backup (``.bak-YYYYMMDD-HHMMSS``, mode 600) is the file as it was.
    The new content is whole in a part file of the same folder before one
    ``os.replace`` puts it over the config, whose mode is kept."""
    config = _config(tmp_path, _others())
    old = config.read_bytes()
    if POSIX:
        config.chmod(0o640)
    replaced: list[tuple[Path, Path]] = []
    real = os.replace

    def spy(src: Any, dst: Any) -> None:
        """Check the part file and the config at the moment of the rename."""
        src, dst = Path(src), Path(dst)
        assert src.parent == config.parent and src != config
        assert _read(src)["mcpServers"]["olmoearth-agent"] == ENTRY
        assert src.read_bytes().endswith(b"}\n")
        assert config.read_bytes() == old
        assert [p.read_bytes() for p in _backups(config)] == [old]
        replaced.append((src, dst))
        real(src, dst)

    monkeypatch.setattr(install.os, "replace", spy)
    assert _run("install", "--config", str(config)) == 0
    assert [dst for _, dst in replaced] == [config]
    backups = _backups(config)
    assert len(backups) == 1 and backups[0].read_bytes() == old
    assert re.fullmatch(re.escape(NAME) + r"\.bak-\d{8}-\d{6}", backups[0].name)
    if POSIX:
        assert _mode(backups[0]) == 0o600 and _mode(config) == 0o640
    assert sorted(p.name for p in config.parent.iterdir()) == sorted(
        [NAME, backups[0].name]
    )
    assert _read(config)["mcpServers"]["olmoearth-agent"] == ENTRY


def test_a_write_that_fails_leaves_the_config_and_no_part_file(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The rename fails: exit 1, the config as it was, no part file left; the
    backup made before it stays."""
    config = _config(tmp_path, _others())
    old = config.read_bytes()

    def broken(src: Any, dst: Any) -> None:
        """The disk refuses the rename."""
        raise PermissionError("no")

    monkeypatch.setattr(install.os, "replace", broken)
    assert _run("install", "--config", str(config)) == 1
    assert config.read_bytes() == old
    names = sorted(p.name for p in config.parent.iterdir())
    assert names == sorted([NAME, _backups(config)[0].name])
    said = capsys.readouterr()
    assert "could not be written (PermissionError); it was not changed." in said.err
    assert f"A backup copy was made beside it: {_backups(config)[0]}" in said.err
    assert "Added" not in said.out and OTHER_VALUE not in said.out + said.err
    assert _run("uninstall", "--config", str(_config(tmp_path / "u", ENTRY_ONLY))) == 1
    said = capsys.readouterr()
    assert "could not be written (PermissionError); it was not changed." in said.err
    assert "Removed" not in said.out


def test_a_failed_write_of_a_read_only_config_leaves_no_part_file(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """As on Windows, where a read-only file can be neither replaced nor
    deleted: the part file (a full copy of the new config) is still cleaned
    up, and the config is as it was."""
    config = _config(tmp_path, _others())
    old = config.read_bytes()
    config.chmod(0o444)
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        """Refuse to delete a read-only file, as Windows does."""
        if self.exists() and not self.stat().st_mode & 0o200:
            raise PermissionError(str(self))
        real(self, missing_ok=missing_ok)

    def broken(src: Any, dst: Any) -> None:
        """A read-only config cannot be replaced."""
        raise PermissionError("no")

    monkeypatch.setattr(install.Path, "unlink", unlink)
    monkeypatch.setattr(install.os, "replace", broken)
    assert _run("install", "--no-download", "--config", str(config)) == 1
    assert config.read_bytes() == old
    names = sorted(p.name for p in config.parent.iterdir())
    assert names == sorted([NAME, _backups(config)[0].name])
    assert "it was not changed." in capsys.readouterr().err


def test_ordinary_numbers_are_kept(tools: Tools, tmp_path: Path) -> None:
    """A config with ordinary floats, a large one included, is read and
    written back with the same values."""
    config = _config(tmp_path, {"ratio": 1.5, "big": 1e308, "count": 10**30})
    assert _run("install", "--no-download", "--config", str(config)) == 0
    new = _read(config)
    assert (new["ratio"], new["big"], new["count"]) == (1.5, 1e308, 10**30)
    assert new["mcpServers"]["olmoearth-agent"] == ENTRY


def test_two_backups_in_one_second_do_not_overwrite(tmp_path: Path) -> None:
    """A second backup in the same second gets its own name."""
    config = _config(tmp_path, raw=b"{}")
    first = install.backup(config, b"one")
    second = install.backup(config, b"two")
    assert first != second
    assert (first.read_bytes(), second.read_bytes()) == (b"one", b"two")


# ----------------------------------------------------------- ref, uvx, dry run


def test_ref_pins_a_release(tools: Tools, tmp_path: Path) -> None:
    """``--ref v1.5.3``: the entry and the download name that release."""
    config = _config(tmp_path)
    assert _run("install", "--ref", "v1.5.3", "--config", str(config)) == 0
    pinned = ["--python", "3.12", "--from", SPEC + "@v1.5.3", "olmoearth-agent-mcp"]
    assert _read(config)["mcpServers"]["olmoearth-agent"] == {
        "command": UVX,
        "args": pinned,
    }
    assert tools.commands == [[UVX, *pinned, "--help"]]
    assert install.package_spec(None) == SPEC
    assert install.package_spec("release/1.5_x-2") == SPEC + "@release/1.5_x-2"


@pytest.mark.parametrize("ref", ["", "v1 5", "a;b", "$(id)", "v1@x", "x" * 101, "é"])
def test_a_bad_ref_exits_2(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str], ref: str
) -> None:
    """A ref with any other character, or over 100 characters: exit 2, and
    nothing is downloaded or written."""
    config = _config(tmp_path)
    assert _run("install", "--ref", ref, "--config", str(config)) == 2
    assert not config.exists() and tools.commands == []
    assert "--ref takes a release tag" in capsys.readouterr().err
    assert install.REF.fullmatch("x" * 100)


def test_no_uvx_exits_2_with_the_uv_link(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """uv is not installed: exit 2 and one plain line with uv's link."""
    tools.found.clear()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
    config = _config(tmp_path)
    assert _run("install", "--config", str(config)) == 2
    assert not config.exists() and tools.commands == []
    said = capsys.readouterr()
    assert said.err.strip() == (
        "uv is needed and was not found. Install it first: https://docs.astral.sh/uv/"
    )
    assert said.out == ""


def test_uvx_is_found_beside_the_uv_that_runs_the_program(
    tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not on the PATH: uvx beside the uv named in ``UV`` is used, by its
    absolute path."""
    tools.found.clear()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
    folder = tmp_path / "uv-bin"
    folder.mkdir()
    name = "uvx.exe" if sys.platform.startswith("win") else "uvx"
    (folder / name).write_bytes(b"")
    monkeypatch.setenv("UV", str(folder / "uv"))
    assert install.find_uvx() == str(folder / name)
    tools.found["uvx"] = UVX
    assert install.find_uvx() == os.path.abspath(UVX)


def test_dry_run_writes_nothing_and_runs_nothing(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run``: the path and the entry (command and args only) are
    shown; no file is written, nothing is downloaded, no command runs."""
    data = _others()
    data["mcpServers"]["olmoearth-agent"] = {"command": "x", "env": {"K": FAKE_KEY}}
    config = _config(tmp_path, data)
    before = config.read_bytes()
    tools.found["claude"] = CLAUDE
    code = _run("install", "--dry-run", "--claude-code", "--config", str(config))
    assert code == 0 and tools.commands == []
    assert config.read_bytes() == before
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    said = capsys.readouterr()
    assert f"Would edit Claude Desktop's config: {config}" in said.out
    assert json.dumps(ENTRY, indent=2) in said.out
    assert install.shown(ADD) in said.out
    assert "Nothing was written, downloaded or run." in said.out
    for secret in (FAKE_KEY, OTHER_VALUE):
        assert secret not in said.out + said.err, secret
    missing = _config(tmp_path / "none")
    assert _run("install", "--dry-run", "--config", str(missing)) == 0
    assert not missing.parent.exists() and tools.commands == []
    assert _run("uninstall", "--dry-run", "--config", str(config)) == 0
    assert config.read_bytes() == before and tools.commands == []


# ------------------------------------------------------------------ download


def test_the_download_runs_the_entrys_own_command_first(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The download is the host's own command with ``--help``, run before the
    config is touched, after one line that says it can take minutes."""
    config = _config(tmp_path)
    seen: list[bool] = []

    def answer(command: list[str]) -> tuple[int, str]:
        """Note whether the config exists while the download runs."""
        seen.append(config.exists())
        return 0, "usage: olmoearth-agent-mcp\n"

    tools.answer = answer
    assert _run("install", "--config", str(config)) == 0
    assert tools.commands == [
        [UVX, "--python", "3.12", "--from", SPEC, "olmoearth-agent-mcp", "--help"]
    ]
    assert seen == [False] and config.exists()
    out = capsys.readouterr().out
    line = "Downloading the app and its parts once; this can take a few minutes."
    assert out.index(line) < out.index("Added olmoearth-agent")


def test_a_failed_download_changes_nothing(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The download fails: exit 1, uv's last 20 lines, the config untouched."""
    config = _config(tmp_path, _others())
    before = config.read_bytes()
    tools.found["claude"] = CLAUDE
    tools.answer = lambda command: (1, "".join(f"line {n}\n" for n in range(1, 31)))
    assert _run("install", "--claude-code", "--config", str(config)) == 1
    assert config.read_bytes() == before
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    assert len(tools.commands) == 1 and tools.commands[0][-1] == "--help"
    said = capsys.readouterr()
    lines = said.err.splitlines()
    assert lines[0] == "The download failed; nothing was changed. uv's last lines:"
    assert lines[1:] == [f"line {n}" for n in range(11, 31)]
    assert "Added" not in said.out and "Updated" not in said.out
    absent = _config(tmp_path / "none")
    assert _run("install", "--config", str(absent)) == 1
    assert not absent.parent.exists()


def test_no_download_skips_it(tools: Tools, tmp_path: Path) -> None:
    """``--no-download``: no command runs, and the config is written."""
    config = _config(tmp_path)
    assert _run("install", "--no-download", "--config", str(config)) == 0
    assert tools.commands == []
    assert _read(config) == {"mcpServers": {"olmoearth-agent": ENTRY}}


def test_a_command_that_cannot_start_reads_as_a_failure(
    tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uvx vanished between the look and the run: exit 1, nothing written."""

    def gone(command: list[str], **kwargs: Any) -> None:
        """The program is not there."""
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(install.subprocess, "run", gone)
    config = _config(tmp_path)
    assert _run("install", "--config", str(config)) == 1
    assert not config.exists()


# --------------------------------------------------------------- Claude Code


def test_claude_code_gets_the_exact_add_command(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--claude-code``: after the config, one ``claude mcp add`` for this
    user with the entry's own command."""
    tools.found["claude"] = CLAUDE
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 0
    assert tools.commands == [ADD]
    assert tools.commands[0] == [
        CLAUDE,
        "mcp",
        "add",
        "--scope",
        "user",
        "--transport",
        "stdio",
        "olmoearth-agent",
        "--",
        UVX,
        "--python",
        "3.12",
        "--from",
        SPEC,
        "olmoearth-agent-mcp",
    ]
    out = capsys.readouterr().out
    assert "Added olmoearth-agent to Claude Code, for this user." in out
    assert "run this again with --claude-code" not in out
    assert _read(config)["mcpServers"]["olmoearth-agent"] == ENTRY


def test_a_name_that_exists_in_claude_code_is_removed_then_added(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Claude Code says the name exists: remove it, then add again, and say
    that the old entry's ``--env`` settings were not kept."""
    tools.found["claude"] = CLAUDE
    answers = iter(
        [
            (1, "MCP server olmoearth-agent already exists in user config\n"),
            (0, "Removed MCP server olmoearth-agent\n"),
            (0, "Added stdio MCP server olmoearth-agent\n"),
        ]
    )
    tools.answer = lambda command: next(answers)
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 0
    assert tools.commands == [ADD, REMOVE, ADD]
    said = capsys.readouterr()
    assert "Replaced the olmoearth-agent entry in Claude Code, for this" in said.out
    assert "Any --env settings of the old entry were not kept" in said.out
    assert "add them again (--env KEY=value before --transport)" in said.out
    assert "Added olmoearth-agent to Claude Code" not in said.out
    assert said.err == ""


def test_an_old_claude_code_entry_that_is_not_removed_is_left(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name exists and Claude Code does not remove it: no second add,
    exit 1, and it says the old entry was not changed."""
    tools.found["claude"] = CLAUDE
    answers = iter(
        [
            (1, "MCP server olmoearth-agent already exists in user config\n"),
            (1, "error: the config is locked\n"),
        ]
    )
    tools.answer = lambda command: next(answers)
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 1
    assert tools.commands == [ADD, REMOVE]
    said = capsys.readouterr()
    assert "did not remove it; it was not changed. Its last lines:" in said.err
    assert "error: the config is locked" in said.err
    assert "was removed from Claude Code" not in said.err
    assert "Claude Code, for this user" not in said.out
    last = "Claude Desktop's config is set up; only the Claude Code step failed."
    assert said.err.strip().splitlines()[-1] == last


def test_an_old_claude_code_entry_removed_and_not_added_again_is_said(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The old entry is removed and the second add fails: exit 1, it says the
    old entry is gone, and shows the command that adds it."""
    tools.found["claude"] = CLAUDE
    answers = iter(
        [
            (1, "MCP server olmoearth-agent already exists in user config\n"),
            (0, "Removed MCP server olmoearth-agent\n"),
            (1, "error: no room\n"),
        ]
    )
    tools.answer = lambda command: next(answers)
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 1
    assert tools.commands == [ADD, REMOVE, ADD]
    said = capsys.readouterr()
    assert (
        "The old olmoearth-agent entry was removed from Claude Code and the new "
        "one was not added. Its last lines:"
    ) in said.err
    assert "error: no room" in said.err
    assert "To add it, run:\n  " + install.shown(ADD) in said.err
    assert "Claude Code, for this user" not in said.out


def test_claude_code_that_refuses_exits_1_with_its_words(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Another failure of ``claude mcp add``: exit 1 with its last lines; no
    remove is tried. Claude Desktop's config was already written."""
    tools.found["claude"] = CLAUDE
    tools.answer = lambda command: (1, "error: not signed in\n")
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 1
    assert tools.commands == [ADD]
    said = capsys.readouterr()
    assert "Claude Code did not add olmoearth-agent." in said.err
    assert "error: not signed in" in said.err
    last = "Claude Desktop's config is set up; only the Claude Code step failed."
    assert said.err.strip().splitlines()[-1] == last
    assert _read(config)["mcpServers"]["olmoearth-agent"] == ENTRY
    tools.commands.clear()
    assert _run("uninstall", "--claude-code", "--config", str(config)) == 1
    assert tools.commands == [REMOVE] and "mcpServers" in _read(config)
    assert "olmoearth-agent" not in _read(config)["mcpServers"]
    said = capsys.readouterr()
    assert "Claude Code did not remove olmoearth-agent." in said.err
    assert (
        "olmoearth-agent is out of Claude Desktop's config; only the Claude Code "
        "step failed."
    ) in said.err
    assert _run("uninstall", "--claude-code", "--config", str(config)) == 1
    assert "is out of Claude Desktop's config" not in capsys.readouterr().err


def test_without_claude_the_command_is_printed(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``claude`` is not on the PATH: nothing runs, and the exact command to
    run is printed."""
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 0
    assert tools.commands == []
    out = capsys.readouterr().out
    assert "Claude Code (the claude command) was not found." in out
    assert install.shown(["claude", *ADD[1:]]) in out
    assert "claude mcp add --scope user --transport stdio olmoearth-agent -- " in out
    assert _run("uninstall", "--claude-code", "--config", str(config)) == 0
    assert tools.commands == []
    out = capsys.readouterr().out
    assert "claude mcp remove --scope user olmoearth-agent" in out


def test_on_windows_uvx_exe_is_found_and_the_shown_command_is_quoted_for_it(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With ``sys.platform`` set to Windows: uvx is looked for as ``uvx.exe``,
    and the command a user is told to paste has the package (it holds
    spaces) in double quotes, as that system's terminal takes it."""
    monkeypatch.setattr(sys, "platform", "win32")
    tools.found.clear()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python.exe"))
    folder = tmp_path / "uv-bin"
    folder.mkdir()
    (folder / "uvx.exe").write_bytes(b"")
    monkeypatch.setenv("UV", str(folder / "uv.exe"))
    assert install.find_uvx() == str(folder / "uvx.exe")
    quoted = '--from "' + SPEC + '" olmoearth-agent-mcp'
    line = install.shown(["claude", *ADD[1:]])
    assert quoted in line and "'" not in line
    tools.found["uvx"] = UVX
    config = _config(tmp_path)
    args = ("install", "--claude-code", "--no-download", "--config", str(config))
    assert _run(*args) == 0
    assert tools.commands == []
    assert quoted in capsys.readouterr().out


@pytest.mark.parametrize("dry", [(), ("--dry-run",)], ids=["run", "dry-run"])
@pytest.mark.parametrize("blank", ["", "  "], ids=["empty", "spaces"])
@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_an_empty_config_option_is_refused_and_the_real_config_is_untouched(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    blank: str,
    dry: tuple[str, ...],
) -> None:
    """``--config ""`` (an unset shell variable, say) must not fall through to
    Claude Desktop's own config: exit 2 with one line, nothing made, changed
    or backed up under the home folder, and no command run."""
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    tools.found["claude"] = CLAUDE
    default = install.config_paths()[0]
    assert tmp_path in default.parents
    argv = (command, *dry, "--claude-code", "--config", blank)
    assert _run(*argv) == 2
    assert not default.parent.exists() and tools.commands == []
    said = capsys.readouterr()
    assert said.err.strip() == "--config takes the path of a config file."
    assert said.out == ""
    default.parent.mkdir(parents=True)
    old = {"mcpServers": {"olmoearth-agent": {"command": "x", "env": {"K": FAKE_KEY}}}}
    default.write_text(json.dumps(old), encoding="utf-8")
    before = default.read_bytes()
    assert _run(*argv) == 2
    assert default.read_bytes() == before and tools.commands == []
    assert [p.name for p in default.parent.iterdir()] == [NAME]
    said = capsys.readouterr()
    assert str(default) not in said.out + said.err and said.out == ""


@pytest.mark.parametrize("blank", ["", "  "])
def test_an_empty_override_never_reads_as_the_default_path(
    tools: Tools, blank: str
) -> None:
    """``config_paths`` itself refuses an empty override."""
    with pytest.raises(ValueError, match="--config takes the path"):
        install.config_paths(blank)


# --------------------------------------------------------------------- paths


def test_the_config_path_of_each_system_is_read_at_call_time(
    tools: Tools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS, Linux and Windows, by ``sys.platform`` and the environment as
    they are when the function runs; ``--config`` overrides them all."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert install.config_paths() == [
        tmp_path / "Library" / "Application Support" / "Claude" / NAME
    ]
    monkeypatch.setattr(sys, "platform", "linux")
    assert install.config_paths() == [tmp_path / ".config" / "Claude" / NAME]
    monkeypatch.setattr(sys, "platform", "win32")
    roaming = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert install.config_paths() == [roaming / "Claude" / NAME]
    monkeypatch.delenv("APPDATA")
    monkeypatch.delenv("LOCALAPPDATA")
    assert install.config_paths() == [roaming / "Claude" / NAME]
    assert install.config_paths(str(tmp_path / "mine.json")) == [tmp_path / "mine.json"]


def test_windows_store_copies_are_found_and_each_is_written(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On Windows a Microsoft Store copy of the config (under a made-up
    LOCALAPPDATA) is found; both files get the entry, each with its own
    backup when it existed, and the copy is named."""
    monkeypatch.setattr(sys, "platform", "win32")
    roaming = tmp_path / "Roaming"
    local = tmp_path / "Local"
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    inner = Path("LocalCache") / "Roaming" / "Claude" / NAME
    store = local / "Packages" / "Claude_pzs8sxrjxfjjc" / inner
    decoy = local / "Packages" / "Other_abc" / inner
    for path in (store, decoy):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_others()), encoding="utf-8")
    old = store.read_bytes()
    standard = roaming / "Claude" / NAME
    assert install.config_paths() == [standard, store]
    assert _run("install", "--no-download") == 0
    assert _read(standard) == {"mcpServers": {"olmoearth-agent": ENTRY}}
    assert _backups(standard) == []
    new = _read(store)
    assert new["mcpServers"].pop("olmoearth-agent") == ENTRY and new == _others()
    assert [p.read_bytes() for p in _backups(store)] == [old]
    assert decoy.read_bytes() == old and _backups(decoy) == []
    said = capsys.readouterr()
    out = said.out
    assert f"Found a Microsoft Store copy of the config too: {store}" in out
    assert f"Added olmoearth-agent to Claude Desktop's config: {standard}" in out
    assert f"Added olmoearth-agent to Claude Desktop's config: {store}" in out
    assert OTHER_VALUE not in said.out + said.err


# ----------------------------------------------------------------- uninstall


def test_uninstall_removes_only_the_entry(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only ``olmoearth-agent`` goes, after a backup; the rest is as it was,
    in the same order. The saved key is not opened or deleted, nor its folder
    listed (every open and listing is watched while the command runs), and
    the last line says how the user removes it."""
    key_file = tmp_path / ".config" / "olmoearth-agent" / "studio.env"
    key_file.parent.mkdir(parents=True)
    key_file.write_text(f"OLMOEARTH_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    key_stat = key_file.stat()
    data = _others()
    data["mcpServers"] = {
        "zeta": data["mcpServers"]["zeta"],
        "olmoearth-agent": {**ENTRY, "env": {"OLMOEARTH_API_KEY": FAKE_KEY}},
        "alpha": data["mcpServers"]["alpha"],
    }
    config = _config(tmp_path, data)
    old = config.read_bytes()
    tools.found["claude"] = CLAUDE
    touched: list[str] = []
    key_dir = str(key_file.parent)

    def spy(real: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an opener or a folder lister, to note a path in the key's folder."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Note the path when it is in the key's folder; then do the real thing."""
            path = args[0] if args else None
            if isinstance(path, str | bytes | os.PathLike):
                where = os.fsdecode(path)
                if where.startswith(key_dir):
                    touched.append(where)
            return real(*args, **kwargs)

        return wrapper

    with monkeypatch.context() as watch:
        watched = [(io, "open"), (os, "open"), (os, "scandir"), (os, "listdir")]
        for owner, name in watched:
            watch.setattr(owner, name, spy(getattr(owner, name)))
        watch.setattr("builtins.open", spy(open))
        assert _run("uninstall", "--claude-code", "--config", str(config)) == 0
    assert touched == []
    new = _read(config)
    assert new == _others() and list(new["mcpServers"]) == ["zeta", "alpha"]
    assert list(new) == ["globalShortcut", "mcpServers", "preferences"]
    backups = _backups(config)
    assert [p.read_bytes() for p in backups] == [old]
    if POSIX:
        assert _mode(backups[0]) == 0o600
    assert tools.commands == [REMOVE]
    said = capsys.readouterr()
    assert f"Removed olmoearth-agent from Claude Desktop's config: {config}" in said.out
    assert f"A backup of the config as it was: {backups[0]}" in said.out
    assert "Removed olmoearth-agent from Claude Code, for this user." in said.out
    assert install.KEY_NOTE in said.out and "press Sign out" in install.KEY_NOTE
    assert "~/.config/olmoearth-agent/studio.env" in install.KEY_NOTE
    assert FAKE_KEY not in said.out + said.err
    assert key_file.stat().st_mtime_ns == key_stat.st_mtime_ns
    assert key_file.read_text(encoding="utf-8") == f"OLMOEARTH_API_KEY={FAKE_KEY}\n"
    # The command's own code never names the key file; only that line does.
    assert "studio.env" not in inspect.getsource(install.uninstall)
    assert "If you saved a Studio key in the app" in said.out.splitlines()[-1]


def test_uninstall_says_when_there_was_no_entry(
    tools: Tools, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No entry, no servers, or no file: exit 0, nothing written, said so."""
    config = _config(tmp_path, _others())
    before = config.read_bytes()
    assert _run("uninstall", "--config", str(config)) == 0
    assert config.read_bytes() == before
    assert [p.name for p in config.parent.iterdir()] == [NAME]
    out = capsys.readouterr().out
    assert f"No olmoearth-agent entry in Claude Desktop's config: {config}" in out
    assert "Removed" not in out and tools.commands == []
    bare = _config(tmp_path / "bare", {"theme": "dark"})
    assert _run("uninstall", "--config", str(bare)) == 0
    assert _read(bare) == {"theme": "dark"} and _backups(bare) == []
    absent = _config(tmp_path / "none")
    assert _run("uninstall", "--config", str(absent)) == 0
    assert not absent.parent.exists()
    assert "No olmoearth-agent entry" in capsys.readouterr().out


# ---------------------------------------------------------------------- main


def test_main_serves_with_no_subcommand_and_installs_with_one(
    tools: Tools,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main([])`` still starts the server on stdio and never the installer;
    ``main(["install", ...])`` runs the installer and never the server."""
    pytest.importorskip("mcp")
    anyio = pytest.importorskip("anyio")
    from olmoearth_agent.mcp_app import server

    started: list[Any] = []
    installs: list[argparse.Namespace] = []
    monkeypatch.setattr(server, "_defaults", lambda env: None)
    monkeypatch.setattr(server, "_enter_output_root", lambda env: None)
    monkeypatch.setattr(server, "AppServer", lambda: object())
    monkeypatch.setattr(server, "build_server", lambda app: object())
    monkeypatch.setattr(server, "HOST_LOG", logging.getLogger("r5q.test.host"))
    monkeypatch.setattr(anyio, "run", started.append)
    real = install.run

    def spy(args: argparse.Namespace) -> int:
        """Note each run of the installer."""
        installs.append(args)
        return real(args)

    monkeypatch.setattr(install, "run", spy)
    assert server.main([]) == 0
    assert len(started) == 1 and callable(started[0]) and installs == []
    # stdout carries the protocol: the serve path writes nothing to it.
    assert capsys.readouterr().out == ""
    config = _config(tmp_path)
    assert server.main(["install", "--dry-run", "--config", str(config)]) == 0
    assert len(started) == 1 and len(installs) == 1
    assert installs[0].command == "install" and installs[0].dry_run is True
    assert not config.exists() and tools.commands == []
    assert f"Would edit Claude Desktop's config: {config}" in capsys.readouterr().out
    assert server.main(["uninstall", "--dry-run", "--config", str(config)]) == 0
    assert len(started) == 1 and installs[1].command == "uninstall"
    # As a host starts it: no argument list, the program's own command line.
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["olmoearth-agent-mcp"])
    assert server.main() == 0
    assert len(started) == 2 and len(installs) == 2
    assert capsys.readouterr().out == ""


def test_help_names_install_and_uninstall_in_one_line_each(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--help`` starts nothing and lists the two subcommands, each on one
    line of an 80-column terminal; each subcommand has its own help."""
    pytest.importorskip("mcp")
    from olmoearth_agent.mcp_app import server

    monkeypatch.setenv("COLUMNS", "80")
    with pytest.raises(SystemExit) as stop:
        server.main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: olmoearth-agent-mcp")
    assert re.search(r"^ +install +add the app to Claude Desktop's config$", out, re.M)
    assert re.search(
        r"^ +uninstall +remove the app from Claude Desktop's config$", out, re.M
    )
    with pytest.raises(SystemExit) as stop:
        server.main(["install", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--claude-code", "--ref REF", "--config PATH", "--dry-run"):
        assert flag in out, flag
    assert "--no-download" in out
    with pytest.raises(SystemExit) as stop:
        server.main(["uninstall", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "--claude-code" in out and "--ref" not in out and "--no-download" not in out


# ---------------------------------------------------------------------- docs


def test_the_readme_and_the_setup_page_show_the_install_line() -> None:
    """Both give the exact one line; the setup page keeps the JSON by hand,
    has the row for a config that is not valid JSON and removes with
    ``uninstall``; the changelog and the developer reference say it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    setup = (ROOT / "docs" / "mcp-app-setup.md").read_text(encoding="utf-8")
    for page in (readme, setup):
        assert "```bash\n" + INSTALL_LINE + "\n```" in page
    assert "**Quick setup.** Needs [uv](https://docs.astral.sh/uv/) and git." in readme
    assert "**Quick setup with an agent.**" in readme
    assert "Quick setup by hand" not in readme and '"mcpServers"' not in readme
    assert "quit Claude Desktop fully and open it again" in readme
    assert "\n### 3. Run the install command\n" in setup
    assert "ask before running it" in setup
    assert "\n#### By hand, if the command cannot run\n" in setup
    assert '"command": "/absolute/path/to/uvx",' in setup
    assert "| `install` says the config is not valid JSON |" in setup
    assert "moves it aside" in setup
    assert UNINSTALL_LINE in setup.split("\n## Remove\n")[1]
    assert "except uv's own download from GitHub and PyPI" in setup
    for gone in ("step 5a", "step 5b", "### 7.", "### 8."):
        assert gone not in setup, gone
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = log.split("## [Unreleased]")[1].split("\n## [1.5.2]")[0]
    assert "### Added\n- Setup in one line: `olmoearth-agent-mcp install`" in unreleased
    usage = (ROOT / "spec" / "Usage.md").read_text(encoding="utf-8")
    assert "**The install command.** `olmoearth-agent-mcp install`" in usage
    assert "it never prints the config" in usage
