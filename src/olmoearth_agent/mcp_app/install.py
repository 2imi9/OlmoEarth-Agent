# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""``olmoearth-agent-mcp install`` and ``uninstall`` (round r5-q, 20 Nov 2026).

Until this round the setup page made the user paste a JSON entry into Claude
Desktop's config by hand, because that file may hold a Studio key or another
server's token and an agent must not open it. These two commands edit the file
themselves and never show it, so a person, or a coding agent with the user's
yes, sets the app up with one line.

What ``install`` does, in order (:func:`install`):

1. Finds uvx's absolute path (:func:`find_uvx`); Claude Desktop does not start
   a command from the shell's PATH.
2. Builds the entry (:func:`entry_for`): uvx, Python 3.12, this repository
   from GitHub (main, or the release or commit ``--ref`` names).
3. Downloads everything once (:func:`download`), by running the very command
   the host will start, with ``--help``: uv's cache then holds the package and
   the host's first start fits in its wait. A failed download changes nothing.
4. Finds Claude Desktop's config (:func:`config_paths`), reads it
   (:func:`load`), and merges the entry in (:func:`merge`): every other server
   and every other top-level key stays as it is, in the same order, and an
   ``olmoearth-agent`` entry already there keeps its ``env`` block and any
   other key; only its ``command`` and ``args`` are replaced.
5. Writes safely: a backup beside the file first (mode 600, only when the file
   existed), then the new content to a part file in the same folder and an
   atomic rename over the config (:func:`write_config`). The file's mode is
   kept (600 when new). Nothing is written when the entry is already the same.
6. With ``--claude-code``, adds the same command to Claude Code for this user
   (an entry already there is replaced, without its ``--env`` settings).

What these commands never do: print the config, any ``env`` value or any other
server's entry (a config that is not valid JSON is left as it is, unshown);
open the saved Studio key (``~/.config/olmoearth-agent/``); send anything
anywhere, except uv's own download from GitHub and PyPI.

Exit codes: 0 done, or nothing to do; 1 a failure (a config that could not be
written is as it was; a Claude Code step that fails comes after Claude
Desktop's config is set up, and says so); 2 a usage problem (no uv, a ``--ref``
that is not a tag, branch or commit name, an empty ``--config``). The system,
the home folder and the environment are read when a command runs, not at
import.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

#: The server's name in a host's config, and in Claude Code.
SERVER_NAME = "olmoearth-agent"
#: The command the entry starts, and the name of this program.
PROGRAM = "olmoearth-agent-mcp"
#: The package uvx installs: this repository, with the app's two extras.
PACKAGE = (
    "olmoearth-agent[mcp,inferencex] @ git+https://github.com/2imi9/OlmoEarth-Agent"
)
#: The Python the entry pins (olmoearth-inferencex does not install on 3.14).
PYTHON = "3.12"
#: Where uv comes from, said when uvx is not found.
UV_LINK = "https://docs.astral.sh/uv/"
#: Claude Desktop's config file, by name.
CONFIG_NAME = "claude_desktop_config.json"
#: A ``--ref``: a tag, a branch or a commit of the repository, by its letters.
REF = re.compile(r"[A-Za-z0-9._/-]{1,100}")
#: How many of uv's (or Claude Code's) last lines a failure shows.
TAIL_LINES = 20

DOWNLOADING = "Downloading the app and its parts once; this can take a few minutes."
NEXT_STEP = (
    "Quit Claude Desktop fully (on macOS, Cmd+Q; on Windows, quit it from the tray) "
    "and open it again. Then ask Claude: Show what the OlmoEarth Agent can do."
)
CLAUDE_CODE_OPTION = (
    "For Claude Code in a terminal too, run this again with --claude-code."
)
KEY_NOTE = (
    "If you saved a Studio key in the app, it is still on this computer. To remove "
    "it, delete ~/.config/olmoearth-agent/studio.env. (While the app is installed, "
    "you can instead press Sign out on the app's start card.)"
)
#: Said, with exit 2, when ``--config`` is given an empty value.
NO_CONFIG_PATH = "--config takes the path of a config file."


class ConfigError(Exception):
    """A config that cannot be read as a JSON object; its message is one plain
    line for the user and never holds any of the file's contents."""


class WriteError(Exception):
    """A config that could not be written; the config itself is as it was.

    ``error`` is the name of the system's error, and ``copy`` the backup made
    before the write was tried (None when none was made).
    """

    def __init__(self, error: str, copy: Path | None) -> None:
        """Keep the error's name and the backup's path."""
        super().__init__(error)
        self.error = error
        self.copy = copy


def add_subcommands(parser: argparse.ArgumentParser) -> None:
    """Add ``install`` and ``uninstall`` to the program's parser.

    Neither is required: with no subcommand the parsed ``command`` is None and
    the program serves on stdio, as before the round.
    """
    commands = parser.add_subparsers(dest="command", title="commands")
    add = commands.add_parser(
        "install",
        help="add the app to Claude Desktop's config",
        description=(
            "Add the OlmoEarth Agent to Claude Desktop's config, without showing "
            "the file. Every other entry is kept, and a backup is made first."
        ),
    )
    add.add_argument(
        "--claude-code",
        action="store_true",
        help="also add it to Claude Code in a terminal, for this user",
    )
    add.add_argument(
        "--ref",
        metavar="REF",
        help="pin a release tag or a commit, such as v1.5.0 (default: main)",
    )
    add.add_argument(
        "--config",
        metavar="PATH",
        help="the config file to edit, in place of Claude Desktop's own",
    )
    add.add_argument(
        "--dry-run",
        action="store_true",
        help="show the file and the entry; write, download and run nothing",
    )
    add.add_argument(
        "--no-download",
        action="store_true",
        help="do not download the app first",
    )
    remove = commands.add_parser(
        "uninstall",
        help="remove the app from Claude Desktop's config",
        description=(
            "Remove only the olmoearth-agent entry from Claude Desktop's config, "
            "without showing the file. A backup is made first."
        ),
    )
    remove.add_argument(
        "--claude-code",
        action="store_true",
        help="also remove it from Claude Code, for this user",
    )
    remove.add_argument(
        "--config",
        metavar="PATH",
        help="the config file to edit, in place of Claude Desktop's own",
    )
    remove.add_argument(
        "--dry-run",
        action="store_true",
        help="show the file it would edit; write and run nothing",
    )


def run(args: argparse.Namespace) -> int:
    """Run the parsed subcommand and return the program's exit code."""
    if args.command == "uninstall":
        return uninstall(args)
    return install(args)


def _fail(line: str) -> None:
    """Say one plain line of failure, on stderr."""
    print(line, file=sys.stderr)


def _windows() -> bool:
    """Whether this is Windows, read now (a test sets ``sys.platform``)."""
    return sys.platform.startswith("win")


def find_uvx() -> str | None:
    """uvx's absolute path, or None when uv is not installed.

    The PATH first; then beside the uv that runs this program (uv names itself
    in ``UV`` for what it starts), beside a uv on the PATH, and beside this
    interpreter (uv installed into the environment).
    """
    found = shutil.which("uvx")
    if found:
        return os.path.abspath(found)
    name = "uvx.exe" if _windows() else "uvx"
    for known in (os.environ.get("UV"), shutil.which("uv"), sys.executable):
        if not known:
            continue
        beside = Path(known).parent / name
        if beside.is_file():
            return os.path.abspath(beside)
    return None


def package_spec(ref: str | None) -> str:
    """The package uvx installs: main, or pinned to ``ref`` (a tag or commit)."""
    return PACKAGE + ("@" + ref if ref else "")


def uvx_args(ref: str | None) -> list[str]:
    """What follows uvx in the entry: the Python, the package, the program."""
    return ["--python", PYTHON, "--from", package_spec(ref), PROGRAM]


def entry_for(uvx: str, ref: str | None) -> dict[str, Any]:
    """The ``olmoearth-agent`` entry of a host's config: ``command``, ``args``."""
    return {"command": uvx, "args": uvx_args(ref)}


def claude_add_command(claude: str, uvx: str, ref: str | None) -> list[str]:
    """The command that adds the app to Claude Code, for this user."""
    head = [claude, "mcp", "add", "--scope", "user", "--transport", "stdio"]
    return [*head, SERVER_NAME, "--", uvx, *uvx_args(ref)]


def claude_remove_command(claude: str) -> list[str]:
    """The command that removes the app from Claude Code, for this user."""
    return [claude, "mcp", "remove", "--scope", "user", SERVER_NAME]


def shown(command: list[str]) -> str:
    """A command as one line the user can paste into this system's terminal."""
    if _windows():
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def config_paths(override: str | None = None) -> list[Path]:
    """The config file or files to edit.

    ``override`` (``--config``) alone when given; an empty one raises
    ValueError, so it never reads as Claude Desktop's own. Else that one:
    macOS ``~/Library/Application Support/Claude``, Windows ``%APPDATA%\\Claude``,
    Linux ``~/.config/Claude``. On Windows the Microsoft Store build may read
    its own copy, under ``%LOCALAPPDATA%\\Packages\\Claude_*``: every such copy
    that exists follows the first path.
    """
    if override is not None:
        if not override.strip():
            raise ValueError(NO_CONFIG_PATH)
        return [Path(override).expanduser()]
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library" / "Application Support" / "Claude" / CONFIG_NAME]
    if not _windows():
        return [home / ".config" / "Claude" / CONFIG_NAME]
    roaming = os.environ.get("APPDATA")
    base = Path(roaming) if roaming else home / "AppData" / "Roaming"
    paths = [base / "Claude" / CONFIG_NAME]
    local = os.environ.get("LOCALAPPDATA")
    packages = (Path(local) if local else home / "AppData" / "Local") / "Packages"
    if packages.is_dir():
        pattern = f"Claude_*/LocalCache/Roaming/Claude/{CONFIG_NAME}"
        paths += [copy for copy in sorted(packages.glob(pattern)) if copy not in paths]
    return paths


def _real(path: Path) -> Path:
    """The file a config path names: a link's target, so the link is kept."""
    return Path(os.path.realpath(path)) if path.is_symlink() else path


def _not_json(path: Path) -> str:
    """The line for a config that is not a JSON object; none of its contents."""
    return (
        f"Claude Desktop's config at {path} is not valid JSON; it was not changed. "
        "Fix it in Settings, Developer, Edit Config, or move it aside, then run "
        "this again."
    )


def _no_constant(token: str) -> Any:
    """Refuse ``NaN`` and ``Infinity``: Python reads them, JSON has neither."""
    raise ValueError(token)


def _finite_float(text: str) -> float:
    """A JSON number as a float; one too large to keep (``1e400``) is refused,
    since it would be written back as ``Infinity``, which is not JSON."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


def load(path: Path) -> tuple[dict[str, Any], bytes | None]:
    """Read a config: its JSON object, and its bytes (None when there is no file).

    An empty file reads as ``{}``. A file that is not valid JSON, not a JSON
    object, or whose ``mcpServers`` is not an object raises
    :class:`ConfigError`, as does a file that cannot be read, or one that
    holds ``NaN``, ``Infinity`` or a number too large to keep; the message
    never holds the file's contents.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        raise ConfigError(
            f"Claude Desktop's config at {path} could not be read "
            f"({type(exc).__name__}); it was not changed."
        ) from None
    try:
        text = raw.decode("utf-8-sig")
        data = (
            json.loads(text, parse_constant=_no_constant, parse_float=_finite_float)
            if text.strip()
            else {}
        )
    except (ValueError, RecursionError):
        raise ConfigError(_not_json(path)) from None
    if not isinstance(data, dict):
        raise ConfigError(_not_json(path))
    if "mcpServers" in data and not isinstance(data["mcpServers"], dict):
        raise ConfigError(_not_json(path))
    return data, raw


def merge(data: dict[str, Any], entry: dict[str, Any]) -> str:
    """Put the entry into a config's object, in place.

    Returns ``"added"``, ``"updated"`` or ``"same"``. ``mcpServers`` is made
    when missing. An entry already there keeps its ``env`` and every other
    key, in their order; only ``command`` and ``args`` are replaced.
    """
    servers = data.setdefault("mcpServers", {})
    if SERVER_NAME not in servers:
        servers[SERVER_NAME] = dict(entry)
        return "added"
    old = servers[SERVER_NAME]
    if not isinstance(old, dict):
        servers[SERVER_NAME] = dict(entry)
        return "updated"
    if old.get("command") == entry["command"] and old.get("args") == entry["args"]:
        return "same"
    old["command"] = entry["command"]
    old["args"] = list(entry["args"])
    return "updated"


def _open_new(path: Path) -> int:
    """Open a new file for writing bytes, mode 600; it must not exist yet."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o600)


def backup(path: Path, raw: bytes) -> Path:
    """Copy a config's bytes beside it before a change, mode 600.

    The name is the config's plus ``.bak-YYYYMMDD-HHMMSS`` (this computer's
    time); a second backup in the same second gets ``-2``, and so on.
    """
    name = f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    for count in range(1, 100):
        copy = path.with_name(name if count == 1 else f"{name}-{count}")
        try:
            fd = _open_new(copy)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.chmod(copy, 0o600)
        return copy
    raise FileExistsError(f"no free backup name beside {path}")


def write_config(path: Path, data: dict[str, Any], existed: bool) -> None:
    """Write a config: indent 2, a final newline, by an atomic rename.

    The content goes to a part file in the config's own folder, which then
    replaces the config (``os.replace``); a failure leaves the config as it
    was and no part file. The config's mode is kept; a new one gets 600.
    """
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
        body = (text + "\n").encode("utf-8")
    except UnicodeEncodeError:
        body = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("ascii")
    mode = stat.S_IMODE(path.stat().st_mode) if existed else 0o600
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with os.fdopen(_open_new(part), "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(part, mode)
        os.replace(part, path)
    except BaseException:
        # Windows: a read-only part file cannot be deleted; clear that first.
        with contextlib.suppress(OSError):
            os.chmod(part, 0o600)
        with contextlib.suppress(OSError):
            part.unlink(missing_ok=True)
        raise


def _call(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command with no shell and no input; its output comes back as text.

    A command that cannot start reads as a failure (code 127) with the
    error's name as its output.
    """
    try:
        return subprocess.run(  # noqa: S603 - a fixed argument list, no shell
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, type(exc).__name__, None)


def _tail(output: str | None) -> str:
    """The last lines of a command's output, for a failure message."""
    return "\n".join((output or "").strip().splitlines()[-TAIL_LINES:])


def download(uvx: str, ref: str | None) -> bool:
    """Fill uv's cache with what the host will start; True when it worked.

    Runs the entry's own command with ``--help`` (no server starts, no folder
    is made). On failure, says so with uv's last lines.
    """
    print(DOWNLOADING, flush=True)
    done = _call([uvx, *uvx_args(ref), "--help"])
    if done.returncode == 0:
        return True
    _fail("The download failed; nothing was changed. uv's last lines:")
    _fail(_tail(done.stdout))
    return False


def _load_all(paths: list[Path]) -> list[tuple[Path, dict[str, Any], bytes | None]]:
    """Read every config before any is changed: (the file, its object, its bytes)."""
    loaded = []
    for path in paths:
        real = _real(path)
        data, raw = load(real)
        loaded.append((real, data, raw))
    return loaded


def _say_copies(paths: list[Path]) -> None:
    """On Windows, name the Microsoft Store copies that are edited too."""
    for extra in paths[1:]:
        print(f"Found a Microsoft Store copy of the config too: {extra}")


def _change(path: Path, data: dict[str, Any], raw: bytes | None) -> Path | None:
    """Back a config up (when it existed) and write its new content.

    Returns the backup's path, or None for a new file. A failure raises
    :class:`WriteError` with the backup already made, if any; the config is
    then as it was.
    """
    copy: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            copy = backup(path, raw)
        write_config(path, data, existed=raw is not None)
    except OSError as exc:
        raise WriteError(type(exc).__name__, copy) from None
    return copy


def _say_not_written(path: Path, failure: WriteError) -> None:
    """Say that a config could not be written, that it is as it was, and where
    the backup made before the try is."""
    _fail(
        f"Claude Desktop's config at {path} could not be written "
        f"({failure.error}); it was not changed."
    )
    if failure.copy is not None:
        _fail(f"A backup copy was made beside it: {failure.copy}")


def _claude_code_add(uvx: str, ref: str | None, dry_run: bool) -> int:
    """Add the app to Claude Code for this user; 0 when done or only shown.

    When Claude Code says the name exists, the old one is removed and the new
    one added: the old entry's ``--env`` settings are not kept, and it says
    so (the old entry is never read or printed). A remove that fails stops
    there; an add that fails after the remove says the old entry is gone and
    shows the command. When ``claude`` is not found, the command is shown.
    """
    claude = shutil.which("claude")
    if claude is None or dry_run:
        command = claude_add_command(claude or "claude", uvx, ref)
        if dry_run:
            print("For Claude Code it would run:")
        else:
            print("Claude Code (the claude command) was not found. To add the app")
            print("to it, run this in a terminal where claude works:")
        print("  " + shown(command))
        return 0
    command = claude_add_command(claude, uvx, ref)
    done = _call(command)
    replaced = False
    if done.returncode != 0 and "already exists" in (done.stdout or "").lower():
        gone = _call(claude_remove_command(claude))
        if gone.returncode != 0:
            _fail(
                "Claude Code already has an olmoearth-agent entry and did not "
                "remove it; it was not changed. Its last lines:"
            )
            _fail(_tail(gone.stdout))
            return 1
        replaced = True
        done = _call(command)
    if done.returncode != 0 and replaced:
        _fail(
            "The old olmoearth-agent entry was removed from Claude Code and the "
            "new one was not added. Its last lines:"
        )
        _fail(_tail(done.stdout))
        _fail("To add it, run:")
        _fail("  " + shown(command))
        return 1
    if done.returncode != 0:
        _fail("Claude Code did not add olmoearth-agent. Its last lines:")
        _fail(_tail(done.stdout))
        return 1
    if replaced:
        print(
            "Replaced the olmoearth-agent entry in Claude Code, for this user. Any "
            "--env settings of the old entry were not kept; add them again (--env "
            "KEY=value before --transport). Start a new session; /mcp shows it."
        )
        return 0
    print(
        "Added olmoearth-agent to Claude Code, for this user. Start a new session; "
        "/mcp shows it."
    )
    return 0


def _claude_code_remove(dry_run: bool) -> int:
    """Remove the app from Claude Code for this user; 0 when done or only shown."""
    claude = shutil.which("claude")
    if claude is None or dry_run:
        if dry_run:
            print("For Claude Code it would run:")
        else:
            print("Claude Code (the claude command) was not found. To remove the app")
            print("from it, run this in a terminal where claude works:")
        print("  " + shown(claude_remove_command(claude or "claude")))
        return 0
    done = _call(claude_remove_command(claude))
    if done.returncode != 0:
        _fail("Claude Code did not remove olmoearth-agent. Its last lines:")
        _fail(_tail(done.stdout))
        return 1
    print("Removed olmoearth-agent from Claude Code, for this user.")
    return 0


def install(args: argparse.Namespace) -> int:
    """``olmoearth-agent-mcp install``: the steps of the module's docstring."""
    if args.config is not None and not args.config.strip():
        _fail(NO_CONFIG_PATH)
        return 2
    ref = args.ref
    if ref is not None and not REF.fullmatch(ref):
        _fail(
            "--ref takes a release tag, a branch or a commit of the repository: "
            "letters, digits and . - _ / only, at most 100 characters."
        )
        return 2
    uvx = find_uvx()
    if uvx is None:
        _fail(f"uv is needed and was not found. Install it first: {UV_LINK}")
        return 2
    entry = entry_for(uvx, ref)
    paths = config_paths(args.config)
    if args.dry_run:
        for path in paths:
            print(f"Would edit Claude Desktop's config: {path}")
        print("The olmoearth-agent entry it would write:")
        print(json.dumps(entry, indent=2))
        if args.claude_code:
            _claude_code_add(uvx, ref, dry_run=True)
        print("Nothing was written, downloaded or run.")
        return 0
    if not args.no_download and not download(uvx, ref):
        return 1
    try:
        loaded = _load_all(paths)
    except ConfigError as exc:
        _fail(str(exc))
        return 1
    _say_copies(paths)
    for path, data, raw in loaded:
        did = merge(data, entry)
        if did == "same":
            print(f"Already set up in Claude Desktop's config: {path}")
            continue
        try:
            copy = _change(path, data, raw)
        except WriteError as failure:
            _say_not_written(path, failure)
            return 1
        added = did == "added"
        word = "Added olmoearth-agent to" if added else "Updated olmoearth-agent in"
        print(f"{word} Claude Desktop's config: {path}")
        if copy is not None:
            print(f"A backup of the config as it was: {copy}")
    print(NEXT_STEP)
    if args.claude_code:
        code = _claude_code_add(uvx, ref, dry_run=False)
        if code != 0:
            _fail(
                "Claude Desktop's config is set up; only the Claude Code step failed."
            )
        return code
    print(CLAUDE_CODE_OPTION)
    return 0


def uninstall(args: argparse.Namespace) -> int:
    """``olmoearth-agent-mcp uninstall``: remove only the app's entry.

    A backup is made first and everything else is kept. It says when there
    was no entry. The saved Studio key is not opened and not deleted; the
    last line says how the user removes it.
    """
    if args.config is not None and not args.config.strip():
        _fail(NO_CONFIG_PATH)
        return 2
    paths = config_paths(args.config)
    if args.dry_run:
        for path in paths:
            print(f"Would remove olmoearth-agent from Claude Desktop's config: {path}")
        if args.claude_code:
            _claude_code_remove(dry_run=True)
        print("Nothing was written or run.")
        return 0
    try:
        loaded = _load_all(paths)
    except ConfigError as exc:
        _fail(str(exc))
        return 1
    _say_copies(paths)
    removed = False
    for path, data, raw in loaded:
        servers = data.get("mcpServers")
        if raw is None or not servers or SERVER_NAME not in servers:
            print(f"No olmoearth-agent entry in Claude Desktop's config: {path}")
            continue
        del servers[SERVER_NAME]
        try:
            copy = _change(path, data, raw)
        except WriteError as failure:
            _say_not_written(path, failure)
            return 1
        removed = True
        print(f"Removed olmoearth-agent from Claude Desktop's config: {path}")
        print(f"A backup of the config as it was: {copy}")
    if removed:
        print("Quit Claude Desktop fully and open it again.")
    code = _claude_code_remove(dry_run=False) if args.claude_code else 0
    if code != 0 and removed:
        _fail(
            "olmoearth-agent is out of Claude Desktop's config; only the Claude Code "
            "step failed."
        )
    print(KEY_NOTE)
    return code
