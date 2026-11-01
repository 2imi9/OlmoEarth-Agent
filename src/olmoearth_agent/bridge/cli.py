# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""``olmoearth-agent-bridge``: be the agent's model, one model call per command.

Each command is a fresh process; the run lives in its directory
(:mod:`olmoearth_agent.bridge.store`).

    olmoearth-agent-bridge begin --run-dir D --brief "How many projects do I have?"
    olmoearth-agent-bridge step --run-dir D --response-file response.json --call 2
    olmoearth-agent-bridge step --run-dir D --response '{"content": "..."}'
    olmoearth-agent-bridge show --run-dir D --text
    olmoearth-agent-bridge prompt
    olmoearth-agent-bridge tools [--run-dir D]

``begin``, ``step`` and ``show`` print the step's view as JSON (``--text``:
readable); an error prints ``{"status": "error", "error": ...}`` and exits 1.
``step --call N`` names the model call the response answers, and is refused
when another is pending; ``step`` with no response starts a run whose first
step failed.
``--brief-file -`` and ``--response-file -`` read the text from stdin, as
``scripts/oe-bridge.sh`` sends it, so it passes through no shell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from olmoearth_agent.bridge.errors import BridgeError
from olmoearth_agent.bridge.session import (
    DEFAULT_MAX_TURNS,
    begin,
    render_text,
    show,
    step,
    system_prompt,
    tool_specs,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="olmoearth-agent-bridge",
        description=(
            "Let a Claude Code session be the OlmoEarth Agent's model: the "
            "agent's loop, tools and checks run here, one model call per command."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("begin", help="start a run and show its first call")
    start.add_argument("--run-dir", required=True, help="a new or empty directory")
    brief = start.add_mutually_exclusive_group(required=True)
    brief.add_argument("--brief", help="the user's request, word for word")
    brief.add_argument("--brief-file", help="a file holding the request (- stdin)")
    start.add_argument(
        "--history-file",
        help='prior turns, a JSON list of {"role": "user"|"assistant", "content"}',
    )
    start.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"cap on tool-calling turns (default: {DEFAULT_MAX_TURNS})",
    )
    start.add_argument("--text", action="store_true", help="print readable text")

    move = commands.add_parser("step", help="answer the pending call, show the next")
    move.add_argument("--run-dir", required=True)
    response = move.add_mutually_exclusive_group()
    response.add_argument(
        "--response-file", help="a file holding the response JSON (- stdin)"
    )
    response.add_argument("--response", help="the response JSON itself")
    move.add_argument(
        "--call",
        type=int,
        help="the model call the response answers (the view's model_call); "
        "refused when another is pending",
    )
    move.add_argument("--text", action="store_true", help="print readable text")

    again = commands.add_parser("show", help="the pending call again; nothing runs")
    again.add_argument("--run-dir", required=True)
    again.add_argument("--text", action="store_true", help="print readable text")

    commands.add_parser("prompt", help="the system prompt a new run's model gets")

    specs = commands.add_parser("tools", help="the offered tools' specs as JSON")
    specs.add_argument("--run-dir", help="the tools of this run's pending call")
    return parser.parse_args(argv)


def _read(path: str, what: str) -> str:
    """A file's text, as UTF-8; ``-`` is stdin."""
    try:
        if path == "-":
            return sys.stdin.buffer.read().decode("utf-8")
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        source = "stdin" if path == "-" else f"the file {path}"
        raise BridgeError(f"cannot read the {what} from {source}: {exc}") from exc


def _json(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"the {what} is not JSON: {exc}") from exc


def _response(args: argparse.Namespace) -> Any:
    """The response from ``--response-file`` or ``--response``, else ``None``."""
    if args.response_file:
        return _json(_read(args.response_file, "response"), "response")
    if args.response is not None:
        return _json(args.response, "response")
    return None


def _view(args: argparse.Namespace) -> dict[str, Any]:
    """Run ``begin``, ``step`` or ``show``."""
    if args.command == "show":
        return show(args.run_dir)
    if args.command == "begin":
        brief = args.brief
        if brief is None:
            brief = _read(args.brief_file, "brief")
        history = None
        if args.history_file:
            history = _json(_read(args.history_file, "history"), "history")
        return asyncio.run(
            begin(args.run_dir, brief, max_turns=args.max_turns, history=history)
        )
    return asyncio.run(step(args.run_dir, _response(args), call=args.call))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 = the command ran)."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    text = bool(getattr(args, "text", False))
    try:
        if args.command == "prompt":
            print(system_prompt())
            return 0
        if args.command == "tools":
            print(json.dumps(tool_specs(args.run_dir), indent=2, ensure_ascii=False))
            return 0
        view = _view(args)
    except (BridgeError, RuntimeError, ValueError) as exc:
        # RuntimeError: e.g. OLMOEARTH_API_KEY unset, or the loop's own
        # MiddlewareError; ValueError: a corrupt run file.
        if text:
            print(f"bridge error: {exc}", file=sys.stderr)
        else:
            print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(render_text(view) if text else json.dumps(view, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
