# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Command-line entrypoint: run the OlmoEarth Agent on one brief.

Wires the verified pieces together: LLM client (env ``LLM_*``), Studio
client (env ``OLMOEARTH_API_KEY``), the default tool registry, the
vendored-skill index, into a :class:`~olmoearth_agent.harness.LeadAgent`
and runs a natural-language brief.

    olmoearth-agent "How many OlmoEarth projects do I have?"
    python -m olmoearth_agent --show-trace "Map alfalfa in the Klamath basin"

After the answer and a blank line it prints each figure a tool of the run
wrote (a check route's determination, ``olmoearth_agent.reporting.figures``):
its path, then its caption.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

import httpx
import openai

from olmoearth_agent.harness import AgentResult, LeadAgent
from olmoearth_agent.harness.capabilities import without_map
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.skills import SkillLoader, build_default_registry
from olmoearth_agent.studio import StudioClient
from olmoearth_agent.tools.registry import ToolRegistry

#: The note for a run that ended with no answer at the turn cap, and for one
#: whose model returned no text before it (round r5-n agent/F5: a blank answer
#: printed an empty line and exited 0, and no answer blamed the turn cap
#: though it was not reached): the flag is named only where the trace is not
#: shown already (:func:`no_answer_note`); the Claude Code bridge, which has no
#: such flag and shows the trace, gives the plain note.
# WORDING: r5n agent/F5
NO_ANSWER_AT_CAP = "(no answer: the run reached the turn cap)"
NO_ANSWER_TEXT = "(no answer: the model returned no text before the turn cap)"
NO_ANSWER_TEXT_HINT = (
    "(no answer: the model returned no text before the turn cap; run with "
    "--show-trace to see its tool calls)"
)
#: Said after a failure to reach the model server (round r5-n errors/B7).
LLM_DOWN_HINT = (
    "Is the model server running? Start it with make serve, or set "
    "LLM_ENDPOINT to one that is."
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="olmoearth-agent",
        description="Run the OlmoEarth Studio agent on a natural-language brief.",
    )
    parser.add_argument("brief", help="the task, in plain English")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="cap on LLM round-trips (default: 8)",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="print the tool-call trace to stderr",
    )
    return parser.parse_args(argv)


def figure_of(event: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(figure_path, figure_caption)`` of a tool result whose tool wrote a figure.

    ``event`` is one of :meth:`LeadAgent.run_stream`'s events; anything but a
    successful ``tool_result`` whose result names a ``figure_path`` gives None.
    """
    if event.get("type") != "tool_result" or not event.get("ok"):
        return None
    envelope = event.get("result")
    inner = envelope.get("result") if isinstance(envelope, Mapping) else None
    if not isinstance(inner, Mapping):
        return None
    path, caption = inner.get("figure_path"), inner.get("figure_caption")
    if not isinstance(path, str) or not path:
        return None
    return path, (caption if isinstance(caption, str) else "")


def figure_lines(figures: Iterable[tuple[str, str]]) -> list[str]:
    """The lines printed after the answer for ``(figure_path, caption)`` pairs.

    A blank line first, so a Markdown answer that ends in a table or a list
    stays apart from them, then each figure's ``figure: <path>`` and its
    caption (none when it has no caption); no line at all without a figure.
    """
    lines: list[str] = []
    for path, caption in figures:
        lines.append(f"figure: {path}")
        if caption:
            lines.append(caption)
    return ["", *lines] if lines else []


def no_answer_note(result: AgentResult, *, trace_shown: bool = False) -> str | None:
    """The note for a run with no answer, or None when it answered.

    A blank answer is no answer; the turn cap is blamed only when the run
    reached it (round r5-n agent/F5): a local model whose output budget went
    on thinking returns no text and no tool call before the cap. Its note
    names ``--show-trace`` only when the trace is not shown already
    (``trace_shown``: the flag was given, or the Claude Code bridge, whose
    final view holds the trace and which has no such flag).
    """
    if (result.final_content or "").strip():
        return None
    if result.hit_max_turns:
        return NO_ANSWER_AT_CAP
    return NO_ANSWER_TEXT if trace_shown else NO_ANSWER_TEXT_HINT


def error_line(exc: BaseException) -> str:
    """The one line ``olmoearth-agent`` prints for a run that failed to run.

    The error's own text (its class name when it has none), and, when the
    model server could not be reached, :data:`LLM_DOWN_HINT` (round r5-n
    errors/B7: a missing key or a model server not running printed a
    Python traceback).
    """
    text = str(exc).strip() or type(exc).__name__
    if isinstance(exc, openai.APIConnectionError):
        text = f"{text} {LLM_DOWN_HINT}"
    return f"olmoearth-agent: {text}"


def cli_registry() -> ToolRegistry:
    """The command line's tools: the default registry with no draw tool.

    ``olmoearth_request_aoi`` only tells a map in the web UI or the MCP App
    to open, and the command line shows none (round r5-n agent/F6): it is
    left out, and the capability card says to ask for a saved area's id
    (:func:`~olmoearth_agent.harness.capabilities.without_map`).
    """
    return without_map(build_default_registry())


class _FigureAgent(LeadAgent):
    """A :class:`LeadAgent` that keeps each figure its run's tools wrote.

    ``figures`` maps each ``figure_path`` to its caption, in the order the
    tools wrote them; a figure written again takes the newer caption.
    """

    def __init__(self, *args: Any, figures: dict[str, str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.figures = figures

    async def run_stream(
        self,
        brief: str,
        *,
        max_turns: int = 8,
        history: list[Message] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """The run's events as :meth:`LeadAgent.run_stream` yields them."""
        async for event in super().run_stream(
            brief, max_turns=max_turns, history=history
        ):
            figure = figure_of(event)
            if figure is not None:
                self.figures[figure[0]] = figure[1]
            yield event


async def run_brief(
    brief: str,
    *,
    max_turns: int = 8,
    llm: OlmoEarthLLM | None = None,
    studio: StudioClient | None = None,
    registry: object | None = None,
    skill_index: str | None = None,
    figures: dict[str, str] | None = None,
) -> AgentResult:
    """Assemble a :class:`LeadAgent` and run one brief.

    Dependencies default to the env-configured production objects but can
    be injected (for tests); the default registry is :func:`cli_registry`,
    with no draw tool. Owns the :class:`StudioClient` lifecycle
    only when it creates one. ``figures``, when given, receives each figure
    a tool of the run wrote (:func:`figure_of`): its path and caption.
    """
    llm = llm or OlmoEarthLLM()
    registry = registry or cli_registry()
    if skill_index is None:
        skill_index = SkillLoader().index()

    owns_studio = studio is None
    if studio is None:
        studio = StudioClient.from_env()
    try:
        agent = _FigureAgent(
            llm,
            registry,
            studio,
            state=ThreadState(),
            skill_index=skill_index,
            local=True,  # the CLI runs the local model; keep answers within budget
            figures={} if figures is None else figures,
        )
        return await agent.run(brief, max_turns=max_turns)
    finally:
        if owns_studio:
            await studio.aclose()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 = answered).

    stdout is the answer followed by any figure lines (:func:`figure_lines`:
    a blank line, then each figure's path and caption); the trace and the
    notes go to stderr. A run that could not start or finish (no Studio key,
    the model server down, an error of the model's API) prints one line,
    :func:`error_line`, and returns 2; a run with no answer returns 1
    (:func:`no_answer_note`).
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    figures: dict[str, str] = {}
    try:
        result = asyncio.run(
            run_brief(args.brief, max_turns=args.max_turns, figures=figures)
        )
    except (RuntimeError, openai.APIError, httpx.HTTPError) as exc:
        print(error_line(exc), file=sys.stderr)
        return 2

    if args.show_trace:
        for name, ok in result.tool_calls:
            mark = "ok" if ok else "FAIL"
            print(f"  [{mark}] {name}", file=sys.stderr)
        for check in result.grounding_checks:
            if check.get("action") == "revise":
                listed = ", ".join(check.get("unsupported", []))
                print(
                    f"  [numbers] in no tool result, rewrite asked: {listed}",
                    file=sys.stderr,
                )
        for check in result.checks:
            if check.get("action") == "revise" and check.get("check") != "numbers":
                n = len(check.get("violations", []))
                print(
                    f"  [{check.get('check')}] {n} sentence(s) flagged, rewrite asked",
                    file=sys.stderr,
                )
        if result.state is not None:
            print(
                f"  ({result.turns} turn(s), "
                f"{len(result.state.provenance.entries)} provenance entries)",
                file=sys.stderr,
            )

    missing = no_answer_note(result, trace_shown=args.show_trace)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 1
    if result.hit_max_turns:
        # The web UI shows the same note above the answer (webui/js/run.js).
        print(
            f"(reached the turn cap ({args.max_turns}): the answer was written "
            "from what the tools returned, without further tool calls)",
            file=sys.stderr,
        )
    for check in result.grounding_checks:
        if check.get("action") == "shown":
            # The web UI shows the same note above the answer (webui/js/run.js).
            print(
                "(numbers in the answer that no tool returned: "
                + ", ".join(check.get("unsupported", []))
                + ")",
                file=sys.stderr,
            )
    for check in result.checks:
        if check.get("action") == "marked" and check.get("check") != "numbers":
            # The answer carries "[unverified: <check>]" after each sentence.
            print(
                f"(the {check.get('check')} check still flags "
                f"{len(check.get('violations', []))} statement(s) of the answer; "
                "each is marked [unverified])",
                file=sys.stderr,
            )
    print(result.final_content)
    # The web UI shows each figure under its tool's result (webui/js/viz.js).
    for line in figure_lines(figures.items()):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
