# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Command-line entrypoint: run the OlmoEarth Agent on one brief.

Wires the verified pieces together: LLM client (env ``LLM_*``), Studio
client (env ``OLMOEARTH_API_KEY``), the default tool registry, the
vendored-skill index, into a :class:`~olmoearth_agent.harness.LeadAgent`
and runs a natural-language brief.

    olmoearth-agent "How many OlmoEarth projects do I have?"
    python -m olmoearth_agent --show-trace "Map alfalfa in the Klamath basin"

After the answer it prints each figure a tool of the run wrote (a check
route's determination, ``olmoearth_agent.reporting.figures``): its path, then
its caption.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from typing import Any

from olmoearth_agent.harness import AgentResult, LeadAgent
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm import OlmoEarthLLM
from olmoearth_agent.llm.types import Message
from olmoearth_agent.skills import SkillLoader, build_default_registry
from olmoearth_agent.studio import StudioClient


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
    be injected (for tests). Owns the :class:`StudioClient` lifecycle
    only when it creates one. ``figures``, when given, receives each figure
    a tool of the run wrote (:func:`figure_of`): its path and caption.
    """
    llm = llm or OlmoEarthLLM()
    registry = registry or build_default_registry()
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
    """CLI entrypoint. Returns a process exit code (0 = answered)."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    figures: dict[str, str] = {}
    result = asyncio.run(
        run_brief(args.brief, max_turns=args.max_turns, figures=figures)
    )

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

    if result.final_content is None:
        print("(no answer, hit the turn cap)", file=sys.stderr)
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
    for path, caption in figures.items():
        print(f"figure: {path}")
        if caption:
            print(caption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
