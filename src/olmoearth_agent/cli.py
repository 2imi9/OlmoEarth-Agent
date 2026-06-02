# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Command-line entrypoint: run the OlmoEarth Agent on one brief.

Wires the verified pieces together: LLM client (env ``LLM_*``), Studio
client (env ``OLMOEARTH_API_KEY``), the default tool registry, the
vendored-skill index, into a :class:`~olmoearth_agent.harness.LeadAgent`
and runs a natural-language brief.

    olmoearth-agent "How many OlmoEarth projects do I have?"
    python -m olmoearth_agent --show-trace "Map alfalfa in the Klamath basin"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from olmoearth_agent.harness import AgentResult, LeadAgent
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm import OlmoEarthLLM
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


async def run_brief(
    brief: str,
    *,
    max_turns: int = 8,
    llm: OlmoEarthLLM | None = None,
    studio: StudioClient | None = None,
    registry: object | None = None,
    skill_index: str | None = None,
) -> AgentResult:
    """Assemble a :class:`LeadAgent` and run one brief.

    Dependencies default to the env-configured production objects but can
    be injected (for tests). Owns the :class:`StudioClient` lifecycle
    only when it creates one.
    """
    llm = llm or OlmoEarthLLM()
    registry = registry or build_default_registry()
    if skill_index is None:
        skill_index = SkillLoader().index()

    owns_studio = studio is None
    if studio is None:
        studio = StudioClient.from_env()
    try:
        agent = LeadAgent(
            llm,
            registry,  # type: ignore[arg-type]
            studio,
            state=ThreadState(),
            skill_index=skill_index,
            local=True,  # the CLI runs the local model; keep answers within budget
        )
        return await agent.run(brief, max_turns=max_turns)
    finally:
        if owns_studio:
            await studio.aclose()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 = answered)."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = asyncio.run(run_brief(args.brief, max_turns=args.max_turns))

    if args.show_trace:
        for name, ok in result.tool_calls:
            mark = "ok" if ok else "FAIL"
            print(f"  [{mark}] {name}", file=sys.stderr)
        if result.state is not None:
            print(
                f"  ({result.turns} turn(s), "
                f"{len(result.state.provenance.entries)} provenance entries)",
                file=sys.stderr,
            )

    if result.final_content is None:
        print("(no answer, hit the turn cap)", file=sys.stderr)
        return 1
    print(result.final_content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
