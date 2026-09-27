# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The round-1 harness fixes end to end: a scripted model against a real tool.

exp86 round 1 ended brief 4 on a Studio result with no answer: the model
retried a refused olmoearth_plan_label_sample until the turn cap. Round 1
added the "stop retrying" hint on a tool's second alike failure and a
forced answer, without tools, at the cap; round 2 never triggered either
(no run reached the cap, and no two refusals were alike). This test drives
both through the real plan tool, the registry's dispatch, the loop's
messages, the CLI and the web bridge, with a model that never gives up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.agent import TURN_CAP_PROMPT, LeadAgent
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolRegistry

pytest.importorskip("oe_inferencex.estimate")

#: Forty windows of two-class scores: any budget above 40 is refused alike.
_SCORES = [[0.1 + 0.02 * i, 0.9 - 0.02 * i] for i in range(40)]
_ANSWER = (
    "At most 40 labels can be planned from these scores, so the 300-label "
    "plan could not be made; plan 40 or fewer."
)


class _NeverGivesUp:
    """Asks for a plan of 300, 250, 200 ... labels while tools are offered.

    Offered none (the harness's call at the turn cap), it answers.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], Any]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        self.calls.append((list(messages), tools))
        if not tools:
            return ChatResponse(content=_ANSWER, tool_calls=[], finish_reason="stop")
        n = len(self.calls)
        call = ToolCall(
            id=f"c{n}",
            name="olmoearth_plan_label_sample",
            arguments={"scores": _SCORES, "budget": 350 - 50 * n},
        )
        return ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls")

    async def aclose(self) -> None:
        return None


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    return registry


@pytest.fixture(autouse=True)
def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_SCORES_ROOT", str(tmp_path))
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(tmp_path / "workspace"))


def _check_events(events: list[dict[str, Any]]) -> None:
    """What the loop yields, whichever front end streams it."""
    results = [e for e in events if e["type"] == "tool_result"]
    assert [r["ok"] for r in results] == [False, False, False]
    hints = [r["result"]["hint"] for r in results]
    assert "Stop retrying" not in hints[0]
    assert all("Stop retrying" in h for h in hints[1:])
    assert [r["result"].get("same_error_count") for r in results] == [None, 2, 3]
    assert "at most 40 labels" in results[0]["result"]["error"]
    tail = [e["type"] for e in events][-2:]
    assert tail == ["max_turns", "final"]
    assert events[-2]["final_answer_forced"] is True
    assert events[-1]["forced_by_turn_cap"] is True
    assert events[-1]["content"] == _ANSWER


@pytest.mark.asyncio
async def test_the_loop_warns_then_forces_the_answer() -> None:
    llm = _NeverGivesUp()
    agent = LeadAgent(llm, _registry(), studio=None)  # type: ignore[arg-type]
    events = [e async for e in agent.run_stream("plan 300 labels", max_turns=3)]
    _check_events(events)
    assert len(llm.calls) == 4
    # The hint reached the model: the third turn's messages carry the second
    # refusal's "Stop retrying" as the tool's own message.
    third, tools = llm.calls[2]
    assert tools
    tool_messages = [m for m in third if m.role == "tool"]
    assert len(tool_messages) == 2
    assert "Stop retrying" in (tool_messages[-1].content or "")
    assert "Stop retrying" not in (tool_messages[0].content or "")
    # The forced call offers no tools and says why, after the last result.
    last, tools = llm.calls[3]
    assert tools is None
    assert last[-1].role == "user" and last[-1].content == TURN_CAP_PROMPT
    assert last[-2].role == "tool"
    assert len(agent.state.provenance.entries) == 3


def test_the_cli_prints_the_forced_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from olmoearth_agent import cli

    class _Studio:
        @classmethod
        def from_env(cls) -> _Studio:
            return cls()

        async def aclose(self) -> None:
            return None

    class _Skills:
        def index(self) -> str:
            return ""

    llm = _NeverGivesUp()
    monkeypatch.setattr(cli, "OlmoEarthLLM", lambda: llm)
    monkeypatch.setattr(cli, "StudioClient", _Studio)
    monkeypatch.setattr(cli, "build_default_registry", _registry)
    monkeypatch.setattr(cli, "SkillLoader", _Skills)
    monkeypatch.setattr(cli, "preferences_block", lambda: "")
    code = cli.main(["plan 300 labels", "--max-turns", "3", "--show-trace"])
    out = capsys.readouterr()
    assert code == 0
    assert out.out.strip() == _ANSWER
    assert "turn cap (3)" in out.err
    assert out.err.count("[FAIL] olmoearth_plan_label_sample") == 3


def test_the_web_bridge_streams_the_same_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from olmoearth_agent import serve

    monkeypatch.setattr(serve, "preferences_block", lambda: "")
    with TestClient(serve.app) as client:
        serve.app.state.llm = _NeverGivesUp()
        serve.app.state.registry = _registry()
        serve.app.state.skill_index = ""
        resp = client.post(
            "/api/run",
            json={"brief": "plan 300 labels", "max_turns": 3},
            headers={"X-Olmoearth-Key": "k"},
        )
    assert resp.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == {"type": "done"}
    _check_events(events[:-1])
