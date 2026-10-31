# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""``olmoearth-agent-bridge``: each command a fresh process on one run directory.

The session's production pieces (the default registry, the Studio client
from the environment, the skill index and the memory block) are swapped for
fakes, as ``tests/harness/test_answer_check_loop.py`` swaps them in
``cli.py``. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.bridge import cli, session
from olmoearth_agent.harness.agent import CHECK_ANSWER_ENV, CHECK_NUMBERS_ENV, LeadAgent
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_BRIEF = "How many windows does map a have?"
_ANSWER = "Map a has 1,570 windows."
_SCHEMA = {"type": "object", "properties": {"map": {"type": "string"}}}


def _registry() -> ToolRegistry:
    async def measure(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"map": args.get("map", "a"), "n_windows": 1570}

    registry = ToolRegistry()
    spec = ToolSpec("measure", "count a map's windows", _SCHEMA)
    registry.register(RegisteredTool(spec, measure))
    return registry


class _Studio:
    @classmethod
    def from_env(cls) -> _Studio:
        return cls()

    async def aclose(self) -> None:
        return None


class _Skills:
    def index(self) -> str:
        return ""


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CHECK_NUMBERS_ENV, raising=False)
    monkeypatch.delenv(CHECK_ANSWER_ENV, raising=False)
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "outputs"))
    monkeypatch.setattr(session, "build_default_registry", _registry)
    monkeypatch.setattr(session, "StudioClient", _Studio)
    monkeypatch.setattr(session, "SkillLoader", _Skills)
    monkeypatch.setattr(session, "preferences_block", lambda: "")


def _out(capsys: pytest.CaptureFixture[str]) -> Any:
    return json.loads(capsys.readouterr().out)


def test_a_brief_runs_one_command_per_model_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = str(tmp_path / "run")
    assert cli.main(["begin", "--run-dir", run, "--brief", _BRIEF]) == 0
    first = _out(capsys)
    assert first["status"] == "model_call" and first["tools"] == ["measure"]

    response = tmp_path / "response.json"
    call = {"name": "measure", "arguments": {"map": "a"}}
    response.write_text(json.dumps({"tool_calls": [call]}))
    argv = ["step", "--run-dir", run, "--response-file", str(response)]
    assert cli.main(argv) == 0
    second = _out(capsys)
    assert second["model_call"] == 2 and second["messages"][0]["role"] == "tool"

    answer = json.dumps({"content": _ANSWER})
    argv = ["step", "--run-dir", run, "--response", answer, "--text"]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out.splitlines()
    assert text[0].startswith("== final answer") and text[1] == _ANSWER

    assert cli.main(["show", "--run-dir", run]) == 0
    final = _out(capsys)
    assert final["status"] == "final" and final["content"] == _ANSWER
    assert sorted(p.name for p in Path(run).iterdir()) == [
        "brief.json",
        "events.jsonl",
        "pending.json",
        "provenance.json",
        "responses.jsonl",
        "tool_cache.jsonl",
    ]


def test_a_brief_can_come_from_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    brief = tmp_path / "brief.txt"
    brief.write_text("What's in map a? It's \"the\" map.")
    run = str(tmp_path / "run")
    argv = ["begin", "--run-dir", run, "--brief-file", str(brief), "--text"]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert text.startswith("== model call 1 (turn 1 of 8) ==")
    assert "What's in map a? It's \"the\" map." in text


def test_an_error_is_printed_as_json_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = str(tmp_path / "run")
    assert cli.main(["begin", "--run-dir", run, "--brief", _BRIEF]) == 0
    capsys.readouterr()
    assert cli.main(["step", "--run-dir", run, "--response", "{"]) == 1
    error = _out(capsys)
    assert error["status"] == "error" and "not JSON" in error["error"]
    assert cli.main(["step", "--run-dir", run]) == 1
    assert "waiting for a model response" in _out(capsys)["error"]
    assert cli.main(["show", "--run-dir", str(tmp_path / "none")]) == 1
    assert "no bridge run" in _out(capsys)["error"]


def test_prompt_and_tools_print_what_the_model_is_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["prompt"]) == 0
    expected = LeadAgent(
        object(),  # type: ignore[arg-type]
        _registry(),
        None,  # type: ignore[arg-type]
        skill_index="",
        memory_block="",
    ).system_prompt
    assert capsys.readouterr().out == expected + "\n"
    assert cli.main(["tools"]) == 0
    spec = {"name": "measure", "description": "count a map's windows"}
    assert _out(capsys) == [{**spec, "parameters": _SCHEMA}]
