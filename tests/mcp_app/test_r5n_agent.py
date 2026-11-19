# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Round r5-n (19 Nov 2026), track AGENT: the pre-release bug fixes of the
agent's own surfaces, its model clients, the web UI bridge and the command
line (``specs/bughunt-2026-11-19/confirmed.json``).

- agent/F1: no sampling parameter goes to Claude; the payload binds to the
  real SDK's ``messages.create``.
- agent/F2: ChatGPT and Gemini get no 32,768-token cap; OpenAI's API takes
  ``max_completion_tokens``, and its reasoning models get no sampling.
- agent/F3: a sentence an earlier web turn left marked ``[unverified]`` is no
  source for the next turn.
- agent/F4: auto-route moves a run to the local model only when it answers.
- agent/F5: a blank answer is no answer; the turn cap is blamed only when hit;
  the flag is named only where the trace is not shown.
- agent/F6: the command line and the Claude Code bridge offer no draw tool.
- agent/F7: arguments that are not one JSON object are refused unrun.
- agent/F8: a refused key on the pixel value is an error with its hint.
- agent/F9: a project created or a prediction submitted in a run drops the
  Projects tree's cached reads.
- errors/B2: a Studio timeout, rate limit or 5xx says so, with a hint to try
  again later, not to check the ids, on the Claude Code bridge too; a lost
  create's own hint (errors/B1) reaches every surface.
- errors/B7: the command line prints one line for a missing key or a model
  server that is down.
- errors/B10: the web UI's difference scan keeps cells Studio did not answer
  apart from no-data, and a refused key apart from both.
- security/SEC-5: the web UI bridge answers loopback names only, and the
  names an operator adds as a ``Host`` carries them.
- packaging/PKG-2: the checkout pins Python 3.12.
- packaging/PKG-4: the bridge says where the web UI is missing.
- numbers/F1: the stakeholder report states no rate from a withdrawn result,
  nor from a zone or certificate on its sheet, and never reads it unmarked.

Nothing here reaches Studio or a model: answers are mocked or stand-ins.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from pytest_httpx import HTTPXMock

from olmoearth_agent import cli
from olmoearth_agent.bridge import session
from olmoearth_agent.bridge.replay import ToolCache, replay_registry, replayed_error
from olmoearth_agent.bridge.store import RunDir
from olmoearth_agent.harness.agent import AgentResult
from olmoearth_agent.harness.capabilities import (
    DRAW_AREA_TOOL,
    NO_MAP,
    capability_card,
)
from olmoearth_agent.harness.checks import (
    RunEvidence,
    ToolRecord,
    check_forbidden_claims,
    check_numbers,
)
from olmoearth_agent.harness.claim_check import claim_check_messages
from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm import OlmoEarthLLM, ServingConfig
from olmoearth_agent.llm.anthropic_client import AnthropicLLM
from olmoearth_agent.llm.presets import (
    CLAIM_CHECK_MAX_TOKENS,
    CLAIM_CHECK_MODE,
    DEFAULT_AGENT_MODE,
)
from olmoearth_agent.llm.types import (
    RAW_ARGUMENTS,
    ChatResponse,
    Message,
    ToolCall,
    ToolSpec,
)
from olmoearth_agent.reporting.narrative import (
    CERTIFY_TOOL,
    ESTIMATE_TOOL,
    NOT_MEASURED,
    WITHDRAWN,
    WITHDRAWN_NOT_MEASURED,
    WITHDRAWN_SAME_SHEET,
    WRITE_CERTIFICATE_TOOL,
    ZONE_MEANING,
    accuracy_section,
)
from olmoearth_agent.skills import build_default_registry
from olmoearth_agent.studio.client import (
    KEY_REJECTED_HINT,
    OUTCOME_UNKNOWN_HINT,
    READ_REFUSED_HINT,
    StudioAccessError,
    StudioClient,
    StudioConfig,
    StudioOutcomeUnknown,
)
from olmoearth_agent.tools.predict import build_predict_tools
from olmoearth_agent.tools.registry import (
    STOP_RETRYING_HINT,
    STUDIO_BUSY_HINT,
    TIMEOUT_TEXT,
    RegisteredTool,
    ToolContext,
    ToolRegistry,
    own_hint,
)

#: The stage's root: this file is tests/mcp_app/test_r5n_agent.py.
ROOT = Path(__file__).resolve().parents[2]
STUDIO = "http://mock-studio/api/v1"
_EMPTY = {"type": "object", "properties": {}, "required": []}


# --------------------------------------------------------------------------- agent/F1


class _FakeMessages:
    """The SDK's ``messages`` resource: keeps the payload, answers one text."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def create(self, **payload: Any) -> Any:
        """Keep the payload; a reply as ``Message.model_dump`` gives it."""
        self._captured.update(payload)
        return types.SimpleNamespace(
            model_dump=lambda: {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        )


@pytest.mark.asyncio
async def test_f1_claudes_payload_binds_to_the_real_sdk() -> None:
    """anthropic 1.8.0 (the lock's) takes no temperature, top_p or top_k:
    every Claude call raised TypeError before any request. The payload now
    binds to the installed SDK's own ``messages.create`` signature."""
    pytest.importorskip("anthropic")
    resources = importlib.import_module("anthropic.resources")
    messages_cls = getattr(resources, "AsyncMessages", None)
    if messages_cls is None:
        messages_cls = getattr(
            importlib.import_module("anthropic.resources.messages"),
            "AsyncMessages",
            None,
        )
    if messages_cls is None:
        pytest.skip("this anthropic SDK names no AsyncMessages")
    llm = AnthropicLLM(model="claude-x", api_key="sk-test")
    captured: dict[str, Any] = {}
    fake = types.SimpleNamespace(messages=_FakeMessages(captured))
    llm._client = fake  # type: ignore[assignment]
    spec = ToolSpec(name="f", description="d", parameters={"type": "object"})
    await llm.chat(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        tools=[spec],
        mode=DEFAULT_AGENT_MODE,
    )
    assert not {"temperature", "top_p", "top_k"} & set(captured)
    # Raises TypeError for a keyword the SDK does not take.
    inspect.signature(messages_cls.create).bind(None, **captured)


# --------------------------------------------------------------------------- agent/F2


def _payload(
    endpoint: str,
    model: str,
    *,
    mode: Any = DEFAULT_AGENT_MODE,
    cap: int | None = None,
) -> dict[str, Any]:
    """The Chat Completions payload a hosted backend (openai_compat) sends."""
    llm = OlmoEarthLLM(
        ServingConfig(endpoint=endpoint, model=model, api_key="k"), openai_compat=True
    )
    return llm._build_payload(
        messages=[Message(role="user", content="hi")],
        tools=None,
        mode=mode,
        max_tokens=cap,
        preserve_thinking=True,
    )


OPENAI = "https://api.openai.com/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_f2_chatgpt_and_gemini_get_no_local_cap() -> None:
    """gpt-4o allows 16,384 output tokens and gemini-2.0-flash 8,192: the local
    model's 32,768 was an HTTP 400 on every call. No cap is sent unless the
    caller gives one."""
    for endpoint, model in ((OPENAI, "gpt-4o"), (GEMINI, "gemini-2.0-flash")):
        payload = _payload(endpoint, model)
        assert "max_tokens" not in payload and "max_completion_tokens" not in payload
        assert payload["temperature"] == 1.0  # no reasoning model: sampling stays
    assert _payload(GEMINI, "gemini-2.0-flash", cap=4096)["max_tokens"] == 4096
    # The local model keeps its own budget.
    local = OlmoEarthLLM(ServingConfig(endpoint="http://localhost:8000/v1"))
    built = local._build_payload(
        messages=[Message(role="user", content="hi")],
        tools=None,
        mode=DEFAULT_AGENT_MODE,
        max_tokens=None,
        preserve_thinking=True,
    )
    assert built["max_tokens"] == 32768


def test_f2_a_reasoning_model_gets_its_cap_by_its_name_and_no_sampling() -> None:
    """o1, o3 and gpt-5 refuse max_tokens and the preset's sampling: the claim
    check's 4,096 arrives as max_completion_tokens, with no temperature, top_p
    or penalty."""
    for model in ("o3", "o1-mini", "gpt-5"):
        payload = _payload(
            OPENAI, model, mode=CLAIM_CHECK_MODE, cap=CLAIM_CHECK_MAX_TOKENS
        )
        assert payload["max_completion_tokens"] == CLAIM_CHECK_MAX_TOKENS
        assert "max_tokens" not in payload and "extra_body" not in payload
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            assert key not in payload, (model, key)


# --------------------------------------------------------------------------- agent/F3

_FORBIDDEN = [{"id": "accuracy_not_measured", "why": "x"}]
_ENVELOPE = {"ok": True, "result": {"forbidden_claims": _FORBIDDEN}}
_REPORT = ToolRecord("olmoearth_case_narrative", {}, _ENVELOPE)
#: Turn 1 as the web UI sent it back: a figure no tool returned, left marked.
_TURN_ONE = (
    "The sample has 300 labels. The map's error rate is 23%. "
    "[unverified: numbers] [unverified: forbidden_claims]\n\n"
    "[unverified: numbers] The rest is 41% of the windows."
)


def _web_run() -> RunEvidence:
    """Turn 2 on the web UI, with turn 1 in its history."""
    return RunEvidence(
        tools=[_REPORT],
        user_messages=["Summarise that for my report."],
        surface="web",
        assistant_messages=[_TURN_ONE],
    )


def test_f3_a_sentence_left_marked_is_no_source_for_the_next_turn() -> None:
    """A figure flagged on turn 1 and shown marked passed unmarked when
    restated on turn 2: the earlier turn was a source for the number check
    and exempted the accuracy claim. Only its sentences that passed are."""
    run = _web_run()
    restated = "The error rate measured earlier is 23%."
    assert [v["detail"] for v in check_numbers(restated, run)] == ["23%"]
    assert [v["text"] for v in check_forbidden_claims(restated, run)] == [restated]
    # A line mark_answer appended at the end is left out too.
    assert [v["detail"] for v in check_numbers("It is 41%.", run)] == ["41%"]
    # The sentence of the same turn that passed is still a source.
    assert check_numbers("The sample has 300 labels.", run) == []
    record = claim_check_messages("A.", run, "")[1].content or ""
    assert "300 labels" in record and "23%" not in record and "41%" not in record


# --------------------------------------------------------------------------- agent/F4 and F9


class _CapAgent:
    """Stands in for LeadAgent: keeps whether the run is local; its events."""

    seen: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    def __init__(self, *_a: Any, **kw: Any) -> None:
        _CapAgent.seen["local"] = kw.get("local")

    async def run_stream(self, *_a: Any, **_kw: Any) -> Any:
        """The scripted events, then a final answer."""
        for event in _CapAgent.events:
            yield event
        yield {"type": "final", "turn": 1, "content": "ok"}


def _bridge(monkeypatch: pytest.MonkeyPatch) -> Any:
    """serve, with LeadAgent stood in for and the read cache empty."""
    pytest.importorskip("fastapi")
    from olmoearth_agent import serve

    monkeypatch.delenv(serve.ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.setattr(serve, "LeadAgent", _CapAgent)
    _CapAgent.seen.clear()
    _CapAgent.events = []
    serve._READ_CACHE.clear()
    return serve


HOSTED = {
    "X-Olmoearth-Key": "k",
    "X-LLM-Backend": "openai",
    "X-LLM-Key": "sk-x",
    "X-LLM-Route": "auto",
}


def test_f4_auto_route_keeps_the_hosted_backend_when_no_local_model_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cloud-only user with auto-route on got a connection error on every
    simple question: the run moved to a local model that was not running."""
    serve = _bridge(monkeypatch)
    from fastapi.testclient import TestClient

    probed: list[str] = []

    async def local_down(endpoint: str, _api_key: str | None = None) -> bool:
        """No local model answers."""
        probed.append(endpoint)
        return False

    monkeypatch.setattr(serve, "_local_llm_up", local_down)
    with TestClient(serve.app, base_url="http://127.0.0.1") as client:
        client.post("/api/run", json={"brief": "list my projects"}, headers=HOSTED)
        assert _CapAgent.seen["local"] is False and len(probed) == 1
        # A brief that would not move is not probed at all.
        client.post(
            "/api/run",
            json={"brief": "create and train a flood model"},
            headers=HOSTED,
        )
    assert _CapAgent.seen["local"] is False and len(probed) == 1


def _result(name: str, ok: bool) -> dict[str, Any]:
    """A tool_result event as LeadAgent.run_stream yields it."""
    inner = {"id": "p9", "name": "New project"} if ok else None
    envelope = {"ok": True, "result": inner} if ok else {"ok": False, "error": "x"}
    return {
        "type": "tool_result",
        "turn": 1,
        "id": "c1",
        "name": name,
        "ok": ok,
        "result": envelope,
    }


def test_f9_a_project_created_or_a_prediction_submitted_drops_the_cached_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Projects tree kept its cached list for up to 300 s (predictions
    30 s) after the agent created a project or submitted a prediction."""
    serve = _bridge(monkeypatch)
    from fastapi.testclient import TestClient

    tag = serve._key_hash("k")
    keys = {
        "projects": f"projects|{tag}",
        "preds": f"preds|{tag}|pr1",
        "areas": f"areas|{tag}|pr1",
    }
    run = {"brief": "make a project", "max_turns": 2}
    try:
        with TestClient(serve.app, base_url="http://127.0.0.1") as client:
            for key in keys.values():
                serve._read_cache_put(key, {"ok": True}, serve._READ_TTL_LONG)
            # A failed write drops nothing.
            _CapAgent.events = [_result("olmoearth_create_project", False)]
            client.post("/api/run", json=run, headers={"X-Olmoearth-Key": "k"})
            assert set(keys.values()) <= set(serve._READ_CACHE)
            _CapAgent.events = [_result("olmoearth_create_project", True)]
            client.post("/api/run", json=run, headers={"X-Olmoearth-Key": "k"})
            assert keys["projects"] not in serve._READ_CACHE
            assert keys["preds"] in serve._READ_CACHE
            _CapAgent.events = [_result("olmoearth_submit_prediction", True)]
            client.post("/api/run", json=run, headers={"X-Olmoearth-Key": "k"})
            assert keys["preds"] not in serve._READ_CACHE
            assert keys["areas"] in serve._READ_CACHE  # not the run's to drop
    finally:
        serve._READ_CACHE.clear()
    # The browser drops its own copy on the same event.
    run_js = (ROOT / "webui" / "js" / "run.js").read_text(encoding="utf-8")
    assert (
        "import { apiRunStream, apiSkills, invalidateApi } from './api.js';" in run_js
    )
    assert (
        "if (ev.ok && ev.name === 'olmoearth_create_project') "
        "invalidateApi('/api/projects');" in run_js
    )
    assert (
        "if (ev.ok && ev.name === 'olmoearth_submit_prediction') "
        "invalidateApi('/predictions');" in run_js
    )


# --------------------------------------------------------------------------- agent/F5


class _Silent:
    """A local model that spent its output budget thinking: no text, no call."""

    async def chat(self, *_a: Any, **_kw: Any) -> ChatResponse:
        """The empty answer of a reasoning parser at finish_reason 'length'."""
        return ChatResponse(content="", tool_calls=[], finish_reason="length")


class _Studio:
    """StudioClient.from_env, offline."""

    @classmethod
    def from_env(cls) -> _Studio:
        """A client that reaches nothing."""
        return cls()

    async def aclose(self) -> None:
        """Nothing to close."""


class _NoSkills:
    """SkillLoader with no index."""

    def index(self) -> str:
        """No skill index."""
        return ""


def test_f5_a_blank_answer_is_no_answer_and_names_no_turn_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line printed an empty line and exited 0 for a blank answer,
    and blamed the turn cap for no answer though it was not reached."""
    monkeypatch.setattr(cli, "OlmoEarthLLM", _Silent)
    monkeypatch.setattr(cli, "StudioClient", _Studio)
    monkeypatch.setattr(cli, "build_default_registry", ToolRegistry)
    monkeypatch.setattr(cli, "SkillLoader", _NoSkills)
    assert cli.main(["How many projects do I have?"]) == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err.splitlines() == [cli.NO_ANSWER_TEXT_HINT]
    assert "reached the turn cap" not in printed.err
    # With the trace shown the note names no flag: the trace is above it.
    assert cli.main(["How many projects do I have?", "--show-trace"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and len(printed.err.splitlines()) >= 2
    assert printed.err.splitlines()[-1] == cli.NO_ANSWER_TEXT
    assert "--show-trace" not in printed.err
    # The Claude Code bridge gives the notes of `olmoearth-agent --show-trace`:
    # its view holds the trace, and it has no such flag.
    blank = AgentResult(final_content="  ", turns=1, tool_calls=[])
    assert session.cli_notes(blank, 8) == [cli.NO_ANSWER_TEXT]
    assert "--show-trace" not in cli.NO_ANSWER_TEXT
    capped = AgentResult(final_content=None, turns=9, tool_calls=[], hit_max_turns=True)
    assert session.cli_notes(capped, 8) == [cli.NO_ANSWER_AT_CAP]


# --------------------------------------------------------------------------- agent/F6


def test_f6_the_command_line_and_the_bridge_offer_no_draw_tool() -> None:
    """The draw tool only opens a map in the web UI or the MCP App; on the
    command line a brief with no area ended at a map that is not there."""
    assert DRAW_AREA_TOOL in build_default_registry().names()  # web UI, MCP App
    for registry in (cli.cli_registry(), session._default_registry()):
        assert DRAW_AREA_TOOL not in registry.names()
        assert registry.cannot == {"draw_area": NO_MAP}
        assert f"- {NO_MAP}." in capability_card(registry)
    # A registry with no draw tool keeps its card as it was.
    empty = ToolRegistry()
    assert not empty.leave_out(DRAW_AREA_TOOL, reason="draw_area", line=NO_MAP)
    assert empty.cannot == {}


# --------------------------------------------------------------------------- agent/F7


class _CountingStudio:
    """A Studio client that counts the searches it is asked for."""

    def __init__(self) -> None:
        self.searches = 0

    async def search_predictions(self, *_a: Any, **_kw: Any) -> Any:
        """Never reached: the call is refused before its handler."""
        self.searches += 1
        raise AssertionError("searched")


@pytest.mark.asyncio
async def test_f7_arguments_that_are_not_one_json_object_are_refused_unrun() -> None:
    """Truncated JSON came through as {'__raw_arguments': ...}, passed the
    schema check (no required argument) and searched every prediction of the
    account instead of the project's."""
    registry = ToolRegistry()
    registry.register_all(build_predict_tools())
    studio = _CountingStudio()
    raw = '{"project_id": "abc"'
    out = await registry.dispatch(
        ToolCall("c1", "olmoearth_search_predictions", {RAW_ARGUMENTS: raw}),
        ToolContext(studio=studio, state=ThreadState()),  # type: ignore[arg-type]
    )
    assert out["ok"] is False and studio.searches == 0
    assert "not one JSON object" in out["error"] and repr(raw) in out["error"]
    assert "expected_arguments" in out


# --------------------------------------------------------------------------- agent/F8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "hint"), [(403, READ_REFUSED_HINT), (401, KEY_REJECTED_HINT)]
)
async def test_f8_a_refused_key_on_the_pixel_value_is_an_error_with_its_hint(
    httpx_mock: HTTPXMock, status: int, hint: str
) -> None:
    """A refused key came back ok, 'off-raster, nodata, or transient': the
    refusal's hint and the sentence the model must state were lost."""
    httpx_mock.add_response(
        url=re.compile(r".*/prediction-results/r1/pixel-value\?.*"),
        status_code=status,
        json={"errors": [{"code": "permission_error", "message": "no"}]},
    )
    registry = ToolRegistry()
    registry.register_all(build_predict_tools())
    args = {"result_id": "r1", "lon": 0.0, "lat": 0.0}
    async with StudioClient(StudioConfig(api_key="k", base_url=STUDIO)) as studio:
        out = await registry.dispatch(
            ToolCall("c1", "olmoearth_pixel_value", args),
            ToolContext(studio=studio, state=ThreadState()),
        )
    assert out["ok"] is False and out["error"].startswith("StudioAccessError: ")
    assert out["hint"] == hint
    assert "nothing was created or changed" in out["must_state"][0]
    assert len(httpx_mock.get_requests()) == 1  # never retried


# --------------------------------------------------------------------------- errors/B2


def _raising(exc: Exception) -> ToolRegistry:
    """A registry of one tool that raises ``exc``."""

    async def boom(_args: dict[str, Any], _ctx: ToolContext) -> None:
        """Studio's failure, as the client raises it."""
        raise exc

    registry = ToolRegistry()
    registry.register(RegisteredTool(ToolSpec("olmoearth_read", "r", _EMPTY), boom))
    return registry


@pytest.mark.asyncio
async def test_b2_studio_failing_says_so_and_asks_to_try_again_later() -> None:
    """httpx's timeouts have an empty text ('ReadTimeout: '), and a 429 or 5xx
    got the hint to check the ids and not to retry, ruling out the one step
    that works."""
    state = ThreadState()
    call = ToolCall("c1", "olmoearth_read", {})
    ctx = ToolContext(studio=None, state=state)  # type: ignore[arg-type]
    out = await _raising(httpx.ReadTimeout("")).dispatch(call, ctx)
    assert out["error"] == f"ReadTimeout: {TIMEOUT_TEXT}"
    assert out["hint"] == STUDIO_BUSY_HINT
    # The same failure again: the stop hint still bounds the retries.
    again = await _raising(httpx.ReadTimeout("")).dispatch(call, ctx)
    assert again["hint"] == STOP_RETRYING_HINT.format(count=2)
    request = httpx.Request("GET", f"{STUDIO}/projects/p1")
    for status, busy in ((503, True), (429, True), (404, False)):
        exc = httpx.HTTPStatusError(
            f"HTTP {status}", request=request, response=httpx.Response(status)
        )
        fresh = ToolContext(studio=None, state=ThreadState())  # type: ignore
        out = await _raising(exc).dispatch(call, fresh)
        assert out["error"] == f"HTTPStatusError: HTTP {status}"
        assert (out["hint"] == STUDIO_BUSY_HINT) is busy, status
    # A card says the words alone.
    from olmoearth_agent.mcp_app import server

    assert server._card_error(f"ReadTimeout: {TIMEOUT_TEXT}") == TIMEOUT_TEXT


def _fresh() -> ToolContext:
    """A new run's tool context, with no Studio client (the tool raises)."""
    return ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]


def _status(code: int) -> httpx.HTTPStatusError:
    """Studio's answer ``code`` to a read, as httpx raises it."""
    request = httpx.Request("GET", f"{STUDIO}/projects/p1")
    return httpx.HTTPStatusError(
        f"HTTP {code}", request=request, response=httpx.Response(code)
    )


@pytest.mark.asyncio
async def test_b2_the_claude_code_bridge_gives_the_live_envelope(
    tmp_path: Path,
) -> None:
    """The bridge serves even the live call from its cache record, which kept
    a timeout's empty text and only a refused key's hint: every step said
    'ReadTimeout: ' with the hint to check the ids (round r5-n review)."""
    call = ToolCall("c1", "olmoearth_read", {})
    lost = StudioOutcomeUnknown("POST", "/projects", httpx.ReadTimeout(""))
    cases: list[tuple[Exception, str | None]] = [
        (httpx.ReadTimeout(""), STUDIO_BUSY_HINT),
        (_status(503), STUDIO_BUSY_HINT),
        (_status(404), None),  # the generic hint, as live
        (StudioAccessError("GET", "/projects/p1", 403), READ_REFUSED_HINT),
        (lost, OUTCOME_UNKNOWN_HINT),
    ]
    steps: list[dict[str, Any]] = []
    for n, (exc, hint) in enumerate(cases):
        live = await _raising(exc).dispatch(call, _fresh())
        run = RunDir(tmp_path / f"run{n}")
        run.path.mkdir()
        steps = []
        for step in (1, 2):  # the live call, then its record served again
            cache = ToolCache(run, saved=step - 1, model_call=step)
            bridged = replay_registry(_raising(exc), cache)
            steps.append(await bridged.dispatch(call, _fresh()))
        assert steps[0] == steps[1] == live, type(exc).__name__
        if n == 0:  # the timeout: its words, not "ReadTimeout: " alone
            assert steps[0]["error"] == f"ReadTimeout: {TIMEOUT_TEXT}"
        if hint is None:
            assert "Check that referenced ids exist" in live["hint"]
        else:
            assert live["hint"] == hint
        (record,) = run.read_tool_cache()
        assert own_hint(replayed_error(record["error"])) == hint
    assert steps[0]["must_state"] == lost.contract["must_state"]
    # The timeout's words, on the bridge too.
    run = RunDir(tmp_path / "run0")
    (record,) = run.read_tool_cache()
    assert record["error"]["message"] == TIMEOUT_TEXT
    # A record cached before this kept no hint and an empty text: it replays
    # as the model first read it, the generic hint and 'ReadTimeout: '.
    old = replayed_error({"type": "ReadTimeout", "message": "", "contract": None})
    assert own_hint(old) is None and str(old) == ""


def _app_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP App's output root and key file, the test's own: no key."""
    from olmoearth_agent.mcp_app import auth

    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setenv("OLMOEARTH_OUTPUT_ROOT", str(out))
    monkeypatch.delenv("OLMOEARTH_SCORES_ROOT", raising=False)
    monkeypatch.delenv(auth.KEY_ENV, raising=False)
    monkeypatch.setenv(auth.KEY_FILE_ENV, str(tmp_path / "cfg" / "studio.env"))


def test_b2_a_timed_out_review_sets_view_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness's STUDIO_TIMEOUT_VIEW copies this, word for word: the
    card's words without the exception's name, no hint, and the review set's
    lead as _lead gives a failed call (result.html draws no lead on an error
    card)."""
    from olmoearth_agent.mcp_app import auth, server

    _app_roots(tmp_path, monkeypatch)
    app = server.AppServer(None, keys=auth.KeyStore(), base_url=STUDIO)
    name = "olmoearth_review_set_from_result"
    envelope = {
        "ok": False,
        "tool": name,
        "error": f"ReadTimeout: {TIMEOUT_TEXT}",
        "hint": STUDIO_BUSY_HINT,
    }
    assert app._view_data(name, envelope, None) == {
        "tool": name,
        "ok": False,
        "error": "Studio did not answer in time",
        "hint": None,
        "needs_sign_in": False,
        "must_state": None,
        "lead": {
            "what": None,
            "next": "To learn how wrong the map is, check a random pick of points.",
            "cannot": (
                "It does not say these are wrong or how wrong the map is; no "
                "labels were used."
            ),
        },
    }


# --------------------------------------------------------------------------- errors/B7


def test_b7_a_run_that_cannot_start_prints_one_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing key or a model server that is down printed a traceback, the
    key's ending in advice no entry point follows (copying .env.example)."""

    async def no_key(*_a: Any, **_kw: Any) -> AgentResult:
        """What StudioClient.from_env raises with no key."""
        StudioConfig.from_env()
        raise AssertionError("a key was set")

    monkeypatch.delenv("OLMOEARTH_API_KEY", raising=False)
    monkeypatch.setattr(cli, "run_brief", no_key)
    assert cli.main(["How many projects do I have?"]) == 2
    (line,) = capsys.readouterr().err.splitlines()
    assert line.startswith("olmoearth-agent: OLMOEARTH_API_KEY is not set.")
    assert "export OLMOEARTH_API_KEY" in line and ".env.example" not in line
    down = openai.APIConnectionError(
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    )
    assert cli.error_line(down).endswith(" " + cli.LLM_DOWN_HINT)

    async def server_down(*_a: Any, **_kw: Any) -> AgentResult:
        """The local model server is not running."""
        raise down

    monkeypatch.setattr(cli, "run_brief", server_down)
    assert cli.main(["How many projects do I have?"]) == 2
    assert capsys.readouterr().err.splitlines() == [cli.error_line(down)]


# --------------------------------------------------------------------------- errors/B10


def test_b10_the_difference_scan_keeps_cells_studio_did_not_answer_apart() -> None:
    """The scan stacked 4 retries on the server's 3, and counted a cell that
    still failed as no-data, so a rate limit or a refused key read as 'no
    overlapping valid pixels'."""
    viz = (ROOT / "webui" / "js" / "viz.js").read_text(encoding="utf-8")
    assert "const PIXEL_FAILED = 'failed';" in viz
    # An HTTP 5xx from the bridge is tried once more only.
    assert "if (t === tries - 1 || (bridged && t >= 1)) return PIXEL_FAILED;" in viz
    assert "const skipped = total - m - failed - refused;" in viz
    assert (
        "'Studio did not answer for ' + failed + ' of ' + total + ' cells; "
        "try again later.'" in viz
    )
    assert "' cells could not be read from Studio'" in viz
    assert "n_failed: failed," in viz
    # A failed cell is never painted or counted as a value.
    assert "if (va === PIXEL_FAILED || vb === PIXEL_FAILED) {\n      failed++;" in viz
    # A refused key (the bridge passes on its 401 or 403) is no outage: no
    # retry, and no "try again later" (round r5-n review).
    assert "const PIXEL_REFUSED = 'refused';" in viz
    refused = "if (/HTTP 40[13]/.test(String(e && e.message))) return PIXEL_REFUSED;"
    assert refused in viz and viz.index(refused) < viz.index("const bridged = ")
    assert (
        "if (va === PIXEL_REFUSED || vb === PIXEL_REFUSED) {\n      refused++;" in viz
    )
    assert (
        "'Studio refused the key for ' + refused + ' of ' + total + ' cells; "
        "check the Studio key.'" in viz
    )
    assert "' · Studio refused the key for ' + refused + ' cells'" in viz
    assert "n_refused: refused," in viz
    assert viz.count("// WORDING: r5n") >= 2


@pytest.mark.parametrize("status", [401, 403])
def test_b10_a_refused_key_on_the_pixel_value_keeps_its_status(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """The bridge turned a refused key into a 502, which the scan read as
    Studio not answering: 'try again later', after four tries."""
    serve = _bridge(monkeypatch)
    from fastapi.testclient import TestClient

    class _Refusing:
        """A pooled Studio client whose key Studio refuses."""

        async def pixel_value(self, *_a: Any, **_kw: Any) -> Any:
            """What StudioClient raises for a 401 or 403."""
            raise StudioAccessError("GET", "/prediction-results/r1/pixel-value", status)

    async def pooled(_key: str) -> _Refusing:
        """The stand-in, for any key."""
        return _Refusing()

    monkeypatch.setattr(serve, "_pooled_studio", pooled)
    serve._PV_CACHE.clear()
    query = {"result_id": "r1", "lon": "0.0", "lat": "0.0"}
    with TestClient(serve.app, base_url="http://127.0.0.1") as client:
        got = client.get(
            "/api/pixel-value", params=query, headers={"X-Olmoearth-Key": "k"}
        )
    assert got.status_code == status
    assert got.json()["detail"] == f"Studio API returned HTTP {status}"
    assert not serve._PV_CACHE  # nothing cached: the key may be fixed


# --------------------------------------------------------------------------- security/SEC-5


def test_sec5_the_bridge_answers_loopback_names_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page that rebinds its own name to 127.0.0.1 could run the agent on
    the user's model and tools with any key and read the replies."""
    serve = _bridge(monkeypatch)
    from fastapi.testclient import TestClient

    async def local_down(_endpoint: str, _api_key: str | None = None) -> bool:
        """No probe leaves the test."""
        return False

    monkeypatch.setattr(serve, "_local_llm_up", local_down)
    with TestClient(serve.app, base_url="http://evil.example:8088") as client:
        health = client.get("/api/health")
        run = client.post(
            "/api/run",
            json={"brief": "list my projects"},
            headers={"X-Olmoearth-Key": "x"},
        )
    assert health.status_code == run.status_code == 403
    assert health.text == serve.FOREIGN_HOST and "local" not in _CapAgent.seen
    assert "127.0.0.1, localhost or [::1]" in health.text
    for base in ("http://127.0.0.1:8088", "http://localhost:8088"):
        with TestClient(serve.app, base_url=base) as client:
            assert client.get("/api/health").status_code == 200, base
    # IPv6 loopback: starlette's TestClient cannot split an IPv6 netloc, so
    # the Host header is sent by hand.
    with TestClient(serve.app, base_url="http://127.0.0.1:8088") as client:
        ipv6 = client.get("/api/health", headers={"host": "[::1]:8088"})
        assert ipv6.status_code == 200
    assert serve._host_name("[::1]:8088") == "[::1]"
    # A LAN name, added by the operator.
    monkeypatch.setenv(serve.ALLOWED_HOSTS_ENV, " lan-box , other ")
    with TestClient(serve.app, base_url="http://LAN-BOX:8088") as client:
        assert client.get("/api/health").status_code == 200
    # Names written with a port, and a bare IPv6 address, match as a Host
    # header carries them (round r5-n review: they were refused for good).
    monkeypatch.setenv(serve.ALLOWED_HOSTS_ENV, "lan-box:9000, fe80::1")
    with TestClient(serve.app, base_url="http://127.0.0.1:8088") as client:
        for host in ("lan-box:8088", "[fe80::1]:8088", "[fe80::1]"):
            got = client.get("/api/health", headers={"host": host})
            assert got.status_code == 200, host
        foreign = client.get("/api/health", headers={"host": "evil.example"})
        assert foreign.status_code == 403
    assert serve._entry_name("fe80::1") == "[fe80::1]"
    assert serve._entry_name(" LAN-box:9000 ") == "lan-box"


# --------------------------------------------------------------------------- packaging/PKG-2


def test_pkg2_the_checkout_pins_python_3_12() -> None:
    """Nothing pinned the interpreter: where uv picked 3.14, `make setup`
    succeeded and installed no olmoearth-inferencex, numpy or rasterio."""
    assert (ROOT / ".python-version").read_text(encoding="utf-8") == "3.12\n"
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".python-version" not in [line.strip() for line in ignored]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "python_version < '3.14'" in pyproject  # the extra's own limit
    assert 'requires-python = ">=3.11"' in pyproject  # the core stays open


# --------------------------------------------------------------------------- packaging/PKG-4


def test_pkg4_the_bridge_says_where_the_web_ui_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """From a wheel install the web UI is not there, and the bridge answered
    every page with a bare 404."""
    serve = _bridge(monkeypatch)
    ran: list[dict[str, Any]] = []

    def run(*_a: Any, **kw: Any) -> None:
        """Keep how uvicorn was asked to serve, serving nothing."""
        ran.append(kw)

    fake = types.ModuleType("uvicorn")
    fake.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    missing = tmp_path / "webui"
    monkeypatch.setattr(serve, "_WEBUI", missing)
    assert serve.main([]) == 2 and ran == []
    assert capsys.readouterr().err == serve.WEBUI_MISSING.format(path=missing) + "\n"
    missing.mkdir()
    assert serve.main(["--port", "8099"]) == 0
    assert ran == [{"host": "127.0.0.1", "port": 8099, "log_level": "info"}]


# --------------------------------------------------------------------------- numbers/F1

#: An estimate's result as olmoearth_estimate_map_error returns it (trimmed).
_ESTIMATE = {
    "available": True,
    "design_path": "/w/part.json",
    "design": "random",
    "seed": 4,
    "n_labelled": 80,
    "n_population": 4000,
    "nominal_coverage": 0.95,
    "valid_interval": {"estimate": 0.15, "low": 0.08, "high": 0.25, "method": "m"},
    "intervals": [{"interval": "valid", "low": 0.08, "high": 0.25, "method": "m"}],
}


#: The estimate's call's arguments.
_GRADED = {"design_path": "/w/part.json", "labels_path": "/w/part.csv"}


def _card(withdrawn: bool) -> dict[str, Any]:
    """An estimate's card as server._estimate_result gives it (trimmed)."""
    gone = {"title": "This result is out of date"} if withdrawn else None
    return {
        "view": {"withdrawn": gone},
        "say_first": ["Said first."],
        "notes": [],
        "note": "Note.",
    }


async def _estimate_in_the_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    withdrawn: bool,
    fails: bool = False,
) -> tuple[Any, list[Any], list[str]]:
    """The estimate through the app's own call, its tool and card stood in
    for: the app, what the run's record held while the card was built, and
    the text blocks Claude reads."""
    from mcp import types

    from olmoearth_agent.mcp_app import auth, server

    _app_roots(tmp_path, monkeypatch)
    app = server.AppServer(None, keys=auth.KeyStore(), base_url=STUDIO)

    async def estimate(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        """The tool's result, or its failure."""
        if fails:
            raise ValueError("the design file cannot be read")
        return dict(_ESTIMATE)

    spec = ToolSpec(ESTIMATE_TOOL, "e", _EMPTY)
    app.registry.register(RegisteredTool(spec, estimate))
    mid: list[Any] = []

    def mark(*_a: Any) -> None:
        """server._grade_mark, while the card is built: what a report reads."""
        verification = list(app.state.provenance.verification)
        mid.append((verification, accuracy_section(verification)[1]))

    monkeypatch.setattr(server, "_grade_mark", mark)
    monkeypatch.setattr(
        server, "_estimate_result_safe", lambda *_a, **_kw: _card(withdrawn)
    )
    result = await app.call_agent_tool(ESTIMATE_TOOL, dict(_GRADED), views=True)
    await app.aclose()
    texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
    return app, mid, texts


@pytest.mark.asyncio
async def test_numbers_f1_the_report_states_no_rate_from_a_withdrawn_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app withdrew an estimate on its card and to Claude, but the report
    rebuilds its accuracy section from the run's record, which never learned
    it: the withdrawn rate was stated as a valid one. The record was then
    marked only after the card's worker threads had run, so a report made in
    between still stated it (round r5-n review)."""
    state = ThreadState()
    envelope = {"ok": True, "result": dict(_ESTIMATE)}
    state.provenance.record_tool_call(ESTIMATE_TOOL, dict(_GRADED), envelope)
    lines, accuracy = accuracy_section(state.provenance.verification)
    assert accuracy["measured"] is True  # as recorded, the rate is stated
    assert any(line.startswith("- Error rate: 15.0%") for line in lines)
    app, mid, texts = await _estimate_in_the_app(tmp_path, monkeypatch, withdrawn=True)
    # While the card was built the record held nothing, so no rate.
    ((held, during),) = mid
    assert held == [] and during["measured"] is False and not during["statements"]
    (call,) = app.state.provenance.verification
    assert call.ok is True and call.result["withdrawn"] is True
    assert "withdrawn" not in texts[-1]  # the model's copy, as the tool gave it
    lines, accuracy = accuracy_section(app.state.provenance.verification)
    assert accuracy["measured"] is False
    assert not any("Error rate:" in line or "15.0%" in line for line in lines)
    assert WITHDRAWN_NOT_MEASURED in lines and WITHDRAWN in lines
    assert NOT_MEASURED not in lines  # it said no call graded labels
    (statement,) = accuracy["statements"]
    assert statement["kind"] == "withdrawn" and "estimate" not in statement


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_numbers_f1_any_other_estimate_is_recorded_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """A result that stands, and an estimate that failed, are recorded as
    they were, once the call ends."""
    app, mid, _texts = await _estimate_in_the_app(
        tmp_path, monkeypatch, withdrawn=False, fails=fails
    )
    (call,) = app.state.provenance.verification
    if fails:
        assert call.ok is False and call.result is None and mid == []
    else:
        assert call.ok is True and "withdrawn" not in call.result
        _lines, accuracy = accuracy_section(app.state.provenance.verification)
        assert accuracy["measured"] is True


def _whole(low: float, high: float) -> dict[str, Any]:
    """A whole_map_estimate fact of 10% wrong, with its interval."""
    return {
        "id": "whole_map_estimate",
        "estimate": 0.1,
        "low": low,
        "high": high,
        "sentence": f"10.0% wrong (range {low:.1%} to {high:.1%})",
    }


def test_numbers_f1_a_zone_on_a_withdrawn_sheet_states_nothing() -> None:
    """A zone graded on the sheet of a withdrawn estimate stated the withdrawn
    rate as its 'Error rate from the same labels', under its verdict, and
    counted as measured: its whole-map figure comes from the same labels
    (round r5-n review). A zone on another design file is stated as before."""
    graded = {"design_path": "/w/z.json", "labels_path": "/w/s.csv"}
    estimate = {
        **_ESTIMATE,
        "design_path": "/w/z.json",
        "withdrawn": True,
        "facts": [_whole(0.06, 0.16)],
    }
    zone = {
        "available": True,
        "design_path": "/w/z.json",
        "design": "random",
        "seed": 4,
        "n_labelled": 80,
        "certified": True,
        "alpha": 0.1,
        "delta": 0.1,
        "rule": "prefix",
        "n_zone": 900,
        "coverage": 0.5,
        "verdict": "Certified: the zone's error rate is at most 0.1.",
        "facts": [_whole(0.07, 0.14)],
    }
    certificate = {
        "available": True,
        "design_path": "/w/z.json",
        "design": "random",
        "seed": 4,
        "n_labelled": 80,
        "certificate_written": True,
        "certificate_path": "/w/c.json",
        "statement": "Certified: at most 0.1 of the zone is wrong.",
        "facts": [_whole(0.07, 0.14)],
    }
    tools = (ESTIMATE_TOOL, CERTIFY_TOOL, WRITE_CERTIFICATE_TOOL)
    calls = [
        {"tool": tool, "arguments": graded, "ok": True, "result": result}
        for tool, result in zip(tools, (estimate, zone, certificate), strict=True)
    ]
    lines, accuracy = accuracy_section(calls)
    assert not any(
        line.startswith("- Error rate from the same labels") for line in lines
    )
    assert ZONE_MEANING not in lines
    assert not any("Certified" in line or "10.0%" in line for line in lines)
    assert accuracy["measured"] is False and lines[2] == WITHDRAWN_NOT_MEASURED
    assert lines.count(WITHDRAWN) == 1 and lines.count(WITHDRAWN_SAME_SHEET) == 2
    kinds = [(s["kind"], s["tool"]) for s in accuracy["statements"]]
    assert kinds == [("withdrawn", tool) for tool in tools]
    # A zone on another design file is no grading of that sheet.
    other = {"design_path": "/w/other.json", "labels_path": "/w/o.csv"}
    elsewhere = {**zone, "design_path": "/w/other.json"}
    calls.append(
        {"tool": CERTIFY_TOOL, "arguments": other, "ok": True, "result": elsewhere}
    )
    lines, accuracy = accuracy_section(calls)
    assert accuracy["measured"] is True and ZONE_MEANING in lines
