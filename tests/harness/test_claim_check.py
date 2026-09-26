# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The claim check: its prompt, its reply, and its place in the answer checks.

exp86 round 9: most material errors were where the tools said nothing, and
the rules caught about half of new wordings. A scripted model plays both
the agent and the verifier here: a comparison tool returns exp86 B7's facts,
the draft offers a step no tool takes ("rerun the comparison with
labels_date, which grades both maps") and claims what no tool checked, and
the verifier's scripted reply flags them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from olmoearth_agent.harness.agent import (
    CHECK_ANSWER_ENV,
    CHECK_CLAIMS_ENV,
    CHECK_NUMBERS_ENV,
    LeadAgent,
)
from olmoearth_agent.harness.answer_checks import AnswerChecksMiddleware
from olmoearth_agent.harness.checks import (
    CLAIMS,
    MARK,
    RunEvidence,
    ToolRecord,
    revision_prompt,
)
from olmoearth_agent.harness.claim_check import (
    CLAIM_CHECK_VERSION,
    CONTRADICTS,
    MAX_RECORD_CHARS,
    OFFER,
    UNSUPPORTED,
    VERIFIER_PROMPT,
    MalformedReply,
    check_claims,
    claim_check_messages,
    match_sentences,
    parse_reply,
)
from olmoearth_agent.harness.spill import SPILL_BYTES_ENV, spill_result_for_llm
from olmoearth_agent.llm.presets import (
    CLAIM_CHECK_MAX_TOKENS,
    CLAIM_CHECK_MODE,
    DEFAULT_AGENT_MODE,
    REVISION_MODE,
)
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.security.paths import OUTPUT_ROOT_ENV
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext, ToolRegistry

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "required": []}
_COMPARE = {
    "n_differing": 1570,
    "classes": {"0": "not water", "1": "water"},
    "facts": [
        {
            "id": "dominant_change",
            "from_class": 0,
            "to_class": 1,
            "n": 1247,
            "share": 0.794268,
            "sentence": "1,247 of the 1,570 differing windows change 0 -> 1.",
        }
    ],
}
_COMPARE_DOC = (
    "Compare two maps window by window; takes date_a, date_b and labels_date."
)
_FIRST = "The maps differ on 1,570 windows, most of them 0 -> 1."
_NO_LABELS = "No ground-truth labels exist for this area."
_OFFER = "I can rerun the comparison with labels_date to grade both maps."
_DRAFT = f"{_FIRST} {_NO_LABELS} {_OFFER}"
_FIXED = (
    f"{_FIRST} To say which map is right, label a sample drawn with "
    "olmoearth_plan_label_sample."
)


def _answer(content: str | None) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=[], finish_reason="stop")


def _flags(*items: tuple[str, str, str]) -> ChatResponse:
    """The verifier's reply flagging each ``(sentence, kind, why)``."""
    return _answer(
        json.dumps([{"sentence": s, "kind": k, "why": w} for s, k, w in items])
    )


def _compare_call() -> ChatResponse:
    call = ToolCall(id="c1", name="compare", arguments={"date_a": "2018-10-01"})
    return ChatResponse(content=None, tool_calls=[call], finish_reason="tool_calls")


class _Scripted:
    """Returns scripted responses in order; records every call's arguments."""

    def __init__(self, responses: Iterable[ChatResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **kw: Any
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": tools, **kw})
        step = self.responses.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def aclose(self) -> None:
        return None

    def verifier_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("mode") == CLAIM_CHECK_MODE]


def _registry() -> ToolRegistry:
    async def compare(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return _COMPARE

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(ToolSpec("compare", _COMPARE_DOC, _EMPTY_SCHEMA), compare)
    )
    return registry


def _agent(llm: Any, **kw: Any) -> LeadAgent:
    return LeadAgent(llm, _registry(), studio=None, **kw)  # type: ignore[arg-type]


async def _events(agent: LeadAgent, brief: str = "Which map is right?") -> Any:
    return [e async for e in agent.run_stream(brief)]


def _check_events(events: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    return [
        (e["check"], e["action"], len(e["violations"]))
        for e in events
        if e["type"] == "check"
    ]


@pytest.fixture(autouse=True)
def _claims_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim check at its default (tests/conftest.py turns it off)."""
    for env in (CHECK_CLAIMS_ENV, CHECK_NUMBERS_ENV, CHECK_ANSWER_ENV):
        monkeypatch.delenv(env, raising=False)


# --- the loop ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_flagged_sentence_goes_into_the_rewrite() -> None:
    llm = _Scripted(
        [
            _compare_call(),
            _answer(_DRAFT),
            _flags(
                (_NO_LABELS, "unsupported", "No tool checked for labels."),
                (
                    _OFFER,
                    "offer",
                    "labels_date is a date; the comparison takes no labels.",
                ),
            ),
            _answer(_FIXED),
            _answer("[]"),
        ]
    )
    events = await _events(_agent(llm))
    assert _check_events(events) == [("claims", "revise", 2), ("claims", "passed", 0)]
    revise = next(e for e in events if e["type"] == "check")
    assert [v["text"] for v in revise["violations"]] == [_NO_LABELS, _OFFER]
    assert [v["kind"] for v in revise["violations"]] == [UNSUPPORTED, OFFER]
    assert revise["draft"] == _DRAFT
    # The draft, the verifier, the rewrite (no tools, the rewrite's sampling),
    # the verifier again: two calls more than the rule checks cost.
    modes = [c.get("mode", DEFAULT_AGENT_MODE) for c in llm.calls]
    assert modes == [
        DEFAULT_AGENT_MODE,
        DEFAULT_AGENT_MODE,
        CLAIM_CHECK_MODE,
        REVISION_MODE,
        CLAIM_CHECK_MODE,
    ]
    rewrite = llm.calls[3]
    assert rewrite["tools"] is None
    note = rewrite["messages"][-1].content or ""
    assert note == revision_prompt({CLAIMS: revise["violations"]})
    assert _NO_LABELS in note and _OFFER in note
    assert "offer instead a step a tool can take" in note
    final = events[-1]
    assert final["content"] == _FIXED
    assert final["revised"] is True and final["marked"] == []


@pytest.mark.asyncio
async def test_a_sentence_still_flagged_after_the_rewrite_is_marked() -> None:
    still = f"{_FIRST} {_NO_LABELS}"
    llm = _Scripted(
        [
            _compare_call(),
            _answer(_DRAFT),
            _flags((_NO_LABELS, "unsupported", "no tool checked")),
            _answer(still),
            _flags((_NO_LABELS, "unsupported", "no tool checked")),
        ]
    )
    events = await _events(_agent(llm))
    assert _check_events(events) == [("claims", "revise", 1), ("claims", "marked", 1)]
    final = events[-1]
    assert final["content"] == f"{still} {MARK.format(check='claims')}"
    assert final["marked"] == ["claims"] and final["revised"] is True
    assert len(llm.verifier_calls()) == 2 and not llm.responses  # never a third


@pytest.mark.asyncio
async def test_a_malformed_reply_fails_open() -> None:
    reply = "The third sentence is not supported by any tool."
    llm = _Scripted([_compare_call(), _answer(_DRAFT), _answer(reply)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [("claims", "failed_open", 0)]
    event = next(e for e in events if e["type"] == "check")
    assert event["reply"] == reply and event["error"].startswith("malformed reply")
    assert events[-1]["content"] == _DRAFT and events[-1]["revised"] is False
    assert len(llm.calls) == 3  # no rewrite


@pytest.mark.asyncio
async def test_a_failed_call_fails_open_and_names_only_the_error_type() -> None:
    address = "http://10.0.0.7:8000/v1 refused"
    llm = _Scripted([_compare_call(), _answer(_DRAFT), ConnectionError(address)])
    events = await _events(_agent(llm))
    event = next(e for e in events if e["type"] == "check")
    assert event["action"] == "failed_open" and event["reply"] is None
    assert event["error"] == "the call failed (ConnectionError)"
    assert address not in json.dumps(events)
    assert events[-1]["content"] == _DRAFT


@pytest.mark.asyncio
async def test_the_claims_join_the_rules_in_one_rewrite() -> None:
    backwards = "Most differing windows flip class 1 -> 0."
    draft = f"{backwards} {_OFFER}"
    llm = _Scripted(
        [
            _compare_call(),
            _answer(draft),
            _flags((_OFFER, "offer", "the comparison takes no labels")),
            _answer(_FIXED),
            _answer("[]"),
        ]
    )
    events = await _events(_agent(llm))
    assert _check_events(events) == [
        ("direction", "revise", 1),
        ("claims", "revise", 1),
        ("claims", "passed", 0),
    ]
    rewrites = [c for c in llm.calls if c.get("mode") == REVISION_MODE]
    assert len(rewrites) == 1
    note = rewrites[0]["messages"][-1].content or ""
    assert backwards in note and _OFFER in note


@pytest.mark.asyncio
async def test_a_clean_answer_costs_one_call_and_records_the_reply() -> None:
    llm = _Scripted([_compare_call(), _answer(_FIRST), _answer("[]")])
    events = await _events(_agent(llm))
    assert _check_events(events) == [("claims", "passed", 0)]
    assert events[-1]["content"] == _FIRST and len(llm.calls) == 3


@pytest.mark.asyncio
async def test_a_quote_found_in_no_sentence_is_reported_not_marked() -> None:
    llm = _Scripted(
        [
            _compare_call(),
            _answer(_FIRST),
            _flags(("Both maps are wrong in the north.", "unsupported", "x")),
        ]
    )
    events = await _events(_agent(llm))
    event = next(e for e in events if e["type"] == "check")
    assert event["action"] == "passed"
    assert event["unmatched"] == ["Both maps are wrong in the north."]
    assert events[-1]["content"] == _FIRST and len(llm.calls) == 3


@pytest.mark.asyncio
async def test_a_required_statement_alone_is_still_appended_after_the_claims_pass() -> (
    None
):
    limit = "No recorded experiment grades which of these two maps is right."

    async def compare(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"n_differing": 1570, "must_state": [limit]}

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(ToolSpec("compare", _COMPARE_DOC, _EMPTY_SCHEMA), compare)
    )
    answer = "The two maps differ on 1,570 windows."
    llm = _Scripted([_compare_call(), _answer(answer), _answer("[]")])
    agent = LeadAgent(llm, registry, studio=None)  # type: ignore[arg-type]
    events = await _events(agent)
    assert _check_events(events) == [
        ("must_state", "appended", 1),
        ("claims", "passed", 0),
    ]
    assert events[-1]["content"].rstrip().endswith(f"Note from the tool: {limit}")
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_the_claim_check_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _Scripted([_compare_call(), _answer(_DRAFT)])
    events = await _events(_agent(llm, check_claims=False))
    assert _check_events(events) == [] and len(llm.calls) == 2

    monkeypatch.setenv(CHECK_CLAIMS_ENV, "off")
    llm = _Scripted([_compare_call(), _answer(_DRAFT)])
    events = await _events(_agent(llm))
    assert _check_events(events) == [] and len(llm.calls) == 2

    # The other switches leave it on: it is the one check that calls the model.
    monkeypatch.delenv(CHECK_CLAIMS_ENV)
    llm = _Scripted([_compare_call(), _answer(_DRAFT), _answer("[]")])
    events = await _events(_agent(llm, check_numbers=False, check_answer=False))
    assert _check_events(events) == [("claims", "passed", 0)]


def test_the_default_chain_runs_the_claims_with_the_agents_client() -> None:
    llm = _Scripted([])
    agent = _agent(llm)
    checks = agent.default_middleware()[2]
    assert isinstance(checks, AnswerChecksMiddleware)
    assert checks.checks[-1] == CLAIMS and checks.llm is llm
    assert checks.registry is agent.registry
    agent.check_claims = False
    assert CLAIMS not in agent.default_middleware()[2].checks  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="LLM client"):
        AnswerChecksMiddleware([CLAIMS])


@pytest.mark.asyncio
async def test_the_event_shapes() -> None:
    llm = _Scripted(
        [
            _compare_call(),
            _answer(_DRAFT),
            _flags((_OFFER, "offer", "no labels")),
            _answer(_DRAFT),
            _flags((_OFFER, "offer", "no labels")),
        ]
    )
    events = await _events(_agent(llm))
    revise, marked = [e for e in events if e["type"] == "check"]
    audit = {"version", "reply", "unmatched", "reply_complete", "finish_reason"}
    base = {"type", "turn", "check", "violations", "action"}
    assert set(revise) == base | audit | {"draft"}
    assert set(marked) == base | audit
    assert revise["version"] == CLAIM_CHECK_VERSION
    assert json.loads(revise["reply"])[0]["sentence"] == _OFFER
    assert revise["violations"] == [
        {"check": CLAIMS, "text": _OFFER, "detail": "offer: no labels", "kind": OFFER}
    ]
    json.dumps(events)  # JSON-serializable, as run_stream promises
    # The events the exp86 driver reads keep their shape.
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["tool_call", "tool_result"] and kinds[-1] == "final"
    assert {"id", "name", "ok", "result", "turn"} <= set(events[1])


@pytest.mark.asyncio
async def test_run_collects_the_claims_events() -> None:
    llm = _Scripted([_compare_call(), _answer(_FIRST), _answer("[]")])
    result = await _agent(llm).run("Which map is right?")
    assert [(c["check"], c["action"]) for c in result.checks] == [("claims", "passed")]


# --- the verifier's call and record -----------------------------------------


@pytest.mark.asyncio
async def test_the_verifier_is_the_agents_client_with_no_tools_and_the_check_preset() -> (
    None
):
    llm = _Scripted([_compare_call(), _answer(_FIRST), _answer("[]")])
    await _events(_agent(llm), brief="Which map is right, A or B?")
    (call,) = llm.verifier_calls()
    assert call["tools"] is None and call["preserve_thinking"] is False
    assert call["max_tokens"] == CLAIM_CHECK_MAX_TOKENS
    system, user = call["messages"]
    assert system.role == "system" and system.content == VERIFIER_PROMPT
    record = user.content or ""
    assert "Which map is right, A or B?" in record
    assert f"- compare: {_COMPARE_DOC}" in record  # the capability card
    assert '[1] compare {"date_a": "2018-10-01"}' in record
    assert json.dumps({"ok": True, "result": _COMPARE}) in record
    assert f"<answer>\n{_FIRST}\n</answer>" in record


def test_a_spilled_result_is_read_as_the_model_read_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(SPILL_BYTES_ENV, "200")
    big = {
        "ok": True,
        "result": {"n": 7, "rows": ["x" * 50] * 40, "must_state": ["M."]},
    }
    seen, saved = spill_result_for_llm("rank", big)
    assert saved is not None
    run = RunEvidence(tools=[ToolRecord("rank", {}, big, saved)], user_messages=["b"])
    record = claim_check_messages("A.", run, "")[1].content or ""
    assert seen in record and saved in record and "x" * 50 not in record


def test_earlier_answers_are_in_the_record_on_the_web_only() -> None:
    run = RunEvidence(
        user_messages=["compare", "and now?"],
        assistant_messages=["About 1,111 windows sit together."],
    )
    cli = claim_check_messages("A.", run, "")[1].content or ""
    assert "1,111" not in cli and "[2] and now?" in cli
    run.surface = "web"
    web = claim_check_messages("A.", run, "")[1].content or ""
    assert "<earlier_answers" in web and "1,111" in web


def test_the_prompt_is_fixed_and_versioned() -> None:
    """Editing the verifier's instructions must come with a new version."""
    digest = hashlib.sha256(VERIFIER_PROMPT.encode()).hexdigest()
    assert (CLAIM_CHECK_VERSION, digest[:16]) == ("claim-check-1", "d569943aba9c81d7")


@pytest.mark.asyncio
async def test_a_record_too_long_is_not_sent() -> None:
    llm = _Scripted([])
    big = {"ok": True, "result": {"blob": "x" * 10}}
    run = RunEvidence(
        tools=[ToolRecord("t", {}, big)], user_messages=["y" * MAX_RECORD_CHARS]
    )
    result = await check_claims(llm, "A.", run, "")
    assert result.failed and "over" in result.failed and llm.calls == []


# --- the reply --------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "sentences", "complete"),
    [
        ("[]", [], True),
        (
            '```json\n[{"sentence": "A b.", "kind": "offer", "why": "w"}]\n```',
            ["A b."],
            True,
        ),
        (
            'Here they are: [{"sentence": "A b.", "kind": "c"}] That is all.',
            ["A b."],
            True,
        ),
        ('<think>hm</think>[{"quote": "A b.", "reason": "r"}]', ["A b."], True),
        ('<think>Is ["C d."] one?</think>[]', [], True),
        ('I would list ["C d."] but no.</think>\n[]', [], True),
        ('{"violations": [{"sentence": "A b."}]}', ["A b."], True),
        ('{"sentence": "A b.", "kind": "b"}', ["A b."], True),
        ("[{'sentence': 'A b.', 'kind': 'offer'}]", ["A b."], True),
        ('["A b.", "C d."]', ["A b.", "C d."], True),
        ('[{"sentence": "A b.", "why": "w"}, {"sentence": "C', ["A b."], False),
        ('[{"kind": "offer"}, {"sentence": "A b."}]', ["A b."], True),
    ],
)
def test_the_reply_is_read_leniently(
    reply: str, sentences: list[str], complete: bool
) -> None:
    items, whole = parse_reply(reply)
    assert [i["sentence"] for i in items] == sentences and whole is complete


@pytest.mark.parametrize(
    "reply", [None, "", "  ", "No sentence is unsupported.", "Sentence [1] is fine."]
)
def test_a_reply_with_no_list_is_malformed(reply: str | None) -> None:
    with pytest.raises(MalformedReply):
        parse_reply(reply)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("unsupported", UNSUPPORTED),
        ("(a) unsupported claim", UNSUPPORTED),
        ("contradiction", CONTRADICTS),
        ("B", CONTRADICTS),
        ("impossible_offer", OFFER),
        ("c", OFFER),
        ("something else", UNSUPPORTED),
    ],
)
def test_kinds_are_read_as_the_three_the_prompt_names(kind: str, expected: str) -> None:
    items, _ = parse_reply(json.dumps([{"sentence": "A b.", "kind": kind}]))
    assert items[0]["kind"] == expected


# --- the quotes -------------------------------------------------------------

_ANSWER = (
    "**Why these windows:** ranked by the model's own confidence.\n"
    "- The top window sits at (14, 29); it is most likely mislabeled.\n"
    "| 1 | (14, 29) | woodland_forest | 0.108 |\n"
    "Want me to set up a direct model run?\n"
    "**No labels.**"
)


@pytest.mark.parametrize(
    ("quote", "flagged"),
    [
        # Exact, and without the Markdown, bullet and case.
        (
            "Want me to set up a direct model run?",
            ["Want me to set up a direct model run?"],
        ),
        (
            "why these windows: ranked by the model's own confidence",
            ["**Why these windows:** ranked by the model's own confidence."],
        ),
        # Part of a sentence names the sentence.
        ("it is most likely mislabeled", ["it is most likely mislabeled."]),
        # A quote over two sentences names both.
        (
            "The top window sits at (14, 29); it is most likely mislabeled.",
            ["- The top window sits at (14, 29);", "it is most likely mislabeled."],
        ),
        # A table row; a dash or quote the model changed.
        (
            "1 | (14, 29) | woodland_forest | 0.108",
            ["| 1 | (14, 29) | woodland_forest | 0.108 |"],
        ),
        (
            "Why these windows: ranked by the model’s own confidence.",
            ["**Why these windows:** ranked by the model's own confidence."],
        ),
        # Punctuation the model changed, in a short quote.
        ("it is most likely mis-labeled", ["it is most likely mislabeled."]),
        # A near copy of a long sentence, one word changed.
        (
            "Want me to set up the direct model run?",
            ["Want me to set up a direct model run?"],
        ),
        # A short quote equal to a sentence but for its Markdown.
        ("No labels.", ["**No labels.**"]),
        # Not in the answer; too short to place.
        ("The north is all water.", []),
        ("model", []),
    ],
)
def test_quotes_are_matched_to_the_answers_sentences(
    quote: str, flagged: list[str]
) -> None:
    violations, unmatched = match_sentences(
        [{"sentence": quote, "kind": OFFER, "why": ""}], _ANSWER
    )
    assert [v["text"] for v in violations] == flagged
    assert unmatched == ([] if flagged else [quote])


def test_a_sentence_named_twice_is_flagged_once() -> None:
    items = [
        {
            "sentence": "Want me to set up a direct model run?",
            "kind": OFFER,
            "why": "a",
        },
        {"sentence": "set up a direct model run", "kind": UNSUPPORTED, "why": "b"},
    ]
    violations, _ = match_sentences(items, _ANSWER)
    assert [(v["kind"], v["detail"]) for v in violations] == [(OFFER, "offer: a")]
