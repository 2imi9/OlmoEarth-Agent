# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The capability card (harness/capabilities.py) and its place in the system prompt.

exp86 round 9: most material errors sat where the tools said nothing, and 7 of
16 were offers of actions no tool can do or whose preconditions did not hold.
The card lists what each tool does, needs and cannot do, from the tools' own
declarations, and what no tool can do, read off the registry; the soul tells
the model to propose only what it lists.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.harness.capabilities import (
    CARD_TITLE,
    NO_TOOL_CAN,
    READ_ONLY_STUDIO,
    beyond_tools,
    capability_card,
)
from olmoearth_agent.harness.soul import load_soul
from olmoearth_agent.llm.types import ChatResponse, Message, ToolCall, ToolSpec
from olmoearth_agent.skills import READ_ONLY_ENV, build_default_registry
from olmoearth_agent.tools.registry import (
    Capability,
    RegisteredTool,
    ToolContext,
    ToolRegistry,
)

#: The card's budget in the system prompt, in tokens of four characters
#: (1,500 until the direct-run step added the certificate tools' lines, 1,600
#: until the label-inventory and Studio-files steps added their lines:
#: test_the_card_fits_its_budget).
CARD_TOKEN_BUDGET = 1700

_SCHEMA = {"type": "object", "properties": {}, "required": []}


async def _echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    return args


def _tool(
    name: str, capability: Capability | None, description: str = "A tool. More."
) -> RegisteredTool:
    return RegisteredTool(
        spec=ToolSpec(name=name, description=description, parameters=_SCHEMA),
        handler=_echo,
        capability=capability,
    )


def _tokens(text: str) -> float:
    return len(text) / 4


# --------------------------------------------------------------------------- the card


def test_the_card_lists_every_core_tool_with_its_required_arguments() -> None:
    registry = build_default_registry()
    card = capability_card(registry)
    assert card.startswith(CARD_TITLE)
    for name in registry.names():
        group = registry.group_of(name)
        line = f"\n- {name}"
        if group is None:
            assert line in card, name
        else:
            assert line not in card, name
            assert group in card, group
    assert "- olmoearth_certify_zone(design_path, alpha): " in card
    assert "- olmoearth_load_context: " in card


def test_needs_and_cannot_are_on_the_tool_line() -> None:
    """The four offers of exp86 round 9 each meet their tool's precondition."""
    card = capability_card(build_default_registry())
    lines = {line.split("(")[0].split(":")[0]: line for line in card.splitlines()}
    review = lines["- olmoearth_review_set_from_result"]
    assert "Needs: threshold, for a regression band not in [0, 1]." in review
    # A class map with top-k scores is ranked; one without them is not.
    assert "Cannot: rank a class map with no top-k scores" in review
    compare = lines["- olmoearth_compare_review"]
    assert "Cannot: say which map is right, even with labels_date" in compare
    estimate = lines["- olmoearth_estimate_map_error"]
    assert "Cannot: use a review set, or windows chosen by margin" in estimate
    certify = lines["- olmoearth_certify_zone"]
    assert "Needs: a design='random' plan and its labels." in certify


def test_a_loaded_group_is_listed_in_full() -> None:
    """A loaded group's tools are listed like the core ones; another is named.

    On a registry of its own: no tool of the agent is deferred.
    """
    registry = ToolRegistry()
    registry.register(_tool("core", Capability(does="a core thing")))
    registry.register(_tool("first", Capability(does="a first thing")), group="a")
    registry.register(_tool("second", Capability(does="a second thing")), group="b")
    card = capability_card(registry, loaded_groups=["a"])
    assert "\n- core: a core thing." in card
    assert "\n- first: a first thing." in card
    assert "\n- second" not in card
    tail = card.split("More tools come with these skills", 1)[1].split("\n\n")[0]
    assert tail.endswith(": b.")


def test_what_no_tool_can_do_is_read_off_the_registry() -> None:
    """exp86 round 9: "I'll set up a direct model run"; "no ground-truth labels exist"."""
    registry = build_default_registry()
    card = capability_card(registry)
    head, beyond = card.split("No tool of this agent can:", 1)
    for text in NO_TOOL_CAN.values():
        assert f"- {text}." in beyond
    # A tool that covers one of them takes it off the list.
    registry.register(
        _tool(
            "fetch_labels",
            Capability(does="fetch labels", covers=frozenset({"labels"})),
        )
    )
    assert NO_TOOL_CAN["labels"] not in capability_card(registry)
    assert NO_TOOL_CAN["labels"] not in beyond_tools(registry)
    assert NO_TOOL_CAN["run_model"] in capability_card(registry)


def test_the_card_counts_studio_annotations_but_takes_no_sample_or_training() -> None:
    """The label inventory counts the Studio annotations a model's training
    selector selects now, and their splits; what no tool can do still says they
    are no probability sample and that no tool trains a model (the
    label-inventory step of 31 October 2026 and its review: the selector is
    read now, so the card does not say what the model was trained on)."""
    card = capability_card(build_default_registry())
    head, beyond = card.split("No tool of this agent can:", 1)
    line = (
        "\n- olmoearth_studio_label_inventory: count the Studio annotations a "
        "model's training selector selects now, or a project's, by status, split "
        "and class."
    )
    assert line in head
    assert NO_TOOL_CAN["labels"].endswith(
        "use Studio annotations as a probability sample"
    )
    assert NO_TOOL_CAN["training"].startswith("train a model")
    for key in ("labels", "training"):
        assert f"- {NO_TOOL_CAN[key]}." in beyond
    assert "say whether any exist" not in card


def test_a_read_only_key_puts_its_line_on_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With OLMOEARTH_READ_ONLY on, the card says why no project or prediction."""
    default = capability_card(build_default_registry())
    assert READ_ONLY_STUDIO not in default
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    registry = build_default_registry()
    card = capability_card(registry)
    head, beyond = card.split("No tool of this agent can:", 1)
    assert f"- {READ_ONLY_STUDIO}." in beyond
    assert "read-only" in READ_ONLY_STUDIO
    for gone in ("olmoearth_create_project", "olmoearth_submit_prediction"):
        assert f"- {gone}" not in head, gone
    # What no tool can do is still listed in full beside it.
    for text in NO_TOOL_CAN.values():
        assert f"- {text}." in beyond
    assert beyond_tools(registry)[-1] == READ_ONLY_STUDIO


def test_covers_names_only_what_no_tool_can() -> None:
    registry = build_default_registry()
    for name in registry.names():
        capability = registry.capability_of(name)
        assert capability is not None
        assert set(capability.covers) <= set(NO_TOOL_CAN), name


def test_the_card_fits_its_budget() -> None:
    """About 1,390 tokens (chars / 4) for the default registry's core tools.

    Measured before the trim of 27 October 2026, which took three core tools
    and six deferred groups off the default card; opted in, it still fits.
    The direct-run step adds, counted by hand from the declarations, the
    lines of olmoearth_write_certificate and olmoearth_verify_certificate
    (357 characters with their line breaks) and "or from-olmoearth's files" on
    olmoearth_scores_from_file's line (27): 384 characters, 96 tokens, on
    the default card and the opted-in one alike. The opted-in card, which
    holds the default one, three more core lines and the deferred groups'
    line, was estimated near 1,460 tokens before it (an estimate from the
    declarations' text, not a measurement), so the budget moves from 1,500 to
    1,600. The fixes of 28 October add 18 characters to
    olmoearth_estimate_map_error's line (", the valid interval first, then
    its design's" for " with its design's interval"): 5 tokens, on both
    cards, within the budget. The fixes of agent trial 12 leave
    olmoearth_plan_label_sample's line at 425 characters: seed_from_map's
    need (57 with its separator) in place of "; save the design" (17),
    "(no confidence)" (16), "for 'disagreement', " (4 net), "a Studio result
    at grid 16 plans them all" shortened (17) and ", or" (3). The top-k step
    adds 25 characters, 6 tokens, on both cards: " or its top-k margin" (20)
    on olmoearth_review_set_from_result's line, whose cannot is 6 shorter
    ("a class map with no top-k scores (no margin)"), "a class map with no
    top-k scores" on olmoearth_plan_label_sample's need (14) and on
    olmoearth_certify_zone's cannot (3 shorter). The label-inventory step of
    31 October adds, counted the same way, olmoearth_studio_label_inventory's
    line (133 characters with its line break), and rewords what no tool can
    do: the labels line 14 characters longer ("use Studio annotations as a
    probability sample" for "look up ... or say whether any exist") and the
    training line 5 shorter ("train a model" for "what its training data
    held"): 142 characters, 36 tokens, on both cards. The review of that step
    adds 17 characters to the tool's line ("a model's training selector
    selects now" for "a model was trained on"): 159 characters, 40 tokens.
    The Studio-files step adds olmoearth_scores_from_studio's line, counted
    by hand from its declaration: 192 characters with its line break (what it
    does, 100, verb first as the others, and its cannot, 37), 48 tokens on
    both cards. The two steps together: 88 tokens. The opted-in card,
    estimated near 1,567 tokens before them (1,460 + 96 + 5 + 6, estimates
    from the declarations' text, not measurements), would then be near
    1,655, so the budget moves by the same 88, from 1,600 to 1,688: it is
    1,700. The final code step of 31 October rewords
    olmoearth_search_predictions' cannot, counted by hand: "say a model's
    metrics, or count its Studio annotations (olmoearth_studio_label_inventory
    does)" (94 characters) for "say what a model's training data held, or
    its metrics" (53): 41 characters, 10 tokens, on both cards. The opted-in
    card, estimated near 1,655, would be near 1,665, under the budget of
    1,700, which stays. The trim of 31 October takes olmoearth_litsearch's
    core line and four deferred groups' names off the opted-in card only; the
    default card is unchanged. Its next step removes the last opt-in tools
    with their code, so the opted-in card is the default card, and the step
    after it the opt-in setting, so there is one card.
    """
    card = capability_card(build_default_registry())
    assert _tokens(card) < CARD_TOKEN_BUDGET, _tokens(card)


def test_the_card_names_no_removed_tool_or_group() -> None:
    """A removed tool is off the card, and so is its group."""
    card = capability_card(build_default_registry())
    for gone in (
        "olmoearth_change_detect",
        "olmoearth_grade_review_rule",
        "olmoearth-rslearn",
    ):
        assert gone not in card, gone


def test_an_undeclared_tool_is_listed_from_its_description() -> None:
    registry = ToolRegistry()
    registry.register(_tool("plain", None, "Does a plain thing. Then more."))
    registry.register(_tool("known", Capability(does="a known thing", needs=("x",))))
    card = capability_card(registry)
    assert "- plain: Does a plain thing." in card
    assert "- known: a known thing. Needs: x." in card


def test_an_empty_registry_has_no_card() -> None:
    assert capability_card(ToolRegistry()) == ""


# --------------------------------------------------------------------------- the system prompt


class _Recording:
    """Answers from a script and records every call's messages."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], *, tools: Any = None, **_kw: Any
    ) -> ChatResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


def test_the_system_prompt_is_the_soul_then_the_card_then_the_clauses() -> None:
    registry = build_default_registry()
    agent = LeadAgent(
        None,  # type: ignore[arg-type]
        registry,
        studio=None,  # type: ignore[arg-type]
        skill_index="- some-skill: does things",
        local=True,
    )
    prompt = agent.system_prompt
    soul = load_soul()
    card = capability_card(registry)
    assert prompt.startswith(soul + "\n\n" + card)
    assert prompt.index(card) < prompt.index("Available instruction skills")


@pytest.mark.asyncio
async def test_a_group_loaded_mid_run_joins_the_card_from_the_next_turn() -> None:
    registry = ToolRegistry()
    registry.register(_tool("core_tool", Capability(does="a core thing")))
    registry.register(
        _tool("deep_tool", Capability(does="a deferred thing")), group="deep-skill"
    )
    llm = _Recording(
        [
            ChatResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="deep_tool", arguments={})],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="done", tool_calls=[], finish_reason="stop"),
        ]
    )
    agent = LeadAgent(
        llm,  # type: ignore[arg-type]
        registry,
        studio=None,  # type: ignore[arg-type]
        check_numbers=False,
        check_answer=False,
    )
    await agent.run("use the deep tool")
    first, second = (call[0].content for call in llm.calls)
    assert "- core_tool: a core thing." in first
    assert "- deep_tool" not in first and "deep-skill" in first
    assert "- deep_tool: a deferred thing." in second
    assert agent.system_prompt == second
