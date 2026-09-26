# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The capability card (harness/capabilities.py).

exp86 round 9: most material errors sat where the tools said nothing, and 7 of
16 were offers of actions no tool can do or whose preconditions did not hold.
The card lists what each tool does, needs and cannot do, from the tools' own
declarations, and what no tool can do, read off the registry; the soul tells
the model to propose only what it lists.
"""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from olmoearth_agent.harness.capabilities import (
    CARD_TITLE,
    NO_TOOL_CAN,
    beyond_tools,
    capability_card,
)
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.skills import build_default_registry
from olmoearth_agent.tools.registry import (
    Capability,
    RegisteredTool,
    ToolContext,
    ToolRegistry,
)

#: The card's budget in the system prompt, in tokens of four characters.
CARD_TOKEN_BUDGET = 1500

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
    assert "Cannot: rank a classification band" in review
    compare = lines["- olmoearth_compare_review"]
    assert "Cannot: say which map is right, even with labels_date" in compare
    estimate = lines["- olmoearth_estimate_map_error"]
    assert "Cannot: use a review set, or windows chosen by margin" in estimate
    certify = lines["- olmoearth_certify_zone"]
    assert "Needs: a design='random' plan and its labels." in certify


def test_a_loaded_group_is_listed_in_full() -> None:
    registry = build_default_registry()
    card = capability_card(registry, loaded_groups=["olmoearth-evaluate"])
    assert "\n- olmoearth_nndm_cv(points, pred_points): " in card
    assert "\n- olmoearth_rslearn_compose" not in card
    tail = card.split("More tools come with these skills", 1)[1].split("\n\n")[0]
    assert "olmoearth-evaluate" not in tail and "olmoearth-rslearn" in tail


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


def test_the_opt_in_python_tool_takes_files_and_model_runs_off_the_list() -> None:
    with mock.patch.dict(os.environ, {"OLMOEARTH_RUN_PYTHON": "1"}):
        registry = build_default_registry()
    beyond = beyond_tools(registry)
    assert NO_TOOL_CAN["files"] not in beyond
    assert NO_TOOL_CAN["run_model"] not in beyond
    assert NO_TOOL_CAN["labels"] in beyond and NO_TOOL_CAN["training"] in beyond


def test_covers_names_only_what_no_tool_can() -> None:
    with mock.patch.dict(os.environ, {"OLMOEARTH_RUN_PYTHON": "1"}):
        registry = build_default_registry()
    for name in registry.names():
        capability = registry.capability_of(name)
        assert capability is not None
        assert set(capability.covers) <= set(NO_TOOL_CAN), name


def test_the_card_fits_its_budget() -> None:
    """About 1,390 tokens (chars / 4) for the default registry's core tools."""
    card = capability_card(build_default_registry())
    assert _tokens(card) < CARD_TOKEN_BUDGET, _tokens(card)


def test_an_undeclared_tool_is_listed_from_its_description() -> None:
    registry = ToolRegistry()
    registry.register(_tool("plain", None, "Does a plain thing. Then more."))
    registry.register(_tool("known", Capability(does="a known thing", needs=("x",))))
    card = capability_card(registry)
    assert "- plain: Does a plain thing." in card
    assert "- known: a known thing. Needs: x." in card


def test_an_empty_registry_has_no_card() -> None:
    assert capability_card(ToolRegistry()) == ""
