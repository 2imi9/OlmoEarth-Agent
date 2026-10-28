# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth-case-narrative`` tool bundle (skill #14).

Turns a run's results + provenance into a stakeholder Markdown report,
with a freshness gate that refuses to render stale tiles, and a "How
accurate is this map" section that states what the request's verification
tools measured, from their recorded results (``ProvenanceLog.verification``,
read as the provenance is), never from figures the model types. A request
is one run of the agent: on the web UI, one message.

The title and results are the model's typing, which the report renders as
given: a title or result that states an accuracy or error rate as a figure
("Flood extent: 94% accurate") is refused
(:func:`~olmoearth_agent.harness.checks.states_accuracy_figure`), whether or
not the request measured one, since the accuracy section states every
figure the tools returned. The report's own Markdown is no source of numbers
for the answer's number check (``not_a_source``).

A request that estimated no error rate gets a report that says so and names
the tools that estimate one, and the output contract
(:mod:`olmoearth_agent.tools.statistical_rules`): the answer must convey
that no error rate was estimated (``must_state``), and may state no
accuracy figure for the map (``accuracy_not_measured``, read by the answer
checks).
"""

from __future__ import annotations

from typing import Any

from olmoearth_agent.harness.checks import sentences, states_accuracy_figure
from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.reporting.narrative import build_narrative, typed_text
from olmoearth_agent.tools import statistical_rules as rules
from olmoearth_agent.tools.registry import Capability, RegisteredTool, ToolContext

#: Why a title or result typed with an accuracy figure is refused. It names
#: where, never the figure: the error goes back to the model, and a figure
#: echoed in a result would be a source for the answer's number check.
TYPED_ACCURACY_REFUSED = (
    "an accuracy or error rate is typed as a figure in the report's {where}: "
    "the report states only the accuracy this request's verification tools "
    "measured, in its accuracy section, as they returned it. Remove the figure "
    "from the {where} and call again."
)


def _refuse_typed_accuracy(title: str, results: list[dict[str, Any]]) -> None:
    """Raise when the text the report renders from its caller states an accuracy.

    Read sentence by sentence (:func:`~olmoearth_agent.harness.checks.sentences`)
    as the answer checks read ``accuracy_not_measured``; the registry returns
    the ``ValueError`` as a failed call the model can correct.
    """
    texts = typed_text(title, results)
    for where, group in (("title", texts[:1]), ("results", texts[1:])):
        if any(
            states_accuracy_figure(sentence.text)
            for text in group
            for sentence in sentences(text)
        ):
            raise ValueError(TYPED_ACCURACY_REFUSED.format(where=where))


async def _case_narrative(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    results = args.get("results", [])
    _refuse_typed_accuracy(args["title"], results)
    provenance = ctx.state.provenance
    out = build_narrative(
        args["title"],
        results,
        provenance=provenance,
        verification=provenance.verification,
        window_hours=float(args.get("freshness_window_hours", 24)),
    )
    accuracy = out["accuracy"]
    if accuracy["measured"]:
        return out
    # No error rate estimated: say so, and forbid any accuracy figure for it.
    which_map = bool(accuracy["which_map"])
    stated = (
        rules.NEITHER_MAP_MEASURED_MUST_STATE
        if which_map
        else rules.NOT_MEASURED_MUST_STATE
    )
    return rules.add_contract(
        out,
        must_state=[stated],
        forbidden_claims=[rules.accuracy_not_measured(which_map=which_map)],
    )


def build_narrative_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth-case-narrative`` tool bundle."""
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_case_narrative",
                description=(
                    "Write a stakeholder Markdown report from prediction "
                    "results (tile URLs + properties) and the run's "
                    "provenance; tiles older than the freshness window are "
                    "struck through, not rendered. It states the accuracy "
                    "this run's verification tools measured, as returned, or "
                    "that none was measured; put no accuracy figure in title "
                    "or results. Call at the end of a task."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "results": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Each: result_id, tile_urls, property_names, "
                                "creation_time (ISO-8601)."
                            ),
                        },
                        "freshness_window_hours": {
                            "type": "number",
                            "default": 24,
                        },
                    },
                    "required": ["title", "results"],
                },
            ),
            handler=_case_narrative,
            capability=Capability(
                does="a Markdown report from prediction results and this run's "
                "provenance, with the accuracy this run measured"
            ),
        ),
    ]
