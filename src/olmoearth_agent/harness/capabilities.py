# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The capability card: what this run's tools do, need and cannot do.

exp86 round 9's audit found that the model answers only from its tools, and
that most of its material errors sat where the tools said nothing and the
model filled the gap from general knowledge. 7 of the 16 were offers of
actions no tool can do or whose preconditions did not hold: a review set of a
regression band with no threshold, "I'll set up a direct model run", labels
on flagged windows turned into "a proper error-rate estimate", a comparison
"rerun with labels_date", which takes no labels. The owner wants the agent to
keep proposing next steps, but only possible ones, with their preconditions.

The card is the list the soul tells the model to propose from. It is
assembled from code: each tool's required arguments (its spec) and its
:class:`~olmoearth_agent.tools.registry.Capability`, declared beside its
handler, whose ``needs`` and ``cannot`` a test calls the tool in. The core
tools are listed, and the deferred groups the run has loaded; a group not
loaded is named, since ``olmoearth_load_skill`` lists and loads its tools. :data:`NO_TOOL_CAN` ends the card: what no tool of the registry
can do, each dropped as soon as a registered tool declares it ``covers`` it.
"""

from __future__ import annotations

from collections.abc import Iterable

from olmoearth_agent.tools.registry import Capability, ToolRegistry

__all__ = ["CARD_TITLE", "NO_TOOL_CAN", "beyond_tools", "capability_card"]

#: The card's heading in the system prompt (the soul's rules name it).
CARD_TITLE = "## Capability card"

#: What no agent tool can do, by the id a tool's ``Capability.covers`` names.
#: Read off the registry: ``olmoearth_scores_from_file`` reads a run and
#: ``olmoearth_submit_prediction`` runs a Studio model; the labels tools take
#: labels from their caller; a model's summary gives its name, type,
#: prediction_type and no-data value only (``tools/sampling.summarize_model``);
#: file paths are confined to the workspace and the scores root
#: (``security/paths.safe_path``, ``tools/review_set.read_roots``). exp86
#: round 9 claimed "no ground-truth labels exist" while Studio's model records
#: showed both models fine-tuned on the project's label fields, which no tool
#: reads, and offered to "set up" a direct model run.
NO_TOOL_CAN: dict[str, str] = {
    "run_model": (
        "run a model outside Studio, on a GPU cluster or locally "
        "(olmoearth_scores_from_file reads a run already made)"
    ),
    "labels": (
        "look up or fetch ground-truth labels (they come from the user or a "
        "reviewer), or say whether any exist"
    ),
    "training": (
        "read what a Studio model was trained on (label fields, training data, "
        "metrics)"
    ),
    "files": (
        "read files outside the workspace, the scores root and, with no scores "
        "root set, the working directory"
    ),
}

_HEAD = (
    "What each tool of this run does, from its code (required arguments in "
    "parentheses), what it needs and what it cannot do. Propose only these "
    "actions, with their preconditions."
)


def _required(registry: ToolRegistry, name: str) -> list[str]:
    """The spec's required arguments of the tool called ``name``."""
    spec = registry.spec_of(name)
    params = spec.parameters if spec is not None else {}
    required = params.get("required") if isinstance(params, dict) else None
    return [str(r) for r in required or []]


def _first_sentence(text: str) -> str:
    """The first sentence of a description, for a tool that declares nothing."""
    head = text.strip().split(". ", 1)[0].rstrip(".")
    return head[:160]


def tool_line(registry: ToolRegistry, name: str) -> str:
    """One tool's line on the card: ``- name(required): does. Needs: ... Cannot: ...``.

    A tool with no declaration (a test's or a plugin's) is listed with its
    description's first sentence, so the card never omits a callable tool.
    """
    capability = registry.capability_of(name)
    required = _required(registry, name)
    head = f"- {name}({', '.join(required)})" if required else f"- {name}"
    if capability is None:
        spec = registry.spec_of(name)
        return f"{head}: {_first_sentence(spec.description if spec else '')}."
    parts = [f"{head}: {capability.does}."]
    if capability.needs:
        parts.append("Needs: " + "; ".join(capability.needs) + ".")
    if capability.cannot:
        parts.append("Cannot: " + "; ".join(capability.cannot) + ".")
    return " ".join(parts)


def beyond_tools(registry: ToolRegistry) -> list[str]:
    """What no registered tool can do: :data:`NO_TOOL_CAN` less what one covers.

    Every registered tool counts, core or deferred: a deferred tool is one
    ``olmoearth_load_skill`` call away.
    """
    covered: set[str] = set()
    for name in registry.names():
        capability: Capability | None = registry.capability_of(name)
        if capability is not None:
            covered |= set(capability.covers)
    return [text for key, text in NO_TOOL_CAN.items() if key not in covered]


def capability_card(registry: ToolRegistry, loaded_groups: Iterable[str] = ()) -> str:
    """The capability card for ``registry``, with ``loaded_groups`` listed in full.

    Parameters
    ----------
    registry
        The run's tools.
    loaded_groups
        The deferred groups the run has loaded (``ThreadState.loaded_groups``);
        their tools are listed like the core ones. Any other group is named
        on one line.

    Returns
    -------
    str
        The card as Markdown, starting with :data:`CARD_TITLE`; empty for a
        registry with no tools.
    """
    names = registry.names()
    if not names:
        return ""
    loaded = set(loaded_groups)
    listed = [
        n
        for n in names
        if registry.group_of(n) is None or registry.group_of(n) in loaded
    ]
    lines = [CARD_TITLE, "", _HEAD, ""]
    lines += [tool_line(registry, n) for n in listed]
    # A group not loaded is named only: olmoearth_load_skill's description
    # lists its tools, and loading it puts them on the card.
    waiting = [g for g in registry.groups() if g not in loaded]
    if waiting:
        lines += [
            "",
            "More tools come with these skills, once olmoearth_load_skill loads "
            "one: " + ", ".join(waiting) + ".",
        ]
    beyond = beyond_tools(registry)
    if beyond:
        lines += ["", "No tool of this agent can:"]
        lines += [f"- {text}." for text in beyond]
    return "\n".join(lines)
