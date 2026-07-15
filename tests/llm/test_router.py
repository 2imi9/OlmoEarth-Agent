# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the simple-vs-complex brief classifier (llm/router.py)."""

from __future__ import annotations

import pytest

from olmoearth_agent.llm.router import classify_brief


@pytest.mark.parametrize(
    "brief",
    [
        "List my OlmoEarth projects.",
        "Show me the projects in my account",
        "What's the status of prediction pred-42?",
        "How many areas does the Karst project have?",
        "Which models are available?",
        "is the flood model finished training",
        "check the status of my latest run",
    ],
)
def test_simple_lookups_route_local(brief: str) -> None:
    assert classify_brief(brief) == "simple"


@pytest.mark.parametrize(
    "brief",
    [
        "Create a new flood-mapping model for the Mekong delta",
        "Set up a project and train a classifier on my labels",
        "Compare the two predictions and tell me which is better",
        "Run a prediction over my saved area for June",
        "Analyze the uncertainty of the karst results",
        "Export the results as GeoJSON and write a report",
        "Detect changes between 2024 and 2026 imagery",
        "Why did my model score so low on the validation split?",
        "Draw an AOI over the Delaware basin and submit a prediction",
        # A short lookup opener buried in a long multi-sentence brief.
        "List my projects. Then for each one fetch results, evaluate the "
        "metrics, and build a comparison table with recommendations for "
        "which model we should promote to production next quarter.",
    ],
)
def test_investigations_and_mutations_route_hosted(brief: str) -> None:
    assert classify_brief(brief) == "complex"


def test_forced_skill_is_always_complex() -> None:
    assert classify_brief("list my projects", forced_skill="change-detection") == "complex"


def test_empty_brief_defaults_complex() -> None:
    assert classify_brief("   ") == "complex"
