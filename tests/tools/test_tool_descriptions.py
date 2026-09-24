# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""No tool description states a measured result.

Descriptions reach every model call, whether or not the tool runs. In exp86
round 1 (brief 3, Studio run 1) the model quoted "51-70%" from
olmoearth_compare_review's description as a finding, in a run that never
called the tool, about two regression bands it does not apply to. A measured
figure belongs in a tool's output, where it applies and grounds the number.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from olmoearth_agent.skills import build_default_registry

#: Forms a measured result takes; rules and limits ("2-8 results", "2-16",
#: "0.05 = the most suspect 5%") are not in them.
_FINDINGS = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(?:%|percent)"),
    re.compile(r"\b\d+\s*/\s*\d+\s+(?:scenes|tasks|tiles|windows|events)"),
    re.compile(r"\bp\s*=\s*\d"),
    re.compile(r"\ball \d+ (?:scored )?(?:tasks|scenes|tiles|testbeds)"),
    re.compile(r"\b\d+ of \d+ (?:scenes|tasks|tiles|windows|events)"),
    re.compile(r"capture of \d"),
]


def _texts(name: str, schema: Any, where: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(schema, dict):
        if isinstance(schema.get("description"), str):
            out.append((f"{name}{where}", schema["description"]))
        for key, sub in (schema.get("properties") or {}).items():
            out.extend(_texts(name, sub, f"{where}.{key}"))
        if isinstance(schema.get("items"), dict):
            out.extend(_texts(name, schema["items"], f"{where}[]"))
    return out


def _all_texts() -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for spec in build_default_registry().specs():
        texts.append((spec.name, spec.description))
        texts.extend(_texts(spec.name, spec.parameters))
    return texts


@pytest.mark.parametrize("pattern", _FINDINGS, ids=lambda p: p.pattern)
def test_no_tool_description_states_a_measured_figure(pattern: re.Pattern[str]) -> None:
    hits = [
        (where, m.group(0))
        for where, text in _all_texts()
        for m in pattern.finditer(text)
    ]
    assert not hits, f"measured figures in tool descriptions: {hits}"


def test_the_compare_review_description_carries_no_evidence_figure() -> None:
    spec = next(
        s
        for s in build_default_registry().specs()
        if s.name == "olmoearth_compare_review"
    )
    assert "51" not in spec.description and "70" not in spec.description
    assert "%" not in spec.description
