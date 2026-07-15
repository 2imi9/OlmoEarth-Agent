# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for JSON-Schema tool-argument validation (tools/validate.py)."""

from __future__ import annotations

from olmoearth_agent.tools.validate import schema_summary, validate_arguments

_SCHEMA = {
    "type": "object",
    "properties": {
        "project_id": {"type": "string"},
        "limit": {"type": "integer"},
        "mode": {"type": "string", "enum": ["fast", "full"]},
        "bbox": {"type": "array", "items": {"type": "number"}},
        "options": {
            "type": "object",
            "properties": {"threshold": {"type": "number"}},
            "required": ["threshold"],
        },
    },
    "required": ["project_id"],
}


def test_valid_arguments_pass() -> None:
    args = {
        "project_id": "p1",
        "limit": 5,
        "mode": "fast",
        "bbox": [1.0, 2, 3.5, 4],
        "options": {"threshold": 0.5},
    }
    assert validate_arguments(args, _SCHEMA) == []


def test_missing_required_is_named() -> None:
    problems = validate_arguments({"limit": 5}, _SCHEMA)
    assert problems == ["missing required argument 'project_id'"]


def test_wrong_type_is_named_with_expectation() -> None:
    problems = validate_arguments({"project_id": "p", "limit": "five"}, _SCHEMA)
    assert len(problems) == 1
    assert "'limit'" in problems[0]
    assert "integer" in problems[0]
    assert "str" in problems[0]


def test_enum_violation_lists_choices() -> None:
    problems = validate_arguments({"project_id": "p", "mode": "turbo"}, _SCHEMA)
    assert len(problems) == 1
    assert "'fast'" in problems[0] and "'full'" in problems[0]


def test_array_items_and_nested_object_checked() -> None:
    problems = validate_arguments(
        {"project_id": "p", "bbox": [1, "two"], "options": {}}, _SCHEMA
    )
    assert any("bbox[1]" in p for p in problems)
    assert any("missing required key 'threshold'" in p for p in problems)


def test_string_accepts_coercible_scalars() -> None:
    # Handlers str()-coerce, so an int id must not be rejected.
    assert validate_arguments({"project_id": 42}, _SCHEMA) == []


def test_bool_is_not_a_number() -> None:
    problems = validate_arguments({"project_id": "p", "limit": True}, _SCHEMA)
    assert len(problems) == 1 and "'limit'" in problems[0]


def test_undeclared_arguments_are_permitted() -> None:
    assert validate_arguments({"project_id": "p", "extra": 1}, _SCHEMA) == []


def test_non_object_schema_is_ignored() -> None:
    assert validate_arguments({"anything": 1}, {}) == []
    assert validate_arguments({"anything": 1}, {"type": "string"}) == []


def test_schema_summary_is_compact() -> None:
    summary = schema_summary(_SCHEMA)
    assert summary["required"] == ["project_id"]
    assert summary["properties"]["project_id"] == "string"
    assert summary["properties"]["mode"] == {"enum": ["fast", "full"]}
