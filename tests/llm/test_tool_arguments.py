# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tool-call argument decoding must yield a dict whatever the model sent."""

from __future__ import annotations

import json

from olmoearth_agent.llm.client import decode_tool_arguments


def test_single_encoded_object_decodes() -> None:
    assert decode_tool_arguments(json.dumps({"budget": 0.05})) == {"budget": 0.05}


def test_double_encoded_object_is_decoded_twice() -> None:
    """Qwen through a shim sends the object as a JSON string; that used to crash the turn."""
    twice = json.dumps(json.dumps({"scores_path": "/x/scores.json"}))
    assert decode_tool_arguments(twice) == {"scores_path": "/x/scores.json"}


def test_malformed_json_is_surfaced_not_dropped() -> None:
    out = decode_tool_arguments("{not json")
    assert out == {"__raw_arguments": "{not json"}


def test_a_string_that_is_not_an_object_is_surfaced() -> None:
    assert decode_tool_arguments(json.dumps("just text")) == {"__raw_arguments": "just text"}


def test_a_non_object_json_value_is_surfaced() -> None:
    assert decode_tool_arguments("[1, 2]") == {"__raw_arguments": "[1, 2]"}
