# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the export helper functions."""

from __future__ import annotations

from olmoearth_agent.reporting.export import curate, group_items, slugify, to_json


def test_slugify() -> None:
    assert slugify("PA Karst Final") == "pa-karst-final"
    assert slugify("") == "untitled"
    assert slugify(None) == "untitled"
    assert slugify("a/b c!") == "a-b-c"


def test_group_items() -> None:
    grouped = group_items([{"s": "a"}, {"s": "b"}, {"s": "a"}], "s")
    assert set(grouped) == {"a", "b"}
    assert len(grouped["a"]) == 2


def test_group_items_missing_key() -> None:
    grouped = group_items([{"x": 1}], "s")
    assert "unknown" in grouped


def test_curate_keeps_only_present_fields() -> None:
    assert curate({"a": 1, "b": 2, "c": 3}, ("a", "c", "z")) == {"a": 1, "c": 3}


def test_to_json_is_sorted() -> None:
    blob = to_json({"b": 1, "a": 2})
    assert blob.index('"a"') < blob.index('"b"')
