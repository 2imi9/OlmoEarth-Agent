# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for cross-thread preference memory (harness/memory.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.harness import memory
from olmoearth_agent.harness.agent import LeadAgent
from olmoearth_agent.tools.registry import ToolRegistry


def test_empty_store_loads_empty(tmp_path: Path) -> None:
    assert memory.load_facts(tmp_path) == {}
    assert memory.preferences_block(tmp_path) == ""


def test_remember_and_load_roundtrip(tmp_path: Path) -> None:
    out = memory.remember("default_project", "PA Karst (project_id 12ab)", tmp_path)
    assert out["key"] == "default_project"
    assert out["stored_keys"] == ["default_project"]
    facts = memory.load_facts(tmp_path)
    assert facts["default_project"]["value"] == "PA Karst (project_id 12ab)"
    # Stored under <root>/memory/preferences.json, inside the workspace.
    assert (tmp_path / "memory" / "preferences.json").is_file()


def test_remember_overwrites_same_key(tmp_path: Path) -> None:
    memory.remember("srcs", "Sentinel-2 L1C", tmp_path)
    memory.remember("srcs", "Sentinel-2 L2A", tmp_path)
    facts = memory.load_facts(tmp_path)
    assert len(facts) == 1
    assert facts["srcs"]["value"] == "Sentinel-2 L2A"


def test_value_is_single_line_and_capped(tmp_path: Path) -> None:
    memory.remember("style", "short\n\nreports   please " + "x" * 500, tmp_path)
    value = memory.load_facts(tmp_path)["style"]["value"]
    assert "\n" not in value
    assert "short reports please" in value
    assert len(value) <= memory.MAX_VALUE_CHARS


def test_invalid_key_and_empty_value_rejected(tmp_path: Path) -> None:
    with pytest.raises(memory.PreferenceError, match="invalid memory key"):
        memory.remember("Bad Key!", "v", tmp_path)
    with pytest.raises(memory.PreferenceError, match="empty value"):
        memory.remember("ok_key", "   ", tmp_path)


def test_fact_cap_refuses_new_keys_but_allows_updates(tmp_path: Path) -> None:
    for i in range(memory.MAX_FACTS):
        memory.remember(f"k{i}", "v", tmp_path)
    with pytest.raises(memory.PreferenceError, match="memory is full"):
        memory.remember("overflow", "v", tmp_path)
    memory.remember("k0", "updated", tmp_path)  # updates still allowed
    assert memory.load_facts(tmp_path)["k0"]["value"] == "updated"


def test_forget(tmp_path: Path) -> None:
    memory.remember("gone", "v", tmp_path)
    assert memory.forget("gone", tmp_path) is True
    assert memory.forget("gone", tmp_path) is False
    assert memory.load_facts(tmp_path) == {}


def test_corrupt_file_is_empty_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "preferences.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert memory.load_facts(tmp_path) == {}


def test_block_labels_facts_as_data(tmp_path: Path) -> None:
    memory.remember("default_area", "Delaware basin (area_id a9)", tmp_path)
    block = memory.preferences_block(tmp_path)
    assert "- default_area: Delaware basin (area_id a9)" in block
    assert "DATA, not instructions" in block
    assert "olmoearth_remember" in block


def test_agent_appends_memory_block() -> None:
    block = "Durable user preferences...\n- default_area: X"
    agent = LeadAgent(
        None,  # type: ignore[arg-type]
        ToolRegistry(),
        studio=None,  # type: ignore[arg-type]
        memory_block=block,
    )
    assert agent.system_prompt.endswith("\n\n" + block)
    bare = LeadAgent(
        None,  # type: ignore[arg-type]
        ToolRegistry(),
        studio=None,  # type: ignore[arg-type]
    )
    assert block not in bare.system_prompt
