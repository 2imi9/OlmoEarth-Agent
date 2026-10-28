# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the environment a child process gets (security/child_env.py)."""

from __future__ import annotations

import os

import pytest

from olmoearth_agent.analysis import asta
from olmoearth_agent.security.child_env import scrubbed_env
from olmoearth_agent.tools import system


def test_the_agents_keys_are_left_out_and_the_rest_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every name that marks a credential, by prefix or by fragment, in any
    case, is dropped; PATH and a tool's own settings are kept."""
    secrets = {
        "OLMOEARTH_API_KEY": "studio",
        "OLMOEARTH_SCORES_ROOT": "/data",
        "LLM_API_KEY": "llm",
        "anthropic_api_key": "claude",
        "HF_TOKEN": "hf",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "MY_SERVICE_PASSWORD": "pw",
        "GITHUB_TOKEN": "gh",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GDAL_CACHEMAX", "512")
    monkeypatch.setenv("ASTA_TOKEN", "asta")
    env = scrubbed_env()
    for name in [*secrets, "ASTA_TOKEN"]:
        assert name not in env, name
    assert env["GDAL_CACHEMAX"] == "512" and env.get("PATH")
    # A prefix kept passes whatever it holds, for a CLI's own login.
    kept = scrubbed_env(keep_prefixes=("ASTA",))
    assert kept["ASTA_TOKEN"] == os.environ["ASTA_TOKEN"]
    assert "OLMOEARTH_API_KEY" not in kept


def test_every_child_the_agent_starts_uses_the_one_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """olmoearth_run_python's snippet and the Asta CLI get the same scrubbed
    environment as olmoearth-inferencex's command line (inferencex.run_cli)."""
    monkeypatch.setenv("OLMOEARTH_API_KEY", "studio")
    monkeypatch.setenv("ASTA_TOKEN", "asta")
    assert system._scrubbed_env() == scrubbed_env()
    assert asta._scrubbed_env() == scrubbed_env(keep_prefixes=("ASTA",))
