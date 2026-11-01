# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the environment a child process gets (security/child_env.py)."""

from __future__ import annotations

import pytest

from olmoearth_agent.security.child_env import scrubbed_env


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
    env = scrubbed_env()
    for name in secrets:
        assert name not in env, name
    assert env["GDAL_CACHEMAX"] == "512" and env.get("PATH")
