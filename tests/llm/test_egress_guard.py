# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The LLM client guards its (env-overridable) endpoint via the egress policy.

A malicious ``LLM_ENDPOINT`` must not exfiltrate the conversation or a
hosted-provider key. Local inference is loopback (``llm-local``); a hosted
provider must be a known cloud host (``llm-cloud``).
"""

from __future__ import annotations

import pytest

from olmoearth_agent.llm.client import OlmoEarthLLM
from olmoearth_agent.llm.config import ServingConfig
from olmoearth_agent.security import egress


def _llm(endpoint: str) -> OlmoEarthLLM:
    return OlmoEarthLLM(ServingConfig(endpoint=endpoint, api_key="EMPTY"))


def test_local_loopback_endpoint_allowed_even_in_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    # the default local llama.cpp endpoint -> llm-local (loopback) -> allowed
    _llm("http://localhost:8000/v1")
    _llm("http://127.0.0.1:8000/v1")


def test_known_cloud_provider_allowed_in_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    # a hosted provider on the llm-cloud allowlist -> allowed
    _llm("https://api.anthropic.com/v1")
    _llm("https://api.openai.com/v1")


def test_malicious_endpoint_blocked_in_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    with pytest.raises(egress.EgressError):
        _llm("https://evil.example.com/v1")
    # SSRF to the cloud metadata endpoint must also be blocked
    with pytest.raises(egress.EgressError):
        _llm("http://169.254.169.254/v1")


def test_malicious_endpoint_audited_not_blocked_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # default mode is audit: log a warning, but never break a deployment
    monkeypatch.delenv("OLMOEARTH_EGRESS", raising=False)
    _llm("https://evil.example.com/v1")  # constructs without raising
