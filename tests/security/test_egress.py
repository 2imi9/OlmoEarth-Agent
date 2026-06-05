# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Unit tests for the in-process egress guard."""

from __future__ import annotations

import pytest

from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.security import egress


# --------------------------------------------------------------------------- #
# check_endpoint: allow cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("url", "capability"),
    [
        ("https://olmoearth.allenai.org/api/v1", "studio"),
        ("https://api.anthropic.com", "llm-cloud"),
        ("https://api.openai.com/v1", "llm-cloud"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "llm-cloud"),
        ("http://export.arxiv.org/api/query", "litsearch"),
        ("https://api.openalex.org/works", "litsearch"),
        ("https://datasets-server.huggingface.co/rows", "hf"),
    ],
)
def test_allowlisted_hosts_pass(url: str, capability: str) -> None:
    decision = egress.check_endpoint(url, capability)
    assert decision.allowed is True
    assert decision.capability == capability


@pytest.mark.parametrize(
    "url", ["http://localhost:8000/v1", "http://127.0.0.1:8000", "http://[::1]:8000"]
)
def test_loopback_allowed_only_for_local_inference(url: str) -> None:
    assert egress.check_endpoint(url, "llm-local").allowed is True
    # The same loopback endpoint is denied for a credentialed cloud capability.
    assert egress.check_endpoint(url, "llm-cloud").allowed is False
    assert egress.check_endpoint(url, "studio").allowed is False


# --------------------------------------------------------------------------- #
# check_endpoint: SSRF / deny cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.5",  # RFC1918
        "192.168.1.1",  # RFC1918
        "172.16.9.9",  # RFC1918
        "169.254.169.254",  # link-local cloud metadata
        "100.64.0.1",  # CGNAT (RFC6598)
        "0.0.0.0",  # unspecified  # noqa: S104 - fixture asserting this is blocked
        "[fe80::1]",  # IPv6 link-local
        "[fd00::1]",  # IPv6 ULA
        "[::ffff:10.0.0.1]",  # IPv4-mapped IPv6 -> private v4
    ],
)
def test_private_and_internal_ips_are_blocked(host: str) -> None:
    # Even for the loopback-permitting capability, non-loopback internal
    # addresses stay blocked (loopback != link-local/private).
    decision = egress.check_endpoint(f"https://{host}/x", "llm-local")
    assert decision.allowed is False
    assert "IP address" in decision.reason


@pytest.mark.parametrize(
    "host",
    ["metadata.google.internal", "foo.internal", "service.local", "anything.localhost"],
)
def test_reserved_names_blocked_for_credentialed_capability(host: str) -> None:
    assert egress.check_endpoint(f"https://{host}/x", "studio").allowed is False


def test_non_allowlisted_public_host_denied() -> None:
    decision = egress.check_endpoint("https://evil.example.com/steal", "studio")
    assert decision.allowed is False
    assert "allowlist" in decision.reason


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "gopher://h"])
def test_unsupported_schemes_denied(url: str) -> None:
    decision = egress.check_endpoint(url, "studio")
    assert decision.allowed is False
    assert "scheme" in decision.reason


def test_unknown_capability_denied() -> None:
    assert egress.check_endpoint("https://api.openai.com", "made-up").allowed is False


def test_decision_records_host_only_not_url() -> None:
    decision = egress.check_endpoint(
        "https://api.openalex.org/works?secret=abc", "litsearch"
    )
    assert decision.host == "api.openalex.org"
    assert "secret" not in decision.host


# --------------------------------------------------------------------------- #
# OLMOEARTH_EGRESS_ALLOW: operator host extension
# --------------------------------------------------------------------------- #
def test_env_allowlist_extends_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        egress.check_endpoint(
            "https://studio.internal-corp.example/api", "studio"
        ).allowed
        is False
    )
    monkeypatch.setenv(
        "OLMOEARTH_EGRESS_ALLOW", "studio.internal-corp.example, other.example"
    )
    assert (
        egress.check_endpoint(
            "https://studio.internal-corp.example/api", "studio"
        ).allowed
        is True
    )


# --------------------------------------------------------------------------- #
# validate_endpoint: mode handling
# --------------------------------------------------------------------------- #
def test_audit_mode_allows_bad_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLMOEARTH_EGRESS", raising=False)  # default is audit
    # Returns the url (does not raise) even though the host is off-allowlist.
    assert (
        egress.validate_endpoint("https://evil.example.com", "studio")
        == "https://evil.example.com"
    )


def test_enforce_mode_raises_on_bad_host() -> None:
    with pytest.raises(egress.EgressError) as excinfo:
        egress.validate_endpoint(
            "https://169.254.169.254/latest/meta-data", "studio", mode="enforce"
        )
    assert "studio" in str(excinfo.value)


def test_enforce_mode_allows_good_host() -> None:
    url = "https://olmoearth.allenai.org/api/v1"
    assert egress.validate_endpoint(url, "studio", mode="enforce") == url


def test_off_mode_skips_check() -> None:
    # off returns the url without classifying it at all.
    bad = "https://169.254.169.254/x"
    assert egress.validate_endpoint(bad, "studio", mode="off") == bad


def test_env_mode_enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_EGRESS", "enforce")
    with pytest.raises(egress.EgressError):
        egress.validate_endpoint("https://evil.example.com", "studio")


def test_invalid_env_mode_falls_back_to_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLMOEARTH_EGRESS", "banana")
    # Falls back to audit -> does not raise.
    assert (
        egress.validate_endpoint("https://evil.example.com", "studio")
        == "https://evil.example.com"
    )


# --------------------------------------------------------------------------- #
# validate_endpoint: provenance audit hook
# --------------------------------------------------------------------------- #
def test_validate_records_to_auditor_in_audit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLMOEARTH_EGRESS", raising=False)
    log = ProvenanceLog(run_id="r")
    egress.validate_endpoint("https://evil.example.com", "studio", auditor=log)
    assert len(log.egress) == 1
    assert log.egress[0]["host"] == "evil.example.com"
    assert log.egress[0]["allowed"] is False


def test_validate_records_to_auditor_before_enforce_raise() -> None:
    log = ProvenanceLog()
    with pytest.raises(egress.EgressError):
        egress.validate_endpoint(
            "https://10.0.0.1", "studio", mode="enforce", auditor=log
        )
    # The decision is recorded even though the call then raised.
    assert len(log.egress) == 1
    assert log.egress[0]["allowed"] is False
