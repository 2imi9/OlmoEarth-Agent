# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""In-process egress guard: validate an outbound endpoint before the agent
sends a request -- and possibly a credential -- to it.

Informed by NVIDIA NemoClaw's network-policy model (per-integration host
allowlists with ``enforce`` / ``audit`` enforcement) and its SSRF guard
(``nemoclaw/src/blueprint/{ssrf,private-networks}.ts``), re-scoped for a
single-user *in-process* agent. We have no container and cannot enforce at a
network layer, so this is a defence-in-depth check at the boundaries where an
attacker-influenceable URL meets a real credential: the env-configured Studio
base URL (``OLMOEARTH_BASE_URL``) and the hosted-LLM endpoints. It also gives
every outbound call an auditable destination record (see
:meth:`olmoearth_agent.provenance.log.ProvenanceLog.record_egress`).

What it stops
-------------
- Sending a Studio / LLM credential to a private, internal, or cloud-metadata
  address (SSRF + credential exfiltration) reached via a malicious or typo'd
  endpoint override.
- Reaching a host that is not on the per-capability allowlist.

Honest limits
-------------
- This is **not** a sandbox. It guards the URLs *this code* builds; it does not
  constrain arbitrary egress from opt-in executed code
  (:mod:`olmoearth_agent.tools.system`) or anything else on the host.
- It does **not** resolve DNS (NemoClaw pins post-resolution IPs to defeat DNS
  rebinding; an always-running container can afford that, a per-call interactive
  agent should not block on a lookup). The allowlist is therefore the primary
  control -- a rebinding hostname would still have to be allowlisted to be
  reached in ``enforce`` mode.
- The default mode is ``audit`` (log only). Set ``OLMOEARTH_EGRESS=enforce`` to
  block; ``OLMOEARTH_EGRESS_ALLOW=host1,host2`` to allowlist self-hosted Studio
  or LLM hosts; ``OLMOEARTH_EGRESS=off`` to disable the guard entirely.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from olmoearth_agent.provenance.log import ProvenanceLog

logger = logging.getLogger(__name__)

#: Outbound capabilities and the hosts each is allowed to reach. Mirrors
#: NemoClaw's ``policies/presets/*.yaml`` (one named bundle per integration),
#: right-sized to the agent's known dependencies. ``allow_loopback`` is the
#: local-inference exception (cf. NemoClaw ``presets/local-inference.yaml``,
#: which must explicitly allowlist the loopback gateway its SSRF guard would
#: otherwise reject).
_CAPABILITIES: dict[str, "_Capability"] = {}


@dataclass(frozen=True)
class _Capability:
    name: str
    hosts: frozenset[str]
    allow_loopback: bool = False


def _register(
    name: str, hosts: tuple[str, ...], *, allow_loopback: bool = False
) -> None:
    _CAPABILITIES[name] = _Capability(name, frozenset(hosts), allow_loopback)


_register("studio", ("olmoearth.allenai.org",))
_register(
    "llm-cloud",
    (
        "api.anthropic.com",
        "api.openai.com",
        "generativelanguage.googleapis.com",
        "integrate.api.nvidia.com",
    ),
)
_register("llm-local", (), allow_loopback=True)
_register("litsearch", ("export.arxiv.org", "api.openalex.org"))
_register("hf", ("datasets-server.huggingface.co",))

#: Reserved IPv4 ranges that ``ipaddress`` does not flag as private on every
#: supported interpreter: RFC 6598 carrier-grade NAT and the 0.0.0.0/8 block.
_EXTRA_BLOCKED_V4: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("100.64.0.0/10"),
    ipaddress.IPv4Network("0.0.0.0/8"),
)

#: Hostname suffixes that name internal/loopback services and must never carry
#: a credential off-box.
_RESERVED_NAME_SUFFIXES = (".local", ".internal", ".localhost")
#: Exact reserved names (cloud metadata services, etc.).
_RESERVED_NAMES = frozenset({"metadata", "metadata.google.internal"})

_VALID_MODES = frozenset({"off", "audit", "enforce"})


class EgressError(RuntimeError):
    """Raised in ``enforce`` mode when an outbound endpoint is not allowed."""


@dataclass(frozen=True)
class EgressDecision:
    """The verdict for one ``(url, capability)`` pair.

    ``host`` is the bare hostname only (never the path or query), so recording
    a decision in the provenance manifest cannot leak a key embedded in a URL.
    """

    allowed: bool
    host: str
    capability: str
    reason: str


def _mode() -> str:
    """Active enforcement mode from ``OLMOEARTH_EGRESS`` (default ``audit``)."""
    raw = os.environ.get("OLMOEARTH_EGRESS", "audit").strip().lower()
    return raw if raw in _VALID_MODES else "audit"


def _extra_allow() -> frozenset[str]:
    """Operator-added hosts from ``OLMOEARTH_EGRESS_ALLOW`` (comma-separated)."""
    raw = os.environ.get("OLMOEARTH_EGRESS_ALLOW", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` as an IP literal, unwrapping IPv4-mapped IPv6; else None."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any non-public IP (private / link-local / reserved / etc.).

    Loopback is classified separately by the caller so the local-inference
    capability can permit it; everything else non-global is blocked.
    """
    if ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return True
    if ip.is_unspecified:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _EXTRA_BLOCKED_V4)
    return False


def _is_loopback_name(host: str) -> bool:
    return host == "localhost" or host.endswith(".localhost")


def _is_reserved_name(host: str) -> bool:
    return host in _RESERVED_NAMES or host.endswith(_RESERVED_NAME_SUFFIXES)


def check_endpoint(url: str, capability: str) -> EgressDecision:
    """Classify ``url`` for ``capability`` without raising or logging.

    Pure verdict function -- :func:`validate_endpoint` wraps it with mode
    handling, logging, and the optional provenance record.
    """
    cap = _CAPABILITIES.get(capability)
    if cap is None:
        return EgressDecision(
            False, "", capability, f"unknown capability {capability!r}"
        )

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return EgressDecision(
            False, "", capability, f"unsupported URL scheme {scheme or '(none)'!r}"
        )
    host = (parts.hostname or "").lower()
    if not host:
        return EgressDecision(False, "", capability, "no host in URL")

    ip = _parse_ip(host)
    is_loopback = ip.is_loopback if ip is not None else _is_loopback_name(host)
    if is_loopback:
        if cap.allow_loopback:
            return EgressDecision(
                True, host, capability, "loopback allowed for this capability"
            )
        return EgressDecision(
            False,
            host,
            capability,
            "loopback/localhost not allowed for this capability",
        )
    if ip is not None and _ip_blocked(ip):
        return EgressDecision(
            False, host, capability, "private/internal/reserved IP address"
        )
    if ip is None and _is_reserved_name(host):
        return EgressDecision(False, host, capability, "reserved/internal hostname")

    if host in (cap.hosts | _extra_allow()):
        return EgressDecision(True, host, capability, "host on allowlist")
    return EgressDecision(
        False, host, capability, f"host not on {capability!r} allowlist"
    )


def validate_endpoint(
    url: str,
    capability: str,
    *,
    mode: str | None = None,
    auditor: "ProvenanceLog | None" = None,
) -> str:
    """Validate ``url`` for ``capability``; return it (for call-site chaining).

    In ``enforce`` mode a disallowed endpoint raises :class:`EgressError`. In
    ``audit`` mode (the default) it is logged at WARNING and allowed through, so
    turning the guard on never breaks a working deployment. When ``auditor`` is
    given, the decision is appended to the provenance manifest regardless of
    mode. ``off`` skips the check entirely.
    """
    resolved = (mode or _mode()).lower()
    if resolved == "off":
        return url

    decision = check_endpoint(url, capability)
    if auditor is not None:
        try:
            auditor.record_egress(decision)
        except Exception:  # provenance is best-effort; never break the request
            logger.debug("egress: failed to record provenance entry", exc_info=True)

    if decision.allowed:
        logger.debug(
            "egress allow: %s -> %s (%s)", capability, decision.host, decision.reason
        )
        return url

    host = decision.host or "(no host)"
    if resolved == "enforce":
        logger.error("egress BLOCK: %s -> %s (%s)", capability, host, decision.reason)
        raise EgressError(
            f"Blocked outbound request for capability {capability!r}: {decision.reason} "
            f"(host={host!r}). Allowlist a custom host with OLMOEARTH_EGRESS_ALLOW, "
            f"or disable the guard with OLMOEARTH_EGRESS=off."
        )
    logger.warning(
        "egress audit: would block %s -> %s (%s); set OLMOEARTH_EGRESS=enforce to block "
        "or OLMOEARTH_EGRESS_ALLOW to allowlist",
        capability,
        host,
        decision.reason,
    )
    return url
