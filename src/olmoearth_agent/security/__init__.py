# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Operational-safety helpers for the in-process agent.

Currently exposes the :mod:`~olmoearth_agent.security.egress` guard, a
defence-in-depth check on the outbound endpoints the agent talks to.
"""

from __future__ import annotations

from olmoearth_agent.security.egress import (
    EgressDecision,
    EgressError,
    check_endpoint,
    validate_endpoint,
)

__all__ = [
    "EgressDecision",
    "EgressError",
    "check_endpoint",
    "validate_endpoint",
]
