# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Operational-safety helpers for the in-process agent.

Exposes the :mod:`~olmoearth_agent.security.egress` guard (a defence-in-depth
check on the outbound endpoints the agent talks to),
:mod:`~olmoearth_agent.security.paths` (confines model-controlled tool file I/O
to a workspace root) and :mod:`~olmoearth_agent.security.child_env` (the
environment a subprocess the agent starts receives, its keys removed).
"""

from __future__ import annotations

from olmoearth_agent.security.egress import (
    EgressDecision,
    EgressError,
    check_endpoint,
    validate_endpoint,
)
from olmoearth_agent.security.paths import (
    OUTPUT_ROOT_ENV,
    PathTraversalError,
    safe_path,
    workspace_root,
)

__all__ = [
    "OUTPUT_ROOT_ENV",
    "EgressDecision",
    "EgressError",
    "PathTraversalError",
    "check_endpoint",
    "safe_path",
    "validate_endpoint",
    "workspace_root",
]
