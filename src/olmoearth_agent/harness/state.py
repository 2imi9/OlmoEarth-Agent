# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Shared thread state, mutated by tools across a single agent run.

Mirrors the reducer-merged ``ThreadState`` pattern from DeerFlow v2,
trimmed to what the OlmoEarth Agent needs today. Tools read/write this
via :class:`olmoearth_agent.tools.registry.ToolContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.provenance.log import ProvenanceLog
from olmoearth_agent.types import StudioContext


@dataclass
class ThreadState:
    """Per-run state shared across tool calls."""

    studio_context: StudioContext | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    todos: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)
    turn_count: int = 0
    #: One-line natural-language reflections from the grounded verify-and-retry
    #: gate (Reflexion-style): each is a verifier's reason for rejecting a tool
    #: result, fed back to the model before a single retry. Append-only.
    reflections: list[str] = field(default_factory=list)
    #: Append-only provenance log (rule §3.13); one entry per tool call.
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
