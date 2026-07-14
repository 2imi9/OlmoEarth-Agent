# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The agent's soul: the system prompt as a versioned markdown artifact.

Shippy-style agent anatomy separates the *soul* (persona + behavioral
boundaries) from code, so guardrails are auditable and revisable without a
code change. The packaged ``soul.md`` next to this module is the default;
operators can point ``OLMOEARTH_SOUL_PATH`` at a different markdown file to
swap the soul without touching the package (a config change, not a rebuild).

The file's whole text becomes the base system prompt; the harness then
appends run-specific clauses (skill index, forced skill, local-model budget)
on top — see :mod:`olmoearth_agent.harness.agent`.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Env var pointing at an alternative soul markdown file.
SOUL_PATH_ENV = "OLMOEARTH_SOUL_PATH"

#: The versioned soul shipped with the package.
PACKAGED_SOUL = Path(__file__).with_name("soul.md")


def soul_path() -> Path:
    """The soul file in effect: ``OLMOEARTH_SOUL_PATH`` if set, else packaged."""
    raw = os.environ.get(SOUL_PATH_ENV, "").strip()
    return Path(raw) if raw else PACKAGED_SOUL


def load_soul() -> str:
    """Read the soul markdown (stripped), falling back to the packaged file.

    A broken ``OLMOEARTH_SOUL_PATH`` (missing/unreadable file) must not turn
    into an agent with no behavioral boundaries, so it falls back to the
    packaged soul rather than raising or returning an empty prompt.
    """
    path = soul_path()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if not text and path != PACKAGED_SOUL:
        text = PACKAGED_SOUL.read_text(encoding="utf-8").strip()
    return text
