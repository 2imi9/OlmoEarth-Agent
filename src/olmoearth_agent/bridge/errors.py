# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The bridge's two errors: one for its commands, one that stops a replay."""

from __future__ import annotations


class BridgeError(RuntimeError):
    """A bridge command cannot go on: no run, a bad response, a diverged replay.

    The command line prints it as ``{"status": "error", "error": ...}`` and
    exits 1. A step that fails saves no response and leaves ``pending.json``
    as it was; tool calls it already ran stay in the cache as provisional
    records, which the same response sent again reuses and another response
    drops (``replay.ToolCache``).
    """


class BridgeHalt(BaseException):
    """Stops a replayed loop where it stands.

    A ``BaseException``, so that it passes the ``except Exception`` of the
    registry, which would turn it into a failed tool envelope and let the loop
    run the turn's next tool call live, and of the claim check, which fails
    open. The step catches it and raises :class:`BridgeError`.
    """
