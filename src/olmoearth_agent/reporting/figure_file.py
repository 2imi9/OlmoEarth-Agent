# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Reading a determination figure a tool wrote, for a surface that shows it.

A figure tool (:mod:`olmoearth_agent.reporting.figures`) returns its SVG's
path as ``figure_path``. That path comes from a tool result, so the web
bridge (``serve.py``) and the MCP App (``mcp_app``) read it only through
:func:`read_figure_svg`: a regular file named ``*_figure.svg`` (the only
files the figure tools write) whose real path lies under the tools' write
root or the workspace root, of at most :data:`FIGURE_MAX_BYTES`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from olmoearth_agent.security.paths import workspace_root

#: The largest figure (bytes) a surface reads and inlines.
FIGURE_MAX_BYTES = 2 * 1024 * 1024


def read_figure_svg(path: Any) -> str | None:
    """The SVG text at ``path``, or ``None`` unless it is a figure a tool wrote.

    Anything else (no path, a path elsewhere or that cannot be resolved, a
    NUL byte, another file, a directory or a pipe, an unreadable, too large
    or non-UTF-8 file) gives ``None``: the surface shows no figure.
    """
    # The tools' write root lives beside the review-set tools, which import
    # the figure module; imported here, at call time, to keep imports acyclic.
    from olmoearth_agent.tools.review_set import write_root

    if not isinstance(path, str) or not path:
        return None
    try:
        real = Path(os.path.realpath(path))
        roots = (write_root(), workspace_root())
        inside = any(real.is_relative_to(os.path.realpath(root)) for root in roots)
        if not inside or not real.name.endswith("_figure.svg"):
            return None
        if not real.is_file():  # never open a directory or a pipe
            return None
        with real.open("rb") as fh:
            data = fh.read(FIGURE_MAX_BYTES + 1)
        if len(data) > FIGURE_MAX_BYTES:
            return None
        return data.decode("utf-8")
    except (OSError, ValueError, UnicodeDecodeError):
        return None
