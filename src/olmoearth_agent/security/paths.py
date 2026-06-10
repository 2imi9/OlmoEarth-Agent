# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Confine model-controlled tool file I/O to a workspace root (path-traversal guard).

Tool handlers receive file-path arguments straight from the model
(``output_path``, ``out_dir``, ``positives_path``, ...). The LLM is a
semi-trusted component, so a prompt-injection payload (carried in a Studio
project name, a fetched abstract, or any tool output the model reads) can steer
it to a traversing (``../../``) or absolute path -- yielding arbitrary-location
file read/write under the operator's account: clobbering served ``webui/js``
modules, planting a startup script, or reflecting a JSON-ish secrets file back
into chat.

:func:`safe_path` resolves every such path under a single workspace root and
rejects anything that escapes it. The *containment check on the resolved path*
is the real guard -- a ``..`` escape, an absolute path elsewhere, a symlink
target outside the root, or a repo-relative ``webui/js/x.js`` clobber are all
refused, on POSIX and Windows alike.

The root is ``OLMOEARTH_OUTPUT_ROOT`` if set, else ``<cwd>/olmoearth_outputs``.
It is deliberately NOT the process CWD / repo, so a relative path lands
harmlessly inside the workspace instead of overwriting served code or sources.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Env var that relocates the workspace root (e.g. to a mounted data volume).
OUTPUT_ROOT_ENV = "OLMOEARTH_OUTPUT_ROOT"
_DEFAULT_SUBDIR = "olmoearth_outputs"


class PathTraversalError(ValueError):
    """Raised when a model-supplied path resolves outside the workspace root."""


def workspace_root() -> Path:
    """The directory all tool file I/O is confined to (resolved, absolute)."""
    raw = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    root = Path(raw) if raw else Path.cwd() / _DEFAULT_SUBDIR
    return root.resolve()


def safe_path(user_path: str, *, root: Path | None = None) -> Path:
    """Resolve ``user_path`` inside the workspace root, or raise.

    Relative paths are joined under the root; an absolute path is taken as-is.
    Either way the *resolved* path must sit inside the root -- a ``..`` escape,
    an absolute path outside the root, or a symlink target outside it are all
    rejected. No filesystem entry is created here; callers ``mkdir`` as needed.

    Raises
    ------
    PathTraversalError
        If ``user_path`` is empty or resolves outside the workspace root.
    """
    base = (root or workspace_root()).resolve()
    raw = str(user_path).strip()
    if not raw:
        raise PathTraversalError("empty path")
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise PathTraversalError(
            f"path {user_path!r} resolves outside the workspace root {base} "
            f"(set {OUTPUT_ROOT_ENV} to relocate the workspace, or pass a path "
            "inside it)"
        )
    return resolved
