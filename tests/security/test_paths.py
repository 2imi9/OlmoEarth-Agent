# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the workspace path-traversal guard (security/paths.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from olmoearth_agent.security.paths import (
    OUTPUT_ROOT_ENV,
    PathTraversalError,
    safe_path,
    workspace_root,
)


def test_relative_path_joined_under_root(tmp_path: Path) -> None:
    out = safe_path("sub/out.json", root=tmp_path)
    assert out == (tmp_path / "sub" / "out.json").resolve()
    assert tmp_path.resolve() in out.parents


def test_absolute_path_inside_root_allowed(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    assert safe_path(str(target), root=tmp_path) == target.resolve()


def test_root_itself_allowed(tmp_path: Path) -> None:
    assert safe_path(str(tmp_path), root=tmp_path) == tmp_path.resolve()


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        safe_path("../escape.json", root=tmp_path)


def test_deep_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        safe_path("a/b/../../../../etc/passwd", root=tmp_path)


def test_absolute_outside_root_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "sibling" / "x.json"
    with pytest.raises(PathTraversalError):
        safe_path(str(outside), root=tmp_path)


def test_empty_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathTraversalError):
        safe_path("   ", root=tmp_path)


def test_workspace_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path))
    assert workspace_root() == tmp_path.resolve()
    # safe_path with no explicit root uses the env workspace.
    assert safe_path("f.json") == (tmp_path / "f.json").resolve()


def test_default_root_is_a_subdir_not_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no env override the workspace is a dedicated subdir, never the bare
    # CWD/repo (so a relative `webui/js/x.js` can't clobber served code).
    monkeypatch.delenv(OUTPUT_ROOT_ENV, raising=False)
    root = workspace_root()
    assert root.name == "olmoearth_outputs"
    assert root != Path.cwd().resolve()
