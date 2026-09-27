# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Lazy access to the optional companion package ``olmoearth-inferencex``.

The design-based estimation tools (``olmoearth_plan_label_sample``,
``olmoearth_estimate_map_error``, ``olmoearth_certify_zone``) and the dates
reading of ``olmoearth_compare_review`` call ``oe_inferencex`` (>= 1.3.0), which
brings numpy. The core package has no numpy, so the package is an optional
extra, ``pip install 'olmoearth-agent[inferencex]'``, imported only when a tool
needs it. When it is missing the tool answers with :func:`missing_extra`
instead of raising.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from types import ModuleType
from typing import Any

#: The install line a missing-extra answer names.
INSTALL_HINT = "pip install 'olmoearth-agent[inferencex]'"

#: The companion package's distribution name on PyPI.
DISTRIBUTION = "olmoearth-inferencex"


class InferencexMissingError(ImportError):
    """The optional ``olmoearth-inferencex`` package is not importable."""


def load(module: str) -> ModuleType:
    """Import ``oe_inferencex.<module>``, or raise :class:`InferencexMissingError`.

    Any ``ImportError`` (the package absent, or numpy absent under it) is
    reported the same way: the extra is not installed.
    """
    try:
        return importlib.import_module(f"oe_inferencex.{module}")
    except ImportError as exc:
        raise InferencexMissingError(str(exc)) from exc


def version() -> str | None:
    """The installed companion package's version, or ``None``."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def missing_extra(what: str) -> dict[str, Any]:
    """The answer a tool gives when the extra is not installed."""
    return {
        "available": False,
        "reason": (
            f"{what} needs the optional package olmoearth-inferencex (>= 1.3.0), "
            "which is not installed in this environment. Nothing was computed; "
            "do not substitute a hand-made formula (a simple-random-sample "
            "interval does not apply to a stratified or targeted design)."
        ),
        "install": INSTALL_HINT,
    }
