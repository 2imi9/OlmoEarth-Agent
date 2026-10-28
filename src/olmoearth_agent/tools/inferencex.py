# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Lazy access to the optional companion package ``olmoearth-inferencex``.

The design-based estimation tools (``olmoearth_plan_label_sample``,
``olmoearth_estimate_map_error``, ``olmoearth_certify_zone``) and the dates
reading of ``olmoearth_compare_review`` call ``oe_inferencex`` (>= 1.20.1), which
brings numpy. The core package has no numpy, so the package is an optional
extra, ``pip install 'olmoearth-agent[inferencex]'``, imported only when a tool
needs it. When it is missing, or older than :data:`MIN_VERSION`, the tool
answers with :func:`missing_extra` instead of raising.
"""

from __future__ import annotations

import importlib
import re
from importlib import metadata
from types import ModuleType
from typing import Any

#: The install line a missing-extra answer names.
INSTALL_HINT = "pip install 'olmoearth-agent[inferencex]'"

#: The install line when an older companion package is installed.
UPGRADE_HINT = "pip install -U 'olmoearth-agent[inferencex]'"

#: The companion package's distribution name on PyPI.
DISTRIBUTION = "olmoearth-inferencex"

#: The oldest companion release the tools are written for: 1.19.0 moved the
#: modules they load into feature folders (``review.assess``,
#: ``common.raster``, ``certify.zone``), and 1.20.1 is the release the agent is
#: checked against.
MIN_VERSION = "1.20.1"


class InferencexMissingError(ImportError):
    """The optional ``olmoearth-inferencex`` package is not importable."""


def load(module: str) -> ModuleType:
    """Import ``oe_inferencex.<module>``, or raise :class:`InferencexMissingError`.

    Any ``ImportError`` (the package absent, a release older than
    :data:`MIN_VERSION` without the module, or numpy absent under it) is
    reported the same way; :func:`missing_extra` tells an older release apart
    by its version.
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


def _release(text: str) -> tuple[int, ...]:
    """A version's leading release numbers: ``"1.20.1.post1"`` -> ``(1, 20, 1)``."""
    numbers: list[int] = []
    for part in text.split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        numbers.append(int(digits.group()))
        if digits.end() < len(part):
            break
    return tuple(numbers)


def older_version() -> str | None:
    """The installed version when it is older than :data:`MIN_VERSION`, else ``None``.

    ``None`` too when the package is not installed at all. Versions compare by
    their release numbers, so 1.3.1 is older than 1.20.1.
    """
    installed = version()
    if installed is None or _release(installed) >= _release(MIN_VERSION):
        return None
    return installed


def unavailable() -> tuple[str, str]:
    """Why the package cannot be used here, as a clause, and the line that fixes it.

    A release older than :data:`MIN_VERSION` lacks the modules the tools load,
    so its import fails as an absent package's does; the clause then names the
    installed version and the line upgrades it, instead of saying the package
    is not installed.
    """
    old = older_version()
    if old is not None:
        return f"and this environment has version {old}, which is older", UPGRADE_HINT
    return "which is not installed in this environment", INSTALL_HINT


def missing_extra(what: str) -> dict[str, Any]:
    """The answer a tool gives when the extra is not installed, or is too old."""
    status, install = unavailable()
    return {
        "available": False,
        "reason": (
            f"{what} needs the optional package olmoearth-inferencex "
            f"(>= {MIN_VERSION}), {status}. Nothing was computed; "
            "do not substitute a hand-made formula (a simple-random-sample "
            "interval does not apply to a stratified or targeted design)."
        ),
        "install": install,
    }
