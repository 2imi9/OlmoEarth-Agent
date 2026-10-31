# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Lazy access to the optional companion package ``olmoearth-inferencex``.

The design-based estimation tools (``olmoearth_plan_label_sample``,
``olmoearth_estimate_map_error``, ``olmoearth_certify_zone``) and the dates
reading of ``olmoearth_compare_review`` call ``oe_inferencex`` (>= 1.20.3), which
brings numpy. The core package has no numpy, so the package is an optional
extra, ``pip install 'olmoearth-agent[inferencex]'``, imported only when a tool
needs it. When it is missing, or older than :data:`MIN_VERSION` (:func:`load`
refuses an older release even when its modules import), the tool answers with
:func:`missing_extra` instead of raising.

A certificate rests on files the package's command line writes (a sample
drawn with ``--seed-from-map`` and its sidecar, then the zone ``certify``
writes), so the certificate route also runs ``oe-inferencex`` itself:
:func:`run_cli` runs one command with this interpreter (``-P``: the working
folder never on its import path), never a shell, without the agent's keys in
its environment, under a time limit, and turns any failure into a refusal;
:func:`run_cli_async` runs it off the event loop.
"""

from __future__ import annotations

import asyncio
import importlib
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from olmoearth_agent.security.child_env import scrubbed_env

#: The install line a missing-extra answer names.
INSTALL_HINT = "pip install 'olmoearth-agent[inferencex]'"

#: The install line when an older companion package is installed.
UPGRADE_HINT = "pip install -U 'olmoearth-agent[inferencex]'"

#: The companion package's distribution name on PyPI.
DISTRIBUTION = "olmoearth-inferencex"

#: The oldest companion release the tools are written for. 1.19.0 moved the
#: modules they load into feature folders (``review.assess``,
#: ``common.raster``, ``certify.zone``), so an older release fails their
#: import. 1.20.3 changed ``certify.certificate``: a certificate's zone mask
#: may carry GDAL's ModelTransformation matrix, so a map that is south up,
#: rotated or has no georeferencing gets a certificate (1.17.0 to 1.20.2
#: refuse that mask: they write no such certificate and fail an honest one),
#: and verify refuses a mask with a negative PixelScale y step (which passed
#: through 1.20.2 while libgeotiff mirrors the zone), a perspective term, a
#: second tiepoint or a tag given twice. 1.20.2 changed only the
#: documentation. 1.19.0 to 1.20.2 hold every module the tools load, so
#: :func:`load` checks the installed version as well.
MIN_VERSION = "1.20.3"

#: The package's command line, run as a module of this interpreter: the
#: package has no ``oe_inferencex/__main__.py``, and the console script
#: ``oe-inferencex`` is ``oe_inferencex.cli:main``.
CLI_MODULE = "oe_inferencex.cli"

#: Seconds one command of the package's command line may run before it is
#: stopped and the call refused.
CLI_TIMEOUT = 600.0


class InferencexMissingError(ImportError):
    """The optional ``olmoearth-inferencex`` package is not importable, or too old."""


def load(module: str) -> ModuleType:
    """Import ``oe_inferencex.<module>``, or raise :class:`InferencexMissingError`.

    Any ``ImportError`` (the package absent, a release older than
    :data:`MIN_VERSION` without the module, or numpy absent under it) is
    reported the same way. So is an installed release older than
    :data:`MIN_VERSION` whose module imports (1.19.0 to 1.20.2 hold every
    module the tools load): run on it, a tool would use that release's
    certificate code without a word. Every tool loads the package through
    here, so each refuses such a release as it refuses an absent one;
    :func:`missing_extra` tells an older release apart by its version.
    """
    try:
        found = importlib.import_module(f"oe_inferencex.{module}")
    except ImportError as exc:
        raise InferencexMissingError(str(exc)) from exc
    old = older_version()
    if old is not None:
        # No package name: a caller that reports the message prefixes it, as
        # it does an import error's.
        msg = f"version {old} is installed; the tools need >= {MIN_VERSION}"
        raise InferencexMissingError(msg)
    return found


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
    their release numbers, so 1.3.1 is older than 1.20.3, and so is 1.20.2.
    """
    installed = version()
    if installed is None or _release(installed) >= _release(MIN_VERSION):
        return None
    return installed


def unavailable() -> tuple[str, str]:
    """Why the package cannot be used here, as a clause, and the line that fixes it.

    A release older than :data:`MIN_VERSION` lacks the modules the tools load
    (before 1.19.0), so its import fails as an absent package's does, or
    :func:`load` refuses it (1.19.0 to 1.20.2); the clause then names the
    installed version and the line upgrades it, instead of saying the package
    is not installed.
    """
    old = older_version()
    if old is not None:
        return f"and this environment has version {old}, which is older", UPGRADE_HINT
    return "which is not installed in this environment", INSTALL_HINT


def run_cli(args: list[str], *, cwd: Path | str, timeout: float = CLI_TIMEOUT) -> str:
    """Run ``oe-inferencex <args>`` in ``cwd``; its standard output, or a refusal.

    The command runs as ``[sys.executable, "-P", "-m", "oe_inferencex.cli",
    *args]`` with no shell, in ``cwd`` (the run's working folder, never the
    package), under ``timeout`` seconds. ``-P`` keeps ``cwd`` off the child's
    ``sys.path``: ``cwd`` is a data folder under the scores root, so a
    ``json.py`` or ``numpy.py`` left there is never imported. The child gets
    the agent's environment without its Studio, LLM and cloud keys
    (:func:`~olmoearth_agent.security.child_env.scrubbed_env`; the package
    reads no environment variable). The package's command line ends a
    refusal with ``SystemExit``, a message on standard error and a non-zero
    status; that, a time-out, or a command that cannot start is raised here
    as ``ValueError`` with the command's own last line, so the registry
    answers with it and the agent keeps running. A tool handler awaits
    :func:`run_cli_async`, which runs this in a worker thread so the event
    loop (the web bridge's other sessions) keeps running.

    Parameters
    ----------
    args
        The subcommand and its arguments, e.g. ``["certify", "sample.csv"]``.
    cwd
        The directory the command runs in.
    timeout
        Seconds before the command is stopped (:data:`CLI_TIMEOUT`).

    Returns
    -------
    str
        What the command printed on standard output.

    Raises
    ------
    ValueError
        When the command exits with a non-zero status, runs past ``timeout``
        or cannot be started.

    Examples
    --------
    >>> run_cli(["certify", "sample.csv"], cwd="work")  # doctest: +SKIP
    """
    command = [sys.executable, "-P", "-m", CLI_MODULE, *args]
    name = f"oe-inferencex {args[0]}" if args else "oe-inferencex"
    try:
        done = subprocess.run(  # noqa: S603 - this interpreter, no shell
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=scrubbed_env(),  # the agent's Studio/LLM keys stay out of the child
        )
    except subprocess.TimeoutExpired:
        msg = f"{name} did not finish within {timeout:g} s and was stopped"
        raise ValueError(msg) from None
    except OSError as exc:
        raise ValueError(f"{name} could not be started: {exc}") from None
    if done.returncode != 0:
        text = done.stderr or done.stdout or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        said = lines[-1] if lines else "it printed nothing"
        msg = f"{name} refused (exit status {done.returncode}): {said}"
        raise ValueError(msg)
    return done.stdout or ""


async def run_cli_async(
    args: list[str], *, cwd: Path | str, timeout: float = CLI_TIMEOUT
) -> str:
    """:func:`run_cli` in a worker thread, for an ``async`` tool handler.

    A command may run for up to ``timeout`` seconds; run inline, it would
    hold the event loop that serves every other session of the web bridge.

    Parameters
    ----------
    args, cwd, timeout
        As :func:`run_cli` takes them.

    Returns
    -------
    str
        What the command printed on standard output.

    Examples
    --------
    >>> await run_cli_async(["certify", "sample.csv"], cwd="work")  # doctest: +SKIP
    """
    return await asyncio.to_thread(run_cli, args, cwd=cwd, timeout=timeout)


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
