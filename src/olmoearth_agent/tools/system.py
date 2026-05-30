# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth_run_python`` sandbox tool (PLAN.md §1 ``system.python``).

Lets the agent run a short Python snippet for LIGHT orchestration/inspection
between API calls: open an rslearn ``Dataset``, tag a train/val split, compute
a quick stat, drive a FAISS/embeddings step (skills #4/#9 assume this tool).

This is ARBITRARY CODE EXECUTION, so it is **opt-in**: the bundle is empty
unless ``OLMOEARTH_RUN_PYTHON`` is truthy. Even then it is a deliberately small
first cut: an isolated subprocess of the project interpreter, with a
wall-clock timeout and an output cap, in a throwaway working directory. It is
NOT the PLAN.md in-process no-import sandbox: state does not persist across
calls and there is no import ban, and the heavy geospatial/rslearn/GDAL stack
is not guaranteed installed. So the agent should use it for light work and
SURFACE the long ``rslearn ingest/materialize/model fit`` jobs for the user to
run (see the ``olmoearth-rslearn`` skill).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from olmoearth_agent.llm.types import ToolSpec
from olmoearth_agent.tools.registry import RegisteredTool, ToolContext

#: Default wall-clock cap per call (seconds); override OLMOEARTH_RUN_PYTHON_TIMEOUT.
_DEFAULT_TIMEOUT = 30.0
#: Per-stream output cap (chars), so a noisy run can't flood the context window.
_MAX_OUTPUT = 8000
_TRUTHY = {"1", "true", "yes", "on"}


def _enabled() -> bool:
    """True only if the operator opted into code execution."""
    return os.environ.get("OLMOEARTH_RUN_PYTHON", "").strip().lower() in _TRUTHY


def _timeout() -> float:
    """Wall-clock cap per call (env-overridable)."""
    try:
        return float(os.environ.get("OLMOEARTH_RUN_PYTHON_TIMEOUT", _DEFAULT_TIMEOUT))
    except ValueError:
        return _DEFAULT_TIMEOUT


def _cap(raw: bytes) -> tuple[str, bool]:
    """Decode bytes and truncate to the output cap; flag if truncated."""
    text = raw.decode("utf-8", errors="replace")
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "\n…(truncated)", True
    return text, False


async def _run_python(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
    """Execute a Python snippet in an isolated, time-bounded subprocess."""
    code = str(args.get("code", ""))
    if not code.strip():
        return {"ok": False, "error": "no code provided"}
    timeout = _timeout()
    with tempfile.TemporaryDirectory(prefix="oe-run-") as workdir:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # isolated: ignore env/user-site, don't prepend the cwd
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "ok": False,
                "error": f"timed out after {timeout:g}s",
                "returncode": None,
            }
    stdout, out_capped = _cap(out)
    stderr, err_capped = _cap(err)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": out_capped or err_capped,
    }


def build_system_tools() -> list[RegisteredTool]:
    """Return the ``olmoearth_run_python`` bundle, empty unless opted in.

    Set ``OLMOEARTH_RUN_PYTHON=1`` to enable. Off by default because the tool
    runs unrestricted Python in a subprocess.
    """
    if not _enabled():
        return []
    return [
        RegisteredTool(
            spec=ToolSpec(
                name="olmoearth_run_python",
                description=(
                    "Run a short Python snippet; returns its stdout, stderr, "
                    "and exit code. Use for LIGHT orchestration/inspection "
                    "between API calls: open an rslearn Dataset, tag a "
                    "train/val split, compute a quick statistic, drive a "
                    "FAISS/embeddings step. Runs in an isolated subprocess "
                    "with a wall-clock timeout and an output cap, in a "
                    "throwaway working directory; STATE DOES NOT PERSIST "
                    "across calls. The heavy geospatial/rslearn/GDAL stack may "
                    "not be installed, and long jobs (rslearn ingest / "
                    "materialize / model fit) WILL time out. Surface those "
                    "for the user to run instead. print() what you want back."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python source to execute. print() the results "
                                "you want returned. Keep it short and fast."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            ),
            handler=_run_python,
        ),
    ]
