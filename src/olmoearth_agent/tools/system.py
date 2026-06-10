# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The ``olmoearth_run_python`` sandbox tool (PLAN.md §1 ``system.python``).

Lets the agent run a short Python snippet for LIGHT orchestration/inspection
between API calls: open an rslearn ``Dataset``, tag a train/val split, compute
a quick stat, drive an embeddings step (e.g. skill #3's notebook glue).

This is ARBITRARY CODE EXECUTION, so it is **opt-in**: the bundle is empty
unless ``OLMOEARTH_RUN_PYTHON`` is truthy. Even then it is a deliberately small
first cut: an isolated subprocess of the project interpreter, with a
wall-clock timeout and an output cap, in a throwaway working directory. It is
NOT the PLAN.md in-process no-import sandbox: state does not persist across
calls and there is no import ban, and the heavy geospatial/rslearn/GDAL stack
is not guaranteed installed. So the agent should use it for light work and
SURFACE the long ``rslearn ingest/materialize/model fit`` jobs for the user to
run (see the ``olmoearth-rslearn`` skill).

The subprocess runs with a **credential-scrubbed environment** (the agent's
Studio/LLM keys are removed from ``os.environ`` before the snippet sees it), so
executed code cannot trivially read and exfiltrate them. This is
defence-in-depth informed by NemoClaw's sandbox-hardening posture, NOT a
sandbox: the subprocess is still NOT network-isolated and inherits the rest of
the environment. See ``docs/nemoclaw-assessment.md``.
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

#: Env-var name fragments / prefixes that mark a credential. Matched
#: case-insensitively; any match is dropped from the subprocess environment so
#: opt-in executed code cannot read the agent's own Studio/LLM/cloud keys.
_SECRET_FRAGMENTS = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
)
_SECRET_PREFIXES = (
    "OLMOEARTH_",
    "LLM_",
    "AWS_",
    "ANTHROPIC_",
    "OPENAI_",
    "GEMINI_",
    "HF_",
)


def _enabled() -> bool:
    """True only if the operator opted into code execution."""
    return os.environ.get("OLMOEARTH_RUN_PYTHON", "").strip().lower() in _TRUTHY


def _scrubbed_env() -> dict[str, str]:
    """A copy of ``os.environ`` with known-secret variables removed.

    Keeps OS-essential variables (PATH, SystemRoot, TEMP, ...) so the
    interpreter still launches on every platform, but drops the agent's own
    credentials. Defence-in-depth, not a complete allowlist sandbox: a snippet
    still inherits non-secret env and is not network-isolated.
    """
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith(_SECRET_PREFIXES):
            continue
        if any(fragment in upper for fragment in _SECRET_FRAGMENTS):
            continue
        env[name] = value
    return env


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
        # A tool-input error -> raise so dispatch reports ok=False. (A returned
        # dict is wrapped as {"ok": True, "result": ...}, which would mask the
        # failure as a successful call in provenance + the agent loop.)
        raise ValueError("no code provided")
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
            env=_scrubbed_env(),  # drop the agent's Studio/LLM keys from the child
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
                    "train/val split, compute a quick statistic, drive an "
                    "embeddings step. Runs in an isolated subprocess "
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
