# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The optional companion package's loader and its missing-extra answers.

olmoearth-inferencex 1.19.0 moved the modules the tools load into feature
folders (``review.assess``, ``common.raster``, ``certify.zone``), so on an
older release their import fails as an absent package's does. The answer then
names the installed version and the upgrade line instead of saying the
package is not installed. These need no package: the failed import and the
installed version are simulated.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from olmoearth_agent.harness.state import ThreadState
from olmoearth_agent.llm.types import ToolCall
from olmoearth_agent.tools import inferencex
from olmoearth_agent.tools.certificate import build_certificate_tools
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_scores_file_tools())
    registry.register_all(build_certificate_tools())
    ctx = ToolContext(studio=None, state=ThreadState())  # type: ignore[arg-type]
    out = await registry.dispatch(ToolCall(id="1", name=name, arguments=args), ctx)
    assert out["ok"] is True, out
    return out["result"]


@pytest.mark.parametrize(
    ("installed", "older"),
    [
        (None, None),
        ("1.20.1", None),
        ("1.20.2", None),
        ("1.21.0", None),
        ("2.0", None),
        ("1.20.1.post1", None),
        ("1.20.0", "1.20.0"),
        ("1.20", "1.20"),
        ("1.19.0", "1.19.0"),
        ("1.18.1", "1.18.1"),
        ("1.3.1", "1.3.1"),  # by release numbers: as text, "1.3.1" > "1.20.1"
    ],
)
def test_an_older_release_is_told_apart_by_its_release_numbers(
    monkeypatch: pytest.MonkeyPatch, installed: str | None, older: str | None
) -> None:
    assert inferencex.MIN_VERSION == "1.20.1"
    monkeypatch.setattr(inferencex, "version", lambda: installed)
    assert inferencex.older_version() == older


def test_missing_extra_says_not_installed_only_when_no_older_release_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for installed in (None, "1.20.1"):  # absent, or numpy missing under it
        monkeypatch.setattr(inferencex, "version", lambda v=installed: v)
        absent = inferencex.missing_extra("Certifying a trusted zone")
        assert absent["available"] is False
        assert absent["reason"].startswith(
            "Certifying a trusted zone needs the optional package "
            "olmoearth-inferencex (>= 1.20.1), which is not installed in this "
            "environment. Nothing was computed;"
        )
        assert absent["install"] == "pip install 'olmoearth-agent[inferencex]'"
    monkeypatch.setattr(inferencex, "version", lambda: "1.18.1")
    old = inferencex.missing_extra("Certifying a trusted zone")
    assert old["available"] is False
    assert "not installed" not in old["reason"]
    assert (
        "olmoearth-inferencex (>= 1.20.1), and this environment has version "
        "1.18.1, which is older. Nothing was computed;" in old["reason"]
    )
    assert "do not substitute a hand-made formula" in old["reason"]
    assert old["install"] == "pip install -U 'olmoearth-agent[inferencex]'"
    assert old["install"] == inferencex.UPGRADE_HINT


@pytest.mark.asyncio
async def test_certify_zone_on_a_release_without_certify_zone_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """certify_zone loads ``oe_inferencex.certify.zone`` (for ISSUED_NOTE),
    which 1.18.1 does not have: the answer names 1.18.1, not 'not installed'."""
    monkeypatch.setitem(sys.modules, "oe_inferencex.certify.zone", None)
    monkeypatch.setattr(inferencex, "version", lambda: "1.18.1")
    out = await _ok("olmoearth_certify_zone", {"design_path": "x.json", "alpha": 0.05})
    assert out["available"] is False
    assert "version 1.18.1, which is older" in out["reason"]
    assert "not installed" not in out["reason"]
    assert out["install"] == inferencex.UPGRADE_HINT


@pytest.mark.asyncio
async def test_scores_from_file_on_a_release_without_review_assess_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scores_from_file loads ``oe_inferencex.review.assess``, which 1.18.1
    does not have (its ``assess`` was top-level): the answer names 1.18.1."""
    monkeypatch.setitem(sys.modules, "oe_inferencex.review.assess", None)
    monkeypatch.setattr(inferencex, "version", lambda: "1.18.1")
    out = await _ok("olmoearth_scores_from_file", {"run_dir": "anything"})
    assert out["available"] is False
    assert out["reason"].startswith(
        "Reading a model run's scores raster needs the optional package "
        "olmoearth-inferencex (>= 1.20.1) with rasterio (its geo extra), and "
        "this environment has version 1.18.1, which is older."
    )
    assert "not installed" not in out["reason"]
    assert "do not rank" in out["reason"]
    assert "olmoearth-inferencex" in out["missing"]
    assert out["install"] == inferencex.UPGRADE_HINT
    monkeypatch.setattr(inferencex, "version", lambda: None)
    absent = await _ok("olmoearth_scores_from_file", {"run_dir": "anything"})
    assert "which is not installed in this environment" in absent["reason"]
    assert absent["install"] == inferencex.INSTALL_HINT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args", "what"),
    [
        ("olmoearth_write_certificate", {"design_path": "x.json"}, "Writing"),
        ("olmoearth_verify_certificate", {"certificate_path": "x"}, "Checking"),
    ],
)
async def test_the_certificate_tools_without_the_package_say_so(
    tool: str, args: dict[str, Any], what: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both certificate tools load ``oe_inferencex.certify.certificate``
    (1.17.0 and later): without it they answer with the install line."""
    monkeypatch.setitem(sys.modules, "oe_inferencex.certify.certificate", None)
    monkeypatch.setattr(inferencex, "version", lambda: None)
    out = await _ok(tool, args)
    assert out["available"] is False
    assert out["reason"].startswith(f"{what} a certificate needs the optional")
    assert out["install"] == inferencex.INSTALL_HINT


def test_run_cli_runs_this_interpreter_with_no_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The package's command line as ``python -P -m oe_inferencex.cli``
    (``-P``: the working folder, a data folder, is not on the child's import
    path), under a time limit, in the folder given; a non-zero exit is a
    refusal in the command's own last words. The subprocess is replaced: no
    package needed."""
    import subprocess

    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> Any:
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(inferencex.subprocess, "run", run)
    assert inferencex.run_cli(["certify", "s.csv"], cwd=tmp_path) == "done\n"
    assert seen["command"] == [
        sys.executable,
        "-P",
        "-m",
        "oe_inferencex.cli",
        "certify",
        "s.csv",
    ]
    assert seen["cwd"] == str(tmp_path) and seen["timeout"] == inferencex.CLI_TIMEOUT
    assert seen["capture_output"] is True and seen["text"] is True
    assert "shell" not in seen and seen["check"] is False

    def refuse(command: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="\n\n")

    monkeypatch.setattr(inferencex.subprocess, "run", refuse)
    with pytest.raises(ValueError, match=r"exit status 2\): it printed nothing"):
        inferencex.run_cli(["certify", "s.csv"], cwd=tmp_path)

    def broken(command: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError("no interpreter")

    monkeypatch.setattr(inferencex.subprocess, "run", broken)
    with pytest.raises(ValueError, match="could not be started: no interpreter"):
        inferencex.run_cli(["certify", "s.csv"], cwd=tmp_path)


def test_run_cli_keeps_the_agents_keys_out_of_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The child gets the agent's environment without its Studio, LLM and
    cloud keys (``security.child_env``, the list ``olmoearth_run_python``
    uses), and keeps the rest: PATH, and GDAL's settings."""
    import subprocess

    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("OLMOEARTH_API_KEY", "studio-secret")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-secret")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "token-secret")
    monkeypatch.setenv("GDAL_CACHEMAX", "512")
    monkeypatch.setattr(inferencex.subprocess, "run", run)
    inferencex.run_cli(["certify", "s.csv"], cwd=tmp_path)
    env = seen["env"]
    for key in (
        "OLMOEARTH_API_KEY",
        "LLM_API_KEY",
        "ANTHROPIC_API_KEY",
        "SOME_SERVICE_TOKEN",
    ):
        assert key not in env, key
    assert env.get("GDAL_CACHEMAX") == "512" and env.get("PATH")


@pytest.mark.asyncio
async def test_run_cli_async_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A tool handler awaits the command in a worker thread: the loop that
    serves the web bridge's other sessions keeps running meanwhile, and a
    refusal reaches the handler as the same ValueError."""
    import asyncio
    import subprocess
    import threading

    loop_thread = threading.get_ident()
    ran_in: list[int] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        ran_in.append(threading.get_ident())
        return subprocess.CompletedProcess(command, 0, stdout="done\n", stderr="")

    monkeypatch.setattr(inferencex.subprocess, "run", run)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        ticks += 1

    out, _ = await asyncio.gather(
        inferencex.run_cli_async(["certify", "s.csv"], cwd=tmp_path), tick()
    )
    assert out == "done\n" and ticks == 1
    assert ran_in and ran_in[0] != loop_thread

    def refuse(command: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="no\n")

    monkeypatch.setattr(inferencex.subprocess, "run", refuse)
    with pytest.raises(ValueError, match=r"exit status 1\): no"):
        await inferencex.run_cli_async(["certify", "s.csv"], cwd=tmp_path)
