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
from olmoearth_agent.tools.estimation import build_estimation_tools
from olmoearth_agent.tools.registry import ToolContext, ToolRegistry
from olmoearth_agent.tools.scores_file import build_scores_file_tools


async def _ok(name: str, args: dict[str, Any]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register_all(build_estimation_tools())
    registry.register_all(build_scores_file_tools())
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
