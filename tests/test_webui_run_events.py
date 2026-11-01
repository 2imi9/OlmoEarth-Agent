# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The web UI reads every action a ``check`` event carries.

The claim check (``harness/claim_check.py``) emits a ``check`` event on
each call, flags or not. ``webui/js/run.js`` showed any action but
``marked`` as "asked for a rewrite", so a claim check that passed, or failed
open, would have been shown as a rewrite that never happened.
"""

from __future__ import annotations

from pathlib import Path

from olmoearth_agent.harness.answer_checks import CLAIM_ACTIONS

_RUN_JS = Path(__file__).resolve().parents[1] / "webui" / "js" / "run.js"


def test_run_js_names_every_action_but_the_rewrite() -> None:
    source = _RUN_JS.read_text(encoding="utf-8")
    shown_apart = [a for a in (*CLAIM_ACTIONS, "appended") if a != "revise"]
    for action in shown_apart:
        assert f"ev.action === '{action}'" in source, action
