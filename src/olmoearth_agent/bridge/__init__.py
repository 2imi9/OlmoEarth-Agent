# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The replay bridge: a Claude Code session as the agent's model.

The role Qwen plays behind ``OlmoEarthLLM`` is taken by whoever writes the
responses in a run directory, one model call per command, with the agent's
real loop, tools and answer checks (``LeadAgent.run_stream``, unchanged)
and no process, allocation or connection held between calls. Each command
(``olmoearth-agent-bridge``, :mod:`olmoearth_agent.bridge.cli`) starts a
fresh agent that replays the run from its start: the recorded responses
through :class:`~olmoearth_agent.bridge.scripted.ScriptedLLM`, the tool
calls from the run's cache (:mod:`olmoearth_agent.bridge.replay`), then the
response just given, and stops at the next model call or at the answer
(:mod:`olmoearth_agent.bridge.session`). See ``docs/CLAUDE_AS_MODEL.md``.

This is not the web UI's bridge (``serve.py``, ``make bridge``), which
serves the agent to a browser.
"""

from olmoearth_agent.bridge.errors import BridgeError
from olmoearth_agent.bridge.session import begin, show, step

__all__ = ["BridgeError", "begin", "show", "step"]
