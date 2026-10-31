# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tool handlers served from a run's cache when the loop is replayed.

Every bridge step replays the run from its start in a fresh process, so each
tool call the model made earlier is made again. Those calls must not reach
Studio again: :class:`ToolCache` wraps each ``RegisteredTool``'s handler
(not ``ToolRegistry.dispatch``), so the registry's argument validation,
result envelopes and deferred-group loading, and the loop's retry hint (with
its failure counts), spill, records and provenance, all run as live, around
a handler that returns what it returned the first time.

The k-th handler invocation of a replay is served from the cache's k-th
record when there is one, after checking it is the same call (same tool,
same arguments; else "the run diverged at call k"). Past the cache, the
handler runs and its outcome is appended, before the loop goes on. A handler
that raised is replayed by raising an exception with the same class name,
text and ``contract``, which is all the registry builds the failed envelope
from (``ToolRegistry._execute``).

Each record names the model call whose response made it. A step saves the
response it was given only once it has stopped cleanly, so a step that
failed after a tool call ran (a killed process, the srun time limit, a
bridge stop) leaves records no saved response made. Those are provisional:
the next step serves them while it makes the same calls, and drops them from
the first call it makes otherwise, or at its end if it makes fewer. A
dropped call did run: a Studio write among them may already have happened,
which the step logs and its view states.

Results are kept as JSON. The first, live call already gets its result back
through JSON, so every step sees the same value (a tuple comes back a list;
the text the model reads, ``json.dumps`` of the envelope, is unchanged).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from olmoearth_agent.bridge.errors import BridgeHalt
from olmoearth_agent.tools.registry import Handler, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from olmoearth_agent.bridge.store import RunDir

logger = logging.getLogger(__name__)

__all__ = [
    "RERUN_TOOLS",
    "ReplayedToolError",
    "ToolCache",
    "replay_registry",
    "replayed_error",
]

#: The handlers that run again on every replay instead of being served from
#: the cache: their job is the run's own state, which a fresh process does not
#: have. Of the default registry's handlers, these change ``ctx.state`` (read
#: at d1b4548):
#:
#: - ``olmoearth_load_skill`` (``tools/skill_tools.py``) adds a deferred
#:   group to ``loaded_groups``, which decides the tools offered and the
#:   capability card from the next turn on. It reads only the packaged
#:   SKILL.md files, so it runs again; a result other than the cached one
#:   stops the replay.
#: - ``olmoearth_load_context`` (``tools/studio.py``) sets ``studio_context``
#:   and ``olmoearth_submit_prediction`` (``tools/predict.py``) appends to
#:   ``prediction_ids``. Both call Studio, so they are served from the cache
#:   and the two fields stay unset on replay; no code reads either field.
#:
#: The registry itself (a deferred tool's group, ``tool_failures``) and the
#: loop (``provenance``, ``turn_count``) change the rest, and run as live.
#: ``olmoearth_provenance_summary`` and ``olmoearth_case_narrative`` read the
#: provenance, whose timestamps a replay renews: served from the cache, they
#: return what the model read.
RERUN_TOOLS = frozenset({"olmoearth_load_skill"})


class ReplayedToolError(Exception):
    """A handler's exception as the cache recorded it, raised again on replay.

    Each is an instance of a subclass named as the original class, with the
    original text and ``contract``, so ``type(exc).__name__``, ``str(exc)``
    and ``exc.contract`` give the registry the envelope the live call gave.
    """

    contract: dict[str, Any] | None = None


_ERROR_CLASSES: dict[str, type[ReplayedToolError]] = {}


def replayed_error(error: dict[str, Any]) -> ReplayedToolError:
    """The exception to raise for a cached ``{"type", "message", "contract"}``.

    Examples
    --------
    >>> exc = replayed_error({"type": "ValueError", "message": "bad budget"})
    >>> f"{type(exc).__name__}: {exc}"
    'ValueError: bad budget'
    """
    name = str(error.get("type") or "Exception")
    cls = _ERROR_CLASSES.get(name)
    if cls is None:
        cls = cast("type[ReplayedToolError]", type(name, (ReplayedToolError,), {}))
        _ERROR_CLASSES[name] = cls
    exc = cls(str(error.get("message", "")))
    contract = error.get("contract")
    if isinstance(contract, dict):
        exc.contract = contract
    return exc


def _as_json(value: Any, what: str) -> Any:
    """``value`` through JSON, as the cache keeps it and every step reads it."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        # The live loop would fail too: it serializes every result for the
        # model (harness/spill.py).
        raise BridgeHalt(f"{what} is not JSON: {exc}") from exc


def _outcome(record: dict[str, Any]) -> dict[str, Any]:
    """The part of a cache record a re-run must reproduce: the result or error."""
    return {k: record[k] for k in ("result", "error") if k in record}


def _serve(record: dict[str, Any]) -> Any:
    """Return a record's result, or raise its error, as the handler did."""
    error = record.get("error")
    if isinstance(error, dict):
        raise replayed_error(error)
    return copy.deepcopy(record.get("result"))


def _model_call(record: dict[str, Any]) -> int:
    """The model call whose response made a record's tool call."""
    return int(record.get("model_call") or 0)


def _same_call(record: dict[str, Any], name: str, arguments: Any) -> bool:
    """Whether a record is of the call to ``name`` with ``arguments``."""
    return bool(record.get("name") == name and record.get("arguments") == arguments)


class ToolCache:
    """A run's handler outcomes in call order (``tool_cache.jsonl``).

    Parameters
    ----------
    run
        The run directory; its cache is read now, and each new outcome is
        appended to it as soon as the handler returns.
    saved
        How many model responses the run has saved. The records their tool
        calls made are the run's; any after them are provisional (see the
        module docstring).
    model_call
        The model call whose response this step plays: every tool call that
        runs in this step is one of its calls, and its record says so.

    Attributes
    ----------
    records
        The outcomes, in call order: the cached ones, then this step's.
    cached
        How many records the cache held when this step began.
    committed
        How many of those the saved responses made.
    position
        Handler invocations so far in this step (served, re-run or new).
    dropped
        ``{"call", "name"}`` of each provisional record this step dropped.
    """

    def __init__(self, run: RunDir, *, saved: int, model_call: int) -> None:
        self.run = run
        self.model_call = model_call
        self.records: list[dict[str, Any]] = run.read_tool_cache()
        self.cached = len(self.records)
        self.committed = next(
            (i for i, r in enumerate(self.records) if _model_call(r) > saved),
            self.cached,
        )
        self.position = 0
        self.dropped: list[dict[str, Any]] = []

    def wrap(self, name: str, handler: Handler) -> Handler:
        """``handler`` of the tool ``name``, served from the cache when it can be."""

        async def replayed(args: dict[str, Any], ctx: ToolContext) -> Any:
            self.position += 1
            k = self.position
            arguments = _as_json(args, f"the arguments of call {k} ({name})")
            if k <= len(self.records):
                cached = self.records[k - 1]
                if _same_call(cached, name, arguments):
                    if name not in RERUN_TOOLS:
                        return _serve(cached)
                    fresh = await self._call(k, name, arguments, handler, args, ctx)
                    if _outcome(fresh) == _outcome(cached):
                        return _serve(fresh)
                    if k <= self.committed:
                        raise BridgeHalt(
                            f"the run diverged at call {k}: {name} runs again on "
                            "each step and returned another result than the "
                            "cached one"
                        )
                    self._drop(k)
                    return self._keep(fresh)
                if k <= self.committed:
                    raise BridgeHalt(
                        f"the run diverged at call {k}: the loop called {name} "
                        f"with {json.dumps(arguments)[:200]}, the cache holds "
                        f"{cached.get('name')} with "
                        f"{json.dumps(cached.get('arguments'))[:200]}"
                    )
                # A provisional record of another call: this response is not
                # the one the failed step played.
                self._drop(k)
            return self._keep(await self._call(k, name, arguments, handler, args, ctx))

        return replayed

    def drop_unread(self) -> None:
        """Drop the provisional records the step's replay did not reach.

        Called once the loop has stopped: the failed step's response made
        more tool calls than the one this step played.
        """
        if self.committed <= self.position < len(self.records):
            self._drop(self.position + 1)

    def _keep(self, record: dict[str, Any]) -> Any:
        """Append a new outcome to the cache, then serve it."""
        self.run.append_tool_record(record)
        self.records.append(record)
        return _serve(record)

    def _drop(self, k: int) -> None:
        """Drop the provisional records from call ``k`` on, in the file too."""
        start = k - 1
        gone = self.records[start:]
        del self.records[start:]
        self.run.write_tool_cache(self.records)
        self.dropped += [{"call": r.get("call"), "name": r.get("name")} for r in gone]
        logger.warning(
            "bridge: dropped %d tool call(s) (%s) that a step made for model "
            "call %d before it failed; its response was not saved, and a Studio "
            "write among them may already have happened",
            len(gone),
            ", ".join(str(r.get("name")) for r in gone),
            self.model_call,
        )

    async def _call(
        self,
        k: int,
        name: str,
        arguments: Any,
        handler: Handler,
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> dict[str, Any]:
        """Run ``handler`` once and give its outcome as a cache record."""
        record: dict[str, Any] = {
            "call": k,
            "model_call": self.model_call,
            "name": name,
            "arguments": arguments,
        }
        if name in RERUN_TOOLS:
            record["rerun"] = True
        try:
            value = await handler(args, ctx)
        except Exception as exc:  # noqa: BLE001 - the registry makes it the envelope
            contract = getattr(exc, "contract", None)
            if isinstance(contract, dict):
                contract = _as_json(contract, f"the contract of call {k} ({name})")
            else:
                contract = None
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "contract": contract,
            }
            return record
        record["result"] = _as_json(value, f"the result of call {k} ({name})")
        return record


def replay_registry(registry: ToolRegistry, cache: ToolCache) -> ToolRegistry:
    """A copy of ``registry`` whose handlers go through ``cache``.

    The same tools in the same order, with the same specs, capabilities and
    deferred groups, so the tools offered and the capability card are the
    live ones.
    """
    replayed = ToolRegistry()
    for name in registry.names():
        tool = registry.tool_of(name)
        if tool is None:  # pragma: no cover - names() lists registered tools only
            continue
        replayed.register(
            replace(tool, handler=cache.wrap(name, tool.handler)),
            group=registry.group_of(name),
        )
    return replayed
