# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""A prediction id where a result id is asked: the prediction's one result is used.

A user (or the model) often has a prediction's id at hand where a tool asks
for one of its results' ids. :func:`resolving` wraps the handler of each tool
that takes a result id (applied where the tool is registered), so both
dispatch paths (the MCP App's ``call_agent_tool`` and the agent's own loop)
and the replay bridge, which wraps handlers too, see one handler call:

1. The handler gets a copy of the arguments, never the caller's dict. A
   value that is a UUID is written the API's way (lower case, with hyphens);
   any other value is stripped and refused, with no Studio call, if it holds
   a character no Studio id has (only letters, digits, ``-`` and ``_`` are
   taken), so no ``/``, ``?``, ``#`` or ``%`` reaches a request's path.
2. When the handler raises a 404 (``httpx.HTTPStatusError``) or Studio's
   "Studio API error" (an HTTP 200 carrying errors), or, with ``caught``,
   returns ``available: false`` with a reason holding "404 Not Found" (the
   pixel value catches its errors), each UUID-valued argument is asked of
   Studio as a prediction (``GET /predictions/{id}``). A 429 or 5xx is
   retried by the client and never starts this check.
3. A prediction with one result: that result is put in its place and the
   handler runs once more, never again; the result says so in
   ``resolved_ids`` (and ``prediction_id``, and a card row when it has a
   card) and names the result id it ran on under the argument's own name
   (``result_id``, where the tool's result did not set it), so the
   provenance summary repeats the call with that result id. The card row
   comes after the tool's own rows (at most
   :data:`~olmoearth_agent.tools.statistical_rules.CARD_MAX_ROWS`), one per
   id resolved. A prediction with none or several is refused with a
   ``ValueError`` that lists them (at most :data:`MAX_LISTED`). No
   prediction among them: the first outcome stands, unchanged.

A refused key (:class:`~olmoearth_agent.studio.client.StudioAccessError`)
during the check is raised as it is, so the sign-in is still offered; any
other failure of the check leaves the first outcome as it was. What Studio
answers for ``GET /prediction-results/{prediction id}`` was not probed: if it
answers 200 with no record, nothing here starts and the tools behave as
before.
"""

from __future__ import annotations

import functools
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from olmoearth_agent.studio.client import StudioAccessError
from olmoearth_agent.tools.statistical_rules import CARD_KEY

if TYPE_CHECKING:
    from collections.abc import Callable

    from olmoearth_agent.tools.registry import Handler, ToolContext

__all__ = [
    "MAX_LISTED",
    "RESULT_ID_ARGS",
    "canonical_id",
    "resolving",
]

#: The attribute :func:`resolving` sets on the handler it returns: the names of
#: the arguments it resolves.
RESULT_ID_ARGS = "result_id_args"

#: Most results a refusal lists by id.
MAX_LISTED = 10

#: Results asked of Studio for one prediction.
SEARCH_LIMIT = 200

#: The characters a Studio id that is not a UUID may hold.
_ID_CHARS = re.compile(r"[A-Za-z0-9_-]*")

#: Said of a value with a character no id has; nothing was asked of Studio.
UNSAFE_ID = (
    "{argument} must be a Studio result id, a UUID (the API's format for it), "
    "or a prediction id: {value!r} holds a character no Studio id has (only "
    "letters, digits, '-' and '_'), so nothing was asked of Studio"
)

#: A prediction with several results: which one is the caller's to say.
SEVERAL_RESULTS = (
    "{argument} {prediction} is a prediction id, not a result id. Its {n} "
    "results: {listed}{more}. Pass one of them as {argument}: the one whose "
    "property the question is about, or ask the user which."
)

#: A prediction with no result yet.
NO_RESULTS = (
    "{argument} {prediction} is a prediction id, not a result id, and it has no "
    "results yet (status {status}). olmoearth_get_prediction gives its status."
)

#: The text Studio's client raises for an HTTP 200 that carries errors.
_API_ERROR = "Studio API error"

#: In a caught failure's reason: Studio answered the request with 404.
_CAUGHT_404 = "404 Not Found"


@dataclass(frozen=True)
class _Slot:
    """One argument value that holds a UUID: ``result_id`` or ``result_ids[1]``."""

    label: str
    name: str
    index: int | None
    value: str


@dataclass(frozen=True)
class _Swap:
    """A prediction id found in a slot, and its one result put in its place."""

    slot: _Slot
    result_id: str


class _CheckFailed(Exception):
    """The check could not tell: the call's first outcome stands as it was."""


def _as_uuid(text: str) -> str | None:
    """``text`` as the API writes a UUID (lower case, hyphens), or None."""
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return None


def canonical_id(value: Any, argument: str) -> Any:
    """One id argument as the handler gets it.

    A UUID is written the API's way; another string is stripped and refused
    when it holds a character no Studio id has; a value that is not a string
    is passed on as it is (the schema check, or the handler, answers it).

    Examples
    --------
    >>> canonical_id(" 0A0B0C0D-0000-4000-8000-00000000000F ", "result_id")
    '0a0b0c0d-0000-4000-8000-00000000000f'
    >>> canonical_id(" kelp-1 ", "result_id")
    'kelp-1'
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    as_uuid = _as_uuid(text)
    if as_uuid is not None:
        return as_uuid
    if not _ID_CHARS.fullmatch(text):
        raise ValueError(UNSAFE_ID.format(argument=argument, value=text[:80]))
    return text


def _canonical_args(args: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """A copy of ``args`` with each named id (each item of a list) canonical."""
    work = dict(args)
    for name in names:
        value = work.get(name)
        if isinstance(value, list):
            work[name] = [canonical_id(v, f"{name}[{i}]") for i, v in enumerate(value)]
        elif value is not None:
            work[name] = canonical_id(value, name)
    return work


def _uuid_slots(work: dict[str, Any], names: tuple[str, ...]) -> list[_Slot]:
    """The named argument values that hold a UUID, in argument and list order."""
    slots: list[_Slot] = []
    for name in names:
        value = work.get(name)
        if isinstance(value, list):
            slots += [
                _Slot(f"{name}[{i}]", name, i, item)
                for i, item in enumerate(value)
                if isinstance(item, str) and _as_uuid(item) == item
            ]
        elif isinstance(value, str) and _as_uuid(value) == value:
            slots.append(_Slot(name, name, None, value))
    return slots


def _missed(exc: BaseException) -> bool:
    """Whether ``exc`` is Studio not finding an id: a 404, or its API error."""
    if isinstance(exc, StudioAccessError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    return isinstance(exc, RuntimeError) and str(exc).startswith(_API_ERROR)


def _caught_404(result: Any) -> bool:
    """Whether a tool that catches its errors answered a 404 as its reason."""
    if not isinstance(result, dict) or result.get("available") is not False:
        return False
    return _CAUGHT_404 in str(result.get("reason") or "")


def _listed(records: list[dict[str, Any]], ids: list[str]) -> str:
    """The first :data:`MAX_LISTED` results by id, each with its properties."""
    parts = []
    for record, rid in list(zip(records, ids))[:MAX_LISTED]:
        props = [str(p) for p in record.get("property_names") or []]
        parts.append(f"{rid} ({', '.join(props)})" if props else rid)
    return "; ".join(parts)


async def _one_result(ctx: ToolContext, slot: _Slot) -> str | None:
    """The one result of the prediction ``slot.value``, or None if it is none.

    Raises the refusal (a ``ValueError``) for a prediction with no result or
    several, :class:`StudioAccessError` as it came, and :class:`_CheckFailed`
    when Studio could not be read otherwise.
    """
    try:
        prediction = await ctx.studio.get_prediction(slot.value)
    except StudioAccessError:
        raise
    except Exception as exc:  # noqa: BLE001 - a 404 or API error is no prediction
        if _missed(exc):
            return None
        raise _CheckFailed from exc
    if not prediction:
        return None
    try:
        env = await ctx.studio.search_prediction_results(
            prediction_id=slot.value, limit=SEARCH_LIMIT
        )
    except StudioAccessError:
        raise
    except Exception as exc:  # noqa: BLE001 - the first outcome stands
        raise _CheckFailed from exc
    records: list[dict[str, Any]] = []
    ids: list[str] = []
    for record in env.records:
        rid = record.get("id")
        if not rid:
            continue
        of = record.get("prediction_id")
        if of and (_as_uuid(str(of).strip()) or str(of).strip()) != slot.value:
            continue  # a record of another prediction is never this one's result
        records.append(record)
        ids.append(_as_uuid(str(rid).strip()) or str(rid).strip())
    if len(ids) == 1:
        return ids[0]
    if not ids:
        status = prediction.get("status") or "unknown"
        raise ValueError(
            NO_RESULTS.format(argument=slot.label, prediction=slot.value, status=status)
        )
    n = len(ids)
    if len(env.records) >= SEARCH_LIMIT and isinstance(env.total, int):
        n = max(n, env.total)  # a full page: Studio's count says how many
    more = (
        f" (and {n - MAX_LISTED} more; olmoearth_fetch_results lists them)"
        if n > MAX_LISTED
        else ""
    )
    raise ValueError(
        SEVERAL_RESULTS.format(
            argument=slot.label,
            prediction=slot.value,
            n=n,
            listed=_listed(records, ids),
            more=more,
        )
    )


async def _swaps(
    ctx: ToolContext, work: dict[str, Any], names: tuple[str, ...]
) -> list[_Swap]:
    """Each UUID-valued argument that is a prediction id, with its one result."""
    found: dict[str, str | None] = {}
    swaps: list[_Swap] = []
    for slot in _uuid_slots(work, names):
        if slot.value not in found:
            found[slot.value] = await _one_result(ctx, slot)
        result_id = found[slot.value]
        if result_id is not None:
            swaps.append(_Swap(slot, result_id))
    return swaps


async def _checked(
    ctx: ToolContext, work: dict[str, Any], names: tuple[str, ...]
) -> list[_Swap]:
    """:func:`_swaps`, or none when Studio could not be read for the check."""
    try:
        return await _swaps(ctx, work, names)
    except _CheckFailed:
        return []


def _swapped(work: dict[str, Any], swaps: list[_Swap]) -> dict[str, Any]:
    """A copy of ``work`` with each prediction id replaced by its result's id."""
    out = dict(work)
    for swap in swaps:
        slot = swap.slot
        if slot.index is None:
            out[slot.name] = swap.result_id
        else:
            items = list(out[slot.name])
            items[slot.index] = swap.result_id
            out[slot.name] = items
    return out


def _noted(result: Any, swaps: list[_Swap], used: dict[str, Any]) -> Any:
    """``result`` saying which prediction ids were read as which result ids.

    ``used`` holds the arguments the handler ran on (:func:`_swapped`). Each
    resolved argument is also set in the result, where the tool did not set
    it itself (``result_id``, ``result_id_b``, ``result_ids``), to the result
    id it ran on: the provenance recipe takes a verification call's maps from
    its result first (``provenance.log.VerificationCall``), so it repeats the
    call with the result id, as the design file and the certificate name it,
    not with the prediction id as the call was made.
    """
    if not isinstance(result, dict):
        return result
    result["resolved_ids"] = [
        {
            "argument": swap.slot.label,
            "prediction_id": swap.slot.value,
            "result_id": swap.result_id,
        }
        for swap in swaps
    ]
    for name in dict.fromkeys(swap.slot.name for swap in swaps):
        result.setdefault(name, used[name])
    if len(swaps) == 1:
        result.setdefault("prediction_id", swaps[0].slot.value)
    card = result.get(CARD_KEY)
    if isinstance(card, list):
        for swap in swaps:
            detail = f"the one result of prediction {swap.slot.value}"
            if len(swaps) > 1:
                detail += f" ({swap.slot.label})"
            card.append(["Result used", swap.result_id, detail])
    return result


def resolving(*names: str, caught: bool = False) -> Callable[[Handler], Handler]:
    """Wrap a tool's handler so a prediction id works where a result id is asked.

    ``names`` are the arguments that take a result id (a string, or a list of
    them); ``caught`` also starts the check on a returned ``available:
    false`` whose reason holds "404 Not Found" (a tool that catches its
    errors). The returned handler carries ``names`` as
    :data:`RESULT_ID_ARGS`. See the module docstring for what it does.
    """
    wanted = tuple(names)

    def wrap(handler: Handler) -> Handler:
        """The resolving handler around ``handler``."""

        @functools.wraps(handler)
        async def resolved(args: dict[str, Any], ctx: ToolContext) -> Any:
            """``handler`` on canonical ids, then on a prediction's one result."""
            work = _canonical_args(args, wanted)
            try:
                result = await handler(work, ctx)
            except Exception as exc:
                if not _missed(exc):
                    raise
                swaps = await _checked(ctx, work, wanted)
                if not swaps:
                    raise
            else:
                if not (caught and _caught_404(result)):
                    return result
                swaps = await _checked(ctx, work, wanted)
                if not swaps:
                    return result
            # Once more with the results in place, and never again.
            used = _swapped(work, swaps)
            return _noted(await handler(used, ctx), swaps, used)

        setattr(resolved, RESULT_ID_ARGS, wanted)
        return resolved

    return wrap
