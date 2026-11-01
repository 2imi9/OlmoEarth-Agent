# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Shared pixel-value sampling for every tool that reads a result on a grid.

Two tools sample Studio's ``pixel-value`` endpoint on a grid
(``olmoearth_compare_results``, in all four of its modes and for the two-map
design of ``olmoearth_plan_label_sample``, and
``olmoearth_review_set_from_result``). They share three things that used to be
copied per tool and are now one implementation each:

- :func:`sample_records` -- bounded-concurrency sampling (a semaphore of
  :data:`SAMPLE_CONCURRENCY`), a failed call resolving to ``None`` and a
  point Studio answers with HTTP 404, where it answered other points of the
  same result, to an :class:`OutsideRecord` (read as a point where Studio
  serves no file, status :data:`OUTSIDE`; which file covers which point was
  not tested);
- :func:`read_sample` -- one band's value with the no-data decision of
  :func:`~olmoearth_agent.analysis.raster_compare.band_is_nodata` applied, so a
  sentinel such as ``-1.0`` on a ``[0, 1]`` band never reaches a statistic;
- :func:`result_nodata_context` -- the declared outputs of a result and the
  model's ``nodata_value`` (best effort: a failed lookup leaves the declared
  range as the only no-data rule).

:func:`prediction_meta` reads a result's date and model from its prediction,
once, for both the no-data context and the comparison's choice of mode;
:func:`prediction_of` is that one lookup, and :func:`prediction_period` the
period a prediction's map describes (its start and end times), which the
two-map label plan reads its dates from.

A class map with top-k scores (a Studio segmentation prediction submitted
with ``probability_output``) carries each rank's class and softmax score as
bands of their own: :func:`topk_scores` recognises one from a result's
property names and declared outputs, and :func:`read_topk` reads one window's
rank-1 class, its rank-1 and rank-2 scores and their margin, each band by its
own name (:func:`select_band`'s fall back to the first band never applies
there).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import httpx

from olmoearth_agent.analysis.raster_compare import (
    band_is_nodata,
    declared_fields,
    declared_range,
)
from olmoearth_agent.tools.registry import ToolContext

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
SAMPLE_CONCURRENCY = 8

#: Sample status values returned by :func:`read_sample`.
OK, NODATA, FAILED = "ok", "nodata", "failed"
#: The status of a point Studio answered with HTTP 404 while answering other
#: points of the result (:class:`OutsideRecord`), read as a point where no
#: file is served; which file covers which point was not tested.
#: :data:`FAILED` is every other failed call.
OUTSIDE = "outside"

#: What a point of status :data:`OUTSIDE` is, as the sampling tools state it:
#: what was observed (a 404 beside answered points), not where the result's
#: files lie, which no probe tested.
OUTSIDE_REASON = (
    "answered with HTTP 404 (not_found_error) by Studio, which answered other "
    "points of the result"
)


class OutsideRecord(dict[str, Any]):
    """The record :func:`sample_records` gives a point Studio answers with HTTP 404.

    Probe 5 (31 October 2026) sampled the kelp result at the cell centres of
    the review set's 16 x 16 grid and got 60 answers of HTTP 404
    ``not_found_error`` beside 196 answers of 200, the same count the live
    run had reported as 60 failed calls. A 404 where the same result answered
    other points is read as a point where Studio serves no file and counted
    on its own (:data:`OUTSIDE`); the probe did not test which file covers
    which point, so the tools state the 404 they saw (:data:`OUTSIDE_REASON`),
    not that the point lies outside the result's files. The record holds no
    band, so a reader that does not ask for its status finds no value in it,
    as in a record with no bands.
    """


def dropped_text(n_nodata: int, n_failed: int, n_outside: int = 0) -> str:
    """The dropped windows in words: ``3 no-data, 2 failed``, then the 404s.

    The windows Studio answered with HTTP 404 (:data:`OUTSIDE`) are named
    only when there are some, so a sampling with none reads as it did before
    they were counted apart from the failed calls.
    """
    text = f"{n_nodata} no-data, {n_failed} failed"
    if n_outside:
        text += (
            f", {n_outside} answered HTTP 404 (not_found_error) where Studio "
            "answered other points"
        )
    return text


#: How a sampled value is judged no-data, stated in every sampling result.
NODATA_RULE = (
    "a sample is no-data, and never enters a statistic, when its band is "
    "missing or NaN, equals the model's nodata_value, lies outside the band's "
    "declared regression range, or carries no class on a classification band"
)


def select_band(
    record: dict[str, Any], property_name: str | None
) -> dict[str, Any] | None:
    """Pick the pixel-value band to read: the named property, else the first.

    The first band stands in for a named property the record lacks: the
    grid tools' fakes and older records carry one band with no
    ``property_name``. A read that must not take another band, a class map's
    top-k scores above all, uses :func:`band_by_name`.
    """
    bands: list[dict[str, Any]] = record.get("bands") or []
    if not bands:
        return None
    if property_name:
        return next(
            (b for b in bands if b.get("property_name") == property_name), bands[0]
        )
    return bands[0]


def band_by_name(record: dict[str, Any], property_name: str) -> dict[str, Any] | None:
    """The pixel-value band of ``property_name`` itself, or ``None``: never another."""
    bands: list[Any] = record.get("bands") or []
    return next(
        (
            b
            for b in bands
            if isinstance(b, dict) and b.get("property_name") == property_name
        ),
        None,
    )


def default_property(record: dict[str, Any], prop: str | None) -> str | None:
    """The property a sample of ``record`` will read, from its declared names.

    Mirrors :func:`select_band`: the named property when the result has it,
    else the first. ``None`` when the record lists no property names.
    """
    names = [n for n in record.get("property_names") or [] if isinstance(n, str)]
    if not names:
        names = list(declared_fields(record))
    if not names:
        return None
    if prop and prop in names:
        return prop
    return names[0]


def class_value(classification: Any) -> Any:
    """A hashable class value: Studio's ``{label, color}`` object reads as its label."""
    if isinstance(classification, dict):
        return classification.get("label", classification.get("value"))
    return classification


def band_value(record: dict[str, Any], property_name: str | None) -> Any:
    """A band's value as sampled: the class label if set, else ``raw_value``.

    No no-data decision is made here; grid-sampling tools use
    :func:`read_sample`, which applies it.
    """
    band = select_band(record, property_name)
    if band is None:
        return None
    cls = band.get("classification")
    return class_value(cls) if cls is not None else band.get("raw_value")


def is_categorical(record: dict[str, Any], property_name: str | None) -> bool:
    """True when the selected band carries a class rather than a number."""
    band = select_band(record, property_name)
    return band is not None and band.get("classification") is not None


@dataclass
class ResultNodata:
    """What decides no-data for one result: its declared outputs and the model's value."""

    declared: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodata_value: float | None = None
    model: dict[str, Any] | None = None


def summarize_model(record: dict[str, Any]) -> dict[str, Any]:
    """The fields of a Studio model record the agent describes a model with.

    ``prediction_type`` (from ``wizard_answers``, e.g. ``per_pixel_regression``)
    says what the output is: a regression output is a value per pixel, not a
    class probability and not a confidence.

    ``trained_on_labels`` is True when the fine-tuning wizard names a label
    field (``label_field_id``, with its ``split`` of the labels into train,
    val and test), else None: not known from the record. exp86 round 9
    (B3/studio run 1) said "no ground-truth labels exist" of two models whose
    records named a label field of the user's project; the tools had passed
    on only name, type and output. ``split_field_named`` says whether the
    record also names a split field (``split_field_id``): then which of it
    and the proportions training followed is not recorded, and the proportions
    are not stated as the training split
    (``statistical_rules.labels_in_studio_fact``). No person's id is read
    (the record's ``requester_id`` is left out).
    """
    wiz = record.get("wizard_answers") or {}
    out: dict[str, Any] = {
        "model_id": record.get("id"),
        "name": record.get("name"),
        "model_type": record.get("model_type"),
        "prediction_type": wiz.get("prediction_type"),
        "nodata_value": wiz.get("nodata_value"),
        "trained_on_labels": True if wiz.get("label_field_id") else None,
    }
    if wiz.get("label_field_id"):
        out["label_field_id"] = wiz.get("label_field_id")
        out["split_field_named"] = bool(wiz.get("split_field_id"))
        split = wiz.get("split_proportions")
        if isinstance(split, dict):
            out["split"] = {
                k: v
                for k, v in split.items()
                if k in ("train_prop", "val_prop", "test_prop")
            }
    return out


async def model_summary(
    ctx: ToolContext,
    model_id: str,
    cache: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """Best-effort :func:`summarize_model` of ``GET /models/{id}`` (None on failure).

    ``None`` also when the server serves no ``/models`` route; the client's
    ``models_unavailable`` then tells that apart from a model without a record
    (``olmoearth_search_predictions`` states it, so no label field read is not
    taken for no labels).
    """
    if cache is not None and model_id in cache:
        return cache[model_id]
    try:
        record = await ctx.studio.get_model(model_id)
    except Exception:  # enrichment only; never fails the calling tool
        summary = None
    else:
        summary = summarize_model(record) if record else None
    if cache is not None:
        cache[model_id] = summary
    return summary


async def prediction_of(
    ctx: ToolContext, record: dict[str, Any]
) -> dict[str, Any] | None:
    """The prediction a result came from (``GET /predictions/{id}``), or ``None``.

    Best effort: a missing ``prediction_id``, a failed lookup or a record that
    is not an object gives ``None`` and never fails the calling tool.
    """
    prediction_id = record.get("prediction_id")
    if not prediction_id:
        return None
    try:
        prediction = await ctx.studio.get_prediction(str(prediction_id))
    except Exception:  # best effort, as above
        return None
    return prediction if isinstance(prediction, dict) else None


def prediction_period(prediction: dict[str, Any] | None) -> str | None:
    """The period a prediction's map describes, as ``start/end``, or the one time given.

    Studio runs a prediction over the imagery of ``[start_time, end_time]``;
    ``oe_inferencex.compare.dates_reading`` reads ``start/end`` as that
    period (an ISO date or date-time each). ``None`` when the prediction
    gives neither time, or is unknown.
    """
    if not prediction:
        return None
    raw_start, raw_end = prediction.get("start_time"), prediction.get("end_time")
    start: str | None = str(raw_start).strip() if raw_start else None
    end: str | None = str(raw_end).strip() if raw_end else None
    if start and end and start != end:
        return f"{start}/{end}"
    return start or end


async def prediction_meta(
    ctx: ToolContext, record: dict[str, Any]
) -> tuple[str | None, str | None]:
    """A result's ``(date, model_id)`` from its prediction record.

    Result records carry no date of their own; the prediction they came from
    does (``start_time``, else ``end_time``), and it names the model. Best
    effort: a missing ``prediction_id`` or a failed lookup gives
    ``(None, None)`` and never fails the calling tool.
    """
    prediction = await prediction_of(ctx, record)
    if prediction is None:
        return None, None
    date = prediction.get("start_time") or prediction.get("end_time")
    model_id = prediction.get("model_id")
    return (str(date) if date else None), (str(model_id) if model_id else None)


async def model_for_result(
    ctx: ToolContext,
    record: dict[str, Any],
    cache: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """The model behind a result (via its prediction), or ``None`` if unknown."""
    _date, model_id = await prediction_meta(ctx, record)
    if not model_id:
        return None
    return await model_summary(ctx, model_id, cache)


def _nodata_number(value: Any) -> float | None:
    """The model's ``nodata_value`` as a float, or ``None`` when unset/unusable."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def result_nodata_context(
    ctx: ToolContext,
    record: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    lookup_model: bool = True,
    cache: dict[str, dict[str, Any] | None] | None = None,
) -> ResultNodata:
    """The no-data context of one result record.

    ``model`` may be passed when the caller already resolved it; otherwise it
    is looked up from the result's prediction when ``lookup_model`` is true.
    """
    if model is None and lookup_model:
        model = await model_for_result(ctx, record, cache)
    return ResultNodata(
        declared=declared_fields(record),
        nodata_value=_nodata_number((model or {}).get("nodata_value")),
        model=model,
    )


async def sample_records(
    ctx: ToolContext,
    result_id: str,
    points: list[tuple[float, float]],
    sem: asyncio.Semaphore,
) -> list[dict[str, Any] | None]:
    """Sample one result at every ``(lon, lat)``; a failed call yields ``None``.

    A point Studio answers with HTTP 404 yields an :class:`OutsideRecord`
    (:data:`OUTSIDE`, read as a point where no file is served) when the same
    result answered at least one other point; with no answer at any point,
    every 404 stays ``None``: a removed result, or files Studio no longer
    serves, answer 404 everywhere, and that says nothing of where the files
    lie. Every other failure, a 401 or 403 among them, stays ``None``
    (:data:`FAILED`).
    """

    async def one(lon: float, lat: float) -> dict[str, Any] | None:
        """One bounded pixel-value call."""
        async with sem:
            try:
                return await ctx.studio.pixel_value(result_id, lon, lat)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:  # read as no file served there
                    return OutsideRecord()
                return None
            except Exception:  # transient or any other failure -> drop the point
                return None

    records = list(await asyncio.gather(*[one(lo, la) for lo, la in points]))
    if not any(r is not None and not isinstance(r, OutsideRecord) for r in records):
        # No point answered: a 404 is then no sign of the files' extent.
        return [None if isinstance(r, OutsideRecord) else r for r in records]
    return records


def read_sample(
    record: dict[str, Any] | None,
    property_name: str | None,
    nodata: ResultNodata | None = None,
    *,
    exact: bool = False,
) -> tuple[Any, bool, str]:
    """One sampled value: ``(value, categorical, status)``.

    ``status`` is :data:`OK`, :data:`NODATA` (no band, or
    :func:`~olmoearth_agent.analysis.raster_compare.band_is_nodata` said so),
    :data:`OUTSIDE` (Studio answered 404 where it answered other points,
    read as a point where no file is served, :class:`OutsideRecord`) or
    :data:`FAILED` (the pixel-value call itself failed otherwise). ``value``
    is ``None`` unless the status is :data:`OK`; a class reads as its label.
    With ``exact``, a named property is read from its own band only
    (:func:`band_by_name`): a record without it is :data:`NODATA`, never
    read from the first band (the class of a map with top-k scores is read
    this way).
    """
    if record is None:
        return None, False, FAILED
    if isinstance(record, OutsideRecord):
        return None, False, OUTSIDE
    if exact and property_name:
        band = band_by_name(record, property_name)
    else:
        band = select_band(record, property_name)
    if band is None:
        return None, False, NODATA
    ctx = nodata or ResultNodata()
    declared = ctx.declared.get(str(band.get("property_name")))
    categorical = band.get("classification") is not None or (
        (declared or {}).get("value_type") == "classification"
    )
    if band_is_nodata(band, nodata_value=ctx.nodata_value, declared=declared):
        return None, categorical, NODATA
    cls = band.get("classification")
    value = class_value(cls) if cls is not None else band.get("raw_value")
    return value, categorical, OK


# --------------------------------------------------------------------------- a class map with top-k scores

#: Rank ``k``'s softmax score of class property ``P`` (``P_rank<k>_prob``) or
#: rank ``k``'s class (``P_rank<k>_class``): the bands a Studio segmentation
#: result submitted with ``probability_output`` adds beside ``P``, whose own
#: band is rank 1's class.
_TOPK_PART_RE = re.compile(r"^(?P<base>.+)_rank(?P<rank>[1-9][0-9]*)_(?:prob|class)$")

#: The declared range of Studio's uint8 top-k scores, read as
#: ``(raw - 1) / 254`` (the files zip's README; raw 255, above the declared
#: maximum and the files' no-data value, is a score of 1.0, and raw 0 reads
#: 0.0: :func:`topk_probability`; the margin is read from the raw levels,
#: :func:`read_topk`), and the range of a score already in ``[0, 1]``, read
#: as it is.
TOPK_UINT8_RANGE = (0.0, 254.0)
TOPK_UNIT_RANGE = (0.0, 1.0)

#: How a window of a class map with top-k scores is judged no-data, stated in
#: every sampling of one.
TOPK_NODATA_RULE = (
    "a window is no-data, and never enters a statistic, when its rank-1 class "
    "band is missing or NaN, equals the model's nodata_value or carries no "
    "class; a score never makes a window no-data (a raw 255 is a score of "
    "1.0), and a window with a class but no rank-1 or rank-2 score, or a "
    "rank-2 score above its rank-1, refuses the sampling instead of being "
    "dropped"
)

#: The status :func:`read_topk` gives a window whose rank-1 class band holds
#: a class but whose rank-1 or rank-2 score band is missing or holds no
#: number: not no-data, since a score never decides whether a pixel is valid,
#: and not readable, so a sampling that meets one is refused rather than
#: dropping a valid window of the map.
UNSCORED = "unscored"


def _is_number(value: Any) -> bool:
    """True for an int or float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def class_key(value: Any) -> str:
    """How a class is matched: its text, case and outer spaces aside; ``3.0`` is ``3``.

    As the estimation tools match the classes a reviewer writes.
    """
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


@dataclass(frozen=True)
class TopKScores:
    """A class map with top-k scores, as a Studio result names its bands.

    ``base`` is the class property ``P``, whose band is the map's (rank-1)
    class. ``score_properties`` and ``class_properties`` name rank 1's to
    rank ``k``'s score and class (rank 1's class is ``base``; a rank whose
    class the result does not name is ``None``). ``classes`` are the declared
    classes, ``(value, label)`` in ``allowed_values`` order, or empty where
    the result declares none.
    """

    base: str
    k: int
    score_properties: tuple[str, ...]
    class_properties: tuple[str | None, ...]
    classes: tuple[tuple[Any, str], ...] = ()

    @property
    def labels(self) -> list[str]:
        """The declared classes' labels, in ``allowed_values`` order."""
        return [label for _value, label in self.classes]

    def reads_a_part(self, prop: str | None) -> bool:
        """Whether ``prop`` names a rank's score or class of ``base``, not the map.

        By the rank's name (``P_rank<k>_prob`` or ``P_rank<k>_class``), named
        by the result or not. Any other property (a mistyped ``P`` among
        them) reads the map, as :func:`default_property` falls back to the
        result's first property.
        """
        part = _TOPK_PART_RE.match(prop or "")
        return part is not None and part["base"] == self.base

    def describe(self) -> dict[str, Any]:
        """The field as a result states it: property names and counts only."""
        return {
            "class_property": self.base,
            "k": self.k,
            "score_properties": list(self.score_properties),
            "class_properties": list(self.class_properties),
            "n_classes_declared": len(self.classes),
        }

    def code_classes(self, values: list[Any]) -> tuple[list[int], list[Any]]:
        """Each window's rank-1 class as a code, and the classes the codes stand for.

        Codes follow the declared classes' order; a class read is matched to
        a declared one by its label, else its value (:func:`class_key`). A
        class read that the legend does not hold (every class, where the
        result declares none) gets the next code, written as read.
        """
        table: list[Any] = self.labels
        code_of: dict[str, int] = {}
        for code, label in enumerate(table):
            code_of.setdefault(class_key(label), code)
        for code, (value, _label) in enumerate(self.classes):
            if value is not None:
                code_of.setdefault(class_key(value), code)
        codes: list[int] = []
        for value in values:
            key = class_key(value)
            if key not in code_of:
                code_of[key] = len(table)
                whole = isinstance(value, float) and value.is_integer()
                table.append(int(value) if whole else value)
            codes.append(code_of[key])
        return codes, table


def _field_named(fields: Any, name: str) -> dict[str, Any] | None:
    """The declared output field of property ``name`` in ``fields``, or ``None``."""
    return next(
        (
            f
            for f in fields or []
            if isinstance(f, dict) and f.get("property_name") == name
        ),
        None,
    )


def topk_scores(record: dict[str, Any], prop: str | None = None) -> TopKScores | None:
    """The class map with top-k scores read as ``prop`` of ``record``, or ``None``.

    The property read is resolved as :func:`default_property` resolves it:
    ``prop`` when the result names it, else the first the result names; a
    rank's score or class of a top-k field the result names
    (``P_rank3_prob`` of a top-2 result) is read as that rank, so it is
    refused as the map (:meth:`TopKScores.reads_a_part`), never read as
    ``P``. It meets a top-k field when it is a class property ``P``, or one
    of ``P``'s ranks' properties, and the result names ``P``,
    ``P_rank1_prob`` and ``P_rank2_prob``; ``k`` counts the score ranks
    named from 1 up. Where ``result_metadata`` declares the result's fields,
    Studio's own pairing is required: ``P`` is a classification field whose
    ``confidence_band_index`` is the ``band_index`` of ``P_rank1_prob``,
    else the bands are not taken for top-k scores; its ``allowed_values``
    are the classes. ``None`` for any other property, so a result without
    top-k scores is read as before.
    """
    names = [n for n in record.get("property_names") or [] if isinstance(n, str)]
    if not names:
        names = list(declared_fields(record))
    if not names:
        return None
    asked = _TOPK_PART_RE.match(prop or "")
    target = names[0]
    if prop and (
        prop in names or (asked is not None and f"{asked['base']}_rank1_prob" in names)
    ):
        target = prop
    part = _TOPK_PART_RE.match(target)
    base = target if f"{target}_rank1_prob" in names else None
    if base is None and part is not None:
        base = part["base"]
    if base is None or not (
        {base, f"{base}_rank1_prob", f"{base}_rank2_prob"} <= set(names)
    ):
        return None
    meta = record.get("result_metadata") or {}
    field = _field_named(meta.get("classification_fields"), base)
    rank1 = _field_named(meta.get("regression_fields"), f"{base}_rank1_prob")
    # With no result_metadata (search records carry it null), or one that
    # declares no field, the bands are taken by their names alone, as Studio
    # names a top-k result's bands (P_rank<k>_prob is rank k's score of P,
    # rank 1's class is P itself: the files zip's README and the pixel-value
    # records of 31 October 2026), and the classes are the ones the reads
    # carry. Metadata that declares fields but not P as a classification
    # field paired with its rank-1 score says the bands are something else.
    if meta.get("classification_fields") or meta.get("regression_fields"):
        band = (field or {}).get("confidence_band_index")
        if band is None or rank1 is None or band != rank1.get("band_index"):
            return None
    k = 2
    while f"{base}_rank{k + 1}_prob" in names:
        k += 1
    runners = [f"{base}_rank{r}_class" for r in range(2, k + 1)]
    values = [
        v for v in (field or {}).get("allowed_values") or [] if isinstance(v, dict)
    ]
    return TopKScores(
        base=base,
        k=k,
        score_properties=tuple(f"{base}_rank{r}_prob" for r in range(1, k + 1)),
        class_properties=tuple([base, *[n if n in names else None for n in runners]]),
        classes=tuple(
            (v.get("value"), str(v.get("label", v.get("value")))) for v in values
        ),
    )


def topk_part_reason(prop: str, topk: TopKScores) -> str:
    """Why a rank's score or class is refused as the property read: it is no map."""
    return (
        f"{prop!r} is a rank's score or class of {topk.base!r}, a class map with "
        f"top-k scores: the map is {topk.base!r}, its rank-1 class, read with its "
        f"scores; pass property_name={topk.base!r} or leave it out"
    )


def _range_reason(name: str, rng: tuple[float, float] | None) -> str:
    """Why a top-k score band's declared range is not read."""
    shown = f"the range [{rng[0]:g}, {rng[1]:g}]" if rng else "no range"
    return (
        f"the top-k score {name!r} declares {shown}: a top-k score is read only "
        "as Studio's uint8 score, declared [0, 254] and read as (raw - 1) / 254, "
        "or as a score declared [0, 1], so no margin is read from this one"
    )


def topk_range_refusal(
    topk: TopKScores, declared: dict[str, dict[str, Any]]
) -> str | None:
    """Why a rank-1 or rank-2 score's declared range is not read, or ``None``.

    From the result's declared outputs (:func:`declared_fields`), before any
    pixel is sampled; a score the result declares nothing about is judged by
    its band's own range when read (:func:`topk_probability`).
    """
    for name in topk.score_properties[:2]:
        decl = declared.get(name)
        if decl is not None:
            rng = declared_range(None, decl)
            if rng not in (TOPK_UINT8_RANGE, TOPK_UNIT_RANGE):
                return _range_reason(name, rng)
    return None


def topk_probability(raw: float, rng: tuple[float, float] | None, name: str) -> float:
    """One top-k score as a probability, by the range its band declares.

    Declared ``[0, 254]`` (Studio's uint8 scores): ``(raw - 1) / 254``, so
    raw 1 reads 0.0 and raw 255 reads 1.0; raw 0 reads 0.0 too. The README
    calls raw 0 reserved and unused, but the two-class kelp result measured
    on 31 October 2026 has ``p1 + p2 = 252/254`` on every valid pixel and a
    median ``p1`` of raw 254, so its rank-2 score is raw 0 on at least half
    of them: it is read as the least score, and never drops a window.
    Declared ``[0, 1]``: as it is.

    Raises
    ------
    ValueError
        For any other declared range, or none, and for a value that reads
        outside ``[0, 1]`` otherwise.
    """
    if rng == TOPK_UINT8_RANGE:
        prob = 0.0 if raw == 0.0 else (raw - 1.0) / 254.0
    elif rng == TOPK_UNIT_RANGE:
        prob = raw
    else:
        raise ValueError(_range_reason(name, rng))
    if not -1e-9 <= prob <= 1.0 + 1e-9:
        msg = (
            f"the top-k score {name!r} holds {raw:g}, which reads {prob:.6g}, "
            "outside [0, 1]: no score of Studio's top-k format, so no margin is "
            "read from it"
        )
        raise ValueError(msg)
    return min(1.0, max(0.0, prob))


def topk_quantized(
    topk: TopKScores,
    nodata: ResultNodata | None,
    records: list[dict[str, Any] | None],
) -> bool:
    """Whether the rank-1 score is Studio's uint8 score (declared ``[0, 254]``).

    From the result's declared outputs, else from the first sampled record
    whose rank-1 score band declares its range.
    """
    name = topk.score_properties[0]
    decl = (nodata or ResultNodata()).declared.get(name)
    rng = declared_range(None, decl) if decl else None
    for rec in records:
        if rng is not None:
            break
        if rec is not None:
            rng = declared_range(band_by_name(rec, name))
    return rng == TOPK_UINT8_RANGE


def read_topk(
    record: dict[str, Any] | None,
    topk: TopKScores,
    nodata: ResultNodata | None = None,
) -> tuple[tuple[Any, float, float, float] | None, str]:
    """One window of a class map with top-k scores, and the window's status.

    The window is ``(class, p1, p2, margin)``: ``class`` is the rank-1 class
    as :func:`read_sample` reads a class (its label), ``p1`` and ``p2`` are
    the rank-1 and rank-2 scores as probabilities (:func:`topk_probability`,
    by the range each band declares), and ``margin`` is the rank-1 minus the
    rank-2 score. For Studio's uint8 scores the margin is their raw levels'
    difference over 254 (at most 1), which is ``p1 - p2`` under the README's
    ``(raw - 1) / 254`` and under ``raw / 254`` alike (the measured kelp
    scores fit the second better), and keeps a rank-2 raw 0 one level below
    a raw 1, although both read 0.0. Every band is found by its own property
    name (:func:`band_by_name`), never by position: :func:`select_band`
    would read the class band for a missing score.

    ``status`` is :data:`FAILED` for a failed call, :data:`OUTSIDE` for a
    point Studio answered with HTTP 404 where it answered others
    (:class:`OutsideRecord`), :data:`NODATA` when the
    rank-1 class band is missing or no-data
    (:func:`~olmoearth_agent.analysis.raster_compare.band_is_nodata`, the
    band judged as a class band whether or not the result declares it, so a
    band with no class is no-data) and :data:`UNSCORED` when that band holds
    a class but the rank-1 or rank-2 score band is missing or holds no
    number. A score never makes a window no-data: the files zip's README
    says a valid pixel's score may equal the no-data value, and that only
    the rank-1 class band says whether a pixel is valid.

    Raises
    ------
    ValueError
        From :func:`topk_probability`, for a score whose declared range is
        neither ``[0, 254]`` nor ``[0, 1]``, or that reads outside ``[0, 1]``;
        and for a rank-2 score above the rank-1 score, which no one top-k
        field holds at one pixel.
    """
    if record is None:
        return None, FAILED
    if isinstance(record, OutsideRecord):
        return None, OUTSIDE
    ctx = nodata or ResultNodata()
    band = band_by_name(record, topk.base)
    # P is a class property by construction: a band of it with no class is
    # no-data, whether or not the result declares P (topk_scores takes the
    # bands by their names alone where the result declares no field).
    declared = ctx.declared.get(topk.base) or {"value_type": "classification"}
    if band is None or band_is_nodata(
        band, nodata_value=ctx.nodata_value, declared=declared
    ):
        return None, NODATA
    cls = band.get("classification")
    value = class_value(cls) if cls is not None else band.get("raw_value")
    names = topk.score_properties[:2]
    raws: list[float] = []
    ranges: list[tuple[float, float] | None] = []
    for name in names:
        score = band_by_name(record, name)
        raw = (score or {}).get("raw_value")
        if raw is None or not _is_number(raw) or not isfinite(float(raw)):
            return None, UNSCORED
        raws.append(float(raw))
        ranges.append(declared_range(score, ctx.declared.get(name)))
    p1 = topk_probability(raws[0], ranges[0], names[0])
    p2 = topk_probability(raws[1], ranges[1], names[1])
    if ranges[0] == ranges[1] == TOPK_UINT8_RANGE:
        margin = min(1.0, (raws[0] - raws[1]) / 254.0)
    else:
        margin = p1 - p2
    if margin < -1e-9:
        msg = (
            f"a window's rank-2 score ({names[1]!r}, raw {raws[1]:g}) is above its "
            f"rank-1 score ({names[0]!r}, raw {raws[0]:g}): no one top-k field "
            "holds that at one pixel, so these bands were not read together and "
            "no margin is read from them"
        )
        raise ValueError(msg)
    return (value, p1, p2, max(0.0, margin)), OK
