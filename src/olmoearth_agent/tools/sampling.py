# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Shared pixel-value sampling for every tool that reads a result on a grid.

Two tools sample Studio's ``pixel-value`` endpoint on a grid
(``olmoearth_compare_results``, in all four of its modes, and
``olmoearth_review_set_from_result``). They share three things that used to be
copied per tool and are now one implementation each:

- :func:`sample_records` -- bounded-concurrency sampling (a semaphore of
  :data:`SAMPLE_CONCURRENCY`), a failed call resolving to ``None``;
- :func:`read_sample` -- one band's value with the no-data decision of
  :func:`~olmoearth_agent.analysis.raster_compare.band_is_nodata` applied, so a
  sentinel such as ``-1.0`` on a ``[0, 1]`` band never reaches a statistic;
- :func:`result_nodata_context` -- the declared outputs of a result and the
  model's ``nodata_value`` (best effort: a failed lookup leaves the declared
  range as the only no-data rule).

:func:`prediction_meta` reads a result's date and model from its prediction,
once, for both the no-data context and the comparison's choice of mode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from olmoearth_agent.analysis.raster_compare import band_is_nodata, declared_fields
from olmoearth_agent.tools.registry import ToolContext

#: Concurrency for grid pixel-value sampling (bounds load on Studio + proxy).
SAMPLE_CONCURRENCY = 8

#: Sample status values returned by :func:`read_sample`.
OK, NODATA, FAILED = "ok", "nodata", "failed"

#: How a sampled value is judged no-data, stated in every sampling result.
NODATA_RULE = (
    "a sample is no-data, and never enters a statistic, when its band is "
    "missing or NaN, equals the model's nodata_value, lies outside the band's "
    "declared regression range, or carries no class on a classification band"
)


def select_band(
    record: dict[str, Any], property_name: str | None
) -> dict[str, Any] | None:
    """Pick the pixel-value band to read: the named property, else the first."""
    bands: list[dict[str, Any]] = record.get("bands") or []
    if not bands:
        return None
    if property_name:
        return next(
            (b for b in bands if b.get("property_name") == property_name), bands[0]
        )
    return bands[0]


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
    on only name, type and output. No person's id is read (the record's
    ``requester_id`` is left out).
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
    """Best-effort :func:`summarize_model` of ``GET /models/{id}`` (None on failure)."""
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


async def prediction_meta(
    ctx: ToolContext, record: dict[str, Any]
) -> tuple[str | None, str | None]:
    """A result's ``(date, model_id)`` from its prediction record.

    Result records carry no date of their own; the prediction they came from
    does (``start_time``, else ``end_time``), and it names the model. Best
    effort: a missing ``prediction_id`` or a failed lookup gives
    ``(None, None)`` and never fails the calling tool.
    """
    prediction_id = record.get("prediction_id")
    if not prediction_id:
        return None, None
    try:
        prediction = await ctx.studio.get_prediction(str(prediction_id))
    except Exception:  # best effort, as above
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
    """Sample one result at every ``(lon, lat)``; a failed call yields ``None``."""

    async def one(lon: float, lat: float) -> dict[str, Any] | None:
        """One bounded pixel-value call."""
        async with sem:
            try:
                return await ctx.studio.pixel_value(result_id, lon, lat)
            except Exception:  # off-raster / transient -> drop the point
                return None

    return list(await asyncio.gather(*[one(lo, la) for lo, la in points]))


def read_sample(
    record: dict[str, Any] | None,
    property_name: str | None,
    nodata: ResultNodata | None = None,
) -> tuple[Any, bool, str]:
    """One sampled value: ``(value, categorical, status)``.

    ``status`` is :data:`OK`, :data:`NODATA` (no band, or
    :func:`~olmoearth_agent.analysis.raster_compare.band_is_nodata` said so)
    or :data:`FAILED` (the pixel-value call itself failed). ``value`` is
    ``None`` unless the status is :data:`OK`; a class reads as its label.
    """
    if record is None:
        return None, False, FAILED
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
