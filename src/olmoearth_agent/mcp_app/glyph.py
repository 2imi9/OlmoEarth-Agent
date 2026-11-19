# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""The property glyph's fields, computed here from Studio records.

The Studio panel draws a small glyph that says what a thing is, never how
good it is (spec/Usage.md, "In Claude, with views"):

- a prediction: the months its requested range covers, as a ring of twelve,
  with the year beside it;
- a model: its output (per pixel or per window; classes, values, detection or
  embeddings) as a legend tile, its size and version as words, and the
  sources and months its record names;
- a project: how many of its predictions cover each month, in fixed steps,
  years and areas folded together, with the count beside it.

A view gets only the small typed fields these functions return, never a
record: model, prediction and result records carry a person's id
(``requester_id``), credentials (``checkpoint_token``, ``download_token``)
and coordinates. A field the record does not hold is ``None`` ("not
recorded"), never a guess: no output kind is assumed, and no source either.

Months are counted in UTC. A month counts as covered when any instant of the
range falls in it; a range that ends exactly at 00:00 on the first of a month
ends before that month (an exclusive end). Twelve months or more cover the
whole ring, and the span in words says how long it is.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

# fmt: off
MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
# fmt: on

#: The model sizes Studio's builder offers, and Large (open weights), by the
#: word a record may hold in ``wizard_answers.encoder_variant``.
SIZES = {
    "nano": "Nano",
    "tiny": "Tiny",
    "small": "Small",
    "base": "Base",
    "large": "Large",
}

#: Badge texts for a model's image sources (``wizard_answers.imagery_sources``,
#: a list of strings on a live record, 11 Nov 2026), in the badge order S2,
#: S1, L8/9: a name, lowercased with everything but letters and digits taken
#: out, that starts with one of the prefixes or equals one of the short names.
#: Landsat 8 and 9 are one source in OlmoEarth (``landsat_8_9_c2_l1``).
SENSOR_BADGES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("S2", ("sentinel2",), ("s2",)),
    ("S1", ("sentinel1",), ("s1",)),
    ("L8/9", ("landsat",), ("l8", "l9", "l89")),
)

#: Month names a start-month constraint may use, by their first three letters.
_FULL_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_MONTH_KEYS = {
    **{name.lower(): i for i, name in enumerate(MONTH_NAMES, start=1)},
    **{name: i for i, name in enumerate(_FULL_MONTHS, start=1)},
}
_MONTH_NUMBER = re.compile(r"0?[1-9]|1[0-2]")

#: The project ring's fixed steps: none, 1, 2 to 4, 5 or more predictions.
PROJECT_STEPS = ((0, 0), (1, 1), (2, 2), (5, 3))

_VERSION = re.compile(r"^v?(\d+(?:\.\d+){0,2})$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def parse_time(value: Any) -> datetime | None:
    """An ISO 8601 time as an aware UTC datetime; ``None`` when it is not one.

    A time without a zone is read as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _range(start: Any, end: Any) -> tuple[datetime, datetime] | None:
    """The range as two UTC datetimes with an inclusive end, or ``None``."""
    first, last = parse_time(start), parse_time(end)
    if first is None or last is None or last < first:
        return None
    month_start = last.day == 1 and last.time() == datetime.min.time()
    if month_start and last > first:
        last = last - timedelta(microseconds=1)  # an exclusive end
    return first, last


def months_covered(start: Any, end: Any) -> list[int] | None:
    """Twelve 0/1 flags, January first, for the months the range covers.

    ``None`` when either time is missing or not a time, or the end comes
    before the start (the ring then says "not recorded").
    """
    span = _range(start, end)
    if span is None:
        return None
    first, last = span
    n_months = (last.year - first.year) * 12 + (last.month - first.month) + 1
    if n_months >= 12:
        return [1] * 12
    flags = [0] * 12
    for k in range(n_months):
        flags[(first.month - 1 + k) % 12] = 1
    return flags


def year_text(start: Any, end: Any) -> str | None:
    """The range's years: "2024", "2023-24" or "2019-2024"."""
    span = _range(start, end)
    if span is None:
        return None
    a, b = span[0].year, span[1].year
    if a == b:
        return str(a)
    if b == a + 1:
        return f"{a}-{str(b)[-2:]}"
    return f"{a}-{b}"


def span_text(start: Any, end: Any) -> str | None:
    """The range in words: "Apr 2024", "Apr to Jun 2024", "Nov 2023 to Feb 2024"."""
    span = _range(start, end)
    if span is None:
        return None
    first, last = span
    a = MONTH_NAMES[first.month - 1]
    b = MONTH_NAMES[last.month - 1]
    if (first.year, first.month) == (last.year, last.month):
        return f"{a} {first.year}"
    if first.year == last.year:
        return f"{a} to {b} {first.year}"
    return f"{a} {first.year} to {b} {last.year}"


def prediction_glyph(record: Mapping[str, Any]) -> dict[str, Any]:
    """A prediction's glyph fields, from its requested range (no extra request)."""
    start, end = record.get("start_time"), record.get("end_time")
    return {
        "months": months_covered(start, end),
        "years": year_text(start, end),
        "span": span_text(start, end),
    }


def _step(count: int) -> int:
    """The project ring's step for a month covered by ``count`` predictions."""
    level = 0
    for threshold, step in PROJECT_STEPS:
        if count >= threshold:
            level = step
    return level


def project_glyph(
    predictions: Iterable[Mapping[str, Any]], *, limit: int
) -> dict[str, Any]:
    """A project's glyph fields from its prediction records (one page).

    ``months`` holds each month's step (0 to 3), ``count`` the predictions
    read, ``dated`` how many of them have a readable range, and ``capped``
    whether the page was full (more may exist; the view then says "+").
    """
    records = list(predictions)
    totals = [0] * 12
    dated = 0
    for record in records:
        flags = months_covered(record.get("start_time"), record.get("end_time"))
        if flags is None:
            continue
        dated += 1
        totals = [t + f for t, f in zip(totals, flags, strict=True)]
    # Predictions but none with readable dates: "not recorded", never the empty
    # ring that means "covers no month".
    months = None if records and not dated else [_step(t) for t in totals]
    return {
        "months": months,
        "count": len(records),
        "dated": dated,
        "capped": len(records) >= limit,
    }


def _output(
    record: Mapping[str, Any], answers: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """The output's grain (pixel, window) and kind (classes, values, detection,
    embeddings), each ``None`` when the record does not say."""
    prediction_type = str(answers.get("prediction_type") or "").lower()
    model_type = str(record.get("model_type") or "").lower()
    grain = (
        "pixel"
        if "pixel" in prediction_type
        else "window" if "window" in prediction_type else None
    )
    if "embed" in model_type or "embed" in prediction_type:
        return grain, "embeddings"
    if "detect" in prediction_type:
        return grain, "detection"
    if "classif" in prediction_type:
        return grain, "classes"
    if "regress" in prediction_type:
        return grain, "values"
    return grain, None


def _size(answers: Mapping[str, Any]) -> str | None:
    """The size word for ``encoder_variant``; ``None`` for an unknown value."""
    variant = str(answers.get("encoder_variant") or "").lower()
    for key, word in SIZES.items():
        if re.search(rf"(^|[^a-z]){key}($|[^a-z])", variant):
            return word
    return None


def _version(answers: Mapping[str, Any]) -> str | None:
    """``foundation_model_version`` as "v1.2"; ``None`` when it is not a version."""
    match = _VERSION.match(str(answers.get("foundation_model_version") or "").strip())
    return f"v{match.group(1)}" if match else None


def sensor_badges(names: Any) -> list[str]:
    """Badge texts for the source names a record holds, in the order S2, S1, L8/9.

    Only sources the record names get a badge; an unknown name gets none.
    """
    if not isinstance(names, (list, tuple)):
        return []
    plain = [re.sub(r"[^a-z0-9]", "", n.lower()) for n in names if isinstance(n, str)]
    return [
        badge
        for badge, prefixes, short in SENSOR_BADGES
        if any(p.startswith(prefixes) or p in short for p in plain)
    ]


def _start_month(value: Any) -> int | None:
    """A start-month constraint as 1 to 12, from an exact spelling only: "4",
    "04", "Apr", "April" (any case). Anything else is ``None`` (not read)."""
    text = str(value).strip().lower()
    if _MONTH_NUMBER.fullmatch(text):
        return int(text)
    return _MONTH_KEYS.get(text)


def _context_mode(answers: Mapping[str, Any]) -> str | None:
    """``wizard_answers.temporal_context`` as "moment" or "period"; ``None``
    when it is missing, not a string, or neither. The moment words are tested
    first, so a "moment with ... period" spelling is a moment."""
    value = answers.get("temporal_context")
    if not isinstance(value, str):
        return None
    text = value.lower()
    if any(word in text for word in ("moment", "single", "instant", "point")):
        return "moment"
    if "period" in text:
        return "period"
    return None


def _temporal(answers: Mapping[str, Any]) -> dict[str, Any]:
    """A model's months and its context in words.

    Only a period context gives months: each allowed start month and the
    ``period_duration`` months after it (a live record holds an int; read as
    months, 1 to 12, as Studio's wizard offers). No start-month constraint
    (missing, null or an empty list) means any start, so every month. A
    constraint in any other shape, or a month not read, gives no months and
    says "start months not recorded". A moment context gives no months, only
    words; an unread context is ``None`` everywhere ("not recorded").
    """
    mode = _context_mode(answers)
    if mode != "period":
        words = None
        if mode == "moment":
            words = (
                str(answers.get("temporal_context")).replace("_", " ").lower().strip()
            )
        return {"mode": mode, "months": None, "context": words}
    duration = answers.get("period_duration")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not 1 <= duration <= 12
    ):
        return {
            "mode": mode,
            "months": None,
            "context": "a period of a length not recorded",
        }
    period = f"{duration}-month period"
    constraints = answers.get("start_month_constraints")
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        return {
            "mode": mode,
            "months": None,
            "context": period + ", start months not recorded",
        }
    parsed = [_start_month(c) for c in constraints]
    if any(s is None for s in parsed):
        return {
            "mode": mode,
            "months": None,
            "context": period + ", start months not recorded",
        }
    starts = sorted({s for s in parsed if s is not None})
    if not starts:
        return {
            "mode": mode,
            "months": [1] * 12,
            "context": period + ", any start month",
        }
    months = [0] * 12
    for s in starts:
        for k in range(duration):
            months[(s - 1 + k) % 12] = 1
    words = period + ", starts " + " or ".join(MONTH_NAMES[s - 1] for s in starts)
    return {"mode": mode, "months": months, "context": words}


def model_glyph(record: Mapping[str, Any]) -> dict[str, Any]:
    """A model's glyph fields from its record's ``wizard_answers`` (key names
    read from live records on 11 Nov 2026; value spellings not seen, so an
    unknown value is ``None`` or no badge, never a guess).

    A record whose fields cannot be read at all still gives its id and name,
    with every other field not recorded, so one odd record never hides the
    project's other models.
    """
    out: dict[str, Any] = {
        "id": record.get("id"),
        "name": record.get("name") or "(unnamed)",
        "grain": None,
        "kind": None,
        "size": None,
        "version": None,
        "sensors": [],
        "context_mode": None,
        "months": None,
        "context": None,
    }
    try:
        answers = record.get("wizard_answers")
        answers = answers if isinstance(answers, Mapping) else {}
        # The grain comes from prediction_type only: spatial_context is the
        # surrounding area (the patch), not the output's grain.
        grain, kind = _output(record, answers)
        temporal = _temporal(answers)
        out.update(
            grain=grain,
            kind=kind,
            size=_size(answers),
            version=_version(answers),
            sensors=sensor_badges(answers.get("imagery_sources")),
            context_mode=temporal["mode"],
            months=temporal["months"],
            context=temporal["context"],
        )
    except Exception:  # noqa: BLE001 - an unreadable record: "not recorded"
        logger.info("a model record could not be read for its glyph")
    return out
