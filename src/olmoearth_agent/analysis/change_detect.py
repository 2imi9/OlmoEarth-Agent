# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Change-detection trajectory diff (skill #6, ``olmoearth-change-detect``).

Pure Python, no raster deps. The agent extracts one summary ``value`` per
prediction date (e.g. positive-class fraction or mean score over the AOI,
from a skill #5 result) and this module turns the dated series into
trajectory metrics: per-step deltas, net change, the largest-change
interval, a reversal count, and a trend label.

The headline behaviour is a refusal: a two-date diff reports net change
but cannot distinguish a steady trend from a reversal (a flood that
peaked then receded reads as "no change"), so :func:`enforce_min_3_dates`
requires at least three distinct dates (``SKILLS.md`` #6; Ma et al.
arXiv:2601.00857).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

#: Minimum distinct dates a trajectory needs; below this we refuse.
MIN_DATES = 3

#: Output numbers are rounded to this many places for clean JSON; the
#: trend/reversal logic always runs on the raw (unrounded) deltas so
#: rounding can never flip a sign.
_PLACES = 6


class TooFewDatesError(ValueError):
    """Raised when a change request has fewer than :data:`MIN_DATES` dates."""


def _parse_date(value: str) -> datetime:
    """Parse an ISO-8601 date or datetime (tolerating a trailing ``Z``)."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unparseable ISO-8601 date: {value!r}") from exc


def enforce_min_3_dates(dates: Sequence[str]) -> None:
    """Refuse a trajectory with fewer than three distinct dates.

    Parameters
    ----------
    dates
        The date strings of the trajectory points.

    Raises
    ------
    TooFewDatesError
        If there are fewer than :data:`MIN_DATES` distinct dates. A
        two-date diff hides gradual drift and cannot detect reversals, so
        the skill refuses it rather than publish a false-change report.
    """
    distinct = len(set(dates))
    if distinct < MIN_DATES:
        raise TooFewDatesError(
            f"change detection needs >= {MIN_DATES} distinct dates; got "
            f"{distinct}. A two-date diff hides gradual drift and cannot "
            "tell a steady trend from a reversal — add at least one "
            "intermediate date."
        )


def _sign(delta: float, tol: float) -> int:
    """Sign of ``delta``, treating ``|delta| <= tol`` as no change."""
    if delta > tol:
        return 1
    if delta < -tol:
        return -1
    return 0


def diff_layers(
    series: Sequence[tuple[str, float]], *, tol: float = 0.0
) -> dict[str, Any]:
    """Compute trajectory metrics from a dated series of layer summaries.

    Parameters
    ----------
    series
        ``(date, value)`` pairs, one per prediction date. ``date`` is
        ISO-8601; ``value`` is a per-date summary of the prediction layer
        (e.g. positive-class fraction or mean score). Order does not
        matter — the series is sorted by date before differencing.
    tol
        Absolute tolerance below which a step is treated as no change
        (default ``0.0``). Used only for the trend/reversal
        classification, not for the reported deltas.

    Returns
    -------
    dict
        ``n_dates``, sorted ``dates`` and aligned ``values``,
        ``step_deltas`` (consecutive differences), ``net_change``
        (last - first), ``max_step`` (the largest-magnitude interval and
        its delta), ``reversals`` (sign changes between steps),
        ``trend`` (``increasing`` / ``decreasing`` / ``stable`` /
        ``oscillating``), and ``monotonic`` (``reversals == 0``).

    Raises
    ------
    TooFewDatesError
        If fewer than :data:`MIN_DATES` distinct dates are supplied.
    ValueError
        If a date is unparseable or a date is duplicated.
    """
    enforce_min_3_dates([d for d, _ in series])
    if len({d for d, _ in series}) != len(series):
        raise ValueError(
            "duplicate dates are not allowed; each trajectory point must be "
            "a distinct time."
        )

    ordered = sorted(series, key=lambda dv: _parse_date(dv[0]))
    dates = [d for d, _ in ordered]
    values = [float(v) for _, v in ordered]

    raw_deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    signs = [_sign(d, tol) for d in raw_deltas]
    nonzero = [s for s in signs if s != 0]
    reversals = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)

    max_i = max(range(len(raw_deltas)), key=lambda i: abs(raw_deltas[i]))
    went_up = any(s > 0 for s in signs)
    went_down = any(s < 0 for s in signs)
    if went_up and went_down:
        trend = "oscillating"
    elif went_up:
        trend = "increasing"
    elif went_down:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "n_dates": len(dates),
        "dates": dates,
        "values": [round(v, _PLACES) for v in values],
        "step_deltas": [round(d, _PLACES) for d in raw_deltas],
        "net_change": round(values[-1] - values[0], _PLACES),
        "max_step": {
            "interval": [dates[max_i], dates[max_i + 1]],
            "delta": round(raw_deltas[max_i], _PLACES),
        },
        "reversals": reversals,
        "trend": trend,
        "monotonic": reversals == 0,
    }
