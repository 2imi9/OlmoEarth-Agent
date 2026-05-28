# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Temporal analysis for the OlmoEarth agent.

Currently the change-detection trajectory diff for skill #6
(``olmoearth-change-detect``). Pure Python (no raster deps): the agent
extracts one summary ``value`` per prediction date and this module turns
the dated series into trajectory metrics, refusing the two-date case
that hides gradual drift (``SKILLS.md`` #6).
"""

from olmoearth_agent.analysis.change_detect import (
    MIN_DATES,
    TooFewDatesError,
    diff_layers,
    enforce_min_3_dates,
)

__all__ = [
    "MIN_DATES",
    "TooFewDatesError",
    "diff_layers",
    "enforce_min_3_dates",
]
