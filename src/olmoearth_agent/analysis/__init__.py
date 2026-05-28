# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Analysis tools for the OlmoEarth agent.

Pure Python (no heavy deps), operating on data the agent already has:

- skill #6 (``olmoearth-change-detect``): the change-detection
  trajectory diff that refuses the two-date case which hides gradual
  drift (``SKILLS.md`` #6).
- skill #10 (``olmoearth-uncertainty``): the Meyer-Pebesma
  Area-of-Applicability out-of-distribution flag (``SKILLS.md`` #10).
"""

from olmoearth_agent.analysis.change_detect import (
    MIN_DATES,
    TooFewDatesError,
    diff_layers,
    enforce_min_3_dates,
)
from olmoearth_agent.analysis.uncertainty import (
    MIN_TRAIN,
    area_of_applicability,
    ood_flag,
)

__all__ = [
    "MIN_DATES",
    "MIN_TRAIN",
    "TooFewDatesError",
    "area_of_applicability",
    "diff_layers",
    "enforce_min_3_dates",
    "ood_flag",
]
