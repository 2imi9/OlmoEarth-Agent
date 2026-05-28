# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Analysis tools for the OlmoEarth agent.

Pure Python (no raster deps), operating on data the agent already has:

- skill #6 (``olmoearth-change-detect``): the change-detection
  trajectory diff that refuses the two-date case which hides gradual
  drift (``SKILLS.md`` #6).
- skill #11 (``olmoearth-cloud-mask-audit``): the cloud-mask ensemble
  disagreement audit + bad-mask-vs-bad-model verdict (``SKILLS.md`` #11).
"""

from olmoearth_agent.analysis.change_detect import (
    MIN_DATES,
    TooFewDatesError,
    diff_layers,
    enforce_min_3_dates,
)
from olmoearth_agent.analysis.cloud_mask import (
    DEFAULT_VERDICT_THRESHOLD,
    MIN_MASKS,
    MaskShapeError,
    ensemble_disagree,
    verdict_classifier,
)

__all__ = [
    "DEFAULT_VERDICT_THRESHOLD",
    "MIN_DATES",
    "MIN_MASKS",
    "MaskShapeError",
    "TooFewDatesError",
    "diff_layers",
    "ensemble_disagree",
    "enforce_min_3_dates",
    "verdict_classifier",
]
