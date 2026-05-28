# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Analysis tools for the OlmoEarth agent.

Pure Python (no heavy deps), operating on data the agent already has:

- skill #6 (``olmoearth-change-detect``): the change-detection
  trajectory diff that refuses the two-date case which hides gradual
  drift (``SKILLS.md`` #6).
- skill #7 (``olmoearth-baseline-compare``): OlmoEarth-vs-AlphaEarth
  per-metric comparison + difference raster (``SKILLS.md`` #7).
- skill #9 (``olmoearth-similarity``): embedding top-K similarity search
  with a geographic-prior warning (``SKILLS.md`` #9).
- skill #10 (``olmoearth-uncertainty``): the Meyer-Pebesma
  Area-of-Applicability out-of-distribution flag (``SKILLS.md`` #10).
- skill #11 (``olmoearth-cloud-mask-audit``): the cloud-mask ensemble
  disagreement audit + bad-mask-vs-bad-model verdict (``SKILLS.md`` #11).
"""

from olmoearth_agent.analysis.baseline import compare_metrics, difference_raster
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
from olmoearth_agent.analysis.similarity import (
    DEFAULT_K,
    DEFAULT_PRIOR_RADIUS_KM,
    geographic_prior_check,
    similarity_search,
)
from olmoearth_agent.analysis.uncertainty import (
    MIN_TRAIN,
    area_of_applicability,
    ood_flag,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_PRIOR_RADIUS_KM",
    "DEFAULT_VERDICT_THRESHOLD",
    "MIN_DATES",
    "MIN_MASKS",
    "MIN_TRAIN",
    "MaskShapeError",
    "TooFewDatesError",
    "area_of_applicability",
    "compare_metrics",
    "diff_layers",
    "difference_raster",
    "ensemble_disagree",
    "enforce_min_3_dates",
    "geographic_prior_check",
    "ood_flag",
    "similarity_search",
    "verdict_classifier",
]
