# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Honest map-accuracy evaluation for skill #7 (``olmoearth-evaluate``).

Pure-Python (no heavy deps) spatial cross-validation and metrics. The
headline is the random-vs-spatial inflation diagnostic that
operationalizes Ploton et al. 2020 (Nat. Commun. 11:4540) and
Meyer & Pebesma 2021 (MEE 12:1620): random k-fold CV reports optimistic
accuracy on spatially autocorrelated data because test points sit next
to training points.
"""

from olmoearth_agent.evaluation.metrics import classification_metrics
from olmoearth_agent.evaluation.spatial_cv import (
    cv_inflation_diagnostic,
    haversine_km,
    random_folds,
    spatial_block_folds,
)

__all__ = [
    "classification_metrics",
    "cv_inflation_diagnostic",
    "haversine_km",
    "random_folds",
    "spatial_block_folds",
]
