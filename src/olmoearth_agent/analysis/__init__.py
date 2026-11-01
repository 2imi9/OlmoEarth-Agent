# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Analysis layer of the OlmoEarth agent.

Pure Python (no heavy deps), operating on data the agent already has. Each
module is imported by its own path (for example
``olmoearth_agent.analysis.raster_compare``); this package re-exports
nothing, so importing one module never loads the others.
"""
