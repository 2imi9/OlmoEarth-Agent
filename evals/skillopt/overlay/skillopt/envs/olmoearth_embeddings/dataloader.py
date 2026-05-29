"""OlmoEarth job-config dataloader.

Each split directory (``train/``, ``val/``, ``test/``) holds an
``items.json`` array of task items::

    {
      "id": "crop_type_01",
      "task_description": "annual crop-type map for the Nile Delta from S2 ...",
      "task_type": "per_pixel_classification",
      "expected": { "output_type": ..., "foundation_model": ..., "time_frame": {...},
                    "imagery_sources": [...], "patch_size_m": 320 }
    }

The base :class:`SplitDataLoader` already reads a JSON array from the first
``.json`` file in each split directory, so no custom loading is needed.
"""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class OlmoEarthEmbeddingsDataLoader(SplitDataLoader):
    """Dataset-backed loader for the OlmoEarth Studio job-config benchmark."""
