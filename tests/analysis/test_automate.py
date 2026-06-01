# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Tests for the automate (embeddings-vs-fine-tune + config) logic."""

from __future__ import annotations

import json

import pytest

from olmoearth_agent.analysis.automate import (
    HF_DATASETS_SERVER,
    automate,
    decide,
    fetch_hf_dataset_profile,
    parse_hf_info_classes,
    parse_hf_size,
    parse_task_string,
    propose_config,
)

_SIZE = {"size": {"dataset": {"num_rows": 1500}}}
_INFO = {
    "dataset_info": {
        "default": {
            "features": {"label": {"_type": "ClassLabel", "names": ["a", "b", "c"]}}
        }
    }
}


def _fetch_map(mapping: dict[str, tuple[int, str]]):
    async def fetch(url: str, params: dict) -> tuple[int, str]:
        return mapping[url]

    return fetch


def test_decide_branches() -> None:
    assert decide(None, None, None, "similarity")["decision"] == "embeddings"
    assert decide(None, None, None, "no_labels")["classifier"].startswith("k-means")
    assert decide(50, 4, "t4", None)["decision"] == "embeddings"  # <100 -> kNN
    assert decide(50, 9, "t4", None)["model"] == "base"  # >5 classes -> base
    assert decide(50000, 9, "a100", "production")["decision"] == "fine_tune"
    assert decide(500, 4, "t4", None)["classifier"] == "linear_probe"  # 100-2000
    assert decide(5000, 9, "a100", None)["decision"] == "embeddings_then_fine_tune"
    default = decide(None, None, None, None)
    assert default["decision"] == "embeddings"
    assert set(default["ask_for"]) >= {"num_samples", "num_classes", "compute"}


def test_parse_task_string() -> None:
    out = parse_task_string("land cover, 9 classes, 200 samples, T4 GPU, production")
    assert out == {
        "num_samples": 200,
        "num_classes": 9,
        "compute": "t4",
        "goal": "production",
    }


def test_propose_config_embeddings_and_finetune() -> None:
    emb = propose_config(decide(50, 4, "t4", None), num_samples=50, num_classes=4)
    assert emb["approach"] == "embeddings"
    assert "--model tiny" in emb["embeddings_plan"]["notebook_command"]
    assert "--num-classes 4" in emb["embeddings_plan"]["notebook_command"]

    ft = propose_config(
        decide(50000, 9, "a100", "production"), num_samples=50000, num_classes=9
    )
    assert ft["approach"] == "fine_tune"
    assert ft["fine_tune_plan"]["studio_route"] == "olmoearth-studio-job-config"


def test_hf_payload_parsers() -> None:
    assert parse_hf_size(_SIZE) == 1500
    assert parse_hf_info_classes(_INFO) == 3
    assert parse_hf_size({}) is None
    assert parse_hf_info_classes({}) is None


@pytest.mark.asyncio
async def test_fetch_hf_dataset_profile() -> None:
    fetch = _fetch_map(
        {
            f"{HF_DATASETS_SERVER}/size": (200, json.dumps(_SIZE)),
            f"{HF_DATASETS_SERVER}/info": (200, json.dumps(_INFO)),
        }
    )
    prof = await fetch_hf_dataset_profile("org/ds", fetch=fetch)
    assert prof["num_samples"] == 1500
    assert prof["num_classes"] == 3
    assert prof["warnings"] == []


@pytest.mark.asyncio
async def test_fetch_hf_dataset_profile_handles_errors() -> None:
    fetch = _fetch_map(
        {
            f"{HF_DATASETS_SERVER}/size": (404, ""),
            f"{HF_DATASETS_SERVER}/info": (500, ""),
        }
    )
    prof = await fetch_hf_dataset_profile("org/missing", fetch=fetch)
    assert prof["num_samples"] is None and prof["num_classes"] is None
    assert len(prof["warnings"]) == 2


@pytest.mark.asyncio
async def test_automate_from_task_string() -> None:
    out = await automate(task="9 classes, 200 samples, t4")
    assert out["decision"] == "embeddings"  # 100-2000 band
    assert out["config"]["model_size"] == "base"  # 9 classes > 5
    assert out["inputs"]["num_samples"] == 200


@pytest.mark.asyncio
async def test_automate_introspects_hf_dataset() -> None:
    fetch = _fetch_map(
        {
            f"{HF_DATASETS_SERVER}/size": (200, json.dumps(_SIZE)),
            f"{HF_DATASETS_SERVER}/info": (200, json.dumps(_INFO)),
        }
    )
    out = await automate(
        hf_dataset="org/ds", compute="a100", goal="production", fetch=fetch
    )
    # 1500 samples + a100 + production -> production+strong -> fine_tune
    assert out["decision"] == "fine_tune"
    assert out["inputs"]["num_samples"] == 1500
    assert out["inputs"]["num_classes"] == 3


@pytest.mark.asyncio
async def test_automate_default_asks_for_inputs() -> None:
    out = await automate()
    assert out["decision"] == "embeddings"
    assert out["ask_for"]  # missing everything -> non-empty
