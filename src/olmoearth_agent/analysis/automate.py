# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Auto-decide embeddings vs fine-tuning and propose a full config (the in-repo facet of skill #3 olmoearth-embeddings).

The decision logic (:func:`decide`, :func:`parse_task_string`) is a faithful
port of the canonical
``vendor/olmoearth-skills/skills/olmoearth-embeddings/scripts/recommend.py``
(the source of truth for the embeddings-vs-fine-tune call, also used by the
SkillOpt harness). Keep the two in sync — change the vendored table and this
port together. On top of the decision this module adds:

* :func:`propose_config` — turns the decision into an actionable plan
  (model size, classifier head, an embeddings-notebook command, a fine-tune
  schedule, and a hand-off to ``olmoearth-studio-job-config`` for the Studio
  wizard) rather than just a yes/no.
* :func:`fetch_hf_dataset_profile` — best-effort introspection of a Hugging
  Face dataset (row count + ClassLabel classes) via the public
  datasets-server, so a user can point at a dataset instead of typing counts.

All inputs are task metadata (counts, compute tier, goal) — no geometry — so
results are provenance-safe.
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from olmoearth_agent.security import egress

# --- compute tiers + goal vocab (ported from the vendored recommend.py) ---
COMPUTE_TIERS = ["cpu", "t4", "colab", "free", "v100", "a100", "h100", "multi-gpu"]
STRONG_COMPUTE = {"a100", "h100", "multi-gpu"}
_PROTOTYPING = {
    "prototype",
    "prototyping",
    "explore",
    "exploring",
    "baseline",
    "sanity",
}
_PRODUCTION = {"production", "ship", "shipping", "deploy", "deployment", "sla"}
_SIMILARITY = {
    "similarity",
    "similar",
    "search",
    "cluster",
    "clustering",
    "knn-only",
    "find-similar",
    "nearest neighbor",
    "nearest-neighbor",
}
_NO_LABELS = {"no-labels", "unlabeled", "no labels", "unlabelled"}

#: HF datasets-server (public; no key for public datasets).
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"

#: Injected async getter ``(url, params) -> (status, body_text)`` for testability.
Fetcher = Callable[[str, dict[str, Any]], Awaitable[tuple[int, str]]]
_TIMEOUT_SECONDS = 20.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_USER_AGENT = "OlmoEarth-Agent-automate (+https://github.com/2imi9/OlmoEarth-Agent)"


def parse_task_string(task: str) -> dict[str, Any]:
    """Best-effort parse of a free-form task description (port of recommend.py)."""
    t = task.lower()
    out: dict[str, Any] = {}
    m = re.search(r"(\d[\d,]*)\s*(?:samples|labels|points|examples)", t)
    if m:
        out["num_samples"] = int(m.group(1).replace(",", ""))
    m = re.search(r"(\d+)\s*classes?\b", t)
    if m:
        out["num_classes"] = int(m.group(1))
    for tier in COMPUTE_TIERS:
        if tier in t:
            out["compute"] = tier
            break
    for kw in _SIMILARITY:
        if kw in t:
            out["goal"] = "similarity"
            break
    for kw in _NO_LABELS:
        if kw in t:
            out["goal"] = "no_labels"
            break
    for kw in _PRODUCTION:
        if kw in t:
            out.setdefault("goal", "production")
            break
    for kw in _PROTOTYPING:
        if kw in t:
            out.setdefault("goal", "prototyping")
            break
    return out


def decide(
    num_samples: int | None,
    num_classes: int | None,
    compute: str | None,
    goal: str | None,
) -> dict[str, Any]:
    """Embeddings vs fine-tune decision (port of the vendored ``recommend.decide``)."""
    if goal == "similarity":
        return {
            "decision": "embeddings",
            "classifier": "none (cosine kNN ranking)",
            "model": "tiny",
            "rationale": "Similarity search needs raw embeddings, not a classifier head. L2-normalize and rank by cosine.",
        }
    if goal == "no_labels":
        return {
            "decision": "embeddings",
            "classifier": "k-means or HDBSCAN",
            "model": "tiny",
            "rationale": "No labels — fine-tuning is not on the table. Cluster embeddings to discover structure, then hand-label clusters.",
        }
    if num_samples is not None and num_samples < 100:
        return {
            "decision": "embeddings",
            "classifier": "kNN (cosine, k=min(20, len(train)))",
            "model": "base" if (num_classes or 0) > 5 else "tiny",
            "rationale": f"{num_samples} samples < 100 → fine-tuning would overfit. kNN on frozen features is the small-N standard.",
        }
    if goal == "production" and compute in STRONG_COMPUTE:
        return {
            "decision": "fine_tune",
            "classifier": None,
            "model": (
                "base"
                if (num_classes or 0) > 5 or (num_samples or 0) < 2000
                else "tiny"
            ),
            "rationale": "Production deployment + strong compute (A100+) → commit to full fine-tune (30+ epochs). Tutorial: ~82–87% vs ~70–75% for embeddings.",
        }
    if goal == "production" and compute not in STRONG_COMPUTE and compute is not None:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "tiny",
            "rationale": f"Production goal but compute={compute} can't sustain a 2–3 h fine-tune. Ship the LP baseline; flag the accuracy gap explicitly.",
            "warning": "Embeddings cap at ~70–75% on the tutorial benchmark. If your SLA needs more, you need stronger compute.",
        }
    if num_samples is not None and 100 <= num_samples <= 2000:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "base" if (num_classes or 0) > 5 else "tiny",
            "rationale": f"{num_samples} samples (100–2000) → linear probe first. Calibrated baseline in minutes; fine-tune only if LP plateaus below requirement.",
            "next_step_after": "If LP < target accuracy, fine-tune the full 30-epoch schedule.",
        }
    if num_samples is not None and num_samples > 2000:
        if compute in STRONG_COMPUTE:
            return {
                "decision": "embeddings_then_fine_tune",
                "classifier": "linear_probe (validation baseline)",
                "model": "base" if (num_classes or 0) > 5 else "tiny",
                "rationale": f"{num_samples} samples + strong compute → run embeddings first to validate the pipeline (minutes), then commit to fine-tune (2–3 h).",
            }
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "tiny",
            "rationale": f"{num_samples} samples but compute is limited → LP is the best you can sustainably run. Revisit fine-tune when you get better compute.",
        }
    if compute and compute not in STRONG_COMPUTE:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "tiny",
            "rationale": f"Compute={compute} → embeddings path. Fine-tune is impractical on this hardware.",
        }
    return {
        "decision": "embeddings",
        "classifier": "linear_probe",
        "model": "tiny",
        "rationale": "Insufficient info to recommend fine-tuning confidently. Default to embeddings + LP as the safe baseline; revisit after seeing results.",
        "ask_for": [
            x
            for x, v in [
                ("num_samples", num_samples),
                ("num_classes", num_classes),
                ("compute", compute),
                ("goal (prototyping vs production)", goal),
            ]
            if v is None
        ],
    }


def propose_config(
    rec: dict[str, Any],
    *,
    num_samples: int | None,
    num_classes: int | None,
) -> dict[str, Any]:
    """Turn a :func:`decide` result into an actionable plan + Studio hand-off."""
    decision = rec["decision"]
    model = rec["model"]
    plan: dict[str, Any] = {
        "approach": decision,
        "model_size": model,
        "classifier": rec.get("classifier"),
    }

    if decision in ("embeddings", "embeddings_then_fine_tune"):
        nb = ["python make_notebook.py", f"--model {model}"]
        if num_classes is not None:
            nb.append(f"--num-classes {num_classes}")
        nb.append("--out my_embeddings_workflow.ipynb")
        plan["embeddings_plan"] = {
            "normalize": "cosine" in str(rec.get("classifier", "")).lower()
            or "none" in str(rec.get("classifier", "")).lower(),
            "head": rec.get("classifier"),
            "notebook_command": " ".join(nb),
            "via_skill": "olmoearth-embeddings",
        }
    if decision in ("fine_tune", "embeddings_then_fine_tune"):
        plan["fine_tune_plan"] = {
            "epochs": 30,
            "encoder": f"OlmoEarth {model.capitalize()}",
            "studio_route": "olmoearth-studio-job-config",
            "note": "Use the olmoearth-studio-job-config skill for the Studio wizard "
            "(output type, imagery sources, time frame, patch size).",
        }
    return plan


def _hf_label_classes(features: Any) -> int | None:
    """Find a ClassLabel feature's class count anywhere in a features tree."""
    if isinstance(features, dict):
        if features.get("_type") == "ClassLabel" and isinstance(
            features.get("names"), list
        ):
            return len(features["names"])
        for value in features.values():
            found = _hf_label_classes(value)
            if found is not None:
                return found
    elif isinstance(features, list):
        for value in features:
            found = _hf_label_classes(value)
            if found is not None:
                return found
    return None


def parse_hf_size(payload: dict[str, Any]) -> int | None:
    """Total row count from a datasets-server ``/size`` payload."""
    size = (payload.get("size") or {}).get("dataset") or {}
    rows = size.get("num_rows")
    return int(rows) if isinstance(rows, (int, float)) else None


def parse_hf_info_classes(payload: dict[str, Any]) -> int | None:
    """ClassLabel class count from a datasets-server ``/info`` payload."""
    info = payload.get("dataset_info") or {}
    # /info nests per-config; scan each config's features.
    configs = info.values() if isinstance(info, dict) else []
    for cfg in configs:
        if isinstance(cfg, dict):
            found = _hf_label_classes(cfg.get("features"))
            if found is not None:
                return found
    return _hf_label_classes(info.get("features")) if isinstance(info, dict) else None


def _httpx_fetcher(client: httpx.AsyncClient) -> Fetcher:
    import asyncio

    async def fetch(url: str, params: dict[str, Any]) -> tuple[int, str]:
        egress.validate_endpoint(url, "hf")
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError:
                if attempt >= _MAX_ATTEMPTS - 1:
                    raise
            else:
                if not (
                    resp.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1
                ):
                    return resp.status_code, resp.text
            await asyncio.sleep(0.4 * (attempt + 1))
        raise RuntimeError("unreachable retry state")  # pragma: no cover

    return fetch


async def fetch_hf_dataset_profile(
    dataset: str, *, fetch: Fetcher | None = None
) -> dict[str, Any]:
    """Best-effort {num_samples, num_classes, warnings} for a HF dataset.

    Uses the public datasets-server ``/size`` and ``/info``. Any failure is
    reported in ``warnings`` rather than raised, so the caller can still decide
    from whatever signals it has.
    """
    dataset = (dataset or "").strip()
    if not dataset:
        raise ValueError("dataset id is required (e.g. 'org/name').")

    async def _run(owned: Fetcher) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset": dataset,
            "num_samples": None,
            "num_classes": None,
            "warnings": [],
        }
        try:
            status, body = await owned(
                f"{HF_DATASETS_SERVER}/size", {"dataset": dataset}
            )
            if status == 200:
                out["num_samples"] = parse_hf_size(_json.loads(body))
            else:
                out["warnings"].append(f"size: HTTP {status}")
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            out["warnings"].append(f"size: {type(exc).__name__}: {exc}")
        try:
            status, body = await owned(
                f"{HF_DATASETS_SERVER}/info", {"dataset": dataset}
            )
            if status == 200:
                out["num_classes"] = parse_hf_info_classes(_json.loads(body))
            else:
                out["warnings"].append(f"info: HTTP {status}")
        except Exception as exc:  # noqa: BLE001
            out["warnings"].append(f"info: {type(exc).__name__}: {exc}")
        return out

    if fetch is not None:
        return await _run(fetch)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        return await _run(_httpx_fetcher(client))


async def automate(
    *,
    task: str | None = None,
    num_samples: int | None = None,
    num_classes: int | None = None,
    compute: str | None = None,
    goal: str | None = None,
    hf_dataset: str | None = None,
    fetch: Fetcher | None = None,
) -> dict[str, Any]:
    """Auto-decide approach + propose a config from a task and/or a HF dataset."""
    parsed = parse_task_string(task) if task else {}
    warnings: list[str] = []
    if hf_dataset:
        prof = await fetch_hf_dataset_profile(hf_dataset, fetch=fetch)
        warnings.extend(prof.get("warnings", []))
        if num_samples is None and parsed.get("num_samples") is None:
            num_samples = prof.get("num_samples")
        if num_classes is None and parsed.get("num_classes") is None:
            num_classes = prof.get("num_classes")

    num_samples = num_samples if num_samples is not None else parsed.get("num_samples")
    num_classes = num_classes if num_classes is not None else parsed.get("num_classes")
    compute = compute or parsed.get("compute")
    goal = goal or parsed.get("goal")

    rec = decide(num_samples, num_classes, compute, goal)
    return {
        "inputs": {
            "num_samples": num_samples,
            "num_classes": num_classes,
            "compute": compute,
            "goal": goal,
            "hf_dataset": hf_dataset,
        },
        "decision": rec["decision"],
        "rationale": rec["rationale"],
        "config": propose_config(rec, num_samples=num_samples, num_classes=num_classes),
        "warning": rec.get("warning"),
        "ask_for": rec.get("ask_for", []),
        "warnings": warnings,
    }
