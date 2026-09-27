# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ziming Qi
"""
recommend.py — Decide embeddings vs fine-tuning for an OlmoEarth task.

Usage:
    python recommend.py --task "land cover, 9 classes, 200 samples, T4 GPU"
    python recommend.py --num-samples 50 --num-classes 4 --compute t4
    python recommend.py --num-samples 50000 --num-classes 9 --compute a100 --goal production

Emits JSON with the decision, recommended classifier (if embeddings), recommended
model size, and a one-liner suggesting the next command to run.

Stdlib only. The decision logic lives in `decide()` and follows the table in
references/when_to_use.md — change it there and here together.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Compute tiers, ordered weakest → strongest.
COMPUTE_TIERS = ["cpu", "t4", "colab", "free", "v100", "a100", "h100", "multi-gpu"]
STRONG_COMPUTE = {"a100", "h100", "multi-gpu"}

# Goals
PROTOTYPING = {"prototype", "prototyping", "explore", "exploring", "baseline", "sanity"}
PRODUCTION = {"production", "ship", "shipping", "deploy", "deployment", "sla"}
SIMILARITY = {"similarity", "similar", "search", "cluster", "clustering",
              "knn-only", "find-similar", "nearest neighbor", "nearest-neighbor"}
NO_LABELS = {"no-labels", "unlabeled", "no labels", "unlabelled"}


def parse_task_string(task: str) -> dict:
    """Best-effort parse of a free-form task description."""
    t = task.lower()
    out: dict = {}

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

    for kw in PROTOTYPING:
        if kw in t:
            out["goal"] = "prototyping"
            break
    for kw in PRODUCTION:
        if kw in t:
            out["goal"] = "production"
            break
    for kw in SIMILARITY:
        if kw in t:
            out["goal"] = "similarity"
            break
    for kw in NO_LABELS:
        if kw in t:
            out["goal"] = "no_labels"
            break

    return out


def decide(num_samples: int | None, num_classes: int | None,
           compute: str | None, goal: str | None) -> dict:
    """Apply the decision table from references/when_to_use.md."""

    # Hard overrides
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

    # Sample-size driven
    if num_samples is not None and num_samples < 100:
        return {
            "decision": "embeddings",
            "classifier": "kNN (cosine, k=min(20, len(train)))",
            "model": "base" if (num_classes or 0) > 5 else "tiny",
            "rationale": f"{num_samples} samples < 100 → fine-tuning would overfit. kNN on frozen features is the small-N standard.",
        }

    # Production + strong compute → fine-tune
    if goal == "production" and compute in STRONG_COMPUTE:
        return {
            "decision": "fine_tune",
            "classifier": None,
            "model": "base" if (num_classes or 0) > 5 or (num_samples or 0) < 2000 else "tiny",
            "rationale": "Production deployment + strong compute (A100+) → commit to full fine-tune (30+ epochs). Tutorial: ~82–87% vs ~70–75% for embeddings.",
        }

    # Production but weak compute → embeddings, with a flag
    if goal == "production" and compute not in STRONG_COMPUTE and compute is not None:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "tiny",
            "rationale": f"Production goal but compute={compute} can't sustain a 2–3 h fine-tune. Ship the LP baseline; flag the accuracy gap explicitly.",
            "warning": "Embeddings cap at ~70–75% on the tutorial benchmark. If your SLA needs more, you need stronger compute.",
        }

    # Mid-size dataset
    if num_samples is not None and 100 <= num_samples <= 2000:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "base" if (num_classes or 0) > 5 else "tiny",
            "rationale": f"{num_samples} samples (100–2000) → linear probe first. Calibrated baseline in minutes; fine-tune only if LP plateaus below requirement.",
            "next_step_after": "If LP < target accuracy, fine-tune the full 30-epoch schedule.",
        }

    # Large dataset
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

    # Compute-only signal
    if compute and compute not in STRONG_COMPUTE:
        return {
            "decision": "embeddings",
            "classifier": "linear_probe",
            "model": "tiny",
            "rationale": f"Compute={compute} → embeddings path. Fine-tune is impractical on this hardware.",
        }

    # Default: not enough information — recommend the safe baseline
    return {
        "decision": "embeddings",
        "classifier": "linear_probe",
        "model": "tiny",
        "rationale": "Insufficient info to recommend fine-tuning confidently. Default to embeddings + LP as the safe baseline; revisit after seeing results.",
        "ask_for": [
            x for x, v in [
                ("num_samples", num_samples),
                ("num_classes", num_classes),
                ("compute", compute),
                ("goal (prototyping vs production)", goal),
            ] if v is None
        ],
    }


def next_step_command(result: dict, dataset_path: str | None, num_classes: int | None,
                      class_names: str | None) -> str:
    if result["decision"] == "fine_tune":
        return "See the AWF tutorial Part B or `olmoearth_run` for the fine-tune pipeline."
    parts = ["python make_notebook.py"]
    if dataset_path:
        parts.append(f"--dataset-path {dataset_path}")
    if num_classes is not None:
        parts.append(f"--num-classes {num_classes}")
    if class_names:
        parts.append(f'--class-names "{class_names}"')
    parts.append(f"--model {result['model']}")
    parts.append("--out my_embeddings_workflow.ipynb")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recommend embeddings vs fine-tune for OlmoEarth.")
    p.add_argument("--task", help="Free-form task description.")
    p.add_argument("--num-samples", type=int, help="Labeled sample count.")
    p.add_argument("--num-classes", type=int, help="Number of classes.")
    p.add_argument("--compute", choices=COMPUTE_TIERS, help="Compute tier.")
    p.add_argument("--goal", choices=["prototyping", "production", "similarity", "no_labels"],
                   help="Workflow goal.")
    p.add_argument("--dataset-path", help="Forwarded to suggested make_notebook.py command.")
    p.add_argument("--class-names", help="Forwarded to suggested make_notebook.py command.")
    args = p.parse_args(argv)

    parsed = parse_task_string(args.task) if args.task else {}
    num_samples = args.num_samples if args.num_samples is not None else parsed.get("num_samples")
    num_classes = args.num_classes if args.num_classes is not None else parsed.get("num_classes")
    compute = args.compute or parsed.get("compute")
    goal = args.goal or parsed.get("goal")

    result = decide(num_samples, num_classes, compute, goal)
    result["next_step"] = next_step_command(result, args.dataset_path, num_classes, args.class_names)
    result["inputs"] = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "compute": compute,
        "goal": goal,
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
