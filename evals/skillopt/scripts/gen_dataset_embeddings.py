"""Generate the olmoearth_embeddings benchmark from the skill's own oracle.

Ground truth = the vendored `recommend.py:decide()` so the benchmark scores the
target model against the exact rule the skill documents. Tasks are plain-English
paraphrases with the decisive signals (samples / classes / compute / goal)
embedded in prose, spanning every branch of `decide()`.

    python olmoearth_local/gen_dataset_embeddings.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT = os.path.join(os.path.dirname(ROOT), "OlmoEarth Agent")
ORACLE = os.path.join(
    AGENT, "vendor", "olmoearth-skills", "skills",
    "olmoearth-embeddings", "scripts", "recommend.py",
)

spec = importlib.util.spec_from_file_location("emb_recommend", ORACLE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[union-attr]

# (id, task_description, num_samples, num_classes, compute, goal)
TASKS: list[tuple] = [
    # similarity (hard override)
    ("sim_01", "I want to find areas that look like this reference wetland across the whole basin — just nearest-neighbor search, no training.", None, None, None, "similarity"),
    ("sim_02", "Build a similarity search over Sentinel-2 tiles to retrieve scenes resembling a query patch.", None, None, "t4", "similarity"),
    ("sim_03", "Find more sites like these three known illegal dumps; we have A100s but only need ranking.", None, None, "a100", "similarity"),
    # no labels (hard override)
    ("nolab_01", "I have no labels yet — cluster the AOI into natural groups, then hand-label the clusters.", None, None, None, "no_labels"),
    ("nolab_02", "Unlabeled imagery over a delta; discover structure via clustering before any annotation.", None, None, "v100", "no_labels"),
    ("nolab_03", "Thousands of unlabeled chips and no annotations — group them before we start labeling.", 5000, None, None, "no_labels"),
    # < 100 samples
    ("small_01", "Crop-type mapping but I only have 50 labeled field polygons across 4 classes, running on a T4.", 50, 4, "t4", None),
    ("small_02", "Tiny pilot: 30 labeled landslide scars, binary, on Colab free tier.", 30, 2, "colab", None),
    ("small_03", "I have 80 labeled samples spread over 9 land-cover classes, on a single V100.", 80, 9, "v100", None),
    ("small_04", "Only 60 annotated points for a 7-class wetland typology, basically no GPU.", 60, 7, "cpu", None),
    ("small_05", "45 labeled samples, 3 classes, and an A100 sitting idle.", 45, 3, "a100", None),
    # production + strong compute → fine-tune
    ("prodstrong_01", "Production deployment of a 3-class flood model, about 30k samples, we have A100s.", 30000, 3, "a100", "production"),
    ("prodstrong_02", "Ship a 12-class land-cover product at SLA; 50k labels; H100 cluster available.", 50000, 12, "h100", "production"),
    ("prodstrong_03", "Deploy a canopy-height regressor to production; only 1500 labels; multi-GPU node.", 1500, None, "multi-gpu", "production"),
    ("prodstrong_04", "Production crop classifier, 8 classes, 100k samples, A100.", 100000, 8, "a100", "production"),
    ("prodstrong_05", "Production, exactly 2000 samples, 4 classes, on A100.", 2000, 4, "a100", "production"),
    # production + weak compute → embeddings + LP (+warning)
    ("prodweak_01", "We need to ship a mangrove model next week but only have a T4 — production goal.", None, None, "t4", "production"),
    ("prodweak_02", "Production-bound soil-moisture model, but compute is just a Colab free tier.", None, None, "colab", "production"),
    ("prodweak_03", "Deploy a 2-class burn-scar detector; only a V100 available.", None, 2, "v100", "production"),
    # 100–2000 samples → embeddings + LP
    ("mid_01", "Land-cover mapping with about 800 labeled samples, 5 classes, on a V100.", 800, 5, "v100", None),
    ("mid_02", "Mangrove extent, roughly 1200 labels, binary, T4.", 1200, 2, "t4", None),
    ("mid_03", "Ecosystem typology, 1500 samples, 8 classes, V100.", 1500, 8, "v100", None),
    ("mid_04", "Around 500 labeled plots for a 3-class problem; modest compute.", 500, 3, "colab", None),
    ("mid_05", "Exactly 100 labeled samples, 4 classes, V100.", 100, 4, "v100", None),
    # > 2000 + strong compute → embeddings_then_fine_tune
    ("bigstrong_01", "12-class crop map, 40k samples, A100 — want a fast baseline before committing.", 40000, 12, "a100", None),
    ("bigstrong_02", "Big dataset, 25k labels, 4 classes, H100 cluster.", 25000, 4, "h100", None),
    ("bigstrong_03", "Tree-height regression, 10k samples, multi-GPU.", 10000, None, "multi-gpu", None),
    ("bigstrong_04", "60k samples, 20 classes, multi-GPU; want quick validation first.", 60000, 20, "multi-gpu", None),
    # > 2000 + weak compute → embeddings + LP (tiny)
    ("bigweak_01", "Land cover, 8000 samples, 6 classes, but only a T4.", 8000, 6, "t4", None),
    ("bigweak_02", "30k labels, 3 classes, on a CPU-only box.", 30000, 3, "cpu", None),
    ("bigweak_03", "Crop mapping with 5000 labeled fields, Colab free.", 5000, None, "colab", None),
    # compute-only weak signal
    ("computeonly_01", "I'm on a T4 — what's the right approach for an EO classifier?", None, None, "t4", None),
    ("computeonly_02", "Only have Colab free tier for this satellite project.", None, None, "colab", None),
    # default — insufficient info
    ("default_01", "Help me pick an approach for a land-cover model.", None, None, None, None),
    ("default_02", "What should I do for my Sentinel-2 classification task?", None, None, None, None),
]


def main() -> None:
    items = []
    for id_, desc, ns, nc, comp, goal in TASKS:
        d = mod.decide(ns, nc, comp, goal)
        items.append({
            "id": id_,
            "task_description": desc,
            "task_type": d["decision"],
            "inputs": {"num_samples": ns, "num_classes": nc, "compute": comp, "goal": goal},
            "expected": {
                "decision": d["decision"],
                "model": d["model"],
                "classifier": d.get("classifier"),
            },
        })

    rng = random.Random(42)
    rng.shuffle(items)
    n = len(items)
    n_train, n_val = 14, 9
    splits = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
    base = os.path.join(ROOT, "data", "olmoearth_embeddings_split")
    for split, rows in splits.items():
        d = os.path.join(base, split)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "items.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"total={n}  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    from collections import Counter
    print("decision dist:", dict(Counter(i["expected"]["decision"] for i in items)))
    print("model dist:   ", dict(Counter(i["expected"]["model"] for i in items)))


if __name__ == "__main__":
    main()
