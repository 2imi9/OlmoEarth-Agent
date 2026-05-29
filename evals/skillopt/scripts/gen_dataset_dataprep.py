"""Generate the olmoearth_dataprep pitfall-diagnosis benchmark.

Each task is a plain-English description of a data-prep situation/error; the
expected answer is which of the skill's 8 documented pitfalls it is plus the
corrective action family. Scenarios are generic EO symptoms (no project-specific
fixtures) tied to the pitfalls table in the data-prep SKILL.md.

    python olmoearth_local/gen_dataset_dataprep.py
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (id, description, pitfall_id, action_family)
TASKS = [
    # 1 — wrong field names / schema confusion / es_*→oe_labels rename
    ("schema_01", "Studio rejected my upload — my GeoJSON features use a `label` property for the class and a `tag` field for the category.", 1, "fix_schema"),
    ("schema_02", "I exported annotations from Studio and the features have `es_label` and `es_start_time`; olmoearth_run can't load them.", 1, "fix_schema"),
    ("schema_03", "My labels carry a top-level `category` field but label loading silently fails.", 1, "fix_schema"),
    ("schema_04", "Which property does olmoearth_run actually read for the class label, and what do I rename my fields to?", 1, "fix_schema"),
    # 2 — bbox AOIs instead of real watersheds
    ("aoi_01", "For my watershed water-quality study I drew rectangular bounding boxes around each gauge station.", 2, "real_aoi"),
    ("aoi_02", "I'm using bbox AOIs around each lake; the embeddings seem polluted by surrounding farmland.", 2, "real_aoi"),
    ("aoi_03", "Should I use a square around each station point, or fetch the actual basin polygon?", 2, "real_aoi"),
    # 3 — Studio range-locking when uploading multiple metrics
    ("metric_01", "I uploaded one file containing three different metrics and Studio locked the value range across all of them.", 3, "per_metric_file"),
    ("metric_02", "My import bundles turbidity, chlorophyll, and temperature in one labelset and the legend/scale is wrong.", 3, "per_metric_file"),
    ("metric_03", "Uploading multiple metrics at once gives them a single shared color ramp.", 3, "per_metric_file"),
    # 4 — .geojson MIME-rejected as octet-stream
    ("mime_01", "On Windows, my `.geojson` upload is rejected as application/octet-stream.", 4, "emit_json"),
    ("mime_02", "Studio won't accept my file — the browser sends it as octet-stream instead of JSON.", 4, "emit_json"),
    ("mime_03", "What extension should I save my GeoJSON as so the browser doesn't MIME-reject it?", 4, "emit_json"),
    # 5 — quantile binning gave severe imbalance
    ("bin_01", "I binned my continuous target into quartiles and got a 96 / 2.5 / 1.3 / 0.1 % class split.", 5, "equal_frequency"),
    ("bin_02", "My regression-as-classification bins are wildly imbalanced after quantile binning.", 5, "equal_frequency"),
    ("bin_03", "How should I bin a continuous variable into classes so the classes aren't lopsided?", 5, "equal_frequency"),
    # 6 — random splits inflate accuracy
    ("split_01", "I did a random 80/20 train/val split and my reported accuracy looks suspiciously high.", 6, "spatial_split"),
    ("split_02", "With a random split my validation points sit right next to training points.", 6, "spatial_split"),
    ("split_03", "How do I split labels so accuracy isn't inflated by spatial autocorrelation?", 6, "spatial_split"),
    # 7 — too many records time out the 1-hour Studio upload
    ("size_01", "My 14,000-record upload to Studio times out after an hour.", 7, "shard"),
    ("size_02", "I have 30k label features and the import never finishes.", 7, "shard"),
    ("size_03", "I'm hitting Studio's 1-hour upload limit with too many records — what do I do?", 7, "shard"),
    # 8 — class imbalance with no negative class
    ("neg_01", "My classifier flags positives everywhere — I only have target-class labels and no background.", 8, "negative_class"),
    ("neg_02", "I trained on presence-only points and get tons of false positives.", 8, "negative_class"),
    ("neg_03", "Do I need a non-target / background class to prevent false positives?", 8, "negative_class"),
]


def main() -> None:
    items = [
        {"id": id_, "task_description": desc, "task_type": f"pitfall_{pid}",
         "expected": {"pitfall_id": pid, "action": action}}
        for id_, desc, pid, action in TASKS
    ]
    rng = random.Random(42)
    rng.shuffle(items)
    n = len(items)
    n_train, n_val = 10, 7
    splits = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
    base = os.path.join(ROOT, "data", "olmoearth_dataprep_split")
    for split, rows in splits.items():
        d = os.path.join(base, split)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "items.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"total={n}  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("pitfall dist:", dict(sorted(Counter(i['expected']['pitfall_id'] for i in items).items())))


if __name__ == "__main__":
    main()
