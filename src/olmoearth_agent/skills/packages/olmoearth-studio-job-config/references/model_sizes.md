# Foundation model size — Nano / Tiny / Base

OlmoEarth's foundation model is a Vision Transformer (ViT) pre-trained on Sentinel-1/2/Landsat composites. Studio offers three sizes; pick by *representation-quality need*, not by dataset size alone.

**Studio trains on Ai2's compute.** Model size is a quality vs Studio-job cost/turnaround tradeoff — never a question about the user's own hardware. Do not ask what GPU they have.

| Model | Parameters | Speed-up vs Base | When to pick |
|-------|------------|------------------|--------------|
| **Nano** | ~1.4M | ~60× faster training, ~30× cheaper inference | Binary tasks (water/not-water, mangrove/not), embeddings at continental scale, when Studio-job cost/turnaround is the binding constraint |
| **Tiny** | ~6.2M | ~14× faster training | **Default for most fine-tuning.** Good balance of cost and representation for 2–5 class problems with 1K+ samples. |
| **Base** | ~90M | 1× (baseline) | Multi-class (>5) segmentation, small or imbalanced datasets, fine-grained detection (small objects), regression with high label variance |

Approximate parameter counts come from the `allenai/olmoearth_pretrain` repo on GitHub — confirm exact figures and download links from there before quoting them to the user.

## Heuristic decision tree

```
Number of label classes?
├── 1 (regression) or 2 (binary)
│   ├── <2K samples and job-cost-sensitive → Nano
│   ├── 2–20K samples → Tiny
│   └── >20K samples + high variance → Base (only if Tiny plateaus)
│
├── 3–5 classes
│   ├── Balanced, >1K per class → Tiny
│   └── Imbalanced or <500 per class → Base
│
└── >5 classes
    └── Almost always → Base
```

## Why these are heuristics, not rules

- **Nano** sometimes wins on dataset sizes Base would dominate, because more params overfit small data — but only after careful regularization. Default to Tiny first.
- **Base** sometimes loses on huge datasets to Tiny — bottleneck shifts from representation to optimization. Worth a Base-vs-Tiny ablation if the job budget allows.
- **Tiny** is the right starting point for almost every task. Train Tiny first, then upgrade to Base only if you hit a clear ceiling.

## Training/inference cost rough orders

These are *Studio-side* job costs (Ai2's compute) — they shape job price and turnaround, not anything the user runs. Take them as ballpark; actual numbers depend on AOI size, patch size, time-frame length, and modality count:

| Model | Train (10K samples, 30 epochs) | Inference (10K km²) |
|-------|-------------------------------|---------------------|
| Nano | ~10 min | seconds |
| Tiny | ~1 h | low minutes |
| Base | ~12 h | 10–30 min |

Multi-modal (S2 + S1) and longer time frames (12 months > 3 months) push these proportionally higher.

## Pairing with the wizard's other fields

- **Detection + small objects (vessels, oil drums)** — Base helps more than for classification, because small-object features are harder to represent.
- **Embeddings** — Tiny is usually the sweet spot. Nano embeddings (128-dim) are fine for clustering but lose fine-grain similarity. Base (768-dim) is overkill unless the downstream model is also large.
- **Single moment + before context** — Base is more sample-efficient on the temporal stack since it has more capacity to use the extra months; if your only signal is temporal, the model-size gain is real.

## What this skill won't decide for you

- Whether to upgrade Nano → Tiny → Base mid-project. That's an *ablation* decision; recommend running both and comparing val accuracy.
- Whether to use a non-OE foundation model. Out of scope; this skill is OlmoEarth-specific.
