# OlmoEarth Embeddings Recommender + Notebook Generator

OlmoEarth's foundation models (Nano / Tiny / Base / Large) produce dense feature vectors from satellite imagery without any task-specific training. For many problems — small labeled datasets, rapid prototyping, similarity search, clustering, weak-label workflows — frozen embeddings plus a simple classifier (kNN, linear probe, MLP head) beat full fine-tuning on both wall-clock time and GPU memory, and sometimes on accuracy too.

This skill does two things:

1. **Decides** whether embeddings or fine-tuning is right for the user's task, with rationale grounded in the official AllenAI OlmoEarth tutorial.
2. **Generates** a runnable Jupyter notebook (`make_notebook.py`) that extracts embeddings for the user's rslearn dataset and trains kNN + linear-probe heads — adapted from the tutorial but parameterized for the user's paths, class count, and model size.

## Quick decision rules

Work through these in order — the **first** rule that matches wins. They make the precedence in the tables below explicit (the model otherwise tends to jump to fine-tuning and to over-pick Base).

1. **Goal overrides everything.** Similarity / search / retrieval → **embeddings, no classifier** (L2-normalize, rank by cosine), Tiny. No labels yet → **embeddings + clustering** (k-means / HDBSCAN), Tiny. Ignore sample-count and compute for these.
2. **Fewer than 100 labeled samples → embeddings + kNN** (cosine). Fine-tuning overfits at this size; use kNN, not a linear probe.
3. **Compute gates fine-tuning.** Full fine-tuning — and the *embeddings-then-fine-tune* staged path — are only on the table with **strong compute: A100 / H100 / multi-GPU**. On a **T4, V100, Colab/free tier, or CPU**, the answer is **embeddings + linear probe**, *regardless of dataset size or a production goal*. Recommend the linear-probe baseline and flag the accuracy gap explicitly rather than promising a fine-tune the hardware cannot sustain.
4. **Production goal + strong compute → fine_tune.** Commit to the full 30-epoch schedule.
5. **More than 2000 samples + strong compute → embeddings_then_fine_tune.** Run the embeddings + linear-probe baseline first to validate the data and pipeline (minutes), *then* fine-tune.
6. **100–2000 samples → embeddings + linear probe.** Fine-tune later only if the probe plateaus below the requirement.
7. **Otherwise** (weak or missing signals) → **embeddings + linear probe** as the safe baseline; name the missing inputs (samples, classes, compute, goal).

**Model size (applies to every branch above):** default **Tiny**. Choose **Base only when classes > 5 OR samples < 2000** — representation quality dominates with many classes or little data. A large dataset on few classes stays **Tiny** even when fine-tuning.

## When to use this skill

Trigger on any of:

- "Should I use embeddings or fine-tune for X?"
- Dataset-size questions: "I only have 50 / 200 / 1000 labels — what should I do?"
- Compute-constraint questions: "I'm on a T4 / Colab free tier / no GPU"
- Speed questions: "How fast can I get a baseline?"
- Workflow questions: "I want similarity search / find similar areas / cluster the AOI"
- Generation requests: "Make me a notebook that extracts OlmoEarth embeddings on my data"
- Comparison questions: "kNN vs linear probe", "Nano embeddings vs Base embeddings"

If the user is choosing **Studio-side** settings (output type, patch size, model size at job-creation time), route to `olmoearth-studio-job-config` instead. This skill operates at the *Python code / notebook* level.

If the user is preparing labels (schema, AOI, audit), route to `olmoearth-data-prep` first.

## Decision: embeddings vs fine-tuning

The right answer comes from the user's task, not their preference. Use this table (verified directly against the tutorial's "When to Use Which Approach?" cell):

| Scenario | Recommended |
|----------|-------------|
| Rapid prototyping, exploring the data | **Embeddings + kNN/LP** |
| Limited GPU resources (T4, Colab free, no GPU) | **Embeddings + kNN/LP** |
| Small labeled dataset (<100 samples) | **Embeddings + kNN/LP** |
| Similarity search / clustering / nearest-neighbor lookup | **Embeddings** (no classifier needed) |
| Weak labels or no labels yet | **Embeddings** + downstream weak-label pipeline |
| Maximum accuracy needed | **Fine-tuning** |
| Production deployment | **Fine-tuning** |
| Task very different from pre-training (e.g., SAR-only, thermal-only) | **Fine-tuning** |
| Need to update model with new modalities | **Fine-tuning** |

**Cost / accuracy reference** (from the tutorial's "Why Two Approaches?" table, AWF Kenya LULC, T4 GPU):

| Approach | Accuracy | Time | GPU memory |
|----------|----------|------|------------|
| Embeddings + kNN/LP | ~70–75% | Minutes | ~2–3 GB |
| Fine-tuning (4 epochs) | ~70–75% | ~15–20 min | ~4–6 GB |
| Fine-tuning (30 epochs) | ~82–87% | ~2–3 h | ~4–6 GB |

Two takeaways:

- Short fine-tunes don't beat embeddings — if you're going to fine-tune, budget the full schedule.
- Embeddings get you to a working baseline in *minutes*. Always run embeddings first to validate the data and task, even if you intend to fine-tune later.

See [`references/when_to_use.md`](references/when_to_use.md) for the full decision tree, disambiguation prompts for ambiguous cases, and edge cases (label-noise, multi-modal, regression).

## Which model size?

Embeddings come in dimensions matching the model:

| Model | Parameters | Embedding dim | Tradeoff |
|-------|------------|---------------|----------|
| **Nano** | ~1.4M | 128 | Fastest extraction (~2–3 min for 1K windows on T4), good for clustering and binary tasks |
| **Tiny** | ~6.2M | 192 | Default for most embedding-based classifiers |
| **Base** | ~90M | 768 | Strongest representations; use for fine-grained or multi-class downstream tasks |
| **Large** | ~300M | 1024 | Diminishing returns; ~85% on the tutorial benchmark vs Base ~87% |

The tutorial uses **Nano** because it's the fastest path to a working baseline. For real downstream tasks with >5 classes or imbalanced data, prefer **Tiny** or **Base**.

See [`references/model_sizes.md`](references/model_sizes.md) for per-task recommendations.

## Which classifier on top?

Three good defaults, in order of complexity:

| Classifier | When |
|------------|------|
| **kNN (cosine, k=10–20)** | Tiny datasets (<100 labels per class), exploratory, similarity-search adjacent. The tutorial uses k=20. |
| **Linear probe (logistic regression on StandardScaler-normalized embeddings)** | Default for classification. Robust, fast, interpretable coefficients. |
| **Small MLP head (1–2 hidden layers, 256 units)** | When linear probe plateaus and you have >2K samples. Captures non-linear class boundaries without unfreezing the encoder. |

The notebook this skill generates trains both kNN and linear probe by default and reports both accuracies side-by-side. If neither breaks 75 % of the tutorial-benchmark accuracy on the user's task, that's the signal to graduate to fine-tuning.

See [`references/classifier_choice.md`](references/classifier_choice.md) for tuning notes and when to add a per-class threshold for imbalanced data.

## Generating a notebook

`scripts/make_notebook.py` emits a runnable `.ipynb` parameterized for the user's data:

```bash
python scripts/make_notebook.py \
  --dataset-path /path/to/rslearn_dataset \
  --num-classes 9 \
  --class-names "tree_cover,shrubs,cropland,builtup,bare,grassland,water,wetland,other" \
  --model nano \
  --out my_embeddings_workflow.ipynb
```

The generated notebook mirrors the AWF tutorial's flow but with the user's paths and class taxonomy:

1. Install OlmoEarth + rslearn + scikit-learn
2. Detect CUDA / MPS / CPU and load the chosen model
3. Define the `load_window_data` + `extract_embedding` helpers (copied verbatim from the tutorial — they're what the model expects)
4. Extract embeddings for `train` and `val` window groups
5. Train kNN (cosine, k=20) and linear probe (logistic regression on StandardScaler-normalized embeddings)
6. Report accuracy + confusion matrix
7. Optional: t-SNE / UMAP visualization of the embedding space

The notebook is a starting point — the user is expected to read it, not just run it. Add a markdown cell up top calling that out.

## Workflow

When the user asks "should I use embeddings for X" or "generate the notebook":

1. **Confirm the task is classification or similarity** — embeddings don't directly do detection or regression heads (you'd still need a custom head; flag it).
2. **Read the labeled-sample count** — under 100 → embeddings + kNN, no debate. 100–2000 → embeddings + linear probe first, fine-tune second. 2K+ → either works; embeddings give a fast baseline before fine-tuning.
3. **Read the compute budget** — Colab free / T4 / no GPU → embeddings. A100 / 4×GPU / a week → fine-tune.
4. **Recommend** with the rationale table above, citing the specific row that applies.
5. **Generate the notebook** if the user wants to run it, parameterized for their dataset path / class count / model size.

`scripts/recommend.py` packages steps 1–4 into a single command:

```bash
python scripts/recommend.py \
  --task "land cover classification, 9 classes, 200 samples, T4 GPU"

# Output:
# {
#   "decision": "embeddings",
#   "classifier": "linear_probe",
#   "model": "tiny",
#   "rationale": "...",
#   "next_step": "python make_notebook.py --num-classes 9 --model tiny ..."
# }
```

## What this skill does NOT do

- **Generate the labeled data** — that's `olmoearth-data-prep`.
- **Train the fine-tuned model** — if the answer is "fine-tune", point at `olmoearth_run` or the tutorial's Part B and stop. This skill is about the embeddings path.
- **Run the notebook for the user** — it emits a `.ipynb`; the user runs it.
- **Tune the classifier hyperparameters** — kNN k=20 and LR max_iter=1000 are the tutorial defaults; tune if the user reports plateauing.
- **Handle non-S2 modalities natively** — the notebook template is S2-focused (mirrors the tutorial). For S1 or multi-modal, edit `BANDSET_INFO` and `BAND_NAMES` in the generated notebook.

## Reference docs (loaded on demand)

- [`references/when_to_use.md`](references/when_to_use.md) — full embeddings-vs-fine-tune decision tree, edge cases, disambiguation.
- [`references/classifier_choice.md`](references/classifier_choice.md) — kNN vs linear probe vs MLP head, with tuning notes.
- [`references/model_sizes.md`](references/model_sizes.md) — embedding-dim-vs-model-size tradeoffs from the tutorial benchmark.

## Bundled scripts (work standalone, no skill imports)

- [`scripts/recommend.py`](scripts/recommend.py) — task description + (samples, compute) → embeddings-vs-fine-tune decision.
- [`scripts/make_notebook.py`](scripts/make_notebook.py) — emit a parameterized `.ipynb` that mirrors the AWF Kenya tutorial.
