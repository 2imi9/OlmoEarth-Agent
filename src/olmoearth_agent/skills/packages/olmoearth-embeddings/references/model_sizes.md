# Model size for OlmoEarth embeddings

OlmoEarth ships in four sizes. Embedding dimension scales with model size. From the tutorial's "Scaling Up" cell:

| Model | Parameters | Embedding dim | Tutorial benchmark accuracy* |
|-------|------------|---------------|------------------------------|
| **Nano** | 1.4M | 128 | ~82% |
| **Tiny** | 6.2M | 192 | ~85% |
| **Base** | 90M | 768 | ~87% |
| **Large** | 300M | 1024 | ~85%† |

\* AWF Kenya LULC fine-tuning benchmark; embedding-only is ~5–10 percentage points lower per model size.
† Large shows diminishing or slightly negative returns vs Base on this specific task — confirm on your own data before paying the extra cost.

The model is loaded via `ModelID`:

```python
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)   # or _TINY / _BASE / _LARGE
encoder = model.encoder
```

## Picking a size for embeddings

For embeddings specifically — i.e., not fine-tuning — the considerations differ slightly from the Studio job-creation choice:

| Use case | Recommended |
|----------|-------------|
| Sanity-check the pipeline, prototype in <5 min | Nano |
| Similarity search, clustering | Tiny or Base — higher-dim embeddings carry more semantic discrimination |
| Linear probe / kNN for downstream classification | **Tiny** is the working default; Base if classes are fine-grained (>5) |
| Few-shot (<10 samples / class) | Base — more capacity helps when you can't tune the classifier much |
| Massive scale embedding extraction (millions of windows) | Nano — 30× cheaper than Base per window |

## Embedding-dim tradeoffs

- **128 dim (Nano)**: fast cosine kNN, easy to store (4 KB per window in float32, 1 KB in int8 quantized), but loses fine-grained distinctions in fine-grained tasks.
- **192 dim (Tiny)**: best default for downstream classification. Big enough for discrimination, small enough for fast retrieval.
- **768 dim (Base)**: noticeable accuracy gain over Tiny for >5 classes or imbalanced data. Storage cost is 6× Nano.
- **1024 dim (Large)**: rarely worth it for embeddings alone — the tutorial benchmark shows Large underperforms Base. Try only after Base has plateaued.

## Cost / throughput rough orders

These are extrapolated from the tutorial's measured numbers on a T4 GPU (1 sample per second-ish for embedding extraction with Nano):

| Model | Time to extract embeddings for 1K windows (T4) | Storage per 1M windows (float32) |
|-------|-----------------------------------------------|----------------------------------|
| Nano | ~2–3 min | ~0.5 GB |
| Tiny | ~5–8 min | ~0.8 GB |
| Base | ~30–60 min | ~3 GB |
| Large | ~2–3 h | ~4 GB |

Multiply by your time-frame depth (e.g., 12 monthly timesteps = ~12× per-window cost vs 1 timestep) and by AOI size for total cost.

## How to switch model size in the generated notebook

The `make_notebook.py` script bakes in the model choice via `--model nano | tiny | base | large`. Switching after the fact:

```python
# In the notebook, change this:
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
# To:
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
```

And update the embedding-dim assertion (if you added one) and the storage allocation.

## Anti-patterns

- **Nano embeddings for a 20-class fine-grained task** — embedding space too compressed; classes collapse on each other.
- **Large embeddings for clustering 10K windows** — 4 GB of float32 features in memory; switch to Tiny or int8-quantize.
- **Mixing model sizes across train and val** — extractor must be the same for both; otherwise the embedding spaces aren't comparable.
