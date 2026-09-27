# When to use OlmoEarth embeddings (vs fine-tuning)

This is the decision framework. All numbers come from the AllenAI OlmoEarth tutorial's measured benchmarks on the AWF Kenya LULC task (9 classes, ~1000 samples, T4 GPU). Your task will differ — use these as priors, not absolute thresholds.

## The headline decision table

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| <100 labeled samples | **Embeddings + kNN** | Fine-tuning a 1.4M+ param ViT on <100 samples overfits hard; kNN on frozen features is the textbook small-N path. |
| 100–2,000 labels | **Embeddings + linear probe** first; fine-tune if LP plateaus below requirement | LP gets you to a calibrated baseline in minutes; fine-tune only if the baseline isn't good enough. |
| >2,000 labels, batched workflow | **Embeddings first, fine-tune for production** | Use embeddings to validate the data pipeline + label quality, then commit to the 2–3 h fine-tune run. |
| Rapid prototyping / exploring data | **Embeddings + kNN/LP** | Iteration speed matters more than peak accuracy. |
| Similarity search ("find areas like this one") | **Embeddings, no classifier** | Cosine similarity on L2-normalized embeddings is the entire workflow. |
| Clustering / unsupervised segmentation | **Embeddings + k-means / HDBSCAN** | No labels needed; embeddings carry the semantic content. |
| Weak labels (noisy, partial, distant supervision) | **Embeddings + label-aware classifier** | Don't propagate noisy labels through fine-tuning; learn a robust head instead. |
| Production deployment, accuracy SLA | **Fine-tuning (full schedule, 30+ epochs)** | Tutorial: fine-tune at 30 epochs hits ~82–87 % vs embeddings at ~70–75 %. |
| Maximum accuracy needed | **Fine-tuning** | Same as above. |
| Task very different from pre-training (SAR-only, thermal, novel sensor) | **Fine-tuning** | Frozen features were trained on multi-modal composites; novel input distributions need the encoder to adapt. |
| Compute budget: T4 / Colab free / no GPU | **Embeddings** | Tutorial: embedding extraction uses ~2–3 GB VRAM vs 4–6 GB for fine-tuning, runs in minutes vs hours. |

## Tutorial-anchored cost / accuracy reference

These numbers are from the tutorial's own "Why Two Approaches?" cell — the AWF Kenya LULC task with the Nano model on a T4 GPU:

```
Approach                       Accuracy    Time          GPU memory
Embeddings + kNN/LP            ~70–75%     minutes       ~2–3 GB
Fine-tuning (4 epochs)         ~70–75%     ~15–20 min    ~4–6 GB
Fine-tuning (30 epochs)        ~82–87%     ~2–3 hours    ~4–6 GB
```

Three reads from this:

1. **A short fine-tune buys nothing over embeddings.** If you only have time for 4 epochs, stay on embeddings.
2. **The 30-epoch gap is real but bounded.** ~10–15 percentage points on a 9-class problem with ~1000 samples. Whether that's worth 2–3 hours of GPU time is a product question.
3. **Embeddings are the safe baseline.** Run them first, always. They take minutes and tell you whether the data pipeline is broken, the labels are usable, and the task is learnable. If kNN gets 5 % on a 9-class task, fine-tuning won't save you — your labels are bad.

## Disambiguation prompts

When the user is vague, ask:

1. "How many labeled samples do you have?" — under 100 → embeddings, no debate. Over 2K → either; ask about compute.
2. "What GPU do you have, and how long do you have?" — T4 / few hours → embeddings. A100 / overnight or longer → fine-tune is on the table.
3. "Are you prototyping or shipping?" — prototyping → embeddings. Shipping with an accuracy SLA → fine-tune.
4. "Do you need to localize objects or just classify?" — detection is a different beast; embeddings alone don't give bounding boxes. Stay on fine-tune for detection.
5. "Is the task in OlmoEarth's pre-training distribution?" — S2 + S1 globally is in-distribution. Thermal-only, hyperspectral, drone imagery is out-of-distribution; fine-tuning is necessary to adapt.

## Edge cases

### Regression targets
Embeddings + linear regression (Ridge / Lasso) works for continuous targets — same as classification but `LinearRegression` instead of `LogisticRegression`. Tutorial doesn't show this, but it's mechanically the same. Threshold for "use embeddings": <2K samples or high label noise.

### Multi-modal (S2 + S1)
The tutorial extracts S2-only embeddings. For S1 + S2 jointly, build a `MaskedOlmoEarthSample` with both `sentinel2_l2a` and `sentinel1` tensors and concatenate the pooled embeddings (or use the joint encoder output if your OE version supports it). Mention this in the notebook if the user asks for multi-modal.

### Very imbalanced classes
Embeddings + linear probe handles imbalance OK if you set `class_weight='balanced'` in `LogisticRegression`. If one class is <2 % of the data, neither LP nor fine-tune will save you without resampling or focal loss — that's a labeling problem, not a model-choice problem.

### Distribution shift between train and inference
Embeddings are *more* robust to distribution shift than fine-tuned heads — the encoder hasn't been pushed to memorize the training distribution. If you expect to deploy in a region or season different from training, embeddings are often the more honest baseline.

### "I have no labels yet"
Embeddings + k-means / HDBSCAN to *discover* structure, then hand-label clusters. This is a common workflow for novel AOIs. Fine-tuning is not on the table without labels.

## Anti-patterns

- **Fine-tune with <50 labels** — guaranteed overfit. Embeddings + kNN with k = min(20, len(train)).
- **Embeddings + softmax linear probe on a regression target** — wrong loss; use `LinearRegression` or `Ridge`.
- **Compare embeddings vs fine-tune at 4 epochs** — the tutorial explicitly warns this is a fake comparison. Fine-tune the full schedule or don't.
- **Run kNN on un-normalized embeddings with Euclidean distance** — OE embeddings live in a high-dim space where cosine similarity is the meaningful metric. Always L2-normalize for kNN.
- **Skip embedding extraction and go straight to fine-tune** — even if you intend to fine-tune, running embeddings first costs you minutes and tells you whether the pipeline works. The tutorial calls this out explicitly: "Start with embeddings to quickly validate your approach, then fine-tune for production."
