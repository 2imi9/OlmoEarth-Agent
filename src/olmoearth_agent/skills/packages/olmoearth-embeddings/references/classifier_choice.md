# What classifier to train on top of OlmoEarth embeddings

OlmoEarth's encoder emits a dense feature vector per window (128 dim for Nano, 192 for Tiny, 768 for Base, 1024 for Large). Once you have those features, the choice is what to put on top. Three good defaults:

## kNN with cosine similarity

**Use when**: <100 labels per class, exploratory, or as a sanity check against the linear probe.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

train_emb_norm = normalize(train_embeddings, norm='l2')
val_emb_norm = normalize(val_embeddings, norm='l2')

k = min(20, len(train_embeddings))
knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
knn.fit(train_emb_norm, train_labels)
preds = knn.predict(val_emb_norm)
```

Hyperparameters:

- `k=20` is the tutorial default. For tiny datasets, `k=5` or `k=10` may work better — but never set `k > len(train) // 2`.
- `metric='cosine'` is non-negotiable for OE embeddings. Euclidean kNN on raw embeddings gives noticeably worse results.
- L2-normalize *both* train and val before fitting — cosine on L2-normalized vectors equals dot product, which is what kNN's cosine metric expects under the hood.

Pros:
- No training step. Predictions are O(N) at inference but you can use FAISS for scale.
- Works with as few as 1 sample per class.
- Naturally captures multi-modal class distributions.

Cons:
- No calibrated probabilities.
- Bad on heavy class imbalance — minority classes get drowned out by majority neighbors.

## Linear probe (logistic regression on StandardScaler-normalized embeddings)

**Use when**: this is the default. 100+ labels, decent class balance, you want a quick, calibrated, interpretable baseline.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
train_emb_scaled = scaler.fit_transform(train_embeddings)
val_emb_scaled = scaler.transform(val_embeddings)

# Handle zero-variance features after scaling
train_emb_scaled = np.nan_to_num(train_emb_scaled, nan=0.0, posinf=0.0, neginf=0.0)
val_emb_scaled = np.nan_to_num(val_emb_scaled, nan=0.0, posinf=0.0, neginf=0.0)

lp = LogisticRegression(max_iter=1000, solver='lbfgs')
lp.fit(train_emb_scaled, train_labels)
preds = lp.predict(val_emb_scaled)
```

Hyperparameters:

- `max_iter=1000` is the tutorial default. Bump to 2000 if you see `ConvergenceWarning`.
- `solver='lbfgs'` works well for small-to-medium datasets. For >10K samples, `saga` is faster.
- `C=1.0` (default) is fine; lower (`C=0.1`) for tiny datasets where regularization helps, higher (`C=10`) for huge datasets where the prior shouldn't dominate.
- `class_weight='balanced'` when minority classes are <10 % of the data.

Pros:
- Calibrated probabilities (`predict_proba`).
- Coefficients are interpretable — `lp.coef_` shows which embedding dims drive each class.
- Trains in seconds; predicts in microseconds.

Cons:
- Linear in embedding space. If the classes need non-linear separation (rare for ViT features), MLP head helps.

## Small MLP head

**Use when**: linear probe plateaus below the accuracy you need AND you have >2K labeled samples.

```python
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, emb_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)
```

Train with Adam, lr=1e-3, 50–100 epochs, batch size 64. Watch val loss for overfitting — embeddings + a small head overfits faster than embeddings + LP.

Pros:
- Captures non-linear class boundaries that LP misses.
- Still way faster than full fine-tuning.

Cons:
- More hyperparameters to tune (hidden dim, dropout, lr, epochs).
- Overfits easily on small datasets. If you have <2K samples, stick with LP.

## When neither works: fine-tune

If LP plateaus and you don't have enough data for an MLP head — or you've tried both and neither hits your accuracy bar — that's the signal to fine-tune the encoder. See the tutorial's Part B for the rslearn fine-tuning workflow. Budget 2–3 hours and >4 GB VRAM.

## Per-task quick lookup

| Task type | First-try classifier |
|-----------|----------------------|
| Multi-class classification, balanced | Linear probe |
| Multi-class classification, imbalanced | Linear probe with `class_weight='balanced'` |
| Binary classification | Linear probe (often LP > kNN here) |
| Few-shot (<10 samples / class) | kNN with `k=5` |
| Similarity search | kNN with `k=1`, or just cosine ranking, no classifier |
| Clustering | k-means or HDBSCAN on raw embeddings |
| Continuous regression | Ridge / Lasso instead of LogisticRegression |
| Multi-label (each sample has ≥1 label) | OneVsRestClassifier(LogisticRegression) |
| Anomaly detection | Isolation Forest or OC-SVM on embeddings |

## Anti-patterns

- **Train LP without StandardScaler** — coefficients become uninterpretable and convergence slows; OE embeddings have wildly varying per-dim scales.
- **Use kNN with k=1 by default** — overfits to noise; k=5–20 is the working range. The tutorial uses k=20.
- **Compare kNN and LP on different normalizations** — kNN expects L2 normalize, LP expects StandardScaler. They aren't comparable on the same normalized matrix.
- **Use Euclidean kNN on raw OE embeddings** — works much worse than cosine; the tutorial uses cosine.
