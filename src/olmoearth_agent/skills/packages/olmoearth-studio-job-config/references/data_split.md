# Data split — Spatial (recommended) vs a metadata field

Studio's "How should your data be split?" step assigns every labeled sample to train, validation, or test. Two options:

| Option | What Studio does | Pick when |
|--------|------------------|-----------|
| **Spatial (recommended)** | Partitions samples by *location* so validation/test areas are spatially separate from training areas | Default for almost every model |
| **Use a metadata field** | Reads a property on each labeled feature whose values assign `train` / `val` / `test` | You bring a curated split |

## Why Spatial is the recommendation

Earth-observation samples are spatially autocorrelated: two labels 50 m apart sit on nearly identical imagery. A random (non-spatial) split scatters neighbors across train and validation, so the model is effectively evaluated on data it has already seen — validation accuracy looks great and transfers nowhere. The spatial split holds out *areas*, not *rows*, which is the honest estimate of how the model performs on new ground.

This effect is strongest exactly where models are usually built: dense label clusters (a surveyed watershed, a digitized county). The denser the labels, the more a random split lies.

## When a metadata field is right

- **You engineered the split yourself** — e.g. spatial cross-validation folds or a nearest-neighbour-distance-matched (NNDM-style) split produced during data prep. The sibling skill `olmoearth-data-prep` can emit such an assignment as a property on each feature.
- **The split must be identical across tools or runs** — comparing Studio against an external baseline on the *same* held-out samples, or re-training without reshuffling.
- **Domain-structured holdouts** — e.g. hold out whole years, whole administrative regions, or whole label campaigns to test a specific kind of generalization the default spatial split doesn't target.

The field must exist on every labeled feature and contain only the split values (e.g. `train` / `val` / `test`). Samples with a missing or unrecognized value are a config error, not a soft default.

## Anti-patterns

- **A random or row-order split for spatial data** — inflates validation accuracy via neighbor leakage (see above). If the user proposes "just split 80/20 randomly", recommend Spatial and explain the leakage.
- **A metadata-field split that was itself produced randomly** — same leakage, now laundered through a field. The field option is only as good as the procedure that filled it.
- **Tiny validation sets** — holding out one small corner of the AOI gives a noisy estimate; if the user controls the field, keep validation a meaningful share (~10–20%) and spatially representative.
- **Splitting after augmentation/duplication** — duplicates of one source label must never straddle the split boundary.

## Pairing with `olmoearth-data-prep`

Data prep decides *what* the labels are; this step decides *how they're partitioned*. If the user already ran data prep's audit and split tooling, prefer **Use a metadata field** pointing at that assignment. Otherwise **Spatial** — never re-implement a split by hand when the wizard's default does it correctly.
