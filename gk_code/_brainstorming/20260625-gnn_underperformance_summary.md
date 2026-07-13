# Why the spatial GNN (GAT/GCN) underperforms on this dataset

## Headline numbers (verified directly, not from memory)

Dataset: `D20260611_E00_C00_F4500KHz_U_lambda_test_manifold_01`, both curve_type variants, all 4
outlier filters (`None`, `lstm_ae_glb_ds1_label_elbow`, `spatial_knn_label_elbow`,
`spatial_grid_label_elbow`):

- **GAT/GCN accuracy**: 55.42% - 80.13%
- **Best non-GNN model on the same dataset/filters**: 99.13% - 100.00%

So the gap is **20-35 percentage points**, and it persists even with outlier-filter
preprocessing applied — it's not just a "noisy baseline" filter problem.

## Two-part explanation

### (a) A real training bug, now fixed — but only partially closed the gap

`03b_gnn_spatial_training.py`'s `_train_step` originally called `optimizer.step()` once
**per graph** instead of once per epoch across all graphs. Since each graph here is one
well, and every well has exactly one class label, stepping the optimizer after every single
graph behaved like alternating single-class SGD across classes rather than real multi-class
training — the model effectively got yanked toward whichever one class it just saw, every
step.

Fixing this (accumulate gradients across *all* graphs in an epoch, then a single optimizer
step) raised real-data accuracy from ~40% to ~62-65% in the first post-fix verification run,
and to 66-80% in this later run with outlier filters applied. That's a real, substantial
improvement — but still well short of the 99-100% every other model gets on the same data.

### (b) A fundamental, structural mismatch — not fixable by further training-loop tuning

This is the dataset-specific reason the gap doesn't close further, no matter how the
training loop is tuned.

Every node in a GNN graph here is one pixel; every graph is one well; and every well has
exactly one label by design — one well = one reaction = one ground truth. (Confirmed
earlier: "each pixel on the same well has the same label... the pixel hardware condition
might be varied... due to noise.") This dataset's metadata has no `well_id` column at all
(`['pixel_row_idx', 'pixel_col_idx', 'temp_group_idx', 'num_active_pixels_in_temp_group',
'well_temp_lin2d_mean', 'well_2d_temp_npr_mean', 'vref_idx']`), so `well_ids` falls back to
the raw `Y_well` — 10 distinct wells, mapped down to 3 effective classes after
`config.LABEL_MAPPINGS` collapses all `NC-*` variants to one `NC` class for this
`nc_subtract`-tree experiment. Every graph is still homogeneous-label.

A GNN's core mechanism — message-passing, where a node's prediction is informed by its
*neighbors' labels/features differing in informative ways* — has nothing to learn from here.
All neighbors trivially share the same label, always. There is no heterogeneous-label
structure for relational reasoning to exploit. The task reduces to "classify a graph from
its nodes' content," which is much closer to a set/sequence classification problem than to
genuine graph-structured learning.

### Tying this to the companion spatial-reconstruction investigation

GNN message-passing (GCN's fixed-weight aggregation, GAT's attention-weighted aggregation)
is mechanistically the same family of operation as the cosine/attention spatial
curve-reconstruction investigated separately (`spatial_recon_lofo_investigation.md`) — both
are forms of learned or fixed local spatial smoothing over neighboring pixels.

That investigation found smoothing specifically **erodes the noise-driven separation
margin** for class pairs that are already borderline in raw curve-shape space, while only
mildly helping pairs that are already well-separated. The same mechanism plausibly compounds
the GNN's more fundamental problem above: it smooths away exactly the per-pixel noise
variation that a sequence classifier (CNN/GRU/Transformer, with no spatial averaging at all)
is free to use directly — and which already gets 99-100% on this same data without ever
looking at pixel position.

### (c) Gradient contamination on the shared encoder — VERIFIED

`EndToEndSpatialGNN` uses **one shared `TemporalEncoder`** across every graph, regardless of
class. In `_train_step`, each of the ~10 per-well graphs calls `loss.backward()`
independently, and PyTorch accumulates gradients additively — so the encoder's `.grad` after
one epoch's worth of graphs is the **sum** of ~10 graphs' gradients, some pushing toward
class A's representation, others toward class B's or C's.

**Verification** (standalone diagnostic, reusing the actual production code —
`build_subset_graphs`/`EndToEndSpatialGNN` from `03b_gnn_spatial_training.py`, not
reimplemented — on `D20260611_E00_C00_F4500KHz_U_lambda_test_manifold_01`, filter=None, a
freshly-initialized model): computed the shared encoder's gradient separately for each of the
10 per-well graphs, then compared pairwise cosine similarity.

- **Same-class graph pairs**: mean cosine similarity = **0.9988** (std 0.0005, 13 pairs) —
  essentially identical gradient direction.
- **Different-class graph pairs**: mean cosine similarity = **-0.4440** (std 0.1602, 32 pairs)
  — gradients from different classes don't just disagree, they actively **oppose** each other
  on the shared encoder.
- Gradient norms across all 10 graphs ranged 5.26-6.19 (1.18x spread) — graphs aren't wildly
  dominating each other by magnitude; the conflict is specifically about *direction*, not one
  graph being disproportionately large.

**Important refinement on what this actually means**: cross-class gradient opposition on
shared parameters isn't unique to the GNN — it's a universal property of any multi-class
cross-entropy classifier (the CNN/GRU models have it too, within their own batches). What's
different is **sample count**: those models' batches mix hundreds of pixels per class in one
512-sample batch, so the per-class direction is a low-variance average over many independent
samples before the conflict ever needs resolving — gradient descent on cross-entropy loss is
*designed* to balance exactly this kind of multi-class pull, but only works smoothly when each
class is well-represented per step. The GNN's effective "batch" is ~10 *graphs* total, ~3-4 per
class — far too few independent samples to average the (normal, expected) cross-class conflict
into a reliable direction each epoch. The measured -0.44 opposition is the same fundamental
phenomenon the other models handle invisibly; the GNN just can't average it away due to its
tiny per-epoch graph count, on top of the (b) spatial-smoothing mechanism above.

### Secondary, lower-priority factors

- `GNN_LAYERS=2` is already at the safe edge for over-smoothing (the well-known GNN failure
  mode beyond ~3-4 layers). Not the cause here — only 2 layers are used — but it does cap how
  much *useful* depth could ever be added as a lever.
- `KNN_K=24` was carried over from an unrelated config constant for consistency, not tuned
  for this model, and risks being capped at `n_well-1` for small wells. Lower-priority, not
  the primary explanation for the gap.

## What GNNs are actually well-suited for

- **Node labels are heterogeneous within a graph**, and a node's neighbors' labels/features
  genuinely predict its own (homophily or heterophily exploitation) — e.g. citation networks
  (a paper's topic correlates with what it cites), social-network fraud detection (an
  account's neighbors' fraud status predicts its own), recommendation systems.
- **Graph topology itself carries information beyond aggregate node features** — e.g.
  molecular property prediction (atom connectivity/bond structure determines chemical
  behavior; specific substructures/rings predict toxicity), traffic forecasting (road network
  topology constrains flow), knowledge graphs.
- **Whole-graph classification with compositional/structural signal** — where the *pattern of
  connections*, not just the bag of node values, determines the label (e.g. "this specific
  subgraph shape predicts X").

## The contrast with this dataset

Here, the label is already fully determined by per-pixel curve kinetics (shape over time),
with no compositional or relational structure needed to recover it. Swapping a GNN for a
model that ignores spatial position entirely (CNN+GRU on individual pixel curves) doesn't
lose any information relevant to the label — it *gains* accuracy (99-100% vs 66-80%), because
the discriminative signal was already fully present in per-pixel curve content. The
graph/spatial framing only re-introduces a smoothing operation that's net-harmful for
borderline cases, without earning anything back via relational structure, because there is no
relational structure here for a GNN to exploit in the first place.
