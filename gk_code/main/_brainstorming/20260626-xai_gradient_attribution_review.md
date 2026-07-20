# XAI Gradient Attribution: Correctness Review & Method Comparison Notes

## 1. Does `07_attribution_vis_all.py` correctly implement gradient-based attribution?

**Yes.** It's a two-stage vanilla-gradient/saliency-map implementation (the Simonyan et al.
2013 family: `|∂target/∂input|`) — not Integrated Gradients, SmoothGrad, or Grad-CAM.

- **Stage 1 — latent importance ranking (`dY/dZ`)**: `tape.gradient(target_master, z_curve)`
  in `extract_xai_artifacts`, then `rank_latents` averages `|gradient|` across the batch to
  rank latent dimensions by importance.
- **Stage 2 — per-latent trace-back to input (`dZ/dX`)**: `compute_latent_saliency_batch`
  computes, for each top-ranked latent dimension individually, the gradient of *that single
  unit's activation* with respect to the raw input — a per-unit/neuron attribution,
  methodologically sound and distinct from (but related to) feature-visualization/unit-
  attribution techniques in the literature.
- **Master saliency (`dY/dX`)**: `master_sal = mean(|tape.gradient(target_master, x_tf_curve)|,
  axis=(0,2))` — the direct, standard vanilla-gradient saliency map over time.

### Specific correctness checks (confirmed by direct code reading)

- `target_master = reduce_max(head_model(...), axis=1)` — a standard simplification.
  Differentiating the max class score is equivalent to differentiating the predicted class's
  score when the model agrees with its own argmax.
- `tape.watch(z_curve)` is **necessary, not redundant** — `GradientTape.gradient(target,
  sources)` requires non-Variable tensors to be explicitly watched before they can be used as
  a gradient target.
- `persistent=True` is correctly used for multiple `.gradient()` calls from one tape, with an
  explicit `del tape` afterward to free it.
- The `extractor`/`head_model` Keras-functional-API split shares the original model's actual
  layer objects/weights, so splitting the forward pass for intermediate-gradient access doesn't
  change what's being computed versus the original end-to-end model.
- `|gradient|` is taken **before** averaging over the batch (not after) — avoids sign
  cancellation across samples while preserving each sample's saliency magnitude.

### Dead code found (not a correctness issue with what's running)

`targeted_latent_attribution` (line 196) and `get_top_dims_from_weights` (line 189) in
`07_attribution_vis_all.py` are defined but never called anywhere in that file or
`07b_cross_dataset_attribution_vis.py` — likely superseded by `extract_xai_artifacts`'s current
approach.

## 2. The underlying design intent (confirmed sound)

Treat each latent dimension as a derived "feature": rank those features by importance
(`dY/dZ`), then trace each important feature back to the input (`dZ/dX`) to see which part of
the input built it. This is a legitimate, recognized pattern — sometimes called neuron/unit
attribution — distinct from Grad-CAM's combined-localization goal (see below).

## 3. How Integrated Gradients / SmoothGrad / Grad-CAM relate

Refined from "all three address a different goal" to a more precise split:

- **Integrated Gradients**: same goal as the current implementation (input attribution via
  gradients) — a more robust *computation*, not a different analysis. It integrates the
  gradient along a path from a baseline to the actual input, fixing vanilla gradients' main
  weakness: saturation (a feature can be highly important but show near-zero gradient if the
  network is in a saturated region for that input). Could be swapped in for the existing
  `dZ/dX` step (or even the `dY/dZ` step) as a drop-in, more saturation-robust replacement.
- **SmoothGrad**: also the same goal — averages vanilla gradients over several noisy copies of
  the input to denoise a visually spiky saliency map. Could be layered onto the existing
  `dZ/dX` computation as a noise-reduction step, not a different analysis.
- **Grad-CAM**: a genuinely *different* goal. Grad-CAM produces one combined,
  class-discriminative localization map per class by pooling the gradient per-channel into a
  weight and summing weighted channels together — deliberately collapsing many channels into
  one heatmap. This is the opposite of the current per-unit decomposition (ranking individual
  latent units separately, tracing each one back to input independently). Grad-CAM is also
  architecturally tied to spatial conv feature maps in a way the recurrent/transformer/dense
  latents here aren't.

## 4. Planned follow-up work (not yet started)

Compare the existing vanilla-gradient method against Integrated Gradients and SmoothGrad as
more robust replacements for the same `dY/dZ`/`dZ/dX` computations, and separately try Grad-CAM
to get a "bird's eye view" (combined, coarser localization) version — a different and
complementary kind of visualization rather than a replacement for the per-unit trace-back
already in place.
