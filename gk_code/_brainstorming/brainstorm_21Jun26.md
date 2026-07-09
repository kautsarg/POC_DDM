# Brainstorm — 21 Jun 2026

Living document. More to be added over time. Two parts so far: (1) this session's
codebase audit ("what's worth doing next" on the existing pipeline), (2) a forward-looking
brainstorm on alternative architectures/XAI methods worth trying given the nature of the
data — DDM/Lacewing chip amplification curves (sigmoid-shaped kinetic time series, per-pixel
within multi-well chips, classified into assay outcome classes).

---

## Part 1 — Codebase audit: what's worth doing next

Ran 3 parallel Explore agents over `gk_code/main/` (core pipeline 01-04, XAI/interpretability
stack, reporting/infra/config), then personally verified the highest-stakes claims directly
before trusting them. Menu, not a queue — pick what's worth doing.

### Tier 1 — Verified correctness/consistency issues (cheap, worth doing soon)

1. **`hpc_jobs/multi_full_pipeline.pbs` vs `slurm_jobs/multi_full_pipeline.sh` run different
   stages with different flags — VERIFIED by reading both directly.** PBS: `01`/`02` calls
   commented out (assumes already cached), `03` runs with no `--fast_mode`/`--force_rerun`
   (strict, cache-respecting). SLURM: `01`/`02` active (always reprocesses raw data), `03`
   runs with `--fast_mode --force_rerun` (fast, always-fresh). Running the "same" pipeline on
   the two clusters currently produces **different results**. Confirm intentional (e.g. PBS =
   incremental re-run after a prior full SLURM run) or sync them.
2. Dead `legacy_save_path` line — `01_curve_preprocessing_v6.py:423`. Assigned, never read
   again (leftover from the now-removed `--strip_unused_curves` block). One-line delete.
3. `xai_gated_utils.py` has zero importers anywhere in the repo — VERIFIED via grep. Fully
   built and smoke-tested standalone, not wired into `07` yet (expected at this stage — see
   Tier 3).
4. `model_for_xai.py` is dead in the active pipeline — VERIFIED. Only reference outside itself
   is a comment in `config.py`; no `.pbs`/`.sh`/`.py` calls it. Superseded by `07`. Candidate
   for deletion or archival to `gk_code/legacy/`.

### Tier 2 — Performance (measured, not speculative)

5. `compute_kinetic_feature_cache` / `_compute_feature_sensitivity_profiles_all` in `07` is a
   real bottleneck — my own smoke test measured **~165s** for one batch of N=60, T=60 (with
   realistic fast-converging sigmoid curves, not noise). The sensitivity-profile pass alone
   does `n_sub=48 × T` finite-difference perturbations, each running a `scipy.curve_fit` twice.
   At production scale (`batch_size=512`, `T~100+`) this runs per (dataset × curve_type ×
   filter) combination — likely minutes each, accumulating across a sweep. Options: parallelize
   with `joblib.Parallel` (same pattern as `01`'s `sigmoid_fitting_5p`), or check whether `02`'s
   `build_kinetic_features()` already computes equivalent params `07` could reuse instead of
   re-fitting from scratch.

### Tier 3 — Integration gaps (sequential — each blocks the next)

6. Wire the 8 gated fusion models (`model_utils_gated.py`) into `03`/`04`'s `models` lists so
   they actually get trained and saved as `.keras` files.
7. Make `07` recognize them: `load_saved_models()` hardcodes the 9 `model_utils.py` names
   (07:111-115) — gated names silently never load even if the files exist. `extract_xai_artifacts()`'s
   branch-detection looks for a `Concatenate` layer, which gated models don't have (they use
   `fuse_gate`/`fuse_mha1-2`/`fuse_gamma1-2` instead) — needs a fusion-aware branch using the
   explicit layer names already documented in `model_utils_gated.py`'s docstring.
8. Call `xai_gated_utils.get_xai_outputs()` from `07` once 6/7 land — the entry point already
   exists and is tested, just needs a trained model to point it at.

### Tier 4 — New insight opportunities (not bug fixes)

9. Cross-reference `08`'s statistical comparison (which model/filter/curve-type wins) with
   `07`'s latent→feature mapping (which kinetic features that model's latents track) to explain
   *why* a winner wins, not just *that* it wins.
10. Use gate/FiLM XAI (once Tier 3 lands) to explain *when* the CNN vs GRU/Transformer branch
    dominates per class — directly extends the gated-fusion brainstorm from earlier this session.

### Tier 5 — Hygiene (confirm intent before treating as drift)

11. Outlier filter list hardcoded identically in both `03` and `04` instead of one shared
    `config.py` constant — in sync today, will silently drift if only one is edited.
12. No automated test suite anywhere in `gk_code/` — everything verified via manual smoke
    tests this session. Worth a small `pytest` suite for joblib schema contracts between
    `01`→`02`→`03` at minimum.
13. `light_pipeline/` doesn't have the `01`/`02` changes (sigmoid-fit gating, y=0 baseline,
    shared joblib) — per the README it's described as an intentionally separate "lighter-weight
    interpretability variant," likely not meant to mirror `01`-`08`. Confirm intent before
    assuming drift.
14. `hpc_jobs/lab_pipeline_manual.pbs`/`chip_lab_pipeline_manual.pbs` have no SLURM counterpart
    — plausibly intentional (HPC-only datasets). Confirm before treating as a gap.

### Checked, found to be non-issues
- `08_statistical_comparison.py` already correctly imports `config.MODEL_KEY_MAP`/etc. rather
  than stale local copies (verified 08:71-75). `06` too.
- `config.PREPROCESSED_CURVES_PATH` only referenced by `01`'s fallback-merge path and
  `adhoc_strip_unused_curves.py` — legacy constant kept for migration, not a live breakage.

### Already known/decided, not re-litigated
- `04`'s `xai_data_{curve_type}.joblib` per-fold snapshot (~303MB/fold) — known wasteful,
  explicitly deferred (can't be cleanly reformatted retroactively).
- LOFO fold×filter array-job splitting — tried, reverted (HPC queue got worse with 32 small
  jobs than 2 large ones). Decision recorded in `lofo-training-speedup.md`: don't retry.

---

## Part 2 — Architecture & XAI brainstorm: what else is worth trying?

Grounded in what the data actually is: per-pixel amplification curves within multi-well chips,
sigmoid-shaped kinetic time series (rise from baseline to plateau), governed (approximately) by
a 5-parameter sigmoid model already used for feature extraction (`Fm, Fb, Sc, Cs, As` → `Ct`,
`Cy0`, slope, amplitude, AUC, asymmetry, etc.). Classification target is assay outcome
(e.g. Target vs NC-Target, concentration level). Known pain points: cross-dataset/cross-chip
generalization (LOFO), outlier pixels, per-pixel vs per-well unit of analysis, seed variance.

### Alternative architectures worth trying

1. **State-space models (S4 / Mamba)** — these curves are smooth, low-frequency-dominated
   signals, not noisy/high-frequency like audio. Selective SSMs are built for exactly this kind
   of long-range smooth dynamics and are typically cheaper than attention for the sequence
   lengths here. Plausible drop-in replacement for the GRU/Transformer branch.

2. **Physics-informed parametric bottleneck** — since the data is *known* to follow a 5-param
   sigmoid, have the curve branch predict `(Fm, Fb, Sc, Cs, As)` directly via a small head, then
   classify on the predicted parameter vector plus a residual/anomaly branch capturing deviation
   from the ideal sigmoid fit. Gets interpretability "for free" since the bottleneck *is* the
   kinetic parameters — and the residual branch doubles as an outlier signal, potentially
   simplifying the separate outlier-detection stage.

3. **Neural ODE / Neural CDE** — the underlying process is a kinetic reaction (ODE-governed).
   A Latent ODE would give a continuous-time latent trajectory naturally interpretable as
   "reaction state at time t," and handles irregular/missing timepoints gracefully — relevant
   since `04` already needs `CurveResampler` to reconcile different timestamp grids across
   chips; an ODE-based model might not need resampling onto a common grid at all.

4. **Multi-task auxiliary regression head** — add an auxiliary loss where the curve branch must
   *also* regress the kinetic features (Ct, Sc, amplitude, ...) alongside the classification
   head. Regularizes the latent space to be kinetically meaningful, which would directly improve
   the latent→feature-mapping XAI (Tier 4 #9 above) since the model is now explicitly trained to
   encode those features, not just incidentally correlated with them.

5. **Contrastive pretraining for cross-chip invariance** — given multiple experiment
   folders/chips with batch effects (temperature, chip-to-chip variation — `nc_subtract` and
   temperature normalization already exist as partial fixes), a SimCLR-style contrastive
   pretraining objective (augmentations: time-warp, amplitude jitter, baseline shift) could learn
   batch-invariant representations before classification fine-tuning. Directly targets the LOFO
   generalization problem that `04` exists to measure.

6. **Set-based / attention-pooled well-level aggregation** — each well has many pixels, each
   currently treated as an independent sample. A Deep-Sets or attention-pooling layer over all
   pixels in a well would produce one well-level prediction (the actual experimental unit) rather
   than implicit per-pixel majority voting, likely more robust to per-pixel noise — and the
   attention-pooling weights would surface *which pixels were most diagnostic* as a new,
   well-level XAI signal for free.

7. **GP/Bayesian curve modeling for uncertainty** — a Gaussian Process curve model would give a
   principled per-sample uncertainty estimate, useful for flagging ambiguous wells/pixels rather
   than forcing a hard classification. Complements rather than replaces the NN approach; could sit
   alongside the existing outlier-detection stage.

8. **Distillation into a lightweight single-branch model** — if dual/gated fusion proves best but
   is too expensive, distilling into a smaller CNN/GRU preserves accuracy with lower latency —
   worth keeping in mind if this is headed toward point-of-care deployment (the "Lacewing chip"
   framing suggests a real diagnostic device, not just a research exercise).

### Alternative/additional XAI methods worth trying

1. **Integrated Gradients / SmoothGrad** — the current saliency is raw `dZ/dX`, known to be
   noisy and prone to saturation. IG (path integral from a baseline curve) or SmoothGrad
   (averaging over noised copies) gives smoother, more reliable per-timepoint attribution with
   the exact same output shape as the existing saliency heatmap — a drop-in quality upgrade.

2. **SHAP on the manual/late-fusion kinetic-feature branch** — proper additive feature
   attributions with theoretical guarantees, vs. the current correlation-based latent→feature
   mapping which is associative, not causal/additive. Answers "how much did *this* feature
   contribute to *this* prediction," which the current method doesn't.

3. **Counterfactual explanations on the sigmoid parameters** — "if Ct had been 2 units earlier,
   the prediction would flip from Target to NC-Target." Directly actionable for a wet-lab
   scientist in a way gradient saliency isn't, and natural here because the parameters are
   already interpretable (unlike pixel-space counterfactuals in vision).

4. **Concept Activation Vectors (TCAV)** — define concepts directly from
   `config.XAI_KINETIC_FEATURE_GROUP`'s 18 main features ("high amplitude," "early Ct," "steep
   slope") as concept vectors in latent space, then statistically test causal importance per
   class. A more rigorous upgrade path from the current correlation-based latent mapping, and
   dovetails directly with Tier 4 #9 (connecting `07` and `08`).

5. **Attention rollout for the co-attention gated models** — once Tier 3 lands, visualizing the
   actual cross-attention weights (`xai_gated_utils.build_coattn_weight_extractor`/
   `extract_attention_weights`, already built) gives a *direct mechanistic* explanation of which
   timepoints each branch attends to — not a post-hoc gradient proxy like current saliency.

6. **Layer-wise Relevance Propagation (LRP)** — for the CNN branch specifically, LRP often gives
   cleaner, more spatially-localized explanations than raw gradients for convolutional layers;
   worth a direct comparison against the current saliency approach on the same curves.

7. **Anchors / rule-based local explanations** — extract simple human-readable rules (e.g. "IF
   Ct < 15 AND amplitude > 0.8 THEN Target, 95% precision on similar curves") as a complement to
   the visual plots — useful for non-ML audiences (lab scientists, regulatory documentation if
   this is headed toward a diagnostic product).

8. **Prototype-based explanations (ProtoPNet-style)** — learn a small set of representative
   "prototype curves" per class, explain new predictions by similarity to prototypes ("87% like
   prototype #3 for Target"). Highly interpretable specifically *because* the data is curve-shaped
   — naturally visualizable by overlaying prototype vs. query curve.

---

*(Space for future brainstorming sessions below)*
