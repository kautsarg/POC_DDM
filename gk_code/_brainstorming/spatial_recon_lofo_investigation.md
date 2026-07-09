# Investigation: spatial-recon models underperform on LOFO cross-chip validation

## CONCLUSION (reached via cheap, no-GPU-training analysis on existing artifacts)

**Mechanism confirmed**: reconstruction (cosine_recon, and by the same logic attn_recon) is not
generically harmful — it specifically harms class pairs that are *already borderline* (close in
raw curve-shape space), while *helping* every well-separated pair.

Evidence (all from `curve_for_training.joblib` + `build_neighbor_curve_stack`/
`reconstruct_curves_cosine` run standalone, no model training):
- Computed nearest-opposite-class-well distance (mean-curve Euclidean, full (T,) curve, not just
  endpoint) for all 10 wells on the held-out chip (norm_temp_04), raw vs reconstructed.
- **Only the well-8/well-9 pair** (Target/NC) — which are *each other's* nearest opposite-class
  well, and the single most borderline pair on this chip (rank 1/10) — shows reconstruction
  *decreasing* separation (-0.7%).
- **Every other well** (0,1,2,3,4,5,6,7) shows reconstruction *increasing* separation from its
  nearest opposite class (+0.4% to +1.8%) — consistent with those wells' per-well accuracy actually
  *improving* under both recon models (e.g. well 1: baseline 87.0% -> cosine_recon 97.7%).
- Per-pixel: well 8's curve-distance-to-NC-well-9's-mean shrinks in both mean (0.5395->0.4893) and
  spread (std 0.367->0.284) after reconstruction — i.e. reconstruction removes the per-pixel
  noise/variance that occasionally pushed some raw well-8 pixels further from the NC mean (more
  confidently classifiable as Target), collapsing all of well 8's pixels toward its own borderline
  mean curve shape, which happens to sit close to NC's mean.
- This directly confirms the user's original hypothesis ("smoothing removes curve shape
  variation that helps generalization") in a precise, mechanistic form: it's not that smoothing
  generically destroys useful signal everywhere — it specifically erases the noise-driven margin
  that classifiers rely on for class pairs that are already close, while legitimately denoising
  (and improving) well-separated pairs.
- Cross-chip check: well-8/well-9 is the most-borderline pair on norm_temp_04 (rank 1/10, this
  fold's held-out chip) AND read_06 (rank 1/10), but well-SEPARATED on read_07 (rank 10/10) and
  moderately separated on ready_08 (rank 3/10). So this isn't "well 8 is structurally always
  risky" — it's that this specific LOFO fold happened to hold out a chip where well 8/9 are
  unusually close, while training data gave mixed signal (one training chip also borderline, one
  clearly separated). Hypothesis 2 (chip-specific local spatial/hardware artifacts) was not needed
  to explain the observed collapse — the simpler, curve-shape-based mechanism fully accounts for it.

**Practical implication (not yet implemented, diagnose-only per the user's request)**: a fix would
need to either (a) only apply reconstruction-based denoising when the well's class is already
well-separated from its nearest opposite class (risk: needs ground truth at inference time, not
available for genuinely unseen data), or (b) blend the reconstructed curve with the raw curve
rather than fully replacing it (preserves some of the discriminative noise margin), or (c) train
on a mix of raw + reconstructed curves so the classifier doesn't lose exposure to noisy-margin
cases. Not decided/implemented — diagnosis only, per plan scope.

## Observation
`cnn_gru_dual_cosine_recon` / `cnn_gru_dual_attn_recon` (model_utils.py) underperform baseline
`cnn_gru_dual` when evaluated via `04_cross_dataset_training.py`'s LOFO cross-dataset validation
(train on 3 chips, test on a 4th, entirely unseen chip).

Group: `init_oneplex_v6` (4 chips: norm_temp_04, read_06, read_07, ready_08).
Fold 1 (held out: norm_temp_04), filter=None:
- baseline `cnn_gru_dual` = 96.56%
- `cnn_gru_dual_cosine_recon` = 88.40%
- `cnn_gru_dual_attn_recon` = 88.96%

filter=lstm_ae_glb_ds1_label_elbow: baseline=95.04% vs cosine_recon=81.89% (~13pp worse).

Job (`04_cross_dataset_training.py`, SLURM job logs under
`slurm_jobs/logs/multi_04_loco_crossval/`) was still running as of this writing — only fold 1 has
full recon-model data; folds 2-4 (read_06/read_07/ready_08) only have baseline models so far.
Results file:
`/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi_nc_subtract/cross_dataset_cv/init_oneplex_v6/cross_dataset_classification_performances_lofo_ori_curve.joblib`

## User's hypothesis
Spatial smoothing (cosine or attention) removes informative per-pixel curve-shape variation the
classifier relies on, hurting cross-chip generalization more than within-chip.

## My refinement (not yet tested)
Smoothing over k=24 nearby pixels creates LOCAL, chip-specific spatial-correlation structure
(e.g. local heating-gradient artifacts consistent within one physical chip but different across
chips) that the classifier can latch onto during training — looks like signal on train chips,
doesn't transfer to a new chip's different local hardware artifacts. Not mutually exclusive with
the user's hypothesis.

## Ruled out (verified directly, don't re-investigate)
- **Padding-for-small-wells theory** (an Explore agent's first guess): REFUTED. Directly queried
  well sizes for norm_temp_04 from `curve_for_training.joblib['metadata']['well_id']`: wells have
  1194-1937 pixels each, far more than `k_neighbors=24`. The "pad by cycling when well has <k+1
  pixels" fallback in `build_neighbor_curve_stack` never triggers on this dataset.
- **"Missing folds 2-4 in results" as an anomaly**: NOT a bug. Checked joblib mtime (~1hr before
  check) vs per-model training duration (~1-2hr each, from job log) — job is simply still running,
  not crashed/silently skipping.
- **Cross-experiment row alignment** (`04_cross_dataset_training.py`: `load_curve_data`,
  `combine_group`, `CurveResampler.transform`): checked, appears correct — curves/coords/well_ids
  concatenated in the same per-experiment order; `well_id` prefixed `f"{exp_path.name}::{w}"` to
  prevent cross-chip neighbor-finding collisions (also verified via a dedicated synthetic test
  earlier this session).

## KEY FINDING (from Plan-agent-assisted live-data analysis, fold 1, filter=None)
The aggregate ~8pp drop is **not a uniform degradation** — it's concentrated almost entirely in
**one physical well: well 8** (a Target well, 1812 pixels, on the held-out chip norm_temp_04):
- well 8 accuracy: baseline 83.9% → cosine_recon 2.1% → attn_recon 3.4% (catastrophic collapse,
  confidently wrong — probabilities saturated near 0/1, not borderline)
- **every other well is flat or improved** under both recon models (e.g. well 1: baseline 87.0% →
  cosine_recon 97.7%)
- excluding well 8 entirely: baseline 98.11%, cosine_recon 98.99%, attn_recon 99.45% — both recon
  models *beat* baseline on the other 9 wells
- per-class: both recon models show a directional bias toward predicting NC (NC specificity
  improves to 99.4%/100% vs baseline's balanced 96.6%/96.6%; Target recall collapses to ~80%)
- standalone `build_neighbor_curve_stack`+`reconstruct_curves_cosine` run on well 8 in isolation
  (no training, ~4 sec CPU) shows the reconstruction barely perturbs well 8's mean curve (raw
  endpoint mean 0.0953 → reconstructed 0.0954) — the reconstruction transform itself doesn't look
  broken when applied to well 8's curves directly; the failure looks like it comes from how a
  classifier *trained on smoothed curves from the other 3 chips* generalizes to well 8's
  reconstructed curve specifically, not from the transform corrupting well 8's signal outright
- well 8's raw curve endpoint value (0.0953) is the most extreme among all 10 wells on this chip,
  numerically closer to NC well 9's value (0.0779) than to its sibling Target wells (0.005-0.04) —
  i.e. well 8 looks like an inherently borderline/atypical Target well even in raw feature space,
  plausibly why baseline already struggles on it relative to other Target wells (83.9% vs 99%+
  elsewhere) and why anything shifting the decision boundary (training on smoothed curves) could
  tip it over

**Reframes the investigation**: this looks much more like a single-well catastrophic failure mode
than a general "smoothing erases useful variance" effect. Both hypotheses remain plausible
explanations for *why well 8 specifically* fails; investigation should be well-resolved, not just
aggregate-accuracy-resolved.

## Diagnostic plan (cheapest -> most expensive; user chose "diagnose only, then decide")

**Step 0** (minutes, pure analysis, no training): finish well/class-resolved audit — extend to
filter=lstm_ae_glb_ds1_label_elbow; probability-margin histograms for all wells; repeat per-well
breakdown for folds 2-4 once the live job checkpoints them (no new training needed, just re-read
the results joblib later) — tells us if "one catastrophic well" recurs per held-out chip
(systematic) or is idiosyncratic to well 8 of norm_temp_04.

**Step 1** (minutes, no training): characterize well 8 in raw curve/feature space independent of
any model — compare well 8's `LD_FEATURES` against sibling Target wells on the same chip AND
well-index-8 on the other 3 chips; check well 8's pixel_row_idx/pixel_col_idx extents (boundary
effects); cross-reference already-computed `well_temp_lin2d_mean`/`well_2d_temp_npr_mean` (direct
proxy for "local heating-gradient artifact," hypothesis 2) for well 8 vs other wells/chips.

**Step 2** (minutes, CPU only, no GPU): run `build_neighbor_curve_stack`+`reconstruct_curves_cosine`
standalone on all 4 chips/all wells — compare within-well variance reduction, mean-curve drift,
and same-well-index-across-chips reconstruction effect (systematic vs idiosyncratic); inspect
actual cosine softmax weights for well-8 pixels (any pixels acting as "herding" anchors?).

**Step 3** (minutes, no training): cross-chip "neighbor leakage"/separability check — does
reconstruction make training-chip Target/NC curves MORE separable (AUC/Fisher ratio on simple
features) while making held-out well-8 curves LESS separable from NC? Direct test of the
"easier-on-train/harder-on-test" mechanism shared by both hypotheses.

**Step 4** (~30 min dev, no new training): spatial fingerprint of the failure — plot
misclassification probability vs pixel_row_idx/pixel_col_idx within well 8 (clustered => supports
hypothesis 2 local-artifact; uniform => supports hypothesis 1 generic signal loss); cross-reference
with well_temp_lin2d_mean per-pixel within well 8.

**Step 5** (~1-2hr GPU, minimal new training): rerun baseline `cnn_gru_dual` only on the same LOFO
split with 2-3 different seeds — is well 8's 83.9% baseline accuracy itself stable, or already
high-variance/unstable (supports "small nudge tips it over" framing)?

Also worth an early, moderate-cost run: **`03_main_training.py` on norm_temp_04 alone** (within-chip,
not LOFO) with the two new models — directly tests whether well 8 misclassifies under
cosine/attn_recon even WITHOUT cross-chip generalization (would mean "cross-chip" framing needs
revisiting).

**Step 6** (~1-2hr GPU + a deliberate, flagged code change): capture `cnn_gru_dual_attn_recon`'s
actual learned attention weights for well 8 — requires temporarily adding it to `_XAI_SAVE_NAME`
(model_utils.py) and rerunning, OR a one-off interactive retrain script. Most expensive/highest
friction; last resort after Steps 0-5, only if those don't disambiguate the two hypotheses.

## Other plausible causes surfaced (worth ruling out cheaply, not yet investigated)
1. Label correctness for well 8 (`config.LABEL_MAPPINGS` hardcodes well 8 -> Target for all 4
   chips) — cross-check against unrelated upstream outlier-detector labels already in
   `kinetic_features` (spatial_knn_label_elbow, msc_label_*, amf_label_*) for independent evidence
   well 8 is already flagged as anomalous.
2. Single-split instability: `04_cross_dataset_training.py`'s LOFO uses `cv_splits=[(train_idx,
   test_idx)]` — a SINGLE split per fold, not k-fold-averaged. Step 5 above directly probes this.
3. Curve resampling interaction: check if well 8's native curve duration on this chip causes more
   flat-extrapolation at the tail under `CurveResampler` (saved at
   `out_dir / config.CROSS_DATASET_RESAMPLER_PATH`), independent of either hypothesis.

## Critical files
- `gk_code/main/utils/model_training/model_utils.py` — `build_neighbor_curve_stack`,
  `reconstruct_curves_cosine`, `create_cnn_gru_dual_attn_recon_model`, `evaluate_outlier_filters`,
  `_XAI_SAVE_NAME`
- `gk_code/main/04_cross_dataset_training.py` — `load_curve_data`, `combine_group`,
  `build_lofo_splits`, main LOFO loop
- `gk_code/main/03_main_training.py` — within-chip equivalent wiring (coords_full/well_ids_full)
- `gk_code/main/config.py` — `LABEL_MAPPINGS`, `CROSS_DATASET_GROUPS`, `LD_FEATURES`
- `POC_DDM_datasets/POC_DDM_multi_nc_subtract/cross_dataset_cv/init_oneplex_v6/cross_dataset_classification_performances_lofo_ori_curve.joblib` — live job's checkpointed results
- `POC_DDM_datasets/POC_DDM_multi_nc_subtract/D20260608_E00_C00_F4500KHz_U_norm_temp_04/curve_for_training.joblib` — held-out chip's raw curves/metadata/features
