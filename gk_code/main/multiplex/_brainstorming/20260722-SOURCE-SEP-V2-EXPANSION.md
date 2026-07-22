# Source-Sep V2 Expansion: 22 Variants, 2 Families

**Date:** 2026-07-22  
**Dataset:** LAB_Multiplex/02_ACA_qdPCR_balanced (N=10,383, T=45, 3 targets: KPC/NDM/VIM)  
**Status:** Design approved, implementation pending

---

## Background

The current source-sep pipeline (Run 1, documented in `20260718-SOURCE-SEP-RUN1-POSTMORTEM.md`)
trained 4 model variants:

| Key | Description |
|-----|-------------|
| `cnn_gru_source_sep` | Standard Phase 1 encoder, sigmoid head |
| `cnn_gru_source_sep_supcon` | Same architecture, "reserved for SupCon encoder" |
| `cnn_gru_source_sep_crf` | Standard Phase 1 encoder, CRF-MRF head |
| `cnn_gru_source_sep_crf_supcon` | Same as CRF, "reserved for SupCon encoder" |

**Problems with Run 1:**
- The `_supcon` variants are architecturally identical to the base (`_StandardMultiLabelModel`,
  same factory function body). There was no separate encoder weights file; both SC0 and SC1
  loaded the same `source_sep_encoder_weights.weights.h5`. The `_supcon` variants were never
  meaningfully different.
- No `_p2` vs `_p3` distinction — Phase 3 (encoder unfreeze) always ran.
- All 4 variants crammed into one SLURM job, preventing the Family B SupCon encoder idea.

**Goal:** Redesign into 22 well-defined variants across two families distinguished by Phase 1
encoder training. Consolidate all source-sep training into `03b_source_sep_pretraining.py`.

---

## SC Level Framework

SC level (0–3) controls **two distinct mechanisms** in different phases. Reading the variant grid
requires understanding both.

### Phase 1 — encoder pretraining (`03b`, reconstruction objective)

SC level defines which parts of `z` are contrastively trained, via **the same projection head
structure and Jaccard-weighted mechanism as Phase 2/3** — applied on top of the reconstruction
loss instead of BCE.

| SC | Phase 1 SupCon heads | Phase 1 loss formula | Encoder family |
|----|:--------------------:|----------------------|---------------|
| 0 | None | `L_recon` only | **A** |
| 1 | 1 on `z_full` | `L_recon + 0.2·L_sc(proj_full)` | **B** |
| 2 | 2 on `z_shared`, `z_target` | `L_recon + 0.1·[L_sc(proj_s) + L_sc(proj_t)]` | **B** |
| 3 | 3 on `z_shared`, `z_target`, `z_full` | `L_recon + 0.1·[L_sc(proj_s) + L_sc(proj_t) + L_sc(proj_f)]` | **B** |

The SupCon weights match the Phase 2/3 convention exactly (`SUPCON_LAMBDA = 0.2`, per-head
`0.1`). Only the **encoder** weights are saved after Phase 1 — the projection heads are
discarded and re-initialised fresh in Phase 2/3.

### Phase 2/3 — classification head training (`03b`, after Phase 1)

SC level sets the **head architecture** and contrastive loss structure applied on top of the encoder.

| SC | Proj heads | Loss formula | New model class |
|----|:----------:|--------------|----------------|
| 0 | None | BCE | `_StandardMultiLabelModel` (existing) |
| 1 | 1 on `z_full` | `0.8·BCE + 0.2·L_sc(proj_full)` | `MultiLabelSourceSepSCModel` |
| 2 | 2 on `z_shared`, `z_target` | `0.8·BCE + 0.1·[L_sc(proj_s) + L_sc(proj_t)]` | `MultiLabelSourceSepBranch2SCModel` |
| 3 | 3 on `z_shared`, `z_target`, `z_full` | `0.7·BCE + 0.1·[L_sc(proj_s) + L_sc(proj_t) + L_sc(proj_f)]` | `MultiLabelSourceSepBranch3SCModel` |

Phase 2/3 SupCon uses **Jaccard-weighted batch contrastive loss on L2-normalised projection head
outputs** — a fundamentally different mechanism from Phase 1 (no raw `z_j`, no per-target pairing).
See SupCon Loss Calculation section for detailed formulas.

### How Phase 1 and Phase 2/3 SC interact across families

|  | **Family A** | **Family B** |
|--|:------------|:------------|
| Phase 1 SC | SC0 — pure reconstruction, no SupCon | SC1/SC2/SC3 — same projection head SupCon as Phase 2/3, on top of `L_recon` |
| Phase 2/3 SC | Any SC0–SC3 (independent choice) | **Same SC as Phase 1** — same head structure, fresh weights |
| Encoder entering Phase 2 | Unstructured (no precon) | Precon-structured with matching SC1/SC2/SC3 geometry |
| `_p2` SC1-SC3 | **Not used** — frozen unstructured `z` → SupCon gradient stops at projection layer, cannot reach encoder | **Used** — frozen precon-structured `z` → head-level SupCon on already-SC-shaped latent |

**Family A** tests: can Phase 2/3 SupCon head structure an encoder that had no precon?
- SC0: pure BCE baseline.
- SC1-SC3 `_p3` only: Phase 3 unfreezes encoder so SupCon gradient can reshape it from unstructured init.
- SC1-SC3 `_p2` omitted: frozen unstructured encoder → SupCon gradient can't reach encoder → same as SC0 p2.

**Family B** tests: does matching Phase 1 SupCon structure to Phase 2/3 improve the classifier?
- SC level is consistent end-to-end: SC1 Phase 1 pretraining → SC1 Phase 2/3 head.
- `_p2`: head-only SupCon on a frozen encoder already structured by the same SC objective.
- `_p3`: SupCon continues to refine the encoder from a SC-aligned init.
- SC0 omitted: no SupCon in Phase 1 = Family A SC0.

---

## Design Decisions

### Decision 1: One encoder file per SC level (4 files total)

Each SC level produces a **distinct** Phase 1 encoder (see SC Level Framework for the file names
and lambda values). Family B SC0 is dropped because λ=0.0 precon = standard encoder = Family A SC0.
Result: **4 encoder files** (1 Family A + 3 Family B), **22 model variants** (10 Family A + 12 Family B).

Family B Phase 1 runs **once per SC level** — 3 separate pretraining runs, each saving its own
SC-specific encoder file.

### Decision 2: Phase distinction (_p2 vs _p3)

- **_p2**: encoder frozen throughout. Phase 3 (unfreeze + fine-tune) is skipped.
- **_p3**: full Phase 2 (frozen) → Phase 3 (unfrozen, Adam 1e-5, 100 epochs max).

Implemented by adding `'_p2' not in m` guard to the existing `if _enc is not None:` block
at line ~3020 in `model_utils_multilabel.py`.

### Decision 3: SC level grid per family

Grid follows directly from the SC Level Framework:

| Condition | SC levels | Rationale |
|-----------|-----------|-----------|
| Family A, `_p2` | SC0 only | Frozen unstructured encoder — SC1-SC3 head gradient can't reach encoder → equivalent to SC0 |
| Family A, `_p3` | SC0–SC3 | Unfrozen encoder — SC1-SC3 reshapes encoder from unstructured Phase 1 init |
| Family B, `_p2` | SC1–SC3 | Frozen precon-structured encoder — SupCon head on structured frozen `z` is meaningful |
| Family B, `_p3` | SC1–SC3 | Full end-to-end refinement from precon-structured init |
| Family B, SC0 | Omitted | λ=0.0 precon encoder = standard encoder = Family A SC0 |

### Decision 4: SC architecture distinction

- **SC0 variants** → `_StandardMultiLabelModel` (no projection head, existing base factories)
- **SC1-SC3 variants** → new `MultiLabelSourceSepSCModel` / `MultiLabelSourceSepCRFSCModel`
  with projection head + custom `train_step` adding SupCon loss (added to `_SUPCON_ML_KEYS`).

SC1/SC2/SC3 projection head placement maps by analogy to the dual-model SC convention
(see SupCon Loss Calculation section below for detailed formulas):
- SC1: 1 head on `z_full = [z_shared ‖ z_target]` (analogous to dual z_fused)
- SC2: 2 heads — `z_shared` and `z_target` separately (analogous to dual z_cnn + z_seq)
- SC3: 3 heads — `z_shared`, `z_target`, and `z_full` (analogous to dual z_cnn + z_seq + z_fused)

---

## Variant Grid (22 total)

### Family A — standard Phase 1 encoder (10 variants)

```
           SC0      SC1      SC2      SC3
p2, no CRF   ✓        —        —        —
p2, CRF      ✓        —        —        —
p3, no CRF   ✓        ✓        ✓        ✓
p3, CRF      ✓        ✓        ✓        ✓
```

### Family B — precon Phase 1 encoder, SC0 dropped (12 variants)

SC0 precon = lambda=0.0 = standard encoder = Family A SC0 → redundant, omitted.

```
                SC1      SC2      SC3
p2, no CRF       ✓        ✓        ✓
p2, CRF          ✓        ✓        ✓
p3, no CRF       ✓        ✓        ✓
p3, CRF          ✓        ✓        ✓
```

**Grand total: 22 variants** (10 Family A + 12 Family B)

---

## Naming Convention

Pattern: `cnn_gru_source_sep[_precon][_crf][_supcon{N}]_p{2|3}`

**Family A (10):**
```
cnn_gru_source_sep_p2
cnn_gru_source_sep_crf_p2
cnn_gru_source_sep_p3
cnn_gru_source_sep_supcon_p3
cnn_gru_source_sep_supcon2_p3
cnn_gru_source_sep_supcon3_p3
cnn_gru_source_sep_crf_p3
cnn_gru_source_sep_crf_supcon_p3
cnn_gru_source_sep_crf_supcon2_p3
cnn_gru_source_sep_crf_supcon3_p3
```

**Family B (12, SC0 dropped):**
```
cnn_gru_source_sep_precon_supcon_p2
cnn_gru_source_sep_precon_supcon2_p2
cnn_gru_source_sep_precon_supcon3_p2
cnn_gru_source_sep_precon_crf_supcon_p2
cnn_gru_source_sep_precon_crf_supcon2_p2
cnn_gru_source_sep_precon_crf_supcon3_p2
cnn_gru_source_sep_precon_supcon_p3
cnn_gru_source_sep_precon_supcon2_p3
cnn_gru_source_sep_precon_supcon3_p3
cnn_gru_source_sep_precon_crf_supcon_p3
cnn_gru_source_sep_precon_crf_supcon2_p3
cnn_gru_source_sep_precon_crf_supcon3_p3
```

Original 4 keys (`cnn_gru_source_sep`, `cnn_gru_source_sep_supcon`, `cnn_gru_source_sep_crf`,
`cnn_gru_source_sep_crf_supcon`) kept for backward compatibility.

---

## Files to Modify

### 1. `config_multiplex.py`

Add `'source_sep_precon'` entry to `RESULT_FILE_BY_FLAG`. Since `RESULT_10FOLD_FILE_BY_FLAG`
is auto-derived from it via `.replace('.joblib', '_10fold.joblib')`, no second change needed.

```python
# In RESULT_FILE_BY_FLAG dict (line 47):
RESULT_FILE_BY_FLAG = {
    ...
    'source_sep':        'classification_performances_ml_source_sep.joblib',
    'source_sep_precon': 'classification_performances_ml_source_sep_precon.joblib',  # NEW
}
# RESULT_10FOLD_FILE_BY_FLAG auto-derives both entries — no change needed there.
```

Also update `get_result_flag_for_model(key)` (line ~163) — check `_precon` before `source_sep`
since the precon keys contain both substrings:

```python
# Was: if 'source_sep' in key: return 'source_sep'
if 'source_sep_precon' in key: return 'source_sep_precon'   # NEW — must come first
if 'source_sep' in key:        return 'source_sep'
```

---

### 2. `model_utils_source_sep.py`

**Phase 2/3 classification model classes** (SC1-SC3 head training):
- `MultiLabelSourceSepSCModel` — non-CRF SC1-SC3 classification head
- `MultiLabelSourceSepCRFSCModel` — CRF SC1-SC3 classification head
- Both override `train_step` to add Jaccard-weighted SupCon loss on projection head output. Mirror `MultiLabelSupConModel` family.

**Phase 1 pretraining model classes** (Family B SC1-SC3 encoder pretraining):
- `MultiLabelSourceSepPhase1SC1Model` — reconstruction + SC1 SupCon on `z_full`
- `MultiLabelSourceSepPhase1SC2Model` — reconstruction + SC2 SupCon on `z_shared`, `z_target`
- `MultiLabelSourceSepPhase1SC3Model` — reconstruction + SC3 SupCon on all three
- All three override `train_step` to add Jaccard-weighted SupCon (same constants: `SUPCON_TEMP=0.1`, per-head lambda `0.1`) to the existing reconstruction loss.
- After Phase 1 training, only the encoder sub-model's weights are extracted and saved. Projection heads are discarded.

---

### 3. `model_utils_multilabel.py`

**Encoder weights helpers** (new module-level utility):

```python
def _precon_encoder_file(m):
    # Family B has no SC0 — precon encoder exists only for SC1/SC2/SC3
    if '_supcon3' in m: return 'source_sep_precon_sc3_encoder_weights.weights.h5'
    if '_supcon2' in m: return 'source_sep_precon_sc2_encoder_weights.weights.h5'
    if '_supcon'  in m: return 'source_sep_precon_sc1_encoder_weights.weights.h5'
    raise ValueError(f'No precon encoder for SC0 key {m!r} — Family B SC0 was dropped')
```

**Training loop — encoder loading (replaces existing single-path check):**

```python
if m in _SS_ML_KEYS:
    if m in _SS_PRECON_ML_KEYS:
        _w = os.path.join(exp_path, _precon_encoder_file(m))
    else:
        _w = encoder_weights_path  # standard Family A path
    if _w and os.path.exists(_w):
        _enc = model.get_layer('source_sep_encoder')
        _enc.load_weights(_w)
        _enc.trainable = False  # Phase 2
```

Note: `encoder_weights_path` is still passed in for Family A (standard file). Family B no longer
uses a single `precon_encoder_weights_path` — instead the path is derived from the model key.

**Training loop — Phase 3 guard:**

```python
# Was: if _enc is not None:
if _enc is not None and '_p2' not in m:
    _enc.trainable = True
    ...
```

**Key sets (append to existing):**

```python
_SS_ML_KEYS |= frozenset(ALL_22_NEW_KEYS)
_SS_PRECON_ML_KEYS = frozenset(k for k in ALL_22_NEW_KEYS if '_precon' in k)
_CRF_ML_KEYS      |= frozenset(k for k in ALL_22_NEW_KEYS if '_crf' in k)
_SUPCON_ML_KEYS   |= frozenset(k for k in ALL_22_NEW_KEYS if '_supcon' in k)
```

**Factory registration:** 8 underlying architectures (SC0 no-CRF, SC0 CRF, SC1 no-CRF,
SC1 CRF, SC2 no-CRF, SC2 CRF, SC3 no-CRF, SC3 CRF). All 22 new keys map to one of these 8.
`_precon` and `_p2`/`_p3` have no architectural effect — same model, training behaviour differs
via key-set lookups.

**`ML_MODEL_PRINT_MAP`:** Add 22 entries, e.g.:
```python
'cnn_gru_source_sep_p2':                     'SrcSep-A SC0 p2',
'cnn_gru_source_sep_precon_crf_supcon2_p3':  'SrcSep-B CRF SC2 p3',
# etc.
```

---

### 4. `03b_source_sep_pretraining.py`

**New CLI args:**

| Arg | Type | Default | Purpose |
|-----|------|---------|---------|
| `--precon` | flag | False | Family B: Phase 1 trains with SC-specific SupCon and saves precon encoder |
| `--train_phase23` | flag | False | Run Phase 2+3 (uses encoder file matching `--supcon` level) |
| `--supcon` | int 0-3 | 0 | SC level — controls Phase 1 SupCon head structure (when `--precon`) and Phase 2+3 head architecture |
| `--n_splits` | int | 5 | CV folds for Phase 2+3 |
| `--force_rerun_phase23` | flag | False | Clear + retrain Phase 2+3 for this SC level |

**Phase 1 encoder file selection and model choice:**

```python
ENCODER_WEIGHTS_FILE = 'source_sep_encoder_weights.weights.h5'  # existing constant

if args.precon:
    assert args.supcon in (1, 2, 3), f"--precon requires --supcon in [1,2,3]; got {args.supcon}"
    out_path = os.path.join(exp_path,
        f'source_sep_precon_sc{args.supcon}_encoder_weights.weights.h5')
    # Phase 1 model: SC-specific (projection heads + Jaccard SupCon on L_recon)
    # uses same SUPCON_LAMBDA/SUPCON_TEMP as Phase 2/3 — no separate lambda_supcon
else:
    out_path = os.path.join(exp_path, ENCODER_WEIGHTS_FILE)
    # Phase 1 model: standard MultiLabelSourceSepPhase1Model (reconstruction only)
```

Family B Phase 1 model: new `MultiLabelSourceSepPhase1SC{N}Model` classes (in `model_utils_source_sep.py`) with SC-specific projection heads and `train_step` that adds Jaccard-weighted SupCon to `L_recon`. After Phase 1, only the encoder sub-model is saved; projection heads are discarded.

**Phase 2+3 key selection:**

```python
def _get_phase23_keys(supcon_level, precon):
    # precon=True only valid for supcon_level in [1,2,3] — SC0 precon dropped
    assert not (precon and supcon_level == 0), "Family B SC0 does not exist"
    precon_sfx = '_precon' if precon else ''
    sc_sfx     = ['', '_supcon', '_supcon2', '_supcon3'][supcon_level]
    base       = f'cnn_gru_source_sep{precon_sfx}'
    keys = [f'{base}{sc_sfx}_p3', f'{base}_crf{sc_sfx}_p3']
    # p2 variants: Family A SC0 only  OR  Family B SC1-SC3 (structured precon encoder)
    if (not precon and supcon_level == 0) or precon:
        keys += [f'{base}{sc_sfx}_p2', f'{base}_crf{sc_sfx}_p2']
    return keys
```

**Phase 2+3 execution block:**

```python
from model_utils_multilabel import train_and_evaluate_ml_models, ...

result_flag = 'source_sep_precon' if args.precon else 'source_sep'
keys = _get_phase23_keys(args.supcon, args.precon)

train_and_evaluate_ml_models(
    models_to_train=keys,
    encoder_weights_path=(out_path if not args.precon else None),
    # Family B: training loop derives path from key via _precon_encoder_file(m)
    # encoder_weights_path is None for Family B (not used)
    result_flag=result_flag,
    n_splits=args.n_splits,
    force_rerun=args.force_rerun_phase23,
    exp_path=exp_path,   # NEW — needed for _precon_encoder_file() lookup
    ...
)
```

Note: `train_and_evaluate_ml_models` needs `exp_path` added so the training loop can
construct the precon encoder path per model key.

---

### 5. `03_main_training.py`

- Remove `--source_sep` argument (~line 116) and its `elif args.source_sep:` routing block
- Remove `_encoder_weights_path` construction (source-sep only)
- Remove `'source_sep'` from `_result_flag` chain
- All other model groups unchanged

---

## SLURM Job Structure

**`lab_multiplex_source_sep_training.sh`** (Family A — update existing, time 12h):

```bash
# Phase 1 — standard encoder (lambda=0.0), run once
python -u 03b_source_sep_pretraining.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" \
    [p1 hyperparams] --lambda_supcon 0.0 [--force_rerun] --validate

# SC0 → 4 variants: _p2, _crf_p2, _p3, _crf_p3
python -u 03b_source_sep_pretraining.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" \
    --train_phase23 --supcon 0 --n_splits 5 --force_rerun_phase23

# SC1-SC3 → 2 variants each: _supcon{N}_p3, _crf_supcon{N}_p3
for SC in 1 2 3; do
    python -u 03b_source_sep_pretraining.py \
        --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" \
        --train_phase23 --supcon $SC --n_splits 5 --force_rerun_phase23
done
```

**`lab_multiplex_source_sep_precon_training.sh`** (Family B — new file, time 12h):

```bash
# SC1/SC2/SC3 only — SC0 precon is identical to Family A SC0 and is omitted.
# Each SC: (1) Phase 1 precon at SC lambda, (2) Phase 2+3 with SC head (4 variants).
for SC in 1 2 3; do
    # Phase 1 — saves source_sep_precon_sc${SC}_encoder_weights.weights.h5
    python -u 03b_source_sep_pretraining.py \
        --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" \
        --precon --supcon $SC \
        [p1 hyperparams] [--force_rerun] --validate
    
    # Phase 2+3 — 4 variants: _supcon{N}_p2, _crf_supcon{N}_p2, _supcon{N}_p3, _crf_supcon{N}_p3
    python -u 03b_source_sep_pretraining.py \
        --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" \
        --precon --train_phase23 --supcon $SC --n_splits 5 --force_rerun_phase23
done
```

Both jobs can be `sbatch`-ed simultaneously. Family A and B write to separate result files
(`source_sep` vs `source_sep_precon`) — no write collision.

---

## SupCon Loss Calculation for SC1/SC2/SC3 *(Technical Reference)*

Detailed formulas for Phase 2/3 SupCon loss. Verified against `model_utils_multilabel.py`
(`MultiLabelSupConModel`, `MultiLabelSupConBranch2STModel`, `MultiLabelSupConBranch3STModel`)
and `model_utils_supcon.py`. Summary is in the SC Level Framework above.

### Global constants (from `model_utils_supcon.py`)

```python
SUPCON_TEMP   = 0.1    # temperature τ
SUPCON_LAMBDA = 0.2    # SC1 total lambda weight
# SC2/SC3 use supcon_lambda_each = 0.1 (per head; total = 0.2 / 0.3)
```

### Positive pair weighting

Continuous Jaccard similarity (not binary):

```python
pos_mask[i,j] = |y_i ∩ y_j| / |y_i ∪ y_j|    (diagonal = 0)
```

Identical multilabel vectors → 1.0; partial overlap → fractional; disjoint → 0.0.

### SupCon loss per projection head (`_supcon_loss_jaccard`)

Each projection head outputs L2-normalised embeddings `p ∈ (N, D)`:

```
sim[i,j]      = p_i · p_j / τ
log_prob[i,j] = sim[i,j] − log Σ_{k≠i} exp(sim[i,k])
L_sc(p)       = −mean_i [ Σ_j pos_mask[i,j] · log_prob[i,j] / Σ_j pos_mask[i,j] ]
```

Anchors with no positives (pos_sum = 0) are masked out of the mean.

### SC variants in the existing dual-model codebase

For dual-branch (CNN + GRU) models, `z_cnn` and `z_seq` are the branch latents;
`z_fused = concat(z_cnn, z_seq)`.

| Variant | Heads | Input(s) | Total loss formula | BCE_w | SC_total |
|---------|-------|----------|--------------------|-------|---------|
| **SC1** | 1 | `z_fused` | `(1−0.2)·BCE + 0.2·L_sc(proj_fused)` | 0.8 | 0.2 |
| **SC2** | 2 | `z_cnn`, `z_seq` | `(1−0.2)·BCE + 0.1·[L_sc(proj_cnn) + L_sc(proj_seq)]` | 0.8 | 0.2 |
| **SC3** | 3 | `z_cnn`, `z_seq`, `z_fused` | `(1−0.3)·BCE + 0.1·[L_sc(proj_cnn) + L_sc(proj_seq) + L_sc(proj_fused)]` | 0.7 | 0.3 |

Classes: `MultiLabelSupConModel` (SC1), `MultiLabelSupConBranch2STModel` (SC2),
`MultiLabelSupConBranch3STModel` (SC3).

### Source-sep branch mapping

The source-sep encoder is **sequential** (CNN→GRU), not dual-branch. Its output splits into:
- `z_shared` — first `d_shared` dims (analogous to CNN branch embedding)
- `z_target = [z_1 ‖ … ‖ z_J]` — remaining `J × d_target` dims (analogous to GRU branch embedding)
- `z_full = [z_shared ‖ z_target]` — full encoder output (analogous to fused)

Mapping SC1/SC2/SC3 by structural analogy:

| Variant | Heads | Input(s) | Total loss | BCE_w | SC_total |
|---------|-------|----------|------------|-------|---------|
| **SC1** | 1 | `z_full` | `(1−0.2)·BCE + 0.2·L_sc(proj_full)` | 0.8 | 0.2 |
| **SC2** | 2 | `z_shared`, `z_target` | `(1−0.2)·BCE + 0.1·[L_sc(proj_shared) + L_sc(proj_target)]` | 0.8 | 0.2 |
| **SC3** | 3 | `z_shared`, `z_target`, `z_full` | `(1−0.3)·BCE + 0.1·[L_sc(proj_shared) + L_sc(proj_target) + L_sc(proj_full)]` | 0.7 | 0.3 |

`_proj_head` (reused from `model_utils_supcon.py`) takes a slice of `z` and produces an
L2-normalised embedding. The split points are known at build time: `z[:, :d_shared]` for
`z_shared` and `z[:, d_shared:]` for `z_target`.

### Gradient flow across all phases

| Phase | Encoder | SC loss acts on | Gradient to encoder | Gradient to proj head |
|-------|---------|-----------------|--------------------|-----------------------|
| Phase 1 Family B (precon) | trainable | `L_recon + SC` | Yes | Yes |
| Phase 2 `_p2`/`_p3` | frozen | `BCE + SC` | None | Yes |
| Phase 3 `_p3` | unfrozen | `BCE + SC` | Yes | Yes |

**Family A SC1-SC3 `_p2`**: frozen unstructured encoder → SupCon head gradient stops at projection layer, cannot reshape encoder → equivalent to SC0 `_p2` → dropped.

**Family B SC1-SC3 `_p2`**: frozen precon-structured encoder → head-level SupCon on a latent already shaped by the same SC structure during Phase 1 → meaningful experiment.

### Phase 1 SupCon (Family B) vs Phase 2/3 SupCon

**Same core mechanism** — Jaccard-weighted batch contrastive on L2-normalised projection head
outputs, same `SUPCON_TEMP = 0.1`, same per-head lambda convention.

**Difference is the base loss:**

| Phase | SupCon sits on top of | Encoder trainable? | Proj head saved? |
|-------|----------------------|-------------------|-----------------|
| Phase 1 (Family B) | `L_recon` (reconstruction + decoder) | Yes | No — discarded, fresh in Phase 2 |
| Phase 2 | BCE | No (`_enc.trainable = False`) | Yes |
| Phase 3 | BCE | Yes (`_enc.trainable = True`) | Yes |

The existing `_per_target_supcon_loss` in `MultiLabelSourceSepPhase1Model` is **replaced** by the
SC-specific projection head SupCon for Family B SC1-SC3. `MultiLabelSourceSepPhase1Model` (SC0)
keeps the original reconstruction-only loss.

---

## Open Questions

1. **`exp_path` plumbing**: The training loop needs `exp_path` to construct the per-model precon
   encoder path via `_precon_encoder_file(m)`. Current proposal: add `exp_path` as a new keyword
   argument to `train_and_evaluate_ml_models` (currently absent from its signature — confirmed at
   line 2805 of `model_utils_multilabel.py`). Alternative: pass a pre-built
   `{model_key: encoder_path}` dict instead. Resolve during implementation.

---

## Phase 1 Reconstruction Quality — Improvement Plan (V2 Follow-Up)

### Problem statement

Phase 1 standard (SC0) reconstruction on `02_ACA_qdPCR_balanced` converges to `val_loss ≈ 0.050`.
Training converges fully (ReduceLR has fired multiple times by epoch 200). **More epochs will not
help** — the model is at its capacity wall, not an optimisation issue.

**On lambda_cons=2.0 (secondary attempt, log 264349):** Did not help for SC0. The precon SC1/2/3
Phase 1 with lambda_cons=1.0 (same log) reaches `val_l_consist ≈ 0.0047` — but this is hidden
inside `val_loss ≈ 0.53` due to the SupCon component dominating. The actual reconstruction
quality IS better with precon training; the high val_loss is a measurement artefact, not a
reconstruction problem.

**Root cause 1 (standard SC0)**: encoder (Conv1D×2 + BiGRU(32), ~16K params) lacks capacity to
disentangle overlapping sigmoid curves at the balanced mixture distribution.

**Root cause 2 (Phase 1 SC models, bug)**: `EarlyStopping(monitor='val_loss')` in
`03b_source_sep_pretraining.py` monitors the total Phase 1 loss, which for SC1/2/3 models
includes a large noisy SupCon term (e.g. `0.2 × l_sc ≈ 0.48` for SC1). This makes val_loss
an unreliable stopping signal — early stopping may fire on SupCon noise before reconstruction
has truly converged. Fix: monitor `val_l_consist` for Phase 1.

---

### Architecture: serial vs dual branch

**Context:** `cnn_gru_dual` empirically outperforms serial `cnn_gru` on single-task classification
in this codebase (same single-modality input). The dual branch here is **not** two different input
types — it is two different processing paths on the same amplification curve:
- CNN branch: local short-range features (rise onset shape, inflection sharpness)
- GRU branch: temporal context (how the curve evolves, plateau behaviour)

This has direct relevance for source separation: identifying individual sigmoid components in an
overlapping mixture benefits from both feature types simultaneously, not sequentially.

| Option | Pros | Cons |
|--------|------|------|
| **Serial bigger** (Conv1D×2→BiGRU(64)) | Simple change, bottleneck unchanged | Only increases sequential depth; doesn't add the dual-feature extraction that already wins in classification |
| **Dual branch bigger** | Empirically proven architecture in this codebase; CNN + GRU independence may help decompose overlapping sigmoids better; more parameters from both branches | Larger architecture change in `build_source_sep_encoder`; longer Phase 1 training time |

**Recommendation**: **dual branch + bigger capacity**. The dual branch MUST be bigger — a dual
branch that doesn't increase total capacity just splits the existing ~16K encoder params into two
~8K branches, each smaller than the current serial model, which is likely to perform worse.
`cnn_gru_dual` wins in benchmarks because it has both dual feature extraction AND more total
parameters. Target: each branch ~16-32K params, total encoder ~48-64K params.

Serial bigger is the fallback if the architectural change creates risk before V2 evaluation.

---

### Primary change: dual branch + bigger capacity

**Encoder** (in `build_source_sep_encoder`, `model_utils_source_sep.py`):

Mirror the existing `cnn_gru_dual` structure — shared trunk, then split into CNN and GRU branches,
merge before the bottleneck Dense. Bottleneck output dim stays `d_shared + n_targets * d_target = 46`.

```python
# Current (serial):
Input(T,1) → Conv1D(16,5) → Conv1D(16,3,s=2) → BiGRU(32) → LayerNorm → Dense(46)

# Proposed (dual branch):
Input(T,1) → Conv1D(32, 5, causal, relu)          # shared trunk
           → split:
               Branch CNN: Conv1D(32,3,s=2,causal) → Conv1D(32,3,causal) → z_cnn (via Dense or GlobalPool)
               Branch GRU: Conv1D(16,3,s=2,causal) → BiGRU(64) → z_gru
           → Concat([z_cnn, z_gru]) → LayerNorm → Dense(46)
```

Exact branch widths TBD during implementation — target total params ~4–6× current encoder
(~48-64K), while keeping the output dim at 46 (invariant). Check against existing
`build_cnn_gru_dual_model` in `model_utils_cnn_gru.py` for layer naming conventions and merge
pattern to follow.

**Decoder** (in `build_parametric_decoder`, `model_utils_source_sep.py`):

The decoder already has `Dense(32, relu, name=f'dec{j}_hidden')` as a single hidden layer
before 5 separate `Dense(1)` heads with constrained activations (softplus, sigmoid×T, etc.).
Increase the hidden layer width rather than adding a new layer:

```python
# Current:
h = Dense(32, activation='relu', name=f'dec{j}_hidden')(inp)  # inp = (d_shared+d_target,)

# Proposed:
h = Dense(128, activation='relu', name=f'dec{j}_hidden')(inp)
```

This is the layer the 5 sigmoid param heads all branch off from — wider = more shared
representation capacity for the parametric mapping.

Gives the decoder more capacity to map the 26-dim slice to sigmoid parameters even when `z_shared`
contains mixed information from multiple active targets.

**Impact**: Phase 1 only. No Phase 2/3 interface changes — encoder output dim unchanged,
`_SS_D_SHARED`/`_SS_D_TARGET` constants unchanged, layer name `source_sep_encoder` unchanged.
All 22 model variants load by layer name and are unaffected by the internal body change.

**Cost**: must retrain Phase 1 (all datasets) and all Phase 2+3 variants with
`--force_rerun_phase23` since encoder weights change.

---

### Fallback: serial bigger (simpler, lower risk)

If dual branch adds too much complexity before V2 evaluation:

```python
# Serial bigger:
Conv1D(32, 5, ...) → Conv1D(32, 3, stride=2) → BiGRU(64) → LayerNorm → Dense(46)
```

Same cost as dual branch (full retrain), lower expected gain but zero architectural risk.

---

### Secondary change: loss weight tuning

Can try before committing to architecture change (no retraining of Phase 2+3 needed for Phase 1
hyperparameter tuning, just retrain Phase 1):

| Hyperparameter | Current | Try | Effect |
|---------------|---------|-----|--------|
| `lambda_cons` | 1.0 | 2.0 | More weight on reconstruction vs anchor/variance |
| `lambda_anch` | 0.5 | 0.3 | Less anchor pull (may help if anchor bank is too rigid) |
| `lambda_var`  | 0.1 | 0.05 | Usually already 0 at convergence — minimal effect |
| `nn_k` (anchor neighbours) | 5 | 3 or 10 | Tighter (3) or looser (10) reference matching |

Expected impact from `lambda_cons 2.0`: small improvement (~0.005) since the model has already
saturated its capacity. Worth trying as a low-cost first step.

---

### Epochs and callbacks

**More epochs needed for the bigger model?** Yes — a 3-4× bigger dual encoder has more parameters
and will converge more slowly. Change `--epochs_p1 200` → `--epochs_p1 400` when running with the
new architecture.

**Not because of lambda_cons amplification** — ReduceLROnPlateau handles that automatically
(steeper gradient → bigger val_loss oscillations → LR drops sooner). lambda_cons doesn't require
tuning the epoch count directly.

**EarlyStopping fix (bug)** — `03b_source_sep_pretraining.py` currently does:
```python
EarlyStopping(monitor='val_loss', patience=30, ...)
```
For SC1/2/3 Phase 1 models, `val_loss` includes a large SupCon term (`0.2×l_sc ≈ 0.48` for SC1),
making it a noisy stopping signal. The reconstruction component (`val_l_consist`) can still be
improving while `val_loss` oscillates due to SupCon noise → early stopping may fire prematurely.

Fix: use `val_l_consist` as the EarlyStopping monitor for all Phase 1 training:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_l_consist', patience=30, restore_best_weights=True),  # was val_loss
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_l_consist', factor=0.3, patience=15, min_lr=1e-5),   # was val_loss
    tf.keras.callbacks.TerminateOnNaN(),
]
```

This also applies to SC0 (where `val_loss == val_l_consist + other_recon_terms` so the direction
is correct, though `val_loss` would also work for SC0 since no SupCon noise).

**Hyperparameter changes that likely won't help for SC0 specifically:**
- **More epochs** (SC0 already converged): model is at capacity wall, ReduceLR has fired — no headroom.
- **Different LR schedule**: ReduceLROnPlateau already adapts adequately.

---

### Suggested experiment order

1. **Fix EarlyStopping to monitor `val_l_consist`** in `03b_source_sep_pretraining.py` —
   zero-cost change, applies immediately to all future runs.

2. **Implement dual branch + bigger encoder (48-64K params) + wider decoder (Dense 32→128)**.
   Run with `--epochs_p1 400`. Monitor `val_l_consist` for stopping (from fix in step 1).
   Expect `val_l_consist` for SC0 standard to drop below 0.030 (better than current ~0.050).

3. **If SC0 standard val_l_consist is still ≥ 0.040 after 400 epochs**: try `lambda_cons 2.0`
   additionally. With a bigger model, the stronger gradient signal may help more.

4. **Only if all above fail**: consider data augmentation (Gaussian noise, amplitude jitter)
   to address the small balanced dataset size.

---

### Implementation scope

Files to change:
- `03b_source_sep_pretraining.py`: EarlyStopping + ReduceLR → monitor `val_l_consist`; bump
  default `--epochs_p1` to 400
- `model_utils_source_sep.py`: `build_source_sep_encoder` — replace serial body with dual branch
  (shared trunk Conv1D(32,5) → split CNN branch + GRU branch → Concat → LayerNorm → Dense(46));
  `build_parametric_decoder` — increase hidden layer `Dense(32)` → `Dense(128)`
- Reference `build_cnn_gru_dual_model` in `model_utils_cnn_gru.py` for branch naming convention
  and merge pattern to stay consistent with existing dual models
- SLURM scripts: update `--epochs_p1 200` → `--epochs_p1 400`; bottleneck dims unchanged
- Run `--force_rerun` on Phase 1 and `--force_rerun_phase23` on Phase 2+3 for all datasets
