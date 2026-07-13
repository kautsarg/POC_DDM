# Inception Smoothing Front-End

## Motivation

Previous smoothing approaches:
- `ori_curves_avg` — moving average, requires manual window size
- `ori_curves_wavelet_sym8` — wavelet denoising, requires manual wavelet choice + threshold

**Goal:** Replace manual hyperparameter with a learned, end-to-end trainable smoothing front-end that adapts to the data. Input remains `ori_curves` (raw). No new curve variant in joblib.

## Architecture

```
raw curve (N, T, 1)
    ↓
Inception Smoothing Block
    Conv1D(8, k=3,  pad='same', relu)  ─┐
    Conv1D(8, k=7,  pad='same', relu)  ─┤─ Concatenate → (N, T, 32)
    Conv1D(8, k=15, pad='same', relu)  ─┤
    Conv1D(8, k=31, pad='same', relu)  ─┘
    Conv1D(8, k=1,  pad='same', relu)     ← 1×1 bottleneck → (N, T, 8)
    ↓
Existing dual-branch model body (CNN branch + GRU/Transformer branch)
```

- **Multiple kernel sizes**: 3, 7, 15, 31 — cover short to long-range temporal context
- **Bottleneck**: 1×1 Conv1D compresses 32 → 8 channels (narrow by design — smoothing, not feature extraction)
- **Shared**: both CNN and GRU/Transformer branches in dual-branch models see the same inception output
- **Learned**: the kernel weights are optimised jointly with the classification loss — the model learns what "smooth" means for this specific data

## Compared to Alternatives

| Method | Hyperparameter | How it's set | Learned? |
|---|---|---|---|
| `ori_curves_avg` | Window size | Manual | No |
| `ori_curves_wavelet_sym8` | Wavelet family + threshold | Manual (sym8 + universal) | No |
| Inception smoothing | Filter count, kernel sizes | Fixed architecture, weights learned | Yes |

## Usage

```bash
python 03_main_training.py --task_id 0 --exp_folder ... --inception_smoothing
```

Flag: `--inception_smoothing` (action="store_true"). Works with any `--curve_type` (typically `ori_curve`).

## Implementation Notes

- Activation: `--inception_smoothing` in `03_main_training.py`, threaded to `evaluate_outlier_filters`
- Models covered: all single-input and late-fusion models in `model_utils.py` + all 8 gated models in `model_utils_gated.py`
- Not applicable: `cnn_gru_dual_attn_recon` (uses `(k+1, T)` TimeDistributed input — structurally incompatible)
- Inference: inception layers are baked into the saved `.keras` model — `06`/`07` load and run it with no extra flags
