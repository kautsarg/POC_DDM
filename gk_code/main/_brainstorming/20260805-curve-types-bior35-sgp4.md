# Add `bior3.5` wavelet and `SG p=4` as new curve types (01 → 07)

## Context

The notebook (`multi_wavelet_denoising_v2.ipynb`) identified `bior3.5` (wavelet) and SG polyorder=4 (with auto window sweep) as strong denoising candidates. The goal is to bake these into the production pipeline so they're available everywhere `ori_curve_wavelet_sym8` already is — preprocessing through attribution reports.

Two new curve types:
- **`ori_curve_wavelet_bior35`** → joblib key `ori_curves_wavelet_bior35` (pywt `bior3.5`, Donoho-Johnstone threshold — same mechanism as sym8)
- **`ori_curve_sg_p4`** → joblib key `ori_curves_sg_p4` (SG polyorder=4, optimal window found by the same PRS/ROS sweep used in the notebook)

The dot in `bior3.5` is replaced by `35` in all key/filename contexts; the `pywt.Wavelet('bior3.5')` call uses the real name.

### Safety confirmation
- **03** builds `_target_names` from the CLI list and loops over `dataset_name` from the joblib, silently skipping any type not present in the data. → **safe to add to defaults** regardless of whether 01 has been rerun.
- **05, 06, 07** each wrap `resolve_curve_dataset_idx` in `try/except ValueError` and print `[SKIP]` if the type is missing. → **safe to add to defaults**.
- Existing datasets that haven't been rerun through the updated 01 will just skip the new types, producing no error.

---

## Changes per file

### 1. `config.py`

**`CURVE_TYPE_ALIASES`** (line 61) — add 2 entries:
```python
"ori_curve_wavelet_bior35": "ori_curves_wavelet_bior35",
"ori_curve_sg_p4":          "ori_curves_sg_p4",
```

**`CURVE_PRINT_MAP`** (line 621) — add 2 entries for `08_statistical_comparison.py` labels:
```python
"ori_curves_wavelet_bior35": "Wavelet bior3.5",
"ori_curves_sg_p4":          "SG p=4 (auto-w)",
```

---

### 2. `01_curve_preprocessing_v6.py` — main work

**Imports** — add after the existing `import pywt` line:
```python
from scipy.signal import savgol_filter
```

**New helper functions** — add after `wavelet_denoise_curves` (~line 325):

```python
def _ensure_odd(w):
    return w + (1 - w % 2)

def _sg_derivative_scores(curves, windows, polyorder, sample_size=400, seed=0):
    """PRS / ROS sweep matching the notebook's _derivative_scores algorithm."""
    rng    = np.random.default_rng(seed)
    sample = curves[rng.choice(len(curves), size=min(sample_size, len(curves)), replace=False)]
    _ref_w = max(polyorder + 2, _ensure_odd(int(sample.shape[1] * 0.03)))
    ref    = savgol_filter(sample, window_length=_ref_w, polyorder=2, axis=1)
    peak_r = np.abs(np.diff(ref, axis=1)).max(axis=1)
    ro_raw = np.std(np.diff(np.diff(sample, axis=1), axis=1), axis=1)
    prs, ros = [], []
    for w in windows:
        w_  = max(_ensure_odd(int(w)), polyorder + 2)
        df  = np.diff(savgol_filter(sample, window_length=w_, polyorder=polyorder, axis=1), axis=1)
        prs.append(np.mean(np.abs(df).max(axis=1) / np.where(peak_r > 0, peak_r, 1)))
        ros.append(np.mean(np.std(np.diff(df, axis=1), axis=1) / np.where(ro_raw > 0, ro_raw, 1)))
    return np.array(prs), np.array(ros)

def _sg_sweet_spot(windows, roughnesses):
    """First window where ROS <= 2x 10th-percentile floor (matches notebook)."""
    floor = np.percentile(roughnesses, 10)
    sweet = np.where(roughnesses <= 2.0 * floor)[0]
    return int(windows[sweet[0]] if len(sweet) else windows[np.argmin(roughnesses)])

def sg_p4_denoise_curves(curves):
    """Auto-sweep optimal window for SG polyorder=4, return (denoised, optimal_w)."""
    T       = curves.shape[1]
    windows = np.unique([_ensure_odd(int(w))
                         for w in np.linspace(5, max(7, int(T * 0.25)), 40)])
    windows = windows[windows >= 6]   # polyorder=4 needs w >= 6
    _, ros  = _sg_derivative_scores(curves, windows.astype(float), polyorder=4)
    opt_w   = _sg_sweet_spot(windows, ros)
    return savgol_filter(curves, window_length=opt_w, polyorder=4, axis=1), opt_w
```

**`save_experiment_data_restructured()` signature** — add two params:
```python
def save_experiment_data_restructured(..., wavelet_bior35=False, sg_p4=False):
```

**Inside the function body** — initialize `_sg_w = None` BEFORE the `if`-blocks, then populate:
```python
_sg_w = None   # set here so save_data can reference it unconditionally below
if wavelet_bior35:
    curves_dict["ori_curves_wavelet_bior35"] = wavelet_denoise_curves(
        processed_curves[0], wavelet="bior3.5", level=5)
if sg_p4:
    curves_dict["ori_curves_sg_p4"], _sg_w = sg_p4_denoise_curves(processed_curves[0])
```

**`save_data` dict** — add `sg_p4_optimal_w` alongside other metadata:
```python
"sg_p4_optimal_w": _sg_w,
```

**CLI flags** — add two new flags:
```python
parser.add_argument("--wavelet_bior35", action="store_true",
    help="Add 'ori_curves_wavelet_bior35': bior3.5 wavelet, Donoho-Johnstone threshold.")
parser.add_argument("--sg_p4", action="store_true",
    help="Add 'ori_curves_sg_p4': SG polyorder=4 with auto window sweep.")
```

**Patch logic** (~lines 469–496) — add two more checks in the `needs_*` block:
```python
needs_bior35_patch = args.wavelet_bior35 and "ori_curves_wavelet_bior35" not in existing_data["curves"]
needs_sg_p4_patch  = args.sg_p4          and "ori_curves_sg_p4"          not in existing_data["curves"]
```
Skip condition update (to include the new checks):
```python
if not any([needs_avg_patch, needs_norm_patch, needs_wavelet_patch,
            needs_bior35_patch, needs_sg_p4_patch]):
```
And the patching block:
```python
if needs_bior35_patch:
    existing_data["curves"]["ori_curves_wavelet_bior35"] = wavelet_denoise_curves(
        existing_data["curves"]["ori_curves"], wavelet="bior3.5", level=5)
    patched_fields.append("ori_curves_wavelet_bior35")
if needs_sg_p4_patch:
    existing_data["curves"]["ori_curves_sg_p4"], existing_data["sg_p4_optimal_w"] = \
        sg_p4_denoise_curves(existing_data["curves"]["ori_curves"])
    patched_fields.append("ori_curves_sg_p4")
```

**Call site** — pass the two new flags when calling `save_experiment_data_restructured`:
```python
wavelet_bior35=args.wavelet_bior35, sg_p4=args.sg_p4
```

---

### 3. `02_outlier_detection_pipeline.py`

**`_build_variant_lists()`** (~lines 208–231) — add two blocks after the `ori_curves_wavelet_sym8` block:
```python
if "ori_curves_wavelet_bior35" in data["curves"]:
    dataset_name.append("ori_curves_wavelet_bior35")
    dataset.append(data["curves"]["ori_curves_wavelet_bior35"])

if "ori_curves_sg_p4" in data["curves"]:
    dataset_name.append("ori_curves_sg_p4")
    dataset.append(data["curves"]["ori_curves_sg_p4"])
```

**`ae_names` filter** (line 627) — add both new variants to the AE inclusion set:
```python
ae_names = [n for n in dataset_name
            if n in ('ori_curves', 'ori_curves_avg', 'ori_curves_wavelet_sym8',
                     'ori_curves_wavelet_bior35', 'ori_curves_sg_p4')]
```

---

### 4. `03_main_training.py`

**Default `--curve_type`** (line 91) — extend to 5 types:
```python
default=["ori_curve", "ori_curve_avg", "ori_curve_wavelet_sym8",
         "ori_curve_wavelet_bior35", "ori_curve_sg_p4"]
```

---

### 5. `04_cross_dataset_training.py`

**Default `--curve_type`** (line 174) — extend to 5 types:
```python
default=["ori_curve", "ori_curve_avg", "ori_curve_wavelet_sym8",
         "ori_curve_wavelet_bior35", "ori_curve_sg_p4"]
```

**⚠️ Fix the iteration order** (line 223) — the current `[2, 1, 0]` index list silently drops indices 3 and 4 when the list grows to 5. Change to:
```python
ordered = list(reversed(args.curve_type))
```

---

### 6. `04b_arch_poc_curve_training.py`

**`CURVE_TYPES`** (line 60):
```python
CURVE_TYPES = ['ori_curve', 'ori_curve_avg', 'ori_curve_wavelet_sym8',
               'ori_curve_wavelet_bior35', 'ori_curve_sg_p4']
```

**`--curve_type_id` choices** (~line 411) — update docstring and choices:
```python
parser.add_argument('--curve_type_id', type=int, default=0, choices=[0, 1, 2, 3, 4],
    help='0=ori_curve  1=ori_curve_avg  2=ori_curve_wavelet_sym8  '
         '3=ori_curve_wavelet_bior35  4=ori_curve_sg_p4')
```

Also update the module-level docstring at line 8.

---

### 7. `05_outlier_visualization_report.py`

**Default `--curve_type`** (line 158) — extend to 5 types (same list as 03).

---

### 8. `06_model_prediction_report.py`

**Default `--curve_type`** (line 703) — extend to 5 types (same list as 03).

---

### 9. `07_attribution_vis_all.py`

**Default `--curve_type`** (line 2220) — extend to 5 types (same list as 03).

---

### 10. `slurm_jobs/multi_full_pipeline.sh`

Add `--wavelet_bior35 --sg_p4` to **both** 01 calls (nc_subtract=0 and nc_subtract=1):
```bash
# nc_subtract=0
python -u .../01_curve_preprocessing_v6.py ... --wavelet_sym8 --wavelet_bior35 --sg_p4

# nc_subtract=1
python -u .../01_curve_preprocessing_v6.py ... --nc_subtract --wavelet_sym8 --wavelet_bior35 --sg_p4
```
No array-range change needed — the script doesn't enumerate curve types directly; 03/05/06 use their (now-updated) defaults.

---

### 11. `notebooks/result_comparison.ipynb`

In **Cell 2** (the config/constants cell), update three dicts:

```python
CURVE_JLIB = {
    'ori':     'Ori Curves',
    'avg':     'Ori Curves Avg',
    'wavelet': 'Ori Curves Wavelet Sym8',
    'bior35':  'Ori Curves Wavelet Bior35',   # NEW — key = name.replace("_"," ").title()
    'sgp4':    'Ori Curves Sg P4',             # NEW
}
```

```python
CURVE_LABELS = {
    'ori':     'Ori (raw)',
    'avg':     'Moving Avg',
    'wavelet': 'Wavelet sym8',
    'bior35':  'Wavelet bior3.5',              # NEW
    'sgp4':    'SG p=4 (auto-w)',              # NEW
}
```

```python
CURVE_COLORS = {
    'ori':     '#4393c3',
    'avg':     '#e08c00',
    'wavelet': '#1a9850',
    'bior35':  '#d6604d',                      # NEW
    'sgp4':    '#7b2d8b',                      # NEW
}
```

The `CURVE_JLIB` keys (`'Ori Curves Wavelet Bior35'` and `'Ori Curves Sg P4'`) are produced by Python's `name.replace("_", " ").title()` in 03 — verified by tracing `clean_title` at line 442.

The `plot_curve_comparison` function uses `bar_w = 0.22`; with 5 curves and 4 families the bars will be tight. Consider changing to `bar_w = 0.15` and `figsize=(18, 5)` for that plot.

---

## Implementation order

1. `config.py`
2. `01_curve_preprocessing_v6.py`
3. `02_outlier_detection_pipeline.py`
4. `03` / `04` / `04b` / `05` / `06` / `07` defaults
5. `04_cross_dataset_training.py` — also fix iteration bug
6. `slurm_jobs/multi_full_pipeline.sh`
7. `notebooks/result_comparison.ipynb`

---

## Verification

1. **01 smoke test** — run on one dataset:
   ```
   python 01_curve_preprocessing_v6.py --task_id 0 --exp_folder <path> \
       --n_wells 10 --n_a_type v06 --wavelet_sym8 --wavelet_bior35 --sg_p4
   ```
   Confirm joblib has `curves["ori_curves_wavelet_bior35"]`, `curves["ori_curves_sg_p4"]`, and `sg_p4_optimal_w`.

2. **01 patch test** — run again without `--force_rerun`; confirm cache hit with no re-compute.

3. **02 smoke test** — run `02_outlier_detection_pipeline.py`; confirm the two new names appear in the `dataset_name` printout.

4. **03 smoke test** — run without `--curve_type` (uses updated default); confirm 5 curve types are processed and result file is created.

5. **05/06 safety test on old dataset** — run on a dataset where 01 was NOT rerun with new flags; confirm `[SKIP]` messages appear for the two new types and the script completes successfully.

6. **config alias check**:
   ```
   python -c "import config; print(config.CURVE_TYPE_ALIASES)"
   ```
   Should show all 6 keys.
