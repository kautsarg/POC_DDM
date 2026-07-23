"""Source separation pretraining (Phase 1) and Phase 2+3 training for multiplex PCR.

Phase 1 — Encoder + n_targets parametric decoders trained with reconstruction losses.
  Family A (--precon absent): standard encoder, lambda_supcon=0.
  Family B (--precon): SC-specific encoder with Jaccard SupCon projection heads.

Phase 2+3 (--train_phase23) — frozen encoder head training then end-to-end fine-tune.
  Run after Phase 1; uses the appropriate pretrained encoder weights.

Usage:
  # Family A Phase 1 + Phase 2+3 SC0:
  python 03b_source_sep_pretraining.py --train_phase23 --supcon 0 --n_splits 5

  # Family B Phase 1 SC1 + Phase 2+3:
  python 03b_source_sep_pretraining.py --precon --supcon 1 --train_phase23 --n_splits 5
"""
import os
import sys
import gc
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib

_THIS          = Path(__file__).resolve()
_MULTIPLEX_DIR = _THIS.parent
_MAIN_DIR      = _MULTIPLEX_DIR.parent

sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(_MULTIPLEX_DIR))
sys.path.insert(0, str(_MULTIPLEX_DIR / "utils" / "model_training"))

import tensorflow as tf
import config_multiplex as config
from safe_io import safe_joblib_dump
from model_utils import set_global_determinism
from nn_anchor_bank import build_nn_anchor_bank
from model_utils_source_sep import (
    build_source_sep_encoder,
    build_parametric_decoder,
    MultiLabelSourceSepPhase1Model,
    MultiLabelSourceSepPhase1SC1Model,
    MultiLabelSourceSepPhase1SC2Model,
    MultiLabelSourceSepPhase1SC3Model,
)
from model_utils_multilabel import (
    ML_MODEL_KEY_MAP,
    ML_MODEL_PRINT_MAP,
    encode_multilabel_for_training,
    evaluate_outlier_filters_ml,
)

ENCODER_WEIGHTS_FILE = 'source_sep_encoder_weights.weights.h5'

_PHASE1_MODEL_MAP = {
    0: MultiLabelSourceSepPhase1Model,
    1: MultiLabelSourceSepPhase1SC1Model,
    2: MultiLabelSourceSepPhase1SC2Model,
    3: MultiLabelSourceSepPhase1SC3Model,
}


def load_training_data(data_path):
    if not os.path.exists(data_path):
        print(f"  -> [SKIP] {data_path} not found. Run 01b first.")
        sys.exit(0)
    return joblib.load(data_path)


def _get_phase23_keys(supcon_level, precon):
    assert not (precon and supcon_level == 0), "Family B SC0 does not exist (SC0 = standard encoder = Family A)"
    base    = 'cnn_gru_source_sep' + ('_precon' if precon else '')
    sc_sfx  = ['', '_supcon', '_supcon2', '_supcon3'][supcon_level]
    crf     = f'{base}_crf'
    keys    = [f'{base}{sc_sfx}_p3', f'{crf}{sc_sfx}_p3']
    # _p2 variants: Family A SC0, and all Family B (encoder already structured)
    if supcon_level == 0 or precon:
        keys += [f'{base}{sc_sfx}_p2', f'{crf}{sc_sfx}_p2']
    return keys


def _run_validation(encoder, d_shared, d_target, n_targets, X, y_binary, sigmoid_params, all_targets):
    """Validation gate 2: Spearman r between z_j[0] and Ct for single-target wells."""
    from scipy.stats import spearmanr

    X_enc  = np.expand_dims(X, axis=-1).astype(np.float32)
    Z      = encoder.predict(X_enc, verbose=0, batch_size=512)
    Y      = np.asarray(y_binary, dtype=np.int32)
    single = Y.sum(axis=1) == 1

    print("\n  [Validation gate 2] Spearman r(z_j[0], Ct) for single-target wells")
    for j, tgt in enumerate(all_targets):
        mask_j = single & (Y[:, j] == 1)
        if mask_j.sum() < 20:
            print(f"    {tgt}: too few samples ({mask_j.sum()})")
            continue
        start = d_shared + j * d_target
        z_j0  = Z[mask_j, start]
        ct_j  = sigmoid_params[mask_j, 3]
        r, p  = spearmanr(z_j0, ct_j)
        status = "OK" if abs(r) >= 0.65 else "WARN <0.65"
        print(f"    {tgt}: r={r:.3f}  p={p:.3e}  n={mask_j.sum()}  [{status}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Separation Phase 1 + Phase 2+3 Training")
    parser.add_argument("--task_id",      type=int,   default=0)
    parser.add_argument("--exp_folder",   type=str,   default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--d_shared",     type=int,   default=16)
    parser.add_argument("--d_target",     type=int,   default=10)
    parser.add_argument("--epochs_p1",    type=int,   default=400)
    parser.add_argument("--lambda_cons",  type=float, default=2.0)
    parser.add_argument("--lambda_anch",   type=float, default=0.3)
    parser.add_argument("--lambda_active", type=float, default=0.5,
                        help="Weight for L_active (min-amplitude floor on active decoders).")
    parser.add_argument("--Fm_floor_frac", type=float, default=0.65,
                        help="Fraction of mean single-target peak used as per-target amplitude floor.")
    parser.add_argument("--nn_k",          type=int,   default=5)
    parser.add_argument("--lambda_var",    type=float, default=0.05)
    parser.add_argument("--lambda_supcon", type=float, default=0.0,
                        help="Legacy per-target SupCon weight for SC0 Phase 1 (ignored for SC1-3).")
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--fast_mode",    action="store_true")
    parser.add_argument("--validate",     action="store_true",
                        help="Run Spearman r validation gate after Phase 1.")
    parser.add_argument("--force_rerun",  action="store_true",
                        help="Retrain Phase 1 even if encoder weights exist.")
    # New args for Family B and Phase 2+3
    parser.add_argument("--precon",       action="store_true",
                        help="Family B: use SC-specific projection heads in Phase 1 "
                             "and save to source_sep_precon_sc{N}_encoder_weights.weights.h5.")
    parser.add_argument("--supcon",       type=int, choices=[0, 1, 2, 3], default=0,
                        help="SC level: 0=no SupCon (Family A only), 1-3=SC1/2/3.")
    parser.add_argument("--train_phase23", action="store_true",
                        help="Run Phase 2+3 (frozen encoder head training + fine-tune) after Phase 1.")
    parser.add_argument("--n_splits",     type=int, default=5,
                        help="CV folds for Phase 2+3 (1=ShuffleSplit, >1=StratifiedKFold).")
    parser.add_argument("--force_rerun_phase23", action="store_true",
                        help="Clear and retrain Phase 2+3 results for this SC level.")
    args = parser.parse_args()

    if args.precon:
        assert args.supcon in [1, 2, 3], "--precon requires --supcon in [1,2,3]"

    set_global_determinism(0, strict=not args.fast_mode)

    subdirs   = sorted([d for d in Path(args.exp_folder).iterdir() if d.is_dir()])
    task_dirs = [d for d in subdirs if d.name in config.FILE_MAPPING]
    if not task_dirs:
        print(f"No valid task directories found in {args.exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]
    folder   = exp_path.name

    if args.precon:
        out_path  = os.path.join(exp_path, f'source_sep_precon_sc{args.supcon}_encoder_weights.weights.h5')
        P1ModelCls = _PHASE1_MODEL_MAP[args.supcon]
    else:
        out_path   = os.path.join(exp_path, ENCODER_WEIGHTS_FILE)
        P1ModelCls = MultiLabelSourceSepPhase1Model

    print(f"\n{'='*70}\n[SOURCE SEP]  {folder}  SC{args.supcon}  {'Precon' if args.precon else 'Standard'}\n{'='*70}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    _data       = None   # raw data dict, loaded lazily
    _X_curves   = None   # (N, T) float32
    _y_binary   = None   # (N, n_targets) int32
    _all_targets = None
    _n_targets  = None
    _T          = None
    _sigmoid_params = None

    run_phase1 = not os.path.exists(out_path) or args.force_rerun

    if not run_phase1:
        print(f"  -> Encoder weights exist: {out_path}. Skipping Phase 1.")
        if args.validate or args.train_phase23:
            data_path     = os.path.join(exp_path, config.TRAINING_DATA_PATH)
            _data         = load_training_data(data_path)
            _X_curves     = _data['curves']['ori_curves'].astype(np.float32)
            _y_binary     = _data['label_binarized'].astype(np.int32)
            _sigmoid_params = _data['sigmoid_curves']['original']['params'].astype(np.float32)
            _all_targets  = _data['all_targets']
            _n_targets    = len(_all_targets)
            _T            = _X_curves.shape[1]
            if args.validate:
                _enc_model = build_source_sep_encoder(_T, args.d_shared, args.d_target, _n_targets)
                _enc_model.load_weights(out_path)
                _run_validation(_enc_model, args.d_shared, args.d_target, _n_targets,
                                _X_curves, _y_binary, _sigmoid_params, _all_targets)
        if not args.train_phase23:
            sys.exit(0)
    else:
        data_path     = os.path.join(exp_path, config.TRAINING_DATA_PATH)
        _data         = load_training_data(data_path)
        _X_curves     = _data['curves']['ori_curves'].astype(np.float32)
        _y_binary     = _data['label_binarized'].astype(np.int32)
        _sigmoid_params = _data['sigmoid_curves']['original']['params'].astype(np.float32)
        _all_targets  = _data['all_targets']
        _n_targets    = len(_all_targets)
        _T            = _X_curves.shape[1]

        print(f"  N={len(_X_curves)}, T={_T}, n_targets={_n_targets}, targets={_all_targets}")
        print(f"  d_shared={args.d_shared}, d_target={args.d_target}  "
              f"→ latent={args.d_shared + _n_targets * args.d_target}")

        nn_bank = build_nn_anchor_bank(_X_curves, _y_binary, n_targets=_n_targets, max_per_target=500)
        for j, (tgt, bank) in enumerate(zip(_all_targets, nn_bank)):
            print(f"  Anchor bank [{tgt}]: {bank.shape[0]} single-target curves")

        tf.keras.backend.clear_session()
        encoder  = build_source_sep_encoder(_T, args.d_shared, args.d_target, _n_targets)
        decoders = [build_parametric_decoder(args.d_shared, args.d_target, T_max=_T, j=j)
                    for j in range(_n_targets)]

        model = P1ModelCls(
            encoder        = encoder,
            decoders       = decoders,
            nn_bank        = nn_bank,
            T              = _T,
            d_shared       = args.d_shared,
            d_target       = args.d_target,
            n_targets      = _n_targets,
            lambda_cons    = args.lambda_cons,
            lambda_anch    = args.lambda_anch,
            lambda_active  = args.lambda_active,
            Fm_floor_frac  = args.Fm_floor_frac,
            lambda_var     = args.lambda_var,
            lambda_supcon  = args.lambda_supcon,
            k              = args.nn_k,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0))

        X_in = np.expand_dims(_X_curves, axis=-1)

        val_split = 0.1
        n_val     = max(1, int(len(X_in) * val_split))
        rng       = np.random.default_rng(0)
        idx       = rng.permutation(len(X_in))
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_l_consist', mode='min', patience=30, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_l_consist', mode='min', factor=0.3, patience=15, min_lr=1e-5),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        print(f"\n  Training Phase 1 [{P1ModelCls.__name__}] ({args.epochs_p1} epochs max)...")
        model.fit(
            X_in[tr_idx], {'cls_out': _y_binary[tr_idx]},
            validation_data=(X_in[val_idx], {'cls_out': _y_binary[val_idx]}),
            epochs=args.epochs_p1, batch_size=args.batch_size,
            shuffle=True, verbose=1, callbacks=callbacks,
        )

        encoder.save_weights(out_path)
        print(f"\n  -> Encoder weights saved: {out_path}")
        for j, dec in enumerate(decoders):
            dec_path = os.path.join(exp_path, f'source_sep_decoder_{j}_weights.weights.h5')
            dec.save_weights(dec_path)
            print(f"  -> Decoder {j} ({_all_targets[j]}) weights saved: {dec_path}")

        if args.validate:
            _run_validation(encoder, args.d_shared, args.d_target, _n_targets,
                            _X_curves, _y_binary, _sigmoid_params, _all_targets)

        tf.keras.backend.clear_session()
        gc.collect()

    # ── Phase 2+3 ─────────────────────────────────────────────────────────────
    if args.train_phase23:
        # Load data if not already loaded
        if _data is None:
            data_path   = os.path.join(exp_path, config.TRAINING_DATA_PATH)
            _data       = load_training_data(data_path)
            _X_curves   = _data['curves']['ori_curves'].astype(np.float32)
            _y_binary   = _data['label_binarized'].astype(np.int32)
            _all_targets = _data['all_targets']
            _n_targets  = len(_all_targets)
            _T          = _X_curves.shape[1]

        X_in = np.expand_dims(_X_curves, axis=-1)   # (N, T, 1)
        y_binary_ml, y_combo_int, _, _ = encode_multilabel_for_training(
            _data['label_lists'], _all_targets)
        features_df = pd.DataFrame(index=range(len(X_in)))

        keys        = _get_phase23_keys(args.supcon, args.precon)
        result_flag = 'source_sep_precon' if args.precon else 'source_sep'
        _fmap       = (config.RESULT_10FOLD_FILE_BY_FLAG if args.n_splits > 1
                       else config.RESULT_FILE_BY_FLAG)
        results_path = os.path.join(exp_path, _fmap[result_flag])

        cached = {}
        if os.path.exists(results_path):
            try:
                cached = joblib.load(results_path)
            except Exception:
                pass

        if args.force_rerun_phase23:
            from model_utils_multilabel import ML_MODEL_KEY_MAP as _KM
            for f_res in cached.values():
                if not isinstance(f_res, dict):
                    continue
                for k in list(f_res):
                    if any(_KM.get(m, (None,))[0] == k for m in keys):
                        del f_res[k]

        def checkpoint_fn(current_results):
            cached.update(current_results)
            safe_joblib_dump(cached, results_path, compress=3)

        mode_label = f'SrcSep {"B" if args.precon else "A"} SC{args.supcon} Phase2+3'
        print(f"\n  Running Phase 2+3 [{mode_label}]: {keys}")

        evaluate_outlier_filters_ml(
            X_curves              = X_in,
            features_df           = features_df,
            y_binary              = y_binary_ml,
            y_combo_int           = y_combo_int,
            all_targets           = _all_targets,
            outlier_filters       = [None],
            dataset_name          = folder,
            mode_name             = mode_label,
            ml_model_key_map      = ML_MODEL_KEY_MAP,
            ml_model_print_map    = ML_MODEL_PRINT_MAP,
            cached_results        = cached,
            models                = keys,
            n_splits              = args.n_splits,
            checkpoint_fn         = checkpoint_fn,
            encoder_weights_path  = out_path if not args.precon else None,
            exp_path              = str(exp_path),
        )

        safe_joblib_dump(cached, results_path, compress=3)
        print(f"\n  -> Phase 2+3 results saved: {results_path}")
        gc.collect()
