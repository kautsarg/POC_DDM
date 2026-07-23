"""Source separation Phase 1 + Phase 2 pretraining with serial-bigger encoder.

Two variants selected via --variant:
  active  — L_active only  (lambda_active=0.5, lambda_balance=0.0)
  balance — L_balance only (lambda_active=0.0, lambda_balance=0.5)

No preconditioning, no SupCon, no CRF, no Phase 3.

Usage:
  python 03c_serial_source_sep_pretraining.py --variant active  --task_id 0 --exp_folder /path
  python 03c_serial_source_sep_pretraining.py --variant balance --task_id 0 --exp_folder /path --train_phase2 --n_splits 5
  python 03c_serial_source_sep_pretraining.py --variant active  --avg_consist --task_id 0 --exp_folder /path --train_phase2
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
    build_serial_bigger_encoder,
    build_parametric_decoder,
    MultiLabelSourceSepPhase1Model,
)
from model_utils_multilabel import (
    ML_MODEL_KEY_MAP,
    ML_MODEL_PRINT_MAP,
    encode_multilabel_for_training,
    evaluate_outlier_filters_ml,
)

_VARIANT_DEFAULTS = {
    'active':  dict(lambda_active=0.5, lambda_balance=0.0),
    'balance': dict(lambda_active=0.0, lambda_balance=0.5),
}


def load_training_data(data_path):
    if not os.path.exists(data_path):
        print(f"  -> [SKIP] {data_path} not found. Run 01b first.")
        sys.exit(0)
    return joblib.load(data_path)


def _run_validation(encoder, d_shared, d_target, X, y_binary, sigmoid_params, all_targets):
    """Spearman r between z_j[0] and Ct for single-target wells."""
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
        from scipy.stats import spearmanr
        r, p  = spearmanr(z_j0, ct_j)
        status = "OK" if abs(r) >= 0.65 else "WARN <0.65"
        print(f"    {tgt}: r={r:.3f}  p={p:.3e}  n={mask_j.sum()}  [{status}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Separation 03c — Serial-Bigger Encoder")
    parser.add_argument("--variant",       type=str,   required=True, choices=['active', 'balance'],
                        help="active=L_active only; balance=L_balance only.")
    parser.add_argument("--task_id",       type=int,   default=0)
    parser.add_argument("--exp_folder",    type=str,   default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--d_shared",      type=int,   default=16)
    parser.add_argument("--d_target",      type=int,   default=10)
    parser.add_argument("--epochs_p1",     type=int,   default=400)
    parser.add_argument("--lambda_cons",   type=float, default=2.0)
    parser.add_argument("--lambda_anch",   type=float, default=0.3)
    parser.add_argument("--lambda_active", type=float, default=None,
                        help="Override variant default for lambda_active.")
    parser.add_argument("--lambda_balance", type=float, default=None,
                        help="Override variant default for lambda_balance.")
    parser.add_argument("--Fm_floor_frac", type=float, default=0.65)
    parser.add_argument("--nn_k",          type=int,   default=5)
    parser.add_argument("--lambda_var",    type=float, default=0.05)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--fast_mode",     action="store_true")
    parser.add_argument("--validate",      action="store_true")
    parser.add_argument("--force_rerun",   action="store_true")
    parser.add_argument("--avg_consist",   action="store_true",
                        help="L_consist uses avg of active decoders (not sum). "
                             "Appropriate when multi-target signal = mean of single-target signals.")
    parser.add_argument("--train_phase2",  action="store_true",
                        help="Run Phase 2 (frozen encoder classification head) after Phase 1.")
    parser.add_argument("--n_splits",      type=int, default=5,
                        help="CV folds for Phase 2 (1=ShuffleSplit, >1=StratifiedKFold).")
    parser.add_argument("--force_rerun_phase2", action="store_true",
                        help="Clear and retrain Phase 2 results for this variant.")
    args = parser.parse_args()

    # Apply variant defaults for any lambda not explicitly overridden
    vd = _VARIANT_DEFAULTS[args.variant]
    if args.lambda_active  is None: args.lambda_active  = vd['lambda_active']
    if args.lambda_balance is None: args.lambda_balance = vd['lambda_balance']

    set_global_determinism(0, strict=not args.fast_mode)

    subdirs   = sorted([d for d in Path(args.exp_folder).iterdir() if d.is_dir()])
    task_dirs = [d for d in subdirs if d.name in config.FILE_MAPPING]
    if not task_dirs:
        print(f"No valid task directories found in {args.exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]
    folder   = exp_path.name

    # variant_key includes _avg suffix when avg_consist is on, keeping filenames unique
    variant_key = f'{args.variant}_avg' if args.avg_consist else args.variant
    out_path    = os.path.join(exp_path, f'source_sep_serial_{variant_key}_encoder_weights.weights.h5')

    print(f"\n{'='*70}")
    print(f"[SS SERIAL]  {folder}  variant={args.variant}  avg_consist={args.avg_consist}  "
          f"λ_active={args.lambda_active}  λ_balance={args.lambda_balance}")
    print(f"{'='*70}")

    run_phase1 = not os.path.exists(out_path) or args.force_rerun

    _data = None  # loaded lazily; shared between Phase 1 and Phase 2

    if not run_phase1:
        print(f"  -> Encoder weights exist: {out_path}. Skipping Phase 1.")
        if args.validate or args.train_phase2:
            data_path   = os.path.join(exp_path, config.TRAINING_DATA_PATH)
            _data       = load_training_data(data_path)
            X_curves    = _data['curves']['ori_curves'].astype(np.float32)
            y_binary    = _data['label_binarized'].astype(np.int32)
            sig_params  = _data['sigmoid_curves']['original']['params'].astype(np.float32)
            all_targets = _data['all_targets']
            n_targets   = len(all_targets)
            T           = X_curves.shape[1]
            if args.validate:
                enc = build_serial_bigger_encoder(T, args.d_shared, args.d_target, n_targets)
                enc.load_weights(out_path)  # out_path already uses variant_key
                _run_validation(enc, args.d_shared, args.d_target,
                                X_curves, y_binary, sig_params, all_targets)
        if not args.train_phase2:
            sys.exit(0)

    if run_phase1:
        data_path   = os.path.join(exp_path, config.TRAINING_DATA_PATH)
        _data       = load_training_data(data_path)
        X_curves    = _data['curves']['ori_curves'].astype(np.float32)
        y_binary    = _data['label_binarized'].astype(np.int32)
        sig_params  = _data['sigmoid_curves']['original']['params'].astype(np.float32)
        all_targets = _data['all_targets']
        n_targets   = len(all_targets)
        T           = X_curves.shape[1]

        print(f"  N={len(X_curves)}, T={T}, n_targets={n_targets}, targets={all_targets}")
        print(f"  d_shared={args.d_shared}, d_target={args.d_target}  "
              f"→ latent={args.d_shared + n_targets * args.d_target}")

        nn_bank = build_nn_anchor_bank(X_curves, y_binary, n_targets=n_targets, max_per_target=500)
        for j, (tgt, bank) in enumerate(zip(all_targets, nn_bank)):
            print(f"  Anchor bank [{tgt}]: {bank.shape[0]} single-target curves")

        tf.keras.backend.clear_session()
        encoder  = build_serial_bigger_encoder(T, args.d_shared, args.d_target, n_targets)
        decoders = [build_parametric_decoder(args.d_shared, args.d_target, T_max=T, j=j)
                    for j in range(n_targets)]

        model = MultiLabelSourceSepPhase1Model(
            encoder        = encoder,
            decoders       = decoders,
            nn_bank        = nn_bank,
            T              = T,
            d_shared       = args.d_shared,
            d_target       = args.d_target,
            n_targets      = n_targets,
            lambda_cons    = args.lambda_cons,
            lambda_anch    = args.lambda_anch,
            lambda_active  = args.lambda_active,
            lambda_balance = args.lambda_balance,
            Fm_floor_frac  = args.Fm_floor_frac,
            avg_consist    = args.avg_consist,
            lambda_var     = args.lambda_var,
            k              = args.nn_k,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0))

        X_in = np.expand_dims(X_curves, axis=-1)

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

        print(f"\n  Training Phase 1 [serial-bigger / {args.variant}] ({args.epochs_p1} epochs max)...")

        model.fit(
            X_in[tr_idx], {'cls_out': y_binary[tr_idx]},
            validation_data=(X_in[val_idx], {'cls_out': y_binary[val_idx]}),
            epochs=args.epochs_p1, batch_size=args.batch_size,
            shuffle=True, verbose=1, callbacks=callbacks,
        )

        encoder.save_weights(out_path)
        print(f"\n  -> Encoder weights saved: {out_path}")
        for j, dec in enumerate(decoders):
            dec_path = os.path.join(exp_path, f'source_sep_serial_{variant_key}_decoder_{j}_weights.weights.h5')
            dec.save_weights(dec_path)
            print(f"  -> Decoder {j} ({all_targets[j]}) weights saved: {dec_path}")

        if args.validate:
            _run_validation(encoder, args.d_shared, args.d_target,
                            X_curves, y_binary, sig_params, all_targets)

        tf.keras.backend.clear_session()
        gc.collect()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.train_phase2:
        if _data is None:
            data_path   = os.path.join(exp_path, config.TRAINING_DATA_PATH)
            _data       = load_training_data(data_path)
            X_curves    = _data['curves']['ori_curves'].astype(np.float32)
            y_binary    = _data['label_binarized'].astype(np.int32)
            all_targets = _data['all_targets']
            n_targets   = len(all_targets)

        X_in = np.expand_dims(X_curves, axis=-1)
        y_binary_ml, y_combo_int, _, _ = encode_multilabel_for_training(
            _data['label_lists'], all_targets)
        features_df = pd.DataFrame(index=range(len(X_in)))

        model_key    = f'serial_source_sep_{variant_key}_p2'
        result_flag  = 'source_sep_serial'
        _fmap        = (config.RESULT_10FOLD_FILE_BY_FLAG if args.n_splits > 1
                        else config.RESULT_FILE_BY_FLAG)
        results_path = os.path.join(exp_path, _fmap[result_flag])

        cached = {}
        if os.path.exists(results_path):
            try:
                cached = joblib.load(results_path)
            except Exception:
                pass

        if args.force_rerun_phase2:
            preds_key = ML_MODEL_KEY_MAP.get(model_key, (None,))[0]
            for f_res in cached.values():
                if isinstance(f_res, dict):
                    f_res.pop(preds_key, None)

        def checkpoint_fn(current_results):
            cached.update(current_results)
            safe_joblib_dump(cached, results_path, compress=3)

        mode_label = f'SrcSep Serial {args.variant} Phase2'
        print(f"\n  Running Phase 2 [{mode_label}]: {model_key}")

        evaluate_outlier_filters_ml(
            X_curves             = X_in,
            features_df          = features_df,
            y_binary             = y_binary_ml,
            y_combo_int          = y_combo_int,
            all_targets          = all_targets,
            outlier_filters      = [None],
            dataset_name         = folder,
            mode_name            = mode_label,
            ml_model_key_map     = ML_MODEL_KEY_MAP,
            ml_model_print_map   = ML_MODEL_PRINT_MAP,
            cached_results       = cached,
            models               = [model_key],
            n_splits             = args.n_splits,
            checkpoint_fn        = checkpoint_fn,
            encoder_weights_path = out_path,
            exp_path             = str(exp_path),
        )

        safe_joblib_dump(cached, results_path, compress=3)
        print(f"\n  -> Phase 2 results saved: {results_path}")
        gc.collect()
