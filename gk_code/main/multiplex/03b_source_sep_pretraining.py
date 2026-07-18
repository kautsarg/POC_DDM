"""Phase 1 source separation pretraining for multiplex PCR.

Trains an Encoder + n_targets parametric decoders on all curves using:
  L = L_absent  +  λ_cons * L_consist  +  λ_anch * Σ L_anchor_j

Saves encoder weights to {exp_path}/source_sep_encoder_weights.weights.h5.
Run this before 03_main_training.py --source_sep (Phase 2/3).

All losses are in rendered curve space. L_anchor uses NN anchoring —
no concentration information is required at training time.
"""
import os
import sys
import gc
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
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
from model_utils import set_global_determinism
from nn_anchor_bank import build_nn_anchor_bank
from model_utils_source_sep import (
    build_source_sep_encoder,
    build_parametric_decoder,
    MultiLabelSourceSepPhase1Model,
)

ENCODER_WEIGHTS_FILE = 'source_sep_encoder_weights.weights.h5'


def load_training_data(data_path):
    if not os.path.exists(data_path):
        print(f"  -> [SKIP] {data_path} not found. Run 01b first.")
        sys.exit(0)
    return joblib.load(data_path)


def _run_validation(encoder, d_shared, d_target, n_targets, X, y_binary, sigmoid_params, all_targets):
    """Validation gate 2: Spearman r between z_j[0] and Ct for single-target wells.
    Target: |r| > 0.65.  Ct is index 3 in params [Fm, Fb, Sc, Cs, As].
    """
    from scipy.stats import spearmanr

    X_enc  = np.expand_dims(X, axis=-1).astype(np.float32)
    Z      = encoder.predict(X_enc, verbose=0, batch_size=512)  # (N, d_shared + n*d_target)
    Y      = np.asarray(y_binary, dtype=np.int32)
    single = Y.sum(axis=1) == 1

    print("\n  [Validation gate 2] Spearman r(z_j[0], Ct) for single-target wells")
    for j, tgt in enumerate(all_targets):
        mask_j = single & (Y[:, j] == 1)
        if mask_j.sum() < 20:
            print(f"    {tgt}: too few samples ({mask_j.sum()})")
            continue
        start  = d_shared + j * d_target
        z_j0   = Z[mask_j, start]                  # first dim of z_j
        ct_j   = sigmoid_params[mask_j, 3]          # Cs = index 3
        r, p   = spearmanr(z_j0, ct_j)
        status = "✓" if abs(r) >= 0.65 else "✗ <0.65"
        print(f"    {tgt}: r={r:.3f}  p={p:.3e}  n={mask_j.sum()}  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Separation Phase 1 Pretraining")
    parser.add_argument("--task_id",      type=int,   default=0)
    parser.add_argument("--exp_folder",   type=str,   default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--d_shared",     type=int,   default=16,
                        help="Shared latent dim (default 16)")
    parser.add_argument("--d_target",     type=int,   default=10,
                        help="Per-target latent dim (default 10)")
    parser.add_argument("--epochs_p1",    type=int,   default=200)
    parser.add_argument("--lambda_cons",  type=float, default=1.0)
    parser.add_argument("--lambda_anch",  type=float, default=0.5)
    parser.add_argument("--nn_k",         type=int,   default=5,
                        help="k nearest neighbours for L_anchor (default 5)")
    parser.add_argument("--lambda_var",   type=float, default=0.1,
                        help="Weight for variance regularisation loss L_var (default 0.1)")
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--fast_mode",    action="store_true")
    parser.add_argument("--validate",     action="store_true",
                        help="After training, compute Spearman r(z_j[0], Ct) "
                             "for each target (validation gate 2)")
    parser.add_argument("--force_rerun",  action="store_true",
                        help="Retrain even if encoder weights already exist")
    args = parser.parse_args()

    set_global_determinism(0, strict=not args.fast_mode)

    subdirs   = sorted([d for d in Path(args.exp_folder).iterdir() if d.is_dir()])
    task_dirs = [d for d in subdirs if d.name in config.FILE_MAPPING]
    if not task_dirs:
        print(f"No valid task directories found in {args.exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]
    folder   = exp_path.name
    out_path = os.path.join(exp_path, ENCODER_WEIGHTS_FILE)
    print(f"\n{'='*70}\n[SOURCE SEP PHASE 1]  {folder}\n{'='*70}")

    if os.path.exists(out_path) and not args.force_rerun:
        print(f"  -> Encoder weights already exist: {out_path}")
        print(f"     Use --force_rerun to retrain.")
        if args.validate:
            data_path     = os.path.join(exp_path, config.TRAINING_DATA_PATH)
            data          = load_training_data(data_path)
            X_curves      = data['curves']['ori_curves'].astype(np.float32)
            y_binary      = data['label_binarized']
            sigmoid_params = data['sigmoid_curves']['original']['params'].astype(np.float32)
            all_targets   = data['all_targets']
            n_targets     = len(all_targets)
            T             = X_curves.shape[1]
            encoder       = build_source_sep_encoder(T, args.d_shared, args.d_target, n_targets)
            encoder.load_weights(out_path)
            _run_validation(encoder, args.d_shared, args.d_target, n_targets,
                            X_curves, y_binary, sigmoid_params, all_targets)
        sys.exit(0)

    # ── Load data ──────────────────────────────────────────────────────
    data_path      = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    data           = load_training_data(data_path)
    X_curves       = data['curves']['ori_curves'].astype(np.float32)     # (N, T)
    y_binary       = data['label_binarized'].astype(np.int32)            # (N, n_targets)
    sigmoid_params = data['sigmoid_curves']['original']['params'].astype(np.float32)  # (N, 5)
    all_targets    = data['all_targets']
    n_targets      = len(all_targets)
    T              = X_curves.shape[1]

    print(f"  N={len(X_curves)}, T={T}, n_targets={n_targets}")
    print(f"  Targets: {all_targets}")
    print(f"  d_shared={args.d_shared}, d_target={args.d_target}  "
          f"→ latent_dim={args.d_shared + n_targets * args.d_target}")

    # ── Build NN anchor bank (no concentration needed) ─────────────────
    nn_bank = build_nn_anchor_bank(X_curves, y_binary, n_targets=n_targets, max_per_target=500)
    for j, (tgt, bank) in enumerate(zip(all_targets, nn_bank)):
        print(f"  Anchor bank [{tgt}]: {bank.shape[0]} single-target curves")

    # ── Build model ────────────────────────────────────────────────────
    tf.keras.backend.clear_session()
    encoder  = build_source_sep_encoder(T, args.d_shared, args.d_target, n_targets)
    decoders = [build_parametric_decoder(args.d_shared, args.d_target, T_max=T, j=j)
                for j in range(n_targets)]

    model = MultiLabelSourceSepPhase1Model(
        encoder      = encoder,
        decoders     = decoders,
        nn_bank      = nn_bank,
        T            = T,
        d_shared     = args.d_shared,
        d_target     = args.d_target,
        n_targets    = n_targets,
        lambda_cons  = args.lambda_cons,
        lambda_anch  = args.lambda_anch,
        lambda_var   = args.lambda_var,
        k            = args.nn_k,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0))

    # ── Prepare dataset (y_dict format matches train_step expectation) ─
    X_in   = np.expand_dims(X_curves, axis=-1)          # (N, T, 1)
    y_dict = {'cls_out': y_binary}

    val_split = 0.1
    n_val     = max(1, int(len(X_in) * val_split))
    rng       = np.random.default_rng(0)
    idx       = rng.permutation(len(X_in))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    X_tr,  X_val  = X_in[tr_idx],          X_in[val_idx]
    y_tr,  y_val  = y_binary[tr_idx],       y_binary[val_idx]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5),
    ]

    print(f"\n  Training Phase 1 ({args.epochs_p1} epochs max, early stopping patience=30)...")
    model.fit(
        X_tr,  {'cls_out': y_tr},
        validation_data=(X_val, {'cls_out': y_val}),
        epochs     = args.epochs_p1,
        batch_size = args.batch_size,
        shuffle    = True,
        verbose    = 1,
        callbacks  = callbacks,
    )

    # ── Save encoder weights + per-decoder weights (for decomposition viz) ──
    encoder.save_weights(out_path)
    print(f"\n  -> Encoder weights saved: {out_path}")
    for j, dec in enumerate(decoders):
        dec_path = os.path.join(exp_path, f'source_sep_decoder_{j}_weights.weights.h5')
        dec.save_weights(dec_path)
        print(f"  -> Decoder {j} ({all_targets[j]}) weights saved: {dec_path}")

    # ── Validation gate 2 (optional) ──────────────────────────────────
    if args.validate:
        _run_validation(encoder, args.d_shared, args.d_target, n_targets,
                        X_curves, y_binary, sigmoid_params, all_targets)

    tf.keras.backend.clear_session()
    gc.collect()
