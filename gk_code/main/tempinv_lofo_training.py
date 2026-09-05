import argparse
import importlib
import os
import types
from pathlib import Path

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
cdt = importlib.import_module("04_cross_dataset_training")

import tensorflow as tf
import model_utils as mu
import model_utils_dann as dann

GROUP_NAME = "final_6_new"
CURVE_TYPE = "ori_curve_sg_p4_norm"
PC_TTP_ANCHOR = "min"
TRAIN_CENTER_FRAC = 0.5
K_NEIGHBORS = 24
ATTN_DIM = 16
EPOCHS = 500
TEMP_COL = "well_2d_temp_npr_mean"


def build_tempinv_model(k_plus_1, input_size_curve, n_classes, mode,
                        attn_dim=ATTN_DIM, temp_weight=1.0):
    tf.keras.backend.clear_session()
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, input_size_curve),
                                        name="neighbor_stack_input")
    per_curve_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size_curve, 1)),
        tf.keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
        tf.keras.layers.Conv1D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(attn_dim, activation="relu"),
    ], name="per_curve_encoder")
    embeddings = tf.keras.layers.TimeDistributed(per_curve_encoder)(stack_input)

    query = mu._QuerySlice()(embeddings)
    scores = mu._AttnScores(attn_dim)([query, embeddings])
    attn_weights = tf.keras.layers.Softmax(axis=-1, name="attn_weights")(scores)

    reconstructed = mu._WeightedRecon()([attn_weights, stack_input])
    reconstructed = tf.keras.layers.Reshape((input_size_curve, 1))(reconstructed)

    z = mu._build_cnn_gru_dual_branches(reconstructed)

    cls_out = tf.keras.layers.Dense(n_classes, activation="softmax", name="cls_out")(z)

    grl = None
    if mode == "adversarial":
        grl = dann.GradientReversalLayer(lambda_init=0.0, name="grl")
        temp_in = grl(z)
    else:
        temp_in = z
    temp_feat = tf.keras.layers.Dense(16, activation="relu", name="temp_feat")(temp_in)
    temp_out = tf.keras.layers.Dense(1, name="temp_out")(temp_feat)

    w = 0.0 if mode == "off" else temp_weight
    model = tf.keras.models.Model(inputs=stack_input, outputs=[cls_out, temp_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={"cls_out": "sparse_categorical_crossentropy", "temp_out": "mse"},
        loss_weights={"cls_out": 1.0, "temp_out": w},
        metrics={"cls_out": "accuracy"},
    )
    model.grl = grl
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--held_out", type=int, required=True,
                    help="index 0-5 into the chip list (one LOFO fold per array task)")
    p.add_argument("--mode", choices=["adversarial", "mtl", "off"], required=True,
                    help="adversarial: GRL -> temp invariance. mtl: cooperative aux "
                         "regression (same shape as the concentration MTL). "
                         "off: zero-weighted head, baseline control.")
    p.add_argument("--variant", choices=["raw", "wellcentered"], default="raw",
                    help="raw: per-pixel temp as-is (also suppresses chip-level offset). "
                         "wellcentered: temp minus its well mean -- purely local thermal "
                         "position, class-correlated component removed by construction.")
    p.add_argument("--temp_weight", type=float, default=1.0)
    p.add_argument("--limit_rows", type=int, default=0,
                    help="smoke-test only: subsample each split to this many rows so a "
                         "full run fits on a contended GPU. 0 = use everything.")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                    help="override for smoke tests; production runs leave this at 500")
    p.add_argument("--dry_run", action="store_true",
                    help="Build the pool, split and model, print shapes, then exit.")
    return p.parse_args()


def main():
    a = parse_args()
    cdt.set_global_determinism(0, strict=True)

    import re
    folder_names = sorted(config.CROSS_DATASET_GROUPS[GROUP_NAME],
                          key=lambda s: int(re.search(r"DDM_0(\d)", s).group(1)))
    held_out = folder_names[a.held_out]
    exp_paths = [Path(config.FINAL_EXP_FOLDER, n) for n in folder_names]

    cache = Path(config.FINAL_EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME / "_cache_pc_ttp"
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = (Path(config.FINAL_EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
               / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}" / "tempinv")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TEMPINV] held-out={held_out}  mode={a.mode}  variant={a.variant}  "
          f"temp_weight={a.temp_weight}")

    combined = cdt.combine_group_pc_aligned(
        exp_paths, GROUP_NAME, curve_type=CURVE_TYPE, held_out_chip=held_out,
        anchor_method=PC_TTP_ANCHOR, anchor_pct=cdt.PC_TTP_ANCHOR_PCT_DEFAULT,
        pc_ttp_cache_dir=cache)
    assert combined is not None

    keep = cdt.is_amplifying_mask(combined["curves"])
    combined["features_df"][cdt.NOAMP_FILTER_NAME] = keep.astype(int)
    print(f"  [*] {cdt.NOAMP_FILTER_NAME}: {int((~keep).sum())}/{len(keep)} flagged")

    args_ns = types.SimpleNamespace(mtl=False, train_center_frac=TRAIN_CENTER_FRAC,
                                    k_neighbors=K_NEIGHBORS, batch_size=a.batch_size,
                                    curve_alignment="pc_ttp", pc_ttp_anchor=PC_TTP_ANCHOR)
    encoder, y_full, _, _, _, _ = cdt._derive_pool_labels(combined, args_ns)

    fdf = combined["features_df"]
    temp_raw = fdf[TEMP_COL].to_numpy(dtype=float)
    wells_all = np.asarray(combined["well_ids"])

    m = keep
    curves = combined["curves"][m]
    coords = combined["coords"][m]
    wells = wells_all[m]
    ds_id = np.asarray(combined["dataset_id"])[m]
    y = y_full[m]
    temp = temp_raw[m]

    if a.variant == "wellcentered":
        for w in np.unique(wells):
            s = wells == w
            temp[s] = temp[s] - np.nanmean(temp[s])
    temp = np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

    test_idx = np.where(ds_id == held_out)[0]
    train_idx = np.where(ds_id != held_out)[0]
    train_idx = mu.crop_train_to_well_centers(train_idx, coords, wells, TRAIN_CENTER_FRAC)
    print(f"  pool={len(y)}  train={len(train_idx)} (center-cropped)  test={len(test_idx)}")

    # Validation split for EarlyStopping. Class-stratified over pixels, matching
    # production exactly (model_utils.py:1218). An earlier well-grouped split was
    # more leak-safe in principle but drew ~4 wells from ~40, which could cover
    # only 1-2 of the 5 classes -- validation accuracy collapsed to ~1%, val loss
    # rose monotonically, and EarlyStopping(restore_best_weights=True) restored a
    # barely-trained epoch-2 model (13.7% vs a 59.1% baseline). Matching production
    # both fixes that and removes the largest remaining harness deviation, which
    # matters because the `off` arm exists to reproduce the published baseline.
    from sklearn.model_selection import train_test_split
    _tr_sub, _val_sub = train_test_split(
        np.arange(len(train_idx)), test_size=0.1,
        stratify=y[train_idx], random_state=0)
    fit_idx, val_idx = train_idx[_tr_sub], train_idx[_val_sub]
    _vc = np.bincount(y[val_idx], minlength=len(np.unique(y)))
    print(f"  fit={len(fit_idx)}  val={len(val_idx)}  val class counts={_vc.tolist()}")
    assert (_vc > 0).all(), f"validation split is missing classes: {_vc.tolist()}"

    mu_t, sd_t = temp[fit_idx].mean(), temp[fit_idx].std() + 1e-9
    print(f"  temp target: mean={mu_t:.3f} sd={sd_t:.3f} "
          f"({len(np.unique(temp[fit_idx]))} distinct values in fit set)")

    if a.limit_rows:
        _r = np.random.default_rng(0)
        def _sub(ix):
            return np.sort(_r.choice(ix, size=min(a.limit_rows, len(ix)), replace=False))
        fit_idx, val_idx, test_idx = _sub(fit_idx), _sub(val_idx), _sub(test_idx)
        print(f"  [SMOKE] subsampled -> fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

    def stack(idx):
        return mu.build_neighbor_curve_stack(
            curves[idx].astype(np.float32, copy=False),
            coords[idx], wells[idx], k=K_NEIGHBORS)

    n_classes = len(encoder.classes_)
    if a.dry_run:
        model = build_tempinv_model(K_NEIGHBORS + 1, curves.shape[1], n_classes,
                                    a.mode, temp_weight=a.temp_weight)
        print(f"\n  model params: {model.count_params():,}")
        print(f"  outputs: {[o.name for o in model.outputs]}")
        print("\n[DRY RUN] pool/split/model built -- exiting before training.")
        return

    X_fit, X_val, X_test = stack(fit_idx), stack(val_idx), stack(test_idx)
    print(f"  stacks: fit={X_fit.shape} val={X_val.shape} test={X_test.shape}")

    model = build_tempinv_model(K_NEIGHBORS + 1, curves.shape[1], n_classes,
                                a.mode, temp_weight=a.temp_weight)
    steps = max(1, int(np.ceil(len(fit_idx) / a.batch_size))) * a.epochs
    cbs = [
        # mode="min" is required: Keras 3 can't infer direction for a composed
        # multi-output metric name like val_cls_out_loss (it can for plain "loss").
        tf.keras.callbacks.EarlyStopping(monitor="val_cls_out_loss", mode="min",
                                         patience=100, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_cls_out_loss", mode="min",
                                             patience=30, factor=0.5, min_lr=1e-5),
    ]
    if model.grl is not None:
        cbs.insert(0, dann.GRLLambdaSchedule(model.grl, total_steps=steps, lambda_max=1.0))
    model.fit(X_fit,
              {"cls_out": y[fit_idx], "temp_out": (temp[fit_idx] - mu_t) / sd_t},
              validation_data=(X_val, {"cls_out": y[val_idx],
                                        "temp_out": (temp[val_idx] - mu_t) / sd_t}),
              epochs=a.epochs, batch_size=a.batch_size, shuffle=True, verbose=2,
              callbacks=cbs)

    out = model.predict(X_test, batch_size=a.batch_size, verbose=0)
    prob, temp_pred = out[0], out[1].ravel()
    pred = np.argmax(prob, axis=1)
    acc = float((pred == y[test_idx]).mean() * 100)

    t_true = (temp[test_idx] - mu_t) / sd_t
    temp_mse = float(np.mean((temp_pred - t_true) ** 2))
    ss = float(np.sum((t_true - t_true.mean()) ** 2))
    temp_r2 = float(1.0 - np.sum((temp_pred - t_true) ** 2) / ss) if ss > 0 else float("nan")
    print(f"  temp head on held-out chip: mse={temp_mse:.4f}  r2={temp_r2:+.4f}")
    print(f"\n[RESULT] {held_out}  mode={a.mode}  variant={a.variant}  LOFO acc = {acc:.2f}%")

    import joblib
    tag = f"tempinv_{a.mode}_{a.variant}_w{a.temp_weight:g}"
    from safe_io import safe_keras_save
    model_path = out_dir / f"{tag}_{CURVE_TYPE}_lofo_{held_out}_model.keras"
    safe_keras_save(model, model_path)
    print(f"  [MODEL] saved -> {model_path}")

    joblib.dump({"held_out": held_out, "mode": a.mode, "variant": a.variant,
                 "temp_weight": a.temp_weight, "acc": acc,
                 "temp_mse": temp_mse, "temp_r2": temp_r2,
                 "y_true": y[test_idx], "y_pred": pred,
                 "well_ids_test": wells[test_idx],
                 "class_names": [str(c) for c in encoder.classes_]},
                out_dir / f"{tag}_{CURVE_TYPE}_lofo_{held_out}.joblib", compress=3)
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
