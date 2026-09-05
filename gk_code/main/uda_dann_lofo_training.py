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


def build_uda_dann_model(k_plus_1, input_size_curve, n_classes, n_domains,
                         attn_dim=ATTN_DIM, domain_weight=1.0):
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

    # No cls_feat Dense -- the plain factory goes straight from the fused
    # embedding to the softmax, and matching it keeps the comparison clean.
    cls_out = tf.keras.layers.Dense(n_classes, activation="softmax", name="cls_out")(z)

    grl = dann.GradientReversalLayer(lambda_init=0.0, name="grl")
    dom_feat = tf.keras.layers.Dense(16, activation="relu", name="dom_feat")(grl(z))
    dom_out = tf.keras.layers.Dense(n_domains, activation="softmax", name="dom_out")(dom_feat)

    model = tf.keras.models.Model(inputs=stack_input, outputs=[cls_out, dom_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss={"cls_out": "sparse_categorical_crossentropy",
              "dom_out": "sparse_categorical_crossentropy"},
        loss_weights={"cls_out": 1.0, "dom_out": domain_weight},
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
    p.add_argument("--domain", choices=["binary", "chip"], default="binary",
                    help="binary: source vs target, the canonical Ganin UDA setup. "
                         "chip: one class per chip WITH the target included (unlike "
                         "04's DANN, which omits it).")
    p.add_argument("--domain_weight", type=float, default=1.0)
    p.add_argument("--target_n", type=int, default=0,
                    help="cap on unlabelled target rows used for the discriminator "
                         "(0 = use all). Target is NOT centre-cropped, so it keeps "
                         "the test-time distribution.")
    p.add_argument("--epochs", type=int, default=EPOCHS,
                    help="override for smoke tests; production runs leave this at 500")
    p.add_argument("--limit_rows", type=int, default=0,
                    help="smoke-test only: subsample each split so a full run fits on "
                         "a contended GPU. 0 = use everything.")
    p.add_argument("--dry_run", action="store_true",
                    help="Build pool, split and model, print shapes, then exit.")
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
               / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}" / "uda_dann")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UDA-DANN] held-out={held_out}  domain={a.domain}  w={a.domain_weight}")
    print("           target chip's CURVES are used (unlabelled); its LABELS are not.")

    # Alignment stats still exclude the target (leak-safe, same as supervised LOFO);
    # the pool itself contains every chip, which is what UDA needs.
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
    encoder, y_full, _, _, chip_id_all, _ = cdt._derive_pool_labels(combined, args_ns)

    m = keep
    curves = combined["curves"][m]
    coords = combined["coords"][m]
    wells = np.asarray(combined["well_ids"])[m]
    ds_id = np.asarray(combined["dataset_id"])[m]
    y = y_full[m]
    chip_id = np.asarray(chip_id_all)[m]

    tgt_idx = np.where(ds_id == held_out)[0]
    src_idx = np.where(ds_id != held_out)[0]
    src_idx = mu.crop_train_to_well_centers(src_idx, coords, wells, TRAIN_CENTER_FRAC)
    print(f"  pool={len(y)}  source={len(src_idx)} (centre-cropped)  target={len(tgt_idx)}")

    # Validation is SOURCE ONLY and class-stratified. Class-stratified because a
    # well-grouped split can draw wells covering only 1-2 of the 5 classes, which
    # collapses val accuracy and makes EarlyStopping restore a barely-trained model.
    from sklearn.model_selection import train_test_split
    _tr, _val = train_test_split(np.arange(len(src_idx)), test_size=0.1,
                                 stratify=y[src_idx], random_state=0)
    fit_idx, val_idx = src_idx[_tr], src_idx[_val]
    _vc = np.bincount(y[val_idx], minlength=len(np.unique(y)))
    print(f"  fit={len(fit_idx)}  val={len(val_idx)}  val class counts={_vc.tolist()}")
    assert (_vc > 0).all(), f"validation split is missing classes: {_vc.tolist()}"

    rng = np.random.default_rng(0)
    if a.target_n and len(tgt_idx) > a.target_n:
        tgt_idx = np.sort(rng.choice(tgt_idx, size=a.target_n, replace=False))
        print(f"  target capped to {len(tgt_idx)} rows for the discriminator")
    if a.limit_rows:
        def _sub(ix):
            return np.sort(rng.choice(ix, size=min(a.limit_rows, len(ix)), replace=False))
        fit_idx, val_idx, tgt_idx = _sub(fit_idx), _sub(val_idx), _sub(tgt_idx)
        print(f"  [SMOKE] fit={len(fit_idx)} val={len(val_idx)} target={len(tgt_idx)}")

    def stack(idx):
        return mu.build_neighbor_curve_stack(
            curves[idx].astype(np.float32, copy=False), coords[idx], wells[idx], k=K_NEIGHBORS)

    n_classes = len(encoder.classes_)
    n_domains = 2 if a.domain == "binary" else len(np.unique(chip_id))

    if a.dry_run:
        model = build_uda_dann_model(K_NEIGHBORS + 1, curves.shape[1], n_classes,
                                     n_domains, domain_weight=a.domain_weight)
        print(f"\n  n_classes={n_classes}  n_domains={n_domains}  "
              f"params={model.count_params():,}")
        print("\n[DRY RUN] pool/split/model built -- exiting before training.")
        return

    X_fit, X_val, X_tgt = stack(fit_idx), stack(val_idx), stack(tgt_idx)
    print(f"  stacks: fit={X_fit.shape} val={X_val.shape} target={X_tgt.shape}")

    # Training set = labelled source + UNLABELLED target.
    X_train = np.concatenate([X_fit, X_tgt], axis=0)
    n_src, n_tgt = len(X_fit), len(X_tgt)

    # Classification targets: real for source, placeholder for target. The target
    # rows are zero-weighted, so their placeholder labels contribute nothing --
    # this is what keeps the run unsupervised.
    y_cls = np.concatenate([y[fit_idx], np.zeros(n_tgt, dtype=y.dtype)])
    w_cls = np.concatenate([np.ones(n_src), np.zeros(n_tgt)])

    if a.domain == "binary":
        y_dom = np.concatenate([np.zeros(n_src, dtype=int), np.ones(n_tgt, dtype=int)])
    else:
        y_dom = np.concatenate([chip_id[fit_idx], chip_id[tgt_idx]]).astype(int)

    # Source outnumbers target ~5:1; inverse-frequency weights stop the
    # discriminator from simply predicting "source".
    dom_classes, dom_counts = np.unique(y_dom, return_counts=True)
    inv = {c: len(y_dom) / (len(dom_classes) * n) for c, n in zip(dom_classes, dom_counts)}
    w_dom = np.array([inv[d] for d in y_dom])

    print(f"  train rows={len(X_train)} (source {n_src} labelled + target {n_tgt} unlabelled)")
    print(f"  domain classes={dom_classes.tolist()} counts={dom_counts.tolist()}")

    model = build_uda_dann_model(K_NEIGHBORS + 1, curves.shape[1], n_classes,
                                 n_domains, domain_weight=a.domain_weight)
    steps = max(1, int(np.ceil(len(X_train) / a.batch_size))) * a.epochs
    cbs = [
        dann.GRLLambdaSchedule(model.grl, total_steps=steps, lambda_max=1.0),
        # mode="min" is required: Keras 3 can't infer direction for a composed
        # multi-output metric name like val_cls_out_loss.
        tf.keras.callbacks.EarlyStopping(monitor="val_cls_out_loss", mode="min",
                                         patience=100, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_cls_out_loss", mode="min",
                                             patience=30, factor=0.5, min_lr=1e-5),
    ]
    # Targets/weights must be LIST-form, matching the model's list outputs: Keras 3
    # cannot map dict-form sample_weight onto list outputs (raises KeyError: 0).
    # List outputs also keep model.predict returning a list, which is what 08's
    # pc_recenter path expects (`out[0] if isinstance(out, list)`).
    #
    # Validation carries source only -- dom_out is zero-weighted there, so
    # val_cls_out_loss stays a clean supervised signal.
    n_val = len(val_idx)
    model.fit(X_train,
              [y_cls, y_dom],
              sample_weight=[w_cls, w_dom],
              validation_data=(X_val,
                               [y[val_idx], np.zeros(n_val, dtype=int)],
                               [np.ones(n_val), np.zeros(n_val)]),
              epochs=a.epochs, batch_size=a.batch_size, shuffle=True, verbose=2,
              callbacks=cbs)

    # Score on the FULL target chip (uncropped, unsubsampled) -- labels used here
    # for evaluation only.
    tgt_all = np.where(ds_id == held_out)[0]
    X_eval = stack(tgt_all)
    prob = model.predict(X_eval, batch_size=a.batch_size, verbose=0)[0]
    pred = np.argmax(prob, axis=1)
    acc = float((pred == y[tgt_all]).mean() * 100)
    print(f"\n[RESULT] {held_out}  domain={a.domain}  UDA-DANN LOFO acc = {acc:.2f}%")

    import joblib
    from safe_io import safe_keras_save
    tag = f"udadann_{a.domain}_w{a.domain_weight:g}"
    safe_keras_save(model, out_dir / f"{tag}_{CURVE_TYPE}_lofo_{held_out}_model.keras")
    joblib.dump({"held_out": held_out, "domain": a.domain,
                 "domain_weight": a.domain_weight, "acc": acc,
                 "n_source": n_src, "n_target_unlabelled": n_tgt,
                 "y_true": y[tgt_all], "y_pred": pred,
                 "well_ids_test": wells[tgt_all],
                 "class_names": [str(c) for c in encoder.classes_]},
                out_dir / f"{tag}_{CURVE_TYPE}_lofo_{held_out}.joblib", compress=3)
    print(f"[DONE] saved -> {out_dir}")


if __name__ == "__main__":
    main()
