"""
model_utils_gated.py  —  Ultimate Dual-Branch Fusion Architectures
===================================================================
CNN + (BiGRU | Transformer) dual-branch models with four alternative fusion mechanisms.
Each replaces the plain Concatenate() used in model_utils.py's cnn_gru_dual / cnn_trans_dual.

Upgrade summary vs. original model_utils.py dual models
---------------------------------------------------------
CNN branch   : Conv1D+BatchNorm+SE (Squeeze-and-Excitation channel attention)
               Returns (conv_seq, emb) — seq used by co-attention models.
GRU branch   : Second BiGRU keeps return_sequences=True; attention-weighted pooling
               replaces plain GlobalAvgPool; returns (seq, emb) or emb.
Transformer  : Unchanged core; returns (pre-pool seq, emb) when needed.

Fusion mechanisms
-----------------
Gate      : 2-layer gate MLP (Dense(64,relu)→Dense(32,σ)) + LayerNorm on output.
            gate ∈ [0,1]^32: merged = gate⊙cnn + (1-gate)⊙other
            XAI: model.get_layer('fuse_gate').output → gate values per sample

Hadamard  : [cnn | other | BatchNorm(cnn⊙other)] — interaction term is BN-normalised
            so its scale doesn't dominate the downstream Dense head.
            Output width: 96 (3×32).

Co-Attn   : Bidirectional cross-attention (co-attention):
              Dir-1 CNN_emb (query) → other_seq (key/value) → context enriches CNN
              Dir-2 other_emb (query) → cnn_seq  (key/value) → context enriches other
            Both directions use residual adds; merged = [cnn_enriched ‖ other_enriched].
            Output width: 64.

FiLM      : Two-stage conditioning: CNN affinely modulates other twice in series.
              Stage-1: mod1 = relu(γ1⊙other + β1)
              Stage-2: mod2 = relu(γ2⊙Dense(mod1) + β2)
            merged = [cnn_emb ‖ mod2].  Output width: 64.

XAI-relevant layer names (all extractable via model.get_layer(...).output)
---------------------------------------------------------------------------
  fuse_gate       : gate values   (gate models)
  fuse_mha1       : cross-attn CNN→other  (co-attention models)
  fuse_mha2       : cross-attn other→CNN  (co-attention models)
  fuse_gamma1/2   : FiLM scale params  (FiLM models)
  fuse_beta1/2    : FiLM shift params  (FiLM models)
  cnn_emb         : CNN branch embedding (all models)
  gru_emb / trans_emb : other branch embedding (all models)

Note on custom layers (_SumPool1D / _OneMinus)
------------------------------------------------
Attention-weighted pooling (GRU branch) and (1-gate) (gate fusion) used to be Lambda
layers. A Lambda's saved bytecode loses its closure's globals on deserialization, so
`tf` was unbound and inference raised NameError the instant the model was reloaded in
a different process — model.save()/load_model() alone didn't catch it since neither
runs the layer. Replaced with tiny registered Layer subclasses, which serialize via
get_config/from_config instead and don't have this problem. Any module that loads
these .keras files (e.g. 07_attribution_vis_all.py) must `import model_utils_gated`
first so the @register_keras_serializable decorators run and the classes are
resolvable by name during deserialization.

Integrated into 03_main_training.py / 04_cross_dataset_training.py (training),
07_attribution_vis_all.py (XAI), and config.py's MODEL_KEY_MAP/MODEL_PRINT_MAP
(06/08 reporting) via the _ALL_FACTORIES dict below.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

_EMB_DIM = 32  # shared branch output dimension


def inception_smoothing_block(inputs, filters=8, kernel_sizes=(3, 7, 15, 31)):
    """Multi-scale 1D CNN front-end for learned curve smoothing.

    Applies parallel Conv1D branches with different kernel sizes (short→long temporal
    context), concatenates, then compresses via a 1×1 bottleneck. Output shape:
    (batch, T, filters) — same temporal length as input due to padding='same'.

    Defined here (not model_utils.py) to avoid the circular import created by
    model_utils_gated importing from model_utils.
    """
    branches = [
        tf.keras.layers.Conv1D(filters, k, padding='same', activation='relu',
                               name=f'inc_k{k}')(inputs)
        for k in kernel_sizes
    ]
    merged = tf.keras.layers.Concatenate(name='inc_merge')(branches)
    return tf.keras.layers.Conv1D(filters, 1, padding='same', activation='relu',
                                  name='inc_bottleneck')(merged)


# Custom layers instead of Lambda: a Lambda's saved bytecode is reconstructed without
# the original module's globals, so `tf` is unbound and inference raises NameError the
# moment the model is reloaded in a different process (caught by an actual inference
# smoke test in 07_attribution_vis_all.py, not by load_model() alone, which doesn't run
# the layer). Subclassed layers serialize via get_config/from_config instead, so this
# doesn't happen, and unsafe Lambda deserialization isn't needed either.
@tf.keras.utils.register_keras_serializable(package="model_utils_gated")
class _SumPool1D(tf.keras.layers.Layer):
    """Sums over axis=1 (the time dimension). Used for attention-weighted pooling."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


@tf.keras.utils.register_keras_serializable(package="model_utils_gated")
class _OneMinus(tf.keras.layers.Layer):
    """Computes 1 - x (used for the gate fusion's complementary weight)."""
    def call(self, x):
        return 1.0 - x


# ============================================================
# BRANCH BUILDERS  (private)
# ============================================================

def _cnn_branch(inp, emb_dim=_EMB_DIM, pfx="cnn"):
    """
    Local-feature CNN branch with BatchNorm + Squeeze-and-Excitation.

    Returns: (conv_seq, emb)
        conv_seq : (batch, T, emb_dim)  — SE-weighted conv output, used by co-attention.
        emb      : (batch, emb_dim)     — pooled embedding for fusion.

    Param reduction vs original: replaces Flatten+Dense(752→32, ~24k) with GAP+Dense(32→32,~1k).
    SE block adds channel-wise recalibration with only emb_dim*(emb_dim/4 + emb_dim) ≈ 1.3k params.
    """
    c = tf.keras.layers.Conv1D(emb_dim, 5, padding='same', activation='relu',
                               name=f"{pfx}_c1")(inp)
    c = tf.keras.layers.BatchNormalization(name=f"{pfx}_bn1")(c)
    c = tf.keras.layers.Conv1D(emb_dim, 3, padding='same', activation='relu',
                               name=f"{pfx}_c2")(c)
    c = tf.keras.layers.BatchNormalization(name=f"{pfx}_bn2")(c)

    # Squeeze-and-Excitation: recalibrate which feature channels matter for this sample
    se = tf.keras.layers.GlobalAveragePooling1D(name=f"{pfx}_se_gap")(c)
    se = tf.keras.layers.Dense(emb_dim // 4, activation='relu',    name=f"{pfx}_se_sq")(se)
    se = tf.keras.layers.Dense(emb_dim,      activation='sigmoid', name=f"{pfx}_se_ex")(se)
    se = tf.keras.layers.Reshape((1, emb_dim), name=f"{pfx}_se_rsh")(se)
    c  = tf.keras.layers.Multiply(name=f"{pfx}_se_mul")([c, se])   # (batch, T, emb_dim)

    emb = tf.keras.layers.GlobalAveragePooling1D(name=f"{pfx}_gap")(c)
    emb = tf.keras.layers.Dense(emb_dim, activation='relu', name=f"{pfx}_emb")(emb)
    return c, emb


def _gru_branch(inp, emb_dim=_EMB_DIM, return_seq=False, pfx="gru"):
    """
    Two-layer BiGRU with LayerNorm + attention-weighted pooling.

    Attention pooling learns which timesteps to emphasise (vs plain GlobalAvgPool).

    return_seq=False → emb (batch, emb_dim)
    return_seq=True  → (seq, emb)
        seq : (batch, T, emb_dim)  — raw BiGRU-2 output, used by co-attention as key/value.
        emb : (batch, emb_dim)     — attention-pooled embedding.
    """
    g = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(emb_dim, return_sequences=True), name=f"{pfx}_bi1")(inp)
    g = tf.keras.layers.LayerNormalization(name=f"{pfx}_ln1")(g)
    # Second BiGRU always returns sequences (needed for co-attention and pooling)
    g2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(emb_dim // 2, return_sequences=True), name=f"{pfx}_bi2")(g)
    g2 = tf.keras.layers.LayerNormalization(name=f"{pfx}_ln2")(g2)
    g2 = tf.keras.layers.Dropout(0.2, name=f"{pfx}_drop")(g2)
    # g2: (batch, T, emb_dim)

    # Attention-weighted pooling: learn which timesteps matter
    attn = tf.keras.layers.Dense(1, use_bias=False, name=f"{pfx}_attn_w")(g2)   # (B, T, 1)
    attn = tf.keras.layers.Softmax(axis=1, name=f"{pfx}_attn_sm")(attn)
    weighted = tf.keras.layers.Multiply(name=f"{pfx}_attn_mul")([g2, attn])      # (B, T, emb_dim)
    pooled   = _SumPool1D(name=f"{pfx}_attn_pool")(weighted)  # (B, emb_dim)
    emb = tf.keras.layers.Dense(emb_dim, activation='relu', name=f"{pfx}_emb")(pooled)

    if not return_seq:
        return emb
    return g2, emb


def _trans_branch(inp, head_size=_EMB_DIM, num_heads=2, ff_dim=32,
                  num_blocks=2, dropout=0.1, return_seq=False, pfx="trans"):
    """
    Transformer encoder with positional embedding (unchanged from original architecture).

    return_seq=False → emb (batch, emb_dim)
    return_seq=True  → (seq, emb)
        seq : (batch, T_reduced, head_size)  — pre-pool transformer output for co-attention.
        emb : (batch, emb_dim)               — pooled embedding.
    """
    t = tf.keras.layers.Conv1D(
        head_size, 5, strides=2, padding="same", activation="relu",
        name=f"{pfx}_proj")(inp)
    t = tf.keras.layers.MaxPooling1D(2, padding="same", name=f"{pfx}_pool")(t)

    seq_len   = t.shape[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb   = tf.keras.layers.Embedding(seq_len, head_size, name=f"{pfx}_pos")(positions)
    t = t + pos_emb

    for i in range(num_blocks):
        a = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            name=f"{pfx}_mha{i}")(t, t)
        a = tf.keras.layers.Dropout(dropout, name=f"{pfx}_da{i}")(a)
        t = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{pfx}_ln1_{i}")(t + a)
        ff = tf.keras.layers.Dense(ff_dim, activation="relu", name=f"{pfx}_ff1_{i}")(t)
        ff = tf.keras.layers.Dropout(dropout, name=f"{pfx}_dff{i}")(ff)
        ff = tf.keras.layers.Dense(head_size, name=f"{pfx}_ff2_{i}")(ff)
        t  = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{pfx}_ln2_{i}")(t + ff)
    # t: (batch, T_reduced, head_size)

    pooled = tf.keras.layers.GlobalAveragePooling1D(
        data_format="channels_last", name=f"{pfx}_gap")(t)
    emb = tf.keras.layers.Dense(_EMB_DIM, activation="relu", name=f"{pfx}_emb")(pooled)

    if not return_seq:
        return emb
    return t, emb


def _head(merged, output_size, pfx="head"):
    """Classifier head: Dense(64,relu) → Dropout → Dense(n_classes, softmax)."""
    z = tf.keras.layers.Dense(64, activation='relu', name=f"{pfx}_d1")(merged)
    z = tf.keras.layers.Dropout(0.2, name=f"{pfx}_drop")(z)
    return tf.keras.layers.Dense(output_size, activation='softmax', name=f"{pfx}_out")(z)


# ============================================================
# FUSION HELPERS  (private)
# ============================================================

def _fuse_gate(cnn_emb, other_emb, emb_dim=_EMB_DIM, pfx="fuse"):
    """
    Two-layer gate MLP + LayerNorm on gated output.

    gate ∈ [0,1]^emb_dim:  merged = gate⊙cnn_emb + (1-gate)⊙other_emb

    Upgrade vs v1: single Dense → 2-layer MLP (more expressive gate prediction)
                   + LayerNorm on merged output (stabilises training).
    Output shape: (batch, emb_dim) = 32.

    XAI: model.get_layer('{pfx}_gate').output → gate values per sample ∈ [0,1]^32
         Values close to 1 → CNN dominates; close to 0 → other branch dominates.
    """
    cat  = tf.keras.layers.Concatenate(name=f"{pfx}_cat")([cnn_emb, other_emb])
    h    = tf.keras.layers.Dense(64, activation='relu', name=f"{pfx}_gate_h")(cat)
    gate = tf.keras.layers.Dense(emb_dim, activation='sigmoid', name=f"{pfx}_gate")(h)
    inv  = _OneMinus(name=f"{pfx}_inv")(gate)
    g_cnn   = tf.keras.layers.Multiply(name=f"{pfx}_g_cnn")([gate, cnn_emb])
    g_other = tf.keras.layers.Multiply(name=f"{pfx}_g_other")([inv, other_emb])
    merged  = tf.keras.layers.Add(name=f"{pfx}_add")([g_cnn, g_other])
    return    tf.keras.layers.LayerNormalization(name=f"{pfx}_merged")(merged)


def _fuse_hadamard(cnn_emb, other_emb, pfx="fuse"):
    """
    [cnn_emb ‖ other_emb ‖ BN(cnn_emb ⊙ other_emb)]

    BN on the product normalises the interaction term's scale relative to the raw embeddings,
    preventing the product from being dominated by large-magnitude features.
    Output shape: (batch, 3 × emb_dim) = 96.
    """
    product = tf.keras.layers.Multiply(name=f"{pfx}_prod")([cnn_emb, other_emb])
    product = tf.keras.layers.BatchNormalization(name=f"{pfx}_prod_bn")(product)
    return tf.keras.layers.Concatenate(name=f"{pfx}_merged")([cnn_emb, other_emb, product])


def _fuse_coattn(cnn_seq, cnn_emb, other_seq, other_emb,
                 emb_dim=_EMB_DIM, num_heads=2, pfx="fuse"):
    """
    Co-attention: both branches attend to EACH OTHER simultaneously.

    Dir-1  CNN_emb (query=1 token) → other_seq (key/value=T tokens) → ctx1
           CNN learns which global timesteps support its local reading.
    Dir-2  other_emb (query=1 token) → cnn_seq (key/value=T tokens) → ctx2
           Global branch learns which local CNN features to focus on.

    Both outputs are added as residuals to the original embeddings before final concat.
    merged = [cnn_emb + ctx1 ‖ other_emb + ctx2].  Output shape: (batch, 2 × emb_dim) = 64.

    XAI: model.get_layer('{pfx}_mha1') → CNN-to-other attention weights
         model.get_layer('{pfx}_mha2') → other-to-CNN attention weights
         (call with return_attention_scores=True in an extractor model)
    """
    key_dim = emb_dim // num_heads

    # Direction 1: CNN_emb queries other_seq
    kv1  = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_kv1")(other_seq)      # (B, T, emb_dim)
    q1   = tf.keras.layers.Reshape((1, emb_dim), name=f"{pfx}_q1")(cnn_emb)
    att1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{pfx}_mha1"
    )(query=q1, key=kv1, value=kv1)                                           # (B, 1, emb_dim)
    att1 = tf.keras.layers.Reshape((emb_dim,), name=f"{pfx}_att1_flat")(att1)
    ctx1 = tf.keras.layers.Dense(emb_dim, activation='relu', name=f"{pfx}_ctx1")(att1)

    # Direction 2: other_emb queries cnn_seq
    kv2  = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_kv2")(cnn_seq)        # (B, T, emb_dim)
    q2   = tf.keras.layers.Reshape((1, emb_dim), name=f"{pfx}_q2")(other_emb)
    att2 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{pfx}_mha2"
    )(query=q2, key=kv2, value=kv2)                                           # (B, 1, emb_dim)
    att2 = tf.keras.layers.Reshape((emb_dim,), name=f"{pfx}_att2_flat")(att2)
    ctx2 = tf.keras.layers.Dense(emb_dim, activation='relu', name=f"{pfx}_ctx2")(att2)

    # Residual enrichment: each branch gets context from the other
    cnn_enriched   = tf.keras.layers.Add(name=f"{pfx}_cnn_enr")([cnn_emb,   ctx1])
    other_enriched = tf.keras.layers.Add(name=f"{pfx}_other_enr")([other_emb, ctx2])
    return tf.keras.layers.Concatenate(name=f"{pfx}_merged")([cnn_enriched, other_enriched])


def _fuse_film(cnn_emb, other_emb, emb_dim=_EMB_DIM, pfx="fuse"):
    """
    Two-stage FiLM (Feature-wise Linear Modulation).

    Stage-1: mod1  = relu(γ1(cnn)⊙other_emb  + β1(cnn))
    Stage-2: mod2  = relu(γ2(cnn)⊙Dense(mod1) + β2(cnn))

    Cascaded conditioning forces CNN to modulate the other branch at two abstraction
    levels: raw embedding → transformed embedding.  Asymmetric: CNN drives both stages.
    merged = [cnn_emb ‖ mod2].  Output shape: (batch, 2 × emb_dim) = 64.

    XAI: model.get_layer('{pfx}_gamma1/2') → scale modulation per sample
         model.get_layer('{pfx}_beta1/2')  → shift modulation per sample
    """
    # Stage 1
    gamma1 = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_gamma1")(cnn_emb)
    beta1  = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_beta1")(cnn_emb)
    scaled1 = tf.keras.layers.Multiply(name=f"{pfx}_sc1")([gamma1, other_emb])
    mod1    = tf.keras.layers.Add(name=f"{pfx}_sh1")([scaled1, beta1])
    mod1    = tf.keras.layers.Activation('relu', name=f"{pfx}_mod1")(mod1)

    # Stage 2: Dense transform then condition again
    mod1_t = tf.keras.layers.Dense(emb_dim, activation='relu', name=f"{pfx}_t2")(mod1)
    gamma2 = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_gamma2")(cnn_emb)
    beta2  = tf.keras.layers.Dense(emb_dim, name=f"{pfx}_beta2")(cnn_emb)
    scaled2 = tf.keras.layers.Multiply(name=f"{pfx}_sc2")([gamma2, mod1_t])
    mod2    = tf.keras.layers.Add(name=f"{pfx}_sh2")([scaled2, beta2])
    mod2    = tf.keras.layers.Activation('relu', name=f"{pfx}_mod2")(mod2)

    return tf.keras.layers.Concatenate(name=f"{pfx}_merged")([cnn_emb, mod2])


# ============================================================
# GATE FUSION MODELS
# ============================================================

def create_cnn_gru_gate_model(input_size_curve, output_size, inception_smoothing=False):
    """
    CNN (SE+BN) + BiGRU (attention pooling), fused via 2-layer gate + LayerNorm.
    Merged width: 32.  Gate layer: 'fuse_gate'.
    """
    inp = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    gru_emb    = _gru_branch(x, pfx="gru")
    merged     = _fuse_gate(cnn_emb, gru_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_gru_gate")
    model.compile(tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_trans_gate_model(input_size_curve, output_size,
                                head_size=32, num_heads=2, ff_dim=32,
                                num_blocks=2, dropout=0.1, inception_smoothing=False):
    """
    CNN (SE+BN) + Transformer, fused via 2-layer gate + LayerNorm.
    Merged width: 32.  Gate layer: 'fuse_gate'.
    """
    inp        = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    trans_emb  = _trans_branch(x, head_size, num_heads, ff_dim, num_blocks, dropout, pfx="trans")
    merged     = _fuse_gate(cnn_emb, trans_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_trans_gate")
    model.compile(tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================
# HADAMARD FUSION MODELS
# ============================================================

def create_cnn_gru_hadamard_model(input_size_curve, output_size, inception_smoothing=False):
    """
    CNN (SE+BN) + BiGRU (attention pooling), fused via [cnn ‖ gru ‖ BN(cnn⊙gru)].
    Merged width: 96.
    """
    inp = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    gru_emb    = _gru_branch(x, pfx="gru")
    merged     = _fuse_hadamard(cnn_emb, gru_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_gru_hadamard")
    model.compile(tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_trans_hadamard_model(input_size_curve, output_size,
                                    head_size=32, num_heads=2, ff_dim=32,
                                    num_blocks=2, dropout=0.1, inception_smoothing=False):
    """
    CNN (SE+BN) + Transformer, fused via [cnn ‖ trans ‖ BN(cnn⊙trans)].
    Merged width: 96.
    """
    inp        = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    trans_emb  = _trans_branch(x, head_size, num_heads, ff_dim, num_blocks, dropout, pfx="trans")
    merged     = _fuse_hadamard(cnn_emb, trans_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_trans_hadamard")
    model.compile(tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================
# CO-ATTENTION (bidirectional cross-attention) FUSION MODELS
# ============================================================

def create_cnn_gru_crossattn_model(input_size_curve, output_size, num_heads=2, inception_smoothing=False):
    """
    CNN (SE+BN) + BiGRU, fused via co-attention (bidirectional cross-attention).
      - BiGRU exposes full sequence as key/value for CNN to attend over.
      - CNN exposes SE-weighted conv sequence as key/value for GRU to attend over.
    Both directions use residual addition before final concat.
    Merged width: 64.  Attention layers: 'fuse_mha1' (CNN→GRU), 'fuse_mha2' (GRU→CNN).
    """
    inp = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    cnn_seq, cnn_emb  = _cnn_branch(x, pfx="cnn")
    gru_seq, gru_emb  = _gru_branch(x, return_seq=True, pfx="gru")
    merged = _fuse_coattn(cnn_seq, cnn_emb, gru_seq, gru_emb,
                          num_heads=num_heads, pfx="fuse")
    out   = _head(merged, output_size)
    model = tf.keras.Model(inp, out, name="cnn_gru_crossattn")
    model.compile(tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_trans_crossattn_model(input_size_curve, output_size,
                                     head_size=32, num_heads=2, ff_dim=32,
                                     num_blocks=2, dropout=0.1, inception_smoothing=False):
    """
    CNN (SE+BN) + Transformer, fused via co-attention.
      - Transformer exposes its pre-pool sequence as key/value for CNN.
      - CNN exposes SE conv sequence as key/value for Transformer.
    Merged width: 64.  Attention layers: 'fuse_mha1', 'fuse_mha2'.
    """
    inp = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    cnn_seq, cnn_emb     = _cnn_branch(x, pfx="cnn")
    trans_seq, trans_emb = _trans_branch(
        x, head_size, num_heads, ff_dim, num_blocks, dropout, return_seq=True, pfx="trans")
    merged = _fuse_coattn(cnn_seq, cnn_emb, trans_seq, trans_emb,
                          num_heads=num_heads, pfx="fuse")
    out   = _head(merged, output_size)
    model = tf.keras.Model(inp, out, name="cnn_trans_crossattn")
    model.compile(tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================
# FiLM FUSION MODELS
# ============================================================

def create_cnn_gru_film_model(input_size_curve, output_size, inception_smoothing=False):
    """
    CNN (SE+BN) + BiGRU (attention pooling), fused via two-stage FiLM.
    CNN drives both conditioning stages.  Merged width: 64.
    FiLM params: 'fuse_gamma1/2', 'fuse_beta1/2'.
    """
    inp = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    gru_emb    = _gru_branch(x, pfx="gru")
    merged     = _fuse_film(cnn_emb, gru_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_gru_film")
    model.compile(tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_trans_film_model(input_size_curve, output_size,
                                head_size=32, num_heads=2, ff_dim=32,
                                num_blocks=2, dropout=0.1, inception_smoothing=False):
    """
    CNN (SE+BN) + Transformer, fused via two-stage FiLM.
    Merged width: 64.  FiLM params: 'fuse_gamma1/2', 'fuse_beta1/2'.
    """
    inp        = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = inception_smoothing_block(inp) if inception_smoothing else inp
    _, cnn_emb = _cnn_branch(x, pfx="cnn")
    trans_emb  = _trans_branch(x, head_size, num_heads, ff_dim, num_blocks, dropout, pfx="trans")
    merged     = _fuse_film(cnn_emb, trans_emb, pfx="fuse")
    out        = _head(merged, output_size)
    model      = tf.keras.Model(inp, out, name="cnn_trans_film")
    model.compile(tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
                  'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================
# CONVENIENCE
# ============================================================

_ALL_FACTORIES = {
    "cnn_gru_gate":        create_cnn_gru_gate_model,
    "cnn_gru_hadamard":    create_cnn_gru_hadamard_model,
    "cnn_gru_crossattn":   create_cnn_gru_crossattn_model,
    "cnn_gru_film":        create_cnn_gru_film_model,
    "cnn_trans_gate":      create_cnn_trans_gate_model,
    "cnn_trans_hadamard":  create_cnn_trans_hadamard_model,
    "cnn_trans_crossattn": create_cnn_trans_crossattn_model,
    "cnn_trans_film":      create_cnn_trans_film_model,
}


def build_all_gated_models(input_size_curve, output_size):
    """Build all 8 ultimate gated-fusion dual-branch models. Returns dict[name → model]."""
    return {name: fn(input_size_curve, output_size) for name, fn in _ALL_FACTORIES.items()}


# ============================================================
# ARCHITECTURE INSPECTION
# ============================================================

if __name__ == "__main__":
    INPUT_SIZE = 100
    N_CLASSES  = 3

    # Expected merged widths per fusion type:
    #   gate      →  32   (Add + LayerNorm, same width as each branch)
    #   hadamard  →  96   (3 × 32, BN-normalised product term)
    #   coattn    →  64   (2 × 32, residual-enriched both branches)
    #   film      →  64   (2 × 32, cnn ‖ stage-2-modulated-other)

    for name, factory in _ALL_FACTORIES.items():
        print(f"\n{'='*70}")
        print(f"  {name.upper()}")
        print(f"{'='*70}")
        m = factory(INPUT_SIZE, N_CLASSES)
        m.summary()
        print(f"  Total params: {m.count_params():,}")
