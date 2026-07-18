"""Source separation pretraining for multiplex PCR classification.

Phase 1 — Encoder + n_targets parametric decoders trained with:
  L = L_absent  +  λ_cons * L_consist  +  λ_anch * Σ L_anchor_j
    + λ_var  * L_var

All losses computed in rendered curve space.  No concentration required at
training time — L_anchor uses a nearest-neighbour reference bank built from
single-target curves only.

Phase 2 / 3 — build_source_sep_classifier attaches a per-target sigmoid
classification head to the frozen (Phase 2) or unfrozen (Phase 3) encoder.
"""

import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────────────
# Sigmoid rendering — pure TF, on gradient path
# Formula matches sigmoid_5p in main/utils/sigmoid_fitting.py:
#   Fm / (1 + exp(-Sc * (t - Cs)))^As + Fb
# Param order: [Fm, Fb, Sc, Cs, As]  (indices 0-4)
# ──────────────────────────────────────────────────────────────────────────────

def render_sigmoid(params, T=45):
    """Render 5-parameter sigmoid curves from a parameter batch.

    Parameters
    ----------
    params : (batch, 5) tensor — [Fm, Fb, Sc, Cs, As]
    T      : number of time steps

    Returns
    -------
    (batch, T) float32 tensor — rendered curves, differentiable w.r.t. params
    """
    Fm = params[:, 0:1]
    Fb = params[:, 1:2]
    Sc = params[:, 2:3]
    Cs = params[:, 3:4]
    As = params[:, 4:5]
    t  = tf.cast(tf.range(T), tf.float32)[tf.newaxis, :]  # (1, T)
    return Fm / (1.0 + tf.exp(-Sc * (t - Cs))) ** As + Fb  # (batch, T)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

def build_source_sep_encoder(T, d_shared, d_target, n_targets=3):
    """Encoder: Input(T,1) → structured bottleneck (d_shared + n_targets*d_target).

    Architecture: two causal Conv1D (stride-2 on second) → BiGRU(32) →
    LayerNorm → Dense bottleneck (linear — no activation to prevent dying-ReLU
    collapse in per-target dims).

    Layer names use stable name= kwargs for positional load_weights compatibility.
    """
    inputs = tf.keras.layers.Input(shape=(T, 1), name='ss_input')
    x = tf.keras.layers.Conv1D(
        16, 5, activation='relu', padding='causal', name='ss_conv1')(inputs)
    x = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='causal', strides=2, name='ss_conv2')(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(32), name='ss_bigru')(x)
    x = tf.keras.layers.LayerNormalization(name='ss_ln')(x)
    z = tf.keras.layers.Dense(
        d_shared + n_targets * d_target, name='ss_bottleneck')(x)  # linear — no relu
    return tf.keras.Model(inputs=inputs, outputs=z, name='source_sep_encoder')


# ──────────────────────────────────────────────────────────────────────────────
# Parametric decoder (one per target channel)
# ──────────────────────────────────────────────────────────────────────────────

def build_parametric_decoder(d_shared, d_target, T_max=45, j=0):
    """MLP decoder: (z_shared || z_j) → 5 constrained sigmoid params.

    Output param order: [Fm, Fb, Sc, Cs, As]  — matches sigmoid_5p convention.
    Activations enforce physical constraints (prevents negative components
    exploiting L_consist degeneracy):
      Fm  → softplus          (strictly positive amplitude)
      Sc  → softplus          (strictly positive slope)
      Cs  → sigmoid * T_max   (threshold ∈ [0, T])
      As  → softplus + 1      (shape exponent ≥ 1)
      Fb  → linear            (baseline, can be slightly negative)
    """
    inp = tf.keras.layers.Input(shape=(d_shared + d_target,), name=f'dec{j}_in')
    h   = tf.keras.layers.Dense(32, activation='relu', name=f'dec{j}_hidden')(inp)

    Fm = tf.keras.layers.Dense(1, name=f'dec{j}_Fm_lin')(h)
    Fm = tf.keras.layers.Activation(tf.math.softplus, name=f'dec{j}_Fm')(Fm)

    Fb = tf.keras.layers.Dense(1, name=f'dec{j}_Fb')(h)

    Sc = tf.keras.layers.Dense(1, name=f'dec{j}_Sc_lin')(h)
    Sc = tf.keras.layers.Activation(tf.math.softplus, name=f'dec{j}_Sc')(Sc)

    Cs = tf.keras.layers.Dense(1, name=f'dec{j}_Cs_lin')(h)
    Cs = tf.keras.layers.Lambda(
        lambda t: tf.sigmoid(t) * float(T_max), name=f'dec{j}_Cs')(Cs)

    As = tf.keras.layers.Dense(1, name=f'dec{j}_As_lin')(h)
    As = tf.keras.layers.Lambda(
        lambda t: tf.math.softplus(t) + 1.0, name=f'dec{j}_As')(As)

    params = tf.keras.layers.Concatenate(name=f'dec{j}_params')([Fm, Fb, Sc, Cs, As])
    return tf.keras.Model(inputs=inp, outputs=params, name=f'decoder_{j}')


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 model
# ──────────────────────────────────────────────────────────────────────────────

class MultiLabelSourceSepPhase1Model(tf.keras.Model):
    """Source separation pretraining model.

    Wraps one encoder and n_targets parametric decoders.
    train_step / test_step compute:
        L = L_absent  +  λ_cons * L_consist  +  λ_anch * Σ_{j active} L_anchor_j
          + λ_var  * L_var

    L_var penalises low per-target-dim variance across the batch to prevent
    representation collapse (needed because the linear bottleneck has no
    activation that would otherwise bound the variance).

    Parameters
    ----------
    encoder   : Keras Model, Input(T,1) → (batch, d_shared + n*d_target)
    decoders  : list of n_targets Keras Models, Input(d_sh+d_t) → (batch, 5)
    nn_bank   : list of n_targets float32 arrays (N_single_j, T) — reference curves
    T         : time steps (must match encoder input)
    d_shared  : shared latent dims
    d_target  : per-target latent dims
    n_targets : number of targets
    lambda_cons, lambda_anch, lambda_var : loss weights
    var_margin : minimum acceptable std for per-target dims (default 0.05)
    k         : k nearest neighbours for L_anchor
    """

    def __init__(self, encoder, decoders, nn_bank, T=45,
                 d_shared=16, d_target=10, n_targets=3,
                 lambda_cons=1.0, lambda_anch=0.5, lambda_var=0.1,
                 var_margin=0.05, k=5, **kwargs):
        super().__init__(**kwargs)
        self.encoder      = encoder
        self.decoders     = decoders
        self.nn_bank      = [tf.constant(b, dtype=tf.float32) for b in nn_bank]
        self.T            = T
        self.d_shared     = d_shared
        self.d_target     = d_target
        self.n_targets    = n_targets
        self.lambda_cons  = lambda_cons
        self.lambda_anch  = lambda_anch
        self.lambda_var   = lambda_var
        self.var_margin   = var_margin
        self.k            = k

    def _split_z(self, z):
        z_shared = z[:, :self.d_shared]
        z_parts  = [
            z[:, self.d_shared + j * self.d_target:
                   self.d_shared + (j + 1) * self.d_target]
            for j in range(self.n_targets)
        ]
        return z_shared, z_parts

    def _encode_and_decode(self, x, training):
        """Run encoder + all decoders; returns (z_parts, rendered_list)."""
        z                 = self.encoder(x, training=training)
        z_shared, z_parts = self._split_z(z)
        rendered          = []
        for j, dec in enumerate(self.decoders):
            dec_in   = tf.concat([z_shared, z_parts[j]], axis=-1)
            params_j = dec(dec_in, training=training)
            rendered.append(render_sigmoid(params_j, T=self.T))
        return z_parts, rendered

    def _nn_anchor_loss(self, rendered_j, ref_bank_j, active_mask):
        """L_anchor for channel j, masked to active samples.

        rendered_j  : (batch, T)
        ref_bank_j  : (N_ref, T)  tf.constant
        active_mask : (batch,)  float — 1.0 where y_j == 1
        """
        diffs = (tf.expand_dims(rendered_j, 1)
                 - tf.expand_dims(ref_bank_j, 0))          # (batch, N_ref, T)
        dists = tf.reduce_sum(diffs ** 2, axis=-1)          # (batch, N_ref)
        _, idx   = tf.math.top_k(-dists, k=self.k)         # (batch, k)
        anchors  = tf.gather(ref_bank_j, idx)               # (batch, k, T)
        anchor   = tf.reduce_mean(anchors, axis=1)          # (batch, T)
        sq_err   = tf.reduce_mean((rendered_j - anchor) ** 2, axis=-1)  # (batch,)
        n_active = tf.reduce_sum(active_mask) + 1e-8
        return tf.reduce_sum(active_mask * sq_err) / n_active

    def _compute_losses(self, x, y_bin, training):
        x_curve  = tf.squeeze(x, axis=-1)          # (batch, T)
        y_f      = tf.cast(y_bin, tf.float32)       # (batch, n_targets)
        z_parts, rendered = self._encode_and_decode(x, training)

        # L_absent: absent channels → zero rendered curve
        l_absent = tf.constant(0.0)
        for j in range(self.n_targets):
            absent = 1.0 - y_f[:, j]               # (batch,)
            sq     = tf.reduce_mean(rendered[j] ** 2, axis=-1)
            l_absent = l_absent + tf.reduce_mean(absent * sq)

        # L_consist: sum of active channels ≈ input
        active_sum = tf.zeros_like(x_curve)
        for j in range(self.n_targets):
            active_sum = active_sum + y_f[:, j:j+1] * rendered[j]
        l_consist = tf.reduce_mean((x_curve - active_sum) ** 2)

        # L_anchor: active channels close to NN reference (no concentration used)
        l_anchor = tf.constant(0.0)
        for j in range(self.n_targets):
            l_anchor = l_anchor + self._nn_anchor_loss(
                rendered[j], self.nn_bank[j], y_f[:, j])

        # L_var: penalise low per-target-dim variance (prevents dead-dim collapse)
        l_var = tf.constant(0.0)
        for j in range(self.n_targets):
            z_std = tf.math.reduce_std(z_parts[j], axis=0)  # (d_target,)
            l_var = l_var + tf.reduce_mean(tf.nn.relu(self.var_margin - z_std))

        loss = (l_absent
                + self.lambda_cons  * l_consist
                + self.lambda_anch  * l_anchor
                + self.lambda_var   * l_var)
        return loss, l_absent, l_consist, l_anchor, l_var

    def call(self, x, training=False):
        _, rendered = self._encode_and_decode(x, training)
        return rendered

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'], tf.int32)
        with tf.GradientTape() as tape:
            loss, l_ab, l_co, l_an, l_va = self._compute_losses(x, y_bin, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'l_absent': l_ab, 'l_consist': l_co,
                'l_anchor': l_an, 'l_var': l_va}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'], tf.int32)
        loss, l_ab, l_co, l_an, l_va = self._compute_losses(x, y_bin, training=False)
        return {'loss': loss, 'l_absent': l_ab, 'l_consist': l_co,
                'l_anchor': l_an, 'l_var': l_va}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 / 3 classifier
# ──────────────────────────────────────────────────────────────────────────────

def build_source_sep_classifier(encoder, d_shared, d_target, n_targets=3):
    """Attach a per-target sigmoid classification head to the pretrained encoder.

    Each target j gets its own MLP head that sees [z_shared || z_j], matching
    the pretraining decoder structure.  This ensures that the structured latent
    decomposition is actually exploited at classification time.

    Parameters
    ----------
    encoder  : Keras Model from build_source_sep_encoder (output dim = d_shared + n*d_target)
    d_shared, d_target, n_targets : must match values used to build the encoder

    Returns
    -------
    Keras functional Model with Input(T,1) → (batch, n_targets) sigmoid output.
    Call `encoder.trainable = False` before compiling for Phase 2.
    Call `encoder.trainable = True`  and recompile with lr=1e-5 for Phase 3.
    """
    T      = encoder.input_shape[1]
    inputs = tf.keras.layers.Input(shape=(T, 1), name='cls_input')
    z      = encoder(inputs)                                    # (batch, d_sh + n*d_t)

    # Slice shared dims — used by every per-target head
    z_shared = tf.keras.layers.Lambda(
        lambda t: t[:, :d_shared], name='cls_z_shared')(z)

    outs = []
    for j in range(n_targets):
        s    = d_shared + j * d_target
        e    = d_shared + (j + 1) * d_target
        z_j  = tf.keras.layers.Lambda(
            lambda t, _s=s, _e=e: t[:, _s:_e], name=f'cls_z{j}')(z)
        h_j  = tf.keras.layers.Concatenate(name=f'cls_cat{j}')([z_shared, z_j])
        h_j  = tf.keras.layers.Dense(
            d_shared + d_target, activation='relu', name=f'cls_h{j}')(h_j)
        out_j = tf.keras.layers.Dense(1, activation='sigmoid', name=f'cls_sig{j}')(h_j)
        outs.append(out_j)

    out = tf.keras.layers.Concatenate(name='cls_out')(outs)
    return tf.keras.Model(inputs=inputs, outputs=out, name='source_sep_classifier')
