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
    # Clip exponent to ±50 to prevent exp overflow (exp(50)≈5e21 is finite; beyond that is inf)
    exponent = tf.clip_by_value(-Sc * (t - Cs), -50.0, 50.0)
    return Fm / (1.0 + tf.exp(exponent)) ** As + Fb  # (batch, T)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

def build_source_sep_encoder(T, d_shared, d_target, n_targets=3):
    """Dual-branch encoder: shared trunk → CNN branch + GRU branch → bottleneck.

    CNN branch captures local sigmoid-rise features; GRU branch captures
    temporal dynamics across the full curve. Merge before a linear bottleneck
    (no activation — prevents dying-ReLU collapse in per-target dims).

    ~50K params vs ~13K for the old serial encoder; empirically dual-branch
    architectures outperform serial on single-modality PCR curves in this codebase.

    Layer names use stable name= kwargs for load_weights compatibility.
    """
    inputs = tf.keras.layers.Input(shape=(T, 1), name='ss_input')

    # Shared trunk: initial feature extraction from raw curve
    trunk = tf.keras.layers.Conv1D(
        32, 5, activation='relu', padding='causal', name='ss_trunk')(inputs)  # (T, 32)

    # CNN branch: local patterns via deeper convolutions + global average pooling
    cnn = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='causal', strides=2, name='ss_cnn1')(trunk)  # (T//2, 32)
    cnn = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='causal', name='ss_cnn2')(cnn)               # (T//2, 32)
    cnn = tf.keras.layers.GlobalAveragePooling1D(name='ss_cnn_gap')(cnn)               # (32,)
    cnn = tf.keras.layers.Dense(64, activation='relu', name='ss_cnn_emb')(cnn)        # (64,)

    # GRU branch: temporal dynamics via strided conv + BiGRU
    gru = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='causal', strides=2, name='ss_gru_conv')(trunk)  # (T//2, 16)
    gru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64), name='ss_bigru')(gru)                                     # (128,)

    # Merge branches → LayerNorm → linear bottleneck
    merged = tf.keras.layers.Concatenate(name='ss_merge')([cnn, gru])  # (192,)
    merged = tf.keras.layers.LayerNormalization(name='ss_ln')(merged)
    z = tf.keras.layers.Dense(
        d_shared + n_targets * d_target, name='ss_bottleneck')(merged)  # linear — no relu
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
    h   = tf.keras.layers.Dense(128, activation='relu', name=f'dec{j}_hidden')(inp)

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

def _per_target_supcon_loss(z_j, y_j, temp=0.07):
    """SupCon loss on per-target latent z_j.

    Positives for anchor i: all other wells where y_j == 1 AND anchor i also has y_j == 1.
    z_j is L2-normalised before similarity computation.

    Parameters
    ----------
    z_j : (batch, d_target) float32
    y_j : (batch,)          int32 — binary label for target j
    temp: float             — temperature (default 0.07)

    Returns
    -------
    Scalar loss.  Returns 0.0 when no positive pairs exist in the batch.
    """
    z_norm   = tf.math.l2_normalize(z_j, axis=-1)                    # (B, d)
    y        = tf.cast(y_j, tf.float32)[:, tf.newaxis]                # (B, 1)
    not_self = 1.0 - tf.eye(tf.shape(z_norm)[0])
    pos_mask = tf.matmul(y, y, transpose_b=True) * not_self           # (B, B)

    sim      = tf.matmul(z_norm, z_norm, transpose_b=True) / temp     # (B, B)
    sim_max  = tf.stop_gradient(tf.reduce_max(sim, axis=1, keepdims=True))
    exp_sim  = tf.exp(sim - sim_max)
    log_den  = tf.math.log(tf.reduce_sum(exp_sim * not_self, axis=1, keepdims=True) + 1e-8)
    log_prob = (sim - sim_max) - log_den
    pos_sum  = tf.reduce_sum(pos_mask, axis=1)
    has_pos  = tf.cast(pos_sum > 0, tf.float32)
    per_anc  = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (pos_sum + 1e-8)
    return tf.reduce_mean(per_anc * has_pos)


class MultiLabelSourceSepPhase1Model(tf.keras.Model):
    """Source separation pretraining model.

    Wraps one encoder and n_targets parametric decoders.
    train_step / test_step compute:
        L = L_absent  +  λ_cons * L_consist  +  λ_anch * Σ_{j active} L_anchor_j
          + λ_var  * L_var  +  λ_supcon * Σ_j SupCon(z_j, y_j)

    L_var penalises low per-target-dim variance across the batch to prevent
    representation collapse.
    L_supcon (optional) pulls same-target per-target latents together; when
    lambda_supcon=0 (default) it is a no-op and SC0 vs SC1 use the same
    pretraining.

    Parameters
    ----------
    encoder   : Keras Model, Input(T,1) → (batch, d_shared + n*d_target)
    decoders  : list of n_targets Keras Models, Input(d_sh+d_t) → (batch, 5)
    nn_bank   : list of n_targets float32 arrays (N_single_j, T) — reference curves
    T         : time steps (must match encoder input)
    d_shared  : shared latent dims
    d_target  : per-target latent dims
    n_targets : number of targets
    lambda_cons, lambda_anch, lambda_var, lambda_supcon : loss weights
    var_margin  : minimum acceptable std for per-target dims (default 0.05)
    supcon_temp : SupCon temperature (default 0.07)
    k           : k nearest neighbours for L_anchor
    """

    def __init__(self, encoder, decoders, nn_bank, T=45,
                 d_shared=16, d_target=10, n_targets=3,
                 lambda_cons=2.0, lambda_anch=0.3, lambda_var=0.05,
                 lambda_active=0.5, Fm_floor_frac=0.65,
                 var_margin=0.05, lambda_supcon=0.0, supcon_temp=0.07,
                 k=5, **kwargs):
        super().__init__(**kwargs)
        self.encoder       = encoder
        self.decoders      = decoders
        self.nn_bank       = [tf.constant(b, dtype=tf.float32) for b in nn_bank]
        self.T             = T
        self.d_shared      = d_shared
        self.d_target      = d_target
        self.n_targets     = n_targets
        self.lambda_cons   = lambda_cons
        self.lambda_anch   = lambda_anch
        self.lambda_var    = lambda_var
        self.lambda_active = lambda_active
        self.var_margin    = var_margin
        self.lambda_supcon = lambda_supcon
        self.supcon_temp   = supcon_temp
        self.k             = k
        # Per-target amplitude floor: 65% of mean single-target peak (self-calibrates to data)
        self.Fm_floor = tf.constant([
            Fm_floor_frac * float(tf.reduce_mean(tf.reduce_mean(b, axis=-1)))
            for b in self.nn_bank
        ], dtype=tf.float32)  # (n_targets,)

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

        # Clip rendered curves to prevent float32 overflow in gradient computation.
        # Valid PCR amplitudes are in [0, ~3]; ±10 is a safe generous bound.
        rendered = [tf.clip_by_value(r, -10.0, 10.0) for r in rendered]

        # L_absent: absent channels → zero rendered curve
        l_absent = tf.constant(0.0)
        for j in range(self.n_targets):
            absent = 1.0 - y_f[:, j]               # (batch,)
            sq     = tf.reduce_mean(rendered[j] ** 2, axis=-1)
            l_absent = l_absent + tf.reduce_mean(absent * sq)

        # L_active: active channels must reach a minimum peak amplitude (prevents
        # the degenerate solution where one decoder absorbs the full mixture signal
        # and other active decoders remain flat).
        l_active = tf.constant(0.0)
        for j in range(self.n_targets):
            mean_j   = tf.reduce_mean(rendered[j], axis=-1)       # (batch,) — stable gradient vs max
            l_active += tf.reduce_mean(
                y_f[:, j] * tf.nn.relu(self.Fm_floor[j] - mean_j))

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

        # L_var: penalise low per-target-dim variance (prevents dead-dim collapse).
        # Use sqrt(var + eps) instead of reduce_std to avoid 0/0 gradient when std=0.
        l_var = tf.constant(0.0)
        for j in range(self.n_targets):
            mean_j = tf.reduce_mean(z_parts[j], axis=0)
            var_j  = tf.reduce_mean((z_parts[j] - mean_j) ** 2, axis=0)
            z_std  = tf.sqrt(var_j + 1e-8)          # (d_target,) — numerically safe
            l_var  = l_var + tf.reduce_mean(tf.nn.relu(self.var_margin - z_std))

        # L_supcon: supervised contrastive on per-target latent z_j.
        # Pulls same-target latents together; no-op when lambda_supcon=0.
        l_supcon = tf.constant(0.0)
        if self.lambda_supcon > 0:
            y_int = tf.cast(y_bin, tf.int32)
            for j in range(self.n_targets):
                l_supcon = l_supcon + _per_target_supcon_loss(
                    z_parts[j], y_int[:, j], temp=self.supcon_temp)

        loss = (l_absent
                + self.lambda_active * l_active
                + self.lambda_cons   * l_consist
                + self.lambda_anch   * l_anchor
                + self.lambda_var    * l_var
                + self.lambda_supcon * l_supcon)
        return loss, l_absent, l_active, l_consist, l_anchor, l_var, l_supcon

    def call(self, x, training=False):
        _, rendered = self._encode_and_decode(x, training)
        return rendered

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'] if isinstance(y_dict, dict) else y_dict, tf.int32)
        with tf.GradientTape() as tape:
            loss, l_ab, l_ac, l_co, l_an, l_va, l_sc = self._compute_losses(x, y_bin, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'l_absent': l_ab, 'l_active': l_ac, 'l_consist': l_co,
                'l_anchor': l_an, 'l_var': l_va, 'l_supcon': l_sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'] if isinstance(y_dict, dict) else y_dict, tf.int32)
        loss, l_ab, l_ac, l_co, l_an, l_va, l_sc = self._compute_losses(x, y_bin, training=False)
        return {'loss': loss, 'l_absent': l_ab, 'l_active': l_ac, 'l_consist': l_co,
                'l_anchor': l_an, 'l_var': l_va, 'l_supcon': l_sc}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 SC1/2/3 extensions (Family B pretraining)
# ──────────────────────────────────────────────────────────────────────────────

# Mirrors SUPCON_TEMP / SUPCON_LAMBDA from model_utils_supcon; defined locally
# to avoid circular import (model_utils_multilabel imports this module).
_SC_TEMP       = 0.1   # SupCon temperature
_SC_LAMBDA_SC1 = 0.2   # SC1: single head weight
_SC_LAMBDA_SCN = 0.1   # SC2/SC3: per-head weight


def _jaccard_pm(y_bin):
    """Pairwise Jaccard pos mask (diagonal = 0) from int binary tensor."""
    y     = tf.cast(y_bin, tf.float32)
    inter = tf.matmul(y, y, transpose_b=True)
    rs    = tf.reduce_sum(y, axis=1, keepdims=True)
    union = rs + tf.transpose(rs) - inter
    return (inter / (union + 1e-8)) * (1.0 - tf.eye(tf.shape(y)[0]))


def _sc_loss_p1(proj_l2, pm):
    """Jaccard-weighted SupCon on L2-normalised embeddings with given pos mask."""
    not_self  = 1.0 - tf.eye(tf.shape(proj_l2)[0])
    sim       = tf.matmul(proj_l2, proj_l2, transpose_b=True) / _SC_TEMP
    sim_max   = tf.stop_gradient(tf.reduce_max(sim, axis=1, keepdims=True))
    exp_sim   = tf.exp(sim - sim_max)
    log_denom = tf.math.log(tf.reduce_sum(exp_sim * not_self, axis=1, keepdims=True) + 1e-8)
    log_prob  = (sim - sim_max) - log_denom
    pos_sum   = tf.reduce_sum(pm, axis=1)
    has_pos   = tf.cast(pos_sum > 0, tf.float32)
    per_anc   = -tf.reduce_sum(log_prob * pm, axis=1) / (pos_sum + 1e-8)
    return tf.reduce_mean(per_anc * has_pos)


def _proj_l2(dense, x, training):
    """Dense(64, relu) → L2-normalise. dense is a pre-created tf.keras.layers.Dense."""
    return tf.math.l2_normalize(dense(x, training=training), axis=-1)


def _make_branch_extractor(encoder):
    """Sub-model that returns (z_cnn, z_gru) from the dual-branch encoder.

    z_cnn : (batch, 64)  — CNN branch output (ss_cnn_emb layer)
    z_gru : (batch, 128) — GRU branch output (ss_bigru layer, BiGRU(64)→128)
    Shares all weights with encoder; used by Phase 1 SC2/SC3 projection heads.
    """
    return tf.keras.Model(
        inputs=encoder.input,
        outputs=[encoder.get_layer('ss_cnn_emb').output,
                 encoder.get_layer('ss_bigru').output],
        name='ss_branch_extractor',
    )


class MultiLabelSourceSepPhase1SC1Model(MultiLabelSourceSepPhase1Model):
    """Phase 1 + SC1: L_recon + 0.2 * SC(proj_full) — projects fused bottleneck z."""

    def __init__(self, *args, **kwargs):
        kwargs.pop('lambda_supcon', None)
        super().__init__(*args, lambda_supcon=0.0, **kwargs)
        self._pf = tf.keras.layers.Dense(64, activation='relu', name='p1sc1_pf')

    def _compute_losses(self, x, y_bin, training):
        loss, l_ab, l_ac, l_co, l_an, l_va, _ = super()._compute_losses(x, y_bin, training)
        z_full = self.encoder(x, training=training)
        proj_f = _proj_l2(self._pf, z_full, training)
        l_sc   = _sc_loss_p1(proj_f, _jaccard_pm(y_bin))
        return loss + _SC_LAMBDA_SC1 * l_sc, l_ab, l_ac, l_co, l_an, l_va, l_sc


class MultiLabelSourceSepPhase1SC2Model(MultiLabelSourceSepPhase1Model):
    """Phase 1 + SC2: L_recon + 0.1*(SC(proj_cnn) + SC(proj_gru)).

    Projects from the CNN branch (z_cnn, 64-dim) and GRU branch (z_gru, 128-dim)
    separately — not from arbitrary bottleneck slices.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop('lambda_supcon', None)
        super().__init__(*args, lambda_supcon=0.0, **kwargs)
        self._pc = tf.keras.layers.Dense(64, activation='relu', name='p1sc2_pc')
        self._pg = tf.keras.layers.Dense(64, activation='relu', name='p1sc2_pg')
        self._branch_ext = _make_branch_extractor(self.encoder)

    def _compute_losses(self, x, y_bin, training):
        loss, l_ab, l_ac, l_co, l_an, l_va, _ = super()._compute_losses(x, y_bin, training)
        z_cnn, z_gru = self._branch_ext(x, training=training)
        pm     = _jaccard_pm(y_bin)
        proj_c = _proj_l2(self._pc, z_cnn, training)
        proj_g = _proj_l2(self._pg, z_gru, training)
        l_sc   = _sc_loss_p1(proj_c, pm) + _sc_loss_p1(proj_g, pm)
        return loss + _SC_LAMBDA_SCN * l_sc, l_ab, l_ac, l_co, l_an, l_va, l_sc


class MultiLabelSourceSepPhase1SC3Model(MultiLabelSourceSepPhase1Model):
    """Phase 1 + SC3: L_recon + 0.1*(SC(proj_cnn) + SC(proj_gru) + SC(proj_full)).

    Projects from CNN branch (z_cnn), GRU branch (z_gru), and fused bottleneck z_full.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop('lambda_supcon', None)
        super().__init__(*args, lambda_supcon=0.0, **kwargs)
        self._pc = tf.keras.layers.Dense(64, activation='relu', name='p1sc3_pc')
        self._pg = tf.keras.layers.Dense(64, activation='relu', name='p1sc3_pg')
        self._pf = tf.keras.layers.Dense(64, activation='relu', name='p1sc3_pf')
        enc = self.encoder
        self._branch_ext = tf.keras.Model(
            inputs=enc.input,
            outputs=[enc.get_layer('ss_cnn_emb').output,
                     enc.get_layer('ss_bigru').output,
                     enc.output],
            name='ss_branch_extractor',
        )

    def _compute_losses(self, x, y_bin, training):
        loss, l_ab, l_ac, l_co, l_an, l_va, _ = super()._compute_losses(x, y_bin, training)
        z_cnn, z_gru, z_full = self._branch_ext(x, training=training)
        pm     = _jaccard_pm(y_bin)
        proj_c = _proj_l2(self._pc, z_cnn, training)
        proj_g = _proj_l2(self._pg, z_gru, training)
        proj_f = _proj_l2(self._pf, z_full, training)
        l_sc   = _sc_loss_p1(proj_c, pm) + _sc_loss_p1(proj_g, pm) + _sc_loss_p1(proj_f, pm)
        return loss + _SC_LAMBDA_SCN * l_sc, l_ab, l_ac, l_co, l_an, l_va, l_sc


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
