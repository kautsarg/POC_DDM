import tensorflow as tf

from model_utils_mtl import (_build_cnn_gru_dual_branches_mtl,
                              _build_cnn_gru_dual_attn_recon_embedding_mtl)


@tf.keras.utils.register_keras_serializable(package="model_utils_dann")
class GradientReversalLayer(tf.keras.layers.Layer):
    """Forward = identity, backward = -lambda_ * gradient (Ganin & Lempitsky 2016)."""
    def __init__(self, lambda_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = tf.Variable(lambda_init, trainable=False, dtype=tf.float32, name="grl_lambda")

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -tf.cast(self.lambda_, dy.dtype) * dy
            return x, grad
        return _reverse(x)


class GRLLambdaSchedule(tf.keras.callbacks.Callback):
    """lambda_p = lambda_max * (2/(1+exp(-10p)) - 1), p = step/total_steps."""
    def __init__(self, grl_layer, total_steps, lambda_max=1.0):
        super().__init__()
        self.grl_layer = grl_layer
        self.total_steps = max(total_steps, 1)
        self.lambda_max = lambda_max
        self._step = 0

    def on_train_batch_begin(self, batch, logs=None):
        import numpy as np
        p = self._step / self.total_steps
        lambda_p = self.lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        self.grl_layer.lambda_.assign(lambda_p)
        self._step += 1


def _dann_wrap(inputs, embedding, n_classes, n_chips, grl_lambda_init=0.0):
    cls_feat = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)

    grl = GradientReversalLayer(lambda_init=grl_lambda_init, name='grl')
    chip_feat = tf.keras.layers.Dense(16, activation='relu', name='chip_feat')(grl(embedding))
    chip_out  = tf.keras.layers.Dense(n_chips, activation='softmax', name='chip_out')(chip_feat)

    model = DANNModel(inputs=inputs, outputs=[cls_out, chip_out])
    model.grl = grl
    return model


@tf.keras.utils.register_keras_serializable(package='model_utils_dann')
class DANNModel(tf.keras.Model):
    """CE + dann_domain_weight*chip_CE, fixed weight (not learned UW-SO -- adversarial,
    not cooperative). Monitor val_cls_ce, not val_loss (chip_ce is meant to plateau
    near chance, not decrease)."""
    def __init__(self, *args, dann_domain_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dann_domain_weight = dann_domain_weight

    def _compute_loss(self, y_cls, y_chip, cls_out, chip_out):
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        ce_chip = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_chip, chip_out))
        loss = ce + self.dann_domain_weight * ce_chip
        return loss, ce, ce_chip

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, chip_out = self(x, training=True)
            loss, ce, ce_chip = self._compute_loss(y_dict['cls_out'], y_dict['chip_out'], cls_out, chip_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'chip_ce': ce_chip, 'grl_lambda': self.grl.lambda_})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, chip_out = self(x, training=False)
        loss, ce, ce_chip = self._compute_loss(y_dict['cls_out'], y_dict['chip_out'], cls_out, chip_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'chip_ce': ce_chip, 'grl_lambda': self.grl.lambda_})


def create_cnn_gru_dual_dann_model(T, n_classes, n_chips):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _dann_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes, n_chips)


def create_cnn_gru_dual_attn_recon_dann_model(k_plus_1, T, n_classes, n_chips, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    embedding = _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim)
    return _dann_wrap(stack_input, embedding, n_classes, n_chips)


DANN_MODEL_KEYS = (
    'cnn_gru_dual_dann', 'cnn_gru_dual_attn_recon_dann',
)
