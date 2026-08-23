import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="model_utils_gnn_recon")
class _QuerySlice(tf.keras.layers.Layer):
    def call(self, x):
        return x[:, 0:1, :]


@tf.keras.utils.register_keras_serializable(package="model_utils_gnn_recon")
class _AttnScores(tf.keras.layers.Layer):
    def __init__(self, attn_dim, **kwargs):
        super().__init__(**kwargs)
        self.attn_dim = attn_dim

    def call(self, inputs):
        q, k = inputs
        return tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.attn_dim, tf.float32))

    def get_config(self):
        return {**super().get_config(), "attn_dim": self.attn_dim}


@tf.keras.utils.register_keras_serializable(package="model_utils_gnn_recon")
class _WeightedRecon(tf.keras.layers.Layer):
    def call(self, inputs):
        weights, stack = inputs
        return tf.matmul(weights, stack)


def create_cnn_gru_dual_gat_recon_model(k_plus_1, input_size_curve, output_size, branch_builder):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, input_size_curve), name="neighbor_stack_input")

    shared_proj = tf.keras.layers.Dense(input_size_curve, activation=None, name="gat_shared_proj")
    Wh = tf.keras.layers.TimeDistributed(shared_proj, name="Wh")(stack_input)      # (N, k+1, T)

    query = _QuerySlice()(Wh)                                                      # (N, 1, T)
    scores = _AttnScores(input_size_curve)([query, Wh])                           # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name="gat_attn_weights")(scores)  # (N, 1, k+1)

    reconstructed = _WeightedRecon()([attn_weights, Wh])                          # (N, 1, T)
    reconstructed = tf.keras.layers.Reshape(
        (input_size_curve, 1), name="reconstructed_curve")(reconstructed)         # (N, T, 1)

    z = branch_builder(reconstructed)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.models.Model(inputs=stack_input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
