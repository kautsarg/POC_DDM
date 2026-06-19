import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# MODEL ARCHITECTURES (Classification)
# ==========================================

def build_cnn_model(input_size, output_size):
    inputs = layers.Input(shape=(input_size, 1))
    x = layers.Conv1D(16, 5, activation='relu')(inputs)
    x = layers.Conv1D(8, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_size, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_gru_model(input_size, output_size):
    inputs = layers.Input(shape=(input_size, 1))
    x = layers.Bidirectional(layers.GRU(32, return_sequences=True))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.GRU(16))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_size, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_gru_dual_model(input_size, output_size):
    inputs = layers.Input(shape=(input_size, 1))

    # Local feature branch (CNN)
    c = layers.Conv1D(16, 5, activation='relu')(inputs)
    c = layers.Conv1D(8, 3, activation='relu')(c)
    c = layers.Flatten()(c)
    cnn_emb = layers.Dense(32, activation='relu')(c)

    # Global feature branch (BiGRU)
    g = layers.Bidirectional(layers.GRU(32, return_sequences=True))(inputs)
    g = layers.LayerNormalization()(g)
    g = layers.Bidirectional(layers.GRU(16))(g)
    g = layers.Dropout(0.2)(g)
    gru_emb = layers.Dense(32, activation='relu')(g)

    merged = layers.Concatenate()([cnn_emb, gru_emb])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.2)(z)
    outputs = layers.Dense(output_size, activation='softmax')(z)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_transformer_model(input_size, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = layers.Input(shape=(input_size, 1))

    x = layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    seq_len = x.shape[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=head_size)(positions)
    x = x + pos_embedding

    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn = layers.Dropout(dropout)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(head_size)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_size, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_transformer_dual_model(input_size, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = layers.Input(shape=(input_size, 1))

    # Local feature branch (CNN)
    c = layers.Conv1D(16, 5, activation='relu')(inputs)
    c = layers.Conv1D(8, 3, activation='relu')(c)
    c = layers.Flatten()(c)
    cnn_emb = layers.Dense(32, activation='relu')(c)

    # Global feature branch (Transformer)
    t = layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(inputs)
    t = layers.MaxPooling1D(pool_size=2, padding="same")(t)

    seq_len = t.shape[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=head_size)(positions)
    t = t + pos_embedding

    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(t, t)
        attn = layers.Dropout(dropout)(attn)
        t = layers.LayerNormalization(epsilon=1e-6)(t + attn)

        ffn = layers.Dense(ff_dim, activation="relu")(t)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(head_size)(ffn)
        t = layers.LayerNormalization(epsilon=1e-6)(t + ffn)

    t = layers.GlobalAveragePooling1D()(t)
    trans_emb = layers.Dense(32, activation="relu")(t)

    merged = layers.Concatenate()([cnn_emb, trans_emb])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.2)(z)
    outputs = layers.Dense(output_size, activation='softmax')(z)

    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ==========================================
# MODEL REGISTRY
# ==========================================
# Maps model name and number of training epochs
MODEL_BUILDERS = {
    "cnn": (build_cnn_model, 1000),
    "gru": (build_gru_model, 500),
    "cnn_gru_dual": (build_cnn_gru_dual_model, 500),
    "transformer": (build_transformer_model, 500),
    "cnn_transformer_dual": (build_cnn_transformer_dual_model, 500),
}


def train_and_evaluate(model_name, X_train, y_train, X_test, y_test, n_classes, batch_size=512):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(MODEL_BUILDERS)}")

    builder, epochs = MODEL_BUILDERS[model_name]
    tf.keras.backend.clear_session()
    model = builder(X_train.shape[1], n_classes)

    start = time.time()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=[early_stop])
    duration = time.time() - start

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == y_test))

    return accuracy, duration
