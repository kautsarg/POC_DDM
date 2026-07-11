import tensorflow as tf

def create_cnn_gru_dual(input_size, n_classes):
    # CNN GRU dual model

    inp = tf.keras.layers.Input(shape=(input_size, 1), name='curve_input')

    # Local branch: CNN
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(inp)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    # Global branch: GRU
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inp)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(32, activation='relu')(g)

    # Fusion (dual model)
    z = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(z)
    z = tf.keras.layers.Dropout(0.2)(z)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(z)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
