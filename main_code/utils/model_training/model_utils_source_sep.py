import tensorflow as tf


def build_serial_bigger_encoder(T, d_shared, d_target, n_targets=3):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='sb_input')
    x = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='causal', name='sb_conv1')(inputs)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='causal', strides=2, name='sb_conv2')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64), name='sb_bigru')(x)
    x = tf.keras.layers.LayerNormalization(name='sb_ln')(x)
    z = tf.keras.layers.Dense(d_shared + n_targets * d_target, name='sb_bottleneck')(x)
    return tf.keras.Model(inputs=inputs, outputs=z, name='serial_bigger_encoder')


def build_parametric_decoder(d_shared, d_target, T_max=45, j=0):
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
