import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def inception_smoothing_block(inputs, filters=8, kernel_sizes=(3, 7, 15, 31)):
    branches = [
        tf.keras.layers.Conv1D(filters, k, padding='same', activation='relu',
                               name=f'inc_k{k}')(inputs)
        for k in kernel_sizes
    ]
    merged = tf.keras.layers.Concatenate(name='inc_merge')(branches)
    return tf.keras.layers.Conv1D(filters, 1, padding='same', activation='relu',
                                  name='inc_bottleneck')(merged)
