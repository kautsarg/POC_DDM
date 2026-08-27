import numpy as np
import tensorflow as tf

_SPATIAL_MODEL_HINT = "attn_recon"


def is_spatial_model(model_key):
    return _SPATIAL_MODEL_HINT in model_key


def cls_layer(model):
    try:
        return model.get_layer('cls_out')
    except ValueError:
        return model.layers[-1]


def compute_embeddings(model, curves):
    embed_model = tf.keras.Model(inputs=model.input, outputs=cls_layer(model).input)
    return embed_model.predict(curves[..., None].astype(np.float32), verbose=0)


def compute_embeddings_stack(model, stack):
    embed_model = tf.keras.Model(inputs=model.input, outputs=cls_layer(model).input)
    return embed_model.predict(stack.astype(np.float32), verbose=0)


def infer_k(model):
    return model.input_shape[1] - 1


def pc_mean_stack(pc_curves, k):
    mean_curve = pc_curves.mean(axis=0)
    return np.tile(mean_curve, (k + 1, 1))[None, ...]


def recentered_predict(model, model_key, X_test, ref_embed, new_chip_embed):
    is_spatial = is_spatial_model(model_key)
    embeddings = (compute_embeddings_stack(model, X_test) if is_spatial
                  else compute_embeddings(model, X_test))
    shift = ref_embed - new_chip_embed
    probs = cls_layer(model)(embeddings + shift).numpy()
    return probs, float(np.linalg.norm(shift))
