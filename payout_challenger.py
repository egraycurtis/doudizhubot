"""Transfer-initialized, stake-aware challenger models.

The existing calibrated win head and every trunk tensor are cloned exactly.  A
small new payout branch consumes the transferred penultimate representation plus
explicit stake/team context.  Production keeps loading its original one-output
models unchanged.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def create_payout_challenger(source: tf.keras.Model) -> tf.keras.Model:
    cloned = tf.keras.models.clone_model(source)
    cloned.set_weights(source.get_weights())
    if not all(np.array_equal(before, after) for before, after in zip(source.get_weights(), cloned.get_weights())):
        raise RuntimeError("source weights did not transfer exactly")
    stake_context = tf.keras.Input(shape=(4,), name="stake_context")
    transferred_trunk = cloned.layers[-2].output
    value_features = layers.Concatenate(name="payout_features")([transferred_trunk, stake_context])
    value_features = layers.Dense(32, activation="relu", name="payout_hidden")(value_features)
    payout = layers.Dense(1, activation="tanh", name="expected_payout")(value_features)
    # The original output layer has an auto-generated name; this identity makes
    # the public multi-head contract stable without changing calibrated values.
    win_probability = layers.Activation("linear", name="win_probability")(cloned.output)
    model = tf.keras.Model(cloned.inputs + [stake_context], {"win_probability": win_probability, "expected_payout": payout}, name="stake_aware_challenger")
    model.compile(optimizer="adam", loss={"win_probability": "mean_squared_error", "expected_payout": "mean_squared_error"}, loss_weights={"win_probability": 1.0, "expected_payout": 0.5})
    return model


def create_challenger_family(destination_model_name: str, source_model_name: str, source_version: int, source_paths, source_hashes) -> int:
    from model_registry import get_checkpoint_path, load_models, _atomic_save_model, _write_initial_metadata

    sources = load_models(source_model_name, compile_model=False, version=source_version)
    challengers = [create_payout_challenger(source) for source in sources]
    paths = [get_checkpoint_path(destination_model_name, position, 0) for position in range(3)]
    for challenger, path in zip(challengers, paths):
        _atomic_save_model(challenger, path)
        challenger.save_weights(path.with_suffix(".weights.h5"))
    _write_initial_metadata(destination_model_name, 0, source_model_name, source_version, source_hashes, paths)
    return 0
