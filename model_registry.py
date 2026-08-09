"""Safe model-family registry for production baselines and training experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Input, layers, models


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
PRODUCTION_MODEL_NAME = "transformer"
EXPERIMENT_MODEL_NAMES = {"transformer_v2", "transformer_payout_v1", "smoke_transformer_v2", "smoke_transformer_payout_v1"}


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "transformer"
    uses_sequence_history: bool = True
    uses_stake_context: bool = False
    use_payout_head: bool = False
    target_mix: float = 0.1
    checkpoint_interval_batches: int = 10
    keep_checkpoint_versions: int = 4
    schema_version: int = 1


MODEL_CONFIGS = {
    "transformer": ModelConfig(checkpoint_interval_batches=0, keep_checkpoint_versions=0),
    "transformer_v2": ModelConfig(),
    "transformer_payout_v1": ModelConfig(uses_stake_context=True, use_payout_head=False, schema_version=2),
    "smoke_transformer_v2": ModelConfig(),
    "smoke_transformer_payout_v1": ModelConfig(uses_stake_context=True, use_payout_head=False, schema_version=2),
}


def get_model_config(model_name: str) -> ModelConfig:
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    raise ValueError(f"unknown model family {model_name!r}; experiments must be explicitly registered")


def get_model_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name


def get_metadata_path(model_name: str) -> Path:
    return get_model_dir(model_name) / "metadata.json"


def _metadata(model_name: str) -> dict:
    path = get_metadata_path(model_name)
    return json.loads(path.read_text()) if path.exists() else {"version": 0}


def get_latest_checkpoint_version(model_name: str) -> int:
    if model_name == PRODUCTION_MODEL_NAME:
        return 0
    return int(_metadata(model_name).get("version", 0))


def archive_destination(path: Path) -> Path:
    """Return a collision-safe archive name without mutating either path."""
    candidate, suffix = path.with_name(f"{path.name}.archived"), 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.archived-{suffix}")
        suffix += 1
    return candidate


def get_checkpoint_path(model_name: str, position: int, version: int | None = None) -> Path:
    if model_name == PRODUCTION_MODEL_NAME:
        return get_model_dir(model_name) / f"transformer{position}.keras"
    version = get_latest_checkpoint_version(model_name) if version is None else version
    return get_model_dir(model_name) / f"{model_name}{position}_v{version:06d}.keras"


def load_models(model_name: str, compile_model: bool = False, version: int | None = None) -> list[tf.keras.Model]:
    # Metadata is atomically published but can advance between loads. Pin one
    # snapshot for the complete three-role family.
    if version is None and model_name != PRODUCTION_MODEL_NAME:
        version = get_latest_checkpoint_version(model_name)
    return [tf.keras.models.load_model(get_checkpoint_path(model_name, position, version), compile=compile_model) for position in range(3)]


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_save_model(model: tf.keras.Model, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.keras")
    model.save(temporary)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_model_hashes() -> dict[str, str]:
    """Read-only portability check for the protected serving checkpoints."""
    return {
        f"transformer{position}": _file_sha256(get_checkpoint_path(PRODUCTION_MODEL_NAME, position))
        for position in range(3)
    }


def validate_experiment_family(model_name: str, source_model: str | None = None) -> dict:
    """Validate one complete, immutable checkpoint snapshot before resuming."""
    _assert_experiment(model_name)
    directory, metadata_path = get_model_dir(model_name), get_metadata_path(model_name)
    if not directory.is_dir() or not metadata_path.is_file():
        raise ValueError(f"cannot resume: experimental family {model_name!r} is missing; recover its directory or start a new run")
    metadata = _metadata(model_name)
    config = get_model_config(model_name)
    if metadata.get("model_name") != model_name or metadata.get("schema_version") != config.schema_version:
        raise ValueError("cannot resume: experimental model metadata has an incompatible schema")
    if source_model is not None and metadata.get("source_model") != source_model:
        raise ValueError("cannot resume: experimental model source does not match the run")
    version, names, hashes = metadata.get("version"), metadata.get("checkpoints"), metadata.get("sha256")
    if not isinstance(version, int) or not isinstance(names, list) or not isinstance(hashes, list) or len(names) != 3 or len(hashes) != 3:
        raise ValueError("cannot resume: experimental model metadata is incomplete")
    for position, (name, expected_hash) in enumerate(zip(names, hashes)):
        path = get_checkpoint_path(model_name, position, version)
        if name != path.name or not path.is_file():
            raise ValueError(f"cannot resume: checkpoint for role {position} at version {version} is missing")
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"cannot resume: checkpoint hash mismatch for role {position}; restore the matching family")
    return metadata


def _assert_experiment(model_name: str) -> None:
    if model_name == PRODUCTION_MODEL_NAME or model_name not in EXPERIMENT_MODEL_NAMES:
        raise ValueError(f"refusing to write model family {model_name!r}; production models are read-only")


def save_models(model_name: str, models_to_save: list[tf.keras.Model], version: int | None = None, source_model: str | None = None) -> int:
    _assert_experiment(model_name)
    config = get_model_config(model_name)
    directory = get_model_dir(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    version = get_latest_checkpoint_version(model_name) + 1 if version is None else version
    paths = [get_checkpoint_path(model_name, position, version) for position in range(3)]
    for model, path in zip(models_to_save, paths):
        _atomic_save_model(model, path)
        model.save_weights(path.with_suffix(".weights.h5"))
    metadata = {
        "version": version,
        "model_name": model_name,
        "schema_version": config.schema_version,
        "source_model": source_model,
        "checkpoints": [path.name for path in paths],
        "sha256": [_file_sha256(path) for path in paths],
        "config": asdict(config),
    }
    _atomic_write_text(get_metadata_path(model_name), json.dumps(metadata, indent=2, sort_keys=True))
    keep_from = version - config.keep_checkpoint_versions + 1
    for path in directory.glob(f"{model_name}[0-2]_v*.keras"):
        old_version = int(path.stem.rsplit("_v", 1)[1])
        if old_version < keep_from:
            path.unlink(missing_ok=True)
            path.with_suffix(".weights.h5").unlink(missing_ok=True)
    return version


def _copy_source_family(destination: str, source: str) -> int:
    paths = []
    for position in range(3):
        source_path = get_checkpoint_path(source, position)
        destination_path = get_checkpoint_path(destination, position, 0)
        shutil.copy2(source_path, destination_path)
        tf.keras.models.load_model(destination_path, compile=False).save_weights(destination_path.with_suffix(".weights.h5"))
        paths.append(destination_path)
    config = get_model_config(destination)
    _atomic_write_text(get_metadata_path(destination), json.dumps({"version": 0, "model_name": destination, "schema_version": config.schema_version, "source_model": source, "checkpoints": [path.name for path in paths], "sha256": [_file_sha256(path) for path in paths], "config": asdict(config)}, indent=2, sort_keys=True))
    return 0


def initialize_model_family(model_name: str, source_model_name: str = PRODUCTION_MODEL_NAME, force_reset: bool = False) -> int:
    _assert_experiment(model_name)
    directory = get_model_dir(model_name)
    if directory.exists() and get_metadata_path(model_name).exists() and not force_reset:
        return get_latest_checkpoint_version(model_name)
    if directory.exists() and force_reset:
        directory.rename(archive_destination(directory))
    directory.mkdir(parents=True, exist_ok=True)
    if model_name.endswith("payout_v1"):
        from payout_challenger import create_challenger_family

        return create_challenger_family(model_name, source_model_name)
    return _copy_source_family(model_name, source_model_name)


def transformer_block(x, num_heads: int, ff_dim: int):
    normalized = layers.LayerNormalization(epsilon=1e-6)(x)
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(normalized, normalized)
    residual = layers.Add()([x, attention])
    normalized = layers.LayerNormalization(epsilon=1e-6)(residual)
    feed_forward = layers.Dense(ff_dim, activation="relu")(normalized)
    return layers.Add()([normalized, layers.Dense(x.shape[-1])(feed_forward)])


def create_transformer_model() -> tf.keras.Model:
    inputs = [Input(shape=shape) for shape in ((85,), (85,), (54,), (54,), (54,), (54,), (54,), (2,), (5,), (5,), (15, 54))]
    history = layers.Flatten()(transformer_block(inputs[-1], num_heads=2, ff_dim=64))
    merged = layers.Concatenate()(inputs[:-1] + [history])
    hidden = merged
    for width in (512, 256, 128, 64):
        hidden = layers.Dense(width, activation="relu")(hidden)
    output = layers.Dense(1, activation="sigmoid", name="win_probability")(hidden)
    model = models.Model(inputs, output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model
