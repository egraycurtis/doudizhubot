"""Golden prediction smoke test for Mac, WSL2, and Linux training hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_registry import get_checkpoint_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_inputs():
    generator = np.random.default_rng(20260809)
    return [generator.random((2, *shape), dtype=np.float32) for shape in ((85,), (85,), (54,), (54,), (54,), (54,), (54,), (2,), (5,), (5,), (15, 54))]


def collect_predictions():
    inputs = fixed_inputs()
    models = []
    for position in range(3):
        path = get_checkpoint_path("transformer", position)
        model = tf.keras.models.load_model(path, compile=False)
        prediction = np.asarray(model(inputs, training=False)).reshape(-1)
        if not np.isfinite(prediction).all():
            raise AssertionError(f"non-finite prediction from production model position {position}")
        prediction = prediction.tolist()
        models.append({"position": position, "path": str(path), "sha256": _sha256(path), "prediction": prediction})
    return {"schema_version": 1, "tolerance": 1e-5, "models": models}


def preflight_production_models():
    """Load every protected model and exercise the exact fixed input schema."""
    result = collect_predictions()
    for model in result["models"]:
        values = model["prediction"]
        print(f"transformer{model['position']}: loaded; finite predictions in [{min(values):.7f}, {max(values):.7f}]; SHA-256 {model['sha256']}")
    return result


def compare(expected, actual):
    tolerance = float(expected.get("tolerance", 1e-5))
    for before, after in zip(expected["models"], actual["models"]):
        if before["sha256"] != after["sha256"]:
            raise AssertionError(f"model hash changed for position {before['position']}")
        if not np.allclose(before["prediction"], after["prediction"], rtol=tolerance, atol=tolerance):
            raise AssertionError(f"prediction drift exceeds {tolerance} for position {before['position']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("experiments/portability-golden.json"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="load all production models and run fixed-schema inference")
    args = parser.parse_args()
    if args.preflight:
        preflight_production_models()
        return
    actual = collect_predictions()
    if args.write:
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        args.golden.write_text(json.dumps(actual, indent=2, sort_keys=True))
        print(f"wrote {args.golden}")
    else:
        compare(json.loads(args.golden.read_text()), actual)
        print("portability smoke passed")


if __name__ == "__main__":
    main()
