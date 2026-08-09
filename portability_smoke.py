"""Golden prediction smoke test for Mac, WSL2, and Linux training hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_registry import get_checkpoint_path
from cards import empty_card_dict, landlord_first_shuffle
from self_play import build_turn_tensors, get_move_options_with_ids, get_previous_turn_info, remove_choice_from_hand
from turn_info import get_turn_info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_inputs():
    """Legal, seeded candidate tensors covering seats and non-empty history."""
    import random
    random.seed(20260809)
    hands = landlord_first_shuffle()
    game = {"turns": [], "cards_played_by_hands": [empty_card_dict(), empty_card_dict(), empty_card_dict()], "stake_multiplier": 1}
    examples = []
    for position in range(3):
        options = get_move_options_with_ids(get_previous_turn_info(game["turns"]), hands[position])
        for _, choice in (options[0], options[-1]):
            examples.append(build_turn_tensors(game, position, hands[position], choice, "transformer"))
        action_id, choice = options[0]
        game["turns"].append({"turn_info": get_turn_info(choice), "action_id": action_id, "position": position})
        for card, count in choice.items():
            game["cards_played_by_hands"][position][card] += count
        remove_choice_from_hand(hands[position], choice)
    keys = ("cards_not_seen_additional_features_tensor", "cards_remaining_additional_feature_tensor", "cards_not_seen_tensor", "cards_person_on_right_has_played_tensor", "cards_person_on_left_has_played_tensor", "choice_tensor", "cards_remaining_tensor", "last_played_tensor", "cards_person_on_left_has_left_tensor", "cards_person_on_right_has_left_tensor", "transformer_tensor")
    return [np.asarray([example[key].reshape(shape) for example in examples], dtype=np.float32) for key, shape in zip(keys, ((85,), (85,), (54,), (54,), (54,), (54,), (54,), (2,), (5,), (5,), (15, 54)))]


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
    return {"schema_version": 2, "probe_version": "legal-state-v1", "tolerance": 1e-5, "models": models}


def preflight_production_models():
    """Load every protected model and exercise the exact fixed input schema."""
    result = collect_predictions()
    for model in result["models"]:
        values = model["prediction"]
        if max(values) - min(values) < 1e-6 or all(value >= 0.999 for value in values) or all(value <= 0.001 for value in values):
            raise AssertionError(f"non-discriminating portability probe for position {model['position']}")
        print(f"transformer{model['position']}: loaded; finite predictions in [{min(values):.7f}, {max(values):.7f}]; SHA-256 {model['sha256']}")
    return result


def compare(expected, actual):
    for record in (expected, actual):
        if record.get("schema_version") != 2 or record.get("probe_version") != "legal-state-v1" or len(record.get("models", [])) != 3:
            raise AssertionError("golden record has an invalid model/schema probe contract")
    tolerance = float(expected.get("tolerance", 1e-5))
    for position, (before, after) in enumerate(zip(expected["models"], actual["models"])):
        if before.get("position") != position or after.get("position") != position:
            raise AssertionError("golden record positions are missing or reordered")
        if before["sha256"] != after["sha256"]:
            raise AssertionError(f"model hash changed for position {before['position']}")
        if np.asarray(before["prediction"]).shape != np.asarray(after["prediction"]).shape:
            raise AssertionError(f"prediction shape changed for position {position}")
        if np.all(np.asarray(after["prediction"]) >= 0.999) or np.all(np.asarray(after["prediction"]) <= 0.001):
            raise AssertionError(f"saturated portability probe for position {position}")
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
