"""Correct, exhaustive self-play primitives shared by training and inference.

Only experimental model families are written by this module.  Production model
files are loaded read-only and remain the source of truth for serving.
"""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import random
import time
from typing import Any

import numpy as np
import redis
import tensorflow as tf

from action_space import action_space
from cards import empty_card_dict, full_card_dict, landlord_first_shuffle, rank
from filtered_options import filtered_options
from model_registry import get_latest_checkpoint_version, get_model_config, load_models
from training_codec import encode_training_batch, hand_to_string
from turn_info import choice_bomb_multiplier, expected_value, get_turn_info


BASE_FEATURE_KEYS = [
    "cards_not_seen_additional_features_tensor",
    "cards_remaining_additional_feature_tensor",
    "cards_not_seen_tensor",
    "cards_person_on_right_has_played_tensor",
    "cards_person_on_left_has_played_tensor",
    "choice_tensor",
    "cards_remaining_tensor",
    "last_played_tensor",
    "cards_person_on_left_has_left_tensor",
    "cards_person_on_right_has_left_tensor",
]
TRANSFORMER_FEATURE_KEY = "transformer_tensor"
CARD_KEYS = tuple(empty_card_dict())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_turn_info(turn_info):
    if isinstance(turn_info, str):
        turn_info = json.loads(turn_info)
    return {"type": turn_info.get("type", "pass"), "size": turn_info.get("size", 0), "rank": turn_info.get("rank", 0)}


def _get_turn_info(turn):
    if isinstance(turn, dict):
        return _normalize_turn_info(turn.get("turn_info", turn))
    mapping = getattr(turn, "_mapping", None)
    if mapping is not None:
        return _normalize_turn_info(mapping.get("turn_info", mapping))
    if hasattr(turn, "turn_info"):
        return _normalize_turn_info(turn.turn_info)
    return {"type": "pass", "size": 0, "rank": 0}


def get_previous_turn_info(turns):
    for turn in reversed(turns[-2:]):
        info = _get_turn_info(turn)
        if info["type"] != "pass":
            return info
    return {"type": "pass", "size": 0, "rank": 0}


def get_previous_played(turns):
    if turns and _get_turn_info(turns[-1])["type"] != "pass":
        return -1
    if len(turns) > 1 and _get_turn_info(turns[-2])["type"] != "pass":
        return -2
    return 0


def string_to_card_dict(action: str):
    result = empty_card_dict()
    for card in action:
        result[card] += 1
    return result


ACTION_STRINGS = list(action_space) + ([""] if "" not in action_space else [])
ACTION_ID_BY_STRING = {action: index for index, action in enumerate(ACTION_STRINGS)}
ACTION_CARD_DICTS = [string_to_card_dict(action) for action in ACTION_STRINGS]
ACTION_COUNT_MATRIX = np.asarray([[option[card] for card in CARD_KEYS] for option in ACTION_CARD_DICTS], dtype=np.int8)
ACTION_INFOS = [get_turn_info(option) for option in ACTION_CARD_DICTS]
PASS_ACTION_ID = ACTION_ID_BY_STRING[""]
ROCKET_ACTION_ID = ACTION_ID_BY_STRING["BR"]
OPEN_ACTION_IDS = tuple(range(len(action_space)))
ACTION_IS_BOMB = np.asarray([choice_bomb_multiplier(option) > 1 for option in ACTION_CARD_DICTS], dtype=bool)

FILTERED_OPTION_ACTION_IDS = {
    move_type: {
        size: {rank_value: [ACTION_ID_BY_STRING[action] for action in actions] for rank_value, actions in ranks.items()}
        for size, ranks in sizes.items()
    }
    for move_type, sizes in filtered_options.items()
}
BOMB_ACTION_IDS = tuple(dict.fromkeys(
    action_id
    for size_to_moves in FILTERED_OPTION_ACTION_IDS["bomb"].values()
    for action_ids in size_to_moves.values()
    for action_id in action_ids
)) + (ROCKET_ACTION_ID,)


def get_action_dict_by_id(action_id: int) -> dict[str, int]:
    return ACTION_CARD_DICTS[action_id]


def can_make_move(move_frequency: dict[str, int], cards_in_hand: dict[str, int]) -> bool:
    return all(cards_in_hand[card] >= count for card, count in move_frequency.items())


def _candidate_action_ids(info: dict[str, int]) -> tuple[int, ...]:
    if info["type"] == "pass":
        return OPEN_ACTION_IDS
    candidates = []
    for rank_value, action_ids in FILTERED_OPTION_ACTION_IDS.get(info["type"], {}).get(str(info["size"]), {}).items():
        if int(rank_value) > info["rank"]:
            candidates.extend(action_ids)
    # Rocket beats every four-card bomb, but no action beats a rocket.  Other
    # bomb responses are restricted to a higher four-card bomb or the rocket.
    if info["type"] == "bomb":
        if info["size"] == 2:  # rocket
            candidates = []
        else:
            candidates.append(ROCKET_ACTION_ID)
    else:
        candidates.extend(BOMB_ACTION_IDS)
    candidates.append(PASS_ACTION_ID)
    return tuple(dict.fromkeys(candidates))


def get_move_options_with_ids_reference(info, hand: dict[str, int]) -> list[tuple[int, dict[str, int]]]:
    """Independent, deliberately simple rules oracle for legality tests."""
    if info["type"] == "pass":
        allowed = list(OPEN_ACTION_IDS)
    else:
        matching, bombs = [], []
        for action_id, action_info in enumerate(ACTION_INFOS):
            if action_info["type"] == info["type"] and action_info["size"] == info["size"] and action_info["rank"] > info["rank"] and info["type"] != "bomb":
                matching.append(action_id)
            if action_info["type"] == "bomb":
                if info["type"] != "bomb" or action_info["size"] == 2 or (info["size"] != 2 and action_info["size"] == 4 and action_info["rank"] > info["rank"]):
                    bombs.append(action_id)
        allowed = matching + ([] if info["type"] == "bomb" and info["size"] == 2 else bombs) + [PASS_ACTION_ID]
    return [(action_id, ACTION_CARD_DICTS[action_id]) for action_id in allowed if can_make_move(ACTION_CARD_DICTS[action_id], hand)]


def get_move_options_with_ids(info, hand: dict[str, int]) -> list[tuple[int, dict[str, int]]]:
    """Exact vectorized replacement for the reference per-action card check."""
    candidate_ids = np.asarray(_candidate_action_ids(info), dtype=np.int32)
    hand_counts = np.fromiter((hand[card] for card in CARD_KEYS), dtype=np.int8, count=len(CARD_KEYS))
    legal = np.all(ACTION_COUNT_MATRIX[candidate_ids] <= hand_counts, axis=1)
    return [(int(action_id), ACTION_CARD_DICTS[int(action_id)]) for action_id in candidate_ids[legal]]


def get_move_options(info, hand: dict[str, int]) -> list[dict[str, int]]:
    return [option for _, option in get_move_options_with_ids(info, hand)]


def prune_options_for_scoring(options_with_ids, max_scored_options=None):
    """Compatibility shim: training v2 deliberately scores every legal action."""
    if max_scored_options is not None:
        raise ValueError("candidate pruning is disabled: every legal action must be scored")
    return options_with_ids


def remove_choice_from_hand(hand, move):
    for card, count in move.items():
        hand[card] -= count
    return hand


def remove_move_from_hand_copy(hand, move):
    copied = hand.copy()
    return remove_choice_from_hand(copied, move)


def cards_not_seen(hand, cards_played):
    full = full_card_dict()
    for card in full:
        full[card] -= hand[card] + sum(played[card] for played in cards_played)
    return full


def dict_to_tensor(card_dict):
    tensor = np.zeros(54, dtype=np.float32)
    for card, count in card_dict.items():
        index = 4 * rank(card)
        tensor[index : index + min(count, 4)] = 1
    return tensor[None, :]


def create_last_played_tensor(offset):
    return np.eye(3, 2, k=offset + 1, dtype=np.float32)[0][None, :]


def additional_features_tensor(card_dict):
    cards = "3456789TJQKA"
    features = []
    for length, count in ((length, 1) for length in range(5, len(cards) + 1)):
        features.extend(int(all(card_dict[card] >= count for card in cards[start : start + length])) for start in range(len(cards) - length + 1))
    for length, count in ((length, 2) for length in range(3, 6)):
        features.extend(int(all(card_dict[card] >= count for card in cards[start : start + length])) for start in range(len(cards) - length + 1))
    for length, count in ((length, 3) for length in range(2, 4)):
        features.extend(int(all(card_dict[card] >= count for card in cards[start : start + length])) for start in range(len(cards) - length + 1))
    features.append(int(card_dict["B"] + card_dict["R"] == 2))
    return np.asarray(features, dtype=np.float32)[None, :]


def card_count(card_dict):
    return sum(card_dict.values())


def cards_left_tensor(played_by_hands, pos):
    remaining = (20 if pos == 0 else 17) - card_count(played_by_hands[pos])
    result = np.zeros(5, dtype=np.float32)
    if 1 <= remaining <= 5:
        result[remaining - 1] = 1
    return result[None, :]


def to_string(card_dict):
    return "pass" if not card_count(card_dict) else "".join(sorted((card * count for card, count in card_dict.items()), key=lambda value: rank(value[0])))


def create_transformer_input(turns):
    result = np.zeros((15, 54), dtype=np.float32)
    for index, turn in enumerate(reversed(turns[-15:])):
        action_id = turn.get("action_id") if isinstance(turn, dict) else None
        result[index] = dict_to_tensor(ACTION_CARD_DICTS[action_id]).reshape(54) if action_id is not None else np.asarray(turn["tensors"]["choice_tensor"]).reshape(54)
    return result


def _uses_sequence_history(model_name):
    return get_model_config(model_name).uses_sequence_history


def create_feature_accumulator(model_name):
    config = get_model_config(model_name)
    return [[] for _ in range(len(BASE_FEATURE_KEYS) + int(config.uses_sequence_history) + int(config.uses_stake_context))]


def build_turn_context(game, position, hand, model_name):
    played = game["cards_played_by_hands"]
    unseen = cards_not_seen(hand, played)
    last = create_last_played_tensor(0)
    if game["turns"]:
        last = create_last_played_tensor(1 if game["turns"][-1]["turn_info"]["type"] != "pass" else 2 if len(game["turns"]) > 1 and game["turns"][-2]["turn_info"]["type"] != "pass" else 0)
    context = {
        "cards_not_seen_additional_features_tensor": additional_features_tensor(unseen),
        "cards_not_seen_tensor": dict_to_tensor(unseen),
        "cards_person_on_right_has_played_tensor": dict_to_tensor(played[(position + 1) % 3]),
        "cards_person_on_left_has_played_tensor": dict_to_tensor(played[(position - 1) % 3]),
        "last_played_tensor": last,
        "cards_person_on_left_has_left_tensor": cards_left_tensor(played, (position - 1) % 3),
        "cards_person_on_right_has_left_tensor": cards_left_tensor(played, (position + 1) % 3),
    }
    if _uses_sequence_history(model_name):
        context[TRANSFORMER_FEATURE_KEY] = create_transformer_input(game["turns"])
    return context


def stake_context_tensor(position, multiplier):
    """Versioned challenger-only context: stake, role, and teammate direction."""
    return np.asarray([[min(multiplier, 16) / 16.0, float(position == 0), float(position != 0), float(position in (1, 2))]], dtype=np.float32)


def build_turn_tensors(game, position, hand, choice_dict, model_name, context=None):
    context = context or build_turn_context(game, position, hand, model_name)
    remaining = remove_move_from_hand_copy(hand, choice_dict)
    tensors = {
        "cards_not_seen_additional_features_tensor": context["cards_not_seen_additional_features_tensor"],
        "cards_remaining_additional_feature_tensor": additional_features_tensor(remaining),
        "cards_not_seen_tensor": context["cards_not_seen_tensor"],
        "cards_person_on_right_has_played_tensor": context["cards_person_on_right_has_played_tensor"],
        "cards_person_on_left_has_played_tensor": context["cards_person_on_left_has_played_tensor"],
        "choice_tensor": dict_to_tensor(choice_dict),
        "cards_remaining_tensor": dict_to_tensor(remaining),
        "last_played_tensor": context["last_played_tensor"],
        "cards_person_on_left_has_left_tensor": context["cards_person_on_left_has_left_tensor"],
        "cards_person_on_right_has_left_tensor": context["cards_person_on_right_has_left_tensor"],
        TRANSFORMER_FEATURE_KEY: context.get(TRANSFORMER_FEATURE_KEY, np.zeros((15, 54), dtype=np.float32)),
    }
    if get_model_config(model_name).uses_stake_context:
        tensors["stake_context"] = stake_context_tensor(position, game.get("stake_multiplier", 1))
    return tensors


def tensors_to_feature_list(tensors, model_name):
    shapes = (85, 85, 54, 54, 54, 54, 54, 2, 5, 5)
    features = [np.asarray(tensors[key]).reshape(shape) for key, shape in zip(BASE_FEATURE_KEYS, shapes)]
    if _uses_sequence_history(model_name):
        features.append(np.asarray(tensors[TRANSFORMER_FEATURE_KEY]).reshape(15, 54))
    if get_model_config(model_name).uses_stake_context:
        features.append(np.asarray(tensors["stake_context"]).reshape(4))
    return features


def append_choice_features(accumulator, tensors, model_name):
    for index, feature in enumerate(tensors_to_feature_list(tensors, model_name)):
        accumulator[index].append(feature)


def _direct_predict(model, features):
    output = model([tf.convert_to_tensor(feature) for feature in features], training=False)
    if isinstance(output, dict):
        return np.asarray(output["win_probability"]).reshape(-1), np.asarray(output.get("expected_payout", output["win_probability"])).reshape(-1)
    if isinstance(output, (list, tuple)):
        return np.asarray(output[0]).reshape(-1), np.asarray(output[1]).reshape(-1)
    values = np.asarray(output).reshape(-1)
    return values, None


def select_best_candidate(predictions, option_game_numbers, action_ids, hands, use_payout=False, payout_predictions=None):
    """Return the true maximum per game; never stop after the first candidate."""
    choices = {}
    for index, prediction in enumerate(predictions):
        game_number = option_game_numbers[index]
        option = ACTION_CARD_DICTS[action_ids[index]]
        remaining = remove_move_from_hand_copy(hands[game_number], option)
        score = float(payout_predictions[index]) if use_payout and payout_predictions is not None else float(expected_value(float(prediction), option, remaining))
        if game_number not in choices or score > choices[game_number]["score"]:
            choices[game_number] = {"action_id": action_ids[index], "prediction": float(prediction), "score": score}
    return choices


def _compact_payload(game_states, model_name):
    return encode_training_batch(model_name, [
        {"hands": [hand_to_string(hand) for hand in game["initial_hands"]], "actions": [turn["action_id"] for turn in game["turns"]], "predictions": [turn["prediction"] for turn in game["turns"]], "landlord_won": game["landlord_won"]}
        for game in game_states
    ])


def batch_seed(actor_seed: int, batch_number: int) -> int:
    """Derive independent deterministic batch seeds without offset collisions."""
    digest = hashlib.sha256(f"{actor_seed}|{batch_number}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def play_self_play_batch(partition, model_name, models, game_batch_size=50, explore_rate=0.2, seed=None, use_payout_head=None):
    if seed is not None:
        random.seed(seed)
    started = time.perf_counter()
    config = get_model_config(model_name)
    use_payout_head = config.use_payout_head if use_payout_head is None else use_payout_head
    game_states = []
    for index in range(game_batch_size):
        initial_hands = landlord_first_shuffle()
        game_states.append({"complete": False, "number": index, "hands": [hand.copy() for hand in initial_hands], "initial_hands": [hand.copy() for hand in initial_hands], "turns": [], "cards_played_by_hands": [empty_card_dict(), empty_card_dict(), empty_card_dict()], "landlord_won": False, "stake_multiplier": 1})
    candidate_rows = scored_rows = random_turns = 0
    for turn_number in range(200):
        position = turn_number % 3
        accumulator = create_feature_accumulator(model_name)
        option_games, action_ids = [], []
        if all(game["complete"] for game in game_states):
            break
        for game in game_states:
            if game["complete"]:
                continue
            options = get_move_options_with_ids(get_previous_turn_info(game["turns"]), game["hands"][position])
            candidate_rows += len(options)
            if random.random() < explore_rate:
                options = [random.choice(options)]
                random_turns += 1
            scored_rows += len(options)
            context = build_turn_context(game, position, game["hands"][position], model_name)
            for action_id, option in options:
                tensors = build_turn_tensors(game, position, game["hands"][position], option, model_name, context)
                append_choice_features(accumulator, tensors, model_name)
                option_games.append(game["number"])
                action_ids.append(action_id)
        features = [np.asarray(values, dtype=np.float32) for values in accumulator]
        win_predictions, payout_predictions = _direct_predict(models[position], features)
        choices = select_best_candidate(win_predictions, option_games, action_ids, {game["number"]: game["hands"][position] for game in game_states}, use_payout_head, payout_predictions)
        for game in game_states:
            choice = choices.get(game["number"])
            if game["complete"] or choice is None:
                continue
            option = ACTION_CARD_DICTS[choice["action_id"]]
            game["turns"].append({"turn_info": get_turn_info(option), "position": position, "action_id": choice["action_id"], "prediction": choice["prediction"]})
            game["stake_multiplier"] *= choice_bomb_multiplier(option)
            remove_choice_from_hand(game["hands"][position], option)
            for card, count in option.items():
                game["cards_played_by_hands"][position][card] += count
            if card_count(game["hands"][position]) == 0:
                game["complete"] = True
                game["landlord_won"] = position == 0
    payload = _compact_payload(game_states, model_name)
    return payload, {"batch_seconds": time.perf_counter() - started, "games": len(game_states), "candidate_rows": candidate_rows, "scored_rows": scored_rows, "random_turns": random_turns, "turns": sum(len(game["turns"]) for game in game_states), "queue_bytes": len(payload)}


def self_play(partition, model_name, stop_event=None, producer_stop_event=None, max_batches=None, stats_queue=None, queue_key="training_data", max_queue_items=64, game_batch_size=50, explore_rate=0.2, cpu_only=True, use_payout_head=None, seed=None, redis_host="localhost", redis_port=6379, producer_done_queue=None):
    if cpu_only:
        # Actors are CPU-only so the one learner process owns the GPU context.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
    current_version = get_latest_checkpoint_version(model_name)
    models = load_models(model_name, compile_model=False, version=current_version)
    client = redis.Redis(host=redis_host, port=redis_port, db=0)
    batches = 0
    while (stop_event is None or not stop_event.is_set()) and (producer_stop_event is None or not producer_stop_event.is_set()):
        if max_batches is not None and batches >= max_batches:
            break
        if client.llen(queue_key) >= max_queue_items:
            time.sleep(0.05)
            continue
        latest = get_latest_checkpoint_version(model_name)
        if latest != current_version:
            models = load_models(model_name, compile_model=False, version=latest)
            current_version = latest
        # A stream advances per batch while remaining stable across runs.
        payload, stats = play_self_play_batch(partition, model_name, models, game_batch_size=game_batch_size, explore_rate=explore_rate, use_payout_head=use_payout_head, seed=None if seed is None else batch_seed(seed, batches))
        client.rpush(queue_key, payload)
        if stats_queue is not None:
            stats_queue.put({"kind": "generation", "model_name": model_name, "version": current_version, **stats})
        batches += 1
    completion = {"kind": "producer_complete", "actor": partition, "batches": batches, "games": batches * game_batch_size}
    if stats_queue is not None:
        stats_queue.put(completion)
    if producer_done_queue is not None:
        producer_done_queue.put(completion)


if __name__ == "__main__":
    workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        pool.starmap(self_play, [(index, "transformer_v2") for index in range(workers)])
