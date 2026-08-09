"""Resident learner for compact self-play trajectories."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import signal
import time

import numpy as np
import redis

from cards import empty_card_dict
from model_registry import get_latest_checkpoint_version, get_model_config, get_metadata_path, load_models, save_models
from self_play import build_turn_tensors, get_action_dict_by_id, remove_choice_from_hand
from training_codec import TRAJECTORY_FORMAT, decode_training_batch, string_to_hand
from turn_info import choice_bomb_multiplier, get_turn_info


def normalized_signed_payout(position: int, landlord_won: bool, multiplier: int) -> float:
    """Final payoff from the acting role's team's perspective, bounded for MSE."""
    team_won = landlord_won if position == 0 else not landlord_won
    return (1.0 if team_won else -1.0) * min(multiplier, 16) / 16.0


def _pop_training_payloads(client, queue_key, min_items=4, max_items=16, drain=False):
    if client.llen(queue_key) < min_items and not drain:
        return []
    pipeline = client.pipeline()
    pipeline.lrange(queue_key, 0, max_items - 1)
    pipeline.ltrim(queue_key, max_items, -1)
    payloads, _ = pipeline.execute()
    return payloads


def _final_multiplier(actions):
    multiplier = 1
    for action_id in actions:
        multiplier *= choice_bomb_multiplier(get_action_dict_by_id(action_id))
    return multiplier


def _replay_payload(raw_payload):
    payload = decode_training_batch(raw_payload)
    model_name = payload["model_name"]
    turns_by_position = [[], [], []]
    if payload.get("format") != TRAJECTORY_FORMAT:
        for turn in payload["turns"]:
            turns_by_position[turn["position"]].append(turn)
        return model_name, turns_by_position
    target_mix = get_model_config(model_name).target_mix
    for compact_game in payload["games"]:
        hands = [string_to_hand(hand) for hand in compact_game["hands"]]
        state = {"turns": [], "cards_played_by_hands": [empty_card_dict(), empty_card_dict(), empty_card_dict()], "stake_multiplier": 1}
        final_multiplier = _final_multiplier(compact_game["actions"])
        for turn_index, (action_id, prediction) in enumerate(zip(compact_game["actions"], compact_game["predictions"])):
            position = turn_index % 3
            choice = get_action_dict_by_id(action_id)
            tensors = build_turn_tensors(state, position, hands[position], choice, model_name)
            team_won = compact_game["landlord_won"] if position == 0 else not compact_game["landlord_won"]
            win_target = float(prediction) + (target_mix * (1.0 - float(prediction)) if team_won else -target_mix * float(prediction))
            # State context is pre-action; target is final realized game payout.
            turns_by_position[position].append({"prediction": win_target, "payout": normalized_signed_payout(position, compact_game["landlord_won"], final_multiplier), "tensors": tensors, "position": position})
            state["stake_multiplier"] *= choice_bomb_multiplier(choice)
            state["turns"].append({"turn_info": get_turn_info(choice), "action_id": action_id, "position": position})
            for card, count in choice.items():
                state["cards_played_by_hands"][position][card] += count
            remove_choice_from_hand(hands[position], choice)
    return model_name, turns_by_position


def _group_turns(payloads, reconstruction_workers=1):
    grouped = defaultdict(lambda: [[], [], []])
    mapper = ThreadPoolExecutor(max_workers=reconstruction_workers) if reconstruction_workers > 1 else None
    results = mapper.map(_replay_payload, payloads) if mapper else map(_replay_payload, payloads)
    try:
        for model_name, positions in results:
            for position, turns in enumerate(positions):
                grouped[model_name][position].extend(turns)
    finally:
        if mapper:
            mapper.shutdown()
    return grouped


def _turns_to_arrays(turns, model_name):
    specs = list(zip(
        ("cards_not_seen_additional_features_tensor", "cards_remaining_additional_feature_tensor", "cards_not_seen_tensor", "cards_person_on_right_has_played_tensor", "cards_person_on_left_has_played_tensor", "choice_tensor", "cards_remaining_tensor", "last_played_tensor", "cards_person_on_left_has_left_tensor", "cards_person_on_right_has_left_tensor", "transformer_tensor"),
        ((85,), (85,), (54,), (54,), (54,), (54,), (54,), (2,), (5,), (5,), (15, 54)),
    ))
    if get_model_config(model_name).uses_stake_context:
        specs.append(("stake_context", (4,)))
    x_train = [np.asarray([np.asarray(turn["tensors"][key]).reshape(shape) for turn in turns], dtype=np.float32) for key, shape in specs]
    win = np.asarray([turn["prediction"] for turn in turns], dtype=np.float32)
    if get_model_config(model_name).uses_stake_context:
        return x_train, {"win_probability": win, "expected_payout": np.asarray([turn.get("payout", 0.0) for turn in turns], dtype=np.float32)}
    return x_train, win


def install_worker_signal_policy():
    """The coordinator alone turns Ctrl+C into a drain/abort transition."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train(stop_event=None, max_batches=None, stats_queue=None, queue_key="training_data", min_queue_items=4, max_payloads=16, reconstruction_workers=1, redis_host="localhost", redis_port=6379, producers_done_event=None, ignore_sigint=False):
    if ignore_sigint:
        install_worker_signal_policy()
    client = redis.Redis(host=redis_host, port=redis_port, db=0)
    loaded_models, versions, dirty, batches_since_save = {}, {}, set(), defaultdict(int)
    completed = 0
    try:
        while stop_event is None or not stop_event.is_set():
            if max_batches is not None and completed >= max_batches:
                break
            draining = producers_done_event is not None and producers_done_event.is_set()
            payloads = _pop_training_payloads(client, queue_key, min_queue_items, max_payloads, drain=draining)
            if not payloads:
                if draining:
                    break
                time.sleep(0.05)
                continue
            replay_started = time.perf_counter()
            grouped = _group_turns(payloads, reconstruction_workers)
            replay_seconds = time.perf_counter() - replay_started
            for model_name, by_position in grouped.items():
                if model_name not in loaded_models:
                    versions[model_name] = get_latest_checkpoint_version(model_name)
                    loaded_models[model_name] = load_models(model_name, compile_model=True, version=versions[model_name])
                fit_started, examples = time.perf_counter(), 0
                for position, turns in enumerate(by_position):
                    if turns:
                        x_train, targets = _turns_to_arrays(turns, model_name)
                        loaded_models[model_name][position].fit(x_train, targets, epochs=1, batch_size=256, verbose=0)
                        examples += len(turns)
                dirty.add(model_name)
                batches_since_save[model_name] += 1
                checkpoint_seconds = 0.0
                if batches_since_save[model_name] >= get_model_config(model_name).checkpoint_interval_batches:
                    checkpoint_started = time.perf_counter()
                    source_model = json.loads(get_metadata_path(model_name).read_text()).get("source_model")
                    versions[model_name] = save_models(model_name, loaded_models[model_name], source_model=source_model)
                    checkpoint_seconds = time.perf_counter() - checkpoint_started
                    batches_since_save[model_name] = 0
                    dirty.remove(model_name)
                if stats_queue is not None:
                    stats_queue.put({"kind": "learner", "model_name": model_name, "examples": examples, "replay_seconds": replay_seconds, "fit_seconds": time.perf_counter() - fit_started - checkpoint_seconds, "checkpoint_seconds": checkpoint_seconds, "version": versions[model_name], "durable_checkpoint": checkpoint_seconds > 0})
            completed += 1
    finally:
        for model_name in dirty:
            checkpoint_started = time.perf_counter()
            source_model = json.loads(get_metadata_path(model_name).read_text()).get("source_model")
            versions[model_name] = save_models(model_name, loaded_models[model_name], source_model=source_model)
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            if stats_queue is not None:
                # This final metric must be consumed before the coordinator
                # summarizes a normal drain, otherwise checkpoint accounting
                # can falsely report an empty learner run.
                stats_queue.put({"kind": "learner", "model_name": model_name, "examples": 0, "replay_seconds": 0.0, "fit_seconds": 0.0, "checkpoint_seconds": checkpoint_seconds, "version": versions[model_name], "durable_checkpoint": True, "final_checkpoint": True})


if __name__ == "__main__":
    train()
