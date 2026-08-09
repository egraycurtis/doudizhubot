"""Deterministic, team-correct evaluation for three role-specific model families."""

from __future__ import annotations

import math
import random

import numpy as np

from cards import empty_card_dict, landlord_first_shuffle
from model_registry import get_model_config, load_models
from self_play import ACTION_CARD_DICTS, _direct_predict, append_choice_features, build_turn_context, build_turn_tensors, card_count, create_feature_accumulator, get_move_options_with_ids, get_previous_turn_info, remove_choice_from_hand, select_best_candidate
from turn_info import choice_bomb_multiplier, get_turn_info


ROLE_NAMES = ("landlord_p0", "peasant_p1", "peasant_p2")


def role_mapping(model_family: str) -> dict[int, str]:
    """Every seat always uses its matching role-specific checkpoint."""
    return {position: model_family for position in range(3)}


def _play_deal(models_by_position, model_names_by_position, initial_hands, payout_positions=frozenset()):
    state = {"hands": [hand.copy() for hand in initial_hands], "turns": [], "cards_played_by_hands": [empty_card_dict(), empty_card_dict(), empty_card_dict()], "stake_multiplier": 1}
    for turn_number in range(200):
        position = turn_number % 3
        options = get_move_options_with_ids(get_previous_turn_info(state["turns"]), state["hands"][position])
        context = build_turn_context(state, position, state["hands"][position], model_names_by_position[position])
        accumulator = create_feature_accumulator(model_names_by_position[position])
        action_ids = []
        for action_id, option in options:
            append_choice_features(accumulator, build_turn_tensors(state, position, state["hands"][position], option, model_names_by_position[position], context), model_names_by_position[position])
            action_ids.append(action_id)
        features = [np.asarray(values, dtype=np.float32) for values in accumulator]
        wins, payouts = _direct_predict(models_by_position[position], features)
        choice = select_best_candidate(wins, [0] * len(action_ids), action_ids, {0: state["hands"][position]}, position in payout_positions, payouts)[0]
        option = ACTION_CARD_DICTS[choice["action_id"]]
        state["turns"].append({"turn_info": get_turn_info(option), "action_id": choice["action_id"], "position": position})
        state["stake_multiplier"] *= choice_bomb_multiplier(option)
        remove_choice_from_hand(state["hands"][position], option)
        for card, count in option.items():
            state["cards_played_by_hands"][position][card] += count
        if card_count(state["hands"][position]) == 0:
            return {"landlord_won": position == 0, "winner_position": position, "stake_multiplier": state["stake_multiplier"]}
    raise RuntimeError("deal exceeded 200 turns")


def _wilson_interval(wins, total, z=1.96):
    if not total:
        return (0.0, 1.0)
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def evaluate_families(baseline_model: str, challenger_model: str, deals: int = 100, seed: int = 1, challenger_uses_payout: bool = False):
    """Mirror each deterministic deal: challenger landlord, then challenger peasants."""
    baseline, challenger = load_models(baseline_model, compile_model=False), load_models(challenger_model, compile_model=False)
    challenger_wins = challenger_landlord_wins = challenger_peasant_wins = 0
    results = []
    for deal_index in range(deals):
        random.seed(seed + deal_index)
        hands = landlord_first_shuffle()
        landlord_result = _play_deal([challenger[0], baseline[1], baseline[2]], [challenger_model, baseline_model, baseline_model], hands, frozenset({0}) if challenger_uses_payout else frozenset())
        peasant_result = _play_deal([baseline[0], challenger[1], challenger[2]], [baseline_model, challenger_model, challenger_model], hands, frozenset({1, 2}) if challenger_uses_payout else frozenset())
        if landlord_result["landlord_won"]:
            challenger_wins += 1
            challenger_landlord_wins += 1
        if not peasant_result["landlord_won"]:
            challenger_wins += 1
            challenger_peasant_wins += 1
        results.extend((landlord_result, peasant_result))
    total = deals * 2
    return {"baseline": baseline_model, "challenger": challenger_model, "challenger_uses_payout": challenger_uses_payout, "deals": deals, "games": total, "challenger_wins": challenger_wins, "challenger_landlord_wins": challenger_landlord_wins, "challenger_peasant_team_wins": challenger_peasant_wins, "win_rate": challenger_wins / total, "confidence_interval_95": _wilson_interval(challenger_wins, total), "results": results}
