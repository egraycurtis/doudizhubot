from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import numpy as np

from cards import empty_card_dict, landlord_first_shuffle
from compete import evaluate_families, role_mapping
from self_play import ACTION_CARD_DICTS, ACTION_ID_BY_STRING, get_move_options_with_ids, get_move_options_with_ids_reference, get_previous_turn_info, select_best_candidate
from train import normalized_signed_payout
from training_codec import decode_training_batch, encode_training_batch, hand_to_string
from turn_info import expected_value


class TrainingSystemTests(unittest.TestCase):
    def test_vectorized_legality_matches_reference(self):
        random.seed(7)
        for _ in range(30):
            for hand in landlord_first_shuffle():
                for info in ({"type": "pass", "size": 0, "rank": 0}, {"type": "single", "size": 1, "rank": 0}, {"type": "pair", "size": 2, "rank": 4}, {"type": "bomb", "size": 4, "rank": 2}):
                    self.assertEqual([item[0] for item in get_move_options_with_ids_reference(info, hand)], [item[0] for item in get_move_options_with_ids(info, hand)])

    def test_candidate_selection_uses_maximum_not_first(self):
        hand = empty_card_dict()
        hand["3"] = hand["4"] = hand["5"] = 1
        action_ids = [ACTION_ID_BY_STRING["3"], ACTION_ID_BY_STRING["4"], ACTION_ID_BY_STRING["5"]]
        choices = select_best_candidate([0.1, 0.2, 0.3], [0, 0, 0], action_ids, {0: hand}, use_payout=True, payout_predictions=[0.1, 0.9, 0.2])
        self.assertEqual(choices[0]["action_id"], ACTION_ID_BY_STRING["4"])

    def test_previous_turn_info_tolerates_row_like_turns(self):
        self.assertEqual(get_previous_turn_info([{"turn_info": {"type": "single", "size": 1, "rank": 2}}, {"turn_info": {"type": "pass", "size": 0, "rank": 0}}])["type"], "single")

    def test_compact_codec_round_trip(self):
        hands = landlord_first_shuffle()
        games = [{"hands": [hand_to_string(hand) for hand in hands], "actions": [1, 22, 300], "predictions": [0.1, 0.5, 0.9], "landlord_won": True}]
        decoded = decode_training_batch(encode_training_batch("transformer_v2", games))
        self.assertEqual(decoded["model_name"], "transformer_v2")
        self.assertEqual(decoded["games"][0]["hands"], games[0]["hands"])
        self.assertEqual(decoded["games"][0]["actions"], games[0]["actions"])
        self.assertTrue(np.allclose(decoded["games"][0]["predictions"], games[0]["predictions"], atol=5e-4))

    def test_role_mapping_is_explicit(self):
        self.assertEqual(role_mapping("transformer_v2"), {0: "transformer_v2", 1: "transformer_v2", 2: "transformer_v2"})

    def test_evaluation_balances_mirrored_team_seats(self):
        outcomes = [
            {"landlord_won": True, "winner_position": 0, "stake_multiplier": 1},
            {"landlord_won": False, "winner_position": 1, "stake_multiplier": 1},
        ]
        with patch("compete.load_models", side_effect=[[object(), object(), object()], [object(), object(), object()]]), patch("compete._play_deal", side_effect=outcomes):
            result = evaluate_families("transformer", "transformer_v2", deals=1, seed=11)
        self.assertEqual(result["games"], 2)
        self.assertEqual(result["challenger_landlord_wins"], 1)
        self.assertEqual(result["challenger_peasant_team_wins"], 1)
        self.assertEqual(result["win_rate"], 1.0)

    def test_peasant_team_targets_are_shared(self):
        self.assertEqual(normalized_signed_payout(1, landlord_won=False, multiplier=2), normalized_signed_payout(2, landlord_won=False, multiplier=2))
        self.assertGreater(normalized_signed_payout(1, landlord_won=False, multiplier=2), 0)

    def test_bomb_value_respects_win_confidence(self):
        bomb = empty_card_dict()
        bomb["3"] = 4
        regular = empty_card_dict()
        regular["4"] = 1
        remaining = empty_card_dict()
        self.assertLess(expected_value(0.45, bomb, remaining), expected_value(0.45, regular, remaining))
        self.assertGreater(expected_value(0.80, bomb, remaining), expected_value(0.80, regular, remaining))


if __name__ == "__main__":
    unittest.main()
