import threading
from typing import Any

import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cards import empty_card_dict, empty_card_id_dict, mapped_values, rank
from self_play import (
    additional_features_tensor,
    cards_left_tensor,
    create_last_played_tensor,
    dict_to_tensor,
    get_move_options,
    get_previous_played,
    get_previous_turn_info,
    remove_move_from_hand_copy,
    to_string,
)
from turn_info import expected_value


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql://" + database_url[len("postgres://") :]
    return database_url


def get_card_ids(card_dict: dict[str, list[int]], choice: dict[str, int]) -> list[int]:
    result_ids: list[int] = []
    for card_value, freq in choice.items():
        if card_value in card_dict:
            card_ids = card_dict[card_value]
            if len(card_ids) >= freq:
                result_ids.extend(card_ids[:freq])

    return result_ids


def create_previous_turns_tensor(card_dicts: list[dict[str, int]]) -> np.ndarray:
    tensor = np.zeros((15, 54), dtype=np.float32)
    for i in range(15):
        card_dict = card_dicts[i]
        for card, count in card_dict.items():
            for j in range(count):
                tensor[i, min(4 * rank(card) + j, 53)] = 1

    return np.expand_dims(tensor, axis=0)


class BotInferenceService:
    def __init__(self, database_url: str):
        normalized_url = normalize_database_url(database_url)
        self.engine = create_engine(normalized_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(bind=self.engine)
        self.models = [
            tf.keras.models.load_model(f"./models/transformer/transformer{position}.keras")
            for position in range(3)
        ]
        self.lock = threading.Lock()

    def choose_move(self, game_id: int, hand_id: int, turn_number: int) -> dict[str, Any]:
        with self.lock:
            with self.session_factory() as session:
                return self._choose_move(session, game_id, hand_id, turn_number)

    def _choose_move(self, session: Any, game_id: int, hand_id: int, turn_number: int) -> dict[str, Any]:
        game = session.execute(
            text(
                """
                select landlord_hand_id
                from games
                where id = :game_id
                """
            ),
            {"game_id": game_id},
        ).fetchone()
        if game is None:
            raise ValueError(f"missing game {game_id}")
        if game.landlord_hand_id is None:
            raise ValueError("game has not chosen a landlord yet")

        cards = session.execute(
            text(
                """
                select
                    c.id,
                    c.hand_id,
                    c.turn_id,
                    c.value,
                    h.position as hand_position,
                    t.number as turn_number
                from cards c
                join hands h on c.hand_id = h.id
                left join turns t on c.turn_id = t.id
                where c.game_id = :game_id
                """
            ),
            {"game_id": game_id},
        ).fetchall()

        played_by_hands = [empty_card_dict() for _ in range(3)]
        cards_in_hand = empty_card_dict()
        cards_not_seen_dict = empty_card_dict()
        cards_in_hand_ids = empty_card_id_dict()
        landlord_offset = next(
            card.hand_position for card in cards if game.landlord_hand_id == card.hand_id
        )
        previous_turns = [empty_card_dict() for _ in range(15)]
        requested_hand_position = None

        for card in cards:
            if card.hand_id == hand_id and requested_hand_position is None:
                requested_hand_position = card.hand_position

            if card.turn_id is not None:
                played_by_hands[(card.hand_position - landlord_offset) % 3][mapped_values(card.value)] += 1
                if card.turn_number >= turn_number - 15:
                    previous_turns[turn_number - card.turn_number - 1][mapped_values(card.value)] += 1

            if card.turn_id is None:
                if card.hand_id == hand_id:
                    cards_in_hand[mapped_values(card.value)] += 1
                    cards_in_hand_ids[mapped_values(card.value)].append(card.id)
                else:
                    cards_not_seen_dict[mapped_values(card.value)] += 1

        if requested_hand_position is None:
            raise ValueError(f"missing hand {hand_id} for game {game_id}")
        if requested_hand_position != turn_number % 3:
            raise ValueError(
                f"hand {hand_id} is at position {requested_hand_position}, "
                f"but turn {turn_number} expects position {turn_number % 3}"
            )

        previous_turns_tensor = create_previous_turns_tensor(previous_turns)

        turns = session.execute(
            text(
                """
                select *
                from (
                    select
                        t.*
                    from turns t
                    where t.game_id = :game_id
                    order by id desc
                    limit 2
                ) recent_turns
                order by id asc
                """
            ),
            {"game_id": game_id},
        ).fetchall()

        turn_info = get_previous_turn_info(turns)
        options = get_move_options(turn_info, cards_in_hand)
        if not options:
            raise ValueError("bot has no legal move options")

        last_played_tensor = create_last_played_tensor(-get_previous_played(turns))

        position = (turn_number - landlord_offset) % 3
        cards_person_on_left_has_played_dict = played_by_hands[(position - 1) % 3]
        cards_person_on_right_has_played_dict = played_by_hands[(position + 1) % 3]
        cards_person_on_left_has_left_tensor = cards_left_tensor(played_by_hands, (position - 1) % 3)
        cards_person_on_right_has_left_tensor = cards_left_tensor(played_by_hands, (position + 1) % 3)

        model = self.models[position]
        choice = options[0]
        max_expected_value = float("-inf")
        feature_tensors_list = [[] for _ in range(11)]
        all_cards_remaining_dicts = []
        for option_dict in options:
            cards_that_would_be_remaining_dict = remove_move_from_hand_copy(cards_in_hand, option_dict)
            all_cards_remaining_dicts.append(cards_that_would_be_remaining_dict)

            feature_tensors = [
                additional_features_tensor(cards_not_seen_dict).reshape(85),
                additional_features_tensor(cards_that_would_be_remaining_dict).reshape(85),
                dict_to_tensor(cards_not_seen_dict).reshape(54),
                dict_to_tensor(cards_person_on_right_has_played_dict).reshape(54),
                dict_to_tensor(cards_person_on_left_has_played_dict).reshape(54),
                dict_to_tensor(option_dict).reshape(54),
                dict_to_tensor(cards_that_would_be_remaining_dict).reshape(54),
                last_played_tensor.reshape(2),
                cards_person_on_left_has_left_tensor.reshape(5),
                cards_person_on_right_has_left_tensor.reshape(5),
                previous_turns_tensor.reshape(15, 54),
            ]

            for i, tensor in enumerate(feature_tensors):
                feature_tensors_list[i].append(tensor)

        model_input_tensors = [np.array(feature_list) for feature_list in feature_tensors_list]
        predictions = model.predict(model_input_tensors, verbose=0)
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        for i, option_dict in enumerate(options):
            prediction = float(predictions[i])
            exp_val = expected_value(prediction, option_dict, all_cards_remaining_dicts[i])
            if exp_val > max_expected_value:
                max_expected_value = exp_val
                choice = option_dict

        selected_cards = get_card_ids(cards_in_hand_ids, choice)
        return {
            "position": position,
            "selected_cards": selected_cards,
            "selected_move": to_string(choice),
        }
