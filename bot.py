import time
from self_play import create_last_played_tensor
from cards import empty_card_dict, empty_card_id_dict, mapped_values, rank
from self_play import dict_to_tensor, get_move_options, remove_move_from_hand_copy, additional_features_tensor, to_string
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import tensorflow as tf
import os
import json
import numpy as np
from turn_info import expected_value

def get_previous_turn_info(turns):
    if len(turns) > 0:
        if turns[0].type != 'pass':
            return {'type': turns[0].type, 'size': turns[0].size, 'rank': turns[0].rank}
    if len(turns) > 1:
        if turns[1].type != 'pass':
            return {'type': turns[1].type, 'size': turns[1].size, 'rank': turns[1].rank}
    return {'type': 'pass', 'size': 0, 'rank': 0}

def get_card_ids(card_dict: dict[str, list], choice: dict[str, int]):
    result_ids = []
    for card_value, freq in choice.items():
        if card_value in card_dict:
            card_ids = card_dict[card_value]
            if len(card_ids) >= freq:
                result_ids.extend(card_ids[:freq])

    return result_ids

def cards_left_tensor(cards_played_by_hand: dict[str, int], position: int):

    cards_left = 17
    if position == 0:
        cards_left = 20

    for _, c in cards_played_by_hand.items():
        cards_left -= c
    
    tensor = np.zeros(5)
    if cards_left < 6: tensor[cards_left-1] = 1
    return np.expand_dims(tensor, axis=0)

def create_previous_turns_tensor(card_dicts: list[dict[str, int]]):
    tensor = np.zeros((15, 54))
    for i in range(15):
        card_dict = card_dicts[i]
        for card, count in card_dict.items():
            for j in range(count):
                tensor[i, min(4*rank(card) + j, 53)] = 1

    return np.expand_dims(tensor, axis=0)

def run_background_process():
    db_url = os.getenv('DATABASE_URL', "postgresql://postgres:password@localhost:5432/ddz")
    # db_url = 'postgresql://dpawcqez:bgMMRQdd4FS4-kCRcDiCjg48rUi5yqd3@castor.db.elephantsql.com/dpawcqez'
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    models = [tf.keras.models.load_model(f"./models/transformer/transformer{position}.keras") for position in range(3)]
    while True:
        time.sleep(1)
        requested_predictions = session.execute(text("""
            select 
                p.*,
                g.landlord_hand_id 
            from predictions p
            join games g on p.game_id = g.id
            where status = 'requested'
        """)).fetchall()

        for req in requested_predictions:

            cards = session.execute(text(f"""
                select 
                    c.*,
                    h.position as hand_position,
                    t.number as turn_number
                from cards c
                join hands h on c.hand_id = h.id
                left join turns t on c.turn_id = t.id
                where c.game_id = {req.game_id}
            """)).fetchall()

            cards_person_on_left_has_played_dict = empty_card_dict()
            cards_person_on_right_has_played_dict = empty_card_dict()
            cards_in_hand = empty_card_dict()
            cards_not_seen_dict = empty_card_dict()
            cards_in_hand_ids = empty_card_id_dict()
            landlord_offset = 0
            previous_turns = [empty_card_dict() for _ in range(15)]
            for card in cards:
                if req.landlord_hand_id == card.hand_id:
                    landlord_offset = card.hand_position

                if card.turn_id != None:
                    if (req.turn_number - 1) % 3 == card.hand_position:
                        cards_person_on_left_has_played_dict[mapped_values(card.value)] += 1
                    if (req.turn_number + 1) % 3 == card.hand_position:
                        cards_person_on_right_has_played_dict[mapped_values(card.value)] += 1 
                    
                    if card.turn_number >= req.turn_number - 15:
                        previous_turns[req.turn_number - card.turn_number - 1][mapped_values(card.value)] += 1

                if card.turn_id == None:
                    if card.hand_position == req.turn_number % 3:
                        cards_in_hand[mapped_values(card.value)] += 1
                        cards_in_hand_ids[mapped_values(card.value)].append(card.id)
                    else:
                        cards_not_seen_dict[mapped_values(card.value)] += 1

            previous_turns_tensor = create_previous_turns_tensor(previous_turns)

            raw_sql = text(f"""
                select 
                    t.*
                from turns t
                where t.game_id = {req.game_id}
                order by id desc limit 2
            """)
            turns = session.execute(raw_sql).fetchall()
            turn_info = get_previous_turn_info(turns)
            options = get_move_options(turn_info, cards_in_hand)

            last_played_tensor = create_last_played_tensor(0)
            if len(turns) > 0:
                if turns[0].type != 'pass':
                    last_played_tensor = create_last_played_tensor(1)
                elif len(turns) > 1 and turns[1] != 'pass':
                    last_played_tensor = create_last_played_tensor(2)

            position = (req.turn_number-landlord_offset)%3
            cards_person_on_left_has_left_tensor = cards_left_tensor(cards_person_on_left_has_played_dict, (position - 1)%3)
            cards_person_on_right_has_left_tensor = cards_left_tensor(cards_person_on_right_has_played_dict, (position + 1)%3)
            print()
            print(f"pos: {position} cards:{to_string(cards_in_hand)}")
            model = models[position]
            choice = options[0]
            max_expected_value = -1
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

            options_to_print = []
            for i, option_dict in enumerate(options):
                prediction = predictions[i]
                exp_val = expected_value(prediction, option_dict, all_cards_remaining_dicts[i])
                options_to_print.append((prediction, option_dict, exp_val))

                if exp_val > max_expected_value:
                    max_expected_value = exp_val
                    choice = option_dict
            
            options_to_print.sort(key=lambda x: x[0], reverse=True)
            for prediction, option_dict, exp_val in options_to_print:
                print(f"p: {prediction:.4f}, ev: {exp_val:.4f} - {to_string(option_dict)}")

            ids = get_card_ids(cards_in_hand_ids, choice)
            session.execute(text("""
                update predictions
                set status = 'sent', args = :args
                where id = :id
            """), {'args': json.dumps({ 'selected_cards': ids }), 'id': req.id})
            session.commit()

if __name__ == "__main__":
    run_background_process()