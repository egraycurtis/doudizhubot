import time
from cards import empty_card_dict, empty_card_id_dict, mapped_values
from train import dict_to_tensor, get_move_options, create_position_tensor, remove_move_from_hand_copy, additional_features_tensor, to_string
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import tensorflow as tf
import os
import json

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

def run_background_process():
    # db_url = os.getenv('DATABASE_URL', "postgresql://postgres:password@localhost:5432/ddz")
    db_url = 'postgresql://dpawcqez:bgMMRQdd4FS4-kCRcDiCjg48rUi5yqd3@castor.db.elephantsql.com/dpawcqez'
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    model = tf.keras.models.load_model('new_model.keras')
    while True:
        time.sleep(1)
        requested_predictions = session.execute(text("""
            select p.*, g.landlord_hand_id from predictions p
            join games g on p.game_id = g.id
            where status = 'requested'
        """)).fetchall()

        for req in requested_predictions:

            cards = session.execute(text(f"""
                select 
                    c.*, h.position as hand_position, t.number as turn_number
                from cards c
                join hands h on c.hand_id = h.id
                left join turns t on c.turn_id = t.id
                where c.game_id = {req.game_id}
            """)).fetchall()

            cards_person_on_right_has_played_dict = empty_card_dict()
            cards_person_on_left_has_played_dict = empty_card_dict()
            cards_in_hand = empty_card_dict()
            cards_not_seen = empty_card_dict()
            cards_in_hand_ids = empty_card_id_dict()
            last_played_turn_number = 0
            landlord_position = 0
            for card in cards:
                if req.landlord_hand_id == card.hand_id:
                    landlord_position = card.hand_position

                if card.turn_id != None:
                    if card.turn_number > last_played_turn_number:
                        last_played_turn_number = card.turn_number
                    if (req.turn_number + 1) % 3 == card.hand_position:
                        cards_person_on_right_has_played_dict[mapped_values(card.value)] += 1 
                    if (req.turn_number - 1) % 3 == card.hand_position:
                        cards_person_on_left_has_played_dict[mapped_values(card.value)] += 1
                if card.turn_id == None:
                    if card.hand_position == req.turn_number % 3:
                        cards_in_hand[mapped_values(card.value)] += 1
                        cards_in_hand_ids[mapped_values(card.value)].append(card.id)
                    else:
                        cards_not_seen[mapped_values(card.value)] += 1

            position_tensor = create_position_tensor(landlord_position, req.turn_number % 3, (last_played_turn_number - req.turn_number)%3)
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

            choice = options[0]
            max_expected_value = -1
            for option_dict in options:
                cards_that_would_be_remaining_dict = remove_move_from_hand_copy(cards_in_hand, option_dict)

                prediction = model.predict([
                    additional_features_tensor(cards_not_seen),
                    additional_features_tensor(cards_that_would_be_remaining_dict),
                    dict_to_tensor(cards_not_seen),
                    dict_to_tensor(cards_person_on_right_has_played_dict),
                    dict_to_tensor(cards_person_on_left_has_played_dict),
                    dict_to_tensor(option_dict),
                    dict_to_tensor(cards_that_would_be_remaining_dict),
                    position_tensor,
                ], verbose=0)

                probability_of_winning = prediction[0][0]
                exp_val = expected_value(probability_of_winning, option_dict, cards_that_would_be_remaining_dict)
                print(to_string(option_dict), probability_of_winning, exp_val)
                if exp_val > max_expected_value:
                    max_expected_value = exp_val
                    choice = option_dict
                
            ids = get_card_ids(cards_in_hand_ids, choice)
            session.execute(text("""
                update predictions
                set status = 'sent', args = :args
                where id = :id
            """), {'args': json.dumps({ 'selected_cards': ids }), 'id': req.id})
            session.commit()
            print(ids)

run_background_process()