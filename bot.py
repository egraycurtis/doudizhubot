import time
from cards import empty_card_dict, empty_card_id_dict, mapped_values
from train import dict_to_tensor, get_move_options, position_tensor, remove_move_from_hand_copy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
import tensorflow as tf
import requests
import os

def get_previous_turn_info(turns):
    if len(turns) > 0:
        if turns[0].type != 'pass':
            return {'type': turns[0].type, 'size': turns[0].size, 'rank': turns[0].rank}
    if len(turns) > 1:
        if turns[1].type != 'pass':
            return {'type': turns[1].type, 'size': turns[1].size, 'rank': turns[1].rank}
    return {'type': 'pass', 'size': 0, 'rank': 0}

def get_card_ids(card_dict, choice):
    result_ids = []

    for card_value, freq in choice.items():
        if card_value in card_dict:
            available_ids = card_dict[card_value]
            if len(available_ids) >= freq:
                result_ids.extend(available_ids[:freq])
            else:
                print(f"Not enough IDs available for card value '{card_value}'. Needed: {freq}, available: {len(available_ids)}")
        else:
            print(f"Card value '{card_value}' not found.")

    return result_ids

def run_background_process():
    # db_url = os.getenv('DATABASE_URL', "postgresql://postgres:password@localhost:5432/ddz")
    db_url = 'postgresql://dpawcqez:bgMMRQdd4FS4-kCRcDiCjg48rUi5yqd3@castor.db.elephantsql.com/dpawcqez'
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    model = tf.keras.models.load_model('my_model.keras')
    while True:
        print('sleep')
        time.sleep(2)
        raw_sql = text("""
            select g.*, h.id as hand_id, h.position as hand_position, h.user_id
            from games g
            join rooms r on g.room_id = r.id
            join hands h on g.id = h.game_id
            join users u on h.user_id = u.id
            where g.created_at > current_timestamp - interval '1 hour'
            and g.landlord_won is null
            and u.type = 'bot'
            and g.turn_number % 3 = h.position
        """)

        games = session.execute(raw_sql).fetchall()
        for game in games:
            user_id = game.user_id
            raw_sql = text(f"""
                select 
                    c.*, h.position as hand_position, t.number as turn_number
                from cards c
                join hands h on c.hand_id = h.id
                left join turns t on c.turn_id = t.id
                where c.game_id = {game.id}
            """)

            cards = session.execute(raw_sql).fetchall()
            cardsPersonOnRightHasPlayed = empty_card_dict()
            cardsPersonOnLeftHasPlayed = empty_card_dict()
            cardsInHand = empty_card_dict()
            cardsInHandIDs = empty_card_id_dict()
            last_played_turn_number = 0
            landlord_position = 0
            for card in cards:
                if game.landlord_hand_id == card.hand_id:
                    landlord_position = card.hand_position

                if card.turn_id != None:
                    if card.turn_number > last_played_turn_number:
                        last_played_turn_number = card.turn_number
                    if (game.hand_position + 1) % 3 == card.hand_position:
                        cardsPersonOnRightHasPlayed[mapped_values(card.value)] += 1 
                    if (game.hand_position - 1) % 3 == card.hand_position:
                        cardsPersonOnLeftHasPlayed[mapped_values(card.value)] += 1
                if card.turn_id == None and card.hand_position == game.hand_position:
                    cardsInHand[mapped_values(card.value)] += 1
                    cardsInHandIDs[mapped_values(card.value)].append(card.id)

            positionStuff = position_tensor(landlord_position, game.hand_position, (last_played_turn_number - game.turn_number)%3)
            raw_sql = text(f"""
                select 
                    t.*
                from turns t
                where t.game_id = {game.id}
                order by id desc limit 2
            """)
            turns = session.execute(raw_sql).fetchall()
            turn_info = get_previous_turn_info(turns)
            options = get_move_options(turn_info, cardsInHand)

            choice = options[0]
            max_prediction = 0
            for option in options:
                cardsInOption = dict_to_tensor(option)
                cardsThatWouldBeRemaining = dict_to_tensor(remove_move_from_hand_copy(cardsInHand, option))

                prediction = model.predict([
                    dict_to_tensor(cardsPersonOnRightHasPlayed),
                    dict_to_tensor(cardsPersonOnLeftHasPlayed),
                    cardsInOption,
                    cardsThatWouldBeRemaining,
                    positionStuff,
                ], verbose=0)
                
                if prediction[0][0] > max_prediction:
                    max_prediction = prediction[0][0]
                    choice = option
                
            ids = get_card_ids(cardsInHandIDs, choice)
            data = {
                'selectedCards': ids,
                'userID': user_id
            }

            # url = os.getenv('URL', "'http://localhost:8080")+'/public/CreateTurn'
            url = 'https://www.doudizhu.online/external/CreateTurn'
            resp = requests.post(url, json=data)
            print(resp.status_code, data)

run_background_process()