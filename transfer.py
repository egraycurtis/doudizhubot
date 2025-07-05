import numpy as np
from self_play import NumpyEncoder, additional_features_tensor, card_count, cards_left_tensor, cards_not_seen, create_last_played_tensor, create_transformer_input, dict_to_tensor, get_move_options, get_previous_turn_info, remove_choice_from_hand, remove_move_from_hand_copy, to_string
import tensorflow as tf
from turn_info import get_turn_info
from cards import empty_card_dict, landlord_first_shuffle
import json
import multiprocessing
import redis

def transfer(partition):
    while True:
        try:
            p0 = tf.keras.models.load_model('./models/deep/deep0.keras')
            p1 = tf.keras.models.load_model('./models/deep/deep1.keras')
            p2 = tf.keras.models.load_model('./models/deep/deep2.keras')
            model_name = "transformer"
            models = [p0, p1, p2]
            game_batch_size = 10
            game_states = [{
                'complete': False,
                'number': i,
                'show_output': i == 0 and partition == 0,
                'hands': landlord_first_shuffle(),
                'turns': [],
                'cards_played_by_hands': [empty_card_dict(), empty_card_dict(), empty_card_dict()],
                'cards_seen': empty_card_dict(),
                'landlord_won': False,
                } for i in range(game_batch_size)]
            
            training_data = []
            for turn_number in range(200):
                position = turn_number%3
                options_across_games = []
                tensors_across_games = []
                option_game_number = []

                feature_tensors_list = [[] for _ in range(10)]

                all_games_complete = True
                for game in game_states:
                    if game['complete'] :
                        continue

                    all_games_complete = False
                    
                    hand = game['hands'][position]
                    cards_person_on_left_has_played_tensor = dict_to_tensor(game['cards_played_by_hands'][(position-1)%3])
                    cards_person_on_right_has_played_tensor = dict_to_tensor(game['cards_played_by_hands'][(position+1)%3])

                    turn_info = get_previous_turn_info(game['turns'])
                    options = get_move_options(turn_info, hand)

                    choice_dict = options[0]
                    cards_not_seen_dict = cards_not_seen(hand, game['cards_played_by_hands'])
                    cards_not_seen_tensor = dict_to_tensor(cards_not_seen_dict)
                    cards_not_seen_additional_features_tensor = additional_features_tensor(cards_not_seen_dict)

                    if game['show_output']: 
                        print()
                        if position == 0:
                            print('L')
                        print(to_string(hand))

                    if game['show_output']:
                        print('options:')
                        
                    last_played_tensor = create_last_played_tensor(0)
                    if len(game['turns']) > 0:
                        if game['turns'][-1]['turn_info']['type'] != 'pass':
                            last_played_tensor = create_last_played_tensor(1)
                        elif len(game['turns']) > 1 and game['turns'][-2]['turn_info']['type'] != 'pass':
                            last_played_tensor = create_last_played_tensor(2)

                    cards_person_on_left_has_left_tensor = cards_left_tensor(game['cards_played_by_hands'], (position - 1)%3)
                    cards_person_on_right_has_left_tensor = cards_left_tensor(game['cards_played_by_hands'], (position + 1)%3)

                    transformer_tensor = np.zeros((15, 54), dtype=np.float32)
                    if model_name == 'transformer':
                        transformer_tensor = create_transformer_input(game['turns'])

                    for option_dict in options:
                        options_across_games.append(option_dict)
                        option_game_number.append(game['number'])

                        cards_that_would_be_remaining_dict = remove_move_from_hand_copy(hand, option_dict)
                        feature_tensors = [
                            cards_not_seen_additional_features_tensor.reshape(85),
                            additional_features_tensor(cards_that_would_be_remaining_dict).reshape(85),
                            cards_not_seen_tensor.reshape(54),
                            cards_person_on_right_has_played_tensor.reshape(54),
                            cards_person_on_left_has_played_tensor.reshape(54),
                            dict_to_tensor(option_dict).reshape(54),
                            dict_to_tensor(cards_that_would_be_remaining_dict).reshape(54),
                            last_played_tensor.reshape(2),
                            cards_person_on_left_has_left_tensor.reshape(5),
                            cards_person_on_right_has_left_tensor.reshape(5),
                        ]

                        tensors_across_games.append({
                            'cards_not_seen_additional_features_tensor': cards_not_seen_additional_features_tensor,
                            'cards_remaining_additional_feature_tensor': additional_features_tensor(cards_that_would_be_remaining_dict),
                            'cards_not_seen_tensor': cards_not_seen_tensor,
                            'cards_person_on_right_has_played_tensor': cards_person_on_right_has_played_tensor,
                            'cards_person_on_left_has_played_tensor': cards_person_on_left_has_played_tensor,
                            'choice_tensor': dict_to_tensor(option_dict),
                            'cards_remaining_tensor': dict_to_tensor(cards_that_would_be_remaining_dict),
                            'last_played_tensor': last_played_tensor,
                            'cards_person_on_left_has_left_tensor': cards_person_on_left_has_left_tensor,
                            'cards_person_on_right_has_left_tensor': cards_person_on_right_has_left_tensor,
                            'transformer_tensor': transformer_tensor,
                        })

                        for i, tensor in enumerate(feature_tensors):
                            feature_tensors_list[i].append(tensor)
                
                if all_games_complete:
                    break

                # print('find max predictions')
                model_input_tensors = [np.array(feature_list) for feature_list in feature_tensors_list]
                predictions = models[position].predict(model_input_tensors, verbose=0)

                if predictions.ndim > 1:
                    predictions = predictions.flatten()


                choices = [{'max_prediction': 0, 'tensors': {}, 'option_dict': {}} for _ in range(game_batch_size)]
                options_to_print = []
                for i, option_dict in enumerate(options_across_games):
                    prediction = predictions[i]
                    game_number = option_game_number[i]
                    training_data.append({'prediction': prediction, 'tensors': tensors_across_games[i], 'position': position})

                    if game_number == 0 and partition == 0:
                        options_to_print.append((prediction, option_dict))

                    if prediction > choices[game_number]['max_prediction']:
                        choices[game_number]['max_prediction'] = prediction
                        choices[game_number]['tensors'] = tensors_across_games[i]
                        choices[game_number]['option_dict'] = option_dict

                options_to_print.sort(key=lambda x: x[0], reverse=True)
                for prediction, option_dict in options_to_print:
                    print(f"{prediction:.5f} - {to_string(option_dict)}")

                # print('update game states')
                for game in game_states:
                    if game['complete'] :
                        continue

                    choice = choices[game['number']]
                    choice_dict = choice['option_dict']
                    hand = game['hands'][position]

                    for card, count in choice_dict.items():
                        game['cards_seen'][card] += count
                        game['cards_played_by_hands'][position][card] += count
                
                    if game['show_output']:
                        print('choice:', to_string(choice_dict))

                    turn_info = get_turn_info(choice_dict)
                    game['turns'].append({ 
                        'turn_info': turn_info,
                        'prediction': choice['max_prediction'],
                        'position': position,
                        'tensors':  choice['tensors'],
                    })
                    remove_choice_from_hand(hand, choice_dict)
                    if card_count(hand) == 0:
                        game['complete'] = True
                        if position == 0:
                            game['landlord_won'] = True

            r = redis.Redis(host='localhost', port=6379, db=0)
            turn_data_json = json.dumps({'turns': training_data, 'model_name': model_name}, cls=NumpyEncoder)  
            count = r.llen('training_data')
            if count < 25:      
                r.rpush('training_data', turn_data_json)


            # end_time = time.perf_counter()
            # time_delta = end_time - start_time
        except Exception as e:
            print(e)
            pass
    

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print("cpu_count", cpu_count)
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.map(transfer, range(cpu_count))


