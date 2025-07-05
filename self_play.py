import random
import numpy as np
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from turn_info import get_turn_info
from cards import empty_card_dict, full_card_dict, landlord_first_shuffle, rank
import json
import redis
import multiprocessing

def self_play(partition: int, model_name: str):
    while True:
        try:
            models = [tf.keras.models.load_model(f"./models/{model_name}/{model_name}{position}.keras") for position in range(3)]
            learning_rate = .1
            game_batch_size = 50
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
            
            
            for turn_number in range(200):
                position = turn_number%3
                options_across_games = []
                tensors_across_games = []
                option_game_number = []

                feature_tensors_list = [[] for _ in range(10)]
                if model_name == 'transformer' or model_name == 'lstm':
                    feature_tensors_list = [[] for _ in range(11)]

                all_games_complete = True
                for game in game_states:
                    if game['complete']:
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

                    # if game['show_output']: 
                    #     print()
                    #     if position == 0:  print('L')
                    #     print(to_string(hand))

                    if random.random() < 0.2:
                        choice_dict = random.choice(options)
                        options = [choice_dict]
                        # if game['show_output']: print('random choice')
                    # else:
                        # if game['show_output']: print('options:')
                        
                    last_played_tensor = create_last_played_tensor(0)
                    if len(game['turns']) > 0:
                        if game['turns'][-1]['turn_info']['type'] != 'pass':
                            last_played_tensor = create_last_played_tensor(1)
                        elif len(game['turns']) > 1 and game['turns'][-2]['turn_info']['type'] != 'pass':
                            last_played_tensor = create_last_played_tensor(2)

                    cards_person_on_left_has_left_tensor = cards_left_tensor(game['cards_played_by_hands'], (position - 1)%3)
                    cards_person_on_right_has_left_tensor = cards_left_tensor(game['cards_played_by_hands'], (position + 1)%3)

                    transformer_tensor = np.zeros((15, 54), dtype=np.float32)
                    if model_name == 'transformer' or model_name == 'lstm':
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

                        if model_name == 'transformer' or model_name == 'lstm':
                            feature_tensors.append(transformer_tensor)

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

                    if game_number == 0 and partition == 0:
                        options_to_print.append((prediction, option_dict))

                    if prediction > choices[game_number]['max_prediction']:
                        choices[game_number]['max_prediction'] = prediction
                        choices[game_number]['tensors'] = tensors_across_games[i]
                        choices[game_number]['option_dict'] = option_dict

                # options_to_print.sort(key=lambda x: x[0], reverse=True)
                # for prediction, option_dict in options_to_print:
                #     print(f"{prediction:.5f} - {to_string(option_dict)}")

                # print('update game states')
                for game in game_states:
                    if game['complete']:
                        continue

                    choice = choices[game['number']]
                    choice_dict = choice['option_dict']
                    hand = game['hands'][position]

                    for card, count in choice_dict.items():
                        game['cards_seen'][card] += count
                        game['cards_played_by_hands'][position][card] += count
                
                    # if game['show_output']: print('choice:', to_string(choice_dict))

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

            training_data = {'turns': [], 'model_name': model_name}
            for game in game_states:
                for turn in game['turns']:
                    if (turn['position'] == 0) == game['landlord_won']:
                        turn['prediction'] += learning_rate * (1 - turn['prediction'])
                    else:
                        turn['prediction'] -= learning_rate * turn['prediction']
                    training_data['turns'].append({'prediction': turn['prediction'], 'tensors': turn['tensors'], 'position': turn['position']})


            r = redis.Redis(host='localhost', port=6379, db=0)
            training_data_json = json.dumps(training_data, cls=NumpyEncoder)        
            r.rpush('training_data', training_data_json)

        except Exception as e:
            print(e)
            pass

def create_transformer_input(turns):
    transformer_input = np.zeros((15, 54), dtype=np.float32)
    for i in range(min(len(turns), 15)):
        transformer_input[i, :] = np.array(turns[-(i+1)]['tensors']['choice_tensor'])
    
    return transformer_input

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def get_previous_turn_info(turns):
    if len(turns) > 0 and turns[-1]['turn_info']['type'] != 'pass':
            return turns[-1]['turn_info']
            
    if len(turns) > 1 and turns[-2]['turn_info']['type'] != 'pass':
            return turns[-2]['turn_info']
            
    return {'type': 'pass', 'size': 0, 'rank': 0}

def get_previous_played(turns):
    if len(turns) > 0 and turns[-1]['turn_info']['type'] != 'pass':
            return -1
    
    if len(turns) > 1 and turns[-2]['turn_info']['type'] != 'pass':
            return -2
    
    return 0

def can_make_move(move: str, cards_in_hand: dict[str, int]):
    move_frequency = string_to_card_dict(move)
    return all(cards_in_hand[c] >= n for c, n in move_frequency.items())


def string_to_card_dict(action: str):
    d = empty_card_dict()
    for a in action:
        d[a] += 1
    return d

def get_move_options(info, hand: dict[str, int]) -> list[dict[str, int]]:
    options = []
    actions = []
    if info['type'] == 'pass':
        actions = action_space

    else:
        for rank, values in filtered_options[info['type']][str(info['size'])].items():
            if int(rank) > info['rank']:
                actions.extend(values)
        if info['type'] != 'bomb':
            for _, sizeToMoves in filtered_options['bomb'].items():
                for _, values in sizeToMoves.items():
                    actions.extend(values)
        if 'BR' not in actions:
            actions.append('BR')
        actions.append('')

    for a in actions:
        if can_make_move(a, hand):
            opt = string_to_card_dict(a)
            options.append(opt)

    return options

def remove_choice_from_hand(hand: dict[str, int], move: dict[str, int]):
    for card, count in move.items():
        hand[card] -= count
    return hand

def cards_not_seen(hand: dict[str, int], cards_played: list[dict[str, int]]):
    full = full_card_dict()
    for card in full.keys():
        full[card] -= (hand[card] + cards_played[0][card] + cards_played[1][card] + cards_played[2][card])
    return full

def remove_move_from_hand_copy(hand: dict[str, int], move: dict[str, int]):
    hand_copy = hand.copy() 
    for card, count in move.items():
        hand_copy[card] -= count
    return hand_copy

def dict_to_tensor(card_dict: dict[str, int]) -> np.ndarray:
    tensor = np.zeros(54, dtype=float)
    for card, count in card_dict.items():
        idx = 4 * rank(card)
        tensor[idx : idx + min(count, 4)] = 1
    return tensor[None, :] 

def create_last_played_tensor(offset: int) -> np.ndarray:
    return np.eye(3, 2, k=offset + 1, dtype=float)[0][None, :]

def additional_features_tensor(card_dict: dict[str, int]) -> np.ndarray:
    cards = "3456789TJQKA"
    features: list[int] = []

    for length in range(5, len(cards) + 1):
        for start in range(len(cards) - length + 1):
            window = cards[start : start + length]
            features.append(int(all(card_dict[c] >= 1 for c in window)))

    for length in range(3, 6):
        for start in range(len(cards) - length + 1):
            window = cards[start : start + length]
            features.append(int(all(card_dict[c] >= 2 for c in window)))

    for length in range(2, 4):
        for start in range(len(cards) - length + 1):
            window = cards[start : start + length]
            features.append(int(all(card_dict[c] >= 3 for c in window)))


    features.append(int(card_dict.get("B", 0) + card_dict.get("R", 0) == 2))

    return np.array(features, dtype=float)[None, :]


def card_count(card_dict: dict[str, int]) -> int:
    return sum(card_dict.values())


def cards_left_tensor(played_by_hands: list[dict[str, int]], pos: int) -> np.ndarray:
    remaining = (20 if pos == 0 else 17) - card_count(played_by_hands[pos])
    tensor = np.zeros(5, dtype=float)
    if 1 <= remaining <= 5:
        tensor[remaining - 1] = 1

    return tensor[None, :]

def to_string(card_dict: dict[str, int]) -> str:
    s = "".join(card * n for card, n in card_dict.items() for n in range(card_dict[card]))
    return "pass" if not s else "".join(sorted(s, key=rank))




if __name__ == "__main__":
    models = ['transformer']
    cpu_count = multiprocessing.cpu_count()
    tasks = [(i, models[i%len(models)]) for i in range(cpu_count - 2)]
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.starmap(self_play, tasks)
