import random
import numpy as np
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from turn_info import get_turn_info
from cards import empty_card_dict, full_card_dict, landlord_first_shuffle, shuffle, rank
import json
import threading
import time

def save_game_data(turns, file_path='game_data.json'):
    with open(file_path, 'a') as file:
        for turn in turns:
            json.dump(turn, file, cls=NumpyEncoder)
            file.write('\n')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def get_previous_turn_info(turns):
    if len(turns) > 0:
        if turns[-1]['turn_info']['type'] != 'pass':
            return turns[-1]['turn_info']
    if len(turns) > 1:
        if turns[-2]['turn_info']['type'] != 'pass':
            return turns[-2]['turn_info']
    return {'type': 'pass', 'size': 0, 'rank': 0}

def get_previous_played(turns):
    if len(turns) > 0:
        if turns[-1]['turn_info']['type'] != 'pass':
            return -1
    if len(turns) > 1:
        if turns[-2]['turn_info']['type'] != 'pass':
            return -2
    return 0

def can_make_move(move: str, cards_in_hand: dict[str, int]):
    move_frequency = empty_card_dict()
    for char in move:
        move_frequency[char] += 1

    for card, count in move_frequency.items():
        if cards_in_hand[card] < count:
            return False
    return True

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
        if 'BR' not in actions: actions.append('BR')
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

def dict_to_tensor(card_dict: dict[str, int]):
    tensor = np.zeros(54)
    for card, count in card_dict.items():
        for i in range(count):
            tensor[min(4*rank(card) + i, 53)] = 1

    return np.expand_dims(tensor, axis=0)

def additional_features_tensor(card_dict: dict[str, int]):
    cards = '3456789TJQKA'
    l = []
    for i in range(len(cards)): # 36 straights
        still_going = True
        for j in range(i, len(cards)):
            if card_dict[cards[j]] == 1:
                still_going = False
            if j - i >= 4:
                if still_going == True:
                    l.append(1)
                else:
                    l.append(0)

    for i in range(len(cards)): # 27 pair straights ignoring len > 5
        still_going = True
        for j in range(i, len(cards)):
            if j - i > 4:
                break
            if card_dict[cards[j]] < 2:
                still_going = False
            if j - i >= 2:
                if still_going == True:
                    l.append(1)
                else:
                    l.append(0)
    
    for i in range(len(cards)): # 21 triple straights ignoring len > 3
        still_going = True
        for j in range(i, len(cards)):
            if j - i > 2:
                break
            if card_dict[cards[j]] < 3:
                still_going = False
            if j - i >= 1:
                if still_going == True:
                    l.append(1)
                else:
                    l.append(0)
    
    if card_dict['B'] + card_dict['R'] == 2:
        l.append(1)
    else:
        l.append(0)

    tensor = np.zeros(85)
    for i in range(len(l)):
        tensor[i] = l[i]
    return np.expand_dims(tensor, axis=0)

def card_count(card_dict: dict[str, int]):
    count = 0
    for _, c in card_dict.items():
        count += c
    return count


def to_string(card_dict: dict[str, int]):
    s = ''
    for card, c in card_dict.items():
        for _ in range(c):
            s += card
    if s == '': 
        return 'pass'
    return ''.join(sorted(s, key=rank))

def train():
    p0 = tf.keras.models.load_model('p0.keras')
    p1 = tf.keras.models.load_model('p1.keras')
    p2 = tf.keras.models.load_model('p2.keras')
    models = [p0, p1, p2]
    learning_rate = .1
    start_time = time.perf_counter()
    game_batch_size = 50
    game_states = [{
        'complete': False,
        'number': i,
        'show_output': i == 0,
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
        feature_tensors_list = [[] for _ in range(9)]
        all_games_complete = True
        for game in game_states:
            if game['complete'] == True:
                continue

            all_games_complete = False
            
            hand = game['hands'][position]
            cards_person_on_left_has_played_tensor = dict_to_tensor(game['cards_played_by_hands'][(position-1)%3])
            cards_person_on_right_has_played_tensor = dict_to_tensor(game['cards_played_by_hands'][(position+1)%3])

            turn_info = get_previous_turn_info(game['turns'])
            options = get_move_options(turn_info, hand)

            choice_dict = options[0]
            # cards_remaining_dict = remove_move_from_hand_copy(hand, choice_dict)

            cards_not_seen_dict = cards_not_seen(hand, game['cards_played_by_hands'])
            cards_not_seen_tensor = dict_to_tensor(cards_not_seen_dict)
            cards_not_seen_additional_features_tensor = additional_features_tensor(cards_not_seen_dict)

            if game['show_output']: 
                print()
                if position == 0:  print('L')
                print(to_string(hand))

            if random.random() < 0.2:
                choice_dict = random.choice(options)
                options = [choice_dict]
                if game['show_output']: print('random choice')
            else:
                if game['show_output']: print('options:')
                
            previous_turn_tensor = dict_to_tensor(empty_card_dict())
            if len(game['turns']) > 0:
                previous_turn_tensor = game['turns'][-1]['choice_tensor']

            penultimate_turn_tensor = dict_to_tensor(empty_card_dict())
            if len(game['turns']) > 1:
                penultimate_turn_tensor = game['turns'][-2]['choice_tensor']

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
                    previous_turn_tensor.reshape(54),
                    penultimate_turn_tensor.reshape(54),
                ]

                tensors_across_games.append({
                    'cards_not_seen_additional_features_tensor': cards_not_seen_additional_features_tensor,
                    'cards_remaining_additional_feature_tensor': additional_features_tensor(cards_that_would_be_remaining_dict),
                    'cards_not_seen_tensor': cards_not_seen_tensor,
                    'cards_person_on_right_has_played_tensor': cards_person_on_right_has_played_tensor,
                    'cards_person_on_left_has_played_tensor': cards_person_on_left_has_played_tensor,
                    'choice_tensor': dict_to_tensor(option_dict),
                    'cards_remaining_tensor': dict_to_tensor(cards_that_would_be_remaining_dict),
                    'previous_turn_tensor': previous_turn_tensor,
                    'penultimate_turn_tensor': penultimate_turn_tensor,
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

            if game_number == 0:
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
            if game['complete'] == True:
                continue

            choice = choices[game['number']]
            choice_dict = choice['option_dict']
            hand = game['hands'][position]

            for card, count in choice_dict.items():
                game['cards_seen'][card] += count
                game['cards_played_by_hands'][position][card] += count
        
            if game['show_output']: print('choice:', to_string(choice_dict))

            turn_info = get_turn_info(choice_dict)
            game['turns'].append({ 
                'turn_info': turn_info,
                'prediction': choice['max_prediction'],
                'position': position,
                'cards_not_seen_additional_features_tensor': choice['tensors']['cards_not_seen_additional_features_tensor'],
                'cards_remaining_additional_feature_tensor': choice['tensors']['cards_remaining_additional_feature_tensor'],
                'cards_not_seen_tensor': choice['tensors']['cards_not_seen_tensor'],
                'cards_person_on_right_has_played_tensor': choice['tensors']['cards_person_on_right_has_played_tensor'],
                'cards_person_on_left_has_played_tensor': choice['tensors']['cards_person_on_left_has_played_tensor'],
                'choice_tensor': choice['tensors']['choice_tensor'],
                'cards_remaining_tensor': choice['tensors']['cards_remaining_tensor'],
                'previous_turn_tensor': choice['tensors']['previous_turn_tensor'],
                'penultimate_turn_tensor': choice['tensors']['penultimate_turn_tensor'],
            })
            remove_choice_from_hand(hand, choice_dict)
            if card_count(hand) == 0:
                game['complete'] = True
                if position == 0:
                    game['landlord_won'] = True

    turns_by_position = [[],[],[]]
    for game in game_states:
        for turn in game['turns']:
            if (turn['position'] == 0) == game['landlord_won']:
                turn['prediction'] += learning_rate * (1 - turn['prediction'])
            else:
                turn['prediction'] -= learning_rate * turn['prediction']
            turns_by_position[turn['position']].append(turn)
    
    for i in range(3):
        turns = turns_by_position[i]
        model = models[i]

        i1 = np.array([turn['cards_not_seen_additional_features_tensor'].reshape(85) for turn in turns])
        i2 = np.array([turn['cards_remaining_additional_feature_tensor'].reshape(85) for turn in turns])
        i3 = np.array([turn['cards_not_seen_tensor'].reshape(54) for turn in turns])
        i4 = np.array([turn['cards_person_on_right_has_played_tensor'].reshape(54) for turn in turns])
        i5 = np.array([turn['cards_person_on_left_has_played_tensor'].reshape(54) for turn in turns])
        i6 = np.array([turn['choice_tensor'].reshape(54) for turn in turns])
        i7 = np.array([turn['cards_remaining_tensor'].reshape(54) for turn in turns])
        i8 = np.array([turn['previous_turn_tensor'].reshape(54) for turn in turns])
        i9 = np.array([turn['penultimate_turn_tensor'].reshape(54) for turn in turns])

        y_train = np.array([turn['prediction'] for turn in turns])
        x_train = [i1, i2, i3, i4, i5, i6, i7, i8, i9]

        model.fit(x_train, y_train, epochs=1, batch_size=64)
        model.save(f"p{i}.keras")

    end_time = time.perf_counter()
    time_delta = end_time - start_time
    print(f"round took {time_delta} seconds")


if __name__ == "__main__":
    while True:
        # try:
        train()
        # except:
        #     pass


