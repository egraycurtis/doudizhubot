import random
import numpy as np
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from turn_info import get_turn_info
from cards import empty_card_dict, full_card_dict, shuffle, rank
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

def get_move_options(info, hand: dict[str, int]):
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

def create_position_tensor(landlordPos: int, turnPos: int, prevTurnOffset: int):
    tensor = np.zeros(6)
    tensor[(landlordPos-turnPos)%3] = 1
    tensor[3 + prevTurnOffset%3] = 1

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
    while True:
        num_games = 5
        threads = []
        shared_results = [None] * num_games  # Placeholder for game results
        model = tf.keras.models.load_model('new_model.keras')
        start_time = time.perf_counter()

        for i in range(num_games):
            thread = threading.Thread(target=play_game, args=(model, i, shared_results))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        turns = [turn for game_result in shared_results for turn in game_result]
        print(len(turns))
        end_time = time.perf_counter()
        time_delta = end_time - start_time
        print(f"The function took {time_delta} seconds to complete.")

        i1 = np.array([turn['cards_not_seen_additional_features_tensor'].reshape(85) for turn in turns])
        i2 = np.array([turn['cards_remaining_additional_feature_tensor'].reshape(85) for turn in turns])
        i3 = np.array([turn['cards_not_seen_tensor'].reshape(54) for turn in turns])
        i4 = np.array([turn['cards_person_on_right_has_played_tensor'].reshape(54) for turn in turns])
        i5 = np.array([turn['cards_person_on_left_has_played_tensor'].reshape(54) for turn in turns])
        i6 = np.array([turn['choice_tensor'].reshape(54) for turn in turns])
        i7 = np.array([turn['cards_remaining_tensor'].reshape(54) for turn in turns])
        i8 = np.array([turn['position_tensor'].reshape(6) for turn in turns])

        y_train = np.array([turn['prediction'] for turn in turns])
        x_train = [i1, i2, i3, i4, i5, i6, i7, i8]

        model.fit(x_train, y_train, epochs=1, batch_size=32)
        model.save('new_model.keras')

def play_game(model, game_id, shared_results):
    learning_rate = .1
    multiplier = 1
    show_output = game_id == 0
    hands = shuffle()
    turns = []
    cards_seen = empty_card_dict()
    cards_played_by_hand = [empty_card_dict(), empty_card_dict(), empty_card_dict()]

    turn_number = 0
    landlord_position = 0
    for j in range(len(hands)):
        if card_count(hands[j]) == 20:
            landlord_position = j
            break
        turn_number += 1

    landlord_won = False
    while True:
        turn = turn_number%3
        hand = hands[turn]
        cards_person_on_right_has_played_tensor = dict_to_tensor(cards_played_by_hand[(turn+1)%3])
        cards_person_on_left_has_played_tensor = dict_to_tensor(cards_played_by_hand[(turn-1)%3])
        position_tensor = create_position_tensor(landlord_position, turn, get_previous_played(turns))

        turn_info = get_previous_turn_info(turns)
        options = get_move_options(turn_info, hand)

        choice_dict = options[0]
        cards_remaining_dict = remove_move_from_hand_copy(hand, choice_dict)
        max_prediction = 0

        cards_not_seen_dict = cards_not_seen(hand, cards_played_by_hand)
        cards_not_seen_tensor = dict_to_tensor(cards_not_seen_dict)
        cards_not_seen_additional_features_tensor = additional_features_tensor(cards_not_seen_dict)

        if show_output: print()
        if turn == landlord_position: 
            if show_output: print('L')
        if show_output: print(to_string(hand))
        if random.random() < 0.2:
            choice_dict = random.choice(options)
            options = [choice_dict]
            if show_output: print('random choice')
        else:
            if show_output: print('options:')

        for option_dict in options:
            cards_that_would_be_remaining_dict = remove_move_from_hand_copy(hand, option_dict)

            prediction = model.predict([
                cards_not_seen_additional_features_tensor,
                additional_features_tensor(cards_that_would_be_remaining_dict),
                cards_not_seen_tensor,
                cards_person_on_right_has_played_tensor,
                cards_person_on_left_has_played_tensor,
                dict_to_tensor(option_dict),
                dict_to_tensor(cards_that_would_be_remaining_dict),
                position_tensor,
            ], verbose=0)
            
            if show_output: print(to_string(option_dict), prediction[0][0])

            if prediction[0][0] > max_prediction:
                max_prediction = prediction[0][0]
                choice_dict = option_dict
                cards_remaining_dict = cards_that_would_be_remaining_dict
            
        for card, count in choice_dict.items():
            cards_seen[card] += count
            cards_played_by_hand[turn][card] += count
        
        if show_output: print('choice:', to_string(choice_dict))

        turn_info = get_turn_info(choice_dict)
        if turn_info['type'] == 'bomb': multiplier *= 2
        turns.append({ 
            'turn_info': turn_info,
            'prediction': max_prediction,
            'landlord': landlord_position == turn,
            'cards_not_seen_additional_features_tensor': cards_not_seen_additional_features_tensor,
            'cards_remaining_additional_feature_tensor': additional_features_tensor(cards_remaining_dict),
            'cards_not_seen_tensor': cards_not_seen_tensor,
            'cards_person_on_right_has_played_tensor': cards_person_on_right_has_played_tensor,
            'cards_person_on_left_has_played_tensor': cards_person_on_left_has_played_tensor,
            'choice_tensor': dict_to_tensor(choice_dict),
            'cards_remaining_tensor': dict_to_tensor(cards_remaining_dict),
            'position_tensor': position_tensor,
        })
        if show_output: print (position_tensor)
        remove_choice_from_hand(hand, choice_dict)
        if card_count(hands[turn]) == 0:
            if landlord_position == turn:
                landlord_won = True
                if show_output: print('landlord_won')
            else: 
                if show_output: print('landlord_lost')
            break
        turn_number += 1

    for turn in turns:
        # reward_multiplier = 1 + (multiplier - 1) * ((turn['landlord'] == True) + 1) * learning_rate
        if turn['landlord'] == landlord_won:
            turn['prediction'] += learning_rate * (1 - turn['prediction'])  
        else:
            turn['prediction'] -= learning_rate * turn['prediction']

    shared_results[game_id] = turns

if __name__ == "__main__":
    train()

2* 4* .0001

