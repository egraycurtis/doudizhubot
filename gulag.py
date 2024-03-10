import random
import numpy as np
from train4 import position_tensor3x2
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from turn_info import get_turn_info
from cards import empty_card_dict, shuffle, rank


learning_rate = .01

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
        len(actions)

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

def remove_move_from_hand_copy(hand: dict[str, int], move: dict[str, int]):
    hand_copy = hand.copy() 
    for card, count in move.items():
        hand_copy[card] -= count
        # if hand_copy[card] == 0:
        #     del hand_copy[card]  # Optionally, remove the card if count goes to 0
    return hand_copy

def dict_to_tensor(card_dict: dict[str, int]):
    tensor = np.zeros((4, 16))
    for card, count in card_dict.items():
        for row in range(count):
            if row < 4:
                tensor[row, rank(card)] = 1

    return np.expand_dims(tensor, axis=0)

def position_tensor(landlord: int, turn: int, prevTurnOffset: int):
    # col 1 is landlord position
    # col 2 is position
    # col 3 is last turn position
    tensor = np.zeros((3, 3))

    tensor[landlord, 0] = 1
    tensor[turn, 1] = 1
    tensor[(turn+prevTurnOffset)%3, 2] = 1

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


def gulag():
    model1 = tf.keras.models.load_model('my_model.keras')
    model2 = tf.keras.models.load_model('c4_model.keras')
    model3 = tf.keras.models.load_model('c2_model.keras')
    models = [model1, model2, model3]
    model_wins = [0, 0, 0]
    while True:
        hands = shuffle()
        turns = []
        cardsSeen = empty_card_dict()
        cardsSeenByHand = [empty_card_dict(), empty_card_dict(), empty_card_dict()]

        i = 0
        landlord_position = 0
        for j in range(len(hands)):
            if card_count(hands[j]) == 20:
                landlord_position = j
                i = j
                break

        print('landlord_position', landlord_position)
        for h in hands:
            print (card_count(h))
        while True:
            hand = hands[i%3]
            model = models[i%3]
            cardsPersonOnRightHasPlayed = dict_to_tensor(cardsSeenByHand[(i+1)%3])
            cardsPersonOnLeftHasPlayed = dict_to_tensor(cardsSeenByHand[(i-1)%3])
            positionStuff = position_tensor(landlord_position, i%3, get_previous_played(turns))
            if i%3 == 1:
                positionStuff = position_tensor3x2(landlord_position, i%3, get_previous_played(turns))


            turn_info = get_previous_turn_info(turns)
            options = get_move_options(turn_info, hand)

            choice = options[0]
            cardsRemaining = dict_to_tensor(remove_move_from_hand_copy(hand, choice))
            max_prediction = 0

            print()
            print(to_string(hand))

            for option in options:
                cardsInOption = dict_to_tensor(option)
                cardsThatWouldBeRemaining = dict_to_tensor(remove_move_from_hand_copy(hand, option))

                prediction = model.predict([
                    cardsPersonOnRightHasPlayed,
                    cardsPersonOnLeftHasPlayed,
                    cardsInOption,
                    cardsThatWouldBeRemaining,
                    positionStuff,
                ], verbose=0)
                
                # print(to_string(option), prediction[0][0])

                if prediction[0][0] > max_prediction:
                    max_prediction = prediction[0][0]
                    choice = option
                    cardsRemaining = cardsThatWouldBeRemaining
                
            for card, count in choice.items():
                cardsSeen[card] += count
                cardsSeenByHand[i%3][card] += count
            
            print('choice:', to_string(choice))

            turn_info = get_turn_info(choice)
            turns.append({ 
                'turn_info': turn_info,
                'choice': dict_to_tensor(choice),
                'prediction': max_prediction,
                'landlord': landlord_position == i%3,
                'cardsPersonOnRightHasPlayed': cardsPersonOnRightHasPlayed,
                'cardsPersonOnLeftHasPlayed': cardsPersonOnLeftHasPlayed,
                'positionStuff': positionStuff,
                'cardsRemaining': cardsRemaining,
            })
            remove_choice_from_hand(hand, choice)
            if card_count(hands[i%3]) == 0:
                if landlord_position == i%3:
                    print('landlord_won')
                    model_wins[landlord_position%3] += 2
                    model_wins[(landlord_position-1)%3] -= 1
                    model_wins[(landlord_position+1)%3] -= 1
                else: 
                    print('landlord_lost')
                    model_wins[landlord_position%3] -= 2
                    model_wins[(landlord_position-1)%3] += 1
                    model_wins[(landlord_position+1)%3] += 1
                break
            i += 1

        print(model_wins)

if __name__ == "__main__":
    gulag()


