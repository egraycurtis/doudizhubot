import random
import numpy as np
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from train import *
from turn_info import get_turn_info
from cards import empty_card_dict, shuffle

def gulag():
    model1 = tf.keras.models.load_model('new_model_save8.keras')
    model2 = tf.keras.models.load_model('model_save.keras')
    model3 = tf.keras.models.load_model('new_model.keras')
    models = [model1, model2, model3]
    model_wins = [0, 0, 0]
    while True:
        multiplier = 1
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

        while True:
            turn = turn_number%3
            hand = hands[turn]
            model = models[turn]
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
                
                if prediction[0][0] > max_prediction:
                    max_prediction = prediction[0][0]
                    choice_dict = option_dict
                    cards_remaining_dict = cards_that_would_be_remaining_dict
                
            for card, count in choice_dict.items():
                cards_seen[card] += count
                cards_played_by_hand[turn][card] += count
            
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
            remove_choice_from_hand(hand, choice_dict)
            if card_count(hand) == 0:
                if landlord_position == turn:
                    print('landlord_won')
                    model_wins[landlord_position%3] += 2*multiplier
                    model_wins[(landlord_position-1)%3] -= 1*multiplier
                    model_wins[(landlord_position+1)%3] -= 1*multiplier
                else: 
                    print('landlord_lost')
                    model_wins[landlord_position%3] -= 2*multiplier
                    model_wins[(landlord_position-1)%3] += 1*multiplier
                    model_wins[(landlord_position+1)%3] += 1*multiplier
                break
            turn_number += 1

        print(model_wins)

if __name__ == "__main__":
    gulag()


