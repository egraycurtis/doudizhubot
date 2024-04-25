import random
import numpy as np
import tensorflow as tf
from filtered_options import filtered_options
from action_space import action_space
from single.train import *
from turn_info import get_turn_info
from cards import empty_card_dict, shuffle

def gulag():
    model1 = tf.keras.models.load_model('new_model9lowering_learning_rate.keras')
    model2 = tf.keras.models.load_model('new_model.keras')
    model3 = tf.keras.models.load_model('model_save_2.keras')
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

            all_cards_remaining_dict = []
            feature_tensors_list = [[] for _ in range(8)]  

            for option_dict in options:
                cards_that_would_be_remaining_dict = remove_move_from_hand_copy(hand, option_dict)
                all_cards_remaining_dict.append(cards_that_would_be_remaining_dict)

                feature_tensors = [
                    cards_not_seen_additional_features_tensor.reshape(85),
                    additional_features_tensor(cards_that_would_be_remaining_dict).reshape(85),
                    cards_not_seen_tensor.reshape(54),
                    cards_person_on_right_has_played_tensor.reshape(54),
                    cards_person_on_left_has_played_tensor.reshape(54),
                    dict_to_tensor(option_dict).reshape(54),
                    dict_to_tensor(cards_that_would_be_remaining_dict).reshape(54),
                    position_tensor.reshape(6),
                ]

                for i, tensor in enumerate(feature_tensors):
                    feature_tensors_list[i].append(tensor)

            # Prepare the input tensors for the model
            model_input_tensors = [np.array(feature_list) for feature_list in feature_tensors_list]

            predictions = model.predict(model_input_tensors, verbose=0)

            if predictions.ndim > 1:
                predictions = predictions.flatten()

            max_prediction_index = np.argmax(predictions)
            choice_dict = options[max_prediction_index]
            cards_remaining_dict = all_cards_remaining_dict[max_prediction_index]
            max_prediction = predictions[max_prediction_index]

                
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
                    model_wins[landlord_position%3] += 1
                else: 
                    print('landlord_lost')
                    model_wins[(landlord_position-1)%3] += 1
                    model_wins[(landlord_position+1)%3] += 1
                break
            turn_number += 1

        print(model_wins)

if __name__ == "__main__":
    gulag()


