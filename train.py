import numpy as np
import tensorflow as tf
import time
import redis
import json

def train():
    j = 0
    while True:
        
        r = redis.Redis(host='localhost', port=6379, db=0)
        try:
            count = r.llen('training_data')
            if count >= 20:
                # start_time = time.perf_counter()
                data = r.lrange('training_data', 0, 99)
                r.ltrim('training_data', 100, -1)
                training_data_list = [json.loads(d) for d in data]
                # end_time = time.perf_counter()
                # time_delta = end_time - start_time
                # print(f"query took {time_delta} seconds")
                j += 1

                model_training_data_by_position = {}
                for training_data in training_data_list:
                    model_training_data_by_position[training_data['model_name']] = [[],[],[]]

                for training_data in training_data_list:
                    model_name = training_data['model_name']
                    for turn in training_data['turns']:
                        model_training_data_by_position[model_name][turn['position']].append(turn)

                for model_name, turns_by_position in model_training_data_by_position.items():
                    for i in range(3):
                        turns = turns_by_position[i]
                        
                        i1 = np.array([np.array(turn['tensors']['cards_not_seen_additional_features_tensor']).reshape(85) for turn in turns])
                        i2 = np.array([np.array(turn['tensors']['cards_remaining_additional_feature_tensor']).reshape(85) for turn in turns])
                        i3 = np.array([np.array(turn['tensors']['cards_not_seen_tensor']).reshape(54) for turn in turns])
                        i4 = np.array([np.array(turn['tensors']['cards_person_on_right_has_played_tensor']).reshape(54) for turn in turns])
                        i5 = np.array([np.array(turn['tensors']['cards_person_on_left_has_played_tensor']).reshape(54) for turn in turns])
                        i6 = np.array([np.array(turn['tensors']['choice_tensor']).reshape(54) for turn in turns])
                        i7 = np.array([np.array(turn['tensors']['cards_remaining_tensor']).reshape(54) for turn in turns])
                        i8 = np.array([np.array(turn['tensors']['last_played_tensor']).reshape(2) for turn in turns])
                        i9 = np.array([np.array(turn['tensors']['cards_person_on_left_has_left_tensor']).reshape(5) for turn in turns])
                        i10 = np.array([np.array(turn['tensors']['cards_person_on_right_has_left_tensor']).reshape(5) for turn in turns])
                        i11 = np.array([np.array(turn['tensors']['transformer_tensor']).reshape(15, 54) for turn in turns])

                        y_train = np.array([turn['prediction'] for turn in turns])
                        x_train = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
                        if model_name == 'transformer' or model_name == 'lstm':
                            x_train.append(i11)

                        model = tf.keras.models.load_model(f"./models/{model_name}/{model_name}{i}.keras")
                        model.fit(x_train, y_train, epochs=1, batch_size=256)
                        model.save(f"./models/{model_name}/{model_name}{i}.keras")
                        if (j % 100) == 1:
                            model.save(f"./models/{model_name}/{model_name}{i}_backup.keras")
                
        except Exception as e:
                print(e)
                pass
        
        time.sleep(1)

if __name__ == "__main__":
    train()