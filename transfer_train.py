import psycopg2
import numpy as np
import tensorflow as tf
import time
import redis
import json

def transfer_train():
    r = redis.Redis(host='localhost', port=6379, db=0)
    try:
        count = r.llen('transfer_data')
        
        if count >= 10:
            start_time = time.perf_counter()
            # Retrieve and remove the first 100 elements atomically
            results = r.lrange('transfer_data', 0, 99)
            r.ltrim('transfer_data', 100, -1)
            turn_data = [json.loads(result) for result in results]
            end_time = time.perf_counter()
            time_delta = end_time - start_time
            print(f"query took {time_delta} seconds")


            turns_by_position = [[],[],[]]
            for tbp in turn_data:
                for t in tbp:
                    turns_by_position[t['position']].append(t)

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

                y_train = np.array([turn['prediction'] for turn in turns])
                x_train = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]

                for name in ['deeperer', 'shallow', 'shallowish']:
                    print(f"{name}{i}.keras")
                    model = tf.keras.models.load_model(f"{name}{i}.keras")
                    model.fit(x_train, y_train, epochs=3, batch_size=256)
                    model.save(f"{name}{i}.keras")

            return count*10
            
    except Exception as e:
        print(e)
        pass
       

if __name__ == "__main__":
    game_count = 0
    while True:
        try:
            count = transfer_train()
            game_count += count
            print(f"games played so far: {game_count}")
        except: pass
