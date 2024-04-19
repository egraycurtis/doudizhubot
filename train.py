import psycopg2
import numpy as np
import tensorflow as tf
import time

def train(db_path="postgresql://postgres:password@localhost:5432"):
    conn = psycopg2.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("select count(*) from game_turns")
        count = cursor.fetchone()[0]

        if count >= 10:
            cursor.execute("select id, turn_data from game_turns LIMIT 100")
            data = cursor.fetchall()
            turn_data = [turn[1]['turns_by_position'] for turn in data]

            turns_by_position = [[],[],[]]
            for tbp in turn_data:
                for i in range(3):
                    for turn in tbp[i]:
                        turns_by_position[i].append(turn)
            ids_to_delete = [turn[0] for turn in data]
            
            p0 = tf.keras.models.load_model('p0.keras')
            p1 = tf.keras.models.load_model('p1.keras')
            p2 = tf.keras.models.load_model('p2.keras')
            models = [p0, p1, p2]
            for i in range(3):
                turns = turns_by_position[i]
                model = models[i]

                i1 = np.array([np.array(turn['cards_not_seen_additional_features_tensor']).reshape(85) for turn in turns])
                i2 = np.array([np.array(turn['cards_remaining_additional_feature_tensor']).reshape(85) for turn in turns])
                i3 = np.array([np.array(turn['cards_not_seen_tensor']).reshape(54) for turn in turns])
                i4 = np.array([np.array(turn['cards_person_on_right_has_played_tensor']).reshape(54) for turn in turns])
                i5 = np.array([np.array(turn['cards_person_on_left_has_played_tensor']).reshape(54) for turn in turns])
                i6 = np.array([np.array(turn['choice_tensor']).reshape(54) for turn in turns])
                i7 = np.array([np.array(turn['cards_remaining_tensor']).reshape(54) for turn in turns])
                i8 = np.array([np.array(turn['last_played_tensor']).reshape(2) for turn in turns])
                i9 = np.array([np.array(turn['cards_person_on_left_has_left_tensor']).reshape(5) for turn in turns])
                i10 = np.array([np.array(turn['cards_person_on_right_has_left_tensor']).reshape(5) for turn in turns])

                y_train = np.array([turn['prediction'] for turn in turns])
                x_train = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]

                model.fit(x_train, y_train, epochs=1, batch_size=64)
                model.save(f"p{i}.keras")
            
            delete_query = "delete from game_turns where id = any(%s)"
            cursor.execute(delete_query, (ids_to_delete,))
            conn.commit()
            print(f"Trained on {len(ids_to_delete)} entries and deleted them from the database.")

    except Exception as e:
        print(e)
        conn.rollback()
    finally:
        conn.close()
       

if __name__ == "__main__":
    while True:
        train()
        time.sleep(5)
