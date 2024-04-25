import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
import json 

def load_game_data(file_path='game_data.json'):
    turns = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                turn = json.loads(line.strip())
                turns.append(turn)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Faulty line: {line}")
    return turns

def create_model():
    i1 = Input(shape=(85))
    i2 = Input(shape=(85))
    i3 = Input(shape=(54))
    i4 = Input(shape=(54))
    i5 = Input(shape=(54))
    i6 = Input(shape=(54))
    i7 = Input(shape=(54))
    i8 = Input(shape=(6))
    merged = layers.concatenate([i1, i2, i3, i4, i5, i6, i7, i8])

    dense1 = layers.Dense(200, activation='relu')(merged)
    dense2 = layers.Dense(100, activation='relu')(dense1)
    output = layers.Dense(1, activation='sigmoid')(dense2)

    model = models.Model(inputs=[i1, i2, i3, i4, i5, i6, i7, i8], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def generate_dummy_data(num_samples=2000):
    i1 = np.random.rand(num_samples, 85)
    i2 = np.random.rand(num_samples, 85)
    i3 = np.random.rand(num_samples, 54)
    i4 = np.random.rand(num_samples, 54)
    i5 = np.random.rand(num_samples, 54)
    i6 = np.random.rand(num_samples, 54)
    i7 = np.random.rand(num_samples, 54)
    i8 = np.random.rand(num_samples, 6)
    y_train = np.full((num_samples, 1), 0.5)
    return [i1, i2, i3, i4, i5, i6, i7, i8], y_train

model = create_model()
model.summary()

x_train, y_train = generate_dummy_data()

model.fit(x_train, y_train, epochs=25, batch_size=32)

model.save('new_model_bad.keras')

print("Model saved successfully.")

loaded_model = tf.keras.models.load_model('new_model_bad.keras')


i1 = np.random.rand(1, 85)
i2 = np.random.rand(1, 85)
i3 = np.random.rand(1, 54)
i4 = np.random.rand(1, 54)
i5 = np.random.rand(1, 54)
i6 = np.random.rand(1, 54)
i7 = np.random.rand(1, 54)
i8 = np.random.rand(1, 6)

prediction = loaded_model.predict([i1, i2, i3, i4, i5, i6, i7, i8])
print("Prediction:", prediction)

# turns = load_game_data()
# i1 = np.array([turn['cards_not_seen_additional_features_tensor'].reshape(85) for turn in turns])
# i2 = np.array([turn['cards_remaining_additional_feature_tensor'].reshape(85) for turn in turns])
# i3 = np.array([turn['cards_not_seen_tensor'].reshape(54) for turn in turns])
# i4 = np.array([turn['cards_person_on_right_has_played_tensor'].reshape(54) for turn in turns])
# i5 = np.array([turn['cards_person_on_left_has_played_tensor'].reshape(54) for turn in turns])
# i6 = np.array([turn['choice_tensor'].reshape(54) for turn in turns])
# i7 = np.array([turn['cards_remaining_tensor'].reshape(54) for turn in turns])
# i8 = np.array([turn['position_tensor'].reshape(6) for turn in turns])

# y_train = np.array([turn['prediction'] for turn in turns])
# x_train = [i1, i2, i3, i4, i5, i6, i7, i8]

# model.fit(x_train, y_train, epochs=1, batch_size=32)
# model.save('new_model_bad.keras')


# 0 1 1 1 1 1 0 1 0 1 1 1 0 0 0 0
# 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 

