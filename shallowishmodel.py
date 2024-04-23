import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input


def create_model():
    i1 = Input(shape=(85))
    i2 = Input(shape=(85))
    i3 = Input(shape=(54))
    i4 = Input(shape=(54))
    i5 = Input(shape=(54))
    i6 = Input(shape=(54))
    i7 = Input(shape=(54))
    i8 = Input(shape=(2))
    i9 = Input(shape=(5))
    i10 = Input(shape=(5))
    merged = layers.concatenate([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10])

    dense1 = layers.Dense(512, activation='relu')(merged)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    dense3 = layers.Dense(512, activation='relu')(dense2)
    output = layers.Dense(1, activation='sigmoid')(dense3)

    model = models.Model(inputs=[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10], outputs=output)
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
    i8 = np.random.rand(num_samples, 2)
    i9 = np.random.rand(num_samples, 5)
    i10 = np.random.rand(num_samples, 5)
    y_train = np.full((num_samples, 1), 0.5)
    return [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10], y_train

model = create_model()
model.summary()

x_train, y_train = generate_dummy_data()

model.fit(x_train, y_train, epochs=25, batch_size=32)

for i in range(3):
    model.save(f"shallowish{i}.keras")


# 0 1 1 1 1 1 0 1 0 1 1 1 0 0 0 0
# 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 

