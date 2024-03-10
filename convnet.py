import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def create_model():
    conv_input1 = Input(shape=(4, 16, 1))
    conv_input2 = Input(shape=(4, 16, 1))
    conv_input3 = Input(shape=(4, 16, 1))
    conv_input4 = Input(shape=(4, 16, 1))
    dense_input = Input(shape=(3, 3))

    def create_conv_pathway(input_tensor):
        conv_1x1 = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        conv_2x1 = layers.Conv2D(filters=1, kernel_size=(2, 1), activation='relu', padding='same')(input_tensor)
        conv_3x1 = layers.Conv2D(filters=1, kernel_size=(3, 1), activation='relu', padding='same')(input_tensor)
        conv_4x1 = layers.Conv2D(filters=1, kernel_size=(4, 1), activation='relu', padding='same')(input_tensor)
        conv_1x5 = layers.Conv2D(filters=1, kernel_size=(1, 5), activation='relu', padding='same')(input_tensor)
        conv_1x6 = layers.Conv2D(filters=1, kernel_size=(1, 6), activation='relu', padding='same')(input_tensor)
        conv_1x7 = layers.Conv2D(filters=1, kernel_size=(1, 7), activation='relu', padding='same')(input_tensor)
        conv_1x8 = layers.Conv2D(filters=1, kernel_size=(1, 8), activation='relu', padding='same')(input_tensor)
        conv_1x9 = layers.Conv2D(filters=1, kernel_size=(1, 9), activation='relu', padding='same')(input_tensor)
        conv_1x10 = layers.Conv2D(filters=1, kernel_size=(1, 10), activation='relu', padding='same')(input_tensor)
        conv_1x11 = layers.Conv2D(filters=1, kernel_size=(1, 11), activation='relu', padding='same')(input_tensor)
        conv_1x12 = layers.Conv2D(filters=1, kernel_size=(1, 12), activation='relu', padding='same')(input_tensor)
        conv_2x3 = layers.Conv2D(filters=1, kernel_size=(2, 3), activation='relu', padding='same')(input_tensor)
        conv_2x4 = layers.Conv2D(filters=1, kernel_size=(2, 4), activation='relu', padding='same')(input_tensor)
        conv_2x5 = layers.Conv2D(filters=1, kernel_size=(2, 5), activation='relu', padding='same')(input_tensor)
        # conv_2x6 = layers.Conv2D(filters=1, kernel_size=(2, 6), activation='relu', padding='same')(input_tensor)
        # conv_2x7 = layers.Conv2D(filters=1, kernel_size=(2, 7), activation='relu', padding='same')(input_tensor)
        # conv_2x8 = layers.Conv2D(filters=1, kernel_size=(2, 8), activation='relu', padding='same')(input_tensor)
        # conv_2x9 = layers.Conv2D(filters=1, kernel_size=(2, 9), activation='relu', padding='same')(input_tensor)
        conv_2x10 = layers.Conv2D(filters=1, kernel_size=(2, 10), activation='relu', padding='same')(input_tensor)
        conv_3x3 = layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        # conv_3x4 = layers.Conv2D(filters=1, kernel_size=(3, 4), activation='relu', padding='same')(input_tensor)
        # conv_3x5 = layers.Conv2D(filters=1, kernel_size=(3, 5), activation='relu', padding='same')(input_tensor)
        # conv_3x6 = layers.Conv2D(filters=1, kernel_size=(3, 6), activation='relu', padding='same')(input_tensor)

        combined = layers.concatenate([conv_1x1, conv_2x1, conv_3x1, conv_4x1, conv_1x5, conv_1x6, conv_1x7, conv_1x8, conv_1x9, conv_1x10, conv_1x11, conv_1x12, conv_2x3, conv_2x4, conv_2x5, conv_2x10, conv_3x3])#, conv_3x4, conv_3x5, conv_3x6, conv_2x6, conv_2x7, conv_2x8, conv_2x9])
        combined = layers.Flatten()(combined)
        return combined

    pathway1 = create_conv_pathway(conv_input1)
    pathway2 = create_conv_pathway(conv_input2)
    pathway3 = create_conv_pathway(conv_input3)
    pathway4 = create_conv_pathway(conv_input4)

    dense_output = layers.Dense(64, activation='relu')(dense_input)
    dense_output = layers.Flatten()(dense_output)
    dense_output = layers.Dropout(0.2)(dense_output)


    merged = layers.concatenate([pathway1, pathway2, pathway3, pathway4, dense_output])
    final_dense = layers.Dense(64, activation='relu')(merged)
    dense_output = layers.Dropout(0.5)(dense_output)
    output = layers.Dense(1, activation='sigmoid')(final_dense)

    model = models.Model(inputs=[conv_input1, conv_input2, conv_input3, conv_input4, dense_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def generate_dummy_data(num_samples=2000):
    x_train_conv1 = np.random.rand(num_samples, 4, 16, 1)
    x_train_conv2 = np.random.rand(num_samples, 4, 16, 1)
    x_train_conv3 = np.random.rand(num_samples, 4, 16, 1)
    x_train_conv4 = np.random.rand(num_samples, 4, 16, 1)
    x_train_dense = np.random.rand(num_samples, 3, 3)
    y_train = np.full((num_samples, 1), 0.5)
    return [x_train_conv1, x_train_conv2, x_train_conv3, x_train_conv4, x_train_dense], y_train

model = create_model()
model.summary()

x_train, y_train = generate_dummy_data()

model.fit(x_train, y_train, epochs=25, batch_size=32)

model.save('my_model.keras')

print("Model saved successfully.")

loaded_model = tf.keras.models.load_model('my_model.keras')


x_test_conv1 = np.random.rand(1, 4, 16, 1)
x_test_conv2 = np.random.rand(1, 4, 16, 1)
x_test_conv3 = np.random.rand(1, 4, 16, 1)
x_test_conv4 = np.random.rand(1, 4, 16, 1)
x_test_dense = np.random.rand(1, 3, 3)

prediction = loaded_model.predict([x_test_conv1, x_test_conv2, x_test_conv3, x_test_conv4, x_test_dense])
print("Prediction:", prediction)

# 0 1 1 1 1 1 0 1 0 1 1 1 0 0 0 0
# 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 