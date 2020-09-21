#!/usr/bin/env python3

import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28,28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cb = [
    tf.keras.callbacks.EarlyStopping(patience=0),
    tf.keras.callbacks.TensorBoard('./logs_keras')
]

model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test), callbacks=cb)

model.summary()
