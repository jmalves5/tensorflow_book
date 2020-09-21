#!/usr/bin/env python3

from datetime import datetime
import io
import itertools
from packaging import version
from six.moves import range

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


num_classes = 10
img_rows, img_cols = 28,28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model_input = tf.keras.layers.Input(shape=input_shape)
output = tf.keras.layers.Flatten()(model_input)
output = tf.keras.layers.Dense(128, activation='relu')(output)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(output)
model = tf.keras.Model(model_input, output)

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cb = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard('./logs_keras')
]

# Sets up a timestamped log directory.
logdir = "logs_keras/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(x_train, (-1, 28, 28, 1))
  tf.summary.image("training data examples", images, step=0)

model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test,
                                                                    y_test), callbacks=cb)