"""A sample CNN network for classification
"""

from __future__ import absolute_import, division, print_function

from keras.layers import Conv2D, Dense, Flatten

# keras modules
from keras.models import Sequential
from keras.optimizers import RMSprop

n_digits = 10
model = Sequential()
model.add(
    Conv2D(
        filters=64,
        kernel_size=3,
        activation="relu",
        strides=2,
        input_shape=(28, 28, 1),
        padding="same",
    )
)
model.add(Conv2D(filters=128, kernel_size=3, activation="relu", strides=2))
model.add(Flatten())
model.add(Dense(n_digits, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
model.summary()
