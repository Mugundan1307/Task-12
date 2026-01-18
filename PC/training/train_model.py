import numpy as np
import tensorflow as tf

X = np.load("X.npy")
Y = np.load("Y.npy")

X = X[..., np.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=X.shape[1:]),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, Y, epochs=20, batch_size=16, validation_split=0.2)
model.save("../models/kws_model.h5")
