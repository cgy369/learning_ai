import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

xy = np.genfromtxt(
    "gpass.csv", delimiter=",", dtype=np.float32, skip_header=1, filling_values=0
)
# xy = np.loadtxt("gpass.csv", delimiter=",", dtype=np.float32, skiprows=1)
x_data = xy[:, 1:]
y_data = xy[:, [0]]
print(x_data, np.int16(y_data))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(activation="relu", units=64))
model.add(tf.keras.layers.Dense(activation="relu", units=128))
model.add(tf.keras.layers.Dense(activation="sigmoid", units=1))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(np.array(x_data), np.array(y_data), epochs=3000, verbose=2)
print(model.predict(np.array([[700, 3.65, 2], [710, 3.82, 3]])))
# print(x_data, y_data)
