import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import numpy as np

x_data = np.array(
    [
        [73.0, 80.0, 75.0],
        [93.0, 88.0, 93.0],
        [89.0, 91.0, 90.0],
        [96.0, 98.0, 100.0],
        [73.0, 66.0, 70.0],
    ]
)

# y_data: 정답 데이터. 각 x_data 샘플에 대한 결과값입니다. (5개의 샘플, 1개의 결과)
# e.g., [기말고사 점수]
y_data = np.array([[152.0], [185.0], [180.0], [196.0], [142.0]])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=64, activation="linear", input_dim=3))
model.add(tf.keras.layers.Dense(units=4, activation="linear", input_dim=64))
model.add(tf.keras.layers.Dense(units=1, activation="linear", input_dim=4))
model.compile(
    loss="mse",
    optimizer="adam",
)
model.fit(x_data, y_data, epochs=500, verbose=1)
print(
    model.predict(
        np.array(
            [
                [73.0, 80.0, 75.0],
                [93.0, 88.0, 93.0],
                [89.0, 91.0, 90.0],
                [96.0, 98.0, 100.0],
                [73.0, 66.0, 70.0],
            ]
        )
    )
)
