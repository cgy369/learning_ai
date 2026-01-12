import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

xy = np.loadtxt("data-03-diabetes.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
# y_data: 정답 데이터. 0 또는 1의 값을 가집니다. (6개의 샘플, 1개의 결과)
# e.g., [0: 불합격, 1: 합격]
y_data = xy[:, [-1]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(activation="sigmoid", units=1, input_dim=8))

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)
history = model.fit(np.array(x_data), np.array(y_data), epochs=3000, verbose=1)
print("Last epoch accuracy: {0}".format(history.history["accuracy"][-1]))
print(model.predict(np.array(x_data)))
