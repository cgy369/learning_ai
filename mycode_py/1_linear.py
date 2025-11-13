import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train = [1, 2, 3, 4]
y_train = [5, 6, 7, 8]

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(64, activation="tanh", input_shape=(1,))
)  # 머신러닝이기 떄문에 뉴런 없이 진행한다.
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
model.compile(loss="mean_squared_error", optimizer="sgd")

history = model.fit(np.array(x_train), np.array(y_train), epochs=500, verbose=1)
model.summary()
print(model.predict(np.array([1, 2, 3, 4, 9, 10])))
# print(model)

plt.plot(history.history["loss"])

# 그래프의 제목을 설정합니다.
plt.title("Model loss")
# 그래프의 Y축 레이블을 설정합니다.
plt.ylabel("Loss")
# 그래프의 X축 레이블을 설정합니다.
plt.xlabel("Epoch")

# 범례(legend)를 추가합니다. 현재는 훈련 데이터의 손실만 있으므로 'Train'만 표시합니다.
# (참고: 검증 데이터를 사용했다면 history.history['val_loss']도 함께 그릴 수 있습니다.)
plt.legend(["Train"], loc="upper left")

# 그래프를 화면에 보여줍니다.
plt.show()
