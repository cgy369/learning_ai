import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

x_train = np.array([1, 2, 3, 4])
# y_train: 출력 데이터 (종속 변수). 모델이 예측해야 할 정답(label)입니다.
# 데이터의 관계를 보면 y = -x + 1 이라는 것을 알 수 있습니다.
# 모델은 이 관계를 학습하여 W는 -1에, b는 1에 가까워지도록 훈련될 것입니다.
y_train = np.array([0, -1, -2, -3])

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(
            32, activation="tanh", input_shape=(1,)
        ),  # 1차원이기 떄문에 명시해야된다. 2차원이면 안해도된다.
        # tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
    ]
)  # 3개의 레이어 마지막 데이터는 0과 1이 나와야 되기 때문에 디자인에 따라 마지막 레이어를 정해둬야한다.

model.add(tf.keras.layers.Dense(1, activation="linear"))

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.compile(optimizer="adam", loss="mean_squared_error")
print("\n--- 훈련 시작 ---")
model.fit(x_train, y_train, epochs=200, verbose=0)
print("--- 훈련 완료 ---\n")
print("x=5에 대한 예측:", model.predict(np.array([5])))
tf.keras.models.save_model(model, "my_shoe_model.h5")
