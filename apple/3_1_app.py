import pandas as pd

data = pd.read_csv("gpass.csv")
data = data.dropna()
print(data.admit)

x_data = []
for i, rows in data.iterrows():
    x_data.append([rows["gre"], rows["gpa"], rows["rank"]])

y_data = data["admit"].values

import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(
            64, activation="tanh"
        ),  # 1차원이기 떄문에 명시해야된다. 2차원이면 안해도된다.
        # tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(128, activation="tanh"),
    ]
)  #
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("\n--- 훈련 시작 ---")
model.fit(np.array(x_data), np.array(y_data), epochs=100)
print("--- 훈련 완료 ---\n")

new_loss, new_model_accuracy = model.evaluate(
    np.array(x_data), np.array(y_data), verbose=1
)


loaded_model = tf.keras.models.load_model("expect_rank.keras")
old_loss, old_model_accuracy = loaded_model.evaluate(
    np.array(x_data), np.array(y_data), verbose=1
)
print(f"올드 모델의 테스트 정확도: {old_model_accuracy:.4f}")
print(f"새로운 모델의 테스트 정확도: {new_model_accuracy:.4f}")
print(f"올드 모델의 테스트 손실: {old_loss:.4f}")
print(f"새로운 모델의 테스트 손실: {new_loss:.4f}")
print("380,3.21,3에 대한 기존 예측:", loaded_model.predict(np.array([[380, 3.21, 3]])))
print("380,3.21,3에 대한 신규 예측:", model.predict(np.array([[380, 3.21, 3]])))

tf.keras.models.save_model(model, "expect_rank.keras")


# print("x=5에 대한 신규 예측:", loaded_model.predict(np.array([5])))

# tf.keras.models.save_model(model, "expect_rank.keras")
