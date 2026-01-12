import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf


# x_raw: 입력 데이터. 4개의 특성(feature)을 가집니다.
x_raw = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]

# y_raw: 정답 데이터. '원-핫 인코딩(One-Hot Encoding)' 형식으로 되어 있습니다.
# 원-핫 인코딩이란?
# - 총 클래스 개수만큼의 길이를 가지는 배열을 만들고,
# - 정답 클래스의 인덱스에만 1을, 나머지는 모두 0으로 채우는 방식입니다.
# - 예: 클래스 C -> [0, 0, 1], 클래스 B -> [0, 1, 0], 클래스 A -> [1, 0, 0]
# 이 예제에서는 3개의 클래스(A, B, C)가 있습니다.
y_raw = [
    [0, 0, 1],  # Class C
    [0, 0, 1],  # Class C
    [0, 0, 1],  # Class C
    [0, 1, 0],  # Class B
    [0, 1, 0],  # Class B
    [0, 1, 0],  # Class B
    [1, 0, 0],  # Class A
    [1, 0, 0],
]  # Class A

pred_class = ["A", "B", "C"]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=3, input_dim=4, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    metrics=["accuracy"],
)
history = model.fit(np.array(x_raw), np.array(y_raw), epochs=100, verbose=1)

print("--------------")
# 새로운 데이터 [1, 11, 7, 9]에 대한 예측
prediction_a = model.predict(np.array([[1, 11, 7, 9]]))
# 예측 결과(a)는 3개 클래스에 대한 확률 분포입니다. e.g., [[0.9, 0.05, 0.05]]
print("Prediction (probabilities):", prediction_a)

# np.argmax()는 배열에서 가장 큰 값의 인덱스를 반환합니다.
# axis=1은 각 행(row)별로 가장 큰 값의 인덱스를 찾으라는 의미입니다.
# 이 인덱스가 바로 모델이 예측한 클래스입니다.
# 예: 확률이 [[0.1, 0.2, 0.7]] 이라면 argmax는 2를 반환 -> 클래스 C


predicted_class_a = np.argmax(prediction_a, axis=1)
for idx in predicted_class_a:
    print("예측된 A인덱스:", predicted_class_a)
    print("예측된 A클래스:", pred_class[idx])


print("--------------")
# 다른 데이터들에 대해서도 예측을 수행하고, argmax로 최종 클래스를 확인합니다.
prediction_b = model.predict(np.array([[1, 3, 4, 3]]))

predicted_class_b = np.argmax(prediction_b, axis=1)
for idx in predicted_class_b:
    print("예측된 B인덱스:", predicted_class_b)
    print("예측된 B클래스:", pred_class[idx])


print("--------------")
# 여러 개의 데이터를 한 번에 예측할 수도 있습니다.
all_predictions = model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
predicted_classes_all = np.argmax(
    all_predictions, axis=-1
)  # axis=-1은 마지막 축을 의미하며, axis=1과 동일하게 동작합니다.
for i, idx in enumerate(predicted_classes_all):
    print(f"{i}번째 입력 → index: {idx}, class: {pred_class[idx]}")


print("훈련 데이터 정확도 계산...")

all_pred = model.predict(np.array(x_raw))
all_pred_class = np.argmax(all_pred, axis=1)
true_class = np.argmax(np.array(y_raw), axis=1)

correct_count = np.sum(all_pred_class == true_class)

print("\n정확도: {:.2f}%".format(correct_count / len(y_raw) * 100))

for i in range(len(y_raw)):
    print(
        "[{}] Pred: {} ({}) | True: {} ({})".format(
            all_pred_class[i] == true_class[i],
            all_pred_class[i],
            pred_class[all_pred_class[i]],
            true_class[i],
            pred_class[true_class[i]],
        )
    )
