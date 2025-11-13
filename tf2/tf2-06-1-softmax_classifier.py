import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 다중 클래스 분류(Multi-class Classification)를 위한 데이터를 정의합니다.
# 3개 이상의 클래스 중 하나를 예측하는 문제입니다.

# x_raw: 입력 데이터. 4개의 특성(feature)을 가집니다.
x_raw = [[1, 2, 1, 1],
         [2, 1, 3, 2],
         [3, 1, 3, 4],
         [4, 1, 5, 5],
         [1, 7, 5, 5],
         [1, 2, 5, 6],
         [1, 6, 6, 6],
         [1, 7, 7, 7]]

# y_raw: 정답 데이터. '원-핫 인코딩(One-Hot Encoding)' 형식으로 되어 있습니다.
# 원-핫 인코딩이란?
# - 총 클래스 개수만큼의 길이를 가지는 배열을 만들고,
# - 정답 클래스의 인덱스에만 1을, 나머지는 모두 0으로 채우는 방식입니다.
# - 예: 클래스 C -> [0, 0, 1], 클래스 B -> [0, 1, 0], 클래스 A -> [1, 0, 0]
# 이 예제에서는 3개의 클래스(A, B, C)가 있습니다.
y_raw = [[0, 0, 1],  # Class C
         [0, 0, 1],  # Class C
         [0, 0, 1],  # Class C
         [0, 1, 0],  # Class B
         [0, 1, 0],  # Class B
         [0, 1, 0],  # Class B
         [1, 0, 0],  # Class A
         [1, 0, 0]]  # Class A

# 데이터를 TensorFlow가 사용하기 좋은 NumPy 배열 형태로 변환합니다.
x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

# 클래스의 개수를 변수에 저장합니다.
nb_classes = 3


# --- 2. 모델 구성 (Model Building) ---
model = tf.keras.Sequential()

# 다중 클래스 분류 모델을 구성합니다.
# input_dim=4: 입력 특성이 4개입니다.
# units=nb_classes: 출력 뉴런의 개수는 반드시 클래스의 개수와 동일해야 합니다(여기서는 3개).
#                    각 뉴런은 해당 클래스에 대한 예측값을 출력합니다.
# activation='softmax': 소프트맥스 활성화 함수.
#                      - 모델의 출력값(logits)을 모든 클래스에 대한 '확률' 분포로 변환합니다.
#                      - 모든 출력값의 합은 항상 1이 됩니다.
#                      - 가장 높은 확률값을 가지는 클래스가 모델의 최종 예측이 됩니다.
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=4, activation='softmax'))


# --- 3. 모델 컴파일 (Model Compilation) ---
# loss='categorical_crossentropy': 다중 클래스 분류에서 정답이 '원-핫 인코딩'된 경우 사용하는 표준 손실 함수입니다.
#                                 모델이 예측한 확률 분포와 실제 정답의 원-핫 분포 사이의 차이를 측정합니다.
# optimizer=SGD(...): 경사 하강법 옵티마이저.
# metrics=['accuracy']: 훈련 시 정확도를 측정합니다.
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])

# 모델의 구조를 요약하여 보여줍니다.
model.summary()


# --- 4. 모델 훈련 (Model Training) ---
history = model.fit(x_data, y_data, epochs=2000, verbose=0)


# --- 5. 예측 (Prediction) ---
print('--------------')
# 새로운 데이터 [1, 11, 7, 9]에 대한 예측
prediction_a = model.predict(np.array([[1, 11, 7, 9]]))
# 예측 결과(a)는 3개 클래스에 대한 확률 분포입니다. e.g., [[0.9, 0.05, 0.05]]
print("Prediction (probabilities):", prediction_a)

# np.argmax()는 배열에서 가장 큰 값의 인덱스를 반환합니다.
# axis=1은 각 행(row)별로 가장 큰 값의 인덱스를 찾으라는 의미입니다.
# 이 인덱스가 바로 모델이 예측한 클래스입니다.
# 예: 확률이 [[0.1, 0.2, 0.7]] 이라면 argmax는 2를 반환 -> 클래스 C
predicted_class_a = np.argmax(prediction_a, axis=1)
print("Predicted class (index):", predicted_class_a)


print('--------------')
# 다른 데이터들에 대해서도 예측을 수행하고, argmax로 최종 클래스를 확인합니다.
prediction_b = model.predict(np.array([[1, 3, 4, 3]]))
predicted_class_b = np.argmax(prediction_b, axis=1)
print("Prediction (probabilities):", prediction_b)
print("Predicted class (index):", predicted_class_b)


print('--------------')
# 여러 개의 데이터를 한 번에 예측할 수도 있습니다.
all_predictions = model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
predicted_classes_all = np.argmax(all_predictions, axis=-1) # axis=-1은 마지막 축을 의미하며, axis=1과 동일하게 동작합니다.
print("All predictions (probabilities):\n", all_predictions)
print("All predicted classes (indices):", predicted_classes_all)
