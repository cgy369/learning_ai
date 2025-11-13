import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 이 예제에서는 훈련 데이터와 별도로 '테스트 데이터'를 준비합니다.

# x_data, y_data: 모델을 훈련시키는 데 사용될 데이터
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],  # One-hot encoded
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# x_test, y_test: 모델의 성능을 '평가'하는 데 사용될 데이터
# 이 데이터는 훈련 과정에서 모델이 전혀 보지 못한 새로운 데이터여야 합니다.
# 이를 통해 모델이 얼마나 새로운 데이터에 대해 일반화를 잘 하는지 측정할 수 있습니다.
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


# --- 2. 모델 구성 및 컴파일 (Model Building & Compilation) ---

# 학습률(Learning Rate): 모델이 가중치를 업데이트하는 보폭(step size)입니다.
# - 너무 크면(e.g., 100): 최적점을 지나쳐 발산(diverge)할 수 있습니다.
# - 너무 작으면(e.g., 1e-10): 학습이 매우 느리거나 지역 최적점(local minimum)에 갇힐 수 있습니다.
# 적절한 학습률을 찾는 것은 매우 중요한 하이퍼파라미터 튜닝 과정입니다.
learning_rate = 0.1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              metrics=['accuracy'])
model.summary()


# --- 3. 모델 훈련 (Model Training) ---
# 모델을 '훈련 데이터(x_data, y_data)'로만 훈련시킵니다.
model.fit(x_data, y_data, epochs=1000, verbose=0)


# --- 4. 모델 평가 및 예측 (Model Evaluation & Prediction) ---

# 예측(Prediction): 훈련된 모델을 사용하여 '테스트 데이터(x_test)'에 대한 예측을 수행합니다.
# model.predict()는 각 클래스에 대한 확률을 반환합니다.
predictions = model.predict(x_test)
# np.argmax()를 사용하여 가장 높은 확률을 가진 클래스의 인덱스를 찾습니다.
predicted_classes = np.argmax(predictions, axis=-1)
print("Prediction (on test data): ", predicted_classes)

# 평가(Evaluation): 모델의 성능을 '테스트 데이터(x_test, y_test)'로 평가합니다.
# model.evaluate()는 손실(loss)과 컴파일 시 지정한 평가지표(metrics)를 계산하여 반환합니다.
# 이 결과가 모델의 일반화 성능을 나타내는 더 신뢰성 있는 지표입니다.
evaluation_results = model.evaluate(x_test, y_test)
print("Accuracy (on test data): ", evaluation_results[1])
