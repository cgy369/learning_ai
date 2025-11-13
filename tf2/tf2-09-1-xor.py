import tensorflow as tf
import numpy as np

# --- 1. XOR 문제 정의 (The XOR Problem) ---
# XOR(Exclusive OR) 문제는 머신러닝 역사에서 매우 중요한 문제입니다.
# 데이터의 분포는 다음과 같습니다.
# (0, 0) -> 0
# (0, 1) -> 1
# (1, 0) -> 1
# (1, 1) -> 0
#
# 이 데이터를 2D 평면에 점으로 찍어보면, 어떤 직선을 긋더라도
# 0과 1을 완벽하게 나눌 수 없다는 것을 알 수 있습니다.
# 즉, XOR 문제는 '선형적으로 분리 불가능(linearly inseparable)'합니다.

# 이 스크립트는 단일 레이어(Logistic Regression) 모델이
# 왜 이 문제를 해결할 수 없는지 보여주는 예제입니다.

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


# --- 2. 모델 구성 및 컴파일 (A Single-Layer Linear Classifier) ---
# 이전 예제에서 사용했던 로지스틱 회귀 모델을 그대로 사용합니다.
# 이 모델은 데이터를 나누기 위해 단 하나의 직선(결정 경계)만을 사용할 수 있습니다.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),  metrics=['accuracy'])
model.summary()


# --- 3. 모델 훈련 (Model Training) ---
# 모델을 1000 에포크 동안 훈련시킵니다.
history = model.fit(x_data, y_data, epochs=1000, verbose=0)


# --- 4. 결과 확인 (Checking the Failure) ---
# 훈련 후, 모델의 예측값과 정확도를 확인합니다.
predictions = model.predict(x_data)
predicted_classes = (predictions > 0.5).astype(int)

print('Predictions: \n', predicted_classes)

# 정확도를 평가합니다.
# 선형 모델은 이 문제를 해결할 수 없으므로, 정확도는 0.5 또는 0.75 근처에 머물게 됩니다.
# 이는 모델이 데이터를 제대로 분류하지 못하고 있다는 것을 의미합니다.
score = model.evaluate(x_data, y_data)
print('Final Accuracy: ', score[1])

# 이 문제를 해결하기 위해서는, 여러 개의 레이어를 쌓아
# 비선형(non-linear) 결정 경계를 만들 수 있는 '신경망(Neural Network)'이 필요합니다.
# 다음 예제에서 신경망을 사용하여 이 문제를 해결하는 방법을 알아봅니다.
