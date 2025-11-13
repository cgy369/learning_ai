import tensorflow as tf
import numpy as np

# --- 1. XOR 데이터 준비 (XOR Data Preparation) ---
# 이전 파일과 동일한, 선형적으로 분리 불가능한 XOR 데이터입니다.
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


# --- 2. 신경망(Neural Network) 모델 구성 ---
# XOR 문제를 해결하기 위해, 여러 개의 레이어를 쌓은 신경망 모델을 구성합니다.
# 이 구조는 입력층(Input Layer) - 은닉층(Hidden Layer) - 출력층(Output Layer)으로 이루어집니다.
model = tf.keras.Sequential()

# 은닉층 (Hidden Layer)
# - units=2: 은닉층에 2개의 뉴런을 둡니다.
# - input_dim=2: 입력 데이터의 특성이 2개입니다.
# - activation='sigmoid': ***매우 중요***. 은닉층의 활성화 함수는 반드시 '비선형(non-linear)' 함수(e.g., sigmoid, relu, tanh)여야 합니다.
#   이 비선형성이 모델이 직선이 아닌, 복잡하고 구부러진 결정 경계(non-linear decision boundary)를 학습할 수 있게 해주는 핵심입니다.
#   만약 은닉층의 활성화 함수가 선형이면, 여러 층을 쌓는 의미가 없어지고 결국 하나의 선형 모델과 같아집니다.
model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'))

# 출력층 (Output Layer)
# - units=1: 최종 출력값은 하나(0 또는 1)입니다.
# - activation='sigmoid': 최종 예측을 0과 1 사이의 확률값으로 변환합니다.
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# 모델의 구조를 요약하여 보여줍니다.
# 2개의 Dense 레이어가 있고, 각 레이어의 파라미터(가중치, 편향) 개수를 확인할 수 있습니다.
model.summary()


# --- 3. 모델 컴파일 및 훈련 (Model Compilation & Training) ---
# 학습률을 0.1로 설정하고, 10000번의 에포크 동안 훈련합니다.
# 신경망은 더 복잡한 모델이므로, 수렴하는 데 더 많은 훈련이 필요할 수 있습니다.
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),  metrics=['accuracy'])
history = model.fit(x_data, y_data, epochs=10000, verbose=0)


# --- 4. 결과 확인 (Checking the Success) ---
# 훈련 후, 모델의 예측값과 정확도를 확인합니다.
predictions = model.predict(x_data)
predicted_classes = (predictions > 0.5).astype(int)

print('Predictions: \n', predicted_classes)

# 정확도를 평가합니다.
# 신경망 모델은 비선형 결정 경계를 학습했기 때문에, XOR 문제를 해결할 수 있습니다.
# 따라서 정확도는 1.0에 매우 가까운 값이 나옵니다.
score = model.evaluate(x_data, y_data)
print('Final Accuracy: ', score[1])
