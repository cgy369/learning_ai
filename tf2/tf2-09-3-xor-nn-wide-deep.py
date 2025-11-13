import tensorflow as tf
import numpy as np

# --- 1. XOR 데이터 준비 (XOR Data Preparation) ---
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


# --- 2. 깊고 넓은 신경망 모델 구성 (Deep & Wide Neural Network) ---
# 이전 예제보다 더 많은 레이어(깊게)와 더 많은 뉴런(넓게)을 가진 모델을 구성합니다.
# 이를 통해 모델의 '용량(capacity)'을 늘려 더 복잡한 문제를 풀 수 있게 합니다.
model = tf.keras.Sequential()

# 4개의 은닉층과 1개의 출력층을 가집니다.
# 각 은닉층은 10개의 뉴런을 가집니다.
model.add(tf.keras.layers.Dense(units=10, input_dim=2, activation='sigmoid')) # Hidden 1
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))             # Hidden 2
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))             # Hidden 3
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))             # Hidden 4
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))              # Output

# 모델 구조 요약
model.summary()


# --- 3. 모델 컴파일: 기울기 소실 문제와 Adam 옵티마이저 ---
#
# *** 기울기 소실 문제 (Vanishing Gradient Problem) ***
# 'sigmoid' 함수를 활성화 함수로 사용하는 깊은 신경망(deep neural network)에서는
# 역전파(backpropagation) 과정에서 기울기(gradient)가 점점 작아져,
# 앞쪽 레이어(입력층에 가까운 레이어)의 가중치가 거의 업데이트되지 않는 문제가 발생할 수 있습니다.
# 이를 '기울기 소실'이라고 하며, 깊은 모델의 학습을 매우 어렵게 만듭니다.
#
# 해결책 1: Adam과 같은 발전된 옵티마이저 사용
# - SGD 대신 Adam 옵티마이저를 사용합니다. Adam은 각 파라미터마다 다른 학습률을 적용하는 '적응형(adaptive)' 옵티마이저로,
#   기울기 소실 문제에 대해 SGD보다 더 강건한 경향이 있습니다.
#
# 해결책 2: ReLU 활성화 함수 사용 (더 근본적인 해결책)
# - 은닉층의 활성화 함수로 'sigmoid' 대신 'ReLU(Rectified Linear Unit)'를 사용하는 것이 현대 딥러닝의 표준입니다.
# - ReLU(f(x) = max(0, x))는 입력이 양수일 때 기울기가 항상 1이므로, 기울기 소실 문제를 크게 완화해줍니다.
#
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])


# --- 4. 모델 훈련 및 결과 확인 (Training & Evaluation) ---
history = model.fit(x_data, y_data, epochs=5000, verbose=0)

# 훈련된 모델의 예측값과 정확도를 확인합니다.
predictions = model.predict(x_data)
predicted_classes = (predictions > 0.5).astype(int)
print('Predictions: \n', predicted_classes)

score = model.evaluate(x_data, y_data)
print('Final Accuracy: ', score[1])
