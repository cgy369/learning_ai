import datetime
import numpy as np
import os
import tensorflow as tf

# --- 1. XOR 데이터 준비 (XOR Data Preparation) ---
# 이전 예제와 동일한 XOR 데이터입니다.
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


# --- 2. 신경망 모델 구성 (Neural Network Model Building) ---
# XOR 문제를 해결하기 위한 2층 신경망 모델입니다.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid')) # 은닉층
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))              # 출력층

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()


# --- 3. TensorBoard 콜백 설정 (TensorBoard Callback Setup) ---
# TensorBoard는 TensorFlow의 시각화 도구로, 모델의 훈련 과정을 실시간으로 모니터링하고 분석하는 데 사용됩니다.
# 손실(loss), 정확도(accuracy) 변화, 모델 그래프, 가중치 분포 등을 시각적으로 확인할 수 있습니다.

# 로그 파일이 저장될 디렉토리를 설정합니다.
# 각 훈련 실행마다 고유한 로그 디렉토리를 생성하여 여러 번의 실험 결과를 비교하기 용이하게 합니다.
# 예: ./logs/fit/20251104-143000
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# tf.keras.callbacks.TensorBoard 콜백을 생성합니다.
# log_dir: 로그 파일이 저장될 경로.
# histogram_freq=1: 각 에포크마다 모델의 가중치(weights)와 편향(biases)의 히스토그램을 기록합니다.
#                   이를 통해 훈련 중 모델 내부 파라미터의 변화를 추적할 수 있습니다.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# --- 4. 모델 훈련 (Model Training) ---
# model.fit() 함수에 callbacks 인자로 TensorBoard 콜백을 전달합니다.
# 훈련이 진행되는 동안 TensorBoard가 지정된 log_dir에 훈련 데이터를 기록합니다.
history = model.fit(x_data, y_data, epochs=10000, callbacks=[tensorboard_callback], verbose=0)


# --- 5. 예측 및 평가 (Prediction & Evaluation) ---
predictions = model.predict(x_data)
predicted_classes = (predictions > 0.5).astype(int)
print('Predictions: \n', predicted_classes)

score = model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])


# --- 6. TensorBoard 실행 방법 ---
# 훈련이 완료된 후, 터미널(명령 프롬프트)을 열어 다음 명령어를 실행하여 TensorBoard를 시작할 수 있습니다.
#
# 1. 이 스크립트가 있는 디렉토리로 이동합니다.
#    cd E:\workspace\python\Study\tf2
#
# 2. TensorBoard를 실행합니다.
#    tensorboard --logdir=./logs/fit
#
# 3. 웹 브라우저에서 TensorBoard가 제공하는 주소(보통 http://localhost:6006)로 접속합니다.
#
# TensorBoard에 대한 더 자세한 정보는 다음 링크를 참조하세요:
# https://www.tensorflow.org/tensorboard/get_started