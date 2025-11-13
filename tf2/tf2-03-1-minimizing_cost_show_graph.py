import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. 데이터 준비 (Data Preparation) ---
# y = -x + 1 관계를 가지는 훈련 데이터를 정의합니다.
x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])


# --- 2. 모델 구성 (Model Building) ---
# Keras Sequential 모델을 생성합니다.
model = tf.keras.Sequential()
# 입력 차원이 1이고 출력이 1인 단순 선형 회귀 모델을 위한 Dense 레이어를 추가합니다.
model.add(tf.keras.layers.Dense(units=1, input_dim=1))


# --- 3. 모델 컴파일 (Model Compilation) ---
# 경사 하강법(SGD) 옵티마이저를 정의하고 학습률(learning_rate)을 0.1로 설정합니다.
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
# 손실 함수로 평균 제곱 오차(mse)를, 옵티마이저로 sgd를 사용하여 모델을 컴파일합니다.
model.compile(loss="mse", optimizer=sgd)


# --- 4. 모델 요약 (Model Summary) ---
# 훈련 전에 모델의 구조를 출력합니다.
model.summary()


# --- 5. 모델 훈련 및 기록 (Model Training & History) ---
# model.fit() 함수는 훈련을 실행하고, 훈련 과정에 대한 정보를 담은 History 객체를 반환합니다.
# 이 History 객체에는 각 에포크(epoch)별 손실(loss) 값과 평가지표(metrics) 값이 기록되어 있습니다.
history = model.fit(x_train, y_train, epochs=100, verbose=1)


# --- 6. 예측 (Prediction) ---
# 훈련된 모델을 사용하여 새로운 데이터 [5, 4]에 대한 예측을 수행합니다.
y_predict = model.predict(np.array([5, 4]))
print(y_predict)


# --- 7. 훈련 과정 시각화 (Visualizing Training History) ---
# Matplotlib 라이브러리를 사용하여 훈련 과정 동안 손실(loss)이 어떻게 변화했는지 그래프로 확인합니다.
# 이는 모델이 잘 학습되고 있는지(손실이 점차 감소하는지)를 직관적으로 파악하는 데 매우 중요합니다.

# history.history['loss']에 저장된, 각 에포크별 훈련 손실 값을 그래프로 그립니다.
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
