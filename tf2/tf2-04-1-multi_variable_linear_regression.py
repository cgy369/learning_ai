import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 다중 변수(Multi-variable) 선형 회귀를 위한 데이터를 정의합니다.
# 이전 예제와 달리, 입력 데이터(x)가 여러 개의 특성(feature)을 가집니다.
# 예를 들어, 3과목의 시험 점수(x_data)로 최종 시험 점수(y_data)를 예측하는 문제입니다.

# x_data: 훈련 데이터. 각 샘플이 3개의 특성(feature)을 가집니다. (5개의 샘플, 3개의 특성)
# e.g., [퀴즈1 점수, 퀴즈2 점수, 중간고사 점수]
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

# y_data: 정답 데이터. 각 x_data 샘플에 대한 결과값입니다. (5개의 샘플, 1개의 결과)
# e.g., [기말고사 점수]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# --- 2. 모델 구성 (Model Building) ---
# 가설(Hypothesis): H(x1, x2, x3) = w1*x1 + w2*x2 + w3*x3 + b
# 입력 특성이 3개이므로, 학습해야 할 가중치(W)도 3개가 됩니다.

model = tf.keras.Sequential()

# Dense 레이어의 input_dim을 3으로 설정하여 모델이 3개의 특성을 입력으로 받는다는 것을 알려줍니다.
# (참고: 최신 Keras에서는 input_shape=(3,) 사용을 권장합니다.)
model.add(tf.keras.layers.Dense(units=1, input_dim=3))

# 활성화 함수(Activation Function): 뉴런의 최종 출력값을 결정하는 함수입니다.
# 'linear'는 입력값을 그대로 출력하는 활성화 함수(f(x) = x)입니다.
# 회귀(Regression) 문제에서는 예측값이 특정 범위에 제한되지 않아야 하므로, 주로 linear 활성화 함수를 사용합니다.
# Dense 레이어의 기본 활성화 함수가 'linear'이므로, 이 라인은 생략해도 동일하게 동작합니다.
model.add(tf.keras.layers.Activation('linear'))


# --- 3. 모델 컴파일 (Model Compilation) ---
# 옵티마이저로 SGD를, 손실 함수로 mse를 사용합니다.
# learning_rate를 1e-5 (0.00001)로 매우 작게 설정했습니다.
# 이는 입력 데이터의 스케일이 클 때(e.g., 70~100점대) 학습 과정에서 발산(divergence)하는 것을 방지하기 위함입니다.
# (참고: 데이터를 정규화(Normalization)하면 더 높은 학습률을 사용할 수 있어 학습이 빨라집니다.)
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))


# --- 4. 모델 요약 및 훈련 (Model Summary & Training) ---
# 모델의 구조를 출력합니다. 파라미터(Params)가 4개인 것을 볼 수 있습니다. (가중치 3개, 편향 1개)
model.summary()

# 훈련을 100 에포크 동안 실행합니다.
history = model.fit(x_data, y_data, epochs=100, verbose=0)


# --- 5. 예측 (Prediction) ---
# 훈련된 모델을 사용하여 새로운 데이터 [72., 93., 90.]에 대한 최종 점수를 예측합니다.
y_predict = model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
