import numpy as np
import tensorflow as tf

# --- 1. 데이터 준비 (Data Preparation) ---
# 훈련(training)에 사용할 데이터를 정의합니다.
# 선형 회귀(Linear Regression)는 데이터의 패턴을 학습하여 직선 형태의 모델을 찾는 것이 목표입니다.
# y = Wx + b 형태의 직선을 예측하게 됩니다. (W: 가중치, b: 편향)

# x_train: 입력 데이터 (독립 변수). 모델이 학습할 특성(feature)입니다.
x_train = [1, 2, 3, 4]
# y_train: 출력 데이터 (종속 변수). 모델이 예측해야 할 정답(label)입니다.
# 데이터의 관계를 보면 y = -x + 1 이라는 것을 알 수 있습니다.
# 모델은 이 관계를 학습하여 W는 -1에, b는 1에 가까워지도록 훈련될 것입니다.
y_train = [0, -1, -2, -3]


# --- 2. 모델 구성 (Model Building) ---
# tf.keras.Sequential: Keras에서 모델을 만드는 가장 간단한 방법으로, 레이어(layer)를 순차적으로 쌓아 구성합니다.
model = tf.keras.Sequential()

# tf.keras.layers.Dense: 가장 기본적인 신경망 레이어(Fully Connected Layer).
# 이 레이어는 입력과 출력 사이에 모든 뉴런이 연결된 구조입니다.
# units=1: 레이어의 출력 뉴런(노드) 개수. 선형 회귀에서는 예측값이 하나이므로 1로 설정합니다.
# input_dim=1: 입력 데이터의 차원(특성의 개수). x_train 데이터는 각 항목이 숫자 하나이므로 1입니다.
# (참고: 최신 Keras에서는 input_shape=(1,) 사용을 권장합니다.)
model.add(tf.keras.layers.Dense(units=1, input_dim=1))


# --- 3. 모델 컴파일 (Model Compilation) ---
# 모델을 훈련시키기 전에, 훈련 과정을 설정하는 단계입니다.
# optimizer: 손실 함수(loss function)의 값을 최소화하기 위해 모델의 가중치(W)와 편향(b)을 업데이트하는 방법.
# sgd (Stochastic Gradient Descent): 확률적 경사 하강법. 가장 기본적인 최적화 알고리즘입니다.
# learning_rate=0.1: 학습률. 한 번의 훈련 스텝에서 가중치를 얼마나 크게 업데이트할지 결정합니다.
# (참고: 이전 버전의 'lr' 대신 'learning_rate' 사용을 권장합니다.)
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)

# loss: 손실 함수. 모델의 예측값과 실제 정답값 사이의 오차를 측정하는 방법.
# 'mse' (Mean Squared Error): 평균 제곱 오차. 회귀 문제에서 가장 널리 사용됩니다.
# (예측값 - 정답값)^2 의 평균으로, 오차가 클수록 더 큰 패널티를 부여합니다.
model.compile(loss='mse', optimizer=sgd)


# --- 4. 모델 요약 (Model Summary) ---
# model.summary(): 구성된 모델의 구조를 텍스트로 출력합니다.
# 각 레이어의 이름, 출력 형태, 파라미터(학습될 가중치) 개수 등을 보여주어 모델이 올바르게 구성되었는지 확인하는 데 유용합니다.
model.summary()


# --- 5. 모델 훈련 (Model Training) ---
# model.fit(): 실제 훈련을 실행하는 함수.
# x_train, y_train: 훈련에 사용할 입력 데이터와 정답 데이터.
# epochs=200: 전체 훈련 데이터를 200번 반복하여 학습하라는 의미.
#             에포크(epoch)가 클수록 모델이 데이터를 더 많이 학습하지만, 너무 많으면 과적합(overfitting)의 위험이 있습니다.
# verbose=0: 훈련 과정의 로그 출력을 생략하여 화면을 깔끔하게 유지합니다.
model.fit(x_train, y_train, epochs=200, verbose=0)


# --- 6. 예측 (Prediction) ---
# model.predict(): 훈련된 모델을 사용하여 새로운 입력값에 대한 예측을 수행합니다.
# np.array([5, 4]): 모델이 본 적 없는 새로운 데이터 [5, 4]에 대한 예측을 요청합니다.
y_predict = model.predict(np.array([5, 4]))

# 예측 결과를 출력합니다.
# 모델이 y = -x + 1 관계를 잘 학습했다면,
# x=5에 대해서는 -4에 가까운 값, x=4에 대해서는 -3에 가까운 값을 예측할 것입니다.
print(y_predict)
