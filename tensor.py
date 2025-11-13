import numpy as np  # NumPy는 수치 계산, 특히 배열(array) 처리에 효율적인 라이브러리입니다.
import tensorflow as tf  # TensorFlow는 구글에서 개발한 오픈소스 머신러닝 라이브러리입니다.

# 1. 데이터 준비 (Training Data)
# --------------------------------------------------------------------------------
# x_train: 입력 데이터 (독립 변수). 여기서는 간단한 선형 관계를 가정합니다.
# y_train: 출력 데이터 (종속 변수). x_train에 -1을 곱한 값에서 1을 뺀 형태입니다 (y = -x - 1).
# 예를 들어, x가 1일 때 y는 0, x가 2일 때 y는 -1 등입니다.
x_train = np.array([1, 2, 3, 4], dtype=np.float32) # Keras는 NumPy 배열을 선호합니다.
y_train = np.array([0, -1, -2, -3], dtype=np.float32) # 데이터 타입을 명시하는 것이 좋습니다.

# 2. 모델 구성 (Model Architecture)
# --------------------------------------------------------------------------------
# tf.keras.Sequential: Keras에서 레이어를 순차적으로 쌓아 모델을 만드는 가장 간단한 방법입니다.
# model = tf.keras.Sequential(): 모델 객체를 생성하고, 일반적으로 'model' 변수에 할당합니다.
# tf.model = ... 보다는 'model = ...'이 일반적인 컨벤션입니다.
model = tf.keras.Sequential()

# tf.keras.layers.Dense: 가장 기본적인 신경망 레이어인 '완전 연결(Fully Connected)' 레이어입니다.
# units=1: 이 레이어의 출력 뉴런(노드)의 개수를 1개로 설정합니다. 선형 회귀에서는 보통 1개입니다.
# input_shape=(1,): 입력 데이터의 형태를 지정합니다.
#                   x_train이 단일 특성(feature)을 가지므로 (1,)로 지정합니다.
#                   이전 버전의 input_dim=1 대신 input_shape=(1,)를 사용하는 것이 권장됩니다.
model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

# 3. 모델 컴파일 (Model Compilation)
# --------------------------------------------------------------------------------
# 모델을 훈련하기 전에, 훈련 과정을 설정하는 '컴파일' 단계가 필요합니다.
# optimizer: 모델의 가중치를 어떻게 업데이트할지 결정하는 알고리즘입니다.
#            SGD (Stochastic Gradient Descent): 가장 기본적인 경사 하강법 최적화 도구입니다.
#            learning_rate=0.1: 한 번 가중치를 업데이트할 때 얼마나 크게 변화시킬지 결정하는 값입니다.
#                               이전 버전의 lr=0.1 대신 learning_rate=0.1을 사용하는 것이 권장됩니다.
sgd = tf.keras.optimizers.SGD(learning_rate=0.1)

# loss: 모델이 예측한 값과 실제 값(정답) 사이의 오차를 계산하는 함수입니다.
#       "mse" (Mean Squared Error): 평균 제곱 오차. 회귀 문제에서 가장 흔히 사용됩니다.
#       (예측값 - 실제값)^2의 평균을 계산하여 오차가 클수록 큰 페널티를 줍니다.
model.compile(loss="mse", optimizer=sgd)

# 4. 모델 요약 (Model Summary)
# --------------------------------------------------------------------------------
# model.summary(): 모델의 구조, 각 레이어의 출력 형태, 파라미터(가중치) 개수 등을 터미널에 출력합니다.
# 모델이 예상대로 구성되었는지 확인하는 데 유용합니다.
model.summary()

# 5. 모델 훈련 (Model Training)
# --------------------------------------------------------------------------------
# model.fit(): 모델을 훈련시키는 함수입니다.
# x_train: 훈련에 사용할 입력 데이터.
# y_train: 훈련에 사용할 정답(레이블) 데이터.
# epochs=200: 전체 훈련 데이터를 200번 반복하여 학습하라는 의미입니다.
#             에포크가 많을수록 모델이 데이터를 더 많이 학습하지만, 과적합의 위험도 있습니다.
model.fit(x_train, y_train, epochs=200, verbose=0) # verbose=0으로 설정하여 훈련 과정 출력을 줄였습니다.

# 6. 예측 (Prediction)
# --------------------------------------------------------------------------------
# model.predict(): 훈련된 모델을 사용하여 새로운 입력에 대한 예측값을 반환합니다.
# np.array([5, 4]): 예측하고자 하는 새로운 입력 데이터입니다.
#                   모델은 이 입력에 대해 학습된 선형 관계를 바탕으로 출력값을 예측합니다.
y_predict = model.predict(np.array([5, 4], dtype=np.float32))

# 예측 결과 출력
print("\nPrediction for [5, 4]:")
print(y_predict)

# 예상 결과: y = -x - 1 이므로,
# x=5일 때 y = -5 - 1 = -6
# x=4일 때 y = -4 - 1 = -5
# 모델이 이 값에 가깝게 예측할 것입니다.
