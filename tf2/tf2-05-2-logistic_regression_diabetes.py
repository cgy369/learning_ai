import tensorflow as tf
import numpy as np

# --- 1. 데이터 불러오기 및 준비 (Loading & Preparing Data) ---
# NumPy를 사용하여 당뇨병 데이터셋(CSV)을 불러옵니다.
# 이 데이터셋은 여러 환자의 의료 기록(8개의 특성)과 당뇨병 발병 여부(1 또는 0)를 포함합니다.
# (참고: 'data-03-diabetes.csv' 파일이 이 스크립트와 동일한 폴더에 존재해야 합니다.)
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

# 데이터를 입력(x_data)과 정답(y_data)으로 분리합니다.
x_data = xy[:, 0:-1]  # 8개의 특성
y_data = xy[:, [-1]]  # 1개의 결과 (0 또는 1)

# 데이터의 형태를 출력하여 확인합니다. (샘플 수, 특성 수)
print(x_data.shape, y_data.shape)


# --- 2. 모델 구성 및 컴파일 (Model Building & Compilation) ---
model = tf.keras.Sequential()

# 입력 특성의 개수(x_data.shape[1]는 8)를 input_dim으로 동적으로 설정합니다.
# 이렇게 하면 데이터의 특성 개수가 변경되어도 코드를 수정할 필요가 없습니다.
# 활성화 함수로 'sigmoid'를 사용하여 이진 분류 모델을 만듭니다.
model.add(tf.keras.layers.Dense(units=1, input_dim=x_data.shape[1], activation='sigmoid'))

# 모델을 컴파일합니다. 손실 함수, 옵티마이저, 평가지표는 이전 로지스틱 회귀 예제와 동일합니다.
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

# 모델 구조를 요약하여 보여줍니다.
model.summary()


# --- 3. 모델 훈련 (Model Training) ---
# 모델을 500 에포크 동안 훈련시킵니다.
history = model.fit(x_data, y_data, epochs=500, verbose=0)


# --- 4. 모델 평가 (Model Evaluation) ---
# 훈련이 끝난 후, 모델의 성능을 평가합니다.

# 방법 1: 훈련 기록(history)에서 마지막 에포크의 정확도 확인하기
# 이는 '훈련 데이터셋'에 대한 정확도입니다.
print("Last epoch accuracy: {0}".format(history.history['accuracy'][-1]))

# 방법 2: model.evaluate() 사용하기
# 주어진 데이터에 대한 모델의 손실(loss)과 평가지표(metrics)를 계산하여 반환합니다.
# 여기서도 '훈련 데이터셋'을 사용하여 평가하고 있습니다.
# (참고: 모델의 진짜 성능(일반화 성능)을 평가하려면, 훈련에 사용하지 않은 별도의 '테스트 데이터셋'으로 평가해야 합니다.)
evaluate_result = model.evaluate(x_data, y_data)
print("Evaluation on training data -> loss: {0}, accuracy: {1}".format(evaluate_result[0], evaluate_result[1]))


# --- 5. 예측 (Prediction) ---
# 훈련된 모델을 사용하여 새로운 데이터 포인트에 대한 예측을 수행합니다.
# 아래 입력 데이터는 각 특성이 정규화(normalized)된 것으로 보입니다.
# 모델을 훈련할 때 사용한 데이터와 동일한 형태로 전처리된 데이터를 예측에 사용해야 합니다.
y_predict = model.predict([[0.176471, 0.155779, 0, 0, 0, 0.052161, -0.952178, -0.733333]])
print("Prediction for a single data point: {0}".format(y_predict))
print("Predicted class (0 or 1): {0}".format((y_predict > 0.5).astype(int)))
