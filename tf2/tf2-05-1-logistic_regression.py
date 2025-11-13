import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 이진 분류(Binary Classification)를 위한 데이터를 정의합니다.
# 이진 분류는 결과를 두 가지 중 하나로 예측하는 문제입니다 (e.g., 0 또는 1, 합격 또는 불합격).

# x_data: 훈련 데이터. 각 샘플이 2개의 특성(feature)을 가집니다. (6개의 샘플, 2개의 특성)
# e.g., [공부한 시간, 출석 일수]
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
# y_data: 정답 데이터. 0 또는 1의 값을 가집니다. (6개의 샘플, 1개의 결과)
# e.g., [0: 불합격, 1: 합격]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]


# --- 2. 모델 구성 (Model Building) ---
# 로지스틱 회귀(Logistic Regression) 모델을 구성합니다.
model = tf.keras.Sequential()

# Dense 레이어에 활성화 함수로 'sigmoid'를 직접 지정합니다.
# input_dim=2: 입력 특성이 2개임을 명시합니다.
# units=1: 출력 결과가 하나(확률값)이므로 1로 설정합니다.
# activation='sigmoid': 시그모이드 함수. 모델의 최종 출력값을 0과 1 사이의 값으로 변환합니다.
#                      이 값은 특정 클래스(e.g., '1' 또는 '합격')에 속할 확률로 해석할 수 있습니다.
model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))


# --- 3. 모델 컴파일 (Model Compilation) ---
# loss='binary_crossentropy': 이진 분류 문제에 사용되는 표준 손실 함수입니다.
#                          모델이 예측한 확률이 실제 정답(0 또는 1)에서 얼마나 벗어났는지를 측정합니다.
#                          'mse'보다 분류 문제에서 훨씬 더 효과적입니다.
# optimizer=SGD(...): 경사 하강법 옵티마이저를 사용합니다.
# metrics=['accuracy']: 훈련 과정 및 테스트 과정에서 모델의 '정확도'를 측정하고 기록하도록 설정합니다.
#                      정확도는 (정확히 예측한 샘플 수) / (전체 샘플 수) 입니다.
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])


# --- 4. 모델 요약 및 훈련 (Model Summary & Training) ---
model.summary()

# 모델을 5000 에포크 동안 훈련시킵니다.
history = model.fit(x_data, y_data, epochs=5000, verbose=0)


# --- 5. 결과 확인 (Checking Results) ---
# 훈련이 끝난 후, 마지막 에포크의 훈련 정확도를 출력합니다.
# history.history['accuracy']는 각 에포크별 정확도를 담고 있는 리스트입니다.
# [-1]은 리스트의 마지막 값을 의미합니다.
print("Final Accuracy: ", history.history['accuracy'][-1])


# --- 6. 예측 (Prediction) ---
# 훈련된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.
# 모델의 출력은 0과 1 사이의 확률값이 됩니다.
new_data_predictions = model.predict([[1, 5], [7, 2]])
print("Predictions for [[1, 5], [7, 2]]: \n", new_data_predictions)

# 예측된 확률값을 0 또는 1의 클래스로 변환합니다.
# 일반적으로 0.5를 기준으로, 크면 1(e.g., 합격), 작으면 0(e.g., 불합격)으로 판단합니다.
predicted_classes = (new_data_predictions > 0.5).astype(int)
print("Predicted classes: \n", predicted_classes)
