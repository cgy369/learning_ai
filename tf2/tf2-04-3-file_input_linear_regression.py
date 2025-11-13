import tensorflow as tf
import numpy as np

# --- 1. 데이터 불러오기 (Loading Data) ---
# NumPy의 loadtxt 함수를 사용하여 외부 CSV 파일에서 데이터를 읽어옵니다.
# 이 방식은 대량의 데이터를 효율적으로 관리하고 코드와 데이터를 분리하는 데 유용합니다.
# delimiter=',' : 파일의 각 값이 쉼표(,)로 구분되어 있음을 의미합니다.
# dtype=np.float32 : 읽어온 데이터를 32비트 실수형으로 지정합니다.
# (참고: 'data-01-test-score.csv' 파일이 이 스크립트와 동일한 폴더에 존재해야 합니다.)
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)


# --- 2. 데이터 분리 (Slicing Data) ---
# 불러온 데이터(xy)를 입력 데이터(x_data)와 정답 데이터(y_data)로 분리합니다.
# NumPy의 슬라이싱(slicing) 기능을 사용합니다.

# x_data: 모든 행(샘플)에 대해, 처음부터 마지막 열을 제외한 모든 열을 선택합니다.
# xy[:, 0:-1] -> ':'는 모든 행을 의미, '0:-1'은 0번째 열부터 마지막(-1) 열 직전까지를 의미.
# 즉, 3개의 입력 특성(feature)을 선택합니다.
x_data = xy[:, 0:-1]

# y_data: 모든 행(샘플)에 대해, 마지막 열만 선택합니다.
# xy[:, [-1]] -> ':'는 모든 행을 의미, '[-1]'은 마지막 열을 의미.
# 즉, 1개의 정답(label)을 선택합니다.
y_data = xy[:, [-1]]

# 불러온 데이터의 형태(shape)와 내용을 확인합니다.
# shape는 (샘플 수, 특성/레이블 수)를 나타냅니다.
print("x_data shape: ", x_data.shape)
print("y_data shape: ", y_data.shape)


# --- 3. 모델 구성 및 컴파일 (Model Building & Compilation) ---
model = tf.keras.Sequential()

# Dense 레이어에 activation 함수를 직접 인자로 전달하는 것이 더 깔끔한 방법입니다.
# input_dim=3: 입력 특성이 3개임을 명시합니다.
# activation='linear': 회귀 문제이므로 선형 활성화 함수를 사용합니다.
model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))

# 모델의 구조를 요약하여 보여줍니다.
model.summary()

# 모델을 컴파일합니다. 학습률은 이전 예제와 동일하게 1e-5로 설정합니다.
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))


# --- 4. 모델 훈련 (Model Training) ---
# 파일에서 읽어온 데이터로 모델을 2000 에포크 동안 훈련시킵니다.
history = model.fit(x_data, y_data, epochs=2000, verbose=0)


# --- 5. 예측 (Prediction) ---
# 훈련된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.
print("My score prediction: ", model.predict([[100, 70, 101]]))
print("Other scores prediction: ", model.predict([[60, 70, 110], [90, 100, 80]]))
