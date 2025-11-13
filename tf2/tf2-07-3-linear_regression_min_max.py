import tensorflow as tf
import numpy as np

# --- 1. 데이터 정규화 함수 정의 (Data Normalization Function) ---

# Min-Max 스케일러 함수를 정의합니다.
# 이 함수는 모든 데이터 값을 0과 1 사이의 값으로 변환합니다.
# 공식: (data - min) / (max - min)
def min_max_scaler(data):
    # np.min(data, 0) / np.max(data, 0): 각 열(axis=0)의 최소값/최대값을 찾습니다.
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    
    # 분모가 0이 되는 것을 방지하기 위해 아주 작은 값(1e-7)을 더해줍니다.
    # (만약 특정 열의 모든 값이 같다면, max와 min이 같아져 분모가 0이 될 수 있습니다.)
    return numerator / (denominator + 1e-7)


# --- 2. 데이터 준비 및 정규화 (Data Preparation & Normalization) ---
# 이전 파일과 동일한, 스케일이 큰 데이터입니다.
xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

# *** 매우 중요한 단계 ***
# 정의한 min_max_scaler 함수를 사용하여 데이터를 정규화합니다.
# 이 과정을 거치면 모든 데이터가 0과 1 사이의 값으로 변환되어,
# 경사 폭주(exploding gradients) 문제를 방지하고 안정적인 학습이 가능해집니다.
xy = min_max_scaler(xy)
print("Normalized data (first 5 rows):\n", xy[:5])

# 정규화된 데이터를 x와 y로 분리합니다.
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# --- 3. 모델 구성 및 컴파일 (Model Building & Compilation) ---
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=4, activation='linear'))
# 이제 데이터가 정규화되었으므로, 이전보다 더 높은 학습률을 사용할 수도 있습니다.
# 하지만 여기서는 동일한 학습률을 사용하여 정규화의 효과를 명확히 확인합니다.
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))
model.summary()


# --- 4. 모델 훈련 (Model Training) ---
# 정규화된 데이터로 훈련을 진행합니다.
# 이제 손실(loss)이 'nan'이 되지 않고 안정적으로 감소하는 것을 볼 수 있습니다.
history = model.fit(x_data, y_data, epochs=1000, verbose=0)


# --- 5. 평가 및 예측 (Evaluation & Prediction) ---
# 훈련된 모델을 평가하고 예측을 수행합니다.
score = model.evaluate(x_data, y_data)
predictions = model.predict(x_data)

print('Cost (loss): ', score)
# (참고) 예측 결과(predictions)는 정규화된 y값에 대한 예측입니다.
# 실제 값으로 해석하려면, 원래 데이터의 min/max 값을 사용하여 역변환(de-normalization) 과정이 필요합니다.
print('Predictions (on scaled data): \n', predictions[:5])
