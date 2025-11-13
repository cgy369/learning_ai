import tensorflow as tf
import numpy as np

# --- 데이터 정규화(Normalization)의 필요성 ---
# 이 스크립트는 데이터 정규화를 하지 않았을 때 경사 하강법(Gradient Descent)에서 발생할 수 있는
# 심각한 문제점을 보여주기 위한 예제입니다.

# --- 1. 데이터 준비 (Data Preparation) ---
# 데이터의 값들을 보면, 어떤 열은 800대인데 어떤 열은 1,000,000대입니다.
# 이렇게 각 특성(feature)의 값의 범위(scale)가 크게 차이 나면,
# 훈련 과정에서 문제가 발생할 수 있습니다.
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# --- 2. 모델 구성 및 컴파일 (Model Building & Compilation) ---
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=4, activation='linear'))
# learning_rate를 1e-5로 매우 작게 설정했음에도 불구하고,
# 데이터의 스케일이 너무 커서 훈련이 제대로 되지 않을 것입니다.
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))
model.summary()


# --- 3. 모델 훈련 및 문제점 확인 (Model Training & Problem) ---
history = model.fit(x_data, y_data, epochs=100, verbose=0)

# 훈련 후 손실(loss) 값을 출력합니다.
# 아마도 손실 값이 'nan' (Not a Number)으로 출력될 것입니다.
print(history.history['loss'])

# 왜 'nan'이 될까요? -> 경사 폭주(Exploding Gradients) 문제
# 1. 입력 데이터의 값이 매우 크면, 손실(loss)을 계산하고 미분하는 과정에서 기울기(gradient) 값이 비정상적으로 커질 수 있습니다.
# 2. 이 거대한 기울기 값에 학습률(learning_rate)이 곱해져 가중치(weight)가 업데이트됩니다.
# 3. 가중치가 너무 큰 폭으로, 비정상적인 방향으로 업데이트되면서 무한대(inf)나 nan 값으로 발산해버립니다.
# 4. 결국 모델의 모든 파라미터가 망가져 더 이상 학습을 진행할 수 없게 됩니다.

# 이 문제에 대한 해결책은 다음 파일에서 다룹니다: 데이터 정규화(Data Normalization)