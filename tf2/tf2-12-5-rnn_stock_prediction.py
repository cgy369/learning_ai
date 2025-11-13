import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 데이터 정규화 및 전처리 함수 (Data Normalization & Preprocessing) ---

# Min-Max 스케일러 함수. 데이터를 0과 1 사이의 값으로 변환합니다.
# 테스트 데이터를 정규화할 때, 훈련 데이터의 min/max 값을 사용해야 하므로 함수를 약간 수정합니다.
def min_max_scaler(data, train_min, train_max):
    numerator = data - train_min
    denominator = train_max - train_min
    return numerator / (denominator + 1e-7)

# 시계열 데이터를 훈련용 데이터셋으로 만드는 함수
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        # seq_length 만큼의 데이터를 입력(X)으로 사용
        x = time_series[i:i + seq_length, :]
        # 바로 다음 날의 종가(Close price)를 정답(Y)으로 사용
        y = time_series[i + seq_length, [-1]]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


# --- 2. 데이터 로드 및 준비 (Data Loading & Preparation) ---
# 하이퍼파라미터
seq_length = 7      # 입력 시퀀스 길이 (7일치 데이터)
data_dim = 5        # 입력 데이터의 차원 (Open, High, Low, Volume, Close)
output_dim = 1      # 출력 데이터의 차원 (다음 날의 종가)
learning_rate = 0.01
iterations = 500

# 주가 데이터 로드 (Open, High, Low, Volume, Close)
# (참고: 'data-02-stock_daily.csv' 파일이 이 스크립트와 동일한 폴더에 존재해야 합니다.)
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # 시간 순서가 되도록 데이터를 뒤집음 (오래된 데이터 -> 최신 데이터)

# 훈련/테스트 데이터 분리 (70% 훈련, 30% 테스트)
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]

# *** 데이터 정규화 (Normalization) ***
# 테스트 데이터를 정규화할 때는 반드시 '훈련 데이터'의 min/max 값을 사용해야 합니다.
# 이는 모델이 훈련 중에 보지 못한 정보(테스트 데이터의 min/max)가 평가 과정에 유입되는
# '데이터 누수(Data Leakage)'를 방지하기 위함입니다.
train_min = np.min(train_set, 0)
train_max = np.max(train_set, 0)

train_set = min_max_scaler(train_set, train_min, train_max)
test_set = min_max_scaler(test_set, train_min, train_max)

# 훈련 및 테스트 데이터셋 생성
trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)


# --- 3. LSTM 모델 구성 (LSTM Model Building) ---
# 시계열 예측을 위한 LSTM 모델을 구성합니다.
model = tf.keras.Sequential()

# LSTM 레이어:
# - units: LSTM의 은닉 상태 크기. 더 복잡한 패턴을 위해 늘릴 수 있습니다.
# - input_shape: 입력 데이터의 형태 (시퀀스 길이, 입력 차원)
#   return_sequences=False (기본값): 마지막 타임스텝의 출력만 반환합니다.
model.add(tf.keras.layers.LSTM(units=16, input_shape=(seq_length, data_dim)))

# 출력층 (Fully Connected Layer):
# - units=1: 예측할 값은 다음 날의 종가 하나이므로 1로 설정합니다.
# - activation='linear' (기본값): 회귀 문제이므로 활성화 함수를 사용하지 않거나 'linear'를 사용합니다.
model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))

model.summary()


# --- 4. 모델 컴파일 및 훈련 (Model Compilation & Training) ---
# 손실 함수로 'mean_squared_error'(mse)를 사용합니다. 이는 회귀 문제의 표준 손실 함수입니다.
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
model.fit(trainX, trainY, epochs=iterations, verbose=0)


# --- 5. 모델 평가 및 결과 시각화 (Evaluation & Visualization) ---
# 훈련된 모델을 사용하여 테스트 데이터에 대한 예측을 수행합니다.
test_predict = model.predict(testX)

# 예측 결과를 실제 값과 함께 그래프로 그려서 성능을 시각적으로 확인합니다.
plt.plot(testY, label='True Price') # 실제 주가
plt.plot(test_predict, label='Predicted Price') # 모델이 예측한 주가
plt.xlabel("Time Period")
plt.ylabel("Stock Price (Normalized)")
plt.legend()
plt.show()
