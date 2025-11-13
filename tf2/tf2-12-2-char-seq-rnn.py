import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 훈련에 사용할 샘플 텍스트입니다.
sample = " if you want you"

# 텍스트에 포함된 모든 고유한 글자들을 찾아 '사전(vocabulary)'을 만듭니다.
# set(sample): 중복된 글자를 제거하여 고유한 글자 집합을 만듭니다.
# list(...): 집합을 리스트로 변환합니다.
idx2char = list(set(sample))  # 인덱스 -> 글자 매핑
char2idx = {c: i for i, c in enumerate(idx2char)}  # 글자 -> 인덱스 매핑

# 하이퍼파라미터 설정
dic_size = len(char2idx)      # 사전의 크기 (고유한 글자의 수)
hidden_size = dic_size        # RNN의 은닉 상태 크기
num_classes = dic_size        # 최종 출력 클래스 수 (글자 수와 동일)
sequence_length = len(sample) - 1  # 입력 시퀀스의 길이
learning_rate = 0.1

# --- 2. 훈련 데이터셋 생성 (Creating the Training Dataset) ---
# "if you want yo" -> "f you want you"
# 각 글자를 예측하는 방식으로 훈련 데이터셋을 만듭니다.

# 샘플 텍스트를 인덱스 시퀀스로 변환합니다.
sample_idx = [char2idx[c] for c in sample]

# x_data: 입력 시퀀스 (처음부터 마지막 글자 직전까지)
x_data = [sample_idx[:-1]]
# y_data: 목표 시퀀스 (두 번째 글자부터 마지막 글자까지)
y_data = [sample_idx[1:]]

# 입력과 목표 데이터를 원-핫 인코딩으로 변환합니다.
# RNN의 입력으로 사용하기 위해 (샘플 수, 시퀀스 길이, 입력 차원) 형태로 만듭니다.
x_one_hot = tf.keras.utils.to_categorical(x_data, num_classes=dic_size)
y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=dic_size)

print("x_one_hot shape:", x_one_hot.shape)
print("y_one_hot shape:", y_one_hot.shape)


# --- 3. RNN 모델 구성 (RNN Model Building) ---
# 이전 예제와 동일한 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 사용합니다.
model = tf.keras.Sequential()

# LSTM 레이어
# - units=hidden_size: LSTM의 은닉 상태 크기.
# - input_shape: 입력 데이터의 형태 (시퀀스 길이, 입력 차원).
# - return_sequences=True: 각 타임스텝마다 출력을 생성합니다.
model.add(tf.keras.layers.LSTM(units=hidden_size, input_shape=(sequence_length, dic_size), return_sequences=True))

# TimeDistributed Dense 레이어
# - 각 타임스텝의 LSTM 출력에 대해 독립적으로 Dense 레이어를 적용하여,
#   다음 글자에 대한 확률 분포를 계산합니다.
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))

model.summary()


# --- 4. 모델 컴파일 및 훈련 (Model Compilation & Training) ---
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

model.fit(x_one_hot, y_one_hot, epochs=50, verbose=0)


# --- 5. 예측 결과 확인 (Prediction) ---
# 훈련된 모델을 사용하여 전체 입력 시퀀스에 대한 예측을 수행합니다.
predictions = model.predict(x_one_hot)

# 예측된 확률 분포에서 가장 높은 확률을 가진 글자를 선택하여 결과 문자열을 만듭니다.
predicted_indices = np.argmax(predictions, axis=2) # axis=2는 각 타임스텝의 클래스 차원을 의미
result_str = [idx2char[c] for c in np.squeeze(predicted_indices)] # np.squeeze로 불필요한 차원 제거

print("\n--- Prediction Result ---")
print("Input string:    ", sample[:-1])
print("Predicted string:", ''.join(result_str))
print("True string:     ", sample[1:])
