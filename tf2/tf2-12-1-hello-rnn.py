import numpy as np
import tensorflow as tf

# --- 1. 데이터 준비 (Data Preparation) ---
# 'hello'라는 단어를 학습하여 다음 글자를 예측하는 간단한 RNN 예제입니다.
# 목표: "hihell" -> "ihello" (한 글자씩 밀린 시퀀스 예측)

# 글자-인덱스 매핑: 각 글자를 숫자로 표현하기 위한 딕셔너리
idx2char = ['h', 'i', 'e', 'l', 'o']
char2idx = {'h': 0, 'i': 1, 'e': 2, 'l': 3, 'o': 4}

# 입력 시퀀스 (x_data): "hihell"
# 각 글자를 해당 인덱스로 변환합니다.
x_data = [char2idx['h'], char2idx['i'], char2idx['h'], char2idx['e'], char2idx['l'], char2idx['l']] # [0, 1, 0, 2, 3, 3]

# 목표 시퀀스 (y_data): "ihello"
y_data = [char2idx['i'], char2idx['h'], char2idx['e'], char2idx['l'], char2idx['l'], char2idx['o']] # [1, 0, 2, 3, 3, 4]

# 하이퍼파라미터 설정
num_classes = len(idx2char)  # 글자 종류의 수 (5개)
input_dim = num_classes      # 각 글자를 원-핫 인코딩할 때의 차원 (5차원)
sequence_length = len(x_data) # 시퀀스의 길이 (6글자)
learning_rate = 0.1

# 입력 데이터를 원-핫 인코딩으로 변환합니다.
# (샘플 수, 시퀀스 길이, 입력 차원) 형태가 됩니다.
# 여기서는 샘플이 하나이므로 (1, 6, 5)
x_one_hot = tf.keras.utils.to_categorical([x_data], num_classes=num_classes)

# 목표 시퀀스도 원-핫 인코딩으로 변환합니다.
# (샘플 수, 시퀀스 길이, 클래스 수) 형태가 됩니다.
y_one_hot = tf.keras.utils.to_categorical([y_data], num_classes=num_classes)

print("x_one_hot shape:", x_one_hot.shape) # (1, 6, 5)
print("y_one_hot shape:", y_one_hot.shape) # (1, 6, 5)


# --- 2. 순환 신경망 (Recurrent Neural Network, RNN) 모델 구성 ---
# RNN은 시퀀스 데이터를 처리하는 데 특화된 신경망입니다.
# 이전 시점의 정보를 기억하여 현재 시점의 예측에 활용합니다.
model = tf.keras.Sequential()

# LSTM (Long Short-Term Memory) 레이어:
# - RNN의 한 종류로, 장기 의존성(long-term dependencies) 문제를 해결하는 데 효과적입니다.
# - units=num_classes: LSTM 셀의 은닉 상태(hidden state) 크기. 여기서는 출력 클래스 수와 동일하게 설정.
# - input_shape=(sequence_length, input_dim): 입력 시퀀스의 형태.
# - return_sequences=True: 각 시점(time step)마다 출력을 반환하도록 설정합니다.
#   (False이면 마지막 시점의 출력만 반환)
model.add(tf.keras.layers.LSTM(units=num_classes, input_shape=(sequence_length, input_dim), return_sequences=True))

# TimeDistributed Dense 레이어:
# - LSTM 레이어의 각 시점(time step) 출력에 대해 독립적으로 Dense 레이어를 적용합니다.
# - 각 시점마다 다음 글자에 대한 확률 분포를 예측합니다.
# - units=num_classes: 각 시점의 출력은 글자 종류의 수와 동일합니다.
# - activation='softmax': 각 시점의 출력을 확률 분포로 변환합니다.
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))

model.summary()


# --- 3. 모델 컴파일 및 훈련 (Model Compilation & Training) ---
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])

# 모델 훈련: 입력 시퀀스 x_one_hot으로 목표 시퀀스 y_one_hot을 예측하도록 학습합니다.
model.fit(x_one_hot, y_one_hot, epochs=50, verbose=0)


# --- 4. 예측 결과 확인 (Prediction & Interpretation) ---
predictions = model.predict(x_one_hot)

# 예측 결과를 시각적으로 확인합니다.
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}:")
    # 각 시점(time step)의 예측 확률 분포에서 가장 높은 확률을 가진 글자의 인덱스를 찾습니다.
    predicted_indices = np.argmax(prediction, axis=1)
    
    # 인덱스를 다시 글자로 변환하여 예측된 단어를 만듭니다.
    result_str = [idx2char[c] for c in predicted_indices]
    print("  Prediction (indices):", predicted_indices)
    print("  Prediction (string): ", ''.join(result_str))
    
    # 실제 정답 시퀀스도 출력하여 비교합니다.
    true_indices = y_data[i]
    true_str = [idx2char[c] for c in true_indices]
    print("  True (indices):    ", true_indices)
    print("  True (string):     ", ''.join(true_str))
