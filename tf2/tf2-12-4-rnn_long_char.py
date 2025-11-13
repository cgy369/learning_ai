import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
# 훈련에 사용할 긴 문장입니다.
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 문장에 포함된 모든 고유한 글자들로 '사전(vocabulary)'을 만듭니다.
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

# 하이퍼파라미터 설정
data_dim = len(char_set)      # 사전의 크기 (입력 차원)
hidden_size = len(char_set)   # RNN의 은닉 상태 크기
num_classes = len(char_set)   # 최종 출력 클래스 수
sequence_length = 10          # 입력 및 출력 시퀀스의 길이
learning_rate = 0.1


# --- 2. 슬라이딩 윈도우를 이용한 훈련 데이터셋 생성 ---
# 긴 문장을 일정 길이(sequence_length)의 시퀀스들로 잘라 여러 개의 훈련 샘플을 만듭니다.
dataX = [] # 입력 시퀀스
dataY = [] # 목표 시퀀스

# 문장 전체를 순회하면서 훈련 샘플을 생성합니다.
for i in range(0, len(sentence) - sequence_length):
    # 입력 시퀀스: i번째 글자부터 10개
    x_str = sentence[i : i + sequence_length]
    # 목표 시퀀스: i+1번째 글자부터 10개 (입력 시퀀스에서 한 글자씩 밀림)
    y_str = sentence[i + 1 : i + sequence_length + 1]
    
    # 글자 시퀀스를 인덱스 시퀀스로 변환
    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

# 생성된 샘플의 개수를 배치 크기로 사용합니다.
batch_size = len(dataX)

# 생성된 데이터를 원-핫 인코딩으로 변환합니다.
X_one_hot = tf.keras.utils.to_categorical(dataX, num_classes)
Y_one_hot = tf.keras.utils.to_categorical(dataY, num_classes)


# --- 3. 스택 LSTM (Stacked LSTM) 모델 구성 ---
# 여러 개의 LSTM 레이어를 쌓아 더 깊은 RNN 모델을 만듭니다.
# 이를 통해 더 복잡하고 장기적인 패턴을 학습할 수 있습니다.
model = tf.keras.Sequential()

# 첫 번째 LSTM 레이어:
# - return_sequences=True: 다음 LSTM 레이어로 전체 시퀀스를 전달하기 위해 필수입니다.
model.add(tf.keras.layers.LSTM(units=hidden_size, input_shape=(sequence_length, data_dim), return_sequences=True))

# 두 번째 LSTM 레이어:
# - 이전 레이어의 출력을 입력으로 받습니다.
model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))

# TimeDistributed Dense 레이어:
# - 각 타임스텝마다 다음 글자를 예측하기 위해 소프트맥스 활성화를 적용합니다.
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))

model.summary()


# --- 4. 모델 컴파일 및 훈련 (Model Compilation & Training) ---
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

model.fit(X_one_hot, Y_one_hot, epochs=100, verbose=0)


# --- 5. 텍스트 생성 (Text Generation) ---
# 훈련된 모델을 사용하여 새로운 텍스트를 생성해봅니다.
# 여기서는 전체 훈련 데이터를 다시 입력으로 넣어, 모델이 학습한 내용을 바탕으로
# 어떻게 문장을 '재구성'하는지 확인합니다.
results = model.predict(X_one_hot)

print("\n--- Text Generation Result ---")
for j, result in enumerate(results):
    # 각 타임스텝에서 가장 확률이 높은 글자의 인덱스를 찾습니다.
    index = np.argmax(result, axis=1)
    
    # 첫 번째 예측 결과(j=0)는 전체 시퀀스를 출력하여 문장의 시작 부분을 만듭니다.
    if j == 0:
        print(''.join([char_set[t] for t in index]), end='')
    # 두 번째 예측 결과부터는 마지막 글자만 이어서 출력합니다.
    else:
        print(char_set[index[-1]], end='')

print("\n--------------------------")
