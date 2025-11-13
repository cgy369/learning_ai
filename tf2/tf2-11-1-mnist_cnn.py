# Lab 11 MNIST and Convolutional Neural Network
import numpy as np
import tensorflow as tf
import random

# --- 1. 하이퍼파라미터 설정 (Hyperparameter Settings) ---
learning_rate = 0.001
training_epochs = 12
batch_size = 128


# --- 2. MNIST 데이터셋 로드 및 전처리 (Load & Preprocess MNIST Dataset) ---
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화 (Normalization): 픽셀 값을 0-1 범위로 스케일링
x_train, x_test = x_train / 255.0, x_test / 255.0

# 이미지 데이터 형태 변경 (Reshaping for CNN):
# CNN은 입력 이미지에 채널(channel) 차원이 필요합니다.
# 흑백 이미지(MNIST)는 1개의 채널을 가집니다.
# (샘플 수, 높이, 너비, 채널) 형태로 변경합니다.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 레이블(정답)을 원-핫 인코딩(One-Hot Encoding)으로 변환
nb_classes = 10 # 0-9까지 10개의 클래스
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)


# --- 3. 합성곱 신경망 (Convolutional Neural Network, CNN) 모델 구성 ---
# CNN은 이미지와 같은 격자형 데이터에서 특징을 추출하고 분류하는 데 매우 효과적입니다.
# 주요 구성 요소: 합성곱(Conv2D) 레이어, 풀링(MaxPooling2D) 레이어, 완전 연결(Dense) 레이어

model = tf.keras.Sequential()

# 첫 번째 합성곱 블록 (Convolutional Block 1)
# Conv2D: 이미지에서 특징(feature)을 추출하는 레이어.
#   - filters=16: 16개의 특징 맵(feature map)을 생성합니다. 각 필터는 다른 특징을 학습합니다.
#   - kernel_size=(3, 3): 3x3 크기의 필터(커널)를 사용합니다.
#   - input_shape=(28, 28, 1): 입력 이미지의 형태 (높이, 너비, 채널). 첫 레이어에서만 지정합니다.
#   - activation='relu': ReLU 활성화 함수를 사용하여 비선형성을 추가합니다.
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
# MaxPooling2D: 특징 맵의 크기를 줄이고(다운샘플링), 중요한 특징을 강조하며, 위치 변화에 덜 민감하게 만듭니다.
#   - pool_size=(2, 2): 2x2 영역에서 가장 큰 값(max)을 선택하여 크기를 절반으로 줄입니다.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 두 번째 합성곱 블록 (Convolutional Block 2)
# 첫 번째 블록과 유사하지만, 더 많은 필터(32개)를 사용하여 더 복잡한 특징을 학습합니다.
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 특징 맵을 1차원 벡터로 펼치기 (Flatten)
# 합성곱 및 풀링 레이어에서 생성된 2차원 특징 맵을 완전 연결(Dense) 레이어에 입력하기 위해 1차원으로 변환합니다.
model.add(tf.keras.layers.Flatten())

# 완전 연결 출력층 (Fully Connected Output Layer)
# - units=10: 10개의 클래스(0-9 숫자)에 대한 출력을 가집니다.
# - activation='softmax': 다중 클래스 분류를 위한 소프트맥스 활성화 함수를 사용하여 확률 분포를 출력합니다.
# - kernel_initializer='glorot_normal': 가중치 초기화 방법으로 Glorot Normal 초기화를 사용합니다.
model.add(tf.keras.layers.Dense(units=nb_classes, kernel_initializer='glorot_normal', activation='softmax'))

# 모델 구조 요약 출력
model.summary()


# --- 4. 모델 컴파일 (Model Compilation) ---
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


# --- 5. 모델 훈련 (Model Training) ---
print("\n--- Start Training ---")
model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1)
print("--- Training Finished ---")


# --- 6. 모델 평가 (Model Evaluation) ---
print("\n--- Evaluation on Test Data ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# --- 7. 무작위 샘플 예측 (Random Sample Prediction) ---
print("\n--- Random 10 Sample Predictions ---")
y_predicted = model.predict(x_test)

for _ in range(10):
    random_index = random.randint(0, x_test.shape[0] - 1)
    print(f"Index: {random_index}, "
          f"Actual Y: {np.argmax(y_test[random_index])}, "
          f"Predicted Y: {np.argmax(y_predicted[random_index])}")