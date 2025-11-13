import numpy as np
import random
import tensorflow as tf

# --- 1. 하이퍼파라미터 설정 (Hyperparameter Settings) ---
random.seed(777)  # 재현성(reproducibility)을 위해 랜덤 시드 설정
learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10


# --- 2. MNIST 데이터셋 로드 및 전처리 (Load & Preprocess MNIST Dataset) ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 정규화 (Normalization): 픽셀 값을 0-1 범위로 스케일링
x_train, x_test = x_train / 255.0, x_test / 255.0

# 이미지 데이터 형태 변경 (Reshaping/Flattening): 28x28 이미지를 784차원 벡터로 펼침
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# 레이블(정답)을 원-핫 인코딩(One-Hot Encoding)으로 변환
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)


# --- 3. 신경망 모델 구성 (Neural Network Model Building) ---
# 가중치 초기화(Weight Initialization)는 신경망 학습의 안정성과 속도에 매우 중요합니다.
# 잘못된 초기화는 기울기 소실(Vanishing Gradient)이나 기울기 폭주(Exploding Gradient)를 유발할 수 있습니다.
model = tf.keras.Sequential()

# 첫 번째 은닉층:
# kernel_initializer='glorot_normal': 가중치 초기화 방법으로 'Glorot Normal' (Xavier Normal) 초기화를 사용합니다.
#   - 이 초기화 방법은 이전 레이어의 뉴런 수와 현재 레이어의 뉴런 수를 고려하여 가중치를 무작위로 초기화합니다.
#   - 활성화 함수가 sigmoid나 tanh일 때 특히 효과적이며, 활성화 값과 기울기의 분산을 일정하게 유지하여
#     학습 초기에 기울기 소실/폭주 문제를 완화하는 데 도움을 줍니다.
model.add(tf.keras.layers.Dense(input_dim=784, units=256, kernel_initializer='glorot_normal', activation='relu'))

# 두 번째 은닉층:
model.add(tf.keras.layers.Dense(units=256, kernel_initializer='glorot_normal', activation='relu'))

# 출력층:
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
