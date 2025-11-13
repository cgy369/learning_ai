import numpy as np
import random
import tensorflow as tf

# --- 1. 하이퍼파라미터 설정 (Hyperparameter Settings) ---
random.seed(777)  # 재현성(reproducibility)을 위해 랜덤 시드 설정
learning_rate = 0.001
batch_size = 100
training_epochs = 15
b_classes = 10
drop_rate = 0.3  # 드롭아웃 비율: 훈련 중 무작위로 비활성화할 뉴런의 비율


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


# --- 3. 드롭아웃(Dropout)을 적용한 깊은 신경망 모델 구성 ---
# 과적합(Overfitting)을 방지하기 위한 정규화(Regularization) 기법인 드롭아웃을 적용합니다.
# 과적합: 모델이 훈련 데이터에 너무 잘 맞춰져서, 새로운(보지 못한) 데이터에 대한 성능이 떨어지는 현상.
model = tf.keras.Sequential()

# 각 은닉층 뒤에 Dropout 레이어를 추가합니다.
# Dropout 레이어는 훈련 중에 무작위로 일부 뉴런의 출력을 0으로 만듭니다.
# 이는 뉴런들이 서로에게 너무 의존하는 것을 방지하고, 모델이 더 강건한 특징을 학습하도록 돕습니다.

# 첫 번째 은닉층
model.add(tf.keras.layers.Dense(input_dim=784, units=512, kernel_initializer='glorot_normal', activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate)) # 드롭아웃 적용

# 두 번째 은닉층
model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate)) # 드롭아웃 적용

# 세 번째 은닉층
model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate)) # 드롭아웃 적용

# 네 번째 은닉층
model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate)) # 드롭아웃 적용

# 출력층
# 출력층에는 일반적으로 드롭아웃을 적용하지 않습니다.
model.add(tf.keras.layers.Dense(units=nb_classes, kernel_initializer='glorot_normal', activation='softmax'))

# 모델 구조 요약 출력
model.summary()


# --- 4. 모델 컴파일 (Model Compilation) ---
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


# --- 5. 모델 훈련 (Model Training) ---
# 드롭아웃은 훈련 시에만 활성화되며, 예측 시에는 비활성화됩니다.
# Keras는 이를 자동으로 처리합니다.
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
