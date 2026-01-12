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


# --- 3. 다층 신경망 (Multi-Layer Neural Network) 모델 구성 ---
# 단일 레이어 소프트맥스 분류기보다 더 복잡한 패턴을 학습할 수 있는 다층 신경망을 구성합니다.
# 이 모델은 두 개의 은닉층(Hidden Layer)을 가집니다.
model = tf.keras.Sequential()

# 첫 번째 은닉층:
# - units=256: 256개의 뉴런을 가집니다.
# - input_dim=784: 입력은 784개의 픽셀 값입니다.
# - activation='relu': ReLU(Rectified Linear Unit) 활성화 함수를 사용합니다.
#   ReLU는 sigmoid보다 기울기 소실 문제에 덜 민감하며, 딥러닝 모델에서 은닉층에 가장 널리 사용됩니다.
model.add(tf.keras.layers.Dense(input_dim=784, units=256, activation="relu"))

# 두 번째 은닉층:
# - units=256: 256개의 뉴런을 가집니다.
# - activation='relu': ReLU 활성화 함수를 사용합니다.
#   이전 레이어의 출력(256개)이 이 레이어의 입력이 됩니다.
model.add(tf.keras.layers.Dense(units=256, activation="relu"))

# 출력층:
# - units=nb_classes (10): 10개의 클래스(0-9 숫자)에 대한 출력을 가집니다.
# - activation='softmax': 다중 클래스 분류를 위한 소프트맥스 활성화 함수를 사용하여 확률 분포를 출력합니다.
model.add(tf.keras.layers.Dense(units=nb_classes, activation="softmax"))

# 모델 구조 요약 출력
model.summary()


# --- 4. 모델 컴파일 (Model Compilation) ---
# 손실 함수, 옵티마이저, 평가지표 설정
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)


# --- 5. 모델 훈련 (Model Training) ---
# 훈련 데이터를 사용하여 모델을 훈련시킵니다.
print("\n--- Start Training ---")
model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1)
print("--- Training Finished ---")


# --- 6. 모델 평가 (Model Evaluation) ---
# 훈련된 모델을 테스트 데이터로 평가하여 일반화 성능을 측정합니다.
print("\n--- Evaluation on Test Data ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# --- 7. 무작위 샘플 예측 (Random Sample Prediction) ---
# 테스트 데이터에서 무작위로 10개의 샘플을 선택하여 모델의 예측 결과를 확인합니다.
print("\n--- Random 10 Sample Predictions ---")
y_predicted = model.predict(x_test)  # 전체 테스트 데이터에 대한 예측을 한 번에 수행

for _ in range(10):  # 10번 반복
    random_index = random.randint(
        0, x_test.shape[0] - 1
    )  # 테스트 데이터 내에서 무작위 인덱스 선택

    # 실제 레이블과 예측된 레이블을 출력합니다.
    # np.argmax는 원-핫 인코딩된 벡터에서 1의 위치(즉, 클래스 인덱스)를 찾아줍니다.
    print(
        f"Index: {random_index}, "
        f"Actual Y: {np.argmax(y_test[random_index])}, "
        f"Predicted Y: {np.argmax(y_predicted[random_index])}"
    )
