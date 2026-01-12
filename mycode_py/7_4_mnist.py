import os

# TensorFlow 로깅 메시지 설정
# oneDNN 최적화를 비활성화 (특정 환경에서 호환성 문제나 디버깅을 위해 사용)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# TensorFlow C++ 로깅 레벨을 '1'로 설정하여 INFO 메시지를 억제하고 WARNING, ERROR만 표시
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 1. MNIST 데이터셋 로드
# tf.keras.datasets.mnist.load_data() 함수는 MNIST 손글씨 이미지 데이터셋을 로드합니다.
# 이 함수는 데이터를 자동으로 훈련 세트와 테스트 세트로 분할하여 반환합니다.
# x_train, y_train: 훈련용 이미지와 레이블 (훈련 데이터)
# x_test, y_test: 테스트용 이미지와 레이블 (평가/검증 데이터)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(
    f"데이터 로드 완료: {len(x_train)}개의 훈련 이미지, {len(x_test)}개의 테스트 이미지."
)

# (추가) 1-1. 데이터 샘플 시각화
print("\n--- 로드된 데이터 샘플 시각화 ---")
# 처음 25개 이미지를 5x5 그리드로 표시
# 새 창을 생성하고, 창의 크기를 가로 10인치, 세로 10인치로 설정
plt.figure(figsize=(10, 10))
for i in range(25):
    # 5x5 그리드에서 i+1 번째 위치에 서브플롯을 생성
    plt.subplot(5, 5, i + 1)
    # x, y 축의 눈금을 제거하여 이미지를 깔끔하게 표시
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # cmap=plt.cm.binary : 이미지를 흑백으로 표시
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # 각 이미지 하단에 실제 레이블(정답)을 표시
    plt.xlabel(f"Label: {y_train[i]}")
plt.suptitle("MNIST Data Samples")
# 시각화 창을 화면에 표시
plt.show()


# 2. 데이터 전처리
# 이미지 픽셀 값 정규화:
# MNIST 이미지의 픽셀 값은 0부터 255까지의 범위를 가집니다.
# 신경망이 더 잘 학습하도록 이 값을 0부터 1 사이로 정규화합니다.
# 이는 각 픽셀 값을 255.0으로 나누어 수행됩니다.
x_train, x_test = x_train / 255.0, x_test / 255.0


# 3. 신경망 모델 구성
# tf.keras.models.Sequential은 레이어를 순서대로 쌓아 올리는 간단한 모델을 만듭니다.
model = tf.keras.models.Sequential(
    [
        # Flatten 레이어: 28x28 픽셀의 2차원 이미지를 784 픽셀의 1차원 벡터로 평탄화합니다.
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Dense (완전 연결) 레이어: 128개의 뉴런을 가진 은닉 레이어입니다.
        tf.keras.layers.Dense(128, activation="relu"),
        # Dropout 레이어: 과적합(overfitting)을 방지하기 위한 정규화 기법입니다.
        tf.keras.layers.Dropout(0.2),
        # Dense (완전 연결) 출력 레이어: 10개의 뉴런(0-9까지의 숫자 클래스)을 가집니다.
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# 4. 모델 컴파일
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 구조 요약 출력
print("\n--- 모델 요약 ---")
model.summary()

# 5. 모델 학습
print("\n--- 학습 시작 ---")
history = model.fit(
    x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2
)
print("--- 학습 완료 ---")


# (추가) 5-1. 학습 과정 시각화
print("\n--- 학습 과정 시각화 ---")
# model.fit()의 결과인 history 객체에는 학습 과정 동안의 성능 지표들이 저장되어 있습니다.
acc = history.history["accuracy"]  # 훈련 세트의 에폭별 정확도
val_acc = history.history["val_accuracy"]  # 검증 세트의 에폭별 정확도
loss = history.history["loss"]  # 훈련 세트의 에폭별 손실
val_loss = history.history["val_loss"]  # 검증 세트의 에폭별 손실
epochs_range = range(len(acc))  # 에폭 범위

# 새 창을 생성하고, 가로 12인치, 세로 5인치로 설정
plt.figure(figsize=(12, 5))

# 정확도 그래프
# 1행 2열의 그리드에서 첫 번째 위치에 서브플롯을 생성
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

# 손실 그래프
# 1행 2열의 그리드에서 두 번째 위치에 서브플롯을 생성
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.suptitle("Training & Validation History")
# 시각화 창을 화면에 표시
plt.show()


# 6. 모델 평가
print("\n--- 테스트 데이터로 모델 평가 ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\n테스트 정확도: {accuracy:.4f}")

# 7. 몇 가지 샘플에 대한 예측 결과 및 실제 값 표시
print("\n--- 샘플 예측 결과 ---")
# 테스트 세트의 첫 5개 이미지에 대한 예측을 수행합니다.
predictions = model.predict(x_test)
# np.argmax를 사용하여 각 예측에서 가장 높은 확률을 가진 클래스(숫자)를 추출합니다.
predicted_classes = np.argmax(predictions, axis=1)

# 첫 5개 이미지에 대해 예측된 숫자와 실제 숫자를 비교하여 출력합니다.
for i in range(len(x_test)):
    if predicted_classes[i] != y_test[i]:
        print(f"이미지 #{i+1} -> 예측: {predicted_classes[i]}, 실제: {y_test[i]}")
