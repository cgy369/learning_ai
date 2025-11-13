import tensorflow as tf
import numpy as np

# --- 1. 하이퍼파라미터 설정 (Hyperparameter Settings) ---
# 모델의 학습 과정에 영향을 미치는 설정 값들입니다.
learning_rate = 0.001      # 학습률: 옵티마이저가 가중치를 업데이트하는 보폭
batch_size = 100           # 배치 크기: 한 번의 가중치 업데이트에 사용되는 샘플의 수
training_epochs = 15       # 에포크 수: 전체 훈련 데이터를 몇 번 반복하여 학습할지
nb_classes = 10            # 클래스 개수: MNIST는 0부터 9까지 10개의 숫자 클래스를 가집니다.


# --- 2. MNIST 데이터셋 로드 및 전처리 (Load & Preprocess MNIST Dataset) ---
# tf.keras.datasets.mnist.load_data(): MNIST 데이터셋을 로드합니다.
# (x_train, y_train): 훈련 데이터 (이미지, 레이블)
# (x_test, y_test): 테스트 데이터 (이미지, 레이블)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화 (Normalization):
# 이미지의 픽셀 값(0-255)을 0.0에서 1.0 사이의 값으로 변환합니다.
# 이는 모델의 학습을 안정화하고 성능을 향상시키는 데 도움이 됩니다.
x_train, x_test = x_train / 255.0, x_test / 255.0

# 이미지 데이터 형태 변경 (Reshaping/Flattening):
# 28x28 픽셀의 2차원 이미지를 784(28*28)개의 픽셀을 가진 1차원 벡터로 펼칩니다.
# 이는 Dense(완전 연결) 레이어의 입력으로 사용하기 위함입니다.
# x_train.shape[0]은 훈련 데이터의 샘플 개수(60000)입니다.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# 레이블(정답)을 원-핫 인코딩(One-Hot Encoding)으로 변환:
# 0~9까지의 정수형 레이블을 해당 클래스 위치만 1이고 나머지는 0인 벡터로 변환합니다.
# 예: 숫자 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)


# --- 3. 모델 구성 (Model Building) ---
# 단일 Dense 레이어를 가진 소프트맥스 분류 모델을 구성합니다.
model = tf.keras.Sequential()
# units=nb_classes (10): 출력 뉴런의 개수는 클래스 개수와 동일합니다.
# input_dim=784: 입력 특성의 개수는 784 (펼쳐진 이미지 픽셀 수)입니다.
# activation='softmax': 다중 클래스 분류를 위한 활성화 함수로, 각 클래스에 대한 확률을 출력합니다.
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=784, activation='softmax'))

# 모델 구조 요약 출력
model.summary()


# --- 4. 모델 컴파일 (Model Compilation) ---
# loss='categorical_crossentropy': 원-핫 인코딩된 레이블을 사용하는 다중 클래스 분류에 적합한 손실 함수입니다.
# optimizer=tf.optimizers.Adam(learning_rate): Adam 옵티마이저를 사용합니다.
#                                             SGD보다 일반적으로 더 좋은 성능을 보이며, 학습률 설정에 덜 민감합니다.
# metrics=['accuracy']: 훈련 및 평가 시 정확도를 측정합니다.
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])


# --- 5. 모델 훈련 (Model Training) ---
# model.fit(): 훈련 데이터를 사용하여 모델을 훈련시킵니다.
# batch_size: 한 번에 처리할 데이터 샘플의 수.
# epochs: 전체 훈련 데이터를 반복할 횟수.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1)


# --- 6. 모델 평가 (Model Evaluation) ---
# 훈련된 모델을 '테스트 데이터(x_test, y_test)'로 평가합니다.
# 이는 모델이 훈련 과정에서 보지 못한 새로운 데이터에 대해 얼마나 잘 작동하는지(일반화 성능)를 측정합니다.
print("\n--- Evaluation on Test Data ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# --- 7. 예측 (Prediction) ---
# 테스트 데이터의 일부를 사용하여 예측을 수행하고 결과를 확인합니다.
# model.predict()는 각 클래스에 대한 확률 분포를 반환합니다.
predictions = model.predict(x_test[:5]) # 테스트 데이터 중 첫 5개 샘플에 대해 예측
print('\n--- Predictions for first 5 test samples ---')
print('Predicted probabilities: \n', predictions)

# np.argmax()를 사용하여 가장 높은 확률을 가진 클래스의 인덱스(예측 숫자)를 찾습니다.
predicted_classes = np.argmax(predictions, axis=-1)
true_classes = np.argmax(y_test[:5], axis=-1) # 실제 정답도 원-핫 인코딩을 역변환하여 확인

print('Predicted classes: ', predicted_classes)
print('True classes:      ', true_classes)