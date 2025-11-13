import tensorflow as tf
import numpy as np

# --- 1. 데이터 불러오기 및 준비 (Loading & Preparing Data) ---
# 동물 데이터셋(CSV)을 불러옵니다.
# 각 행은 동물의 특성(털, 깃털, 알, 우유 등 16개)과 동물의 종류(0~6, 총 7가지)를 나타냅니다.
# (참고: 'data-04-zoo.csv' 파일이 이 스크립트와 동일한 폴더에 존재해야 합니다.)
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]  # 16개의 특성
y_data = xy[:, [-1]]  # 1개의 결과 (0~6 사이의 정수)

# 클래스의 개수를 정의합니다. (동물 종류 0부터 6까지, 총 7종류)
nb_classes = 7

# y_data를 원-핫 인코딩(One-Hot Encoding)으로 변환합니다.
# to_categorical 함수는 정수형 레이블(e.g., 2)을 원-핫 벡터(e.g., [0., 0., 1., 0., 0., 0., 0.])로 쉽게 변환해줍니다.
# 다중 클래스 분류에서 'categorical_crossentropy' 손실 함수를 사용하려면 이 변환이 필수적입니다.
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("y_data(first 5): \n", y_data[:5])
print("y_one_hot(first 5): \n", y_one_hot[:5])


# --- 2. 모델 구성 및 컴파일 (Model Building & Compilation) ---
model = tf.keras.Sequential()

# 입력 특성은 16개, 출력 클래스는 7개인 소프트맥스 분류 모델을 구성합니다.
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))

# 모델을 컴파일합니다.
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()


# --- 3. 모델 훈련 (Model Training) ---
# 훈련 데이터는 x_data, 정답 레이블은 원-핫 인코딩된 y_one_hot을 사용합니다.
history = model.fit(x_data, y_one_hot, epochs=1000, verbose=0)


# --- 4. 예측 및 평가 (Prediction & Evaluation) ---

# 단일 데이터에 대한 테스트
# test_data는 깃털이 있고(feathers), 알을 낳는(eggs) 등의 특성을 가진 조류(type 3) 데이터입니다.
test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])
prediction = model.predict(test_data)
predicted_class = np.argmax(prediction, axis=-1)

print("Prediction (probabilities):", prediction)
print("Predicted class (index):", predicted_class)


# 전체 훈련 데이터에 대한 예측 및 정확도 확인
# 훈련된 모델이 전체 데이터에 대해 얼마나 잘 예측하는지 하나씩 확인해봅니다.
all_predictions = model.predict(x_data)
predicted_classes = np.argmax(all_predictions, axis=-1)

# zip 함수를 사용하여 예측값(p)과 실제 정답(y)을 하나씩 짝지어 반복합니다.
# y_data.flatten()은 [[0], [3], ...] 형태의 2D 배열을 [0, 3, ...] 형태의 1D 배열로 변환합니다.
correct_count = 0
for p, y in zip(predicted_classes, y_data.flatten()):
    is_correct = (p == int(y))
    print("[{}] Prediction: {} True Y: {}".format(is_correct, p, int(y)))
    if is_correct:
        correct_count += 1

print("\nAccuracy on training data: {:.2f}%".format(correct_count / len(y_data) * 100))
