import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

# --- 1. 데이터 준비 (Data Preparation) ---
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# --- 2. 모델 구성 (Model Definition) ---
# XOR 문제는 은닉층(Hidden Layer)이 하나만 있어도 충분히 해결 가능합니다.
# 복잡한 문제일수록 깊게 쌓는 것이 의미가 있지만, XOR에서는 오히려 방해가 될 수 있습니다.
model = tf.keras.Sequential()

# 은닉층 (Hidden Layer): 1개
# - units=10: 은닉층의 뉴런(노드) 수. 2개 이상이면 충분합니다.
# - input_dim=2: 입력 특성은 2개 (x1, x2)
# - activation='sigmoid': 비선형 활성화 함수는 필수입니다.
model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation="relu"))

# 출력층 (Output Layer): 1개
# - units=1: 최종 출력은 1개 (0 또는 1)
# - activation='sigmoid': 출력을 0과 1 사이의 확률 값으로 만듭니다.
# model.add(tf.keras.layers.Dense(units=10, activation="relu"))
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# --- 3. 모델 컴파일 (Model Compilation) ---
# 학습률(learning_rate)을 높여 더 빠르고 효과적으로 학습하도록 설정합니다.
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    metrics=["accuracy"],
)

# --- 4. Early Stopping 콜백 설정 (Early Stopping Callback) ---
# 'patience' 횟수(1000번) 동안 정확도(accuracy)가 개선되지 않으면 학습을 자동으로 멈춥니다.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=500, baseline=1.0, mode="max", verbose=1
)

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="accuracy", patience=3000, mode="max", verbose=1
# )

# --- 5. 모델 훈련 (Model Training) ---
# 충분한 학습을 위해 epochs를 10000으로 설정합니다.
# verbose=2로 설정하여 훈련 과정을 화면에 상세히 출력합니다.
print("\nStarting training...")
# history = model.fit(x_data, y_data, epochs=10000, verbose=2)
history = model.fit(x_data, y_data, epochs=5000, verbose=2, callbacks=[early_stopping])
print("Training finished.")

# --- 6. 결과 확인 (Evaluation) ---
# 훈련된 모델의 가중치와 편향을 확인합니다.
print("\n--- Learned Parameters ---")
try:
    weights0, bias0 = model.layers[0].get_weights()
    weights1, bias1 = model.layers[1].get_weights()
    print("Layer 0 (Hidden Layer) Weights:\n", weights0)
    print("Layer 0 (Hidden Layer) Bias:\n", bias0)
    print("\nLayer 1 (Output Layer) Weights:\n", weights1)
    print("Layer 1 (Output Layer) Bias:\n", bias1)
except Exception as e:
    print("Could not retrieve weights:", e)


# 최종 정확도를 평가하고 출력합니다. 1.0이 나와야 합니다.
print("\n--- Final Accuracy ---")
score = model.evaluate(x_data, y_data, verbose=0)
print("Accuracy: ", score[1])
