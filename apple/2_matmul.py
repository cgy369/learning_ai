import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

# 1. 여러 특성을 가진 2차원 데이터 준비
키_데이터 = np.array([[170, 70], [180, 85], [165, 60], [155, 55]], dtype=np.float32)
신발_데이터 = np.array([[260], [275], [255], [245]], dtype=np.float32)

# 2. 모델 변수 초기화
a = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

# 3. 옵티마이저 정의
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 4. 손실 함수 정의
def loss_func():
    예측값 = tf.matmul(키_데이터, a) + b
    return tf.reduce_mean(tf.square(신발_데이터 - 예측값))

# 5. 훈련 (경사 하강법 - tf.GradientTape 사용)
for i in range(2000):
    with tf.GradientTape() as tape:
        loss = loss_func()
    grads = tape.gradient(loss, [a, b])
    opt.apply_gradients(zip(grads, [a, b]))
    if i % 100 == 0:
        print(f"Epoch {i}: Loss={loss.numpy():.4f}")

# 6. 훈련된 결과 확인
print(f"\n훈련 완료! a = {a.numpy().flatten()}, b = {b.numpy()}")
print(f"키 170cm, 몸무게 70kg에 대한 예측 신발 사이즈: {(tf.matmul(np.array([[170, 70]], dtype=np.float32), a) + b).numpy().flatten()[0]:.2f}")
