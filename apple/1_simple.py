import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

x_train = [1, 2, 3, 4]
# y_train: 출력 데이터 (종속 변수). 모델이 예측해야 할 정답(label)입니다.
# 데이터의 관계를 보면 y = -x + 1 이라는 것을 알 수 있습니다.
# 모델은 이 관계를 학습하여 W는 -1에, b는 1에 가까워지도록 훈련될 것입니다.
y_train = [0, -1, -2, -3]

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
a = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))


def loss_func():
    예측값 = x_train * a + b
    return tf.square(예측값 - y_train)


for i in range(2000):
    with tf.GradientTape() as tape:
        loss = loss_func()
    grads = tape.gradient(loss, [a, b])
    opt.apply_gradients(zip(grads, [a, b]))
    if i % 100 == 0:
        print(f"Epoch {i}: Loss={loss}")

print(f"\n훈련 완료! a = {a.numpy().flatten()}, b = {b.numpy()}")
