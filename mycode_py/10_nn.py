import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model = tf.keras.Sequential(
    [tf.keras.Input(shape=(784,))]
)  # 시퀀셜에서 input의 shape를 지정하는게 최신이다.
model.add(
    tf.keras.layers.Dense(512, activation="relu", kernel_initializer="glorot_normal")
)

# model.add(tf.keras.layers.Dense(512, input_dim=784, activation="relu",kernel_initializer='glorot_normal')) # 가중치 초기화 방법을 넣어준다. Xavier Normal
# model.add(tf.keras.layers.Dense(512, input_shape=(784,), activation="relu")) # 1차원의 경우 둘은 같은 의미이다. 여기서 전처리 없이 mnist값을 쓴다고 하면 input_shape=(28,28,1) 이 된다.
# model.add(tf.keras.layers.Dropout(drop_rate)) # 드롭아웃 적용 0.3 이면 10개중 3번은 안쓴다는 소리이다. 참고로 마지막 레이어는 dropout을 쓰지 않는다.
model.add(
    tf.keras.layers.Dense(
        units=10, activation="softmax", kernel_initializer="glorot_normal"
    )
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, epochs=10, verbose=2)
print("\n--- Final Accuracy ---")
score = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy: ", score[1])
