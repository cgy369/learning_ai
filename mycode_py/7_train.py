import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf


xy = np.loadtxt("data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:-5, 0:-1]
y_data = xy[:-5, [-1]]
x_test = xy[-5:, 0:-1]
y_test = xy[-5:, [-1]]
# 값이 0부터 6까지 7개의 데이터를 갖는다. 이를 원핫 데이터로 만들어야한다.

y_one_hot = tf.keras.utils.to_categorical(y_data, 7)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=7, input_dim=16, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    metrics=["accuracy"],
)

model.summary()
history = model.fit(x_data, y_one_hot, epochs=1000, verbose=1)
# 모델 평가
results = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 7), verbose=0)
print(f"\n테스트 데이터 정확도: {results[1]:.4f}")


prediction = model.predict(x_test)
predicted_class = np.argmax(prediction, axis=-1)

# y_test_hot = tf.keras.utils.to_categorical(y_test, 7)
for idx in range(len(y_test)):
    print("예측된 A클래스:", predicted_class[idx], ", 정답은 :", y_test[idx][0])
    print()

print("\n--- 예측 vs 실제 ---")
for i in range(len(y_test)):
    print(f"예측: {predicted_class[i]}, 실제: {int(y_test[i][0])}")
