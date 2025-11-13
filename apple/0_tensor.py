import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

텐서 = tf.constant([3, 4, 5, 6])  # shape 4
텐서2 = tf.constant([3, 4, 5, 6])
텐서3 = tf.constant([[1, 2], [3, 4]])  # shape 2, 2
텐서4 = tf.constant([[5, 6], [7, 8]])
텐서5 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])  # shape 2,4 2행 4열
텐서6 = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
print(tf.matmul(텐서5, 텐서6))
# print(텐서5)
# print(tf.add(텐서3, 텐서4))
# print((텐서 + 텐서2).numpy())

w = tf.Variable(1.0)  # weight 라고 생각한다.
w.assign = 2.0
print(w.numpy())
