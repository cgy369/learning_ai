import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

hello = tf.constant("Hello Tensor FLow")
print(hello.numpy().decode("utf-8"))
print(hello.numpy())

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.5)
node3 = tf.add(node1, node2)

print(node1.numpy())
print(node2.numpy())
print(node3.numpy())


def adder(a, b):
    return a + b


a = adder(node1.numpy(), node2.numpy())
print(a)


@tf.function
def adder_tf(a, b):
    return a + b


b = adder_tf(node1, node2)
print(b.numpy())
