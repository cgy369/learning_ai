import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np


t = np.array([[1.0, 2.0, 3.0], [5.0, 6, 7], [7.0, 6, 7]])
mat1 = np.array([[1.0, 2], [3.0, 4]])
mat2 = np.array([[4.0], [3.0]])
result = tf.matmul(mat1, mat2)
print(result.shape, result.ndim, result)
# print(t.shape)
# print(t.ndim)
