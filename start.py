import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# TensorFlow 2.x는 즉시 실행(Eager Execution)이 기본입니다.
# 이는 코드가 실행되는 즉시 연산이 수행된다는 의미로,
# TensorFlow 1.x에서 필요했던 tf.Session() 객체를 더 이상 만들거나 사용할 필요가 없습니다.

# --- 1. 상수(Constant) 및 연산 ---
# tf.constant로 텐서(상수)를 정의하면 바로 값을 가집니다.
hello = tf.constant("Hello, TensorFlow!")

# .numpy()를 사용해 텐서의 값을 가져올 수 있습니다.
# 문자열은 바이트(byte) 형태이므로 .decode()로 변환해줍니다.
print(hello.numpy().decode('utf-8'))


# 숫자 텐서의 경우도 마찬가지입니다.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # 자료형을 생략해도 자동으로 tf.float32로 추론됩니다.
node3 = tf.add(node1, node2) # 연산이 즉시 실행됩니다.

# sess.run() 없이 바로 .numpy()로 값을 확인할 수 있습니다.
print("node1:", node1.numpy(), "node2:", node2.numpy())
print("node3:", node3.numpy())


# --- 2. 플레이스홀더(Placeholder)와 feed_dict 대체 ---
# TensorFlow 1.x의 tf.placeholder와 feed_dict는 2.x에서 사라졌습니다.
# 대신, 일반적인 파이썬 함수를 사용합니다.

# a와 b를 입력받아 더하는 간단한 파이썬 함수를 정의합니다.
def adder(a, b):
  return a + b

# 일반 함수처럼 값을 전달하여 바로 호출할 수 있습니다.
# feed_dict가 필요 없습니다.
print("adder(3, 4.5):", adder(3, 4.5))
print("adder([1, 3], [2, 4]):", adder(tf.constant([1, 3]), tf.constant([2, 4])).numpy())


# --- 3. 그래프의 성능이 필요할 때: @tf.function ---
# TensorFlow 1.x의 '그래프 모드'가 주는 성능상의 이점을 2.x에서도 누리고 싶을 때 사용합니다.
# @tf.function 데코레이터를 함수 위에 붙이면,
# TensorFlow가 이 함수를 고성능 그래프로 컴파일하여 실행합니다.
@tf.function
def add_and_triple(a, b):
  # 함수 내부는 일반 TensorFlow 연산을 사용합니다.
  return (a + b) * 3.0

# 컴파일된 함수를 일반 함수처럼 호출합니다.
print("add_and_triple(3, 4.5):", add_and_triple(3, 4.5).numpy())
