import tensorflow as tf

session = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
aplusb = tf.add(a, b)

print session.run(aplusb, {a: [1.0, 2], b: [2.0, 3]})
