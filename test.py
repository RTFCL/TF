# https://www.tensorflow.org/get_started/get_started
import tensorflow as tf
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn.datasets import fetch_20newsgroups

fetch_20newsgroups()

session = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
aplusb = tf.add(a, b)

print session.run(aplusb, {a: [1.0, 2], b: [2.0, 3]})
