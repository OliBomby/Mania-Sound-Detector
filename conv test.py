from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


##sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
##
##x = tf.constant([t for t in range(100)], shape=(1,100), dtype=tf.float32)
##y = tf.constant(4, dtype=tf.float32)
##
##out = tf.scalar_mul(y,x)
##
##
##
##sess.run(tf.global_variables_initializer())

# Creates a graph.
with tf.device("/device:GPU:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
