import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
import os
import random

import Generate_Data

audio_size = Generate_Data.audio_size
folder = Generate_Data.folder
songs_folder = path.join(folder, "Test Songs")


##save_path = "C:\\Users\\Olivier\\Desktop\\NN\\Mania Sound Detector\\Saved models\\model.ckpt"
save_path = "D:\\Downloads\\Mania Sound Detector\\Saved models\\model.ckpt"
save_rate = 100000

namesong = input("Name song ding dong: ")
beatmap_folder = path.join(songs_folder, namesong)
x_data, times = Generate_Data.generate_test(beatmap_folder)


x = tf.placeholder(tf.float32, shape=[None, audio_size])
y = tf.placeholder(tf.float32, shape=[None, 2])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')


W_conv1 = weight_variable([1, 256, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 1, audio_size, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1],
                         strides=[1, 1, 2, 1], padding='SAME')

W_conv2 = weight_variable([1, 256, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1],
                         strides=[1, 1, 2, 1], padding='SAME')

W_fc1 = weight_variable([audio_size//4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, audio_size//4 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
predictions = tf.nn.softmax(y_conv)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, save_path)

    predictions_list = []
    ho_list = []
    xs = [64, 192, 320, 448]
    for index, _x in enumerate(x_data):
        bx = _x.reshape(1,audio_size)
        _predictions = sess.run([predictions], feed_dict={
            x: bx, keep_prob: 1.0})
        if _predictions[0][0][1] > 0.1:
            predictions_list.append(1)
            ho_list.append("%s,192,%s,1,0,0:0:0:0:" % (random.choice(xs), int(np.round(times[index]))))
        else:
            predictions_list.append(0)
for ho in ho_list:
    print(ho)


