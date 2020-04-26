import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
import os
import random
import pyperclip

import Generate_Data

audio_size = Generate_Data.audio_size
folder = Generate_Data.folder
songs_folder = path.join(folder, "Test Songs")

save_path = Generate_Data.save_path


namesong = input("Name directory containing the song you would like to generate beats for: ")

difficulty_setting = int(input("Difficulty Setting(1/2/3): "))
bias = float(input("Bias (-0.5/0.5): "))

beatmap_folder = path.join(songs_folder, namesong)

offsets = [-150, -100, -50, 0, 50, 100, 150]
x_data, times = Generate_Data.generate_test(beatmap_folder, offsets[0])
num_offsets = len(offsets)
num_outputs = len(times)
del x_data

with tf.device('/device:GPU:0'):
    x = tf.placeholder(tf.float32, shape=[None, audio_size])
    y = tf.placeholder(tf.float32, shape=[None, 1])


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

    W_conv3 = weight_variable([1, 256, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 2, 1],
                             strides=[1, 1, 2, 1], padding='SAME')

    W_fc1 = weight_variable([audio_size//8 * 128, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, audio_size//8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 100])
    b_fc2 = bias_variable([100])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    W_fc3 = weight_variable([100, 1])
    b_fc3 = bias_variable([1])

    y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

saver = tf.train.Saver()

clear = lambda: os.system('cls')
clear()
notes = []
all_predictions = np.empty([num_offsets, num_outputs])
split = [150*(n+1) for n in range(int(np.floor(num_outputs / 150)))]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    saver.restore(sess, save_path)

    for oi, offset in enumerate(offsets):
        x_data, times = Generate_Data.generate_test(beatmap_folder, offset)
        x_data_split = np.split(x_data, split)
        pred_list = []
        for xp in x_data_split:
            _predictions = sess.run([y_conv], feed_dict={
                x: xp, keep_prob: 1.0})
            pred_list.append(_predictions[0])
            print('.', end='')
        full_pred = np.vstack(pred_list)
        all_predictions[oi] = np.reshape(full_pred, num_outputs)
        print('')
        

def generate_beats(difficulty_setting, bias):        
    if difficulty_setting == 3:
        max_predictions = np.round(np.amax(all_predictions, 0) + bias)
    elif difficulty_setting == 1:
        max_predictions = np.round(np.amin(all_predictions, 0) + bias)
    else:
        max_predictions = np.round(np.mean(all_predictions, 0) + bias)

    prev = []
    for index, noho in enumerate(max_predictions):
        if noho < 0:
            noho = 0
        if noho > 4:
            noho = 4
        noho = int(noho)
            
        xs = [64, 192, 320, 448]
        for n in prev:
            xs.remove(n)
        if len(xs) == 0:
            xs = [64, 192, 320, 448]
        ta = []
        for n in range(noho):
            xp = random.choice(xs)
            ta.append(xp)
            note = "%s,192,%s,1,0,0:0:0:0:" % (xp, int(np.round(times[index])))
            print(note)
            notes.append(note)
            xs.remove(xp)
            if len(xs) == 0:
                xs = [64, 192, 320, 448]
                for p in ta:
                    xs.remove(p)
        prev = ta
            
    print("Dönner!")
    copy = '\r\n'.join(notes)
    pyperclip.copy(copy)
    input('')   

generate_beats(difficulty_setting, bias)


def save():
    _path = path.join(folder, "predictions.npy")
    np.save(_path, all_predictions)

    
def load():
    _path = path.join(folder, "predictions.npy")
    a = np.load(_path)
    return a


save()




