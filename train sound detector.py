import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import Generate_Data

save_path = Generate_Data.save_path
save_rate = 50000

audio_size = Generate_Data.audio_size
x_data, y_data = Generate_Data.load_training_data()

assert len(x_data) == len(y_data)
p = np.random.permutation(len(x_data))
x_data, y_data = x_data[p], y_data[p]


batch_size = 20

if input("GAMER MODE (y/n): ") == 'y':
    gpu = 1
else:
    gpu = 0

with tf.device('/device:GPU:%s' % gpu):
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

    loss = tf.reduce_mean(tf.losses.absolute_difference(labels=y, predictions=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


def plot(ll, predictions, batchy):
    plt.subplot(1, 3, 1)
    plt.cla()
    plt.plot(ll)

    plt.subplot(1, 3, 3)
    plt.cla()
    plt.plot(batchy, color="red")
    plt.plot(predictions, color="green")

    plt.draw()
    plt.pause(0.0001)


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if input("Do you want to restore the model?(y/n): ") == "y":
        saver.restore(sess, save_path)
        print("Model restored.")

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    d = 0
    while True:
        i = d % int(np.floor(len(x_data)/batch_size))
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batchX = x_data[start_idx:end_idx, :]
        batchY = y_data[start_idx:end_idx, :]

        if d % 1000 == 0:
            train_loss, _predictions = sess.run([loss, y_conv], feed_dict={
                x: batchX, y: batchY, keep_prob: 1.0})
            loss_list.append(train_loss)
            print('step %d, training loss %g' % (d, train_loss))
            plot(loss_list, _predictions, batchY)

        if d % save_rate == 0 and d != 0:
            print("Saving model")
            saver.save(sess, save_path)
            print("Saved")

        train_step.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.3})
        d += 1


plt.ioff()
plt.show()
