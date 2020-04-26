import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
import os
import random
import pyperclip

import osureader
import subprocess
from scipy.io.wavfile import read
from scipy.signal import find_peaks_cwt
import peakutils
import time
import datetime
import json

sampling_rate = 32000
audio_size = 1920
ask_gpu = False

folder = path.dirname(path.abspath("__file__"))
songs_folder = path.join(folder, "Test Songs")

save_folder = path.join(folder, "Saved Models")
save_path = path.join(save_folder, "model2.ckpt")


def get_audio(folder):
    beatmap_list = [f for f in os.listdir(folder) if f[-4:] == ".osu"]
    audio_path = None
    wav_path = None
    audio_filename = None
    if len(beatmap_list) > 0:
        beatmap_path = path.join(folder, beatmap_list[0])

        beatmap = osureader.readBeatmap(beatmap_path)
        audio_filename = beatmap.AudioFilename
        audio_path = path.join(folder, audio_filename)
        wav_path = path.join(folder, "audio.wav.wav")
        
        print(beatmap.Title)
    else:
        mp3_list = [f for f in os.listdir(folder) if f[-4:] == ".mp3"]
        audio_filename = mp3_list[0]
        audio_path = path.join(folder, audio_filename)
        wav_path = path.join(folder, "audio.wav.wav")

        print(mp3_list[0])
        
    if not path.exists(wav_path):
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                         wav_path])
    audio = read(wav_path)
    if not audio[0] == sampling_rate:
        os.remove(wav_path)
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                         wav_path])
        audio = read(wav_path)
        
    audio = audio[1]
    audio_ms = len(audio) / sampling_rate * 1000
    audio = np.divide(np.concatenate((np.zeros(audio_size // 2), audio, np.zeros(audio_size // 2))), 32767)
    return audio, audio_ms, audio_filename


def get_data(audio, start, finish):
    x_list = []
    time = start
    while time < finish:
        audio_index = int(time * sampling_rate / 1000)
        x_list.append(audio[audio_index:audio_index + audio_size])
        time += 1

    x = np.vstack(x_list)
    return x.astype(np.float32, copy=False)

if ask_gpu:
    if input("GAMER MODE?(y/n): ") == "y":
        gpu = 1
    else:
        gpu = 0
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

saver = tf.train.Saver()


def make_split(total, split):
    return [split *(n+1) for n in range(int(np.floor(total / split)))]


def generate_predictions(b=100, s=10000):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path)

        namesong = input("Name directory containing the song you would like to generate beats for: ")
        beatmap_folder = path.join(songs_folder, namesong)

        audio, audio_ms, audio_filename = get_audio(beatmap_folder)
        num_outputs = int(np.ceil(audio_ms))

        split = make_split(num_outputs, s)
        split.append(num_outputs)

        print("Running audio analysis!")
        pred_list = []
        index = 0
        num = 0
        last_time = time.time()
        first_time = time.time()
        for n in split:
            x_data = get_data(audio, index, n)
            index = n
            
            xsplit = make_split(len(x_data), b)
            x_data_split = np.split(x_data, xsplit)
            
            for xp in x_data_split:
                _predictions = sess.run([y_conv], feed_dict={
                    x: xp, keep_prob: 1.0})
                pred_list.append(_predictions[0])
                num += len(_predictions[0])
                
                if time.time() - last_time > 10:
                    print("Progression:", round(num / num_outputs * 100, 3), "%")
                    last_time = time.time()
    
        full_pred = np.reshape(np.vstack(pred_list), num_outputs)
        
        print("Elapsed time:", np.round(time.time() - first_time, 3), "seconds!")
        return full_pred, audio_filename


def find_peaks(a, d, thres=0.25):
    indexes = peakutils.indexes(a, thres=thres/max(a), min_dist=d)
    interpolatedIndexes = peakutils.interpolate(np.array(range(0, len(a))), a, ind=indexes)
    return interpolatedIndexes


def make_snaps(start, finish, increment):
    times = []
    time = start
    while time < peaks[-1] + increment:
        times.append(time)
        time += increment
    return times


def test_offset(peaks, bpm, offset):
    beat_time = 60000 / bpm
    div4 = beat_time / 4
    div3 = beat_time / 3

    times4 = make_snaps(offset, peaks[-1] + beat_time, div4)
    times3 = make_snaps(offset, peaks[-1] + beat_time, div3)

    times = times4 + list(set(times3) - set(times4))
          
    losses = []
    for peak in peaks:
        lowest = min([abs(peak - t) for t in times])
        losses.append(lowest)

    loss = np.mean(losses)
    return loss


def snap_peaks(peaks, bpm, offset):
    print("Snapping to BPM")
    
    beat_time = 60000 / bpm
    div4 = beat_time / 4
    div3 = beat_time / 3

    times4 = make_snaps(offset, peaks[-1] + beat_time, div4)
    times3 = make_snaps(offset, peaks[-1] + beat_time, div3)

    times = times4 + list(set(times3) - set(times4))

    new_peaks = []
    for peak in peaks:
        new_peaks.append(min(times, key=lambda x: abs(x - peak)))
    return new_peaks


def find_timing(a):
    print("Finding BPM and Offset")
    # plt.ion()
    # plt.figure()
    # plt.show()
    # plt.ylim(0, 2500)
    
    a = sorted(a)
    r = range(1, min(1000, len(a)-10))
    delta_peakss = []
    for n in r:
        dp = [abs(j-i) for i, j in zip(a[:-n], a[n:])]
        # plt.plot(dp, '.', color='blue')
        delta_peakss.append(dp)
        
    delta_peaks = []
    for dp in delta_peakss:
        delta_peaks = delta_peaks + dp

    # print("Average interval: ", np.mean(delta_peaks))
    # print("Middle interval: ", sorted(delta_peaks)[int(len(delta_peaks) // 2)])

    round_delta_peaks = np.round(delta_peaks).astype(int)

    dpv = [0 for n in range(0, np.amax(round_delta_peaks) + 1)]
    for x in round_delta_peaks:
        dpv[x] += 1

    # plt.plot(dpv, color='green')

    dpvv = np.convolve(dpv, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 'same') +30
    # plt.plot(dpvv, color='green')

    ppeak = find_peaks(dpvv, 50)
    toppeak = None
    prev = 0
    for p in ppeak:
        h = dpvv[int(round(p))]
        if h > prev:
            prev = h
            toppeak = p

    # for pppp in ppeak:
    #     plt.plot([0, len(a)], [pppp, pppp], color='red')
    
    # print("Top peak: ", toppeak)

    BPM = 960000 / toppeak

    for n in range(7):
        if abs(BPM / 2 - 200) < abs(BPM - 200):
            BPM = BPM / 2

    bpms = []
    for pp in ppeak:
        for n in range(1,len(ppeak)):
            test = (pp / n)
            bpm = 960000 / test
            for no in range(7):
                if abs(bpm / 2 - 220) < abs(bpm - 220):
                    bpm = bpm / 2
            if abs(BPM - bpm) < 0.01:
                bpms.append(bpm)

    BPM = np.mean(bpms)

    for n in range(7):
        if abs(BPM / 2 - 200) < abs(BPM - 200):
            BPM = BPM / 2

    if abs(BPM - round(BPM)) < 0.2:
        BPM = round(BPM)

    beat_time = 60000 / BPM
    # plt.draw()
    # plt.pause(0.0001)

    d = 0.5 * beat_time
    offset = 0
    while d > 0.05:
        right = test_offset(a, BPM, offset + d)
        left = test_offset(a, BPM, offset - d)

        if right < left:
            offset += d
        else:
            offset -= d
        d = d / 2

    offset = int(round(offset))
        
    print("BPM: ", BPM)
    print("Offset: ", offset)
    return BPM, offset


def generate_beats(a):
    print("Generating notes")

    a = np.round(a).astype(int)
    
    notes = []
    prev = []
    for time in a:
        noho = int(round(np.amax(all_predictions[time-2:time+2])))
        
        if noho > 4:
            noho = 4
        elif noho < 0:
            hoho = 0
            
        xs = [64, 192, 320, 448]
        for n in prev:
            xs.remove(n)
        if len(xs) == 0:
            xs = [64, 192, 320, 448]
        ta = []
        for n in range(noho):
            xp = random.choice(xs)
            ta.append(xp)
            note = "%s,192,%s,1,0,0:0:0:0:" % (xp, int(time))
            # print(note)
            notes.append(note)
            xs.remove(xp)
            if len(xs) == 0:
                xs = [64, 192, 320, 448]
                for p in ta:
                    xs.remove(p)
        prev = ta
            
    # print("Dönner!")
    copy = '\r\n'.join(notes)
    pyperclip.copy(copy)
    return  notes


def load():
    _path = path.join(folder, "predictions.npy")
    a = np.load(_path)
    return a, None

    
def save():
    _path = path.join(folder, "predictions.npy")
    np.save(_path, all_predictions)


def savejson():
    with open('data.json', 'w') as outfile:
        json.dump(all_predictions.tolist(), outfile)


def draw(at, N=500, pwid=100, thres=0.25):
    plt.cla()
    x = np.arange(N) + at
    z = all_predictions[at:at+N]
    z_conv = np.convolve(z, np.full(25, 0.04), 'same')
    zf = find_peaks(z_conv, pwid, thres)
    plt.cla()
    plt.plot(x, z)
    plt.plot(x, z_conv)
    zf = np.round(zf).astype(int)
    plt.plot(x[zf], z[zf], '*', ms=20, color='green')
    xlow = np.maximum(np.array(zf) - pwid/2, 0)
    xhigh = np.minimum(np.array(zf) + pwid/2, x.max())
    zguess = 0*xlow # allocate space
    for ii in range(len(zf)):
       zguess[ii] = z[int(xlow[ii]):int(xhigh[ii])].mean()
    plt.plot(x[zf], zguess, 'o', ms=10, color='red')


def export_map(notes, bpm, offset, audio_filename=None, name=None):
    if audio_filename is None:
        audio_filename = "audio.mp3"
    if name is None:
        name = datetime.datetime.now().strftime("beatmap %Y%m%d-%H%M%S.osu")
        
    print("Exporting map to the Exports folder")

    empty_path = path.join(folder, "empty.osu")
    exports_folder = path.join(folder, "Exports")
    export_path = path.join(exports_folder, name)

    timingpoint = "%s,%s,4,2,0,100,1,0" % (offset, 60000 / bpm)
    
    f = open(empty_path, "r")
    full_file = f.read() % (audio_filename, timingpoint, '\n'.join(notes))
    f.close()

    nf = open(export_path, "w+")
    nf.write(full_file)
    nf.close()

    
all_predictions, audio_filename = generate_predictions()
##all_predictions, audio_filename = load()

save()

pwid = 70
conv = 25
thr = 0.4
peaks = find_peaks(np.convolve(all_predictions[5000:50000], np.full(conv, 1 / conv), 'same'), pwid, thr)
####bpm, offset = find_timing(peaks)
##snapped_peaks = snap_peaks(peaks, bpm, offset)
notes = generate_beats(peaks)
export_map(notes, 100, 0, audio_filename)
