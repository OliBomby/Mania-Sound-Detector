import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
import os
import random
import pyperclip

import Generate_Data3 as Generate_Data
import train_sound_detector3 as tsd3
from keras.models import load_model

audio_size = Generate_Data.audio_size
folder = Generate_Data.folder
songs_folder = path.join(folder, "Test Songs")

time_interval = tsd3.time_interval
train_shape, div_shape, label_shape = (-1, 7, 32, 2), (-1, 4), (-1, 1)

##save_path = "C:\\Users\\Olivier\\Desktop\\NN\\Mania Sound Detector\\Saved models\\model.ckpt"
save_path = Generate_Data.save_path
save_rate = 100000

namesong = input("Name directory containing the song you would like to generate beats for: ")
beatmap_folder = path.join(songs_folder, namesong)
x_data, times, tick_data = Generate_Data.generate_test(beatmap_folder)
x_data = np.swapaxes(x_data, 2, 3);
div_data = np.array([[int(k%4==0), int(k%4==1), int(k%4==2), int(k%4==3)] for k in tick_data])

if x_data.shape[0]%time_interval > 0:
    x_data = x_data[:-(x_data.shape[0]%time_interval)];
    div_data = div_data[:-(div_data.shape[0]%time_interval)];
x_data2 = np.reshape(x_data, (-1, time_interval, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))

model = load_model('model3.h5')



test_predictions = model.predict([x_data2, div_data2]).reshape((-1, time_interval, label_shape[1]))

flat_test_preds = test_predictions.reshape(-1, label_shape[1]);

clear = lambda: os.system('cls')
clear()
prev = []
notes = []
for index, p in enumerate(flat_test_preds):
    noho = int(np.round(p))
    if noho < 0:
        noho = 0
    if noho > 4:
        noho = 0

        
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



copy = '\r\n'.join(notes)
pyperclip.copy(copy)
input('')



