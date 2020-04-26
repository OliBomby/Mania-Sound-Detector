import numpy as np

from os import path
import os
import random

import osureader

import subprocess
from scipy.io.wavfile import read

sampling_rate = 32000
audio_size = 1920

folder = path.dirname(path.abspath(__file__))
songs_folder = path.join(folder, "Training Songs")
save_folder = path.join(folder, "Saved Models")
save_path = path.join(save_folder, "model2.ckpt")

data_folder = path.join(folder, "Training_data")
audio_path = path.join(data_folder, "train-audio2.npy")
labels_path = path.join(data_folder, "train-labels2.npy")

# data_folder = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs"


def validate_bpm(beatmap):
    foundanybpm = False
    for t in beatmap.TimingPoints:
        if t[6] == 1:
            foundanybpm = True
            if 60000 / t[1] <= 20 or 60000 / t[1] >= 500:
                return False
    if not foundanybpm:
        return False
    return True


def get_random_song():
    # get random beatmap dir
    beatmap_folder = path.join(songs_folder, random.choice(os.listdir(songs_folder)))
    beatmap_list = [f for f in os.listdir(beatmap_folder) if f[-4:] == ".osu"]
    beatmap_path = path.join(beatmap_folder, random.choice(beatmap_list))

    # open the beatmap
    beatmap = osureader.readBeatmap(beatmap_path)

    # find the audio
    audio_path = path.join(beatmap_folder, beatmap.AudioFilename)

    # open the audio
    wav_path = path.join(beatmap_folder, "audio.wav.wav")
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

    return audio, beatmap


def get_song(i):
    # get random beatmap dir
    beatmap_folder = path.join(songs_folder, os.listdir(songs_folder)[i])
    beatmap_list = [f for f in os.listdir(beatmap_folder) if f[-4:] == ".osu"]
    beatmap_path = path.join(beatmap_folder, random.choice(beatmap_list))

    # open the beatmap
    beatmap = osureader.readBeatmap(beatmap_path)

    # find the audio
    audio_path = path.join(beatmap_folder, beatmap.AudioFilename)

    # open the audio
    wav_path = path.join(beatmap_folder, "audio.wav.wav")
    if not path.exists(wav_path):
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                         wav_path])
    audio = read(wav_path)

    if not audio[0] == sampling_rate:
##        os.remove(wav_path)
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                         wav_path])
        audio = read(wav_path)

    audio = audio[1]

    return audio, beatmap


def get_bpm_at_time(beatmap, time):
    bpm = 0
    index = 0
    while beatmap.TimingPoints[index][0] <= time or bpm == 0:
        if beatmap.TimingPoints[index][6] == 1:
            if 20 < 60000 / beatmap.TimingPoints[index][1] < 500:
                bpm = 60000 / beatmap.TimingPoints[index][1]
        index += 1
        if index == len(beatmap.TimingPoints):
            break
    return bpm


def get_note_at_time(beatmap, time):
    notes = 0
    for ho in beatmap.HitObjects:
        if time + 10 > ho.time > time - 10:
            notes += 1
    return notes


def generate_data(i):
    x_list = []
    y_list = []

    # audio, beatmap = get_random_song()
    audio, beatmap = get_song(i)
    print(beatmap.Title)
    audio_ms = len(audio) / sampling_rate * 1000

    redpoints = []
    for tp in beatmap.TimingPoints:
        if tp[6] == 1:
            redpoints.append(tp)

    for index in range(len(redpoints) - 1):
        rp = redpoints[index]
        time = rp[0]
        while time + 30 < redpoints[index + 1][0]:
            audio_index = int(time * sampling_rate / 1000) - audio_size // 2
            if audio_index >= 0:
                x_list.append(audio[audio_index:audio_index + audio_size])
                y_list.append(get_note_at_time(beatmap, time))
            time += rp[1] / 4

    rp = redpoints[-1]
    time = rp[0]
    while time + 200 < audio_ms:
        audio_index = int(time * sampling_rate / 1000) - audio_size // 2
        if audio_index >= 0:
            x_list.append(audio[audio_index:audio_index + audio_size])
            y_list.append(get_note_at_time(beatmap, time))
        time += rp[1] / 4


    x = np.vstack(x_list)
    y = np.vstack(y_list)

    x = np.divide(x, 32767)

    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

def generate_test(folder, offset):
    beatmap_list = [f for f in os.listdir(folder) if f[-4:] == ".osu"]
    beatmap_path = path.join(folder, beatmap_list[0])
    x_list = []
    times = []

    beatmap = osureader.readBeatmap(beatmap_path)
    audio_path = path.join(folder, beatmap.AudioFilename)
    wav_path = path.join(folder, "audio.wav.wav")
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
    
    print(beatmap.Title)
    audio_ms = len(audio) / sampling_rate * 1000

    redpoints = []
    for tp in beatmap.TimingPoints:
        if tp[6] == 1:
            redpoints.append(tp)

    for index in range(len(redpoints) - 1):
        rp = redpoints[index]
        time = rp[0]
        while time + 30 < redpoints[index + 1][0]:
            audio_index = int(time * sampling_rate / 1000) - audio_size // 2 + offset
            if audio_index >= 0:
                x_list.append(audio[audio_index:audio_index + audio_size])
                times.append(time)
            time += rp[1] / 4

    rp = redpoints[-1]
    time = rp[0]
    while time + 200 < audio_ms:
        audio_index = int(time * sampling_rate / 1000) - audio_size // 2 + offset
        if audio_index >= 0:
            x_list.append(audio[audio_index:audio_index + audio_size])
            times.append(time)
        time += rp[1] / 4

    x = np.vstack(x_list)

    x = np.divide(x, 32767)

    return x.astype(np.float32, copy=False), times

def generate_training_data():
    x_old, y_old = generate_data(0)
    for i in range(1,100):
        x, y = generate_data(i)
        x_old = np.concatenate((x_old, x), axis=0)
        y_old = np.concatenate((y_old, y), axis=0)
    x, y = x_old, y_old
    try:
        x_old = np.load(audio_path)
        y_old = np.load(labels_path)
        print("Loaded existing training data")
        x = np.concatenate((x_old, x), axis=0)
        y = np.concatenate((y_old, y), axis=0)
    except FileNotFoundError:
        pass
    np.save(audio_path, x)
    np.save(labels_path, y)


def load_training_data():
    x = np.load(audio_path)
    y = np.load(labels_path)
    return x, y

if __name__ == "__main__":
    generate_training_data()
