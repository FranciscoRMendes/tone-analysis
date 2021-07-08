# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 07:40:44 2021

@author: Francisco
"""

os.chdir('C:/Users/Francisco/Documents/tone-analysis/dat/bank')
svi, sr = librosa.load('chunk-00.wav')
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import librosa.display
import scipy
fig, ax = plt.subplots()
S = np.abs(librosa.stft(svi,center=False,win_length = frame_length,hop_length = hop_length))
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                        ref=np.max),
                                y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

hop_length = int(10*0.001*sr)
frame_length = int(64*0.001*sr)
frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)

frame = frames[1]
sum = 0

y = frame
S = np.abs(librosa.stft(y,n_fft = 512))
freq_list = np.array(range(0,int((512/2)+1)))*sr/512
power = librosa.power_to_db(S**2,ref=1)
# power  = librosa.amplitude_to_db(S,ref=1)
lower_freqs = freq_list[freq_list<6000]
lower_freqs_power =  power[(freq_list<600),:].sum(axis=0)
peaks,_ = scipy.signal.find_peaks(lower_freqs_power)

max_peak = np.max([lower_freqs_power[i] for i in peaks])

higher_freqs_power = power[(freq_list<8000) & (freq_list>7000),:].sum(axis=0)
LT_marker = max_peak - sum(higher_freqs_power)<30

plt.plot(lower_freqs_power)
plt.plot(higher_freqs_power)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

count = 0
d = []
for frame in frames:
    y = frame
    S = np.abs(librosa.stft(y,n_fft = 512))
    freq_list = np.array(range(0,int((512/2)+1)))*sr/512
    power = librosa.power_to_db(S**2,ref=1)
    # power  = librosa.amplitude_to_db(S,ref=1)
    lower_freqs = freq_list[freq_list<6000]
    lower_freqs_power =  power[(freq_list<600),:].sum(axis=0)
    peaks,_ = scipy.signal.find_peaks(lower_freqs_power)
    if len(peaks)==0:
         max_peak = max(lower_freqs_power)
    else:
        max_peak = np.max([lower_freqs_power[i] for i in peaks])
    higher_freqs_power = power[(freq_list<8000) & (freq_list>7000),:].sum(axis=0)
    LT_marker = max_peak - np.sum(higher_freqs_power)<30
    d.append(max_peak - np.sum(higher_freqs_power))
    count+=LT_marker