# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 06:00:55 2021

@author: Francisco
"""


import numpy as np
import sys
import numpy
import wave
import math
from scipy.signal import lfilter, hamming
# from scikits.talkbox import lpc
import librosa 
import os

file_path = 'ae.wav'
# Read from file.
spf = wave.open(file_path, 'r') # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

svi, sr = librosa.load('ae.wav')
# Get file as numpy array.
x = spf.readframes(-1)
x = np.fromstring(x, 'Int16')

# Get Hamming window.
N = len(x)
w = np.hamming(N)

# Apply window and high pass filter.
x1 = x * w
x1 = lfilter([1], [1., 0.63], x1)

# Get LPC.
# A, e, k = lpc(x1, 8)
Fs = spf.getframerate()
ncoeff = 2 + Fs / 1000
A = librosa.lpc(x1, int(ncoeff))

# A = librosa.lpc(x1,8)

# Get roots.
rts = np.roots(A)
rts = [r for r in rts if np.imag(r) >= 0]

# Get angles.
angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

# Get frequencies.
Fs = spf.getframerate()
frqs = sorted(angz * (Fs / (2 * math.pi)))
s = angz * (Fs / (2 * math.pi))
indices = sorted(range(len(s)), key=lambda k: s[k])

# pi = 3.14
# bw = -1/2*(Fs/(2*np.pi))*np.log(abs(rts(indices)))

bw = [-1/2*(Fs/(2*np.pi))*np.log(abs(rts[i])) for i in indices]
return frqs

os.chdir('C:/Users/Francisco/Documents/tone-analysis/dat/bank')
svi, sr = librosa.load('chunk-00.wav')



# import matplotlib.pyplot as plt
# import librosa.display
# fig, ax = plt.subplots()
# S = np.abs(librosa.stft(svi,center=False,win_length = frame_length,hop_length = hop_length))
# img = librosa.display.specshow(librosa.amplitude_to_db(S,
#                                                         ref=np.max),
#                                 y_axis='log', x_axis='time', ax=ax)
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format="%+2.0f dB")

hop_length = int(10*0.001*sr)
frame_length = int(64*0.001*sr)

def get_formants(x, Fs):
    N = len(x)
    w = np.hamming(N)
    
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    ncoeff = 2 + Fs / 1000
    A = librosa.lpc(x1, int(ncoeff))
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))
    frqs = angz * (Fs / (2 * math.pi))
    rts = np.array(rts)
    bw = -1/2*(Fs/(2*np.pi))*np.log(abs(rts))
    
    frqs = [round(frqs[i],2) for i in range(0,len(frqs)) if bw[i]<400 and frqs[i]>90]
    frqs = sorted(frqs)
    return(frqs[1:3])


import matplotlib.pyplot as plt
frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)
f1 = []
f2 = []
for frame in frames:
    F1, F2 = get_formants(frame, sr)
    f1.append(F1)
    f2.append(F2)
    
    
y = scipy.signal.medfilt(np.array(f1), 99)
f1_1 = scipy.signal.medfilt(np.array(f1), 99)

plt.plot(f1)
plt.plot(y)

plt.plot(f1)
plt.plot(f1_1)
#formant slope is the 
mf1 = [np.polyfit(y = f1[0:i],x=range(0,i),deg=1)[0]/10 for i in range(2,len(f1))]
mf2 = [np.polyfit(y = f2[0:i],x=range(0,i),deg=1)[0]/10 for i in range(2,len(f2))]

mf1 = np.diff(f1_1)
frame = frames[1]
F1, F2 = get_formants(frame, sr)

plt.plot(svi)
plt.plot(mf1)
plt.plot(mf2)

x, Fs = librosa.load('chunk-05.wav')
get_formants(x, Fs)


def get_pitch(x, sr):
    h = librosa.piptrack(x, sr = sr, fmin = 60, fmax= 1600, n_fft = 1024, center = False,threshold=0.9 )
    pitches_detected = h[0][:,1][h[0][:,1]>0]
    average_pitch = np.nanmean(pitches_detected)
    return average_pitch




