# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:58:20 2021

@author: Francisco
"""

import librosa
import numpy as np
import sys
import numpy
import wave
import math
from scipy.signal import lfilter, hamming
# from scikits.talkbox import lpc

svi, sr = librosa.load('chunk-00.wav')
svi, sr = librosa.load('AR31_021108a.wav')
hop_length = int(10*0.001*sr)
frame_length = int(64*0.001*sr)
frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0 )
count = 0
for i, frame in enumerate(frames):
    h = librosa.piptrack(frame,sr = sr, fmin = 60, fmax= 1600, n_fft = 1024, center = False,threshold=0.9 )
    count+=sum(h[1][:,1])>0
    if sum(h[1][:,1])==0:
        xx = frame
    if sum(h[1][:,1])>0:
        yy = frame
    
# for i, frame in enumerate(frames):
#     xx = frame

test = frames[:,1]
h = librosa.piptrack(test,sr = sr, fmin = 60, fmax= 1600, n_fft = 1024, center = False, frame_length=frame+_ )
mag = h[1]

def get_pitch(x, sr):
    h = librosa.piptrack(x, sr = sr, fmin = 60, fmax= 1600, n_fft = 1024, center = False,threshold=0.9 )
    # pitches_detected = h[0][:,1][h[0][:,1]>0]
    pitches_detected = h[0][h[0]>0]
    average_pitch = np.nanmean(pitches_detected)
    return average_pitch

def VC(svi, sr):
    is_VC = 0
    hop_length = int(10*0.001*sr)
    frame_length = int(64*0.001*sr)
    frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0 )
    count = 0
    pitches = []
    for i, frame in enumerate(frames):
        avg_pitch = get_pitch(frame, sr)
        pitches.append(avg_pitch)
        count+=avg_pitch>0
    if count/len(frames)>0.6:
        is_VC = 1
    return [is_VC, np.nanmean(pitches)]
        
    
pitches = []
count= 0
for i, frame in enumerate(frames):
    avg_pitch = get_pitch(frame, sr)
    pitches.append(avg_pitch)
    count+=avg_pitch>0
    if sum(h[1][:,1])==0:
        xx = frame
    if sum(h[1][:,1])>0:
        yy = frame