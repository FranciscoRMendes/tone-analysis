# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 06:01:06 2021

@author: Francisco
"""
import antropy as ant
import entropy as ent
from entropy import spectral_entropy
import os 
import librosa
import matplotlib.pyplot as plt

os.chdir('C:/Users/Francisco/Documents/tone-analysis/dat/bank')
svi, sr = librosa.load('chunk-00.wav')
hop_length = int(10*0.001*sr)
frame_length = int(64*0.001*sr)

frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)


ent.spectral_entropy(svi, sr)
ant.spectral_entropy(svi, sr, method='fft')

def SE(svi, sr, threshold):
    import antropy as ant
    if ant.spectral_entropy(svi, sr)>threshold:
        return True
    return False
