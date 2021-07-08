# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:14:13 2021

@author: Francisco
"""
import antropy as ant
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import lfilter, hamming

def get_pitch(x, sr):
    h = librosa.piptrack(x, sr = sr, fmin = 60, fmax= 1600, n_fft = 1024, center = False,threshold=0.9 )
    pitches_detected = h[0][:,1][h[0][:,1]>0]
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

def get_formants(x, Fs):
    N = len(x)
    w = np.hamming(N)

    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Fs = spf.getframerate()
    ncoeff = 2 + Fs / 1000
    A = librosa.lpc(x1, int(ncoeff))
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * (Fs / (2 * math.pi))
    rts = np.array(rts)
    bw = -1/2*(Fs/(2*np.pi))*np.log(abs(rts))

    frqs = [round(frqs[i],2) for i in range(0,len(frqs)) if bw[i]<400 and frqs[i]>90]
    frqs = sorted(frqs)
    return(frqs[1:3])

def SQ(svi, sr):
    if get_pitch(svi, sr)>600:
        return 1
    else:
        return 0

def LT(svi, sr):
    hop_length = int(10*0.001*sr)
    frame_length = int(64*0.001*sr)
    frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)
    count = 0
    d = []
    for frame in frames:
        y = frame
        S = np.abs(librosa.stft(y,n_fft = 512))
        freq_list = np.array(range(0,int((512/2)+1)))*sr/512
        power = librosa.power_to_db(S**2,ref=1)
        # power  = librosa.amplitude_to_db(S,ref=1)
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

    if count>0.5*len(frames):
        return True
    return False


def HF(svi, sr):
    hop_length = int(10*0.001*sr)
    frame_length = int(64*0.001*sr)
    frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)
    # f, Pxx_spec = signal.periodogram(svi, sr, 'flattop', scaling='spectrum')
    # peaks, _ = scipy.signal.find_peaks(Pxx_spec,prominence=np.percentile(Pxx_spec,99))
    # lowest_peak = f[peaks[0]]
    # if lowest_peak > 1500:
    #     count+=1
    # plt.plot(f,Pxx_spec)
    # plt.plot(f[peaks],Pxx_spec[peaks],'x')
    count = 0
    for frame in frames:
        f, Pxx_spec = signal.periodogram(frame, sr, 'flattop', scaling='spectrum')
        peaks, _ = scipy.signal.find_peaks(Pxx_spec,prominence=np.percentile(Pxx_spec,99))
        lowest_peak = f[peaks[0]]
        if lowest_peak > 1500:
            count+=1
        plt.plot(f,Pxx_spec)
        plt.plot(f[peaks],Pxx_spec[peaks],'x')
        if count/len(frames)>0.5 :
            return True
        return False

def SE(svi, sr, threshold):
    if ant.spectral_entropy(svi, sr)>threshold:
        return True
    return False

def GW(svi, sr):
    if VC(svi, sr)[1]<250:
        return True
    return False


def S(svi, sr):
    if (len(svi) >= int(110*0.001*sr)) & (len(svi)<=int(250*0.001*sr)):
        return True
    return False

def M(svi, sr):
    if len(svi) > int(250*0.001*sr) and len(svi)<=int(600*0.001*sr):
        return True
    return False

def L(svi, sr):
    if len(svi) >int(600*0.001*sr) and len(svi)<=int(900*0.001*sr):
        return True
    return False

def XL(svi, sr):
    if len(svi) >int(900*0.001*sr) and len(svi)<=int(3000*0.001*sr):
        return True
    return False

def dur(svi, sr):
    size = np.array(["S","M","L","XL"])
    check = np.array([S(svi,sr), M(svi,sr),L(svi,sr),XL(svi,sr)])
    return size[check][0]


def WB(x, Fs):
    N = len(x)
    w = np.hamming(N)

    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Fs = spf.getframerate()
    ncoeff = 2 + Fs / 1000
    A = librosa.lpc(x1, int(ncoeff))
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * (Fs / (2 * math.pi))
    rts = np.array(rts)
    bw = -1/2*(Fs/(2*np.pi))*np.log(abs(rts))

    frqs = [round(frqs[i],2) for i in range(0,len(frqs))]
    frqs = sorted(frqs)
    bw = [x for _,x in sorted(zip(frqs,bw))]
    F1,F2 = frqs[1], frqs[2]
    bwF1, bwF2 = bw[1], bw[2]
    print(bwF1)
    print(bwF2)
    if bwF1>400 and bwF2>600:
        return True
    return False


def float_to_pcm16(audio):
    ints = (audio * 32767).astype(np.int16)
    little_endian = ints.astype('<u2')
    buf = little_endian.tobytes()
    return buf
