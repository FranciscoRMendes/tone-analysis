# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:40:08 2021

@author: Francisco
"""

from scipy import signal
from scipy.signal import find_peaks, peak_prominences
frame = frames[1]
f0,_,_ = librosa.pyin(frame, fmin=60,fmax = 8000)
f0 = np.nanmean(f0)
S = np.abs(librosa.stft(frame,n_fft = 512))
scipy.signal.periodogram(frame, sr)


f, Pxx_spec = signal.periodogram(frame, sr, 'flattop', scaling='spectrum')
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.ylim([1e-4, 1e1])
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()

peaks, _ = scipy.signal.find_peaks(Pxx_spec)
f[peaks[0]]

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