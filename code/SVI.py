# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 07:14:34 2021

@author: Francisco
"""
import antropy as ant
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import lfilter, hamming
from scipy import signal
import scipy
import antropy as ant
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import lfilter, hamming


# DRIVER FUNCTIONS
def get_pitch(x, sr):
    h = librosa.piptrack(x, sr=sr, fmin=60, fmax=1600, n_fft=1024, center=False, threshold=0.9)
    # pitches_detected = h[0][:,1][h[0][:,1]>0]
    if np.any(h[0]):
        pitches_detected = h[0][h[0] > 0]
    else:
        pitches_detected = 0
    average_pitch = np.nanmean(pitches_detected)
    return average_pitch


def VC(svi, sr):
    """VC or voicing (i.e., voiced SVIs per SCU) can be portrayed as a measure of the degree to
    which SCUs were acoustically organized to yield clear boundaries between periods of high
    energy phonatory regularity (voicing) and periods lacking that high energy voicing. In practice
    the measure roughly captured a minimum number of voiced islands (very roughly syllables) per
    SCU. Voicing was determined by whether the autocorrelation method was able to assign a pitch
    during each frame. If at least 10 frames or 60% of all frames in an SVI were assigned a pitch
    with 90% confidence, the SVI was categorized as positive for VC, otherwise it was categorized
    as negative."""

    is_VC = False
    hop_length = int(10 * 0.001 * sr)
    frame_length = int(64 * 0.001 * sr)
    frames = librosa.util.frame(svi, frame_length=frame_length, hop_length=hop_length, axis=0)
    count = 0
    pitches = []
    for i, frame in enumerate(frames):
        avg_pitch = get_pitch(frame, sr)
        pitches.append(avg_pitch)
        count += avg_pitch > 0
    if count / len(frames) > 0.6:
        is_VC = True
    return [is_VC, np.nanmean(pitches)]


def SE(svi, sr, threshold):
    """The classification SE was applied to islands with spectral entropy of the FFT exceeding an
    empirically determined value representing a deviation from the pattern of variation that is
    associated with pure normal phonation in childhood. The threshold for classification was set
    low enough that islands showing the sort of spectral variability in entropy that occurs in
    utterances of typical speech were classified as SE."""
    if ant.spectral_entropy(svi, sr) > threshold:
        return True
    return False


def get_formants(x, Fs):
    """https://in.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html"""

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
    bw = -1 / 2 * (Fs / (2 * np.pi)) * np.log(abs(rts))

    frqs = [round(frqs[i], 2) for i in range(0, len(frqs)) if bw[i] < 400 and frqs[i] > 90]
    frqs = sorted(frqs)
    return (frqs[1:3])


def SQ(svi, sr):
    """SQ represented pitch substantially exceeding a maximum value expected for child voices in
    speech-like utterances, set nominally at a mean of 600 Hz for the island."""
    if get_pitch(svi, sr) > 600:
        return True
    else:
        return False


def LT(svi, sr):
    """LT was evidenced by high energy in the highest spectral band (7-8 kHz) relative to the energy
    at the maximum spectral peak from 0-6 kHz. If the highest band’s energy was within 30 dB of
    the maximum spectral peak from 0 through 6 kHz for 50% of the island’s frames, the island
    was classified as LT."""

    hop_length = int(10 * 0.001 * sr)
    frame_length = int(64 * 0.001 * sr)
    frames = librosa.util.frame(svi, frame_length=frame_length, hop_length=hop_length, axis=0)
    count = 0
    d = []
    for frame in frames:
        y = frame
        S = np.abs(librosa.stft(y, n_fft=512))
        freq_list = np.array(range(0, int((512 / 2) + 1))) * sr / 512
        power = librosa.power_to_db(S ** 2, ref=1)
        # power  = librosa.amplitude_to_db(S,ref=1)
        lower_freqs_power = power[(freq_list < 600), :].sum(axis=0)
        peaks, _ = scipy.signal.find_peaks(lower_freqs_power)
        if len(peaks) == 0:
            max_peak = max(lower_freqs_power)
        else:
            max_peak = np.max([lower_freqs_power[i] for i in peaks])
        higher_freqs_power = power[(freq_list < 8000) & (freq_list > 7000), :].sum(axis=0)
        LT_marker = max_peak - np.sum(higher_freqs_power) < 30
        d.append(max_peak - np.sum(higher_freqs_power))
        count += LT_marker

    if count > 0.5 * len(frames):
        return True
    return False


def HF(svi, sr):
    """HF required the first (lowest frequency) spectral peak to occur at above 1.5 kHz for 25% of the
    island’s frames."""

    hop_length = int(10 * 0.001 * sr)
    frame_length = int(64 * 0.001 * sr)
    frames = librosa.util.frame(svi, frame_length=frame_length, hop_length=hop_length, axis=0)
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
        peaks, _ = scipy.signal.find_peaks(Pxx_spec, prominence=np.percentile(Pxx_spec, 99))
        lowest_peak = f[peaks[0]]
        if lowest_peak > 1500:
            count += 1
        plt.plot(f, Pxx_spec)
        plt.plot(f[peaks], Pxx_spec[peaks], 'x')
        if count / len(frames) > 0.5:
            return True
        return False


def GW(svi, sr):
    """GW required pitch to be substantially below an expected minimum level for infant/child voices
    in speech-like utterances, viz., mean < 250 Hz for the island."""
    if VC(svi, sr)[1] < 250:
        return True
    return False


def S(svi, sr):
    if (len(svi) >= int(110 * 0.001 * sr)) & (len(svi) <= int(250 * 0.001 * sr)):
        return True
    return False


def M(svi, sr):
    if int(250 * 0.001 * sr) < len(svi) <= int(600 * 0.001 * sr):
        return True
    return False


def L(svi, sr):
    if int(600 * 0.001 * sr) < len(svi) <= int(900 * 0.001 * sr):
        return True
    return False


def XL(svi, sr):
    if int(900 * 0.001 * sr) < len(svi) <= int(3000 * 0.001 * sr):
        return True
    return False


def dur(svi, sr):
    """Redundant function can quickly assign duration quickly"""
    size = np.array(["S", "M", "L", "XL"])
    check = np.array([S(svi, sr), M(svi, sr), L(svi, sr), XL(svi, sr)])
    return size[check][0]


def WB(x, Fs):
    """WB required that bandwidths for the first two formants (determined by a 3 dB drop from peak
    amplitude) exceed a value empirically determined to correspond to typical bandwidths of
    vowel-like sounds produced with normal phonation by children (> 400 Hz for F1and > 600 Hz
    for F2).
    Broadly, this code is similar to the formant code but extracts bandwidths for both the formants
    Consider merging the two functions later"""

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
    bw = -1 / 2 * (Fs / (2 * np.pi)) * np.log(abs(rts))

    frqs = [round(frqs[i], 2) for i in range(0, len(frqs))]
    frqs = sorted(frqs)
    bw = [x for _, x in sorted(zip(frqs, bw))]
    F1, F2 = frqs[1], frqs[2]
    bwF1, bwF2 = bw[1], bw[2]
    # print(bwF1)
    # print(bwF2)
    if bwF1 > 400 and bwF2 > 600:
        return True
    return False


def float_to_pcm16(audio):
    """Useful function to convert float to PCM16, to use VAD"""
    ints = (audio * 32767).astype(np.int16)
    little_endian = ints.astype('<u2')
    buf = little_endian.tobytes()
    return buf


class SVI(object):
    """Create an SVI island object, the goal here is to take as input a segment of audio generated by 
    VAD and create an SVI object which has all the attributes mentioned in the paper"""

    def __init__(self, svi, sr):
        self.svi = svi
        self.sr = sr
        self.duration = len(svi)
        # The first group of parameters (the rhythmic/syllabification group) identified voicing events, canonical
        # syllables, and moderately high spectral entropy, typical of speech (VC, CS and SE).
        self.VC, self.mean_pitch = VC(svi, sr)
        self.CS = None
        self.SE = SE(svi, sr, 0.6)
        # The second group of parameters (the low spectral tilt and high pitch control group) was represented by
        # the number of islands per sequence showing squeal quality (a technical term in the infant vocalization
        # literature), low spectral tilt, or a first spectral peak at high frequency (SQ, LT, HF):
        self.SQ = SQ(svi, sr)
        self.LT = LT(svi, sr)
        self.HF = HF(svi, sr)
        # The third group (the wide formant bandwidth and low pitch control group) was represented by the
        # number of islands per sequence showing growl quality (again a technical term in the infant vocalization
        # literature) or high bandwidth of resonances (GW, WB):
        self.GW = GW(svi, sr)
        self.WB = WB(svi, sr)
        # The fourth group (the duration group) was represented by the number of islands per sequence showing
        # short, medium, long or extra long durations (S, M, L, XL):
        self.S = S(svi, sr)
        self.M = M(svi, sr)
        self.L = L(svi, sr)
        self.XL = XL(svi, sr)

    def __repr__(self):
        """Improve this later"""
        return repr(f"{self.sr} :: {self.duration}")

# os.chdir('C:/Users/Francisco/Documents/tone-analysis/dat/bank')
# audio, sr = librosa.load('AR31_021108a.wav',sr = None)
# m = SVI(svi, sr)

# vad = webrtcvad.Vad(3)
# hop_length = int(10*0.001*sr)
# frame_length = int(10*0.001*sr)
# frames = librosa.util.frame(svi, frame_length = frame_length,hop_length = hop_length,axis=0)
# voiced_frames = []
# for frame in frames:
#     pcm_dat = float_to_pcm16(frame)
#     if vad.is_speech(pcm_dat, sr):
#         voiced_frames.append(frame)

# voiced_frames = []
# for frame in range(0,len(frames)):
#     pcm_dat = float_to_pcm16(frame)
#     if vad.is_speech(pcm_dat, sr):
#         voiced_frames.append(frame)


# vad = webrtcvad.Vad(2)

# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))
