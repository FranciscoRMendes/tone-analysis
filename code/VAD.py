# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:20:38 2021

@author: Francisco
"""

import librosa
import soundfile
import collections
import contextlib
import sys
import wave
import webrtcvad
import os


# import SVI

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def read_audio_vad(file_path):
    audio, rate = librosa.load(file_path, sr=16000, mono=True)
    tmp_file = 'tmp.wav'
    sf.write(tmp_file, audio, rate, subtype='PCM_16')
    audio, rate = read_pcm16_wave(tmp_file)
    remove(tmp_file)
    return audio, rate


def read_pcm16_wave(file_path):
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def read_audio(file_path, sample_rate=None, mono=False):
    return librosa.load(file_path, sr=sample_rate, mono=mono)


def webrtc_voice(audio, rate):
    voiced_frames = webrtc_split(audio, rate)
    tmp_file = 'tmp.wav'
    for frames in voiced_frames:
        voice_audio = b''.join([f.bytes for f in frames])
        write_pcm16_wave(tmp_file, voice_audio, rate)
        voice_audio, rate = read_audio(tmp_file)
        os.remove(tmp_file)

        start_time = frames[0].timestamp
        end_time = (frames[-1].timestamp + frames[-1].duration)
        start_frame = int(round(start_time * rate / 1e3))
        end_frame = int(round(end_time * rate / 1e3))
        yield voice_audio, rate, start_frame, end_frame


def write_pcm16_wave(path, audio, sample_rate):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def webrtc_split(audio, rate, aggressiveness=3, frame_duration_ms=30, padding_duration_ms=300):
    vad = webrtcvad.Vad(aggressiveness)

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in generate_frames(audio, rate):
        is_speech = vad.is_speech(frame.bytes, rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield voiced_frames


class Frame(object):
    """
    object holding the audio signal of a fixed time interval (30ms) inside a long audio signal
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def generate_frames(audio, sample_rate, frame_duration_ms=30):
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(frame_length) / sample_rate)
    while offset + frame_length < len(audio):
        yield Frame(audio[offset:offset + frame_length], timestamp, duration)
        timestamp += duration
        offset += frame_length
