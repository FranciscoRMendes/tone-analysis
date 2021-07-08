# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:25:47 2021

@author: Francisco
"""
from VAD import *
from SVI import *

class Clip(object):
    def __init__(self, path):
        self.audio, self.sr = read_wave(path)
        k = webrtc_voice(self.audio, self.sr)
        svi_list = []
        feature_list = []
        for i, j in enumerate(k):
            o = SVI(j[0], self.sr)
            svi_list.append(o)
            feature_list.append(o.__dict__)
        self.svi_list = svi_list
        self.feature_list = feature_list
        
