# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:20:38 2021

@author: Francisco
"""

import os
os.chdir('C:/Users/Francisco/Documents/tone-analysis/code')
from clip import Clip
from SVI import *
from VAD import *
import pandas as pd
import librosa.output

import librosa

m = Clip('../dat/bank/AR31_021108a.wav')

feature_df = pd.DataFrame(m.feature_list)

 s = feature_df.sum()

# comments:
# 34 should have a squeal
# 18 is probably also a squeal
# 66 is not a growl but an adult speaking
# really need to remove adult voices

sq = feature_df['svi'][18]
soundfile.write('squeal.wav',sq,16000)
sq = feature_df['svi'][66]
soundfile.write('growl.wav',sq,16000)
sq = feature_df['svi'][34]
soundfile.write('other.wav',sq,16000)

# audio, sr = read_wave('AR31_021108a.wav')
# k = webrtc_voice(audio, sr)
# svi_list = []
# for i,j in enumerate(k):
#     m = j
#     svi_list.append(SVI(m[0],sr))
    
# h = SVI(m[0],sr)

