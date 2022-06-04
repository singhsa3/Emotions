#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:49:19 2022

@author: sanjeev
spectfiles.py
This program does the following:
    1. Mels from each audio files
    2. Runs parallel jobs to save Mels for each audio file in it own pick
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import pickle
import os
import glob
import numpy as np


FIG_SIZE = (15,10)

#recrd = "139_39117_1587600000.wav"
DATASET_PATH = "../data/normalized_samples"
#DATASET_PATH = "../data/voice_samples"
img =  "../data/images"
pkl= "../data/pkl"

filenames= glob.glob(DATASET_PATH+"/*.wav" )
filenames = [os.path.basename(x) for x in filenames]


from joblib import Parallel, delayed

def process(recrd):    
    file = DATASET_PATH +"/"+ recrd
    flo = recrd.split('.')[0]        
    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=16000)    
    #https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
    #https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
    
    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=int(2048/2) ,hop_length=int(512/2))
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    features=mel_spectrogram

    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate)

    plt.savefig(img+'/'+flo+'.png')
 
    with open(pkl+'/'+flo+'.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
results = Parallel(n_jobs=-1)(delayed(process)(recrd) for recrd in filenames)



    
