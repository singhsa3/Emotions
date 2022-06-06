#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:59:00 2022

@author: sanjeev
"""



import librosa, librosa.display
import matplotlib.pyplot as plt

import os
import glob


import cv2

DATASET_PATH = "../data/voice_samples"


filenames= glob.glob(DATASET_PATH+"/*.wav" )
filenames = [os.path.basename(x) for x in filenames]


from joblib import Parallel, delayed

def process(recrd):    
    file = DATASET_PATH +"/"+ recrd
    flo = recrd.split('.')[0]
    img = "../data/greyimg"
    loc = img +'/'+flo+'.png'        
    # load audio file with Librosa
    signal, sample_rate = librosa.load(file, sr=16000)    
   
    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=int(2048/2) ,hop_length=int(512/2))
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)    

    librosa.display.specshow(log_mel_spectrogram, sr=sample_rate)    
    plt.savefig(loc)   


    img = cv2.imread(loc)
    res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(loc,grayscale) 
 
   
    
results = Parallel(n_jobs=-1)(delayed(process)(recrd) for recrd in filenames)
