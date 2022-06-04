# -*- coding: utf-8 -*-
"""
Spyder Editor

wave3vecfeatures.py
This program does the following:
    1. Extract features from each audio files for each transformer layer in wave2 vec2
    2. Please note the pattern is same as that librosa feature extraction
    
"""

#import fairseq
import torch
#from fairseq.models.wav2vec import Wav2Vec2Model,Wav2VecModel
import librosa
import numpy as np
import os
#import IPython
import matplotlib
import matplotlib.pyplot as plt
#import requests

import torchaudio
#from torchvision import datasets, transforms, models
#import librosa, librosa.display
#import matplotlib.pyplot as plt
import pickle
#import os
import glob
import numpy as np

device = torch.device("cpu")


#recrd = "139_39117_1587600000.wav"
DATASET_PATH = "../data/normalized_samples"
#DATASET_PATH = "../data/voice_samples"

pkl= "../data/w2v2_pkl"

filenames= glob.glob(DATASET_PATH+"/*.wav" )
filenames = [os.path.basename(x) for x in filenames]


from joblib import Parallel, delayed

def process(recrd):    
    file = DATASET_PATH +"/"+ recrd
    flo = recrd.split('.')[0]
    bundle = torchaudio.pipelines.WAV2VEC2_LARGE
    model = bundle.get_model().to(device)
    print("model downloaded")        
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.to(device)
    #print(recrd)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)        
    with open(pkl+'/'+flo+'.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
results = Parallel(n_jobs=-1)(delayed(process)(recrd) for recrd in filenames)