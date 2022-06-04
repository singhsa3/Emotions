#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:54:54 2022

@author: sanjeev
features.py:
This file does the following
1. Consolidate the multiple feature pickle files (We did it to process in parallel) into one feature file
2. Create labeled dataset by each therapist and emotion. For example: Yared Alemu_fear.csv
"""

import pickle
import os
import glob
import numpy as np
import json
import pandas as pd


# Consolidating features
pkl = "../data/pkl"
labels="../data/labels"


filenames= glob.glob(pkl+"/*.pickle" )
filenames = [os.path.basename(x) for x in filenames]
features={}
for recrd in filenames:
    #print (recrd)
    flo = recrd.split('.')[0] 
    with open(pkl+'/'+flo+'.pickle', 'rb') as handle:
        features[flo+'.wav'] = pickle.load(handle)

with open('../data/featuresdict.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
fnames=list(features.keys())
with open('../data/filenameslist.pickle', 'wb') as handle:
    pickle.dump(fnames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
 
# Creating labels    
JSON_PATH = "../data/voice_labels.json"
f=open(JSON_PATH)
fj = f.read()  
fj= fj.replace('m4a','wav') 
j = json.loads(fj)

files = list(j.keys())
therapists= list(j[files[0]].keys())
emotions= list(j[files[0]][therapists[0]].keys())

lst=[]
for therapist in therapists:
    for file in files:
        for emotion in emotions:
            try:
                lst.append([therapist,file,emotion, j[file][therapist][emotion]])
            except:
                pass
            
lbl =pd.DataFrame(lst)
lbl.columns=['therapist','name', 'emotion_type','rating']
lbl.to_csv(labels+"/"+"labelsConsolidated.csv")

# Balancing data and creating label set for each emotion and therapist
import pandas as pd

lbl['emotion'] = lbl.rating.apply(lambda x: 1 if x.lower()=="high" else 0)


def data_balanced (therapist, emotion_type, lbl):
    Alemu = lbl[(lbl.therapist==therapist) & (lbl.emotion_type==emotion_type)]    
    emt= Alemu[Alemu.emotion==1]
    emtN=Alemu[Alemu.emotion==0]
    sz = emt.shape[0]
    Alemu=pd.concat([emt, emtN.sample(sz)])
    return Alemu.sample(frac=1)

for therapist in therapists:
    for emotion in emotions:
        df = data_balanced(therapist, emotion, lbl)
        df.to_csv(labels+"/"+therapist+'_'+emotion+'.csv')
        