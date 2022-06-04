#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:41:08 2022

@author: sanjeev

code: data_prep.py
This the first step in the process. We are doing the following:
    1. Converting all m4a files to wav
    2. replacing m4a to wav in the labeled json data
    3. Capturing duration of each sound file
    4. We made all the audio files of equal length by adding segments in the front and back of each soundfile.
    We did this intially by adding Whitnoise but later changed to silence. Final length of each file is 106 sec.

"""


import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "../data/voice_samples"
JSON_PATH = "../data/voice_labels.json"

import os
import argparse

from pydub import AudioSegment

# Format conversion
formats_to_convert = ['.m4a']

for (dirpath, dirnames, filenames) in os.walk(DATASET_PATH):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):

            filepath = dirpath + '/' + filename
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath,
                        file_extension_final)
                wav_filename = filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                print("ERROR CONVERTING " + str(filepath))




import wave

import contextlib
# fname = '/tmp/test.wav'
def dur(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

f=open(JSON_PATH)
fj = f.read()  
fj= fj.replace('m4a','wav') 
j = json.loads(fj)

files = list(j.keys())
therapists= list(j[files[0]].keys())
emotions= list(j[files[0]][therapists[0]].keys())


dursec={}
fileerror=[]
for fname in files:
    fname2=DATASET_PATH +"/"+fname
    try:
        dursec[fname]=dur(fname2)
    except:
        fileerror.append(fname)
        
import pickle

with open('duration.pickle', 'wb') as handle:
    pickle.dump(dursec, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
DATASET_PATH = "../data/voice_samples"
NEW_DATASET_PATH =  "../data/normalized_samples" 
maxdur = int(np.percentile(list(dursec.values()),95)) +5  # In secs

# Add white noise and or silence sound to make it uniform length
 
for recrd in files:
    file = DATASET_PATH +"/"+ recrd
    from os.path import exists
    file_exists = exists(file)
    if file_exists == True:
        #print(dur(file))
        if dur(file)<106:       
            
            filedur= dursec[recrd]
            from pydub import AudioSegment
            from pydub.playback import play
            from pydub.generators import WhiteNoise        
            
            
            # create 1 sec of silence audio segment
            #front = WhiteNoise().to_audio_segment(duration=1000) #duration in milliseconds
            #back = WhiteNoise().to_audio_segment(duration=(maxdur-filedur)*1000)  #duration in millisecon
            front = AudioSegment.silent(duration=1000)
            back = AudioSegment.silent(duration=(maxdur-filedur)*1000)  #duration in millisecon
            
            #read wav file to an audio segment
            clip = AudioSegment.from_wav(file)
            
            #Add above two audio segments    
            final_clip = front+clip+back
            #final_clip = clip
            
            #Either save modified audio
            fmn=NEW_DATASET_PATH +"/"+ recrd    
            final_clip.export(fmn, format="wav")
        
        else:
            print(recrd +" skipped")
            test=0
        test= dur(fmn)
        print(test)

        

