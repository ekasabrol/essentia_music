#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:10 2021

@author: sanjamekas
"""
import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'essentia'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'essentia-tensorflow'])


import numpy as np
import pandas as pd
from essentia.standard import *
import re 
import os


songs = os.listdir('/content/drive/MyDrive/clips')
songs = songs


def feature_extraction(songs):
  column = ['SongName','Genre','BPM','gender','danceable',
            'speechiness',  'tonal', 'mood_acoustic',
            'mood_aggressive','mood_electronic', 'mood_happy',
            'mood_party', 'mood_relaxed', 'mood_sad']

  #Data Frame Initialization; we pass no data, but label column features
  df = pd.DataFrame(columns = column) 

  #running a for loop for each song
  for i in range(0,len(songs)):
    music = "/content/drive/MyDrive/clips/{}".format(songs[i])

    #Extract Name of Song
    sr = 16000
    pattern = re.compile(r'(?<=clips[/]).*')
    name = pattern.findall(music)


    #Get Genre Using msd-musicnn-1.pb
    loader = essentia.standard.MonoLoader(filename = music, sampleRate = sr) #audio sampled at 16000Hz cuz 42000Hz cannot be used
    audio = loader()
    genre_labels = ['rock','pop','alternative','indie','electronic','female vocalists','dance','00s','alternative rock','jazz','beautiful',
                  'metal','chillout','male vocalists','classic rock','soul','indie rock','Mellow','electronica','80s','folk','90s','chill',
                  'instrumental','punk','oldies','blues','hard rock','ambient','acoustic','experimental','female vocalist','guitar','Hip-Hop',
                  '70s','party','country','easy listening','sexy','catchy','funk','electro','heavy metal','Progressive rock','60s','rnb',
                  'indie pop','sad','House','happy']
    predict = TensorflowPredictMusiCNN(graphFilename='/content/essentia_music/packages/msd-musicnn-1.pb')(audio)
    predictions = np.mean(predict, axis=0)
    l = predictions.argsort()[-1] #most weighted label is last one
    genre = genre_labels[l]
    
    #bpm
    loader1 = essentia.standard.MonoLoader(filename = music) #since bpm is an inbuilt method; audio is to be sampled at 42000Hz
    audio1 = loader1()
    getRhythmExtractor = RhythmExtractor2013()
    (bpm,ticks,confidence,estimates,bpmIntervals) = getRhythmExtractor(audio1)
    bpm = [round(bpm,2)]

    #female/male - we do binomial classification by rounding off to 1 0 for female; 1 for male
    predict = TensorflowPredictMusiCNN(graphFilename='/content/essentia_music/packages/gender-musicnn-msd-2.pb')(audio)
    gend = np.mean(predict, axis=0)
    female = round(gend[0],2) 
    if female>0.5:
          gender = 0  #female
    else:
          gender = 1 #male

    #danceable/not danceable
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/danceability-musicnn-msd-2.pb')(audio)
    dance = np.mean(predict, axis=0)
    danceable = round(dance[0],2)

    #vocal/instruments
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/voice_instrumental-musicnn-msd-2.pb')(audio)
    vocal = np.mean(predict, axis=0)
    speechiness = round(vocal[1],2)

    
    #tonal/atonal
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/tonal_atonal-musicnn-msd-2.pb')(audio)
    tones = np.mean(predict, axis=0)
    tonal = round(tones[0],2)

    #mood_acoustic
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_acoustic-musicnn-msd-2.pb')(audio)
    acoust = np.mean(predict, axis=0)
    acoustic = round(acoust[0],2)

    #mood_aggressive
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_aggressive-musicnn-msd-2.pb')(audio)
    agg = np.mean(predict, axis=0)
    aggressive = round(agg[0],2)

    #mood_electronic
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_electronic-musicnn-msd-2.pb')(audio)
    elect = np.mean(predict, axis=0)
    electronic = round(elect[0],2)

    #mood_happy
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_happy-musicnn-msd-2.pb')(audio)
    hap = np.mean(predict, axis=0)
    happy = round(hap[0],2)

    #mood_party
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_party-musicnn-msd-2.pb')(audio)
    part = np.mean(predict, axis=0)
    party = round(part[1],2)

    #mood_relaxed
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_relaxed-musicnn-msd-2.pb')(audio)
    relax = np.mean(predict, axis=0)
    relaxed = round(relax[1],2)

    #mood_sad
    predict = TensorflowPredictMusiCNN(graphFilename = '/content/essentia_music/packages/mood_sad-musicnn-msd-2.pb')(audio) 
    sadd = np.mean(predict, axis=0)
    sad = round(sadd[1],2)

    #making a list of lists of all features corresponding to our DataFrame labels

    features = [name,[genre], bpm, [gender], [danceable],
                [speechiness],[tonal],[acoustic],[aggressive],
                [electronic], [happy], [party], [relaxed], [sad]]
    features = [item for sublist in features for item in sublist]
    
    #pushing every df to end; by taking index size
    df.loc[len(df.index)] = features
    
  df.to_csv(r'/content/musictable.csv', encoding='utf-8', index = False)
  return  


feature_extraction(songs)


