#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:10 2021

@author: sanjamekas
"""
import numpy as np
import pandas as pd
from essentia.standard import *
import re 
import os


songs = os.listdir('/content/drive/MyDrive/songs')
songs = songs


def feature_extraction(songs):
    column_name = ['Name', 'gender']
    df = pd.DataFrame(columns = column_name)
    
    
    for i in range(0, len(songs)):
        music = "/content/drive/MyDrive/songs/{}".format(songs[i])

        #Extract Name of Song
        pattern = re.compile(r'(?<=songs[/]).*')
        name = pattern.findall(music)
        
    
        sr = 16000
        loader = essentia.standard.MonoLoader(filename = music, sampleRate = sr) #audio sampled at 16000Hz cuz 42000Hz cannot be used
        audio = loader()
    
    
        predict = TensorflowPredictMusiCNN(graphFilename='/content/essentia_music/packages/gender-musicnn-msd-2.pb')(audio)
        gend = np.mean(predict, axis=0)
        female = round(gend[0],2) 
    
        if female>0.5:
          gender = 0  #female
        else:
          gender = 1  #male
    
        features = [name, [gender]]
        features = [item for sublist in features for item in sublist] #flattening the list
    
        #pushing every df to end; by taking index size
        df.loc[len(df.index)] = features
    
    df.to_csv(r'/content/musictable.csv', encoding='utf-8', index = False)
    return df


feature_extraction(songs)
