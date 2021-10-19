# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:07:26 2021

@author: mbate
"""

datasetPath = "MDB Drums"
blockSize = 1024
hopSize = 256
sampleRate = 44100

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

def block_audio(x,blockSize,hopSize,fs):
    
    inLen = len(x)
    nBlock = int(np.ceil((inLen-blockSize)/hopSize)+1)

    xb = np.zeros((nBlock,blockSize))
    timeInSample = np.arange(0, hopSize*nBlock, hopSize)
    timeInSec = timeInSample/fs
                  
    for i in range(len(timeInSec)):
        if i == len(timeInSec)-1:
            zeroPad = blockSize - len(x[int(timeInSample[i]):])
            xb[i] = np.pad(x[int(timeInSample[i]):], (0,zeroPad))
        else:
            xb[i] = x[int(timeInSample[i]):int(timeInSample[i]+blockSize)]
            
    return [xb, timeInSec]

 def create_training_dataSet(path, blockSize, hopSize):
     
     mapping = []
     for root, dirs, filenames in os.walk(path_to_musicspeech):
        for directory in dirs:
            if directory == "class":
                classPath = os.path.join(root, directory)
            elif directory == "beats":
                beatsPath = os.path.join(root, directory)
                
    for root, dirs, filenames in os.walk(classPath);
        numFiles = len(filenames)
        training_segment_ndx = np.floor((3/4) * numFiles)
        classMapping = filenames[0:training_segment_ndx]
    
    for root, dirs, filenames in os.walk(beatsPath);
        numFiles = len(filenames)
        training_segment_ndx = np.floor((3/4) * numFiles)
        beatsMapping = filenames[0:training_segment_ndx]
    
    fileNumber = 1
    
    for beatPath in beatsMapping
    for classPath in classMapping:
        
        timeStamp = np.readtxt(txtPath)[:, 0]
        drumClass = np.readtxt(txtPath)[:, 1]
        
        
         

      
                
                
                
                
                
     