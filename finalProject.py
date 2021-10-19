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

def extract_spectral_flux(xb):
    
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = Spectrogram[:,:blockSize_]
    Spectrogram_1 = np.concatenate((np.zeros((1,blockSize_)),Spectrogram_),axis=0)
    
    SpecDiff = np.diff(Spectrogram_1, axis=0)
    SpecFluxVector = np.sqrt(np.sum(SpecDiff**2, axis=1))/(blockSize_/2)
            
    return SpecFluxVector 

def create_training_dataSet(datasetPath):
     
    for root, dirs, filenames in os.walk(datasetPath):
        for directory in dirs:
            if directory == "class":
                classPath = os.path.join(root, directory)
            elif directory == "beats":
                beatsPath = os.path.join(root, directory)

    classMapping = []                
    for root, dirs, filenames in os.walk(classPath):
        numFiles = len(filenames)
        training_segment_ndx = np.floor((3/4) * numFiles)
        trainingSegment = filenames[0:training_segment_ndx]
        for file in trainingSegment:
            classMapping = classMapping.append(os.path.join(root, file))
    
    beatsMapping = []
    for root, dirs, filenames in os.walk(beatsPath):
        numFiles = len(filenames)
        training_segment_ndx = np.floor((3/4) * numFiles)
        trainingSegment = filenames[0:training_segment_ndx]
        for file in trainingSegment:
            beatsMapping = beatsMapping.append(os.path.join(root, file))
        
    # Determines the average inter beat interval length in seconds for each song
    fileNumber = 1
    beatLength = np.ones(len(beatsMapping))
    for i in range(len(beatsMapping)):
        print("---------Evaluating file", fileNumber, "-----------")
        beats = np.readtxt(beatsMapping[i])[0]
        print(beatsMapping[i], ": success read!")
        ibi = np.mean(beats[1:] - beats[0:-2])
        beatLength[i] = ibi
        fileNumber = fileNumber + 1
        
    # Reads class annotations for each song and outputs modified class vectors
    # Each vector corresponds to a class (snare, kick, etc.) and only includes
    # instances of the given instrument that occur with no other occurences within
    # +- 1/2 beatLength of that instance    
    for classPath in classMapping:
        
        timeStamp = np.readtxt(txtPath)[:, 0]
        drumClass = np.readtxt(txtPath)[:, 1]
        
        
         

      
                
                
                
                
                
     