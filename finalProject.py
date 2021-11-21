# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:07:26 2021

@author: mbate
"""

datasetPath = "MDB Drums"
path = "MDB Drums/audio/full_mix/MusicDelta_Britpop_MIX.wav"

blockSize = 1024
hopSize = 256
sampleRate = 44100

import numpy as np
import math
import scipy as sp
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_prominences, butter, filtfilt
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)


def compute_spectrogram(xb, fs):
    numBlocks = xb.shape[0]
    afWindow = 0.5 - (0.5 * np.cos(2 * np.pi / xb.shape[1] * np.arange(xb.shape[1])))
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    f_min = 0
    f_max = fs/2
    f = np.linspace(f_min, f_max, xb.shape[0]+2)
    fInHz = f[1:xb.shape[0]+1]
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        # normalize
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 

    return X, fInHz

def extract_spectral_flux(x, blockSize, hopSize, fs):
    
    xb, t = block_audio(x, blockSize, hopSize, fs)
    
    numBlocks = xb.shape[0]
    afWindow = 0.5 - (0.5 * np.cos(2 * np.pi / xb.shape[1] * np.arange(xb.shape[1])))
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    sFlux = np.zeros(numBlocks)
    
    for n in range(0, numBlocks):
        
        # apply window
        tmp = abs(fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        
        if n == 0:
            sFlux[n] = np.sqrt(np.sum(X[:, n]**2)) / X.shape[0]
        else:
            diff = X[:, n] - X[:, n-1]
            hwr_mask = np.argwhere(diff < 0)
            diff[hwr_mask] = 0
            sFlux[n] = np.sqrt(np.sum(diff**2)) / X.shape[0]
            
    return sFlux 

def draw_spec_flux(x, sFlux, t, fs):
    
    x =  np.max(sFlux) * x / np.max(x)
    time = np.arange(0, len(x)) * (1/fs)
    fig, ax = plt.subplots()
    ax.plot(time, x, color = 'k')
    ax.plot(t, sFlux, color = 'r', marker = '.')
    
# def create_training_dataSet(datasetPath):
     
#     for root, dirs, filenames in os.walk(datasetPath):
#         for directory in dirs:
#             if directory == "class":
#                 classPath = os.path.join(root, directory)
#             elif directory == "beats":
#                 beatsPath = os.path.join(root, directory)

#     classMapping = []                
#     for root, dirs, filenames in os.walk(classPath):
#         numFiles = len(filenames)
#         training_segment_ndx = np.floor((3/4) * numFiles)
#         trainingSegment = filenames[0:training_segment_ndx]
#         for file in trainingSegment:
#             classMapping = classMapping.append(os.path.join(root, file))
    
#     beatsMapping = []
#     for root, dirs, filenames in os.walk(beatsPath):
#         numFiles = len(filenames)
#         training_segment_ndx = np.floor((3/4) * numFiles)
#         trainingSegment = filenames[0:training_segment_ndx]
#         for file in trainingSegment:
#             beatsMapping = beatsMapping.append(os.path.join(root, file))
        
#     # Determines the average inter beat interval length in seconds for each song
#     fileNumber = 1
#     beatLength = np.ones(len(beatsMapping))
#     for i in range(len(beatsMapping)):
#         print("---------Evaluating file", fileNumber, "-----------")
#         beats = np.readtxt(beatsMapping[i])[0]
#         print(beatsMapping[i], ": success read!")
#         ibi = np.mean(beats[1:] - beats[0:-2])
#         beatLength[i] = ibi
#         fileNumber = fileNumber + 1
        
#     # Reads class annotations for each song and outputs modified class vectors
#     # Each vector corresponds to a class (snare, kick, etc.) and only includes
#     # instances of the given instrument that occur with no other occurences within
#     # +- 1/2 beatLength of that instance    
#     for classPath in classMapping:
        
#         timeStamp = np.readtxt(txtPath)[:, 0]
#         drumClass = np.readtxt(txtPath)[:, 1]
        
        
         

      
                
                
                
                
                
     