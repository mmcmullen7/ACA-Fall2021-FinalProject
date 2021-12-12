import pickle
import numpy as np
import math
import os, fnmatch
import librosa.display, librosa

from scipy.io import wavfile
import scipy as sp
from scipy.fftpack import fft
from scipy.signal import find_peaks, peak_prominences, butter, filtfilt

import matplotlib.pyplot as plt 

from sklearn import preprocessing

### Get Audio Files ###
def getFile(path):
    files = []
    for root, dirname, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".wav"):
                files.append(os.path.join(root, file))
    print("found %d audio files"%(len(files)))
    return files

### MFCC ###
def get_MFCC(y, sr=44100, blockSize=2048, hopSize=512, numMel=128, numMFCC=13):
    # Use a pre-computed log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=blockSize, hop_length=hopSize, n_mels=numMel)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=numMFCC)

    # Take the mean of each row
    feature_vec = np.mean(mfcc,1)
    return feature_vec

### Read File And Calculate MFCC Feature ###
def extractMFCC(files, numMFCC=13, fs=44100):
    feature_vec = np.zeros([len(files),numMFCC])
    for ind, file in enumerate(files):
        audio, sr = librosa.load(file, sr=fs)
        audio = audio/audio.max() # normalize audio file
        MFCC_feature = get_MFCC(audio)

        feature_vec[ind] = MFCC_feature
        # feature_vec_test.append(MFCC_feature) # Test

    # # Standardization to Zero-mean & Unit Variance
    # scaler = preprocessing.StandardScaler()
    # feature_vec_scaled = scaler.fit_transform(feature_vec)

    return feature_vec

### Load in the model ###
modelfilePath = 'ACA Final Project/knn_model.sav'
loaded_model = pickle.load(open(modelfilePath, 'rb'))



def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavfile.read(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    return (samplerate, audio)

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

def draw_spec_flux(x, sFlux, t, fs, threshold, result, onsetTimes, onsets):
    
    x =  np.max(sFlux) * x / np.max(x)
    time = np.arange(0, len(x)) * (1/fs)
    fig, ax = plt.subplots()
    ax.plot(time, x, color = 'k')
    ax.plot(t, sFlux, color = 'r', marker = '.')
    ax.plot(t, threshold, color = 'g', marker = '.')
    ax.plot(t, result, color = 'c', marker = '.')
    ax.plot(onsetTimes,onsets, 'o',   color = 'm', marker = '.')
    
def sFlux_smoother(sFlux, f):
    
    b, a = butter(1, f, 'lowpass')
    sFlux_smoothed = filtfilt(b, a, sFlux)
    
    return sFlux_smoothed
    
def movAvg(sFlux, windowSize):
    
    avgVec = np.zeros(sFlux.shape[0])
    beg_zero_pad = np.zeros(windowSize)
    end_zero_pad = np.zeros(1)
    for i in np.arange(0, len(sFlux)):
        
        if i < windowSize:
            avgVec[i] = np.mean(np.concatenate((beg_zero_pad, sFlux[0: i + windowSize])))
            beg_zero_pad = np.delete(beg_zero_pad, 0)
        elif i > avgVec.shape[0] - windowSize:
            avgVec[i] = np.mean(np.concatenate((sFlux[i:sFlux.shape[0]], end_zero_pad)))
            end_zero_pad = np.append(end_zero_pad, 0)
        else:
            avgVec[i] = np.mean(sFlux[i-windowSize:i+windowSize])
            
    
    return avgVec

def peakPicker(sFlux, sFlux_smoothed, avgVec, t):
    
    difference = sFlux_smoothed - avgVec
    difference[difference < 0] = 0
    lam = np.mean(difference) * 3/4
    threshold = avgVec + lam
    
    result = sFlux_smoothed - threshold
    result[result < 0] = 0
    
    peakNDXs = np.empty(0)
    for i in np.arange(0, result.shape[0] - 1):
        
        if i == 0:
            if result[i] > result[i + 1]:
                peakNDXs = np.append(peakNDXs, [i])
        elif (result[i] > result[i - 1]) and (result[i] > result[i + 1]):
                peakNDXs = np.append(peakNDXs, [i])
   
    onsetTimes = np.empty(0)
    onsets = onsetTimes
    for i in np.arange(0, peakNDXs.shape[0]):
        
        iNDX = np.uintp(peakNDXs[i])
        onsetTimes = np.append(onsetTimes, t[iNDX])
        onsets = np.append(onsets, sFlux[iNDX])
    
    return threshold, result, onsetTimes, onsets

def onset_detector(path, blockSize, hopSize, avgWindow, f_over_fn):
    
    # fs, x = wavfile.read(path)
    # if len(np.shape(x)) > 1:
    #     x = 0.5 * (x[:, 0] + x[:, 1])
    fs, x = ToolReadAudio(path)
        
    xb, t = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    sFlux = extract_spectral_flux(x, blockSize, hopSize, fs)
    sFlux_smoothed = sFlux_smoother(sFlux, f_over_fn)
    avgVec = movAvg(sFlux, avgWindow)
    threshold, result, onsetTimes, onsets = peakPicker(sFlux, sFlux_smoothed, avgVec, t)
    # draw_spec_flux(x, sFlux, t, fs, threshold, result, onsetTimes, onsets)
    
    return onsetTimes, onsets


# path = "ACA Final Project/MDBDrums-master/MDB Drums/audio/drum_only/MusicDelta_80sRock_Drum.wav"
path = "ACA Final Project/Test_Oneshots/TestDrumLoop(Easy_Drum2).wav" # input a drum loop file here
blockSize = 1024
hopSize = 256
sampleRate = 44100

onsetTimes, onsets = onset_detector(path, blockSize, hopSize, 15, .125)
print('Onset Time Calculated: \n', onsetTimes)
# print(onsets)
 
fs, audio = ToolReadAudio(path)
# print(fs)
# print(audio.size)

audioframe = np.diff(onsetTimes)
# print(audioframe.shape)
predicted_instrument = []
for i in range(onsetTimes.shape[0]-1):
    first_frame = audio[int(onsetTimes[i]*44100) : int(onsetTimes[i]*44100+audioframe[i]*44100)]
    MFCC_featureVec = get_MFCC(first_frame)
    MFCC_featureVec = MFCC_featureVec.reshape(1,-1)
    # print(MFCC_featureVec)
    predicted_labels = loaded_model.predict(MFCC_featureVec)
    # print(predicted_labels)
    if predicted_labels == 0 or predicted_labels == 3 or predicted_labels == 5:
        predicted_instrument.append('Other')
    if predicted_labels == 1:
        predicted_instrument.append('HiHat')
    if predicted_labels == 2:
        predicted_instrument.append('Kick')
    if predicted_labels == 4:
        predicted_instrument.append('Snare')
print('Predicted:', predicted_instrument)

