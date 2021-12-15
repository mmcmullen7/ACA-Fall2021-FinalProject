### To do ###
# find out how many onsets are missing in the onset function we wrote
# compare the matching onset time deviation (our calculation) with the ground truth 
# see if the instrument predicted at that onset time matches with the ground truth


import numpy as np
import math
import matplotlib.pyplot as plt 

import scipy as sp
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_prominences, butter, filtfilt
import librosa.display, librosa

import os
import pickle

import warnings
warnings.filterwarnings("ignore")


### ____________________________Onset Detection_____________________________________ ###
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

def filter_audio(audio):
    
    kfb, kfa = butter(4, 0.004535, 'lowpass')
    sfb, sfa = butter(4, [0.00907, 0.090703], 'bandpass')
    hfb, hfa = butter(4, 0.3628, 'highpass')
    kickAudio = filtfilt(kfb, kfa, audio)
    snareAudio = filtfilt(sfb, sfa, audio)
    hihatAudio = filtfilt(hfb, hfa, audio)
    
    return kickAudio, snareAudio, hihatAudio

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
    fInHz = np.arange(0, X.shape[0])*fs/(xb.shape[1])
    
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
    ax.plot(onsetTimes, onsets, color = 'm', marker = '.')

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
    draw_spec_flux(x, sFlux, t, fs, threshold, result, onsetTimes, onsets)
    
    return onsetTimes, onsets

# # input a drum loop file here
# path = "ACA Final Project/MDBDrums-master/MDB Drums/audio/drum_only/MusicDelta_80sRock_Drum.wav"
# blockSize = 1024
# hopSize = 256
# sampleRate = 44100
# onsetTimes, onsets = onset_detector(path, blockSize, hopSize, 15, 0.125)
# print('Onset Time Calculated: \n', onsetTimes)



### ____________________________Classifier Model_____________________________________ ###
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
modelfilePath = 'knn_model.sav'
loaded_model = pickle.load(open(modelfilePath, 'rb'))



### ____________________________Evaluation_____________________________________ ###
def evaluation(annotation_filepath,audio_filepath):
    f = open(annotation_filepath, "r")
    lines = f.readlines()
    extractOnsets = []
    extractInstruments = []
    dataset_dict = {}
    for line in lines:
        extractOnsets.append(float(line.split()[0]))
        dataset_dict[float(line.split()[0])]=[]
    for line in lines:
        extractInstruments.append(line.split()[1])
        dataset_dict[float(line.split()[0])].append(line.split()[1])
    remove_repeat_Onsets = list(dict.fromkeys(extractOnsets))

    # Calculate Onset
    blockSize = 1024
    hopSize = 256
    sampleRate = 44100
    onsetTimes, onsets = onset_detector(audio_filepath, blockSize, hopSize, 15, 0.125)
    # print(onsetTimes)
    # print(len(onsetTimes))
    compare_onset(remove_repeat_Onsets, onsetTimes)
    compare_instrument(audio_filepath,remove_repeat_Onsets,dataset_dict)

### Compare onset prediction with dataset ###
def compare_onset(remove_repeat_Onsets, onsetTimes):
    threshold = 0.05 # Threshold of +- 0.05, if within the range, the predicted onset is set to detected
    predicted_onsets_dataset = []
    predicted_onsets = []
    for i in range(len(remove_repeat_Onsets)):
        for j in range(len(onsetTimes)):
            if remove_repeat_Onsets[i]-threshold < onsetTimes[j] < remove_repeat_Onsets[i]+threshold:
                # print(remove_repeat_Onsets[i], onsetTimes[j])
                predicted_onsets_dataset.append(remove_repeat_Onsets[i])
                predicted_onsets.append(onsetTimes[j])
                break
    false_possitives = []
    for i in range(len(onsetTimes)):
        ticker = 0
        for j in range(len(remove_repeat_Onsets)):
            if onsetTimes[i] - threshold < remove_repeat_Onsets[j] < onsetTimes[i] + threshold:
                ticker = ticker + 1
                break
        
        if ticker == 0:
            false_possitives.append(onsetTimes[i])
    false_possitives = len(np.array(false_possitives))        
    predicted_onsets_dataset = np.array(predicted_onsets_dataset)
    predicted_onsets = np.array(predicted_onsets)
    missed_onset = len(remove_repeat_Onsets) - predicted_onsets.shape[0]
    print('missed onset (+-{}):'.format(threshold), missed_onset)
    print('false possitives:', false_possitives)
    SAD_cal = np.sum(np.abs(predicted_onsets_dataset - predicted_onsets))
    print('Sum Absolute Difference:', SAD_cal)

### Predict Instruments ###
def compare_instrument(audio_filepath,remove_repeat_Onsets,dataset_dict):
    fs, audio = ToolReadAudio(audio_filepath)
    detector_releasetime = 0
    audioframe = np.diff(remove_repeat_Onsets)
    predicted_instrument = []
    predicted_dict = {}
    for i in range(len(remove_repeat_Onsets)-1):
        # print(predicted_onsets_dataset[i])
        detecting_frame = audio[int(remove_repeat_Onsets[i]*44100) : int(remove_repeat_Onsets[i]*44100+audioframe[i]*44100 - detector_releasetime*44100)]
        MFCC_featureVec = get_MFCC(detecting_frame)
        MFCC_featureVec = MFCC_featureVec.reshape(1,-1)
        predicted_labels = loaded_model.predict(MFCC_featureVec)

        # Labels numbers are corresponding to the testing file orders
        # 0: Crash, 1: HiHat, 2: Kick, 3: Ride, 4: Snare, 5: Tom
        if predicted_labels == 0 or predicted_labels == 3 or predicted_labels == 5:
            predicted_instrument.append('OT')
            predicted_dict[remove_repeat_Onsets[i]] = 'OT'
        if predicted_labels == 1:
            predicted_instrument.append('HH')
            predicted_dict[remove_repeat_Onsets[i]] = 'HH'
        if predicted_labels == 2:
            predicted_instrument.append('KD')
            predicted_dict[remove_repeat_Onsets[i]] = 'KD'
        if predicted_labels == 4:
            predicted_instrument.append('SD')
            predicted_dict[remove_repeat_Onsets[i]] = 'SD'
        # print(predicted_labels)
        
    # print(dataset_dict)
    # print(predicted_dict)

    true_prediction = 0
    false_prediction = 0
    for key in predicted_dict:
        if predicted_dict[key] in dataset_dict[key]:
            true_prediction += 1
        else:
            false_prediction += 1
    print('true prediction instruments numbers:', true_prediction)
    print('false prediction instruments numbers:', false_prediction)


# annotation_filepath = "ACA Final Project/MDBDrums-master/MDB Drums/annotations/class/MusicDelta_80sRock_class.txt"
# audio_filepath = 'ACA Final Project/MDBDrums-master/MDB Drums/audio/drum_only/MusicDelta_80sRock_Drum.wav'
# evaluation(annotation_filepath,audio_filepath)

Annotation_folder = 'MDB Drums/annotations/class'
Audio_folder = 'MDB Drums/audio/drum_only'
for file in os.listdir(Annotation_folder):
    split1 = file.split('_')
    audioName = split1[0] + '_' + split1[1] + '_Drum.wav'
    annotationName = split1[0] + '_' + split1[1] + '_class.txt'
    # print(audioName)
    annotation_filepath = Annotation_folder + '/' + annotationName
    audio_filepath = Audio_folder + '/' + audioName
    print('__________________________________________')
    print(audioName)
    evaluation(annotation_filepath,audio_filepath)
    

