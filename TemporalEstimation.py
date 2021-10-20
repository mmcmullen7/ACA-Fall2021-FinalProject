import numpy as np
from scipy.io.wavfile import read as wavread
import math
import matplotlib.pyplot as plt

### Block Audio ###
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
### Read Audio ###
def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)
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
    return(samplerate, audio)

def stft(xb):
    w = np.hanning(xb.size)
    xb = xb * w
    X = np.fft.fft(xb)
    return X

def extract_stft(x, blockSize, hopSize, fs):
    xb,t = block_audio(x, blockSize, hopSize, fs)
    block_num = math.ceil(x.size / hopSize)
    X = np.zeros([block_num, blockSize])
    for i in range(0,block_num):
        X[i,:] = stft(xb[i])
    # X = 20 * np.log10(X)
    # X = np.clip(X, -40, 200)
    print(X)
    diff = X[1:] / X[:-1]
    log_diff = 20 * np.log10(diff)
    print(log_diff)
    return X, t


# Test Signal
fs = 44100
x = 1 * np.cos(2*np.pi * 50 * np.arange(0, 1, 1/fs))
x2 = 1 * np.cos(2*np.pi * 1000 * np.arange(0, 1, 1/fs))
sinusoid = np.concatenate([x,x2])

[fs, audio] = ToolReadAudio('ACA Final Project/MusicDelta_80sRock_Drum.wav')
X, t = extract_stft(audio, 2048, 512, fs)
print(X.shape)
print(t.size)
# plt.imshow(X, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
# plt.plot(X)
plt.show()


