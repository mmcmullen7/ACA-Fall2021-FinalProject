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

### Spectral Flux ###
def extract_spectral_flux(xb):
    [nBlocks_,blockSize_] = xb.shape
    SpecFluxVector = np.zeros(nBlocks_)

    Hann = np.hanning(blockSize_)
    Spectrogram = np.abs(np.fft.fft(xb*Hann,2*blockSize_))
    Spectrogram_ = Spectrogram[:,:blockSize_]
    Spectrogram_1 = np.concatenate((np.zeros((1,blockSize_)),Spectrogram_),axis=0)
    
    SpecDiff = np.diff(Spectrogram_1, axis=0)
    SpecFluxVector = np.sqrt(np.sum(SpecDiff**2, axis=1))/(blockSize_/2)
    # logscale = 20 * np.log10(SpecFluxVector[:-1]/SpecFluxVector[1:])
    # logscale = 20 * np.log10(SpecFluxVector)
    return SpecFluxVector

fs, audio = ToolReadAudio('ACA Final Project/Snare/07_SC_SADL_snare_closed_drier.wav')
[xb, time] = block_audio(audio, 1024, 512, fs)
spectral_flux = extract_spectral_flux(xb)
plt.subplot(2,1,1)
plt.plot(np.arange(audio.size)/fs,audio)
plt.title('Audio')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(2,1,2)
plt.plot(time, spectral_flux)
plt.title('Spectral Flux')
plt.tight_layout()
plt.show()

plt.subplot(2,1,1)
freq = np.abs(np.fft.fftfreq(audio.size, 1/fs))
mag = np.abs(np.fft.fft(audio))
max = np.max(mag)
plt.plot(freq, mag/max)
plt.title('Spectrum')
plt.xlabel('Frequency')
plt.ylabel('N Mag')

plt.subplot(2,1,2)
plt.specgram(audio, Fs = fs)
plt.title('Spectrogram')
plt.tight_layout()
plt.show()

