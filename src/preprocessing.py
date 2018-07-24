import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

def wav2spectrogram(filename, max_len=71):
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate)
    
    if (max_len > spectrogram.shape[1]):
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return spectrogram[:,:,np.newaxis]

def wav2lgspectrogram(filename, max_len=71):
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate)
    spectrogram = np.log(spectrogram)

    if (max_len > spectrogram.shape[1]):
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return spectrogram[:,:,np.newaxis]

def wav2mfcc(filename, max_len=11):
    wave, sr = librosa.load(filename, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc[:,:,np.newaxis]
