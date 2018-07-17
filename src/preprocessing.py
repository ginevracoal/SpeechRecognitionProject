import librosa
from scipy import signal
from scipy.io import wavfile

def wav2spectrogram(filename):
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate)
    return spectrogram

def wav2lgspectrogram(filename):
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate, nfft=2048)
    return spectrogram

def wav2mfcc(filename, max_len=11):
    wave, sr = librosa.load(filename, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    mfcc = mfcc[:, :max_len]
    
    return mfcc
