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

