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

def wav2mfcc(filename):
    wave, sr = librosa.load(filename, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc
