
def create_spectrogram(filename):
    """
        From the path of the wav file, produces a png file in the same location
    """
    sampling_rate, samples = wavfile.read(filename)
    f, t, spectrogram = signal.spectrogram(samples, sampling_rate, nfft=2048)
    img = Image.fromarray(np.uint8(spectrogram), mode='L')
    img.save(filename.replace('wav', 'png'))

for root, dirs, files in os.walk('.'):
    create_spectrogram(os.path.join(root, x)) for x in files if x.endswith('.wav')
