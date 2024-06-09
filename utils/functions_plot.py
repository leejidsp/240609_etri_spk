import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')   # To avoide overhead

eps=1e-9


    
def plot_spectrogram(path_data, sr=16000, nfft=1024, figsize=(10, 5)):

    win_length=int(0.020*sr)
    hop_length=win_length//32
    overlap_length = win_length - hop_length
    clim = [-60, 30]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    S = librosa.stft(path_data, n_fft=nfft, hop_length=hop_length, win_length=win_length, center=False)
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=1), hop_length=hop_length,
                               y_axis='linear', x_axis='time', ax=ax, cmap='inferno', sr=sr,
                                  vmax=clim[1], vmin=clim[0], 
                                  )
    
    plt.xlabel('Time(s)',fontsize=13)
    plt.ylabel('Frequency(Hz)',fontsize=13)
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()

    return fig


def plot_mel_spectrogram(wav, sr=16000, num_mels=80, nfft=1024, win_size=1024, hop_size=None):
    hop_size = win_size // 8
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=nfft, 
                                        win_length=win_size, hop_length=hop_size,
                                        n_mels= num_mels, fmax=sr//2)
    S_dB = librosa.power_to_db(S)

    fig = plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr,
                                hop_length=hop_size, fmax=sr//2, cmap='inferno',
                                vmax=20, vmin=-60)
    fig.canvas.draw()
    plt.close()


    return fig

