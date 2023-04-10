import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_mel_spectrogram(mel_spectrogram, title='Mel Spectrogram', cmap='viridis', dynamic_range=80):
    mel_spectrogram_np = mel_spectrogram.squeeze().detach().numpy()

    # Convert to dB scale for better visualization
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram_np, ref=np.max)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram_db, origin='lower', aspect='auto', cmap=cmap, vmin=-dynamic_range)
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()