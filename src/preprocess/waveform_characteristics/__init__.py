import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import soundfile as sf
import torch
import random as rand
from typing import List


# TODO: Only load 50 files for now, but we should load all of them

def load_audio_waveform_from_folder(root_folder: str):
    """
    Load audio waveforms from a folder of folders of .ogg files and split them into 5-second clips.

    Parameters:
        - root_folder: str, the path to the root folder.

    Yields:
        - audio_clip: torch.Tensor, an audio clip as a PyTorch tensor.
    """
    df: pd.DataFrame = pd.read_csv('../../birdclef-2023/train_metadata.csv')
    random_ints = []
    for i in range(50):
        random_ints.append(rand.randint(0, len(df)))
    file_path = df["filename"].tolist()
    for j in random_ints:
        print(file_path[j])
        audio_data, sample_rate = sf.read("/home/meri/Documents/GitHub/BirdCLEF_Glassmasters/src/birdclef-2023"
                                          "/train_audio/" + file_path[j], dtype='float32')
        # Calculate the duration of the audio clip in seconds
        audio_duration = audio_data.shape[0] / sample_rate
        # Split the audio waveform into 5-second clips
        clip_duration = 5
        num_clips = int(audio_duration / clip_duration)
        for i in range(num_clips):
            start = i * clip_duration * sample_rate
            end = (i + 1) * clip_duration * sample_rate
            audio_clip = torch.from_numpy(audio_data[start:end])
            yield audio_clip
            del audio_clip
        del audio_data


def audio_waveform_maximum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: max_value: float, the maximum value of the audio waveform.

    """
    max_value = torch.max(torch.abs(audio_data))
    return max_value.item()


def audio_waveform_minimum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: min_value: float, the minimum value of the audio waveform.

    """
    min_value = torch.min(torch.abs(audio_data))
    return min_value.item()


def audio_waveform_mean(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: mean_value: float, the mean value of the audio waveform.

    """
    mean_value = torch.mean(torch.abs(audio_data))
    return mean_value.item()


def audio_waveform_std(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: std_value: float, the standard deviation value of the audio waveform.

    """
    std_value = torch.std(audio_data)
    return std_value.item()


def audio_waveform_percentile(audio_data: torch.Tensor, percentile: int) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.
        percentile: int, the percentile to calculate.

    Returns: percentile_value: float, the percentile value of the audio waveform.

    """
    percentile_value = torch.quantile(audio_data, percentile / 100)
    return percentile_value.item()


def audio_waveform_to_db(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: db_value: torch.Tensor, the audio waveform in decibels.

    """
    db_value = 20 * torch.log10(torch.abs(audio_data) + 1e-6)
    return db_value


def audio_waveform_fft(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: fft_value: torch.Tensor, the audio waveform in the frequency domain.

    """
    fft_value = torch.fft.fft(audio_data)
    return fft_value


def audio_waveform_envelope(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: envelope_value: torch.Tensor, the audio waveform envelope.

    """
    envelope_value = torch.abs(audio_data)
    return envelope_value


class KNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        # Convert singleton array to 1D array
        if y.ndim == 0:
            y = np.array([y])
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


if __name__ == '__main__':
    # Print cwd
    print(os.getcwd())

    # Load audio waveform data
    print("Loading audio waveform data...")
    path = "../../birdclef-2023/train_audio"
    audio_datas = load_audio_waveform_from_folder(path)
    test_data = load_audio_waveform_from_folder("../../birdclef-2023/test_soundscapes/")

    # Extract audio characteristics
    print("Extracting audio characteristics...")
    audio_characteristics = []
    for audio_data in audio_datas:
        audio_characteristics.append([audio_waveform_maximum(audio_data),
                                      audio_waveform_minimum(audio_data),
                                      audio_waveform_mean(audio_data),
                                      audio_waveform_std(audio_data)])
        del audio_data

    # Fit model and evaluate performance
    print("Fitting model and evaluating performance...")
    knn = KNN()
    knn.fit(audio_characteristics, "class")
    score_before = knn.score(audio_characteristics, "class")
    print(f"Model score before fitting: {score_before}")
    knn.predict(audio_characteristics[0])
    knn.score(audio_characteristics, "class")
    score_after = knn.score(audio_characteristics, "class")
    print(f"Model score after fitting: {score_after}")
    print(f"Difference in score: {score_after - score_before}")
