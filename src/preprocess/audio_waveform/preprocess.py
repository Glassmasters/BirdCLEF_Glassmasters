import os

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import numpy as np

# from src.plotting.plot_audio_wav_augmentation import plot_waveforms


DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


def apply_augmentation(input_wav: np.ndarray) -> np.ndarray:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    transformed_wav = augment(samples=waveform, sample_rate=16000)

    return transformed_wav


def load_waveform(file_path: str) -> np.ndarray:
    y_waveform, sampling_rate_waveform = librosa.load(file_path)

    return y_waveform


if __name__ == '__main__':
    folder_path = os.path.join(DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + r"\abethr1")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        waveform = load_waveform(file_path)
        augmented_waveforms = apply_augmentation(waveform)

        # plot_waveforms(waveform, augmented_waveforms)
