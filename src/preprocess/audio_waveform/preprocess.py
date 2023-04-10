import os

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import matplotlib.pyplot as plt
import numpy as np


DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

def load_waveform(file_path):
    y_waveform, sampling_rate_waveform = librosa.load(file_path)

    return y_waveform


def plot_waveforms(orig_wav, transformed_wav):
    assert orig_wav.shape[0] == transformed_wav.shape[0]
    x_axis_points = [x for x in range(len(orig_wav))]
    plt.subplot(1, 2, 1)
    plt.plot(x_axis_points, orig_wav)
    plt.title("Original waveform")
    plt.xlabel('X-axis ')
    plt.ylabel('Y-axis ')

    plt.subplot(1, 2, 2)
    plt.plot(x_axis_points, transformed_wav)
    plt.title("Transformed waveform")
    plt.xlabel('X-axis ')
    plt.ylabel('Y-axis ')

    plt.show()


if __name__ == '__main__':
    folder_path = os.path.join(DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + r"\abethr1")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        waveform = load_waveform(file_path)
        augmented_waveforms = augment(samples=waveform, sample_rate=16000)

        plot_waveforms(waveform, augmented_waveforms)
