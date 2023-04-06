import os

import numpy as np
import pandas as pd
import glob
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import soundfile as sf


def split_audio():

    bird_dir = "train_audio"

    bird_sounds = []
    bird_features = []
    audio_files = glob.glob('train_audio/**/*.ogg', recursive=True)
    for bird_file in audio_files:
            y, sr = librosa.load(bird_file)
            feature_vector = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
            bird_sounds.append(y)
            bird_features.append(feature_vector)

    long_ogg_path = "test_soundscapes/soundscape_29201.ogg"
    y, sr = librosa.load(long_ogg_path)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    peaks, _ = find_peaks(onset_env, height=0.5)

    start_times = librosa.frames_to_time(peaks[:-1], sr=sr)
    end_times = librosa.frames_to_time(peaks[1:], sr=sr)

    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        segment = y[int(start_time * sr):int(end_time * sr)]
        segment_feature = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20).mean(axis=1)
        similarities = [
            np.dot(segment_feature, bird_feature) / (np.linalg.norm(segment_feature) * np.linalg.norm(bird_feature)) for
            bird_feature in bird_features]
        if max(similarities) > 0.9:
            sf.write(f"chunks/bird_sound_{i}.ogg", segment, int(sr))


def plot_duration():
    audio_files = glob.glob('train_audio/**/*.ogg', recursive=True)
    durations = []
    for audio_file, i in enumerate(audio_files):
        y, sr = librosa.load(audio_file)
        features = librosa.feature.mfcc(y=y, sr=sr)
        duration = (librosa.get_duration(y=y, sr=sr)) * 60  # *60 to convert to microseconds
        durations.append(duration)
        print(i)

    plt.hist(durations, bins=50)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.show()


def extract_features():
    audio_files = glob.glob('train_audio/**/*.ogg', recursive=True)
    for i, audio_file in enumerate(audio_files):
        y, sr = librosa.load(audio_file)
        n_mels = 128
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-frequency spectrogram {i}')
        plt.show()


def import_as_pd():
    df: pd.DataFrame = pd.read_csv("train_metadata.csv")
    return df


if __name__ == '__main__':
    split_audio()
