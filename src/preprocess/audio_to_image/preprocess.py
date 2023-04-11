import os
import random

from src.plotting.plot_mel_spectogram import plot_mel_spectrogram
import torchaudio
import torch
import torchvision.transforms as transforms

DATASET_BASE_FILE_PATH = r"D:\kaggle_competition\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


def load_audio(file_path, sample_rate=32000):
    """
    Load an audio file and resample it to the desired sample rate.
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int, optional): Desired sample rate for the output waveform. Defaults to 32000.
    Returns:
        torch.Tensor: The resampled waveform.
    """
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    resampler = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)
    waveform = resampler(waveform)
    return waveform


def get_mel_spectrogram(waveform, n_fft=2048, n_mels=128):
    """
    Convert a waveform to a Mel spectrogram.
    Args:
        waveform (torch.Tensor): The input waveform.
        n_fft (int, optional): The number of FFT points. Defaults to 2048.
        n_mels (int, optional): The number of Mel bands. Defaults to 128.
    Returns:
        torch.Tensor: The Mel spectrogram.
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, n_mels=n_mels)(waveform)
    return mel_spectrogram


def apply_augmentations(mel_spectrogram):
    """
    Apply augmentations to a Mel spectrogram.
    Args:
        mel_spectrogram (torch.Tensor): The input Mel spectrogram.
    Returns:
        torch.Tensor: The augmented Mel spectrogram.
    """
    # Random power
    power = random.uniform(0.5, 3)
    mel_spectrogram = mel_spectrogram.pow(power)

    # SpecAugment
    spec_augment = transforms.RandomApply([
        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
        torchaudio.transforms.TimeMasking(time_mask_param=2)], p=0.5)
    mel_spectrogram = spec_augment(mel_spectrogram)

    # Gaussian SNR
    noise = torch.randn_like(mel_spectrogram) * random.uniform(0.01, 0.1)
    mel_spectrogram += noise

    return mel_spectrogram


def preprocess_audio(file_path):
    """
    Preprocess an audio file by converting it to a Mel spectrogram and applying augmentations.
    Args:
        file_path (str): Path to the audio file.
    Returns:
        torch.Tensor: The preprocessed Mel spectrogram.
    """
    waveform = load_audio(file_path)
    mel_spectrogram = get_mel_spectrogram(waveform)

    # Apply augmentations
    mel_spectrogram = apply_augmentations(mel_spectrogram)

    return mel_spectrogram


if __name__ == '__main__':
    folder_path = os.path.join(DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + r"\abethr1")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Preprocess the audio file
        mel_spectrogram = preprocess_audio(file_path)

        # Plot the mel spectrogram
        plot_mel_spectrogram(mel_spectrogram, cmap='viridis', dynamic_range=80)
