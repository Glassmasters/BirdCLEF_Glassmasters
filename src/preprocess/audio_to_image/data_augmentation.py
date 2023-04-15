import torch
import torchaudio
import random


class AugmentMelSpectrogram:
    def __init__(self, freq_mask_param=8, time_mask_param=2, noise_std_range=(0.01, 0.1)):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.noise_std_range = noise_std_range

    def __call__(self, mel_spectrogram):
        # Apply frequency and time masking with a random probability
        if random.random() < 0.5:
            mel_spectrogram = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)(mel_spectrogram)
            mel_spectrogram = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param)(mel_spectrogram)

        # Add Gaussian noise
        noise_std = random.uniform(*self.noise_std_range)
        noise = torch.randn_like(mel_spectrogram) * noise_std
        mel_spectrogram += noise

        return mel_spectrogram
