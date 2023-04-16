import os

from audiomentations import Compose, AddGaussianNoise, PitchShift, Reverse, Shift, TimeStretch
import librosa
import numpy as np

from src.plotting.plot_audio_wav_augmentation import plot_waveforms

np.random.seed(42)


DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


def truncate_or_fill(input_waveform: np.ndarray, sampling_rate: float, target_waveform_duration: float):
        target_waveform_length = target_waveform_duration * sampling_rate

        if len(input_waveform) >= target_waveform_length:
            # truncate
            truncated_waveform = input_waveform[0:target_waveform_length]
            return truncated_waveform
        else:
            # fill
            # divide target by current length and add 1 to make sure that the new length is higher than the target length
            repitition_times = int(target_waveform_length/len(input_waveform) + 1)
            repeated_waveform = np.repeat(input_waveform, repitition_times)

            filled_waveform = repeated_waveform[0:target_waveform_length]
            return filled_waveform


class AudioWaveformAugmentation:
    def __init__(self, sampling_rate: float = -1, target_waveform_duration: float = 5):
        self.sampling_rate = sampling_rate
        self.target_waveform_duration = target_waveform_duration
        self.target_waveform_length = target_waveform_duration * sampling_rate

    def truncate_or_fill(self, input_waveform, diff_sampling_rate=None):
        if self.sampling_rate < 1:
            # sampling rate must be greater than 1
            raise ValueError
        if diff_sampling_rate:
            self.target_waveform_length = self.target_waveform_duration * diff_sampling_rate

        if len(input_waveform) >= self.target_waveform_length:
            # truncate
            truncated_waveform = input_waveform[0:self.target_waveform_length]
            return truncated_waveform
        else:
            # fill
            # divide target by current length and add 1 to make sure that the new length is higher than the target length
            repitition_times = int(self.target_waveform_length/len(input_waveform) + 1)
            repeated_waveform = np.repeat(input_waveform, repitition_times)

            filled_waveform = repeated_waveform[0:self.target_waveform_length]
            return filled_waveform

    def _add_gaussian_noise(self):
        self.augment_with_gaussian_noise = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0)
        ])

        return self.augment_with_gaussian_noise

    def _pitch_shift(self):
        self.augment_with_pitch_shift = Compose([
            PitchShift(min_semitones=-4, max_semitones=4, p=1.0)
        ])

    def _reverse(self):
        self.augment_with_reverse = Compose([
            Reverse(p=1.0)
        ])

    def _shift(self):
        self.augment_with_shift = Compose([
            Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0)
        ])

    def _time_stretch(self):
        self.augment_with_time_stretch = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)
        ])

    def add_gaussian_noise(self, input_waveform):
        aug_function = self._add_gaussian_noise()
        transformed_wav = aug_function(samples=input_waveform, sample_rate=16000)

        return transformed_wav

def _add_gaussian_noise(input_waveform: np.ndarray, samplerate: float):
    augment_with_gaussian_noise = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0)
    ])
    transformed_wav = augment_with_gaussian_noise(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _pitch_shift(input_waveform: np.ndarray, samplerate: float):
    augment_with_pitch_shift = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0)
    ])
    transformed_wav = augment_with_pitch_shift(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _reverse(input_waveform: np.ndarray, samplerate: float):
    augment_with_reverse = Compose([
        Reverse(p=1.0)
    ])
    transformed_wav = augment_with_reverse(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _shift(input_waveform: np.ndarray, samplerate: float):
    augment_with_shift = Compose([
        Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0)
    ])
    transformed_wav = augment_with_shift(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _time_stretch(input_waveform: np.ndarray, samplerate: float):
    augment_with_time_stretch = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)
    ])
    transformed_wav = augment_with_time_stretch(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _shift_in_time(input_waveform: np.ndarray, samplerate: float):
    shift_steps = np.random.randint(1, 10)
    shifted_wav = np.roll(input_waveform, shift_steps)
    
    return shifted_wav


available_functions = {
    "add_gaussian_noise": _add_gaussian_noise,
    "pitch_shift": _pitch_shift,
    "reverse": _reverse,
    "shift": _shift,
    "time_stretch": _time_stretch,
    "shift_in_time": _shift_in_time
}
