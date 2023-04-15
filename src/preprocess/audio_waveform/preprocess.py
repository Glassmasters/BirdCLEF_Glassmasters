import os

from audiomentations import Compose, AddGaussianNoise, PitchShift, Reverse, Shift, TimeStretch
import librosa
import numpy as np

from src.plotting.plot_audio_wav_augmentation import plot_waveforms

np.random.seed(42)


DATASET_BASE_FILE_PATH = r"D:\Datasets\birdclef-2023"
TRAIN_SET_FILE_DIR = r"\train_audio"


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

'''
def apply_augmentation(input_wav: np.ndarray) -> np.ndarray:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
        # TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        # PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        # Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0),
    ])

    transformed_wav = augment(samples=input_wav, sample_rate=16000)

    return transformed_wav
'''

def load_waveform(file_path: str) -> np.ndarray:
    y_waveform, sampling_rate_waveform = librosa.load(file_path)
    print(len(y_waveform))
    print(sampling_rate_waveform)
    print(f"Duration in s: {len(y_waveform)/sampling_rate_waveform}")

    return y_waveform, sampling_rate_waveform


if __name__ == '__main__':
    folder_path = os.path.join(DATASET_BASE_FILE_PATH + TRAIN_SET_FILE_DIR + r"\abethr1")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        waveform, sr = load_waveform(file_path)
        # augmented_waveforms = apply_augmentation(waveform)
        # plot_waveforms(waveform, augmented_waveforms)

        audio_augmentation = AudioWaveformAugmentation(sampling_rate=sr)
        # augmented_waveforms = audio_augmentation.add_gaussian_noise(waveform)
        # plot_waveforms(waveform, augmented_waveforms)

        new_wav = audio_augmentation.truncate_or_fill(waveform)
        print(len(new_wav))
