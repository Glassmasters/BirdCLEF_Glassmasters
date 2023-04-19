import os

from audiomentations import Compose, AddGaussianNoise, PitchShift, Reverse, Shift, TimeStretch
import librosa
import numpy as np

from src.plotting.plot_audio_wav_augmentation import plot_waveforms

np.random.seed(42)


def truncate_or_fill(input_waveform: np.ndarray, samplerate: int, target_waveform_duration: float) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the alraedy loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.
    :param target_waveform_duration: float, the duration all waveforms should have.

    :return truncated_waveform or filled_waveform: np.ndarray, the truncated or filled waveform.

    Either truncates the wavefrom when it is longer than the target duration or fills the waveform if it is shorter.
    Filling is done by repating the waveform.
    """
    target_waveform_length = target_waveform_duration * samplerate

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


def _add_gaussian_noise(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Adds gaussain noise onto the waveform.
    """
    augment_with_gaussian_noise = Compose([
        AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0)
    ])
    transformed_wav = augment_with_gaussian_noise(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _pitch_shift(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Applies a pitch to the waveform.
    """
    augment_with_pitch_shift = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0)
    ])
    transformed_wav = augment_with_pitch_shift(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _reverse(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Reverses the complete waveform.
    """
    augment_with_reverse = Compose([
        Reverse(p=1.0)
    ])
    transformed_wav = augment_with_reverse(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _shift(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Shifts a part of the waveform.
    """
    augment_with_shift = Compose([
        Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0)
    ])
    transformed_wav = augment_with_shift(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _time_stretch(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Stretches the waveform in time. Makes the waveform either playing faster or slower.
    """
    augment_with_time_stretch = Compose([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)
    ])
    transformed_wav = augment_with_time_stretch(samples=input_waveform, sample_rate=samplerate)

    return transformed_wav


def _shift_in_time(input_waveform: np.ndarray, samplerate: int) -> np.ndarray:
    """
    :param input_waveform: np.ndarray, the already loaed waveform as a numpy array.
    :param samplerate: int, the sample rate of the waveform.

    :return transformed_wav: np.ndarray, the transfomred waveform.

    Shifts the input waveform by between 1 and up to 9 seconds. The second amount is chosen
    randomly.
    """
    shift_steps = np.random.randint(1, 10)
    shifted_wav = np.roll(input_waveform, int(shift_steps * samplerate))
    
    return shifted_wav


available_functions = {
    "add_gaussian_noise": _add_gaussian_noise,
    "pitch_shift": _pitch_shift,
    "reverse": _reverse,
    "shift": _shift,
    "time_stretch": _time_stretch,
    "shift_in_time": _shift_in_time
}
