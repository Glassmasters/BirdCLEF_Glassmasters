import numpy as np
import torch
from scipy.signal import hilbert


def audio_waveform_maximum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: max_value: float, the maximum value of the audio waveform. Maximum absolute value of the audio data, the value farthest away from zero

    """
    max_value = torch.max(torch.abs(audio_data))
    return max_value.item()


def audio_waveform_minimum(audio_data: torch.Tensor) -> float:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: min_value: float, the minimum value of the audio waveform. Minimum absolute value of the audio data, the value closest to zero

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
    percentile_value = torch.quantile(audio_data, percentile / 100.0)
    return percentile_value.item()


def audio_waveform_to_db(audio_data: torch.Tensor) -> torch.Tensor:
    """
    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns: db_value: torch.Tensor, the audio waveform in decibels.

    """
    eps = 1e-12 #constant to avoid zero in log
    db_value = 20 * torch.log10(torch.abs(audio_data) + eps)
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
    Calculates the envelope of an audio waveform using the Hilbert transform.

    Args:
        audio_data: torch.Tensor, shape (N,), where N is the number of samples in the audio waveform.

    Returns:
        envelope_value: torch.Tensor, the audio waveform envelope.
    """

    # Transform the audio data into a numpy array
    audio_data = audio_data.numpy()
    analytic_signal = hilbert(audio_data)

    # Calculate the amplitude envelope and transform it into a torch.Tensor
    amplitude_envelope = torch.Tensor(np.abs(analytic_signal))

    return amplitude_envelope
